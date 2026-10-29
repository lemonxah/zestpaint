//! Where poses come from: animation files (Mixamo or Quaternius FBX/glTF,
//! CMU mocap BVH), sampled into rig-neutral [`HumanPose`]s a frame at a
//! time; a few built-in photography poses; and the pose library, JSON files
//! in `~/.zestpaint/library/poses`.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};
use glam::{EulerRot, Mat4, Quat, Vec3};

use crate::skeleton::{HumanMap, HumanPose, Joint, Skeleton, Trs, map_humanoid};

/// Frames per second clips are sampled at.
const FPS: f64 = 30.0;
/// Longer clips are sampled more sparsely, to this many frames.
const MAX_FRAMES: usize = 900;

/// An animation file's rig and clips, sampled.
pub struct PoseSource {
    pub path: PathBuf,
    pub skeleton: Skeleton,
    pub map: HumanMap,
    pub rest_world: Vec<Mat4>,
    pub clips: Vec<Clip>,
}

pub struct Clip {
    pub name: String,
    /// World matrices of the mapped joints (indexed like `PoseSource::map`'s
    /// joints: one entry per humanoid bone, None where it has none).
    pub frames: Vec<Vec<Option<Mat4>>>,
    pub seconds_per_frame: f32,
}

impl PoseSource {
    pub fn file_name(&self) -> String {
        self.path.file_name().unwrap_or_default().to_string_lossy().into_owned()
    }

    /// Humanoid bones found, of the 22 body bones (fingers aside).
    pub fn body_bones(&self) -> usize {
        self.map[..22].iter().filter(|b| b.is_some()).count()
    }

    pub fn pose(&self, clip: usize, frame: usize) -> Option<HumanPose> {
        let clip = self.clips.get(clip)?;
        let frame_world = clip.frames.get(frame.min(clip.frames.len().saturating_sub(1)))?;
        // `HumanPose::sample` indexes worlds by joint; lay them out so.
        let mut rest = vec![Mat4::IDENTITY; self.skeleton.joints.len()];
        let mut posed = rest.clone();
        for (b, j) in self.map.iter().enumerate() {
            if let (Some(j), Some(m)) = (j, frame_world[b]) {
                rest[*j] = self.rest_world[*j];
                posed[*j] = m;
            }
        }
        let name = if clip.frames.len() > 1 { format!("{} · {}", clip.name, frame + 1) } else { clip.name.clone() };
        Some(HumanPose::sample(&name, &self.file_name(), &self.map, &rest, &posed))
    }

    fn sample_frames(&self, locals_at: &mut dyn FnMut(f64, &mut [Trs]), begin: f64, end: f64) -> (Vec<Vec<Option<Mat4>>>, f32) {
        let duration = (end - begin).max(0.0);
        let natural = (duration * FPS).round() as usize + 1;
        let count = natural.clamp(1, MAX_FRAMES);
        let step = if count > 1 { duration / (count - 1) as f64 } else { 0.0 };
        let mut locals = self.skeleton.rest_locals();
        let frames = (0..count)
            .map(|k| {
                locals.copy_from_slice(&self.skeleton.rest_locals());
                locals_at(begin + step * k as f64, &mut locals);
                let world = self.skeleton.world(&locals);
                self.map.iter().map(|j| j.map(|j| world[j])).collect()
            })
            .collect();
        (frames, step as f32)
    }
}

/// Read an animation file: `.fbx`, `.glb`/`.gltf` or `.bvh`.
pub fn load(path: &Path) -> Result<PoseSource> {
    let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    let source = match ext.as_str() {
        "fbx" => load_fbx(path),
        "glb" | "gltf" => load_gltf(path),
        "bvh" => load_bvh(path),
        _ => bail!("not an animation file ZestPaint reads (.fbx, .glb, .gltf, .bvh)"),
    }?;
    if source.map[crate::skeleton::HIPS].is_none() {
        bail!("{} has no humanoid skeleton ZestPaint recognises (it needs hips, legs and a head)", source.file_name());
    }
    if source.clips.is_empty() {
        bail!("{} has no animation or pose in it", source.file_name());
    }
    Ok(source)
}

fn names_parents(s: &Skeleton) -> (Vec<String>, Vec<Option<usize>>) {
    (s.joints.iter().map(|j| j.name.clone()).collect(), s.joints.iter().map(|j| j.parent).collect())
}

fn trs_of(t: &ufbx::Transform) -> Trs {
    Trs {
        t: Vec3::new(t.translation.x as f32, t.translation.y as f32, t.translation.z as f32),
        r: Quat::from_xyzw(t.rotation.x as f32, t.rotation.y as f32, t.rotation.z as f32, t.rotation.w as f32).normalize(),
        s: Vec3::new(t.scale.x as f32, t.scale.y as f32, t.scale.z as f32),
    }
}

fn load_fbx(path: &Path) -> Result<PoseSource> {
    use crate::fbx_loader::matrix_to_mat4;
    let opts = ufbx::LoadOpts {
        space_conversion: ufbx::SpaceConversion::TransformRoot,
        target_axes: ufbx::axes_right_handed_y_up(),
        target_unit_meters: 1.0,
        ..Default::default()
    };
    let scene = ufbx::load_file(path.to_str().context("path is not valid UTF-8")?, opts)
        .map_err(|e| anyhow::anyhow!("could not read {}: {:?}", path.display(), e.description))?;
    let nodes = scene.nodes.as_ref();
    // Every node but the root; the root's matrix carries the conversion.
    let mut order: Vec<usize> = (0..nodes.len()).filter(|&i| !nodes[i].is_root).collect();
    order.sort_by_key(|&i| nodes[i].node_depth);
    let mut joint_of = std::collections::HashMap::new();
    let mut joints = Vec::new();
    for &i in &order {
        let node = &nodes[i];
        let parent = node.parent.as_ref().and_then(|p| joint_of.get(&p.element.typed_id).copied());
        let base = match (parent, &node.parent) {
            (None, Some(p)) => matrix_to_mat4(&p.node_to_world),
            _ => Mat4::IDENTITY,
        };
        joint_of.insert(node.element.typed_id, joints.len());
        joints.push(Joint { name: node.element.name.to_string(), parent, rest: trs_of(&node.local_transform), base });
    }
    let mut skeleton = Skeleton { joints };
    let (names, parents) = names_parents(&skeleton);
    let map = map_humanoid(&names, &parents);
    let stacks = scene.anim_stacks.as_ref();
    // A posed character with no animation: its pose is how the nodes stand
    // now, measured from the bind pose its skin was made in.
    let file_pose = skeleton.rest_locals();
    let mut bound = false;
    if stacks.is_empty() {
        let mut bind: std::collections::HashMap<u32, Mat4> = std::collections::HashMap::new();
        for mesh in scene.meshes.as_ref().iter() {
            let Some(instance) = mesh.element.instances.as_ref().first() else { continue };
            let mesh_world = matrix_to_mat4(&instance.geometry_to_world);
            for skin in mesh.skin_deformers.as_ref().iter() {
                for cluster in skin.clusters.as_ref().iter() {
                    if let Some(bone) = &cluster.bone_node {
                        bind.entry(bone.element.typed_id).or_insert_with(|| mesh_world * matrix_to_mat4(&cluster.geometry_to_bone).inverse());
                    }
                }
            }
        }
        bound = !bind.is_empty();
        let mut world: Vec<Mat4> = Vec::with_capacity(skeleton.joints.len());
        for (k, &i) in order.iter().enumerate() {
            let joint = &skeleton.joints[k];
            let parent = joint.parent.map_or(joint.base, |p| world[p]);
            let w = bind.get(&nodes[i].element.typed_id).copied().unwrap_or_else(|| parent * file_pose[k].matrix());
            world.push(w);
        }
        for (k, w) in world.iter().enumerate() {
            let joint = &skeleton.joints[k];
            let parent = joint.parent.map_or(joint.base, |p| world[p]);
            skeleton.joints[k].rest = Trs::from_matrix(&(parent.inverse() * *w));
        }
    }
    let rest_world = skeleton.world(&skeleton.rest_locals());
    let mut source = PoseSource { path: path.to_path_buf(), skeleton, map, rest_world, clips: Vec::new() };
    if bound {
        let (frames, spf) = source.sample_frames(&mut |_, locals| locals.copy_from_slice(&file_pose), 0.0, 0.0);
        let name = path.file_stem().unwrap_or_default().to_string_lossy().into_owned();
        source.clips.push(Clip { name, frames, seconds_per_frame: spf });
    }
    for stack in stacks.iter() {
        let (frames, spf) = source.sample_frames(
            &mut |time, locals| {
                for &i in &order {
                    let j = joint_of[&nodes[i].element.typed_id];
                    locals[j] = trs_of(&ufbx::evaluate_transform(&stack.anim, &nodes[i], time));
                }
            },
            stack.time_begin,
            stack.time_end,
        );
        let name = stack.element.name.to_string();
        // Mixamo calls every clip "mixamo.com": the file names it better.
        let name = if name.is_empty() || name == "mixamo.com" || stacks.len() == 1 {
            path.file_stem().unwrap_or_default().to_string_lossy().into_owned()
        } else {
            name
        };
        source.clips.push(Clip { name, frames, seconds_per_frame: spf });
    }
    Ok(source)
}

fn load_gltf(path: &Path) -> Result<PoseSource> {
    let (doc, buffers, _) = gltf::import(path).with_context(|| format!("could not read {}", path.display()))?;
    let scene = doc.default_scene().or_else(|| doc.scenes().next()).context("the file has no scene")?;
    let (parent, _, depth) = crate::gltf_loader::node_tree(&doc, &scene);
    let mut order: Vec<usize> = (0..parent.len()).collect();
    order.sort_by_key(|&i| depth[i]);
    let nodes: Vec<gltf::Node> = doc.nodes().collect();
    let mut joint_of = vec![usize::MAX; nodes.len()];
    let mut joints = Vec::new();
    for &i in &order {
        let (t, r, s) = nodes[i].transform().decomposed();
        joint_of[i] = joints.len();
        joints.push(Joint {
            name: nodes[i].name().map(str::to_owned).unwrap_or_else(|| format!("Node {i}")),
            parent: parent[i].map(|p| joint_of[p]),
            rest: Trs { t: t.into(), r: Quat::from_array(r).normalize(), s: s.into() },
            base: Mat4::IDENTITY,
        });
    }
    let skeleton = Skeleton { joints };
    let (names, parents) = names_parents(&skeleton);
    let map = map_humanoid(&names, &parents);
    let rest_world = skeleton.world(&skeleton.rest_locals());
    let mut source = PoseSource { path: path.to_path_buf(), skeleton, map, rest_world, clips: Vec::new() };

    for (k, animation) in doc.animations().enumerate() {
        // Each channel: joint, what it drives, key times and values.
        let mut channels: Vec<(usize, Channel)> = Vec::new();
        let (mut begin, mut end) = (f32::MAX, 0.0f32);
        for channel in animation.channels() {
            let j = joint_of[channel.target().node().index()];
            if j == usize::MAX {
                continue;
            }
            let reader = channel.reader(|b| Some(&buffers[b.index()].0));
            let Some(times) = reader.read_inputs() else { continue };
            let times: Vec<f32> = times.collect();
            let cubic = channel.sampler().interpolation() == gltf::animation::Interpolation::CubicSpline;
            let step = channel.sampler().interpolation() == gltf::animation::Interpolation::Step;
            let values: Vec<[f32; 4]> = match reader.read_outputs() {
                Some(gltf::animation::util::ReadOutputs::Translations(v)) => v.map(|p| [p[0], p[1], p[2], 0.0]).collect(),
                Some(gltf::animation::util::ReadOutputs::Scales(v)) => v.map(|p| [p[0], p[1], p[2], 0.0]).collect(),
                Some(gltf::animation::util::ReadOutputs::Rotations(v)) => v.into_f32().collect(),
                _ => continue,
            };
            // Cubic splines store (in-tangent, value, out-tangent): keep values.
            let values: Vec<[f32; 4]> = if cubic { values.chunks(3).filter_map(|c| c.get(1).copied()).collect() } else { values };
            if let (Some(&a), Some(&b)) = (times.first(), times.last()) {
                begin = begin.min(a);
                end = end.max(b);
            }
            let property = match channel.target().property() {
                gltf::animation::Property::Translation => 0,
                gltf::animation::Property::Rotation => 1,
                gltf::animation::Property::Scale => 2,
                _ => continue,
            };
            channels.push((j, Channel { property, times, values, step }));
        }
        if channels.is_empty() {
            continue;
        }
        let (frames, spf) = source.sample_frames(
            &mut |time, locals| {
                for (j, ch) in &channels {
                    let v = ch.at(time as f32);
                    match ch.property {
                        0 => locals[*j].t = Vec3::new(v[0], v[1], v[2]),
                        1 => locals[*j].r = Quat::from_array(v).normalize(),
                        _ => locals[*j].s = Vec3::new(v[0], v[1], v[2]),
                    }
                }
            },
            begin.min(end) as f64,
            end as f64,
        );
        let name = animation.name().map(str::to_owned).unwrap_or_else(|| format!("Animation {}", k + 1));
        source.clips.push(Clip { name, frames, seconds_per_frame: spf });
    }
    Ok(source)
}

struct Channel {
    /// 0 translation, 1 rotation, 2 scale.
    property: u8,
    times: Vec<f32>,
    values: Vec<[f32; 4]>,
    step: bool,
}

impl Channel {
    fn at(&self, t: f32) -> [f32; 4] {
        let n = self.times.len().min(self.values.len());
        if n == 0 {
            return [0.0, 0.0, 0.0, 1.0];
        }
        let i = self.times[..n].partition_point(|&k| k <= t);
        if i == 0 {
            return self.values[0];
        }
        if i >= n {
            return self.values[n - 1];
        }
        let (a, b) = (self.values[i - 1], self.values[i]);
        if self.step {
            return a;
        }
        let span = self.times[i] - self.times[i - 1];
        let f = if span > 0.0 { (t - self.times[i - 1]) / span } else { 0.0 };
        if self.property == 1 {
            Quat::from_array(a).slerp(Quat::from_array(b), f).to_array()
        } else {
            std::array::from_fn(|k| a[k] + (b[k] - a[k]) * f)
        }
    }
}

/// BVH: a text hierarchy of offsets and channels, then one line of channel
/// values per frame (CMU's mocap library is in this format).
fn load_bvh(path: &Path) -> Result<PoseSource> {
    let text = std::fs::read_to_string(path).with_context(|| format!("could not read {}", path.display()))?;
    parse_bvh(&text, path)
}

fn parse_bvh(text: &str, path: &Path) -> Result<PoseSource> {
    let mut tokens = text.split_whitespace().peekable();
    let mut joints: Vec<Joint> = Vec::new();
    // Per joint: its channels, in file order.
    let mut channels: Vec<Vec<String>> = Vec::new();
    let mut stack: Vec<Option<usize>> = Vec::new();
    let mut pending: Option<Option<usize>> = None;
    let mut last: Option<usize> = None;
    let expect = |tokens: &mut std::iter::Peekable<std::str::SplitWhitespace>| -> Result<f32> {
        tokens.next().context("the file ends early")?.parse::<f32>().context("expected a number")
    };
    if tokens.next() != Some("HIERARCHY") {
        bail!("not a BVH file");
    }
    while let Some(tok) = tokens.next() {
        match tok {
            "ROOT" | "JOINT" => {
                let name = tokens.next().context("a joint without a name")?.to_string();
                let parent = stack.last().copied().flatten();
                joints.push(Joint { name, parent, rest: Trs::default(), base: Mat4::IDENTITY });
                channels.push(Vec::new());
                last = Some(joints.len() - 1);
                pending = Some(last);
            }
            "End" => {
                tokens.next();
                pending = Some(None);
            }
            "{" => stack.push(pending.take().flatten()),
            "}" => {
                stack.pop();
            }
            "OFFSET" => {
                let v = Vec3::new(expect(&mut tokens)?, expect(&mut tokens)?, expect(&mut tokens)?);
                // An End Site's offset belongs to no joint.
                if let Some(Some(j)) = stack.last() {
                    if Some(*j) == last {
                        joints[*j].rest.t = v;
                    }
                }
            }
            "CHANNELS" => {
                let n: usize = tokens.next().context("CHANNELS without a count")?.parse()?;
                let j = last.context("CHANNELS outside a joint")?;
                for _ in 0..n {
                    channels[j].push(tokens.next().context("the file ends early")?.to_string());
                }
            }
            "MOTION" => break,
            _ => {}
        }
    }
    if tokens.next() != Some("Frames:") {
        bail!("BVH has no MOTION frames");
    }
    let frame_count: usize = tokens.next().context("no frame count")?.parse()?;
    if (tokens.next(), tokens.next()) != (Some("Frame"), Some("Time:")) {
        bail!("BVH has no frame time");
    }
    let frame_time: f64 = tokens.next().context("no frame time")?.parse()?;
    let per_frame: usize = channels.iter().map(Vec::len).sum();
    let values: Vec<f32> = tokens.take(frame_count * per_frame).map(|t| t.parse::<f32>().unwrap_or(0.0)).collect();
    let frame_count = values.len() / per_frame.max(1);

    let skeleton = Skeleton { joints };
    let (names, parents) = names_parents(&skeleton);
    let map = map_humanoid(&names, &parents);
    let rest_world = skeleton.world(&skeleton.rest_locals());
    let mut source = PoseSource { path: path.to_path_buf(), skeleton, map, rest_world, clips: Vec::new() };
    let rest = source.skeleton.rest_locals();
    let (frames, spf) = source.sample_frames(
        &mut |time, locals| {
            let f = ((time / frame_time.max(1e-6)).round() as usize).min(frame_count.saturating_sub(1));
            let mut at = f * per_frame;
            for (j, chs) in channels.iter().enumerate() {
                let mut t = rest[j].t;
                let mut r = Quat::IDENTITY;
                for ch in chs {
                    let v = values.get(at).copied().unwrap_or(0.0);
                    at += 1;
                    match ch.as_str() {
                        "Xposition" => t.x = v,
                        "Yposition" => t.y = v,
                        "Zposition" => t.z = v,
                        "Xrotation" => r *= Quat::from_rotation_x(v.to_radians()),
                        "Yrotation" => r *= Quat::from_rotation_y(v.to_radians()),
                        "Zrotation" => r *= Quat::from_rotation_z(v.to_radians()),
                        _ => {}
                    }
                }
                locals[j] = Trs { t, r, s: Vec3::ONE };
            }
        },
        0.0,
        frame_time * frame_count.saturating_sub(1) as f64,
    );
    if frame_count > 0 {
        let name = path.file_stem().unwrap_or_default().to_string_lossy().into_owned();
        source.clips.push(Clip { name, frames, seconds_per_frame: spf });
    }
    Ok(source)
}

// --- built-in poses ---------------------------------------------------------------

/// A bone of a built-in pose: where it points (body frame: +X the body's
/// left, +Y up, +Z forward), or its rotation as nod/turn/tilt degrees.
enum Set {
    Aim(f32, f32, f32),
    Turn(f32, f32, f32),
}

/// The canonical rig built-in poses are written on: T-pose, facing +Z,
/// rest rotations all identity, so a bone's rest direction is its offset.
fn canonical() -> Skeleton {
    let mut joints: Vec<Joint> = Vec::new();
    let mut add = |name: &str, parent: Option<&str>, t: Vec3| {
        let parent = parent.map(|p| joints.iter().position(|j| j.name == p).unwrap());
        joints.push(Joint { name: name.into(), parent, rest: Trs { t, ..Default::default() }, base: Mat4::IDENTITY });
    };
    add("Hips", None, Vec3::new(0.0, 0.95, 0.0));
    add("Spine", Some("Hips"), Vec3::new(0.0, 0.1, 0.0));
    add("Chest", Some("Spine"), Vec3::new(0.0, 0.15, 0.0));
    add("UpperChest", Some("Chest"), Vec3::new(0.0, 0.12, 0.0));
    add("Neck", Some("UpperChest"), Vec3::new(0.0, 0.13, 0.0));
    add("Head", Some("Neck"), Vec3::new(0.0, 0.1, 0.0));
    for (side, sx) in [("Left", 1.0f32), ("Right", -1.0)] {
        let s = |n: &str| format!("{side}{n}");
        add(&s("Shoulder"), Some("UpperChest"), Vec3::new(0.03 * sx, 0.09, 0.0));
        add(&s("UpperArm"), Some(&s("Shoulder")), Vec3::new(0.12 * sx, 0.0, 0.0));
        add(&s("LowerArm"), Some(&s("UpperArm")), Vec3::new(0.27 * sx, 0.0, 0.0));
        add(&s("Hand"), Some(&s("LowerArm")), Vec3::new(0.25 * sx, 0.0, 0.0));
        for (k, f) in ["Thumb", "Index", "Middle", "Ring", "Little"].iter().enumerate() {
            let at = Vec3::new(0.08 * sx, 0.0, 0.03 - k as f32 * 0.015);
            let at = if k == 0 { Vec3::new(0.025 * sx, -0.01, 0.035) } else { at };
            add(&s(&format!("{f}1")), Some(&s("Hand")), at);
            add(&s(&format!("{f}2")), Some(&s(&format!("{f}1"))), Vec3::new(0.035 * sx, 0.0, 0.0));
            add(&s(&format!("{f}3")), Some(&s(&format!("{f}2"))), Vec3::new(0.025 * sx, 0.0, 0.0));
        }
        add(&s("UpperLeg"), Some("Hips"), Vec3::new(0.09 * sx, -0.05, 0.0));
        add(&s("LowerLeg"), Some(&s("UpperLeg")), Vec3::new(0.0, -0.42, 0.0));
        add(&s("Foot"), Some(&s("LowerLeg")), Vec3::new(0.0, -0.4, 0.0));
        add(&s("Toes"), Some(&s("Foot")), Vec3::new(0.0, -0.06, 0.13));
    }
    Skeleton { joints }
}

/// Pose the canonical rig: each bone listed is set in world space, others
/// keep their parent's rotation. Left-side entries are mirrored onto the
/// right unless the right is listed too. `drop` lowers the hips (metres).
fn build(name: &str, bones: &[(&str, Set)], drop: f32) -> HumanPose {
    let skeleton = canonical();
    let (names, parents) = names_parents(&skeleton);
    let map = map_humanoid(&names, &parents);
    let mut world_rot: Vec<Quat> = Vec::with_capacity(skeleton.joints.len());
    let lookup = |joint: &str| -> Option<(bool, &Set)> {
        if let Some((_, s)) = bones.iter().find(|(b, _)| *b == joint) {
            return Some((false, s));
        }
        let mirrored = joint.strip_prefix("Right").map(|rest| format!("Left{rest}"))?;
        bones.iter().find(|(b, _)| *b == mirrored).map(|(_, s)| (true, s))
    };
    for joint in &skeleton.joints {
        let parent = joint.parent.map_or(Quat::IDENTITY, |p| world_rot[p]);
        let q = match lookup(&joint.name) {
            Some((mirror, Set::Aim(x, y, z))) => {
                let aim = Vec3::new(if mirror { -x } else { *x }, *y, *z).normalize();
                // The direction the bone's child sits in.
                let child = skeleton.joints.iter().find(|c| c.parent == skeleton.index_of(&joint.name) && !c.name.contains("Thumb"));
                let rest_dir = child.map_or(Vec3::Y, |c| c.rest.t.normalize());
                Quat::from_rotation_arc(rest_dir, aim)
            }
            Some((mirror, Set::Turn(x, y, z))) => {
                let (y, z) = if mirror { (-y, -z) } else { (*y, *z) };
                Quat::from_euler(EulerRot::YXZ, y.to_radians(), x.to_radians(), z.to_radians())
            }
            None => parent,
        };
        world_rot.push(q);
    }
    let mut locals = skeleton.rest_locals();
    for (j, joint) in skeleton.joints.iter().enumerate() {
        let parent = joint.parent.map_or(Quat::IDENTITY, |p| world_rot[p]);
        locals[j].r = parent.inverse() * world_rot[j];
    }
    locals[0].t.y -= drop;
    let rest = skeleton.rest_world();
    let posed = skeleton.world(&locals);
    HumanPose::sample(name, "built in", &map, &rest, &posed)
}

/// Photography starter poses, written for any humanoid.
pub fn builtin() -> Vec<HumanPose> {
    use Set::{Aim, Turn};
    let relaxed_arms = [
        ("LeftUpperArm", Aim(0.22, -0.97, 0.04)),
        ("LeftLowerArm", Aim(0.12, -0.95, 0.28)),
        ("LeftHand", Aim(0.1, -0.95, 0.3)),
    ];
    let mut poses = vec![
        build("A-pose", &[("LeftUpperArm", Aim(0.7, -0.7, 0.0))], 0.0),
        build("Relaxed stand", &relaxed_arms, 0.0),
        build(
            "Contrapposto",
            &[
                ("Hips", Turn(0.0, 6.0, -4.0)),
                ("Spine", Turn(0.0, 3.0, 2.0)),
                ("Chest", Turn(0.0, 0.0, 4.0)),
                ("Head", Turn(4.0, -12.0, -6.0)),
                ("LeftUpperLeg", Aim(0.0, -0.96, 0.25)),
                ("LeftLowerLeg", Aim(0.06, -0.99, -0.08)),
                ("LeftFoot", Aim(0.25, -0.4, 0.88)),
                ("RightUpperLeg", Aim(-0.08, -1.0, 0.0)),
                ("RightLowerLeg", Aim(-0.02, -1.0, 0.0)),
                ("RightFoot", Aim(-0.2, -0.45, 0.87)),
                ("LeftUpperArm", Aim(0.26, -0.96, 0.08)),
                ("LeftLowerArm", Aim(0.1, -0.9, 0.42)),
                ("RightUpperArm", Aim(-0.2, -0.98, -0.02)),
                ("RightLowerArm", Aim(-0.12, -0.97, 0.18)),
            ],
            0.01,
        ),
        build(
            "Hands on hips",
            &[
                ("LeftUpperArm", Aim(0.6, -0.68, -0.42)),
                ("LeftLowerArm", Aim(-0.56, -0.62, 0.55)),
                ("LeftHand", Aim(-0.6, -0.75, 0.25)),
                ("Head", Turn(-3.0, 0.0, 0.0)),
            ],
            0.0,
        ),
        build(
            "Arms crossed",
            &[
                ("LeftUpperArm", Aim(0.25, -0.88, 0.4)),
                ("LeftLowerArm", Aim(-0.92, 0.18, 0.35)),
                ("RightUpperArm", Aim(-0.25, -0.9, 0.36)),
                ("RightLowerArm", Aim(0.92, 0.1, 0.38)),
                ("Head", Turn(-4.0, 0.0, 3.0)),
            ],
            0.0,
        ),
        build(
            "Wave",
            &[
                ("RightUpperArm", Aim(-0.92, 0.12, 0.35)),
                ("RightLowerArm", Aim(-0.12, 0.98, 0.15)),
                ("RightHand", Aim(-0.05, 0.99, 0.1)),
                ("LeftUpperArm", Aim(0.22, -0.97, 0.04)),
                ("LeftLowerArm", Aim(0.12, -0.95, 0.28)),
                ("Head", Turn(0.0, -10.0, -5.0)),
                ("Chest", Turn(0.0, -6.0, 0.0)),
            ],
            0.0,
        ),
        build(
            "Look over shoulder",
            &[
                ("Hips", Turn(0.0, 25.0, 0.0)),
                ("Spine", Turn(0.0, 10.0, 0.0)),
                ("Chest", Turn(0.0, -10.0, 0.0)),
                ("Neck", Turn(0.0, -30.0, 0.0)),
                ("Head", Turn(5.0, -55.0, -8.0)),
                ("LeftUpperArm", Aim(0.25, -0.96, -0.08)),
                ("LeftLowerArm", Aim(0.15, -0.95, 0.2)),
                ("RightUpperArm", Aim(-0.2, -0.97, 0.12)),
                ("RightLowerArm", Aim(-0.1, -0.94, 0.3)),
            ],
            0.0,
        ),
        build(
            "Walking",
            &[
                ("LeftUpperLeg", Aim(0.0, -0.94, 0.34)),
                ("LeftLowerLeg", Aim(0.0, -0.99, 0.12)),
                ("LeftFoot", Aim(0.1, -0.15, 0.98)),
                ("RightUpperLeg", Aim(0.0, -0.96, -0.27)),
                ("RightLowerLeg", Aim(0.0, -0.88, -0.48)),
                ("RightFoot", Aim(-0.1, -0.7, 0.7)),
                ("LeftUpperArm", Aim(0.2, -0.95, -0.24)),
                ("LeftLowerArm", Aim(0.15, -0.97, -0.05)),
                ("RightUpperArm", Aim(-0.2, -0.95, 0.25)),
                ("RightLowerArm", Aim(-0.12, -0.8, 0.58)),
                ("Hips", Turn(0.0, -5.0, 0.0)),
                ("Chest", Turn(0.0, 5.0, 0.0)),
            ],
            0.03,
        ),
        build(
            "Sitting",
            &[
                ("LeftUpperLeg", Aim(0.12, 0.0, 1.0)),
                ("LeftLowerLeg", Aim(0.04, -1.0, 0.08)),
                ("LeftFoot", Aim(0.1, -0.2, 0.97)),
                ("Spine", Turn(4.0, 0.0, 0.0)),
                ("Chest", Turn(6.0, 0.0, 0.0)),
                ("LeftUpperArm", Aim(0.2, -0.9, 0.38)),
                ("LeftLowerArm", Aim(-0.05, -0.62, 0.78)),
                ("Head", Turn(-6.0, 0.0, 0.0)),
            ],
            0.42,
        ),
    ];
    for p in &mut poses {
        p.source = "built in".into();
    }
    poses
}

// --- the pose library ---------------------------------------------------------------

/// `~/.zestpaint/library/poses`.
pub fn library_dir() -> PathBuf {
    crate::library::library_dir().parent().map(|p| p.join("poses")).unwrap_or_else(|| PathBuf::from("poses"))
}

/// Saved poses, by name.
pub fn saved() -> Vec<(PathBuf, HumanPose)> {
    let Ok(dir) = std::fs::read_dir(library_dir()) else { return Vec::new() };
    let mut out: Vec<(PathBuf, HumanPose)> = dir
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .filter_map(|p| {
            let pose = serde_json::from_str::<HumanPose>(&std::fs::read_to_string(&p).ok()?).ok()?;
            Some((p, pose))
        })
        .collect();
    out.sort_by(|a, b| a.1.name.to_lowercase().cmp(&b.1.name.to_lowercase()));
    out
}

/// Save `pose` under its name; returns the file.
pub fn save(pose: &HumanPose) -> Result<PathBuf> {
    let dir = library_dir();
    std::fs::create_dir_all(&dir)?;
    let safe: String = pose.name.chars().map(|c| if c.is_alphanumeric() || " -_".contains(c) { c } else { '_' }).collect();
    let safe = safe.trim();
    let safe = if safe.is_empty() { "Pose" } else { safe };
    let mut path = dir.join(format!("{safe}.json"));
    let mut n = 2;
    while path.exists() {
        path = dir.join(format!("{safe} {n}.json"));
        n += 1;
    }
    std::fs::write(&path, serde_json::to_string_pretty(pose)?)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::{Tweaks, retarget};

    fn bone_pos(s: &Skeleton, locals: &[Trs], name: &str) -> Vec3 {
        s.world(locals)[s.index_of(name).unwrap()].w_axis.truncate()
    }

    /// Built-in poses do what they say on the canonical rig: hands on the
    /// hips, crossed arms in front of the chest, a sitting pose's thighs
    /// level, a wave above the head.
    #[test]
    fn builtin_poses_land_where_they_should() {
        let s = canonical();
        let (n, p) = names_parents(&s);
        let map = map_humanoid(&n, &p);
        assert!(map.iter().all(Option::is_some), "the canonical rig has every bone");
        let poses = builtin();
        let get = |name: &str| {
            let pose = poses.iter().find(|p| p.name == name).unwrap();
            retarget(Some(pose), &s, &map, false, &Tweaks::new())
        };
        let hips = get("Hands on hips");
        let hand = bone_pos(&s, &hips, "LeftHand");
        let hip = bone_pos(&s, &hips, "LeftUpperLeg");
        assert!((hand - (hip + Vec3::new(0.06, 0.08, 0.0))).length() < 0.12, "the hand rests on the hip: {hand} vs {hip}");

        let crossed = get("Arms crossed");
        let (l, r) = (bone_pos(&s, &crossed, "LeftHand"), bone_pos(&s, &crossed, "RightHand"));
        assert!(l.x < 0.0 && r.x > 0.0, "the hands cross over: {l} {r}");
        assert!(l.z > 0.15 && r.z > 0.15, "in front of the chest: {l} {r}");

        let sitting = get("Sitting");
        let (hip, knee) = (bone_pos(&s, &sitting, "LeftUpperLeg"), bone_pos(&s, &sitting, "LeftLowerLeg"));
        assert!((hip.y - knee.y).abs() < 0.05 && knee.z > 0.35, "thighs level and forward: {hip} {knee}");
        let foot = bone_pos(&s, &sitting, "LeftFoot");
        assert!((foot.y - 0.08).abs() < 0.03, "the ankles are as high as standing, feet on the floor: {foot}");

        let wave = get("Wave");
        assert!(bone_pos(&s, &wave, "RightHand").y > bone_pos(&s, &wave, "Head").y, "the waving hand is up by the head");
        assert!(bone_pos(&s, &wave, "LeftHand").y < 1.0, "the other hand is down by the side");
    }

    /// A small BVH (CMU style) reads into a source with the bones mapped and
    /// one clip a frame long.
    #[test]
    fn bvh_reads_hierarchy_and_frames() {
        let joint = |name: &str, off: &str, body: &str| format!("JOINT {name}\n{{\nOFFSET {off}\nCHANNELS 3 Zrotation Xrotation Yrotation\n{body}\n}}\n");
        let end = "End Site\n{\nOFFSET 0 -1 0\n}";
        let leg = |s: &str, x: &str| joint(&format!("{s}UpLeg"), &format!("{x} -1 0"), &joint(&format!("{s}Leg"), "0 -4 0", &joint(&format!("{s}Foot"), "0 -4 0", &joint(&format!("{s}ToeBase"), "0 -1 1", end))));
        let arm = |s: &str, x: f32| {
            let hand = joint(&format!("{s}Hand"), &format!("{} 0 0", 2.0 * x), end);
            let fore = joint(&format!("{s}ForeArm"), &format!("{} 0 0", 2.0 * x), &hand);
            joint(&format!("{s}Shoulder"), &format!("{x} 2 0"), &joint(&format!("{s}Arm"), &format!("{x} 0 0"), &fore))
        };
        let text = format!(
            "HIERARCHY\nROOT Hips\n{{\nOFFSET 0 0 0\nCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n{}{}{}\n}}\nMOTION\nFrames: 2\nFrame Time: 0.5\n",
            leg("Left", "1"),
            leg("Right", "-1"),
            joint("Spine", "0 1 0", &joint("Neck", "0 3 0", &format!("{}{}{}", joint("Head", "0 1 0", end), arm("Left", 1.0), arm("Right", -1.0)))),
        );
        let channels = 6 + 3 * (4 * 2 + 3 + 4 * 2);
        let frame0 = vec!["0"; channels].join(" ");
        let mut frame1 = vec!["0".to_string(); channels];
        frame1[1] = "-2".into();
        let text = format!("{text}{frame0}\n{}\n", frame1.join(" "));
        let source = parse_bvh(&text, Path::new("walk.bvh")).unwrap();
        assert_eq!(source.body_bones(), 20, "every body bone but the chest and upper chest");
        assert_eq!(source.clips.len(), 1);
        let first = source.pose(0, 0).unwrap();
        let hips = first.bones["Hips"];
        assert_eq!(hips.pos, hips.rest_pos, "frame 0 is the rest");
        let last = source.pose(0, source.clips[0].frames.len() - 1).unwrap();
        assert!((last.bones["Hips"].pos[1] - -2.0).abs() < 1e-4, "the hips drop in the last frame");
    }
}

#[cfg(test)]
mod real_tests {
    /// Real-file check: ZESTPAINT_TEST_ANIM=/path/to/clip.(fbx|glb|bvh)
    /// reads as a pose source with humanoid bones and clips.
    #[test]
    fn load_real_animation() {
        let Ok(path) = std::env::var("ZESTPAINT_TEST_ANIM") else {
            eprintln!("ZESTPAINT_TEST_ANIM not set; skipping");
            return;
        };
        let source = super::load(std::path::Path::new(&path)).unwrap();
        eprintln!("{} body bones, {} clips", source.body_bones(), source.clips.len());
        for c in &source.clips {
            eprintln!("  {}: {} frames", c.name, c.frames.len());
        }
        for (b, j) in source.map.iter().enumerate() {
            let Some(j) = j else { eprintln!("  {:<14} —", crate::skeleton::BONES[b]); continue };
            if b < 22 {
                let w = source.rest_world[*j];
                eprintln!("  {:<14} {:<28} rest pos {:?}", crate::skeleton::BONES[b], source.skeleton.joints[*j].name, w.w_axis.truncate());
            }
        }
        let pose = source.pose(0, source.clips[0].frames.len() / 2).unwrap();
        for b in ["LeftUpperArm", "LeftLowerArm", "LeftHand"] {
            if let Some(x) = pose.bones.get(b) {
                eprintln!("  {b} pose pos {:?}", x.pos);
            }
        }
        assert!(pose.bones.contains_key("Hips"));
    }
}
