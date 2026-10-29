//! Skeletons, skins and poses: what lets a rigged avatar stand in a
//! photography pose. The loaders read a model's bones and skin weights into a
//! [`Skeleton`] and one [`Skin`] per part; a pose sets the bones' local
//! rotations and [`Skin::deform`] bends the rest mesh to follow (linear blend
//! skinning on the CPU, like blendshapes, so painting keeps working on the
//! posed mesh: its UVs don't change).
//!
//! Poses come from other rigs (Mixamo, Quaternius, CMU mocap, …) through a
//! rig-neutral [`HumanPose`]: the world rotation and position of each
//! humanoid bone ([`BONES`]) in the source's rest pose and in the pose. The
//! bones are found by name and by where they sit in the hierarchy
//! ([`map_humanoid`]), so Mixamo's `mixamorig:LeftForeArm`, Unity's
//! `Left elbow` and Unreal's `lowerarm_l` all land on `LeftLowerArm`.
//! [`retarget`] then turns the pose into the target's local rotations,
//! correcting for differing rest poses (a T-pose clip on an A-pose avatar)
//! and facing.

use std::collections::BTreeMap;
use std::sync::Arc;

use glam::{EulerRot, Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::mesh::Vertex3D;

/// A local transform: translation, rotation, scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trs {
    pub t: Vec3,
    pub r: Quat,
    pub s: Vec3,
}

impl Default for Trs {
    fn default() -> Self {
        Trs { t: Vec3::ZERO, r: Quat::IDENTITY, s: Vec3::ONE }
    }
}

impl Trs {
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.s, self.r, self.t)
    }

    pub fn from_matrix(m: &Mat4) -> Trs {
        let (s, r, t) = m.to_scale_rotation_translation();
        Trs { t, r: r.normalize(), s }
    }
}

/// The rotation part of a world matrix (scale removed).
fn rotation_of(m: &Mat4) -> Quat {
    m.to_scale_rotation_translation().1.normalize()
}

#[derive(Clone, Debug)]
pub struct Joint {
    pub name: String,
    /// Always an earlier joint, so walking in order visits parents first.
    pub parent: Option<usize>,
    /// Local transform in the rest pose (the pose the mesh was bound in).
    pub rest: Trs,
    /// For a joint without a parent joint: the world transform of what it
    /// hangs from (the file's root, with its unit and axis conversion).
    pub base: Mat4,
}

#[derive(Clone, Debug, Default)]
pub struct Skeleton {
    pub joints: Vec<Joint>,
}

impl Skeleton {
    pub fn rest_locals(&self) -> Vec<Trs> {
        self.joints.iter().map(|j| j.rest).collect()
    }

    /// World matrices for these local transforms.
    pub fn world(&self, locals: &[Trs]) -> Vec<Mat4> {
        let mut world: Vec<Mat4> = Vec::with_capacity(self.joints.len());
        for (j, joint) in self.joints.iter().enumerate() {
            let parent = joint.parent.map_or(joint.base, |p| world[p]);
            world.push(parent * locals[j].matrix());
        }
        world
    }

    pub fn rest_world(&self) -> Vec<Mat4> {
        self.world(&self.rest_locals())
    }

    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.joints.iter().position(|j| j.name == name)
    }
}

/// Up to four bones moving one vertex: indices into the part's
/// [`Skin::joints`] and weights summing to 1 (all zero: the vertex doesn't
/// follow any bone).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Influence {
    pub slots: [u16; 4],
    pub weights: [f32; 4],
}

impl Influence {
    /// The four heaviest of `pairs` (slot, weight), normalised.
    pub fn strongest(pairs: &mut Vec<(u16, f32)>) -> Influence {
        pairs.retain(|p| p.1 > 0.0);
        pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
        pairs.truncate(4);
        let total: f32 = pairs.iter().map(|p| p.1).sum();
        let mut out = Influence::default();
        if total > 0.0 {
            for (k, &(slot, w)) in pairs.iter().enumerate() {
                out.slots[k] = slot;
                out.weights[k] = w / total;
            }
        }
        out
    }
}

/// How one model part follows the skeleton.
#[derive(Clone)]
pub struct Skin {
    pub skeleton: Arc<Skeleton>,
    /// Skeleton joint for each of the part's bone slots.
    pub joints: Vec<u32>,
    /// Per slot: from the part's geometry space to the bone's space in the
    /// rest pose (FBX's geometry-to-bone, glTF's inverse bind matrix).
    pub binds: Vec<Mat4>,
    /// Per vertex.
    pub influences: Arc<Vec<Influence>>,
    /// The undeformed vertices (with no blendshapes applied either).
    pub rest: Arc<Vec<Vertex3D>>,
    /// World matrices of the skeleton's joints in the current pose; None is
    /// the rest pose, which leaves the mesh exactly as it was bound.
    pub pose: Option<Arc<Vec<Mat4>>>,
}

impl Skin {
    /// Bend `vertices` (in the part's geometry space, which `transform`
    /// takes to world) to the current pose. Unchanged in the rest pose.
    pub fn deform(&self, vertices: &mut [Vertex3D], transform: &Mat4) {
        let Some(pose) = &self.pose else { return };
        let to_part = transform.inverse();
        // Geometry space → the bone (bind) → posed world → back to the
        // part's space, which the renderer takes to world again.
        let matrices: Vec<Mat4> = self
            .joints
            .iter()
            .zip(&self.binds)
            .map(|(&j, bind)| pose.get(j as usize).map_or(Mat4::IDENTITY, |w| to_part * *w * *bind))
            .collect();
        for (v, inf) in vertices.iter_mut().zip(self.influences.iter()) {
            if inf.weights[0] == 0.0 {
                continue;
            }
            let p = Vec3::from(v.position);
            let n = Vec3::from(v.normal);
            let (mut pp, mut nn) = (Vec3::ZERO, Vec3::ZERO);
            for k in 0..4 {
                let w = inf.weights[k];
                if w == 0.0 {
                    break;
                }
                let m = &matrices[inf.slots[k] as usize];
                pp += m.transform_point3(p) * w;
                nn += m.transform_vector3(n) * w;
            }
            v.position = pp.to_array();
            v.normal = nn.normalize_or(n).to_array();
        }
    }

    /// The skin of a part whose vertices were rebuilt: new vertex `j` is
    /// old vertex `sources[j]`, with the attributes in `vertices` (new UVs).
    pub fn remapped(&self, sources: &[u32], vertices: &[Vertex3D]) -> Skin {
        let influences = sources.iter().map(|&old| self.influences.get(old as usize).copied().unwrap_or_default()).collect();
        let rest = sources
            .iter()
            .zip(vertices)
            .map(|(&old, now)| Vertex3D { uv: now.uv, ..self.rest.get(old as usize).copied().unwrap_or(*now) })
            .collect();
        Skin { influences: Arc::new(influences), rest: Arc::new(rest), ..self.clone() }
    }
}

// --- humanoid bones -----------------------------------------------------------

/// The humanoid bones a pose moves, in Unity's naming. Fingers run 1 (at the
/// knuckle) to 3 (the tip).
pub const BONES: [&str; 52] = [
    "Hips", "Spine", "Chest", "UpperChest", "Neck", "Head",
    "LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand",
    "RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
    "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes",
    "LeftThumb1", "LeftThumb2", "LeftThumb3", "LeftIndex1", "LeftIndex2", "LeftIndex3",
    "LeftMiddle1", "LeftMiddle2", "LeftMiddle3", "LeftRing1", "LeftRing2", "LeftRing3",
    "LeftLittle1", "LeftLittle2", "LeftLittle3",
    "RightThumb1", "RightThumb2", "RightThumb3", "RightIndex1", "RightIndex2", "RightIndex3",
    "RightMiddle1", "RightMiddle2", "RightMiddle3", "RightRing1", "RightRing2", "RightRing3",
    "RightLittle1", "RightLittle2", "RightLittle3",
];

pub const HIPS: usize = 0;
const SPINE: usize = 1;
const CHEST: usize = 2;
const UPPER_CHEST: usize = 3;
const NECK: usize = 4;
pub const HEAD: usize = 5;
/// Shoulder, upper arm, lower arm, hand: left then right.
const ARMS: [[usize; 4]; 2] = [[6, 7, 8, 9], [10, 11, 12, 13]];
/// Upper leg, lower leg, foot, toes: left then right.
const LEGS: [[usize; 4]; 2] = [[14, 15, 16, 17], [18, 19, 20, 21]];
/// First finger bone of each hand (thumb, index, middle, ring, little × 3).
const FINGERS: [usize; 2] = [22, 37];

#[cfg(test)]
pub fn bone_index(name: &str) -> Option<usize> {
    BONES.iter().position(|b| *b == name)
}

/// The same bone on the other side (itself for the middle line).
pub fn mirror_bone(b: usize) -> usize {
    match b {
        6..=9 => b + 4,
        10..=13 => b - 4,
        14..=17 => b + 4,
        18..=21 => b - 4,
        22..=36 => b + 15,
        37..=51 => b - 15,
        _ => b,
    }
}

/// The bones a bone points at, in order of preference: the first one both
/// rigs have gives its direction.
fn aim_targets(b: usize) -> Vec<usize> {
    match b {
        HIPS => vec![SPINE, CHEST, UPPER_CHEST, NECK],
        SPINE => vec![CHEST, UPPER_CHEST, NECK],
        CHEST => vec![UPPER_CHEST, NECK],
        UPPER_CHEST => vec![NECK],
        NECK => vec![HEAD],
        HEAD => vec![],
        _ => {
            for side in 0..2 {
                let arm = ARMS[side];
                if let Some(k) = arm.iter().position(|&x| x == b) {
                    if k < 3 {
                        return vec![arm[k + 1]];
                    }
                    // Hand → middle finger, else index, else ring.
                    let f = FINGERS[side];
                    return vec![f + 6, f + 3, f + 9];
                }
                let leg = LEGS[side];
                if let Some(k) = leg.iter().position(|&x| x == b) {
                    return if k < 3 { vec![leg[k + 1]] } else { vec![] };
                }
                let f = FINGERS[side];
                if (f..f + 15).contains(&b) {
                    return if (b - f) % 3 < 2 { vec![b + 1] } else { vec![] };
                }
            }
            vec![]
        }
    }
}

/// Which skeleton joint plays each humanoid bone.
pub type HumanMap = [Option<usize>; 52];

/// Lowercase words of a bone name, with namespaces (`mixamorig:`) dropped
/// and camelCase, digits and separators split: `J_Bip_L_UpperArm` →
/// `j bip l upper arm`, `LeftHandIndex1` → `left hand index 1`.
fn words(name: &str) -> Vec<String> {
    let name = name.rsplit(':').next().unwrap_or(name);
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut prev: Option<char> = None;
    for c in name.chars() {
        let split = match prev {
            None => false,
            Some(p) => {
                !c.is_alphanumeric()
                    || (c.is_uppercase() && p.is_lowercase())
                    || (c.is_ascii_digit() != p.is_ascii_digit())
            }
        };
        if split && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        if c.is_alphanumeric() {
            cur.extend(c.to_lowercase());
        }
        prev = Some(c);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    // "upperarm", "forearm", "upleg", "toebase" stay whole; "upper arm" is
    // joined so both spellings read the same.
    let mut joined: Vec<String> = Vec::new();
    for w in out {
        if let Some(last) = joined.last_mut()
            && matches!((last.as_str(), w.as_str()), ("upper" | "lower" | "fore" | "up", "arm" | "leg") | ("toe", "base"))
        {
            last.push_str(&w);
            continue;
        }
        joined.push(w);
    }
    joined
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Side {
    Left,
    Right,
}

fn side_of(name: &str, words: &[String]) -> Option<Side> {
    for w in words {
        match w.as_str() {
            "left" | "l" => return Some(Side::Left),
            "right" | "r" => return Some(Side::Right),
            _ => {}
        }
    }
    // Daz: lShldrBend, rForearmBend.
    let base = name.rsplit(':').next().unwrap_or(name);
    let mut chars = base.chars();
    match (chars.next(), chars.next()) {
        (Some('l'), Some(c)) if c.is_uppercase() => Some(Side::Left),
        (Some('r'), Some(c)) if c.is_uppercase() => Some(Side::Right),
        _ => None,
    }
}

/// Helper bones that aren't part of the posed body.
fn is_helper(words: &[String]) -> bool {
    words.iter().any(|w| {
        matches!(
            w.as_str(),
            "end" | "nub" | "top" | "tip" | "twist" | "roll" | "ik" | "fk" | "pole" | "target" | "ctrl" | "mch" | "org"
                | "helper" | "metacarpal" | "palm" | "socket" | "jiggle" | "breast" | "eye" | "hair"
        )
    })
}

fn has(words: &[String], any: &[&str]) -> bool {
    words.iter().any(|w| any.contains(&w.as_str()))
}

fn finger_of(words: &[String]) -> Option<usize> {
    if has(words, &["thumb"]) {
        Some(0)
    } else if has(words, &["index", "pointer"]) {
        Some(1)
    } else if has(words, &["middle", "mid"]) {
        Some(2)
    } else if has(words, &["ring"]) {
        Some(3)
    } else if has(words, &["little", "pinky", "pinkie", "small"]) {
        Some(4)
    } else {
        None
    }
}

/// Find the humanoid bones of a skeleton (`names` and `parents` of its
/// joints, parents first). Bones are found from the hands, feet and head
/// inwards: whatever sits between a hand and the spine is the arm, so rigs
/// that call the upper arm "arm", "upperarm", "shldr" or "Left arm" all map.
pub fn map_humanoid(names: &[String], parents: &[Option<usize>]) -> HumanMap {
    let mut map: HumanMap = [None; 52];
    let w: Vec<Vec<String>> = names.iter().map(|n| words(n)).collect();
    let children = |j: usize| (0..names.len()).filter(move |&c| parents[c] == Some(j));
    let ancestors = |mut j: usize| {
        let mut out = Vec::new();
        while let Some(p) = parents[j] {
            out.push(p);
            j = p;
        }
        out
    };
    let depth = |j: usize| ancestors(j).len();

    let pick = |pred: &dyn Fn(usize) -> bool| -> Vec<usize> { (0..names.len()).filter(|&j| pred(j)).collect() };
    let hands = pick(&|j| has(&w[j], &["hand", "wrist"]) && finger_of(&w[j]).is_none() && !has(&w[j], &["finger"]) && !is_helper(&w[j]));
    let feet = pick(&|j| has(&w[j], &["foot", "ankle"]) && !has(&w[j], &["toe", "toes", "toebase", "ball"]) && !is_helper(&w[j]));
    let heads = pick(&|j| has(&w[j], &["head"]) && !is_helper(&w[j]));
    let by_side = |list: &[usize], side: Side| -> Option<usize> {
        // The shallowest one on that side: "hand" over "hand_ik_target" leftovers.
        list.iter().copied().filter(|&j| side_of(&names[j], &w[j]) == Some(side)).min_by_key(|&j| depth(j))
    };
    let foot = [by_side(&feet, Side::Left), by_side(&feet, Side::Right)];
    let hand = [by_side(&hands, Side::Left), by_side(&hands, Side::Right)];
    let Some(head) = heads.iter().copied().min_by_key(|&j| depth(j)) else { return map };
    let (Some(lf), Some(rf)) = (foot[0], foot[1]) else { return map };

    // Hips: where both legs and the spine meet.
    let lowest_common = |a: usize, b: usize| -> Option<usize> {
        let up_a: Vec<usize> = std::iter::once(a).chain(ancestors(a)).collect();
        std::iter::once(b).chain(ancestors(b)).find(|x| up_a.contains(x))
    };
    let Some(legs_meet) = lowest_common(lf, rf) else { return map };
    let Some(hips) = lowest_common(legs_meet, head) else { return map };
    map[HIPS] = Some(hips);
    map[HEAD] = Some(head);

    // Spine: the chain from the hips to the head, neck last.
    let mut chain: Vec<usize> = ancestors(head).into_iter().take_while(|&j| j != hips).collect();
    chain.reverse();
    chain.retain(|&j| !is_helper(&w[j]));
    if let Some(at) = chain.iter().position(|&j| has(&w[j], &["neck"])) {
        map[NECK] = Some(chain[at]);
        chain.truncate(at);
    } else if chain.len() >= 2 {
        map[NECK] = chain.pop();
    }
    match chain.len() {
        0 => {}
        1 => map[SPINE] = Some(chain[0]),
        2 => {
            map[SPINE] = Some(chain[0]);
            map[CHEST] = Some(chain[1]);
        }
        n => {
            map[SPINE] = Some(chain[0]);
            map[CHEST] = Some(chain[n / 2]);
            map[UPPER_CHEST] = Some(chain[n - 1]);
        }
    }
    let on_spine: Vec<usize> = std::iter::once(hips).chain(chain.iter().copied()).chain(map[NECK]).collect();

    // Limbs: walk up from the hand or foot, skipping twist and roll bones.
    let limb = |end: usize, stop: &dyn Fn(usize) -> bool| -> Vec<usize> {
        let mut out = vec![end];
        let mut j = end;
        while let Some(p) = parents[j] {
            if stop(p) {
                break;
            }
            if !is_helper(&w[p]) {
                out.push(p);
            }
            j = p;
        }
        out
    };
    for side in 0..2 {
        if let Some(h) = hand[side] {
            let up = limb(h, &|p| on_spine.contains(&p));
            let [shoulder, upper, lower, hand_bone] = ARMS[side];
            map[hand_bone] = Some(up[0]);
            map[lower] = up.get(1).copied();
            map[upper] = up.get(2).copied();
            map[shoulder] = up.get(3).copied();
            // Fingers: the chains under the hand, by name.
            let mut stack: Vec<(usize, usize)> = children(up[0]).map(|c| (c, 0)).collect();
            let mut found: [Vec<usize>; 5] = Default::default();
            while let Some((j, d)) = stack.pop() {
                if d > 5 {
                    continue;
                }
                if let Some(f) = finger_of(&w[j])
                    && !is_helper(&w[j])
                {
                    found[f].push(j);
                }
                stack.extend(children(j).map(|c| (c, d + 1)));
            }
            for (f, list) in found.iter_mut().enumerate() {
                list.sort_by_key(|&j| depth(j));
                for (k, &j) in list.iter().take(3).enumerate() {
                    map[FINGERS[side] + f * 3 + k] = Some(j);
                }
            }
        }
        let ft = foot[side].unwrap();
        let up = limb(ft, &|p| p == hips);
        let [upper, lower, foot_bone, toes] = LEGS[side];
        map[foot_bone] = Some(up[0]);
        map[lower] = up.get(1).copied();
        map[upper] = up.get(2).copied();
        map[toes] = children(ft).find(|&c| has(&w[c], &["toe", "toes", "toebase", "ball"]) && !is_helper(&w[c]));
    }
    map
}

// --- rig-neutral poses ----------------------------------------------------------

/// One humanoid bone of a pose, in the source rig's world space: its rest
/// and posed rotation and position.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct BoneSample {
    pub rest_rot: [f32; 4],
    pub rest_pos: [f32; 3],
    pub rot: [f32; 4],
    pub pos: [f32; 3],
}

/// A pose any humanoid can take: what each bone of the source did, relative
/// to the source's rest pose. Saved as JSON in the pose library.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct HumanPose {
    pub name: String,
    /// Where it came from: a file and clip, or "built in".
    #[serde(default)]
    pub source: String,
    /// By [`BONES`] name.
    pub bones: BTreeMap<String, BoneSample>,
}

impl HumanPose {
    /// Sample a source skeleton: its `rest` and `posed` world matrices.
    pub fn sample(name: &str, source: &str, map: &HumanMap, rest: &[Mat4], posed: &[Mat4]) -> HumanPose {
        let mut bones = BTreeMap::new();
        for (b, joint) in map.iter().enumerate() {
            let Some(j) = *joint else { continue };
            bones.insert(
                BONES[b].to_string(),
                BoneSample {
                    rest_rot: rotation_of(&rest[j]).to_array(),
                    rest_pos: rest[j].w_axis.truncate().to_array(),
                    rot: rotation_of(&posed[j]).to_array(),
                    pos: posed[j].w_axis.truncate().to_array(),
                },
            );
        }
        HumanPose { name: name.into(), source: source.into(), bones }
    }

    fn get(&self, b: usize) -> Option<&BoneSample> {
        self.bones.get(BONES[b])
    }
}

/// Rotation about Y turning a rig to face +Z, from where its legs (or arms)
/// sit: the character's left is +X when it faces +Z.
fn facing(pos: &dyn Fn(usize) -> Option<Vec3>) -> Quat {
    let pairs = [(LEGS[0][0], LEGS[1][0]), (ARMS[0][1], ARMS[1][1]), (LEGS[0][2], LEGS[1][2])];
    for (l, r) in pairs {
        if let (Some(l), Some(r)) = (pos(l), pos(r)) {
            let left = l - r;
            let forward = left.cross(Vec3::Y);
            if forward.length_squared() > 1e-10 {
                let yaw = forward.x.atan2(forward.z);
                return Quat::from_rotation_y(-yaw);
            }
        }
    }
    Quat::IDENTITY
}

/// Rotation about Y turning this rig (in its rest pose) to face +Z.
pub fn rig_facing(skeleton: &Skeleton, map: &HumanMap) -> Quat {
    let world = skeleton.rest_world();
    facing(&|b| map[b].map(|j| world[j].w_axis.truncate()))
}

/// The humanoid map of a skeleton.
pub fn humanoid_map(skeleton: &Skeleton) -> HumanMap {
    let names: Vec<String> = skeleton.joints.iter().map(|j| j.name.clone()).collect();
    let parents: Vec<Option<usize>> = skeleton.joints.iter().map(|j| j.parent).collect();
    map_humanoid(&names, &parents)
}

/// Reflect a rotation across the body's middle plane (x = 0 when facing +Z).
fn mirror_rot(q: Quat) -> Quat {
    Quat::from_xyzw(q.x, -q.y, -q.z, q.w)
}

/// Where each bone points in the canonical T-pose (facing +Z, the body's
/// left +X): up the spine, arms and fingers straight out, legs down, feet
/// forward. None for bones with nothing to point at (head, toes, finger
/// tips), which turn with their parent.
fn t_direction(b: usize) -> Option<Vec3> {
    match b {
        HIPS | SPINE | CHEST | UPPER_CHEST | NECK => Some(Vec3::Y),
        6..=9 => Some(Vec3::X),
        10..=13 => Some(-Vec3::X),
        14 | 15 | 18 | 19 => Some(-Vec3::Y),
        16 | 20 => Some(Vec3::new(0.0, -0.4, 0.92).normalize()),
        _ => {
            for side in 0..2 {
                let f = FINGERS[side];
                if (f..f + 15).contains(&b) {
                    let x = if side == 0 { 1.0 } else { -1.0 };
                    let k = b - f;
                    return match (k / 3, k % 3) {
                        (_, 2) => None,
                        (0, _) => Some(Vec3::new(x, 0.0, 1.0).normalize()),
                        _ => Some(Vec3::new(x, 0.0, 0.0)),
                    };
                }
            }
            None
        }
    }
}

/// The canonical parent of a humanoid bone (where a leaf takes its turn
/// from), skipping bones the rig lacks is left to the caller.
fn bone_parent(b: usize) -> Option<usize> {
    match b {
        HIPS => None,
        SPINE | 14 | 18 => Some(HIPS),
        CHEST => Some(SPINE),
        UPPER_CHEST => Some(CHEST),
        NECK => Some(UPPER_CHEST),
        HEAD => Some(NECK),
        6 | 10 => Some(UPPER_CHEST),
        7..=9 | 11..=13 | 15..=17 | 19..=21 => Some(b - 1),
        _ => {
            for side in 0..2 {
                let f = FINGERS[side];
                if (f..f + 15).contains(&b) {
                    return Some(if (b - f) % 3 == 0 { ARMS[side][3] } else { b - 1 });
                }
            }
            None
        }
    }
}

/// World rotations a rig's humanoid bones would have in the canonical
/// T-pose: each rest rotation swung so the bone points its T-pose way
/// (`pos` and `rot` are rest positions and rotations, canonical frame).
fn t_pose_rotations(pos: &dyn Fn(usize) -> Option<Vec3>, rot: &dyn Fn(usize) -> Option<Quat>) -> [Option<Quat>; 52] {
    let mut swing: [Option<Quat>; 52] = [None; 52];
    let mut out: [Option<Quat>; 52] = [None; 52];
    // Parents come first in BONES order (fingers after hands).
    for b in 0..BONES.len() {
        let Some(r) = rot(b) else { continue };
        let aim = t_direction(b).and_then(|t| {
            let p = pos(b)?;
            let c = aim_targets(b).into_iter().find_map(pos)?;
            let d = c - p;
            (d.length_squared() > 1e-12).then(|| Quat::from_rotation_arc(d.normalize(), t))
        });
        let inherited = || {
            let mut p = bone_parent(b);
            while let Some(q) = p {
                if let Some(s) = swing[q] {
                    return s;
                }
                p = bone_parent(q);
            }
            Quat::IDENTITY
        };
        let s = aim.unwrap_or_else(inherited);
        swing[b] = Some(s);
        out[b] = Some((s * r).normalize());
    }
    out
}

/// Photographer's adjustments on top of a pose, per humanoid bone: degrees
/// of nod (X), turn (Y) and tilt (Z) in the body's frame. A tweak carries
/// everything below the bone, so turning the chest turns the head with it.
pub type Tweaks = BTreeMap<String, [f32; 3]>;

/// Local transforms that put `target` in `pose` (None: its rest pose), with
/// `tweaks` on top. `mirror` swaps left and right.
pub fn retarget(pose: Option<&HumanPose>, target: &Skeleton, map: &HumanMap, mirror: bool, tweaks: &Tweaks) -> Vec<Trs> {
    let rest_world = target.rest_world();
    let mut locals = target.rest_locals();
    let t_pos = |b: usize| map[b].map(|j| rest_world[j].w_axis.truncate());
    let t_rot = |b: usize| map[b].map(|j| rotation_of(&rest_world[j]));
    let yt = facing(&t_pos);
    let yt_inv = yt.inverse();

    let mut desired_rot: [Option<Quat>; 52] = [None; 52];
    let mut hips_pos: Option<Vec3> = None;
    if let Some(pose) = pose {
        let s_pos = |b: usize| pose.get(b).map(|s| Vec3::from(s.rest_pos));
        let s_rot = |b: usize| pose.get(b).map(|s| Quat::from_array(s.rest_rot).normalize());
        let ys = facing(&s_pos);
        // Both rigs straightened into the canonical T-pose: whatever pose
        // each rests in (a T, an A, or an animation's first frame), a
        // bone's change from its T-pose is what carries over.
        let s_t = t_pose_rotations(&|b| s_pos(b).map(|p| ys * p), &|b| s_rot(b).map(|q| ys * q));
        let t_t = t_pose_rotations(&|b| t_pos(b).map(|p| yt * p), &|b| t_rot(b).map(|q| yt * q));
        for b in 0..BONES.len() {
            let sb = if mirror { mirror_bone(b) } else { b };
            let (Some(_), Some(s), Some(from_t), Some(to_t)) = (map[b], pose.get(sb), s_t[sb], t_t[b]) else { continue };
            // The source bone's turn from its T-pose, in the canonical frame.
            let delta = ys * Quat::from_array(s.rot).normalize() * from_t.inverse();
            let delta = if mirror { mirror_rot(delta) } else { delta };
            desired_rot[b] = Some((yt_inv * delta * to_t).normalize());
        }
        // The hips keep their place over the floor, at the source's height
        // above its floor scaled by leg length (no drifting off with an
        // animation's root motion).
        let hips_s = if mirror { mirror_bone(HIPS) } else { HIPS };
        if let (Some(j), Some(s)) = (map[HIPS], pose.get(hips_s)) {
            let floor = |pos: &dyn Fn(usize) -> Option<Vec3>| {
                [LEGS[0][2], LEGS[1][2], LEGS[0][3], LEGS[1][3]].iter().filter_map(|&b| pos(b)).map(|p| p.y).reduce(f32::min)
            };
            let leg = |pos: &dyn Fn(usize) -> Option<Vec3>| -> Option<f32> {
                let [u, l, f, _] = LEGS[0];
                Some((pos(l)? - pos(u)?).length() + (pos(f)? - pos(l)?).length()).filter(|v| *v > 1e-6)
            };
            let t_hips = rest_world[j].w_axis.truncate();
            if let (Some(tf), Some(sf), Some(tl), Some(sl)) = (floor(&t_pos), floor(&s_pos), leg(&t_pos), leg(&s_pos)) {
                hips_pos = Some(Vec3::new(t_hips.x, tf + (s.pos[1] - sf) * tl / sl, t_hips.z));
            }
        }
    }

    let bone_of: Vec<Option<usize>> = {
        let mut v = vec![None; target.joints.len()];
        for (b, j) in map.iter().enumerate() {
            if let Some(j) = j {
                v[*j] = Some(b);
            }
        }
        v
    };
    // Pass 1: the pose. Pass 2: the tweaks, which children follow.
    for pass in 0..2 {
        if pass == 1 && tweaks.is_empty() {
            break;
        }
        let mut world: Vec<Mat4> = Vec::with_capacity(target.joints.len());
        for (j, joint) in target.joints.iter().enumerate() {
            let parent = joint.parent.map_or(joint.base, |p| world[p]);
            let parent_rot = rotation_of(&parent);
            if let Some(b) = bone_of[j] {
                if pass == 0 {
                    if let Some(q) = desired_rot[b] {
                        locals[j].r = (parent_rot.inverse() * q).normalize();
                    }
                    if b == HIPS
                        && let Some(p) = hips_pos
                    {
                        locals[j].t = parent.inverse().transform_point3(p);
                    }
                } else if let Some(&[x, y, z]) = tweaks.get(BONES[b]) {
                    let turn = Quat::from_euler(EulerRot::YXZ, y.to_radians(), x.to_radians(), z.to_radians());
                    let now = parent_rot * locals[j].r;
                    let q = yt_inv * turn * yt * now;
                    locals[j].r = (parent_rot.inverse() * q).normalize();
                }
            }
            world.push(parent * locals[j].matrix());
        }
    }
    locals
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A little humanoid facing +Z, metres, rest pose with arms out
    /// (T-pose) or down at 45° (A-pose). Joint names from `names`.
    fn humanoid(names: &dyn Fn(&str) -> String, a_pose: bool) -> Skeleton {
        let mut joints: Vec<Joint> = Vec::new();
        let add = |name: &str, parent: Option<&str>, t: Vec3, joints: &mut Vec<Joint>| {
            let parent = parent.map(|p| joints.iter().position(|j| j.name == names(p)).unwrap());
            joints.push(Joint { name: names(name), parent, rest: Trs { t, ..Default::default() }, base: Mat4::IDENTITY });
        };
        add("Hips", None, Vec3::new(0.0, 1.0, 0.0), &mut joints);
        add("Spine", Some("Hips"), Vec3::new(0.0, 0.1, 0.0), &mut joints);
        add("Chest", Some("Spine"), Vec3::new(0.0, 0.15, 0.0), &mut joints);
        add("Neck", Some("Chest"), Vec3::new(0.0, 0.2, 0.0), &mut joints);
        add("Head", Some("Neck"), Vec3::new(0.0, 0.1, 0.0), &mut joints);
        let arm = if a_pose { Vec3::new(0.7071, -0.7071, 0.0) } else { Vec3::X };
        for (side, sx) in [("Left", 1.0), ("Right", -1.0)] {
            let s = |n: &str| format!("{side}{n}");
            let mirror = Vec3::new(sx, 1.0, 1.0);
            add(&s("Shoulder"), Some("Chest"), Vec3::new(0.05 * sx, 0.18, 0.0), &mut joints);
            add(&s("UpperArm"), Some(&s("Shoulder")), Vec3::new(0.1 * sx, 0.0, 0.0), &mut joints);
            add(&s("LowerArm"), Some(&s("UpperArm")), arm * mirror * 0.25, &mut joints);
            add(&s("Hand"), Some(&s("LowerArm")), arm * mirror * 0.25, &mut joints);
            add(&s("Middle1"), Some(&s("Hand")), arm * mirror * 0.08, &mut joints);
            add(&s("UpperLeg"), Some("Hips"), Vec3::new(0.1 * sx, -0.05, 0.0), &mut joints);
            add(&s("LowerLeg"), Some(&s("UpperLeg")), Vec3::new(0.0, -0.45, 0.0), &mut joints);
            add(&s("Foot"), Some(&s("LowerLeg")), Vec3::new(0.0, -0.45, 0.0), &mut joints);
            add(&s("Toes"), Some(&s("Foot")), Vec3::new(0.0, -0.05, 0.12), &mut joints);
        }
        Skeleton { joints }
    }

    fn names_and_parents(s: &Skeleton) -> (Vec<String>, Vec<Option<usize>>) {
        (s.joints.iter().map(|j| j.name.clone()).collect(), s.joints.iter().map(|j| j.parent).collect())
    }

    fn mapped(s: &Skeleton) -> HumanMap {
        let (n, p) = names_and_parents(s);
        map_humanoid(&n, &p)
    }

    fn found(s: &Skeleton, map: &HumanMap, bone: &str) -> Option<String> {
        map[bone_index(bone).unwrap()].map(|j| s.joints[j].name.clone())
    }

    /// Mixamo, Unity/VRChat (CATS), Unreal and VRoid names all map to the
    /// same bones, by name and by place in the hierarchy.
    #[test]
    fn humanoid_bones_map_across_naming_schemes() {
        let mixamo = |n: &str| {
            let n = n
                .replace("UpperArm", "Arm")
                .replace("LowerArm", "ForeArm")
                .replace("UpperLeg", "UpLeg")
                .replace("LowerLeg", "Leg")
                .replace("Toes", "ToeBase")
                .replace("Chest", "Spine1")
                .replace("Middle1", "HandMiddle1");
            format!("mixamorig:{n}")
        };
        let s = humanoid(&mixamo, false);
        let m = mapped(&s);
        assert_eq!(found(&s, &m, "LeftUpperArm").as_deref(), Some("mixamorig:LeftArm"));
        assert_eq!(found(&s, &m, "LeftLowerArm").as_deref(), Some("mixamorig:LeftForeArm"));
        assert_eq!(found(&s, &m, "LeftLowerLeg").as_deref(), Some("mixamorig:LeftLeg"), "Mixamo's Leg is the shin");
        assert_eq!(found(&s, &m, "RightUpperLeg").as_deref(), Some("mixamorig:RightUpLeg"));
        assert_eq!(found(&s, &m, "LeftMiddle1").as_deref(), Some("mixamorig:LeftHandMiddle1"));
        assert_eq!(found(&s, &m, "LeftHand").as_deref(), Some("mixamorig:LeftHand"));
        assert_eq!(found(&s, &m, "RightToes").as_deref(), Some("mixamorig:RightToeBase"));

        let cats = |n: &str| match n {
            "Hips" | "Spine" | "Chest" | "Neck" | "Head" => n.to_string(),
            _ => {
                let (side, rest) = if let Some(r) = n.strip_prefix("Left") { ("Left", r) } else { ("Right", n.strip_prefix("Right").unwrap()) };
                let part = match rest {
                    "Shoulder" => "shoulder",
                    "UpperArm" => "arm",
                    "LowerArm" => "elbow",
                    "Hand" => "wrist",
                    "Middle1" => "MiddleFinger1",
                    "UpperLeg" => "leg",
                    "LowerLeg" => "knee",
                    "Foot" => "ankle",
                    "Toes" => "toe",
                    _ => unreachable!(),
                };
                if part.starts_with("Middle") { format!("{part}_{}", &side[..1]) } else { format!("{side} {part}") }
            }
        };
        let s = humanoid(&cats, true);
        let m = mapped(&s);
        assert_eq!(found(&s, &m, "LeftUpperLeg").as_deref(), Some("Left leg"), "CATS' leg is the thigh");
        assert_eq!(found(&s, &m, "LeftLowerLeg").as_deref(), Some("Left knee"));
        assert_eq!(found(&s, &m, "RightLowerArm").as_deref(), Some("Right elbow"));
        assert_eq!(found(&s, &m, "RightHand").as_deref(), Some("Right wrist"));
        assert_eq!(found(&s, &m, "LeftMiddle1").as_deref(), Some("MiddleFinger1_L"));
        assert_eq!(found(&s, &m, "Chest").as_deref(), Some("Chest"));

        let unreal = |n: &str| match n {
            "Hips" => "pelvis".into(),
            "Spine" => "spine_01".into(),
            "Chest" => "spine_02".into(),
            "Neck" => "neck_01".into(),
            "Head" => "head".into(),
            _ => {
                let (s, rest) = if let Some(r) = n.strip_prefix("Left") { ("l", r) } else { ("r", n.strip_prefix("Right").unwrap()) };
                let part = match rest {
                    "Shoulder" => "clavicle",
                    "UpperArm" => "upperarm",
                    "LowerArm" => "lowerarm",
                    "Hand" => "hand",
                    "Middle1" => "middle_01",
                    "UpperLeg" => "thigh",
                    "LowerLeg" => "calf",
                    "Foot" => "foot",
                    "Toes" => "ball",
                    _ => unreachable!(),
                };
                format!("{part}_{s}")
            }
        };
        let s = humanoid(&unreal, false);
        let m = mapped(&s);
        assert_eq!(found(&s, &m, "Hips").as_deref(), Some("pelvis"));
        assert_eq!(found(&s, &m, "LeftShoulder").as_deref(), Some("clavicle_l"));
        assert_eq!(found(&s, &m, "RightLowerLeg").as_deref(), Some("calf_r"));
        assert_eq!(found(&s, &m, "LeftToes").as_deref(), Some("ball_l"));

        let vroid = |n: &str| match n {
            "Hips" | "Spine" | "Chest" | "Neck" | "Head" => format!("J_Bip_C_{n}"),
            _ => {
                let (s, rest) = if let Some(r) = n.strip_prefix("Left") { ("L", r) } else { ("R", n.strip_prefix("Right").unwrap()) };
                format!("J_Bip_{s}_{rest}")
            }
        };
        let s = humanoid(&vroid, false);
        let m = mapped(&s);
        assert_eq!(found(&s, &m, "LeftUpperArm").as_deref(), Some("J_Bip_L_UpperArm"));
        assert_eq!(found(&s, &m, "RightUpperLeg").as_deref(), Some("J_Bip_R_UpperLeg"));
        assert_eq!(found(&s, &m, "Neck").as_deref(), Some("J_Bip_C_Neck"));
    }

    fn world_pos(s: &Skeleton, locals: &[Trs], name: &str) -> Vec3 {
        let j = s.index_of(name).unwrap();
        s.world(locals)[j].w_axis.truncate()
    }

    /// A pose sampled from one rig and put on another puts the bones in
    /// the same directions, even when the rest poses differ (T vs A) and
    /// the rigs face different ways.
    #[test]
    fn poses_retarget_across_rest_poses_and_facing() {
        let plain = |n: &str| n.to_string();
        let source = humanoid(&plain, false);
        let smap = mapped(&source);
        // Source pose: left arm raised straight up, left knee bent forward.
        let mut locals = source.rest_locals();
        let ua = source.index_of("LeftUpperArm").unwrap();
        locals[ua].r = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let ll = source.index_of("LeftLowerLeg").unwrap();
        locals[ll].r = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
        let pose = HumanPose::sample("Test", "unit", &smap, &source.rest_world(), &source.world(&locals));

        // Target: A-pose, facing -X (turned -90° about Y), longer arms.
        let mut target = humanoid(&|n: &str| format!("mixamorig:{n}"), true);
        let turn = Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2);
        target.joints[0].rest.r = turn;
        let tmap = mapped(&target);
        let out = retarget(Some(&pose), &target, &tmap, false, &Tweaks::new());

        let dir = |a: &str, b: &str| (world_pos(&target, &out, b) - world_pos(&target, &out, a)).normalize();
        let up = dir("mixamorig:LeftLowerArm", "mixamorig:LeftHand");
        assert!(up.dot(Vec3::Y) > 0.99, "the raised arm points up on the A-pose rig: {up}");
        // The bent knee's shin points forward: the target faces -X.
        let shin = dir("mixamorig:LeftLowerLeg", "mixamorig:LeftFoot");
        assert!(shin.dot(turn * Vec3::Z) > 0.99, "the shin points the way the target faces: {shin}");
        // The right side didn't move from its rest.
        let rest = target.rest_locals();
        let r_arm = |l: &[Trs]| (world_pos(&target, l, "mixamorig:RightHand") - world_pos(&target, l, "mixamorig:RightLowerArm")).normalize();
        let right = r_arm(&out);
        let source_right = Vec3::new(-1.0, 0.0, 0.0);
        assert!((turn * source_right).dot(right) > 0.99, "the right arm takes the source's T-pose direction: {right} (rest {})", r_arm(&rest));

        // Mirrored: the right arm goes up instead.
        let out = retarget(Some(&pose), &target, &tmap, true, &Tweaks::new());
        let up = (world_pos(&target, &out, "mixamorig:RightHand") - world_pos(&target, &out, "mixamorig:RightLowerArm")).normalize();
        assert!(up.dot(Vec3::Y) > 0.99, "mirrored, the right arm is raised: {up}");
    }

    /// Animation files often rest in their first frame, not a T-pose: the
    /// pose still carries over as directions, not as a change from that
    /// frame.
    #[test]
    fn poses_from_clips_resting_in_their_first_frame() {
        let plain = |n: &str| n.to_string();
        let mut source = humanoid(&plain, false);
        // Rest: arm hanging down and forward, shin kicked back (a run frame).
        let ua = source.index_of("LeftUpperArm").unwrap();
        let ll = source.index_of("LeftLowerLeg").unwrap();
        source.joints[ua].rest.r = Quat::from_rotation_z(-1.3) * Quat::from_rotation_y(-0.4);
        source.joints[ll].rest.r = Quat::from_rotation_x(1.2);
        let smap = mapped(&source);
        // Pose: that arm straight up, the leg straight.
        let mut locals = source.rest_locals();
        locals[ua].r = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        locals[ll].r = Quat::IDENTITY;
        let pose = HumanPose::sample("Up", "unit", &smap, &source.rest_world(), &source.world(&locals));
        let target = humanoid(&plain, true);
        let out = retarget(Some(&pose), &target, &mapped(&target), false, &Tweaks::new());
        let dir = |a: &str, b: &str| (world_pos(&target, &out, b) - world_pos(&target, &out, a)).normalize();
        assert!(dir("LeftLowerArm", "LeftHand").dot(Vec3::Y) > 0.98, "arm up: {}", dir("LeftLowerArm", "LeftHand"));
        assert!(dir("LeftLowerLeg", "LeftFoot").dot(-Vec3::Y) > 0.98, "leg straight: {}", dir("LeftLowerLeg", "LeftFoot"));
        let feet = world_pos(&target, &out, "RightFoot").y;
        assert!((feet - world_pos(&target, &target.rest_locals(), "RightFoot").y).abs() < 0.02, "still standing on the floor: {feet}");
    }

    /// The rest pose with no pose and no tweaks is exactly the rest.
    #[test]
    fn no_pose_is_the_rest_pose() {
        let s = humanoid(&|n: &str| n.to_string(), false);
        let m = mapped(&s);
        assert_eq!(retarget(None, &s, &m, false, &Tweaks::new()), s.rest_locals());
    }

    /// A tweak turns the bone in the body's frame and carries its children.
    #[test]
    fn tweaks_carry_their_children() {
        let s = humanoid(&|n: &str| n.to_string(), false);
        let m = mapped(&s);
        let mut tweaks = Tweaks::new();
        tweaks.insert("Chest".into(), [0.0, 90.0, 0.0]);
        let out = retarget(None, &s, &m, false, &tweaks);
        let hand = world_pos(&s, &out, "LeftHand") - world_pos(&s, &out, "Chest");
        assert!(hand.z.abs() > 0.3 && hand.x.abs() < 0.1, "turning the chest 90° swings the arms round: {hand}");
    }

    /// Skinning follows the bone and leaves the rest pose untouched.
    #[test]
    fn skin_follows_its_bone() {
        let s = humanoid(&|n: &str| n.to_string(), false);
        let rest_world = s.rest_world();
        let ua = s.index_of("LeftUpperArm").unwrap();
        // A vertex at the elbow, in a part whose geometry space is world
        // space scaled by 100 (FBX centimetres).
        let transform = Mat4::from_scale(Vec3::splat(0.01));
        let elbow = rest_world[s.index_of("LeftLowerArm").unwrap()].w_axis.truncate() * 100.0;
        let bind = (transform.inverse() * rest_world[ua]).inverse();
        let v = Vertex3D { position: elbow.to_array(), normal: [0.0, 1.0, 0.0], uv: [0.0; 2] };
        let mut skin = Skin {
            skeleton: Arc::new(s.clone()),
            joints: vec![ua as u32],
            binds: vec![bind],
            influences: Arc::new(vec![Influence { slots: [0; 4], weights: [1.0, 0.0, 0.0, 0.0] }]),
            rest: Arc::new(vec![v]),
            pose: Some(Arc::new(rest_world.clone())),
        };
        let mut out = vec![v];
        skin.deform(&mut out, &transform);
        assert!((Vec3::from(out[0].position) - elbow).length() < 1e-3, "the rest pose doesn't move it");

        let mut locals = s.rest_locals();
        locals[ua].r = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        skin.pose = Some(Arc::new(s.world(&locals)));
        let mut out = vec![v];
        skin.deform(&mut out, &transform);
        let shoulder = rest_world[ua].w_axis.truncate() * 100.0;
        let moved = Vec3::from(out[0].position) - shoulder;
        assert!(moved.normalize().dot(Vec3::Y) > 0.999, "the elbow swings up with the arm: {moved}");
        assert!((Vec3::from(out[0].normal) - Vec3::new(-1.0, 0.0, 0.0)).length() < 1e-3, "normals turn too");
    }

    #[test]
    fn strongest_influences_are_normalised() {
        let mut pairs = vec![(0, 0.1), (1, 0.5), (2, 0.0), (3, 0.2), (4, 0.1), (5, 0.1)];
        let inf = Influence::strongest(&mut pairs);
        assert_eq!(inf.slots[0], 1);
        assert!((inf.weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(Influence::strongest(&mut vec![]).weights, [0.0; 4]);
    }
}
