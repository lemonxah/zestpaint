//! The photo studio around the avatar for path-traced renders: soft lights
//! placed around the subject (three-point and other classic setups), a
//! backdrop (floor, or a seamless sweep curving up into a wall), what shows
//! behind it (the environment, a colour, or nothing: a transparent PNG to
//! put a background behind later), and a lens with depth of field.
//!
//! [`Studio`] is what the panel edits and projects save; [`gpu_block`]
//! packs it, placed around a [`Subject`], into the tracer's uniform block.

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

/// Lights the tracer takes (its uniform block has room for this many).
pub const MAX_LIGHTS: usize = 8;
/// vec4s in the tracer's uniform block.
pub const BLOCK_VEC4S: usize = 26;

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[serde(rename_all = "snake_case")]
pub enum Background {
    /// The environment (HDRI or sky) shows behind.
    #[default]
    Environment,
    /// A flat colour.
    Color,
    /// Nothing: alpha 0, for compositing over a background later.
    Transparent,
}

impl Background {
    pub const ALL: [Background; 3] = [Background::Environment, Background::Color, Background::Transparent];

    pub fn label(self) -> &'static str {
        match self {
            Background::Environment => "Environment",
            Background::Color => "Colour",
            Background::Transparent => "Transparent (green screen)",
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[serde(rename_all = "snake_case")]
pub enum Backdrop {
    #[default]
    None,
    /// An endless floor under the feet.
    Floor,
    /// A photo-studio sweep: the floor curves up into a wall behind.
    Sweep,
}

impl Backdrop {
    pub const ALL: [Backdrop; 3] = [Backdrop::None, Backdrop::Floor, Backdrop::Sweep];

    pub fn label(self) -> &'static str {
        match self {
            Backdrop::None => "None",
            Backdrop::Floor => "Floor",
            Backdrop::Sweep => "Studio sweep",
        }
    }
}

/// A round soft light (a softbox seen from the subject), placed around the
/// subject: azimuth 0° in front of them, 90° at their left; elevation 0° at
/// the height of their middle.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct Light {
    pub enabled: bool,
    pub name: String,
    pub azimuth: f32,
    pub elevation: f32,
    /// Metres from the subject's middle.
    pub distance: f32,
    /// Radius in metres: bigger is softer.
    pub size: f32,
    /// sRGB.
    pub color: [f32; 3],
    /// How bright it lights the subject (1 ≈ a well-exposed key).
    pub intensity: f32,
}

impl Default for Light {
    fn default() -> Self {
        Light { enabled: true, name: "Light".into(), azimuth: 40.0, elevation: 30.0, distance: 2.5, size: 0.4, color: [1.0; 3], intensity: 1.0 }
    }
}

fn light(name: &str, azimuth: f32, elevation: f32, size: f32, color: [f32; 3], intensity: f32) -> Light {
    Light { name: name.into(), azimuth, elevation, size, color, intensity, ..Light::default() }
}

const WARM: [f32; 3] = [1.0, 0.93, 0.84];
const COOL: [f32; 3] = [0.82, 0.9, 1.0];
const WHITE: [f32; 3] = [1.0, 1.0, 1.0];

/// Classic lighting setups: (name, what it looks like, lights).
pub fn presets() -> Vec<(&'static str, &'static str, Vec<Light>)> {
    vec![
        (
            "Three-point",
            "Key light high to one side, soft fill opposite, rim behind to lift the edges",
            vec![light("Key", 40.0, 30.0, 0.5, WARM, 1.6), light("Fill", -50.0, 10.0, 0.9, WHITE, 0.45), light("Rim", 150.0, 35.0, 0.3, COOL, 1.4)],
        ),
        (
            "Rembrandt",
            "One strong key high at 45°: a triangle of light on the shadowed cheek",
            vec![light("Key", 50.0, 45.0, 0.45, WARM, 2.0), light("Fill", -60.0, 5.0, 1.0, WHITE, 0.15)],
        ),
        (
            "Butterfly",
            "Beauty light from straight ahead and above, a reflector below",
            vec![light("Key", 0.0, 40.0, 0.5, WHITE, 1.6), light("Reflector", 0.0, -25.0, 0.8, WARM, 0.35), light("Hair", 180.0, 55.0, 0.3, WHITE, 0.8)],
        ),
        (
            "Split",
            "Light from one side only: half the figure in shadow",
            vec![light("Key", 90.0, 10.0, 0.4, WHITE, 1.8), light("Kicker", -150.0, 20.0, 0.25, COOL, 0.6)],
        ),
        (
            "Rim / dramatic",
            "Two hard lights from behind outline the silhouette on a dark set",
            vec![light("Rim left", 135.0, 25.0, 0.2, COOL, 2.2), light("Rim right", -135.0, 25.0, 0.2, WARM, 2.2), light("Fill", 0.0, 0.0, 1.2, WHITE, 0.08)],
        ),
        (
            "Soft box",
            "Big, even light from both sides of the camera: flat and flattering",
            vec![light("Left", 35.0, 15.0, 1.2, WHITE, 0.9), light("Right", -35.0, 15.0, 1.2, WHITE, 0.9), light("Top", 0.0, 75.0, 1.0, WHITE, 0.4)],
        ),
        ("Environment only", "No studio lights: just the environment", vec![]),
    ]
}

/// The whole studio, as the panel edits it and projects save it.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct Studio {
    pub lights: Vec<Light>,
    /// Environment light, times the viewport's environment intensity.
    pub environment: f32,
    /// Degrees about the vertical.
    pub environment_rotation: f32,
    pub background: Background,
    /// sRGB.
    pub background_color: [f32; 3],
    pub backdrop: Backdrop,
    /// sRGB.
    pub backdrop_color: [f32; 3],
    pub backdrop_roughness: f32,
    /// How far behind the subject the sweep's wall stands, metres.
    pub backdrop_distance: f32,
    /// Radius of the sweep's curve, metres.
    pub backdrop_curve: f32,
    /// With a transparent background, keep the backdrop's shadows (in alpha).
    pub shadow_catcher: bool,
    /// A posing box to sit on, under the hips; part of the set (hidden with
    /// a transparent background, like the backdrop).
    pub seat: bool,
    /// Its height in metres; 0 fits it under the posed thighs.
    pub seat_height: f32,
    /// How far behind the hips its middle is, metres.
    pub seat_back: f32,
    pub depth_of_field: bool,
    pub f_stop: f32,
    /// Focus this far past (or before, negative) the subject's middle, metres.
    pub focus_offset: f32,
    /// Light bounces after the first hit.
    pub bounces: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub render_samples: u32,
}

impl Default for Studio {
    fn default() -> Self {
        Studio {
            lights: Vec::new(),
            environment: 1.0,
            environment_rotation: 0.0,
            background: Background::Environment,
            background_color: [0.18, 0.18, 0.2],
            backdrop: Backdrop::None,
            backdrop_color: [0.62, 0.62, 0.64],
            backdrop_roughness: 0.7,
            backdrop_distance: 2.0,
            backdrop_curve: 0.8,
            shadow_catcher: true,
            seat: false,
            seat_height: 0.0,
            seat_back: 0.05,
            depth_of_field: false,
            f_stop: 2.8,
            focus_offset: 0.0,
            bounces: 3,
            render_width: 1920,
            render_height: 1080,
            render_samples: 256,
        }
    }
}

impl Studio {
    /// A photo-studio starting point: three-point light on a grey sweep.
    pub fn photo_default() -> Studio {
        Studio {
            lights: presets().remove(0).2,
            environment: 0.25,
            backdrop: Backdrop::Sweep,
            ..Studio::default()
        }
    }
}

/// Where the subject stands: the studio is arranged around it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Subject {
    /// Middle of the visible models.
    pub center: Vec3,
    /// The floor: the lowest point of the models.
    pub floor: f32,
    /// Turns the subject to face +Z (from its skeleton; identity without).
    pub facing: Quat,
    /// Under the hips: x and z of the posed hips, y the height a seat
    /// should have to meet the thighs.
    pub seat: Vec3,
}

/// The camera a frame is traced from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lens {
    pub position: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub forward: Vec3,
    pub fov_y: f32,
}

pub fn srgb_to_linear(c: [f32; 3]) -> [f32; 3] {
    c.map(|v| if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) })
}

/// Where a light sits in world space.
pub fn light_position(l: &Light, subject: &Subject) -> Vec3 {
    let (az, el) = (l.azimuth.to_radians(), l.elevation.to_radians());
    let local = Vec3::new(az.sin() * el.cos(), el.sin(), az.cos() * el.cos());
    subject.center + subject.facing.inverse() * local * l.distance
}

/// The tracer's uniform block (see `pathtracer::trace_cs`):
/// 0–15 lights (position, radius; radiance, on), 16 backdrop origin, kind;
/// 17 its forward (x, z), wall distance, curve radius; 18 its albedo and
/// roughness; 19 background colour, mode; 20 environment strength,
/// rotation, shadow catcher, light count; 21 lens aperture radius, focus
/// distance; 22–24 camera right, up, forward; 25 the seat: on, height,
/// middle's x and z.
pub fn gpu_block(studio: &Studio, subject: &Subject, env_intensity: f32, lens: &Lens) -> [[f32; 4]; BLOCK_VEC4S] {
    let mut b = [[0.0f32; 4]; BLOCK_VEC4S];
    let lights: Vec<&Light> = studio.lights.iter().filter(|l| l.enabled && l.intensity > 0.0).take(MAX_LIGHTS).collect();
    for (k, l) in lights.iter().enumerate() {
        let p = light_position(l, subject);
        let r = l.size.max(0.01).min(l.distance * 0.9);
        // Radiance so that it lights the subject's middle with `intensity`
        // whatever its size and distance: E = L·π·r²/d².
        let radiance = l.intensity * l.distance * l.distance / (std::f32::consts::PI * r * r);
        let c = srgb_to_linear(l.color);
        b[k * 2] = [p.x, p.y, p.z, r];
        b[k * 2 + 1] = [c[0] * radiance, c[1] * radiance, c[2] * radiance, 1.0];
    }
    let forward = subject.facing.inverse() * Vec3::Z;
    let kind = match studio.backdrop {
        Backdrop::None => 0.0,
        Backdrop::Floor => 1.0,
        Backdrop::Sweep => 2.0,
    };
    b[16] = [subject.center.x, subject.floor, subject.center.z, kind];
    let curve = studio.backdrop_curve.clamp(0.0, studio.backdrop_distance.max(0.05));
    b[17] = [forward.x, forward.z, studio.backdrop_distance.max(0.05), curve];
    let a = srgb_to_linear(studio.backdrop_color);
    b[18] = [a[0], a[1], a[2], studio.backdrop_roughness.clamp(0.03, 1.0)];
    let bg = srgb_to_linear(studio.background_color);
    let mode = match studio.background {
        Background::Environment => 0.0,
        Background::Color => 1.0,
        Background::Transparent => 2.0,
    };
    b[19] = [bg[0], bg[1], bg[2], mode];
    b[20] = [
        env_intensity * studio.environment,
        studio.environment_rotation.to_radians(),
        if studio.shadow_catcher { 1.0 } else { 0.0 },
        lights.len() as f32,
    ];
    let focus = ((subject.center - lens.position).dot(lens.forward) + studio.focus_offset).max(0.05);
    // Full-frame equivalent: the sensor is 24 mm tall.
    let focal = 0.012 / (lens.fov_y * 0.5).tan();
    let aperture = if studio.depth_of_field { focal / (2.0 * studio.f_stop.max(0.5)) } else { 0.0 };
    b[21] = [aperture, focus, 0.0, 0.0];
    b[22] = lens.right.extend(0.0).to_array();
    b[23] = lens.up.extend(0.0).to_array();
    b[24] = lens.forward.extend(0.0).to_array();
    let height = if studio.seat_height > 0.0 { studio.seat_height } else { subject.seat.y.max(0.1) };
    let middle = Vec3::new(subject.seat.x, 0.0, subject.seat.z) - forward * studio.seat_back;
    b[25] = [if studio.seat { 1.0 } else { 0.0 }, height, middle.x, middle.z];
    b
}

/// Focal length (mm, full-frame equivalent) for a vertical field of view.
pub fn focal_length(fov_y: f32) -> f32 {
    12.0 / (fov_y * 0.5).tan()
}

pub fn fov_for_focal_length(mm: f32) -> f32 {
    2.0 * (12.0 / mm.max(1.0)).atan()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subject() -> Subject {
        Subject { center: Vec3::new(0.0, 1.0, 0.0), floor: 0.0, facing: Quat::IDENTITY, seat: Vec3::new(0.0, 0.5, 0.0) }
    }

    /// Lights sit around the subject: 0° in front, 90° at their left, and
    /// turn with the subject.
    #[test]
    fn lights_are_placed_around_the_subject() {
        let front = Light { azimuth: 0.0, elevation: 0.0, distance: 2.0, ..Light::default() };
        assert!((light_position(&front, &subject()) - Vec3::new(0.0, 1.0, 2.0)).length() < 1e-5);
        let left = Light { azimuth: 90.0, ..front.clone() };
        assert!((light_position(&left, &subject()) - Vec3::new(2.0, 1.0, 0.0)).length() < 1e-5);
        // A subject facing -X (turned -90°) has the front light at -X.
        let turned = Subject { facing: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2), ..subject() };
        assert!((light_position(&front, &turned) - Vec3::new(-2.0, 1.0, 0.0)).length() < 1e-5);
    }

    /// A light's radiance makes its irradiance at the subject its intensity,
    /// whatever its size or distance.
    #[test]
    fn light_intensity_is_irradiance_at_the_subject() {
        let lens = Lens { position: Vec3::new(0.0, 1.0, 3.0), right: Vec3::X, up: Vec3::Y, forward: -Vec3::Z, fov_y: 0.8 };
        for (size, distance) in [(0.2, 1.0), (1.0, 4.0)] {
            let studio = Studio { lights: vec![Light { size, distance, intensity: 2.0, ..Light::default() }], ..Studio::default() };
            let b = gpu_block(&studio, &subject(), 1.0, &lens);
            let r = b[0][3];
            let e = b[1][0] * std::f32::consts::PI * r * r / (distance * distance);
            assert!((e - 2.0).abs() < 1e-4, "size {size} at {distance} m gives {e}");
        }
        let off = Studio { lights: vec![Light { enabled: false, ..Light::default() }], ..Studio::default() };
        assert_eq!(gpu_block(&off, &subject(), 1.0, &lens)[20][3], 0.0, "disabled lights aren't sent");
    }

    #[test]
    fn focal_length_round_trips() {
        for mm in [24.0, 50.0, 85.0, 135.0] {
            assert!((focal_length(fov_for_focal_length(mm)) - mm).abs() < 1e-3);
        }
    }
}
