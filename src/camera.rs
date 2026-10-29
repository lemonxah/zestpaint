use glam::{Mat4, Vec3};

/// Orbit camera: rotates around a target point, Substance-style
/// (LMB-drag orbit, MMB/RMB-drag pan, scroll zoom).
#[derive(Clone)]
pub struct OrbitCamera {
    pub target: Vec3,
    /// Radians, rotation around world Y.
    pub yaw: f32,
    /// Radians, elevation. Clamped short of the poles.
    pub pitch: f32,
    pub distance: f32,
    pub fov_y: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            yaw: 0.6,
            pitch: 0.4,
            distance: 3.0,
            fov_y: 45f32.to_radians(),
        }
    }
}

impl OrbitCamera {
    pub fn position(&self) -> Vec3 {
        let dir = Vec3::new(
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.cos(),
        );
        self.target + dir * self.distance
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.008;
        self.pitch = (self.pitch + dy * 0.008).clamp(-1.54, 1.54);
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        let view = self.view_matrix();
        // Rows of the rotation part are the camera basis vectors in world space.
        let right = Vec3::new(view.x_axis.x, view.y_axis.x, view.z_axis.x);
        let up = Vec3::new(view.x_axis.y, view.y_axis.y, view.z_axis.y);
        let scale = self.distance * 0.0015;
        self.target += (-right * dx + up * dy) * scale;
    }

    pub fn zoom(&mut self, scroll: f32) {
        self.distance = (self.distance * (1.0 - scroll * 0.1)).clamp(0.01, 1e6);
    }

    pub fn view_matrix(&self) -> Mat4 {
        glam::camera::rh::view::look_at_mat4(self.position(), self.target, Vec3::Y)
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let near = (self.distance * 0.01).max(0.001);
        let far = self.distance * 100.0;
        // Vulkan convention: Y-down NDC, Z in [0, 1].
        let proj =
            glam::camera::rh::proj::vulkan::perspective(self.fov_y, aspect.max(0.01), near, far);
        proj * self.view_matrix()
    }

    /// Reposition so a bounding sphere fills the view comfortably.
    pub fn frame(&mut self, center: Vec3, radius: f32) {
        self.target = center;
        let r = radius.max(1e-4);
        self.distance = r / (self.fov_y * 0.5).sin() * 1.2;
    }
}
