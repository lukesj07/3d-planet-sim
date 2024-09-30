import numpy as np
from vispy import app, scene
from vispy.geometry import create_sphere
from noise import snoise3
from vispy import use

use('glfw')

class CelestialBody:
    def __init__(self, name: str, radius: float = 1.0, cols: int = 64, rows: int = 64, mass: float = 1.0, rotation_speed: float = 0.08, 
                 axis_of_rotation: tuple[float, float, float] = (0, 1, 0), position: tuple[float, float, float] = (0, 0, 0), 
                 velocity: tuple[float, float, float] = (0, 0, 0), colors: dict[str, list[float]] = None, continent_freq: float = 1.0, 
                 detail_freq: float = 6.0, scale: float = 0.08, land_threshold: float = -0.1, noise_bias: float = 0.15) -> None:
        """
        params:
        - name: str, name of the celestial body.
        - radius: float, radius of the body in arbitrary units (e.g., Earth = 1.0).
        - cols, rows: int, resolution for the body mesh.
        - mass: float, mass of the body in arbitrary units.
        - rotation_speed: float, speed of the body's rotation.
        - axis_of_rotation: tuple, axis about which the body rotates.
        - position: tuple, (x, y, z) coordinates for the body's position.
        - velocity: tuple, initial velocity vector of the body.
        - colors: dict, custom terrain colors for ocean, plains, hills, and mountains (can exclude oceans).
        - continent_freq: float, frequency for large features (continents/craters).
        - detail_freq: float, frequency for small details (rough terrain, noise).
        - scale: float, scale for height deformation.
        - land_threshold: float, height threshold for determining land vs ocean.
        - noise_bias: float, bias applied to noise to control base terrain elevation.
        """
        self.name = name
        self.radius = radius
        self.cols = cols
        self.rows = rows
        self.mass = mass
        self.rotation_speed = rotation_speed
        self.axis_of_rotation = axis_of_rotation
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.colors = colors if colors else {}
        self.continent_freq = continent_freq
        self.detail_freq = detail_freq
        self.scale = scale
        self.land_threshold = land_threshold
        self.noise_bias = noise_bias
        self.vertices, self.faces, self.vertex_colors = self.generate_terrain()

    def generate_terrain(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # mesh generation
        sphere = create_sphere(radius=self.radius, cols=self.cols, rows=self.rows, method='latitude')
        vertices = np.array(sphere.get_vertices())
        faces = np.array(sphere.get_faces())

        heights = []
        for i, vertex in enumerate(vertices):
            x, y, z = vertex
            continent_noise = snoise3(x * self.continent_freq, y * self.continent_freq, z * self.continent_freq, octaves=4) + self.noise_bias
            detail_noise = snoise3(x * self.detail_freq, y * self.detail_freq, z * self.detail_freq, octaves=6)
            height_value = continent_noise + detail_noise * 0.25

            if height_value > self.land_threshold:
                height_value = (height_value - self.land_threshold) * self.scale + 1.0  # Scale land height
            else:
                height_value = 1.0

            vertices[i] *= height_value  # height deform
            heights.append(height_value)

        # height normalization for color mapping
        heights = (np.array(heights) - np.min(heights)) / (np.max(heights) - np.min(heights))

        vertex_colors = np.zeros((len(vertices), 4))  # RGBA
        for i, h in enumerate(heights):
            if 'ocean' in self.colors and h < 0.4:
                vertex_colors[i] = self.colors['ocean'] 
            elif 'plains' in self.colors and h < 0.55:
                vertex_colors[i] = self.colors['plains']
            elif 'hills' in self.colors and h < 0.7:
                vertex_colors[i] = self.colors['hills']
            else:
                vertex_colors[i] = self.colors.get('mountains', [1.0, 1.0, 1.0, 1.0])  # default

        return vertices, faces, vertex_colors

    def create_visual(self, view: scene.ViewBox) -> scene.visuals.Mesh:
        mesh = scene.visuals.Mesh(vertices=self.vertices, faces=self.faces, vertex_colors=self.vertex_colors, shading='flat', parent=view.scene)
        mesh.transform = scene.transforms.MatrixTransform()
        mesh.transform.translate(self.position)
        return mesh

    def apply_gravity(self, other_body: 'CelestialBody') -> np.ndarray:
        G = 1.0  # random units
        r_vec = self.position - other_body.position
        distance = np.linalg.norm(r_vec)
        if distance == 0:
            return np.array([0.0, 0.0, 0.0])
        force_magnitude = G * self.mass * other_body.mass / (distance ** 2)
        force_direction = r_vec / distance
        acceleration = force_magnitude / other_body.mass * force_direction
        return acceleration


class CloudSystem:
    def __init__(self, radius: float = 1.08, cols: int = 128, rows: int = 128) -> None:
        self.radius = radius
        self.cols = cols
        self.rows = rows
        self.vertices, self.faces = self.generate_cloud_mesh()
        self.colors = np.array([[1.0, 1.0, 1.0, 0.0]] * len(self.vertices))  # transparent by default

    def generate_cloud_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        sphere = create_sphere(radius=self.radius, cols=self.cols, rows=self.rows, method='latitude')
        vertices = np.array(sphere.get_vertices())
        faces = np.array(sphere.get_faces())
        return vertices, faces

    def update_clouds(self, t: float, original_cloud_vertices: np.ndarray, cloud_colors: np.ndarray, 
                      cloud_mesh: scene.visuals.Mesh, cloud_freq: float = 1.0, cloud_threshold: float = 0.7, 
                      depth_intensity: float = 0.05) -> None:
        updated_vertices = cloud_mesh.mesh_data.get_vertices().copy()
        updated = False
        for i in range(len(original_cloud_vertices)):
            x, y, z = original_cloud_vertices[i]
            deformation = (snoise3(x * cloud_freq + t, y * cloud_freq + t, z * cloud_freq + t, octaves=2) + 1) / 2
            new_alpha = smooth_alpha(deformation, cloud_threshold)

            if new_alpha > 0:
                new_position = original_cloud_vertices[i] + original_cloud_vertices[i] * deformation * 0.015
                new_position += snoise3(x * 2, y * 2, z * 2, octaves=1) * depth_intensity * 0.5 * original_cloud_vertices[i]
                updated_vertices[i] = new_position
                cloud_colors[i, 3] = new_alpha
                updated = True
            elif cloud_colors[i, 3] != 0.0:  # stop redundant updates
                cloud_colors[i, 3] = 0.0
                updated = True

        if updated:
            cloud_mesh.set_data(vertices=updated_vertices, faces=cloud_mesh.mesh_data.get_faces(), vertex_colors=cloud_colors)

def smooth_alpha(deformation: float, threshold: float, fade_range: float = 0.15) -> float:
    if deformation > threshold:
        return 0.5  # opaque
    elif threshold - fade_range < deformation <= threshold:
        return 0.5 * ((deformation - (threshold - fade_range)) / fade_range)  # fade
    return 0.0  # transparent


canvas = scene.SceneCanvas(keys='interactive', bgcolor='black', size=(800, 600), show=True, dpi=96)
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(fov=60, azimuth=30, elevation=30, distance=10)

G = 1.0 
earth_mass = 1.0
moon_mass = 0.0123

earth_colors = {
    'ocean': [0/255.0, 105/255.0, 148/255.0, 1.0],
    'plains': [34/255.0, 139/255.0, 34/255.0, 1.0],
    'hills': [139/255.0, 69/255.0, 19/255.0, 1.0],
    'mountains': [1.0, 1.0, 1.0, 1.0]
}
earth = CelestialBody(name="Earth", radius=1.0, mass=earth_mass, rotation_speed=0.08, axis_of_rotation=(0, 1, 0), position=(0, 0, 0), colors=earth_colors)
earth_mesh = earth.create_visual(view)

cloud_system = CloudSystem(radius=1.08)
cloud_mesh = scene.visuals.Mesh(vertices=cloud_system.vertices, faces=cloud_system.faces, vertex_colors=cloud_system.colors, shading=None, parent=view.scene)
cloud_mesh.transform = scene.transforms.MatrixTransform()
cloud_mesh.transform.translate((0, 0, 0))

moon_colors = {
    'plains': [0.5, 0.5, 0.5, 1.0], 
    'hills': [0.65, 0.65, 0.65, 1.0],
    'mountains': [0.8, 0.8, 0.8, 1.0]
}
moon_initial_position = np.array([3.0, 0.0, 0.0])
moon_orbital_radius = np.linalg.norm(moon_initial_position)
moon_initial_speed = np.sqrt(G * earth_mass / moon_orbital_radius)
moon_initial_velocity = np.array([0.0, 0.0, moon_initial_speed])
moon = CelestialBody(name="Moon", radius=0.273, mass=moon_mass, rotation_speed=0.01, axis_of_rotation=(0, 1, 0),
                     position=moon_initial_position, velocity=moon_initial_velocity, colors=moon_colors,
                     continent_freq=2.0, detail_freq=10.0, scale=0.03, land_threshold=-0.05, noise_bias=0.2)
moon_mesh = moon.create_visual(view)

def update(event: app.Timer) -> None:
    t = event.elapsed * 0.01
    dt = event.dt * 0.01
    earth_mesh.transform.rotate(earth.rotation_speed, earth.axis_of_rotation)
    acceleration = earth.apply_gravity(moon)
    moon.velocity += acceleration * dt
    moon.position += moon.velocity * dt
    moon_mesh.transform.reset()
    moon_mesh.transform.translate(moon.position)
    moon_mesh.transform.rotate(moon.rotation_speed, moon.axis_of_rotation)
    cloud_system.update_clouds(t, cloud_system.vertices, cloud_system.colors, cloud_mesh)

timer = app.Timer(interval=0.03, connect=update, start=True)

canvas.show()
app.run()
