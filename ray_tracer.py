import argparse
from PIL import Image
import numpy as np

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere


def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def save_image(image_array):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save("scenes/Spheres.png")

"""
Calc Axis:
"""


def calc_camera_axis(E, up, look_at):
    # return V_x, V_y, V_z
    M = calc_rot_matrix(E, up, look_at)
    return np.dot(M, np.array([1, 0, 0])), np.dot(M, np.array([0, 1, 0])), np.dot(M, np.array([0, 0, 1]))


def calc_rot_matrix(E, up, look_at):
    # find camera z axis (look at)
    look_at_dir = look_at - E
    camera_z_axis = look_at_dir / np.linalg.norm(look_at_dir)

    # find camera y axis (up)
    up_fixed = up - np.dot(up, camera_z_axis) * camera_z_axis
    camera_y_axis = up_fixed / np.linalg.norm(up_fixed)

    # camera x axis (left)
    camera_x_axis = np.cross(camera_y_axis, camera_z_axis)

    x_axis, y_axis, z_axis = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])  # real axis
    sin_x, cos_x = calc_sin_cos_vectors(camera_x_axis, x_axis)
    sin_y, cos_y = calc_sin_cos_vectors(camera_y_axis, y_axis)

    return np.array([[cos_y, 0, sin_y],
                     [-sin_x * sin_y, cos_x, sin_x * cos_y],
                     [-cos_x * sin_y, -sin_x, cos_x * cos_y]])


def calc_sin_cos_vectors(v1, v2):
    # return the sin() and cos() functions on the angle between 2 vectors
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle = np.arccos(cos_angle)
    return np.sin(angle), cos_angle


"""
Check inters:
"""


def check_inters(items, ray):
    # return a list with all the intersections with items in the space
    inters_pairs = [] # paris of item and intersection on it
    for item in items:
        if isinstance(item, Sphere):
            inters_pairs += [(item, inter) for inter in find_intersection_sphere(item, ray)]
        if isinstance(item, InfinitePlane):
            inters_pairs += [(item, inter) for inter in find_intersection_plane(item, ray)]
        if isinstance(item, Cube):
            inters_pairs += [(item, inter) for inter in find_intersection_cube(item, ray)]
    return inters_pairs


def find_intersection_sphere(sphere,ray):
    r = sphere.radius
    m, p = ray
    C = sphere.position
    a = np.dot(m, m)
    b = 2*np.dot(m, p - C)
    c = np.dot(p-C, p - C) - r**2
    disc = b**2-4*a*c
    if disc < 0:
        return []
    else:
        i1 = find_point_in_line(ray, (b+np.sqrt(disc))/(2*a))
        i2 = find_point_in_line(ray, (b - np.sqrt(disc)) / (2 * a))
        return [i1, i2]


def find_intersection_plane(plane,ray):
    n = np.array(plane.normal)
    m, p = ray
    c = plane.offset
    if np.dot(m, n) == 0:
        if (c-np.dot(p, n)) == 0:
            return [p]
        else:
            return []
    else:
        return [find_point_in_line(ray, (c-np.dot(p, n))/np.dot(m, n))]


def find_intersection_cube(cube, ray):
    c_x, c_y, c_z = cube.position
    L = cube.scale
    m, p = ray
    t_x1 = ((c_x - L/2) - p[0])/m[0]
    t_x2 = ((c_x + L/2) - p[0])/m[0]
    t_y1 = ((c_y - L/2) - p[1])/m[1]
    t_y2 = ((c_y + L/2) - p[1])/m[1]
    t_z1 = ((c_z - L/2) - p[2])/m[2]
    t_z2 = ((c_z + L/2) - p[2])/m[2]
    t_x_min, t_x_max = min(t_x1, t_x2), max(t_x1, t_x2)
    t_y_min, t_y_max = min(t_y1, t_y2), max(t_y1, t_y2)
    t_z_min, t_z_max = min(t_z1, t_z2), max(t_z1, t_z2)
    t_min = max(t_x_min, t_y_min, t_z_min)
    t_max = min(t_x_max, t_y_max, t_z_max)
    if t_min > t_max:
        return []
    i1 = find_point_in_line(ray, t_min)
    i2 = find_point_in_line(ray, t_max)
    return [i1, i2]


def find_point_in_line(ray, t):
    return ray[1] + ray[0]*t


"""
nearest inter:
"""


def find_nearest_inter_pair(ray, inters_pairs):
    if inters_pairs == []:
        return None
    return min(inters_pairs, key=lambda inter_pair: np.linalg.norm(ray[1]-inter_pair[1]))


"""
get_inter_color
"""


def get_inter_color(objects, lights, materials, ray, inter_pair, background_color, recursions):
    color = np.array([0, 0, 0], dtype='float64')
    for light in lights:
        normal = normal_to_surf(inter_pair[0], inter_pair[1])
        theta_light = np.dot(normal, inter_pair[1] - light.position)
        reflected_light = 2 * normal * np.cos(theta_light) - (inter_pair[1] - light.position)

        material = materials[inter_pair[0].material_index - 1]
        transparency = material.transparency

        diff_color = np.array(material.diffuse_color) * theta_light * light.color
        spec_color = np.array(material.specular_color) * (np.dot(reflected_light, ray[0]) ** material.shininess) * light.color  # v = ray, r = reflected_light
        reflect_color = np.array([0, 0, 0], dtype='float64')
        if recursions > 0:
            ray_reflect, point_reflect_pair = calc_reflect(objects, inter_pair, ray, normal)
            if point_reflect_pair:
                material_reflected_color = np.array(materials[point_reflect_pair[1].material_index - 1].reflection_color)
                reflect_color = get_inter_color(objects, lights, materials, ray_reflect, point_reflect_pair,
                                                background_color, recursions - 1) * material_reflected_color
        color += background_color * transparency + (diff_color + spec_color) * (1 - transparency) + reflect_color
    return color


def normal_to_surf(item, point):
    if isinstance(item, Cube):
        return normal_to_cube(item, point)
    if isinstance(item, InfinitePlane):
        return np.array(item.normal)
    if isinstance(item, Sphere):
        return normal_to_sphere(item, point)


def normal_to_cube(cube, point):
    x_c, y_c, z_c = cube.position
    l = cube.scale
    half_edge = l / 2
    x_p, y_p, z_p = point
    normals = []
    if np.isclose(x_p, x_c + half_edge):
        normals.append(np.array([1, 0, 0]))
    elif np.isclose(x_p, x_c - half_edge):
        normals.append(np.array([-1, 0, 0]))

    if np.isclose(y_p, y_c + half_edge):
        normals.append(np.array([0, 1, 0]))
    elif np.isclose(y_p, y_c - half_edge):
        normals.append(np.array([0, -1, 0]))

    if np.isclose(z_p, z_c + half_edge):
        normals.append(np.array([0, 0, 1]))
    elif np.isclose(z_p, z_c - half_edge):
        normals.append(np.array([0, 0, -1]))
    average_normal = np.mean(normals, axis=0)
    return average_normal / np.linalg.norm(average_normal)


def normal_to_sphere(sphere, point):
    x_c, y_c, z_c = sphere.position
    x_p, y_p, z_p = point
    normal_vec = np.array([x_p - x_c, y_p - y_c, z_p - z_c])
    normal_vec_normalized = normal_vec / np.linalg.norm(normal_vec)
    return normal_vec_normalized

def calc_reflect(objects, inter_pair, ray, normal):
    theta_ray = np.dot(normal, ray)
    reflected_ray = (2 * normal * np.cos(theta_ray) - ray[0], inter_pair[1])
    objects.remove(inter_pair[0])
    inters_pairs = check_inters(objects, reflected_ray)
    return reflected_ray, find_nearest_inter_pair(ray, inters_pairs)

def get_pixel_color(nearest_inter):
    pass

def light_hits(nearest_inter):
    pass

def produce_soft_shadow(nearest_inter):
    pass


def ray_tracer(args, camera, scene_settings, objects):
    w = int(camera.screen_width)
    h = int(w / args.width * args.height)
    E = np.array(camera.position)
    V_x, V_y, V_z = calc_camera_axis(E, np.array(camera.up_vector), np.array(camera.look_at))
    P = E + V_z * camera.screen_distance  # middle pixel
    # set pos to P0:
    P0 = P - (w / 2) * V_x - (h / 2) * V_y  # this is changing in every loop iterate
    pos = P0
    image_array = np.zeros((500, 500, 3))
    for i in range(h):
        for j in range(w):
            pos = P0
            # 1.2
            ray = (pos - E, E)
            # 2
            inters_pairs = check_inters(objects, ray)
            # 3
            nearest_inter_pair = find_nearest_inter_pair(ray, inters_pairs)
            # 4
            if nearest_inter_pair:
                materials = [item for item in objects if isinstance(item, Material)]
                color = get_inter_color(objects, [item for item in objects if isinstance(item, Light)], materials, ray,
                                        nearest_inter_pair, np.array(scene_settings.background_color),
                                        scene_settings.max_recursions)
            else:
                color = np.array([0, 0, 0], dtype='unit8')
            """#5
            light_hits = light_hits(nearest_inter)
            if light_hits:
                #6
                soft_shadow = produce_soft_shadow(nearest_inter)"""
            image_array[i, j] = color.astype('uint8')
            pos += V_x
        P0 += V_y
    return image_array




def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    image_array = ray_tracer(args, camera, scene_settings, objects)

    # Dummy result
    #image_array = np.zeros((500, 500, 3))

    # Save the output image
    save_image(image_array)


if __name__ == '__main__':
    main()
