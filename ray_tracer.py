import argparse
from PIL import Image
import numpy as np
import random

random.seed(10)

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere

EPSILON = 0.0001  # to avoid numeric errors

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
    camera_z_axis = normalize_vec(look_at_dir)

    # find camera y axis (up)
    up_fixed = up - np.dot(up, camera_z_axis) * camera_z_axis
    camera_y_axis = normalize_vec(up_fixed)

    # camera x axis (right)
    camera_x_axis = np.cross(camera_y_axis, camera_z_axis)

    x_axis, y_axis, z_axis = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])  # real axis
    sin_x, cos_x = calc_sin_cos_vectors(camera_x_axis, x_axis)
    sin_y, cos_y = calc_sin_cos_vectors(camera_y_axis, y_axis)
    return np.array([[cos_y, 0, sin_y],
                     [-sin_x * sin_y, cos_x, sin_x * cos_y],
                     [-cos_x * sin_y, -sin_x, cos_x * cos_y]])


def calc_sin_cos_vectors(v1, v2):
    # return the sin() and cos() functions on the angle between 2 vectors
    dot_product = np.dot(normalize_vec(v1), normalize_vec(v2))
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle = np.arccos(max(-1.0, min(1.0, cos_angle)))  # addeed clap for numeric problem hanling
    return np.sin(angle), cos_angle


"""
Check inters:
"""


def check_inters(items, ray):
    # return a list with all the intersections with items in the space
    inters_pairs = [] # paris of item and intersection on it
    for item in items:
        inters_pairs += check_inter(item, ray)
    return inters_pairs


def check_inter(item, ray):
    if isinstance(item, Sphere):
        return [(item, inter) for inter in find_intersection_sphere(item, ray)]
    if isinstance(item, InfinitePlane):
        return [(item, inter) for inter in find_intersection_plane(item, ray)]
    if isinstance(item, Cube):
        return [(item, inter) for inter in find_intersection_cube(item, ray)]
    return []

def find_intersection_sphere(sphere, ray):
    r = sphere.radius
    m, p = ray
    C = sphere.position
    p_minus_C = p - C

    a = np.dot(m, m)
    b = 2 * np.dot(m, p_minus_C)
    c = np.dot(p_minus_C, p_minus_C) - r ** 2
    disc = b ** 2 - 4 * a * c

    if disc < 0:
        return []  # No intersection

    sqrt_disc = np.sqrt(disc)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)

    # We want to return the closest intersection point in front of the camera
    t_min = min(t1, t2)
    t_max = max(t1, t2)

    if t_max < 0:
        return []  # Both intersections are behind the camera

    if t_min < 0:
        t_min = t_max  # If t_min is negative, choose the farther point which is t_max

    if t_min < 0:
        return []  # All intersections are behind the camera

    # Return the closest valid intersection point
    return [p + t_min * m]



def find_intersection_plane(plane, ray):
    n = np.array(plane.normal)
    n_normalized = normalize_vec(n)
    m, p = ray
    c = plane.offset
    if abs(np.dot(m, n_normalized)) < EPSILON:
        if abs((c-np.dot(p, n_normalized))) < EPSILON:
            return [p]
        else:
            return []
    t = (c - np.dot(p, n_normalized)) / np.dot(m, n_normalized)
    if t < 0 or np.isnan(t):
        return []
    return [find_point_in_line(ray, (c-np.dot(p, n_normalized))/np.dot(m, n_normalized))]


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


def get_inter_color(objects, lights, materials, ray, inter_pair, background_color, recursions, shadows_amount):
    color = np.array([0, 0, 0], dtype='float64')
    our_background_color = np.array([0, 0, 0], dtype='float64')
    normal = normal_to_surf(inter_pair[0], ray, inter_pair[1])
    material = materials[inter_pair[0].material_index - 1]
    transparency = material.transparency
    for light in lights:
        light_minus_point = normalize_vec(light.position - inter_pair[1])
        theta_light = np.dot(normal, light_minus_point)
        reflected_light = light_minus_point - normal * (2 * theta_light)
        start = inter_pair[1] + light_minus_point * 0.001
        shade_ray_fraction = 1.0 / shadows_amount / shadows_amount * light.shadow_intensity
        light_width_r = normalize_vec(np.cross(light_minus_point, light_minus_point + 7)) * light.radius
        light_width_d = np.cross(light_width_r, normalize_vec(light_minus_point)) * light.radius

        illumination = 1.0
        # Iterate over shade ray pairs
        shadows_amount = int(shadows_amount)
        for i in range(shadows_amount):
            for j in range(shadows_amount):
                random_up = random.random()
                random_right = random.random()
                vec = light_width_d * ((-shadows_amount / 2 + j + random_up - 0.5) * 1 / shadows_amount)
                vec2 = light_width_r * ((-shadows_amount / 2 + i + random_right - 0.5) * 1 / shadows_amount)
                nearest_light_point = light.position + vec + vec2
                shade_direction_reversed = normalize_vec(nearest_light_point - start)
                ray_norm = np.linalg.norm(start - nearest_light_point)
                light_left_in_ray = 1.0
                for item in objects:
                    curr_ray = (shade_direction_reversed, start + shade_direction_reversed * 0.01)
                    shadow_hit = find_nearest_inter_pair(curr_ray, check_inters([item], curr_ray))
                    if (shadow_hit is not None) and (np.linalg.norm(shadow_hit[1] - start) < ray_norm):
                        light_left_in_ray *= materials[item.material_index - 1].transparency
                        if light_left_in_ray < 0:
                            light_left_in_ray = 0
                            break
                illumination -= shade_ray_fraction * (1 - light_left_in_ray)
        if illumination > 0:
            diff_color = np.clip(np.array(material.diffuse_color) * 255 * theta_light * light.color, 0, 255)
            spec_color = (np.array(material.specular_color) * 255 * light.specular_intensity *
                          (np.dot(normalize_vec(reflected_light), normalize_vec(ray[0])) ** material.shininess) *
                          light.color)
            color += (diff_color + spec_color) * (1 - material.transparency) * illumination
    reflect_color = np.array([0, 0, 0], dtype='float64')
    if recursions > 0:
        ray_reflect, point_reflect_pair = calc_reflect(objects, inter_pair, ray, normal)
        if point_reflect_pair:
            reflect_color += get_inter_color(objects, lights, materials, ray_reflect, point_reflect_pair,
                                             background_color, recursions - 1, shadows_amount) * (np.array(material.reflection_color))
            color += reflect_color

        next_point, new_objets_tra = point_next_hit(objects, inter_pair, ray, normal)
        if next_point:
            our_background_color += get_inter_color(new_objets_tra, lights, materials, ray, next_point,
                                                    background_color,
                                                    recursions - 1, shadows_amount)
            color += (our_background_color * transparency)

    return np.clip(color, 0, 255)


def normal_to_surf(item, ray, point):
    if isinstance(item, Cube):
        return normal_to_cube(item, point)
    if isinstance(item, InfinitePlane):
        return np.array(item.normal) * (-np.dot(ray[0], item.normal))
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
    return normalize_vec(average_normal)


def normal_to_sphere(sphere, point):
    x_c, y_c, z_c = sphere.position
    x_p, y_p, z_p = point
    normal_vec = np.array([x_p - x_c, y_p - y_c, z_p - z_c])
    normal_vec_normalized = normalize_vec(normal_vec)
    return normal_vec_normalized


def calc_reflect(objects, inter_pair, ray, normal):
    """
    old method:
    theta_ray = np.dot(normalize_vec(normal), normalize_vec(ray[0]))
    reflected_ray = (2 * normal * np.cos(theta_ray) - ray[0], inter_pair[1])
    new method:
    """
    theta_ray = np.dot(normalize_vec(normal), ray[0])
    reflected_ray = (ray[0] - normal * (2 * theta_ray), inter_pair[1]) # (ray[1] + ray[0] - normal * (2 * theta_ray), inter_pair[1])
    objects = [o for o in objects if (o != inter_pair[0])]
    inters_pairs = check_inters(objects, reflected_ray)
    return reflected_ray, find_nearest_inter_pair(ray, inters_pairs)


def point_next_hit(objects, inter_pair, ray, normal):
    new_objects = objects.copy()
    new_objects.remove(inter_pair[0])
    if inter_pair[0] in new_objects:
        new_objects.remove(inter_pair[0])
    inters_pairs = check_inters(new_objects,ray)
    if inters_pairs == []:
        return None, None
    return find_nearest_inter_pair(ray, inters_pairs), new_objects

def get_pixel_color(nearest_inter):
    pass

def light_hits(nearest_inter):
    pass

def produce_soft_shadow(nearest_inter):
    pass


def normalize_vec(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def ray_tracer(args, camera, scene_settings, objects):
    w = args.width
    h = args.height
    screen_height = camera.screen_width / w * h
    pixel_width = camera.screen_width / w
    pixel_height = screen_height / h
    E = np.array(camera.position)
    look_at = normalize_vec(np.array(camera.look_at) - E)
    up = np.array(camera.up_vector)
    up = normalize_vec(up - normalize_vec(look_at) * (np.dot(look_at, up) / np.linalg.norm(look_at)))

    right = normalize_vec(np.cross(up, look_at))
    move_right = right * pixel_width
    move_down = up * (-pixel_height)
    height_vec = look_at * camera.screen_distance
    width_vec = (-camera.screen_width / 2) * right
    up_by_height_vec = up * (screen_height / 2)
    P = E + height_vec  # middle pixel
    # set pos to P0:
    P0 = P + width_vec + up_by_height_vec + move_down * 0.5
    image_array = np.zeros((500, 500, 3))
    for i in range(h):
        pos = P0   # this is changing in every loop iterate
        for j in range(w):
            # 1.2
            ray = (normalize_vec(pos - E), E)
            # 2

            inters_pairs = check_inters(objects, ray)
            # 3
            nearest_inter_pair = find_nearest_inter_pair(ray, inters_pairs)
            # 4
            if nearest_inter_pair:
                materials = [item for item in objects if isinstance(item, Material)]
                color = get_inter_color(objects, [item for item in objects if isinstance(item, Light)], materials, ray,
                                        nearest_inter_pair, np.array(scene_settings.background_color),
                                        scene_settings.max_recursions, scene_settings.root_number_shadow_rays)


            else:
                color = np.array([0, 0, 0], dtype='uint8')
            """#5
            light_hits = light_hits(nearest_inter)
            if light_hits:
                #6
                soft_shadow = produce_soft_shadow(nearest_inter)"""
            image_array[i, j] = color.astype('uint8')
            pos += move_right
        if i % 50 == 0:
            print("progress " + str(i) + "\tcolor " + str(image_array[i, 0]) + "\t0-1 scale "+str(image_array[i, 0] / 255))
        pos -= w * move_right
        pos += move_down
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