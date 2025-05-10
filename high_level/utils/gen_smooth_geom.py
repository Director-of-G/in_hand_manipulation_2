import numpy as np
from scipy import interpolate
from scipy.interpolate import BSpline
from scipy.spatial import ConvexHull


def bezier_curve(P0, P1, P2, P3, num_points=100):
    t = np.linspace(0, 1, num_points)[:, np.newaxis]
    curve = (1 - t)**3 * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * P3
    return curve

def extrude_curve_along_z(curve_2d, z_range, num_layers):
    layers = []
    z_values = np.linspace(z_range[0], z_range[1], num_layers)
    
    for z in z_values:
        layer = np.c_[curve_2d, np.full(curve_2d.shape[0], z)]  # 添加 z 轴坐标
        layers.append(layer)
    
    return np.array(layers)

def generate_obj(vertices, faces, filename="bezier_surface.obj"):
    with open(filename, 'w') as file:
        for v in vertices:
            file.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            face_str = " ".join(map(str, face))
            file.write(f"f {face_str}\n")

def create_faces(num_points, num_layers):
    faces = []
    for i in range(num_layers - 1):
        for j in range(num_points - 1):
            # 每个矩形由四个顶点组成
            v1 = i * num_points + j + 1
            v2 = i * num_points + j + 2
            v3 = (i + 1) * num_points + j + 2
            v4 = (i + 1) * num_points + j + 1
            faces.append([v1, v2, v3, v4])
    return faces

def close_surface_top_bottom(vertices, num_points, num_layers):
    faces = []
    
    # 底部（Z最小值）
    bottom_center_idx = len(vertices) + 1
    bottom_vertices_idx = range(1, num_points + 1)  # 第一层的顶点
    for i in range(num_points - 1):
        faces.append([bottom_center_idx, bottom_vertices_idx[i], bottom_vertices_idx[i+1]])
    faces.append([bottom_center_idx, bottom_vertices_idx[-1], bottom_vertices_idx[0]])  # 闭合最后一个面
    
    # 顶部（Z最大值）
    top_center_idx = len(vertices) + 2
    top_vertices_idx = range((num_layers - 1) * num_points + 1, num_layers * num_points + 1)  # 最后一层的顶点
    for i in range(num_points - 1):
        faces.append([top_center_idx, top_vertices_idx[i+1], top_vertices_idx[i]])
    faces.append([top_center_idx, top_vertices_idx[0], top_vertices_idx[-1]])  # 闭合最后一个面
    
    # 添加顶部和底部的中心点到顶点列表
    bottom_center = np.mean(vertices[:num_points], axis=0)
    top_center = np.mean(vertices[-num_points:], axis=0)
    vertices = np.vstack([vertices, bottom_center, top_center])
    
    return vertices, faces


class GeomGeneratorConfig(object):
    def __init__(self):
        # control points
        self.ordered_points = True

        # for 2d curve
        self.n_control_points_2d = 5
        self.is_curve_convex = True     # set True, can make the curve more convex, not strictly
        self.control_point_range_2d = {
            "x": [-0.04, 0.04],
            "y": [-0.04, 0.04],
            "r": [0.06, 0.07]   # radius (polar coords)
        }
        self.num_2d_points = 5
        self.num_cylinder_layers = 3
        self.cylinder_height = 0.10

        # for 3d surface
        self.num_3d_points = 100


class GeomGenerator(object):
    def __init__(self, config:GeomGeneratorConfig):
        self.config = config

    def generate_from_2d_surve(self, control_points:np.ndarray):
        # TODO(yongpeng): deprecate this
        def generate_random_control_points(num_points, x_range, y_range):
            control_points = np.zeros((num_points, 2))
            control_points[:, 0] = np.random.uniform(x_range[0], x_range[1], num_points)
            control_points[:, 1] = np.random.uniform(y_range[0], y_range[1], num_points)
            
            return control_points

        def generate_ordered_closed_control_points(num_points, r_min, r_max, convex=False):
            """
                The points will be located in a ring, and sampled in a clockwise order.
                :param r_min: the inner radius of the ring
                :param r_max: the outer radius of the ring
            """
            control_points = np.zeros((num_points, 2))

            dtheta = 2 * np.pi / num_points
            polar_ang_samples = np.linspace(0, 2 * np.pi - dtheta, num_points)
            polar_ang_samples += np.random.uniform(-dtheta/2, dtheta/2, num_points)

            polar_mod_samples = np.random.uniform(r_min, r_max, num_points)

            control_points[:, 0] = polar_mod_samples * np.cos(polar_ang_samples)
            control_points[:, 1] = polar_mod_samples * np.sin(polar_ang_samples)

            if convex:
                chull = ConvexHull(control_points)
                control_points = control_points[chull.vertices]

            from matplotlib import pyplot as plt
            for i in range(len(control_points)):
                plt.scatter(control_points[i, 0], control_points[i, 1], alpha=0.5+0.5*i/num_points)
            plt.show()
            plt.grid("on")

            return control_points

        pts_range = self.config.control_point_range_2d
        if self.config.ordered_points:
            control_points = generate_ordered_closed_control_points(
                num_points=self.config.n_control_points_2d,
                r_min=pts_range["r"][0],
                r_max=pts_range["r"][1],
                convex=self.config.is_curve_convex
            )
        else:
            control_points = generate_random_control_points(
                num_points=self.config.n_control_points_2d,
                x_range=pts_range["x"],
                y_range=pts_range["y"]
            )
        # close the control points
        control_points = np.append(control_points, [control_points[0]], axis=0)

        x = control_points[:, 0]
        y = control_points[:, 1]

        # generate smooth 2d curve
        tck, u = interpolate.splprep([x, y], s=0, per=True)

        # sample more points
        num_points = self.config.num_2d_points
        u_fine = np.linspace(0, 1, num_points)
        x_fine, y_fine = interpolate.splev(u_fine, tck)
        curve_2d = np.c_[x_fine, y_fine]

        # extrude as 3d cylinder object
        z_range = [-self.config.cylinder_height/2, self.config.cylinder_height/2]
        num_layers = self.config.num_cylinder_layers
        layers = extrude_curve_along_z(curve_2d, z_range, num_layers)

        # slice side surface
        vertices = layers.reshape(-1, 3)
        faces = create_faces(num_points, num_layers)

        # top and bottom
        vertices, top_bottom_faces = close_surface_top_bottom(vertices, num_points, num_layers)

        # combine and generate obj
        faces.extend(top_bottom_faces)
        generate_obj(vertices, faces, "./geom_bspline.obj")
        print("3D 贝塞尔曲面已保存为 'geom_bspline.obj'")

if __name__ == "__main__":
    # 定义控制点，首尾相连形成闭环
    control_points = np.array([
        [0, 0], 
        [1, 2], 
        [3, 3], 
        [4, 0], 
        [2, -2], 
        [0, 0]  # 关键：首尾相连
    ]) * 0.03

    # config = GeomGeneratorConfig()

    # geom_gen = GeomGenerator(config)
    # geom_gen.generate_from_2d_surve(control_points)

    # exit(0)

    # 提取x, y坐标
    x = control_points[:, 0]
    y = control_points[:, 1]

    # 使用scipy的splprep生成闭合B样条，参数s=0表示没有平滑（插值样条）
    tck, u = interpolate.splprep([x, y], s=0, per=True)

    # 通过splev采样更多点来得到平滑曲线
    num_points = 400
    u_fine = np.linspace(0, 1, num_points)
    x_fine, y_fine = interpolate.splev(u_fine, tck)
    curve_2d = np.c_[x_fine, y_fine]

    # 3. 沿 Z 轴拉伸曲线，生成3D曲面
    z_range = [0, 0.08]  # 拉伸的 z 轴范围
    num_layers = 20  # 拉伸生成的层数
    layers = extrude_curve_along_z(curve_2d, z_range, num_layers)

    # 4. 生成顶点和面片
    vertices = layers.reshape(-1, 3)  # 将所有层的顶点拉平
    faces = create_faces(num_points, num_layers)

    # 5. 封闭顶部和底部表面
    vertices, top_bottom_faces = close_surface_top_bottom(vertices, num_points, num_layers)

    # 6. 合并所有面
    faces.extend(top_bottom_faces)

    # 7. 保存为 .obj 文件
    generate_obj(vertices, faces, "./bezier_surface.obj")

    print("3D 贝塞尔曲面已保存为 'bezier_surface.obj'")
