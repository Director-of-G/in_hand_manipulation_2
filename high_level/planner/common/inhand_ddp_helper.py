############################################################
# This script helps in the visualization process of inhand #
# manipulation. We paint contact locations on the object,  #
# and contact forces on the fingertip.                     #
############################################################


import numpy as np

from pydrake.all import (
    Cylinder,
    Rgba,
    RigidTransform,
    RotationMatrix,
    Sphere,
    Quaternion
)

from pydrake.geometry import (
    MeshcatCone,
)

from common.common_ddp import *

CONTACT_COLOR_MAP = {
    "thumb": Rgba(r=0.086, g=0.023, b=0.541, a=0.5),
    "index": Rgba(r=0.619, g=0.094, b=0.615, a=0.5),
    "middle": Rgba(r=0.925, g=0.470, b=0.325, a=0.5),
    "ring": Rgba(r=0.992, g=0.701, b=0.180, a=0.5)
}

class InhandDDPVizHelper(object):
    def __init__(self, meshcat=None):
        self.meshcat = meshcat
        self.contact_force_scale = 30
        self.point_radius = 0.003
        self.cylinder_radius = 0.001
        self.cylinder_length = 0.3
        self.unit_magnitude = 200.0
        self.cone_height = 0.002
        self.cone_radius = 0.002

        self.n_c = 4
        self.finger_to_index_map = {
            "thumb": -1,
            "index": -1,
            "middle": -1,
            "ring": -1
        }
        self.finger_to_geom_name_map = {
            "thumb": "allegro_hand_right::link_15_tip_collision_2",
            "index": "allegro_hand_right::link_3_tip_collision_1",
            "middle": "allegro_hand_right::link_7_tip_collision_1",
            "ring": "allegro_hand_right::link_11_tip_collision_1"
        }

    def handle_contact_reference_request(self, request:ContactTrackReferenceRequest):
        self.finger_to_geom_name_map = request.finger_to_geom_name_map
        self.n_c = len(self.finger_to_geom_name_map)
        self.finger_to_index_map.clear()
        for name, _ in self.finger_to_geom_name_map.items():
            self.finger_to_index_map[name] = -1

    def set_meshcat(self, meshcat):
        self.meshcat = meshcat

    def parse_finger_order(self, geom_names):
        for key, _ in self.finger_to_index_map.items():
            try:
                self.finger_to_index_map[key] = \
                    geom_names.index(self.finger_to_geom_name_map[key])
            except:
                # lost contact
                self.finger_to_index_map[key] = -1

    def get_ranked_contact_data(self, data):
        ranked_data = np.zeros((0, data.shape[1]))
        for key, _ in self.finger_to_index_map.items():
            index_ = self.finger_to_index_map[key]
            if index_ == -1:
                # lost contact
                ranked_data = np.concatenate(
                    (ranked_data, np.zeros((1, data.shape[1]))), axis=0
                )
            else:
                ranked_data = np.concatenate(
                    (ranked_data, data[index_].reshape(1, -1)), axis=0
                )

        return ranked_data

    def get_ranked_data_1d(self, data):
        ranked_data = []
        for key, _ in self.finger_to_index_map.items():
            index_ = self.finger_to_index_map[key]
            # for lost contact
            if index_ == -1:
                ranked_data.append(0.1)
            ranked_data.append(data[index_])
        
        return ranked_data

    def get_cylinder_transform_force_plot(self, point, force):
        """
            We represent contact forces as cylinders.
            This function returns the transform of the cylinder,
            given the contact point and possibly unnormalized
            contact force.
        """
        magnitude = np.linalg.norm(force)
        # print("magnitude: ", magnitude)
        length = magnitude / self.unit_magnitude * self.cylinder_length
        force_ = force / magnitude
        
        X_WO_cylinder = RigidTransform(
            RotationMatrix.MakeFromOneVector(force_, 2),
            point
        )
        X_OC_cylinder = RigidTransform(
            [0.0, 0.0, -length/2-self.cone_height-self.point_radius]
        )
        X_WC_cylinder = X_WO_cylinder.multiply(X_OC_cylinder)

        X_WO_cone = RigidTransform(
            RotationMatrix.MakeFromOneVector(-force_, 2),
            point
        )
        X_OC_cone = RigidTransform([0.0, 0.0, self.point_radius])
        X_WC_cone = X_WO_cone.multiply(X_OC_cone)

        return X_WC_cylinder, X_WC_cone, length

    def set_elements_in_meshcat(self):
        assert self.meshcat is not None

        for name, rgba in CONTACT_COLOR_MAP.items():
            self.meshcat.SetObject(
                "ddp/{}_contact".format(name),
                Sphere(radius=self.point_radius), rgba
            )
        for name, rgba in CONTACT_COLOR_MAP.items():
            self.meshcat.SetObject(
                "ddp/{}_force".format(name),
                Cylinder(radius=self.cylinder_radius, length=self.cylinder_length), rgba
            )
            self.meshcat.SetObject(
                "ddp/{}_force_cone".format(name),
                MeshcatCone(height=self.cone_height, a=self.cone_radius, b=self.cone_radius),
                rgba
            )

    def plot_contact_point_one_finger(self, p_WCas, name="thumb"):
        assert self.meshcat is not None

        index_ = self.finger_to_index_map[name]
        if index_ == -1:
            return
        self.meshcat.SetTransform(
            "ddp/{}_contact".format(name), RigidTransform(p_WCas[index_])
        )

    def plot_contact_points(
        self, p_W,
        already_in_order=False
    ):
        """
        :param p_W: contact points in world frame
        """
        assert p_W.shape[1] == 3

        if already_in_order:
            self.finger_to_index_map = {"thumb": 0, "index": 1, "middle": 2, "ring": 3}

        for name in CONTACT_COLOR_MAP.keys():
            self.plot_contact_point_one_finger(p_W, name)

    def plot_contact_force_one_finger(self, p_WCas, f_BA_W, name="thumb"):
        assert self.meshcat is not None

        index_ = self.finger_to_index_map[name]
        if index_ == -1:
            return
        p_WCa = p_WCas[index_]
        f_BA_W_ = f_BA_W[index_]
        if np.linalg.norm(f_BA_W_) < 1e-5:
            return
        
        X_WC_cylinder, X_WC_cone, length = self.get_cylinder_transform_force_plot(p_WCa, f_BA_W_)
        self.meshcat.SetObject(
            "ddp/{}_force".format(name),
            Cylinder(radius=self.cylinder_radius, length=length),
            CONTACT_COLOR_MAP[name]
        )
        self.meshcat.SetTransform(
            "ddp/{}_force".format(name), X_WC_cylinder
        )
        self.meshcat.SetTransform(
            "ddp/{}_force_cone".format(name), X_WC_cone
        )

    def plot_contact_forces(
        self, p_W, f_W,
        already_in_order=False
    ):
        """
        :param p_W: contact points in world frame
        :param f_W: contact forces in world frame
        """
        assert p_W.shape[1] == 3

        if already_in_order:
            self.finger_to_index_map = {"thumb": 0, "index": 1, "middle": 2, "ring": 3}

        for name in CONTACT_COLOR_MAP.keys():
            f_W_scaled = self.contact_force_scale * f_W
            self.plot_contact_force_one_finger(p_W, f_W_scaled, name)


class InhandReferenceGenerator(InhandDDPVizHelper):
    def __init__(self, meshcat=None):
        super().__init__(meshcat)
        self.external_contact_geom_names = []

    def handle_contact_reference_request(self, request:ContactTrackReferenceRequest):
        """
            This is the interface with low level controller
            Overwrite the method with support of external contacts
        """
        # handle finger contacts 
        self.finger_to_geom_name_map = request.finger_to_geom_name_map
        self.n_c = len(self.finger_to_geom_name_map)
        self.finger_to_index_map.clear()
        for name, _ in self.finger_to_geom_name_map.items():
            self.finger_to_index_map[name] = -1
        
        # handle external contacts
        self.parse_external_contacts(request.external_contact_geom_names)

    def parse_external_contacts(self, geom_names):
        """
            Add support for external contacts
        """
        self.external_contact_geom_names.clear()
        self.external_contact_geom_names += geom_names

        self.num_ext_contacts = len(geom_names)
        print("detected {} possible external contacts".format(self.num_ext_contacts))
        self.n_c += self.num_ext_contacts
        print("there are {} possible contacts in total".format(self.n_c))

        self.contact_to_index_map = self.finger_to_index_map.copy()
        self.contact_to_index_map.update({'external': [-1] * self.num_ext_contacts})

        if self.meshcat is not None:
            self.set_external_contact_elements_in_meshcat()

    def set_external_contact_elements_in_meshcat(self):
        assert self.meshcat is not None

        rgba = Rgba(r=0, g=0, b=0, a=0.5)

        for i_c in range(self.num_ext_contacts):
            self.meshcat.SetObject(
                "ddp/ext{}_contact".format(i_c),
                Sphere(radius=self.point_radius), rgba
            )
            self.meshcat.SetObject(
                "ddp/ext{}_force".format(i_c),
                Cylinder(radius=self.cylinder_radius, length=self.cylinder_length), rgba
            )
            self.meshcat.SetObject(
                "ddp/ext{}_force_cone".format(i_c),
                MeshcatCone(height=self.cone_height, a=self.cone_radius, b=self.cone_radius),
                rgba
            )

    def parse_contact_order(self, geom_names):
        """
            Get the indices of contacts in the coming data
        """
        # parse finger indices
        for key, finger_name in self.finger_to_geom_name_map.items():
            try:
                self.contact_to_index_map[key] = \
                    geom_names.index(finger_name)
            except:
                raise RuntimeError("finger {} lost contact with object".format(finger_name))

        n_c_ext = len(self.contact_to_index_map['external'])
        if len(self.external_contact_geom_names) != n_c_ext:
            raise ValueError("Number of external contacts is different from initialization!")
        
        # parse external contact indices
        for i_c in range(n_c_ext):
            ext_name = self.external_contact_geom_names[i_c]
            if ext_name in geom_names:
                self.contact_to_index_map['external'][i_c] = \
                    geom_names.index(self.external_contact_geom_names[i_c])
            else:
                # inactive external contact
                self.contact_to_index_map['external'][i_c] = -1

    def plot_one_contact_point_3d(
            self,
            p_WCas,
            index=0,
            name="thumb"
        ):
        assert self.meshcat is not None

        index_ = index
        if index_ == -1:
            return
        self.meshcat.SetTransform(
            "ddp/{}_contact".format(name), RigidTransform(p_WCas[index_])
        )

    def plot_contact_points_3d(
            self,
            p_ACas,
            obj_pos=[0.06, 0.0, 0.072],
            obj_quat=[1, 0, 0, 0],
        ):
        """
            Free-floating object case
        """
        assert p_ACas.shape[1] == 3
        X_WA = RigidTransform(
            Quaternion(obj_quat),
            np.array(obj_pos)
        )
        p_WCas = X_WA.multiply(p_ACas.T).T

        for name in CONTACT_COLOR_MAP.keys():
            self.plot_one_contact_point_3d(
                p_WCas,
                self.contact_to_index_map[name],
                name
            )

        for i_c in range(self.num_ext_contacts):
            self.plot_one_contact_point_3d(
                p_WCas,
                self.contact_to_index_map['external'][i_c],
                "ext{}".format(i_c)
            )

    def plot_one_contact_force_3d(
            self,
            p_WCas,
            f_BA_W,
            index=0,
            name="thumb"
        ):
        assert self.meshcat is not None

        index_ = index
        if index_ == -1:
            return
        p_WCa = p_WCas[index_]
        f_BA_W_ = f_BA_W[index_]
        X_WC_cylinder, X_WC_cone, length = self.get_cylinder_transform_force_plot(p_WCa, f_BA_W_)
        self.meshcat.SetObject(
            "ddp/{}_force".format(name),
            Cylinder(radius=self.cylinder_radius, length=length),
            Rgba(r=0, g=0, b=0, a=0.5) if 'ext' in name else CONTACT_COLOR_MAP[name]
        )
        self.meshcat.SetTransform(
            "ddp/{}_force".format(name), X_WC_cylinder
        )
        self.meshcat.SetTransform(
            "ddp/{}_force_cone".format(name), X_WC_cone
        )

    def plot_contact_forces_3d(
            self,
            p_ACas,
            f_BA_W,
            obj_pos=[0.06, 0.0, 0.072],
            obj_quat=[1, 0, 0, 0],
        ):
        assert p_ACas.shape[1] == 3
        X_WA = RigidTransform(
            Quaternion(obj_quat),
            np.array(obj_pos)
        )
        p_WCas = X_WA.multiply(p_ACas.T).T

        for name in CONTACT_COLOR_MAP.keys():
            self.plot_one_contact_force_3d(
                p_WCas,
                f_BA_W,
                self.contact_to_index_map[name],
                name
            )

        for i_c in range(self.num_ext_contacts):
            self.plot_one_contact_force_3d(
                p_WCas,
                f_BA_W,
                self.contact_to_index_map['external'][i_c],
                "ext{}".format(i_c)
            )

class ContactTrackReference(object):
    """
        Ranked data: in order "thumb", "index", "middle", "ring"
    """
    def __init__(self):
        self.p_ACa_A = np.zeros((0, 3))
        self.f_BA_W = np.zeros((0, 3))
