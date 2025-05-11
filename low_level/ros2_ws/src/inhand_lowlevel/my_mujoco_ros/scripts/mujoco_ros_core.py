#!/home/jyp/anaconda3/envs/inhand/bin/python

import os
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PoseStamped, Wrench
from common_msgs.msg import ContactState
from std_srvs.srv import Trigger
from my_mujoco_ros.srv import ApplyLinkFT
from my_mujoco_ros.utils import get_contact_message, transform_pos_quat
import mujoco
from mujoco import viewer


def get_force_vis_message(force, position, marker_id=0):
    marker = Marker()
    marker.header.frame_id = "base"  # 机械臂的基座坐标系
    # marker.header.stamp = rospy.Time.now()

    marker.ns = ""
    marker.id = marker_id

    marker.type = Marker.ARROW
    marker.action = Marker.ADD

    force = force / 100.

    # 设置箭头起点和终点
    marker.points = []

    start_point = Point(x=position[0], y=position[1], z=position[2])
    
    end_point = Point(x=position[0]+force[0], y=position[1]+force[1], z=position[2]+force[2])
    
    marker.points.append(start_point)
    marker.points.append(end_point)

    # 设置箭头的比例和颜色
    marker.scale.x = 0.0025  # 箭头杆的直径
    marker.scale.y = 0.01  # 箭头头部的宽度
    marker.scale.z = 0.01  # 箭头头部的高度

    marker.color.a = 1.0  # 透明度
    marker.color.r = 1.0  # 红色
    marker.color.g = 0.0  # 绿色
    marker.color.b = 0.0  # 蓝色

    return marker

def get_body_pose(model, data, name):
    """ Get the body pose as 4x4 matrix """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    body_pos = data.xpos[body_id]
    body_mat = data.xmat[body_id].reshape(3, 3)

    body_pose = np.eye(4)
    body_pose[:3, :3] = body_mat
    body_pose[:3, 3] = body_pos

    return body_pose

def get_site_pose(model, data, name):
    """ Get the site pose as 4x4 matrix """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    # fake_world is not found
    if site_id == -1:
        return np.eye(4)
    
    site_pos = data.site_xpos[site_id]
    site_mat = data.site_xmat[site_id].reshape(3, 3)

    site_pose = np.eye(4)
    site_pose[:3, :3] = site_mat
    site_pose[:3, 3] = site_pos

    return site_pose

def normalize(arr):
    return arr / max(np.linalg.norm(arr), 1e-5)


class MuJoCoSimulatorNode(Node):
    def __init__(self):
        node_name = 'mujoco_ros'
        super().__init__(node_name)

        # frequency
        self.declare_parameter('object_state_freq', 100)
        self.declare_parameter('contact_state_freq', 100)
        self.declare_parameter('joint_state_freq', 100)
        object_state_freq = self.get_parameter('object_state_freq').get_parameter_value().integer_value
        contact_state_freq = self.get_parameter('contact_state_freq').get_parameter_value().integer_value
        joint_state_freq = self.get_parameter('joint_state_freq').get_parameter_value().integer_value

        self.declare_parameter('model_package', 'leap_hand_custom_description')
        self.declare_parameter('model_url', 'mjcf/leap_with_free_box_mjpc/task.xml')
        self.declare_parameter('xml_joint_prefix', 'joint_')
        self.declare_parameter('urdf_joint_prefix', 'joint_')
        self.declare_parameter('object_body_name', 'object')
        self.declare_parameter('model_init_qpos', [-0.75, 0.5, 0.75, 0.25, 0.0, 0.5, 0.75, 0.25, 0.75, 0.5, 0.75, 0.25, 0.65, 0.9, 0.75, 0.6])
        self.declare_parameter('object_init_qpos', [0.1, 0.025, 0.035, 1.0, 0.0, 0.0, 0.0])
        self.declare_parameter('joint_idx_range', [0, 16])
        self.declare_parameter('contact_geom_names', \
                               ['leap_hand_right::fingertip_collision',
                                'leap_hand_right::fingertip_2_collision',
                                'leap_hand_right::fingertip_3_collision',
                                'leap_hand_right::thumb_fingertip_collision'])
        self.declare_parameter('object_geom_name', 'object_geom')
        self.declare_parameter('is_mjpc', True)

        # Get the value of 'model_url' parameter
        model_package = self.get_parameter('model_package').get_parameter_value().string_value
        model_url = self.get_parameter('model_url').get_parameter_value().string_value
        xml_joint_prefix = self.get_parameter('xml_joint_prefix').get_parameter_value().string_value
        urdf_joint_prefix = self.get_parameter('urdf_joint_prefix').get_parameter_value().string_value
        object_body_name = self.get_parameter('object_body_name').get_parameter_value().string_value
        model_init_qpos = self.get_parameter('model_init_qpos').get_parameter_value().double_array_value
        object_init_qpos = self.get_parameter('object_init_qpos').get_parameter_value().double_array_value
        contact_geom_names = self.get_parameter('contact_geom_names').get_parameter_value().string_array_value
        object_geom_name = self.get_parameter('object_geom_name').get_parameter_value().string_value
        is_mjpc = self.get_parameter('is_mjpc').get_parameter_value().bool_value

        self.model_init_qpos = model_init_qpos
        self.object_init_qpos = object_init_qpos
        self.is_mjpc = is_mjpc

        # TODO(yongpeng): re-write the following with parameters
        joint_idx_start, joint_idx_end = self.get_parameter('joint_idx_range').get_parameter_value().integer_array_value
        self.joint_names = [f'{xml_joint_prefix}{i}' for i in range(joint_idx_start, joint_idx_end)]   # joint names used in xml
        self.topic_joint_names = [f'{urdf_joint_prefix}{i}' for i in range(joint_idx_start, joint_idx_end)]  # joint names used in ROS2 topics
        self.num_joints = len(self.joint_names)

        # Load MuJoCo model (MJCF or XML)
        model_path = os.path.join(
            get_package_share_directory(model_package),
            model_url
        )
        if not os.path.exists(model_path):
            self.get_logger().error(f"Model file not found: {model_path}")
            return
        self.get_logger().info(f"Loading model from {model_path}")

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.mj_lock = threading.Lock()         # mutex lock for simulation

        ## Parse the model
        ## ----------------------------------------
        # Parse the object
        worldbody_id = 0
        object_body_indice = -1
        for i in range(1, self.model.nbody):
            if self.model.body_parentid[i] == worldbody_id:
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name == object_body_name:
                    object_body_indice = i
                    self.get_logger().info(f"Find object body {body_name} with id {i}, creating the publisher!")
                    break
        if object_body_indice == -1:
            self.get_logger().error(f"Object with name {object_body_name} not found!")

        self.object_state_pub = self.create_publisher(PoseStamped, 'object_states', 10)
        self.object_state_publish_timer = self.create_timer(1.0 / object_state_freq, self.publish_object_states)

        self.object_body_indice = object_body_indice

        # Parse finger contacts
        # geom -> the contact geometry
        # site -> the contact force/point will be represented in site
        geom_ids, site_ids = [], []
        geom_to_site_map, geom_to_name_map = {}, {}
        for cid, name in enumerate(contact_geom_names):
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)

            if geom_id != -1 and site_id != -1:
                self.get_logger().info(f"Find contact geom {name} with id {geom_id}, creating publisher!")
            else:
                self.get_logger().warn(f"Failed to create contact sensor since geom {name} is not defined in the MJCF!")

            geom_ids.append(geom_id)
            site_ids.append(site_id)
            geom_to_site_map[geom_id] = site_id
            geom_to_name_map[geom_id] = name
        
        # Object geom ID for collision filter
        obj_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, object_geom_name)
        if obj_geom_id == -1:
            self.get_logger().error(f"Please set the name of object collision geometry as {object_geom_name} in MJCF!")
        else:
            self.get_logger().info(f"Object geom id {obj_geom_id}!")

        self.object_geom_id = obj_geom_id
        self.contact_geom_ids = geom_ids
        self.contact_site_ids = site_ids
        self.geom_to_site_map = geom_to_site_map
        self.geom_to_name_map = geom_to_name_map

        self.contact_state_pub = self.create_publisher(ContactState, '/contact_sensor_state', 10)
        self.contact_state_publish_timer = self.create_timer(1.0 / contact_state_freq, self.publish_contact_states)
        ## ----------------------------------------

        # parse hand joint ids
        self.qpos_indices, self.qvel_indices, self.qfrc_indices = [], [], []
        hand_jnt_ids = []
        assert self.data.ctrl.shape[0] == self.num_joints
        for i in range(len(self.joint_names)):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.joint_names[i])
            hand_jnt_ids.append(joint_id)
            self.qpos_indices.append(self.model.jnt_qposadr[joint_id])
            self.qvel_indices.append(self.model.jnt_dofadr[joint_id])
            self.qfrc_indices.append(self.model.jnt_dofadr[joint_id])
        # ctrl indices start from 0
        self.ctrl_indices = np.array(self.qpos_indices) - min(self.qpos_indices)

        # parse object joint id (for MJPC, maybe conflicts with original code)
        obj_qpos_indices, obj_qvel_indices = [], []
        obj_body_id_mjpc = -1
        for i in range(self.model.njnt):
            if i in hand_jnt_ids:
                continue
            q_start = self.model.jnt_qposadr[i]
            v_start = self.model.jnt_dofadr[i]
            if i == self.model.njnt - 1:
                q_end = self.model.nq
                v_end = self.model.nv
            else:
                q_end = self.model.jnt_qposadr[i+1]
                v_end = self.model.jnt_dofadr[i+1]
            obj_qpos_indices = list(range(int(q_start), int(q_end)))
            obj_qvel_indices = list(range(int(v_start), int(v_end)))
            
            # object body id
            obj_body_id_mjpc = self.model.jnt_bodyid[i]
            break

        self.obj_qpos_indices = obj_qpos_indices
        self.obj_qvel_indices = obj_qvel_indices
        self.obj_body_id_mjpc = obj_body_id_mjpc
        self.mjpc_object_state_pub = self.create_publisher(Float64MultiArray, '/mjpc_object_state', 10)
        self.object_quat_pub = self.create_publisher(Float64MultiArray, '/object_quat', 10)

        # Set initial state
        self.initialize_state_and_control()

        # Get palm pose
        self.step_simulation()
        with self.mj_lock:
            mujoco.mj_forward(self.model, self.data)        # update site pose
        fake_world_pose = get_site_pose(self.model, self.data, 'fake_world')
        self.fake_world_pose = fake_world_pose
        self.inv_fake_world_pose = np.linalg.inv(fake_world_pose)
        print('fake world pose: ', fake_world_pose)

        # Initialize viewer for rendering
        self.viewer = viewer.launch_passive(self.model, self.data)

        # Create publishers
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.mjpc_joint_state_pub = self.create_publisher(Float64MultiArray, '/mjpc_joint_states', 10)
        self.contact_force_gt_pub = self.create_publisher(Marker, 'contact_force_gt', 10)

        # Create services
        self.reset_service = self.create_service(Trigger, f'{node_name}/reset', self.reset_handler)
        self.srv = self.create_service(ApplyLinkFT, 'apply_link_force_torque', self.apply_link_force_torque_callback)

        # Messages
        self.external_force = ApplyLinkFT.Request()
        self.external_force.link_name = ''

        # Run the simulation
        self.timer = self.create_timer(0.002, self.simulate_callback)
        self.joint_state_publish_timer = self.create_timer(1.0 / joint_state_freq, self.publish_joint_states)
        self.desired_joint_pos_sub = self.create_subscription(JointState, f"/{node_name}/joint_commands", self.set_desired_joint_pos_callback, 10)

    def reset_handler(self, request, response):
        self.reset()

        self.get_logger().info("Mujoco environment has been reset.")
        response.success = True
        return response

    def reset(self):
        with self.mj_lock:
            mujoco.mj_resetData(self.model, self.data)
        self.initialize_state_and_control()
        self.step_simulation()
        self.viewer.sync()

    def initialize_state_and_control(self):
        # initialize hand state
        self.data.qpos[self.qpos_indices] = np.array(self.model_init_qpos)
        self.data.ctrl[self.ctrl_indices] = self.data.qpos[self.qpos_indices]
        print("initial hand qpos: ", self.data.qpos[self.qpos_indices])

        # initialize object state
        if len(self.object_init_qpos) != len(self.obj_qpos_indices):
            self.data.qpos[self.obj_qpos_indices] = 0
        else:
            self.data.qpos[self.obj_qpos_indices] = np.array(self.object_init_qpos)
        self.data.qvel[self.obj_qvel_indices] = 0
        print("initial object qpos: ", self.data.qpos[self.obj_qpos_indices])

        print("initial ctrl: ", self.data.ctrl[self.ctrl_indices])

    def step_simulation(self):
        with self.mj_lock:
            mujoco.mj_step(self.model, self.data)

    def simulate_callback(self):
        # Apply external force
        # self.apply_link_force_torque_runtime()

        # Step the simulation
        with self.mj_lock:
            mujoco.mj_step(self.model, self.data)

        # Render the scene
        self.viewer.sync()

        # Optionally, publish states or joint information via ROS2 topics here
        # e.g. self.publisher.publish(...)

    def destroy(self):
        # Cleanup when shutting down the node
        self.viewer.close()
        super().destroy_node()

    def reset_mjpc(self):
        """ Check terminate condition for MJPC """
        self.reset()

    def publish_joint_states(self):
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        mjpc_joint_state_msg = Float64MultiArray()

        # joint_state_msg.name = self.joint_names
        joint_state_msg.name = self.topic_joint_names

        joint_pos = self.data.qpos[self.qpos_indices]
        joint_vel = self.data.qvel[self.qvel_indices]
        joint_frc = self.data.qfrc_applied[self.qfrc_indices]

        joint_state_msg.position = np.asarray(joint_pos, dtype=np.float32).tolist()
        joint_state_msg.velocity = np.asarray(joint_vel, dtype=np.float32).tolist()
        joint_state_msg.effort = np.asarray(joint_frc, dtype=np.float32).tolist()

        # mjpc joint state uses joint order in MJCF model
        joint_pos_mjpc = self.data.qpos[np.sort(self.qpos_indices)]
        joint_vel_mjpc = self.data.qvel[np.sort(self.qvel_indices)]
        mjpc_joint_state_msg.data = np.concatenate((joint_pos_mjpc, joint_vel_mjpc)).tolist()

        self.joint_state_pub.publish(joint_state_msg)
        self.mjpc_joint_state_pub.publish(mjpc_joint_state_msg)

    def publish_object_states(self):
        object_state_msg = PoseStamped()
        object_state_msg.header.stamp = self.get_clock().now().to_msg()

        mjpc_object_state = Float64MultiArray()

        if self.is_mjpc:
            qpos = self.data.qpos[self.obj_qpos_indices]
            qvel = self.data.qvel[self.obj_qvel_indices]
            mjpc_object_state.data = np.concatenate((qpos, qvel)).tolist()

            self.mjpc_object_state_pub.publish(mjpc_object_state)

            obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'object')
            quat = self.data.xquat[obj_body_id]
            object_quat = Float64MultiArray()
            object_quat.data = quat.tolist()
            self.object_quat_pub.publish(object_quat)

        else:
            position = self.data.xpos[self.object_body_indice]
            quaternion = self.data.xquat[self.object_body_indice]
            position, quaternion = transform_pos_quat(self.inv_fake_world_pose, position, quaternion)

            object_state_msg.pose.position.x = position[0]
            object_state_msg.pose.position.y = position[1]
            object_state_msg.pose.position.z = position[2]

            object_state_msg.pose.orientation.w = quaternion[0]
            object_state_msg.pose.orientation.x = quaternion[1]
            object_state_msg.pose.orientation.y = quaternion[2]
            object_state_msg.pose.orientation.z = quaternion[3]

            self.object_state_pub.publish(object_state_msg)

    def get_contact_geom_names(self):
        """ get the geom names of current contacts """
        geom_names = []

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom[0]
            geom2 = contact.geom[1]

            # filter finger-object contacts
            if geom1 in self.contact_geom_ids and geom2 == self.object_geom_id:
                name = self.geom_to_name_map[geom1]
            elif geom2 in self.contact_geom_ids and geom1 == self.object_geom_id:
                name = self.geom_to_name_map[geom2]
            else:
                name = ''

            geom_names.append(name)

        return geom_names

    def publish_contact_states(self):
        """ Returned contact points and forces are in the world frame """

        ncon = self.data.ncon
        # self.get_logger().info(f"Number of contacts: {ncon}")

        contact_info_msg = ContactState()
        contact_info_msg.header.stamp = self.get_clock().now().to_msg()

        # for i in range(ncon):
        #     contact = self.data.contact[i]

        #     geom1 = contact.geom[0]
        #     geom2 = contact.geom[1]

        #     contact_force = np.zeros(3,)
        #     link_mat = np.zeros((3, 3))

        #     forcetorque = np.zeros(6)
        #     mujoco.mj_contactForce(self.model, self.data, i, forcetorque)

        #     if geom1 in self.contact_geom_ids:
        #         contact_name = self.geom_to_name_map[geom1]
        #         contact_force = -forcetorque[:3]
        #         link_pos = self.data.site_xpos[self.geom_to_site_map[geom1]]
        #         link_mat = self.data.site_xmat[self.geom_to_site_map[geom1]].reshape(3, 3)
        #     elif geom2 in self.contact_geom_ids:
        #         contact_name = self.geom_to_name_map[geom2]
        #         contact_force = forcetorque[:3]
        #         link_pos = self.data.site_xpos[self.geom_to_site_map[geom2]]
        #         link_mat = self.data.site_xmat[self.geom_to_site_map[geom2]].reshape(3, 3)
        #     else:
        #         continue

        #     # World frame
        #     # --------------------------------------------------------------
        #     contact_point = contact.pos
        #     contact_normal = normalize(contact.frame[:3])
        #     contact_mat = contact.frame.reshape(3, 3).T
        #     contact_force = contact_mat @ contact_force
        #     # --------------------------------------------------------------

        #     # Palm lower frame
        #     # --------------------------------------------------------------
        #     contact_point = (self.inv_fake_world_pose @ np.append(contact_point, 1))[:3]
        #     contact_force = (self.inv_fake_world_pose @ np.append(contact_force, 1))[:3]
        #     contact_normal = self.inv_fake_world_pose[:3, :3] @ contact_normal
        #     # --------------------------------------------------------------

        #     # Invert force and normal direction
        #     contact_force = -contact_force
        #     contact_normal = -contact_normal

        #     pos_msg, wrench_msg = get_contact_message(contact_point, contact_force, contact_normal)
        #     contact_info_msg.names.append(contact_name)
        #     contact_info_msg.points.append(pos_msg)
        #     contact_info_msg.wrenches.append(wrench_msg)

        #     # print(f"Contact between geom {geom1} and geom {geom2}")
        #     # print(f"Contact position: {contact_pos}")
        #     # print(f"Contact force: {contact_force}")
        #     # print(f"Contact normal force: {contact_normal_force}")

        current_contact_geom_names = self.get_contact_geom_names()

        for id in self.contact_geom_ids:
            contact_name = self.geom_to_name_map[id]
            if contact_name in current_contact_geom_names:
                index = current_contact_geom_names.index(contact_name)
                contact = self.data.contact[index]

                geom1 = contact.geom[0]
                geom2 = contact.geom[1]

                contact_force = np.zeros(3,)

                forcetorque = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, index, forcetorque)

                if geom1 in self.contact_geom_ids:
                    contact_force = -forcetorque[:3]
                elif geom2 in self.contact_geom_ids:
                    contact_force = forcetorque[:3]

                # World frame
                # --------------------------------------------------------------
                contact_point = contact.pos
                contact_normal = normalize(contact.frame[:3])
                contact_mat = contact.frame.reshape(3, 3).T
                contact_force = contact_mat @ contact_force
                contact_dist = contact.dist
                contact_point_on_geom1 = contact_point + 0.5 * contact_dist * contact_normal
                contact_point_on_geom2 = contact_point - 0.5 * contact_dist * contact_normal
                if geom1 in self.contact_geom_ids:
                    contact_point_on_finger = contact_point_on_geom1
                    contact_point_on_object = contact_point_on_geom2
                elif geom2 in self.contact_geom_ids:
                    contact_point_on_finger = contact_point_on_geom2
                    contact_point_on_object = contact_point_on_geom1
                # --------------------------------------------------------------

                # Palm lower frame
                # --------------------------------------------------------------
                contact_point = (self.inv_fake_world_pose @ np.append(contact_point, 1))[:3]        # mid point
                contact_point_on_finger = (self.inv_fake_world_pose @ np.append(contact_point_on_finger, 1))[:3]
                contact_point_on_object = (self.inv_fake_world_pose @ np.append(contact_point_on_object, 1))[:3]
                contact_force = (self.inv_fake_world_pose @ np.append(contact_force, 1))[:3]
                contact_normal = self.inv_fake_world_pose[:3, :3] @ contact_normal
                # --------------------------------------------------------------

                # Invert force and normal direction
                contact_force = -contact_force
                contact_normal = -contact_normal

                pos_msg, wrench_msg = get_contact_message(contact_point_on_finger, contact_force, contact_normal)

            else:
                pos_msg, wrench_msg = get_contact_message(np.zeros(3,), np.zeros(3,), np.zeros(3,))
            
            contact_info_msg.names.append(contact_name)
            contact_info_msg.points.append(pos_msg)
            contact_info_msg.wrenches.append(wrench_msg)

        self.contact_state_pub.publish(contact_info_msg)

    def set_desired_joint_pos_callback(self, msg:JointState):
        def get_jnt_remap(rec_jnt, des_jnt):
            """ Get the mapping from rec_jnt to des_jnt """
            remap = []
            for jnt in des_jnt:
                remap.append(rec_jnt.index(jnt))
            return remap
        
        remap = get_jnt_remap(msg.name, self.joint_names)
        jnt_cmds = np.array(msg.position)[remap]

        assert len(jnt_cmds) == len(self.qpos_indices)
        self.data.ctrl[self.ctrl_indices] = jnt_cmds

    def apply_link_force_torque_callback(self, request, response):
        try:
            bodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, request.link_name)
            assert bodyid != -1
            # self.external_force = request
            # clear previous force
            self.data.qfrc_applied[:] = 0
            with self.mj_lock:
                mujoco.mj_applyFT(self.model, self.data, request.force, request.torque, request.point, bodyid, self.data.qfrc_applied)
                mujoco.mj_forward(self.model, self.data)
            
            # # debug the mj_applyFT function
            # Jc = np.empty((3, self.model.nv))
            # mujoco.mj_jac(self.model, self.data, Jc, None, request.point, bodyid)
            # tau_ext = Jc.T @ request.force
            # self.get_logger().info(f"external torque: {tau_ext}")

            response.success = True

            # # Publish ground truth force
            # from scipy.spatial.transform import Rotation as R
            # body_pos = self.data.xpos[bodyid]
            # body_quat = self.data.xquat[bodyid]
            # body_rot = R.from_quat(body_quat[[1, 2, 3, 0]])
            # contact_point = body_rot.apply(request.point) + body_pos
            # contact_force = body_rot.apply(request.force)
            contact_point = np.array(request.point).astype(np.float64)
            contact_force = np.array(request.force).astype(np.float64)

            contact_point = (self.inv_fake_world_pose @ np.append(contact_point, 1))[:3]
            contact_force = (self.inv_fake_world_pose @ np.append(contact_force, 1))[:3]

            fext_gt_marker = get_force_vis_message(contact_force, contact_point, marker_id=1)
            fext_gt_marker.header.stamp = self.get_clock().now().to_msg()
            self.contact_force_gt_pub.publish(fext_gt_marker)

            self.get_logger().info(f"Applying force {request.force} at point {request.point} on link {request.link_name}")

            return response
        except Exception as e:
            print(e)
            response.success = False
            return response
        
    def apply_link_force_torque_runtime(self):
        if self.external_force.link_name == '':
            return
        bodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.external_force.link_name)
        assert bodyid != -1
        with self.mj_lock:
            mujoco.mj_applyFT(self.model, self.data, self.external_force.force, self.external_force.torque, self.external_force.point, bodyid, self.data.qfrc_applied)
            mujoco.mj_forward(self.model, self.data)

def main(args=None):
    rclpy.init(args=args)

    # Create and spin the MuJoCo simulator node
    mujoco_simulator = MuJoCoSimulatorNode()

    # try:
    #     rclpy.spin(mujoco_simulator)
    # except KeyboardInterrupt:
    #     pass

    rclpy.spin(mujoco_simulator)

    mujoco_simulator.destroy()
    rclpy.shutdown()

if __name__ == '__main__':
    main()