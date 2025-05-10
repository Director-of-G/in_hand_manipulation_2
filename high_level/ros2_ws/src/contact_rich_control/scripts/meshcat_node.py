#!/usr/bin/env python3

# This node updates meshcat visualization

from copy import deepcopy
import sys
import time
import numpy as np
import yaml

from contact_rich_control.common_hw import *
sys.path.append(DDP_SOLVER_DIR)

import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters
from std_msgs.msg import Float64MultiArray
from common_msgs.msg import MeshcatVis

from qsim.parser import QuasistaticParser
from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer
from common.inhand_ddp_helper import InhandDDPVizHelper
from manipulation.meshcat_utils import AddMeshcatTriad

from pydrake.all import BodyIndex


def get_param(node_handle:Node, node_name:str, param_name:str, timeout=0.0):
    """ Block if timeout is None or negative. Don't wait if 0. """
    node_handle.get_logger().info(f'Wait for parameter /{node_name}/{param_name}...')
    client = node_handle.create_client(GetParameters, f'/{node_name}/get_parameters')
    client.wait_for_service(timeout_sec=timeout)
    request = GetParameters.Request()
    request.names = [param_name]
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node_handle, future, timeout_sec=timeout)
    if future.done():
        response = future.result()
        node_handle.get_logger().info(f'Parameter /{node_name}/{param_name} is set!')
        return response.values[0]
    else:
        node_handle.get_logger().error(f'Get parameter /{node_name}/{param_name} failed!')
        return
    
def get_body_with_name(mbp, name):
    for i in range(mbp.num_bodies()):
        body = mbp.get_body(BodyIndex(i))
        if body.name() == name:
            return body
    return None
class MeshcatNode(Node):
    def __init__(self):
        super().__init__('meshcat_node')
        self.acquire_high_level_params()
        self.prepare_visualization()

        self.get_logger().info('Meshcat node ready to take commands!')

        ## SUBSCRIBERS
        # ----------------------------------------
        # dof states
        self.dof_states_subscriber_ = self.create_subscription(
            MeshcatVis,
            '/meshcat/visualization',
            self.dof_states_callback,
            10
        )

    def acquire_high_level_params(self):
        """ Get high level parameters through ROS service """
        params = get_param(self, 'highlevel_controller_node', 'params', timeout=None).string_value
        self.get_logger().info(f'High level params: {params}')
        self.highlevel_params = yaml.safe_load(params)
        self.model_url = get_param(self, 'highlevel_controller_node', 'model_url', timeout=None).string_value
        self.object_link = self.highlevel_params['object_link']

    def prepare_visualization(self):
        parser = QuasistaticParser(self.model_url)
        q_vis = QuasistaticVisualizer.make_visualizer(parser)
        meshcat = q_vis.q_sim_py.meshcat
        viz_helper = InhandDDPVizHelper(meshcat)
        viz_helper.set_elements_in_meshcat()
        viz_helper.finger_to_geom_name_map = self.highlevel_params['finger_to_geom_name_map']
        
        self.object_body = get_body_with_name( q_vis.q_sim_py.get_plant(), self.object_link)
        if self.object_body is None:
            self.get_logger().error(f'Object body with name {self.object_link} not found!')
            return

        self.q_vis = q_vis
        self.q_sim = self.q_vis.q_sim_py
        self.mbp_context = q_vis.q_sim_py.context_plant
        self.meshcat = meshcat
        self.viz_helper = viz_helper

        self.is_visualizing_traj = False

    def get_object_pose(self):
        """ Must be called after q_vis.draw_configuration """
        if self.object_body is None:
            return
        pose = self.object_body.body_frame().CalcPoseInWorld(self.mbp_context)
        return pose
    
    def plot_object_axes(self, q, q_goal):
        """ Plot axes of object current and goal pose """
        # draw goal
        self.q_sim.update_mbp_positions_from_vector(q_goal)
        X_WG = self.get_object_pose()
        AddMeshcatTriad(
            meshcat=self.meshcat,
            path="drake/frames/goal",
            length=0.1, radius=0.004, opacity=0.5,
            X_PT=X_WG
        )

        # draw current
        self.q_sim.update_mbp_positions_from_vector(q)
        X_WO = self.get_object_pose()
        AddMeshcatTriad(
            meshcat=self.meshcat,
            path="drake/frames/object",
            length=0.1, radius=0.004, opacity=1.0,
            X_PT=X_WO
        )

    def delete_previous_traj(self):
        self.q_vis.publish_trajectory([], h=0.1)

    def dof_states_callback(self, msg:MeshcatVis):
        vis_trajectory = msg.vis_trajectory.data
        if vis_trajectory:
            q_traj_arr = []
            for i in range(len(msg.q_traj)):
                q_traj_arr.append(msg.q_traj[i].data)
            self.q_vis.publish_trajectory(q_traj_arr, h=0.1)
            self.is_visualizing_traj = True
        else:
            # delete previous recording
            if self.is_visualizing_traj:
                self.delete_previous_traj()
                self.is_visualizing_traj = False
            self.plot_object_axes(msg.q.data, msg.q_goal.data)
            self.q_vis.draw_configuration(msg.q.data)
            p_W = np.array(msg.p_world.data).reshape(-1, 3)
            f_W = np.array(msg.f_world.data).reshape(-1, 3)
            self.viz_helper.plot_contact_points(p_W, already_in_order=True)
            self.viz_helper.plot_contact_forces(p_W, f_W, already_in_order=True)

def main(args=None):
    rclpy.init(args=args)
    node = MeshcatNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()