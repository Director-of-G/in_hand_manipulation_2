#!/usr/bin/env python3
import os

import rospy
import yaml
from sensor_msgs.msg import JointState

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
urdf_to_sim = [1, 0, 2, 3, 12, 13, 14, 15, 5, 4, 6, 7, 9, 8, 10, 11]


def save_leap_jpos():
    object_type = rospy.get_param("/object_type", "cube_50mm")

    joint_states = rospy.wait_for_message("/joint_states", JointState)
    joint_names = joint_states.name
    joint_positions = joint_states.position

    jpos_dic = {"zeros": {}, "canonical_pose": []}
    for name, pos in zip(joint_names, joint_positions):
        jpos_dic["zeros"][name] = pos

    for jid in urdf_to_sim:
        jpos_dic["canonical_pose"].append(joint_positions[joint_names.index(f"joint_{jid}")])

    yaml.safe_dump(jpos_dic, open(os.path.join(SAVE_DIR, f"default_leap_{object_type}.yaml"), "w"))


def save_leap_jpos_repeatedly():
    while not rospy.is_shutdown():
        save_leap_jpos()
        rospy.sleep(1)


if __name__ == "__main__":
    nh = rospy.init_node("save_leap_jpos_node", anonymous=True)
    save_leap_jpos_repeatedly()
