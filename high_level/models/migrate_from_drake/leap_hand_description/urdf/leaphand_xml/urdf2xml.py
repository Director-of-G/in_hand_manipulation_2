import mujoco

model = mujoco.MjModel.from_xml_path("leaphand.urdf")

mujoco.mj_saveLastXML("leaphand.xml", model)
