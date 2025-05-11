import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import numpy as np


# urdf_filename = "./push_2d.urdf"
urdf_filename = "./allegro_3d_full.urdf"
mesh_dir = "/home/yongpeng/research/projects/contact_rich/idto/dex_playground/mujoco_playground/contact_model/test_CQDC_model/models/mjcf/assets"

model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_filename, mesh_dir)

data = model.createData()
import pdb; pdb.set_trace()

viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer()
viz.loadViewerModel()

## stored hand configurations

# group1
# 0.3, 0.95, 1.0, 1.0,
# 0.0, 0.55, 1.0, 1.0,
# -0.17, 0.95, 1.0, 1.0,
# 0.45, 1.85, 1., 1.2,

# group2
# 0.4, 0.95, 0.9, 1.0,
# 0.0, 0.45, 1.0, 1.0,
# -0.25, 0.85, 1.0, 1.0,
# 0.45, 1.85, 1., 1.2,

# group3
# 0.2, 0.95, 1.0, 1.0,
# 0.0, 0.6, 1.0, 1.0,
# -0.2, 0.95, 1.0, 1.0,
# 0.5, 1.85, 1.0, 1.0,

q0 = np.array([
              0.2, 0.95, 1.0, 1.0,
              0.0, 0.6, 1.0, 1.0,
              -0.2, 0.95, 1.0, 1.0,
              0.5, 1.85, 1.0, 1.0,
              -0.06, 0.0, 0.07, 0, 0, 0, 1
            ])
viz.display(
  q0
)

# for frame_id in range(model.nframes):
#     import pdb; pdb.set_trace()
#     viz.addFrame(model, frame_id, 0.1)

# import pdb; pdb.set_trace()

# exit(0)

# Load model
model = pin.buildModelFromUrdf(urdf_filename)
 
# Load collision geometries
geom_model = pin.buildGeomFromUrdf(model, urdf_filename, pin.GeometryType.COLLISION)
 
# Add collisition pairs
geom_model.addAllCollisionPairs()
print("num collision pairs - initial:",len(geom_model.collisionPairs))
 
# Create data structures
data = model.createData()
geom_data = pin.GeometryData(geom_model)

q = np.array([0.0, 0.0, np.pi/2, 0.0, -0.12])
 
# Compute all the collisions
pin.computeCollisions(model, data, geom_model, geom_data, q, False)
 
# Print the status of collision for all collision pairs
for k in range(len(geom_model.collisionPairs)): 
  cr = geom_data.collisionResults[k]
  cp = geom_model.collisionPairs[k]
  print("collision pair:",cp.first,",",cp.second,"- collision:","Yes" if cr.isCollision() else "No")

pin.computeDistances(model, data, geom_model, geom_data, q)
import pdb; pdb.set_trace()

for i in range(geom_data.distanceResults.size()):
    distance_result = collision_data.distanceResults[i]
    if distance_result.min_distance < some_threshold:  # some_threshold 是接触的阈值
        contact_point = distance_result.nearest_points[0]

import pdb; pdb.set_trace()
