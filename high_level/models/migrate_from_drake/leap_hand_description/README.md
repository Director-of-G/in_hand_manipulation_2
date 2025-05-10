# Description



## Changelog

- 2024-04-16
    - Move all mesh files to `meshes` folder. Meshes in other places are symlinked, using `python/symlink_stls.py`.
    - Deleted `fingertip_original.stl`, `fingertip.stl`, and `thumb_fingertip.stl`.
    - Rename `fingertip_custom_sim.STL` to `fingertip_custom.stl`, and change all `.STL` to `.stl` for better processing in python. Theoretically I modified every occurrence in urdf files.

## Original Readme

### Generate .urdf from .xacro

1. Change the 'mesh_dir' in .xacro.

1. Auto generation:
    ```
    cd catkin_ws
    source devel/setup.bash

    cd your/path/to/urdf
    rosrun xacro xacro -o leaphand.urdf leaphand.urdf.xacro
    ```

1. Change the 'package_dir' back.


### Generate .xml from .urdf

1. Create a new folder called 'leaphand_xml'
1. Copy the leaphand.urdf to this folder.
1. Copy all the mesh file to this folder (no subfolder).
1. Modify and run the ```urdf2xml.py```. 
1. In the generated leaphand: 
    1. Below ```<worldbody>```, add ```<body name="palm_lower">```.
    1. Above ```</worldbody>```, add ```</body>```.
1. In a new 'leaphand_mujoco.xml', include the 'leaphand.xml', add actuators and excluded contacts.