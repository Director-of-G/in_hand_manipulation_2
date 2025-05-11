# Robust Model-Based In-Hand Manipulation with Integrated Real-Time Motion-Contact Planning and Tracking

[[Project Website](https://director-of-g.github.io/in_hand_manipulation_2/)]

Repository for the paper _Robust Model-Based In-Hand Manipulation with Integrated Real-Time Motion-Contact Planning and Tracking_, submitted to the International Journal of Robotics Research (IJRR).

In this repository, we provide:
- Implementation of the high-level motion-contact planner using the [CQDC model](https://github.com/pangtao22/quasistatic_simulator) and DDP solver based on [Crocoddyl](https://github.com/loco-3d/crocoddyl)
- Implementation of the low-level motion-contact tracker
- A simulator based on MuJoCo for running the planner and tracker in parallel, as well as for a quick test of the proposed framework

We do not provide code related to the hardware implementation, as it depends on your specific hardware setup.

<div align="center">
  <img src="./docs/media/task_snapshots.gif" alt="Task Snapshots" width="75%" />
</div>

## Installation

The following instructions has been tested on an empty Ubuntu 20.04 system and Python 3.8.

### (High-Level) Motion-Contact Planner

1. Install Anaconda3 and create a virtual environment
    ```
    conda create -n inhand python=3.8
    ```

2. Install Python requirements
    ```
    pip install -r requirements.txt
    pip install manipulation --no-dependencies
    ```

3. Install [Pinocchio](https://stack-of-tasks.github.io/pinocchio/download.html) and [Crocoddyl](https://stack-of-tasks.github.io/pinocchio/download.html), we recommend to install using robotpkg

4. Install Drake and Python API, please install exactly version 1.22.0 or you might encounter problems like segmentation fault
    ```
    # install Drake
    wget https://github.com/RobotLocomotion/drake/releases/download/v1.22.0/drake-dev_1.22.0-1_amd64-focal.deb
    sudo apt-get install --no-install-recommends ./drake-dev_1.22.0-1_amd64-focal.deb
    export LD_LIBRARY_PATH=/opt/drake/lib:$LD_LIBRARY_PATH
    
    # install Python API
    pip install --extra-index-url https://drake-packages.csail.mit.edu/whl/nightly/ 'drake==1.22.0'
    ```

5. Clone this repository
    ```
    git clone https://github.com/Director-of-G/in_hand_manipulation_2.git
    ```

6. Write these into `~/.bashrc`
    ```
    export INHAND_HOME=/path/to/in_hand_manipulation_2/high_level
    export QSIM_HOME=${INHAND_HOME}/quasistatic_simulator
    export PYTHONPATH=${INHAND_HOME}/planning_through_contact:${INHAND_HOME}/planner:${QSIM_HOME}:${QSIM_HOME}/quasistatic_simulator_cpp/build/bindings:${PYTHONPATH}
    ```

7. Play with the code (view the MeshCat visualization via http://localhost:7000/)
    ```
    cd high_level/planner/ddp/tasks
    python xxx.py
    ```

#### TroubleShooting
- **About the quasistatic simulator** To simplify installation, we provide the pre-built `*.so` of the quasistatic simulator (compared to the [official repo](https://github.com/pangtao22/quasistatic_simulator), we add features like numeric differentiation and separation of the dynamics and gradient computation). You can locate the `*.so` in `quasistatic_simulator/quasistatic_simulator_cpp/build`. Visit [this repo](https://github.com/Director-of-G/quasistatic_simulator) if you want to explore the source code. Follow the following instructions if you want to re-build.
    ```
    # Assume you have Pinocchio installed

    # Download Drake and unzip to $HOME (this may take long time)
    wget https://github.com/RobotLocomotion/drake/archive/refs/tags/v1.22.0.tar.gz

    # Install dependencies
    sudo /path/to/drake/setup/ubuntu/install_prereqs.sh

    # Build Drake (this may take long time)
    cd $HOME && mkdir drake-build && cd drake-build
    cmake ../drake -DWITH_MOSEK=ON
    make install

    # Add Drake to path
    export PYTHONPATH=$HOME/drake-build/install/lib/python3.8/site-packages:$PYTHONPATH
    export LD_LIBRARY_PATH=$HOME/drake-build/install/lib:$LD_LIBRARY_PATH

    # Install Boost (1.82.0) with python3.8
    # Install Eigen3

    # Clone and build quasistatic simulator
    git clone --recursive https://github.com/Director-of-G/quasistatic_simulator.git
    cd /path/to/quasistatic_simulator/quasistatic_simulator_cpp
    mkdir build && cd build
    cmake -DCMAKE_PREFIX_PATH=$HOME/drake-build/install -DEigen3_DIR:PATH=/usr/lib/cmake/eigen3 -DCMAKE_BUILD_TYPE=Release ..
    make
    ```
- **About the self-collision term** Some of the `xxx.py` in `high_level/planner/ddp/tasks` enable self-collsion terms, the implementation of which requires SDF support of Pinocchio. To enable this feature, you need to build Pinocchio and Crocoddyl from source. Although a detailed procedure is under development, we provide a few suggestions here
    - Follow the instructions. Please refer to the instructions for [Pinocchio](https://stack-of-tasks.github.io/pinocchio/download.html#Install_9) and [Crocoddyl](https://github.com/loco-3d/crocoddyl#file_folder-from-source)
    - Install the specific version. We recommend `v3.3.0` for Pinocchio and `v2.1.0` for Crocoddyl
    - Turn ON the SDF option when configuring Pinocchio. Add `-DBUILD_WITH_SDF_SUPPORT=ON` when running `cmake ..`
    - These steps have not been tested on a new system, and you might encounter problems. Feel free to open an issue

### (Low-Level) Motion-Contact Tracker

1. Install ROS2 Foxy and build tools
    ```
    sudo apt install python3-colcon-common-extensions python3-catkin-pkg
    ```

2. Install some ROS2 packages
    ```
    sudo apt install ros-foxy-xacro ros-foxy-moveit-msgs ros-foxy-urdfdom-py
    ```

3. Install some Python dependencies
    ```
    # Install Eigen (3.4) used by grampc
    pip install git+https://github.com/grampc/pygrampc
    pip install lxml mujoco
    ```

4. Although we do not provide a standalone script that you can test the low-level tracker solely, you can explore the force-motion model in `contact_model_grampc.py` and controller in `contact_controller.py`, which can be located [here](./low_level/ros2_ws/src/inhand_lowlevel/leap_ros2/leap_ros2/)

### Run the Framework with ROS2 and MuJoCo

1. Build the low-level and high-level ROS workspaces
    ```
    cd /path/to/in_hand_manipulation_2
    
    # (Terminal 1) build the low-level
    cd low_level/ros2_ws
    colcon build --symlink-install
    source install/setup.bash

    # (Terminal 2) build the high-level
    # source the low-level
    source low_level/ros2_ws/install/setup.bash
    cd high_level/ros2_ws
    colcon build --symlink-install
    source install/setup.bash
    ```

2. Launch a MuJoCo simulation of the Rotate Sphere task, which requires the high-level and low-level modules to run in parallel
    ```
    # (Terminal 1) 
    ros2 launch contact_rich_control launch_highlevel.launch.py

    # (Terminal 2)
    ros2 launch leap_ros2 launch_lowlevel.launch.py

    # (Terminal 3) execute 100 goal orientations, with a time budget 60s for each
    source low_level/ros2_ws/install/setup.bash
    cd low_level/ros2_ws/src/inhand_lowlevel/leap_ros2/scripts/journal
    python ./test_system_with_reset.py
    ```

#### TroubleShooting

- **About poor performance**. The experiments of our paper is conducted on a Ubuntu 20.04 workstation with Intel i9-13900KF CPU and 32GB RAM. The control frequency may vary with different hardware configuration. The high-level and low-level should run approximately at 10Hz and 30Hz, respectively.

### Citation

Please cite our paper if you find it helpful :)

```
misc{jiang2025robustmodelbasedinhandmanipulation,
      title={Robust Model-Based In-Hand Manipulation with Integrated Real-Time Motion-Contact Planning and Tracking}, 
      author={Yongpeng Jiang and Mingrui Yu and Xinghao Zhu and Masayoshi Tomizuka and Xiang Li},
      year={2025},
      eprint={2505.04978},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.04978}, 
}
```