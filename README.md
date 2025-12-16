# AROB ROS Docker Development Environment

This repository is based on [robo_dev](https://github.com/dvdmc/robo_dev) and provides instructions to set up the ROS 2 development environment for the AROB laboratory sessions using Docker.

## Dockerfile

The base of the environment is the **Dockerfile**. Using **docker-compose** (see below), it will get built into a Docker **image**. Then, docker-compose will use that **image** to instantiate a **container** with mapped **volumes**. We then will **attach** a terminal to that container. Everything that you want to install **persistently** will be included in the Dockerfile or mapped to a volume in the host machine. Anything that is not included in the Dockerfile and is not part of a mapped volume **will be removed when the container is stopped**.


## docker-compose.yaml

This file is the alternative to the usual long Docker commands. Here you will include everything related to **environment variables, mapped volumes, network configuration**. From all the configuration, the main important part for you are the mapped volumes. We map a `workspace` folder from the host machine into the Docker to store persistent changes to the code and keep the build files. You can see this config in the `docker-compose.yaml` file:

```
- ./arob_ws:/root/arob_ws
```
If you want to include additional packages in your workspace, you can either clone them directly into the `src` directory or use `rosdep` inside the container to install dependencies. Since the entire workspace is mounted from the host, all changes will persist even after the container is stopped.

NOTES: 
- The container command is `tail -f /dev/null` which basically keeps that container in idle an idle mode. Then, we will attach from another terminal to the idle container.
-  Additionally, if you want to **store results, logs, or maps from your experiments**, you can create another mapped volume where you will store them so they can be persisted out of the machine.


## 🖥️ Enabling GUI Applications (rqt, rviz, Gazebo) via Xserver

To use GUI applications (e.g. `rqt`, `rviz`, `gazebo`) inside the Docker container, you need to share your display from the host with the container using **X11**.

This works on both Linux and Windows (via VcXsrv).

### On Linux

🔄️**Every time you reboot** make sure to give the right permissions by running
```
xhost +
```

### On windows (using VcXsrv)
1. Install an X server like [VcXsrv](https://sourceforge.net/projects/vcxsrv/)
2. Start it with:
    - Set display number to `:0`;
    - Uncheck `Native opengl`;
    - Check `Disable access control`;
3. Set `DISPLAY` to `host.docker.internal:0.0` and `QT_X11_NO_MITSHM` in your `docker-compose.yaml`:
```
environment:
  DISPLAY: host.docker.internal:0.0
  QT_X11_NO_MITSHM: "1"
```
4. Mount the **X11** socket in the `docker-compose.yaml`:
```
volumes:
    - ~/.Xauthority:/root/.Xauthority
```

### ✅ Testing the GUI 
From inside the container, we can try to run the principal tools that require a graphical interface (GUI), e.g
```
ros2 run rviz2 rviz2        # or gazebo, or rqt, ... 
```

## 🚀Starting the environment

### Step 1: Enable Access to Your Host Display for GUI Applications
If you want to run graphical ROS tools inside the Docker container (like `rqt`, `rviz`, or `gazebo`), you need to allow the Docker container to access your host computer’s display server.
**Important**: This command must be run on your **host machine**, not inside the Docker container.

```
xhost +
```
This temporarily allows any program (including the Docker container) to connect to your X server and display windows on your screen.

### Step 2: Start the Docker Container
Start your container in detached mode `-d` (runs in the background):

```
sudo docker compose up -d
```

### Step 3: Attach to the Running Container

Open a terminal inside the running Docker container:

```
sudo docker exec -it arob_lab /bin/bash
```

### Step 4: Use ROS Tools Inside the Container

Once inside the container shell, you can run ROS commands. 
```
ros2 run <package_name> <node_name>
ros2 launch <package_name> <launch_file.launch.py>
```

### Step 5 (Optional): Build Your Workspace
If you have your own package with source files inside the workspace, you need to build it before running your nodes.
```
cd /root/arob_ws
colcon build --symlink-install 
```
After building, source the workspace setup script to overlay your environment:
```
source install/setup.bash
```

## 🧵 Tmux and Tmuxinator

In order to multiplex the terminal and allow a clear execution of multiple nodes in different terminals we use [tmux](https://github.com/tmux/tmux/wiki). `tmux` is a terminal multiplexer so you don't have to open a new terminal for each node. It also allows to detach and reconnect, which is very useful for `ssh` remote sessions. You can check a tutorial for tmux later [here](https://hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/). To simplify things for you, we already include in the Dockerfile two configs for tmux:
- `set -g mouse on`: enables scroll in tmux
- `move between panes with alt + arrow keys`: enables moving using arrow keys in tmux
- `bind | and - to split vertically and horizontally`: enables split in tmux using `|` and `-` instead of `"` and `%`.

In order to configure layouts in `tmux` that we can load directly, we will use `tmuxinator`. This is used in well-established packages such as [Aerostack2](https://aerostack2.github.io/).
You can check the `launch-ws.yaml`used for the tmuxinator execution inside the `ros_package`. The rest of the package is a mockup but empty.

First, navigate to the directory containing your `launch-ws.yaml` configuration file and run:

```
tmuxinator start -n ros_env -p launch-ws.yaml
```
This will start a new Tmux session named `ros_env` and launch the layout specified in the YAML file.

To exit the tmuxinator session, you can press `ctrl+b`, then `:` to enter command mode and type `detach` or `kill-session`.

## 📦Stopping and Cleaning Up
When you're done working in your Docker environment, stop the container with:

```
sudo docker compose down
```

Remember that anything that is not a mapped volume will be removed. If you want to extract results, logs, or maps, you will need to store them in a mapped volume (see above).


## Example: `arob_basic_pkg`

This folder contains a ROS 2 package, including the `talker` and `listener` nodes written in C++.

When using **Option 2 and 3** (persistent workspace and package), any changes made to files like `.cpp`, `CMakeLists.txt`, or `package.xml` from the **host machine** are automatically reflected inside the Docker container. This allows you to develop outside the container with full persistence.

---

### 🔄 Workflow

After making any changes to the package (e.g., editing source files), follow these steps **inside the container**:
```
cd /root/arob_ws
colcon build --symlink-install
source install/setup.bash
```

To run the nodes (example):
```
ros2 run arob_basic_pkg talker
```
```
ros2 run arob_basic_pkg listener
```

---

### 💡 Notes

- You can edit the package code from your **host machine** using any editor (VS Code, Vim, etc.).
- If you want to add more packages to the workspace, create them inside the `arob_ws/src` directory.

This setup ensures a persistent and flexible ROS development workflow using Docker.

## 🐳 Final Remarks: Rebuilding the container

If you make changes to the `Dockerfile` or `docker-compose.yaml`, rebuild and restart the container with:

```
docker compose up -d --build
```
Nevertheless, it is good practice to rebuild the entire image when substantial changes are made. To this purpose, use
```
docker compose build --no-cache
```

This will reapply any updates while preserving your workspace and code.
