# Select your ROS version or container to start from
FROM osrf/ros:humble-desktop-full

# Some basic tools that you will include everytime
#RUN apt-get update && apt-get upgrade -y
RUN apt-get update --fix-missing && apt-get upgrade -y
RUN apt-get install -y  git \
    tmux \
    tmuxinator \
    vim \
    xterm \
    curl \
    ssh \
    tree \
    python3-tk \
    nano \
    libtool \
    libtool-bin \
    htop \
    gdb \
    net-tools \
    build-essential \
    cmake \
    lsb-release \
    iputils-ping

# Enable scroll in tmux
RUN printf "set -g mouse on\n\
bind -n M-Left select-pane -L\n\
bind -n M-Right select-pane -R\n\
bind -n M-Up select-pane -U\n\
bind -n M-Down select-pane -D\n\
\n\
# split panes using | and -\n\
bind | split-window -h\n\
bind - split-window -v\n\
unbind '\"'\n\
unbind %%\n" > /root/.tmux.conf


# For allowing connection between machines, fastdds has some problem.
RUN apt-get install -y ros-humble-rmw-cyclonedds-cpp

## ROS2 utils
RUN apt-get install -y \
    python3-rosdep \
    python3-setuptools \
    python3-pip \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    ros-dev-tools

# Add the source for the system ROS
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

## NAV2
RUN apt-get install -y --fix-missing \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-turtlebot3*   

# Add gazebo models
ENV TURTLEBOT3_MODEL=burger
ENV GAZEBO_MODEL_PATH=/opt/ros/humble/share/turtlebot3_gazebo/models/
 # Optional: ensure local communication
#ENV ROS_LOCALHOST_ONLY=1 
ENV ROS_NAMESPACE=simulation
# Optional: if using custom domain ID
#ENV ROS_DOMAIN_ID=1       
ENV USE_SIM_TIME=True


#export ROS_LOCALHOST_ONLY=1  # Optional: ensure local communication
#export ROS_NAMESPACE=simulation
#export ROS_DOMAIN_ID=1       # Optional: if using custom domain ID
#export USE_SIM_TIME=True


RUN echo "source /usr/share/gazebo/setup.sh" >> /root/.bashrc

#Laboratory enviornment
#RUN echo "source /root/arob_ws/install/setup.bash" >> /root/.bashrc

RUN apt-get install -y \
    ros-humble-aerostack2 \
    ros-humble-tf-transformations

RUN rm -rf log
WORKDIR /root/
