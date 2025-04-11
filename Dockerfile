FROM nvcr.io/nvidia/pytorch:23.09-py3
USER root

WORKDIR /workspace

COPY rcj_soccer_reinforcement_learning_pybullet ./rcj_soccer_reinforcement_learning_pybullet

RUN apt-get update && apt-get install -y\
    python3-pip\
    git\
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

RUN pip install\
    pybullet\
    stable-baselines3[extra]\
    matplotlib\
    sb3-contrib\
    opencv-contrib-python==4.5.5.64
