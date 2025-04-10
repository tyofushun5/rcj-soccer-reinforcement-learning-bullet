FROM nvcr.io/nvidia/pytorch:23.09-py3
USER root

COPY rcj_soccer_reinforcement_learning_pybullet .

RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

RUN pip install \
    pybullet \
    stable-baselines3[extra] \
    opencv-python \
    imageio[ffmpeg] \
    matplotlib
