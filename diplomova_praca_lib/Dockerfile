# Annotating images
#FROM tensorflow/tensorflow:latest-gpu-py3
FROM nvidia/cuda:10.1-cudnn8-runtime


# Install face recognition dependencies
USER root
RUN apt update -y; apt install -y \
gcc-7 \
git \
cmake \
libsm6 \
libxext6 \
libxrender-dev \
python3 \
python3-pip \
libopenblas-dev \
liblapack-dev \
vim

RUN pip3 install --upgrade pip
RUN pip3 install scikit-build mxnet-cu101

RUN apt install -y software-properties-common libgl1-mesa-glx nano
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt update -y; apt install -y gcc-6 g++-6

# RUN cp /usr/local/cuda-10.1/compat/libcuda.so.1 /usr/local/cuda/lib64

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 50
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-6 50

RUN pip3 install face_recognition
RUN pip3 install tensorflow-gpu>=2.0 Pillow numpy opencv-python scikit-learn MiniSom matplotlib

ENV MXNET_CUDNN_AUTOTUNE_DEFAULT 0

WORKDIR /project
COPY . .
RUN pip install -e .

CMD /bin/bash