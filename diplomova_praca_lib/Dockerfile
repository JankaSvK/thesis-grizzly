# Annotating images
FROM tensorflow/tensorflow:latest-gpu-py3


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
RUN pip3 install scikit-build

RUN apt install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt update -y; apt install -y gcc-6 g++-6

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 50
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-6 50

RUN pip3 install face_recognition

WORKDIR /project
COPY . .
RUN pip install -e .

CMD /bin/bash