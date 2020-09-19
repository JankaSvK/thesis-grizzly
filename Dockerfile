FROM tensorflow/tensorflow:latest-gpu-py3
#FROM nvidia/cuda:10.1-cudnn8-runtime


ENV PYTHONUNBUFFERED 1
RUN apt update -y
RUN apt-get install -y libxrender-dev
RUN apt update && apt install -y libsm6 libxext6 libgl1-mesa-glx nano

RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install mxnet-cu101
RUN pip install tensorflow-gpu>=2.0 Pillow numpy opencv-python scikit-learn MiniSom matplotlib

RUN mkdir /diplomova_praca_lib
COPY diplomova_praca_lib /diplomova_praca_lib
RUN pip install /diplomova_praca_lib

RUN mkdir /diplomova_praca
WORKDIR /diplomova_praca

COPY diplomova_praca/requirements.txt /diplomova_praca
RUN pip3 install -r requirements.txt
COPY diplomova_praca /diplomova_praca/

ENV MXNET_CUDNN_AUTOTUNE_DEFAULT 0

EXPOSE 8000

RUN python3 manage.py migrate
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
