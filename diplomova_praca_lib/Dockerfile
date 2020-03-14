# Annotating images
FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update
RUN apt-get install -y cmake libsm6 libxext6 libxrender-dev
WORKDIR /project
COPY . .
RUN pip install -e .

CMD /bin/bash

#CMD [ "python", "diplomova_praca_lib/annotate_images.py",\
# "--images_dir=images/", "--save_location=annotated.npy", "=--feature_model=resnet50antepenultimate"]

