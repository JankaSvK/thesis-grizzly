# Annotating images
FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update

WORKDIR /project
COPY code code
WORKDIR code/diplomova_praca_lib
RUN pip install -e .

CMD /bin/bash

#CMD [ "python", "diplomova_praca_lib/annotate_images.py",\
# "--images_dir=images/", "--save_location=annotated.npy", "=--feature_model=resnet50antepenultimate"]

