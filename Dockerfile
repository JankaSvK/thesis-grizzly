FROM tensorflow/tensorflow:latest-py3

ARG REGIONS_DATA
ARG SPATIAL_DATA
ARG FACE_DATA
ARG FACE_SOM
ARG IMAGES

ENV PYTHONUNBUFFERED 1
RUN pip install --upgrade pip

RUN mkdir /diplomova_praca_lib
COPY diplomova_praca_lib /diplomova_praca_lib
RUN pip install /diplomova_praca_lib

RUN mkdir /diplomova_praca
WORKDIR /diplomova_praca

COPY diplomova_praca/requirements.txt /diplomova_praca
RUN pip install -r requirements.txt
COPY diplomova_praca /diplomova_praca/

COPY $IMAGES/ static/images/lookup/thumbnails
COPY $REGIONS_DATA/ static/image_representations/regions
COPY $SPATIAL_DATA/ static/image_representations/spatial

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
