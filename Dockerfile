FROM tensorflow/tensorflow:latest-py3

ENV PYTHONUNBUFFERED 1
RUN apt update -y
RUN pip install --upgrade pip

RUN apt update && apt install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev

RUN mkdir /diplomova_praca_lib
COPY diplomova_praca_lib /diplomova_praca_lib
RUN pip install /diplomova_praca_lib

RUN mkdir /diplomova_praca
WORKDIR /diplomova_praca

COPY diplomova_praca/requirements.txt /diplomova_praca
RUN pip install -r requirements.txt
COPY diplomova_praca /diplomova_praca/

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
