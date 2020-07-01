FROM python:3.7
ENV PYTHONUNBUFFERED 1
RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/
RUN pip install -r requirements.txt
COPY . /code/


## Dockerfile
#
## FROM directive instructing base image to build upon
#FROM python:3.7
#
#RUN apt-get update && apt-get install nginx vim -y --no-install-recommends
#COPY nginx.default /etc/nginx/sites-available/default
#RUN ln -sf /dev/stdout /var/log/nginx/access.log \
#    && ln -sf /dev/stderr /var/log/nginx/error.log
#
#RUN mkdir -p /opt/app
#RUN mkdir -p /opt/app/pip_cache
#RUN mkdir -p /opt/app/static
#COPY requirements.txt start-server.sh /opt/app/
#COPY .pip_cache /opt/app/pip_cache/
#COPY martor_demo /opt/app/diplomova_praca/
#WORKDIR /opt/app
#RUN pip install -r requirements.txt --cache-dir /opt/app/pip_cache
#RUN chown -R www-data:www-data /opt/app
#
#EXPOSE 80
#STOPSIGNAL SIGTERM
#CMD ["/opt/app/start-server.sh"]
#
#FROM python:3
#ENV PYTHONUNBUFFERED 1
#RUN mkdir /code
#WORKDIR /code
#COPY requirements.txt /code/
#RUN pip install -r requirements.txt
#COPY . /code/