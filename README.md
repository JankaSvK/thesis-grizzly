# thesis-grizzly

Grizzly is a web-based application to search over a dataset of images based on two techniques:
- search by collage
- search by faces

To run the application, Docker is recommended. Then run following commands:
```
docker build . -t app
docker run -p 8000:8000 -v $PWD/image_representations:/diplomova_praca/static/image_representations -t app
```

To correctly run the application a set of data is required. By default, the
provided demo set is used. It is possible to use application with own data,
although, it requires running few processing pipelines.
