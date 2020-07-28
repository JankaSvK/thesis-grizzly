# thesis-grizzly

Grizzly is a web-based application to search over a dataset of images based on two techniques:
- search by collage
- search by faces

To pull the repository:
```
git clone --depth 1 git@github.com:JankaSvK/thesis-grizzly.git
```

Firstly, download the [demo
data](https://drive.google.com/file/d/1IgRn9_My1dwHno2JGEwXiim7YxuWPnd1/view?usp=sharing)
to play with. Replace current `image_representations` with the directory
obtained from the zip.

The images used in the demo data are from [Open Images
Dataset](https://opensource.google/projects/open-images-dataset). The images
are under CC-by 4.0.

To run the application, Docker is recommended. Then run following commands:
```
$ docker build . -t app
$ docker run -p 8000:8000 -v $PWD/image_representations:/diplomova_praca/static/image_representations -t app
```

The application then can be accessed at: [127.0.0.1:8000](127.0.0.1:8000)


To correctly run the application a set of data is required. By default, the
provided demo set is used. It is possible to use application with own data,
although, it requires running few processing pipelines.

Firstly, if the data is videos, please refer to a VIRET tool for extraction
frames from videos. If they are images, with no temporal context, then you need
to create one more subdirectory level.

```
images_representations/images/000/image1.jpg
images_representations/images/000/image2.jpg
```

If they have temporal context, you can split the data into corresponding
directories. We do not rescale images for the frontend, they should be
proprecessed to a resolution 320x180 (possible setting for VIRET tool).

## Obtaining data

It is possible to use the application over a custom dataset. To do that,
firstly it is needed to obtain the features, which the application uses. It is
possible to generate only some of them, all are not required. After the
obtaining the data, the original in `images_representations` can be replaced,
or a new directory with the same structure can be created. In later case,
change the path to the updated directory, when running the app.

### Building docker for annotations

```
$ cd diplomova_praca_lib
$ docker build . -t lib
```

### Regions Annotation for MobileNetV2
2x4 regions, PCA=128 components, input\_size=96

```
$ images="/path/to/images/"
$ intermediate_output="/path/to/intermediate_output/"
$ features="/path/to/features/"

$ docker run \
  -v $images:/images \
  -v $intermediate_output:/feature_records \
  lib \
   python diplomova_praca_lib/annotate_images.py \
    --images_dir=/images --save_location=/feature_records \
    --feature_model=mobilenetv2 --num_regions=2,4 \
    --input_size=96

$ docker run \
  -v $intermediate_output:/feature_records \
  -v $features:/features \
  lib \
    python diplomova_praca_lib/preprocess_data.py \
      --input=/feature_records --output=/features 
      --transform --regions --explained_ratio=128

```

### Faces Extractions

0.08 is the criteria on the size of face to be accepted.

```
$ images="/path/to/images/"
$ intermediate\_output="/path/to/intermediate_output/"
$ features="/path/to/features/"

$ docker run \
  -v $images:/images \
  -v $intermediate_output:/feature_records \
  lib \
   python diplomova_praca_lib/annotate_images.py \
    --images_dir=/images --save_location=/feature_records \
    --feature_model=faces

$ docker run \
  -v $intermediate\_output:/feature_records \
  -v $features:/features \
  lib \
    python diplomova_praca_lib/preprocess_face_data.py \
      --input=/feature_records --output=/features \
      --crop_size=0.08
```

### Training SOM

```
$ face_features="/path/to/face_features/"
$ trained_som="/path/to/trained_som/"

$ docker run \
  -v $face_features:/features \
  -v $trained_som:/output \
  lib \
   python diplomova_praca_lib/train_som.py \
    --input=/features \
    --iterations=200000
```

