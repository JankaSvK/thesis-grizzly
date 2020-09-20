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

Download resnet_mx models
```
mkdir resnet
wget http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152-0000.params resnet
wget http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152-symbol.json resnet
wget https://isis-data.science.uva.nl/mettes/imagenet-shuffle/mxnet/resnext101_bottomup_12988/resnext-101-1-0040.params resnet
wget https://isis-data.science.uva.nl/mettes/imagenet-shuffle/mxnet/resnext101_bottomup_12988/resnext-101-symbol.json resnet
```

The images used in the demo data are from [Open Images
Dataset](https://opensource.google/projects/open-images-dataset). The images
are under CC-by 4.0.

To run the application, Docker is recommended. Then run following commands:
```
$ docker build . -t grizzly
$ docker run -p 8000:8000 -v $PWD/image_representations:/diplomova_praca/static/image_representations -v $PWD/resnet/:/resnet -t grizzly
```

The application then can be accessed at: [127.0.0.1:8000](http://127.0.0.1:8000/)

## Obtaining data for custom dataset

(not necessary for running the demo)

It is possible to use the application over a custom dataset. To do that,
firstly it is needed to obtain the features, which the application uses. It is
possible to generate only some of them, all are not required. After
obtaining the data, the original in `images_representations` can be replaced,
or a new directory with the same structure can be created. In later case,
change the path to the updated directory, when running the app.

Firstly, if the data are videos, please refer to a VIRET tool for extraction
frames from videos. If they are images, with no temporal context, then you need
to create one more subdirectory level.

```
images_representations/images/000/image1.jpg
images_representations/images/000/image2.jpg
```

If they have temporal context, you can split the data into corresponding
directories. We do not rescale images for the frontend, they should be
proprecessed to a resolution 320x180 (possible setting for VIRET tool).

### Building docker for annotations

```
$ cd diplomova_praca_lib
$ docker build . -t grizzly-annotation
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
  --runtime=nvidia \
  grizzly-annotation \
   python3 diplomova_praca_lib/annotate_images.py \
    --images_dir=/images --save_location=/feature_records \
    --feature_model=resnet_mx --num_regions=2,4 \
    --input_size=96 --batch_size=32


$ docker run \
  -v $intermediate_output:/feature_records \
  -v $features:/features \
  grizzly-annotation \
    python3 diplomova_praca_lib/preprocess_data.py \
      --input=/feature_records --output=/features 
      --no_transform --regions --explained_ratio=128

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

### Why grizzly?

Grizzlies spend most of their waking hours searching for food.
