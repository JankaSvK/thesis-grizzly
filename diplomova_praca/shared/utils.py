# STATIC_DIR = settings.STATICFILES_DIRS[0]
import os
import random
from pathlib import Path

from diplomova_praca_lib.utils import Memoize

THUMBNAILS_PATH = os.path.join("static", "image_representations", "images")
FEATURES_PATH = os.path.join("static", "image_representations")

@Memoize
def available_images():
    """
    Traverse and searches for all images under `THUMBNAILS_PATH`
    :return: List of images as relative path to `THUMBNAILS_PATH`
    """
    return [path for path in Path(THUMBNAILS_PATH).rglob('*.jpg')]

@Memoize
def directories():
    return [x for x in Path(THUMBNAILS_PATH).iterdir() if x.is_dir()]

@Memoize
def dir_files(dir:Path):
    return [path for path in dir.rglob('*.jpg')]

def random_image_path():
    if not available_images():
        return None
    return Path("/", random.choice(available_images()))

def random_subset_image_path(images_allowed):
    chosen = random.choice(list(images_allowed))
    return Path("/", THUMBNAILS_PATH, chosen)


def thumbnail_path(relative_path):
    return str(Path("/") / Path(THUMBNAILS_PATH, relative_path).as_posix())
