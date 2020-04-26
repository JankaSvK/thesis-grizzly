# STATIC_DIR = settings.STATICFILES_DIRS[0]
import random, os
from pathlib import Path

from diplomova_praca_lib.utils import Memoize

THUMBNAILS_PATH = os.path.join("static", "images", "lookup", "thumbnails")


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
    return Path("/", random.choice(available_images()))


def thumbnail_path(relative_path):
    return str(Path("/") / Path(THUMBNAILS_PATH, relative_path).as_posix())
