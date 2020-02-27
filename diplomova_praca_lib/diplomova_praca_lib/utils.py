from pathlib import Path


def cap_value(value, minimum, maximum):
    return max(minimum, min(value, maximum))

def filename_without_extensions(path):
    return Path(path).stem
