from pathlib import Path


def cap_value(value, minimum, maximum):
    return max(minimum, min(value, maximum))

def filename_without_extensions(path):
    return Path(path).stem

def batches(iterator, batch_size):
    if batch_size < 1:
        raise ValueError("`batch_size` must be greater than or equal to 1.")

    batch = []
    for el in iterator:
        batch.append(el)

        if len(batch) >= batch_size:
            yield batch
            batch = []

    yield batch


class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        if not args in self.memo:
            self.memo[args] = self.f(*args)
        # Warning: You may wish to do a deepcopy here if returning objects
        return self.memo[args]
