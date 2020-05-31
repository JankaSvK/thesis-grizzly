from typing import List, Callable, Tuple

import numpy as np


class RankingMechanism:
    def __init__(self):
        pass

    @staticmethod
    def rank_func(ranking: List[Tuple[int, List[float]]],
                  func: Callable[[List[float]], float] = np.min,) -> List[Tuple[int, List[float]]]:
        return sorted(ranking, key=lambda item: func(item[1]))
