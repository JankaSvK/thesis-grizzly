from typing import List, Callable, Tuple

import numpy as np


class RankingMechanism:
    def __init__(self):
        pass

    @staticmethod
    def rank_func(ranking: List[Tuple[int, List[float]]],
                  func: Callable[[List[float]], float] = np.min,) -> List[Tuple[int, List[float]]]:
        return sorted(ranking, key=lambda item: func(item[1]))

    @staticmethod
    def mean_with_threshold(ranking: List[Tuple[int, List[float]]], func: Callable[[List[float]], float] = np.mean,
                            threshold=0.7) -> List[Tuple[int, List[float]]]:
        criterium = lambda x: x <= threshold

        return sorted(ranking,
                      key=lambda item: func(RankingMechanism.filter_distances(item[1], criterium=criterium)))

    @staticmethod
    def filter_distances(distances: List[float], criterium: Callable[[float], bool]) -> List[float]:
        filtered = list(filter(criterium, distances))
        if not filtered:
            return distances
        return filtered
