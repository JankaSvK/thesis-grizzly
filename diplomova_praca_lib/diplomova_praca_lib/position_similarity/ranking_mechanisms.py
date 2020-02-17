from collections import defaultdict
from typing import List


def defaultdic(param):
    pass


class RankingMechanism:
    def __init__(self):
        pass

    @staticmethod
    def summing(rankings: List[str]):
        elements = defaultdict(int)
        for ranking in rankings:
            for rank, item in enumerate(ranking):
                elements[item] += rank

        sorted_ranking = sorted(elements.items(), key=lambda kv: (kv[1], kv[0]))
        return [key for key, value in sorted_ranking]
