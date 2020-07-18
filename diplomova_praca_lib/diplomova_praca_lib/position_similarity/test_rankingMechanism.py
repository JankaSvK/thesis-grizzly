from unittest import TestCase

from diplomova_praca_lib.position_similarity.ranking_mechanisms import RankingMechanism


class TestRankingMechanism(TestCase):
    def test_mean_with_threshold(self):
        res = RankingMechanism.mean_with_threshold([(2, [0.4, 0.7]), (1, [0, 1, 0.7])])
        self.assertEqual([(1, [0, 1, 0.7]), (2, [0.4, 0.7])], res)
