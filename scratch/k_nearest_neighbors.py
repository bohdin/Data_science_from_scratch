from typing import List, NamedTuple, Dict, Tuple
from collections import Counter, defaultdict
import requests 
import csv
import matplotlib.pyplot as plt
import random
import tqdm
try:
    from .linear_algebra import Vector, distance
    from .machine_learning import split_data
    from .statistics import mean
except ImportError:
    from linear_algebra import Vector, distance
    from machine_learning import split_data
    from statistics import mean


def raw_majority_vote(labels: List[str]) -> str:
    votes = Counter(labels)
    winner, _ = votes.most_common(1)[0]
    return winner

def majority_vote(labels: List[str]) -> str:
    vote_counts = Counter(labels)
    winner, winner_counts = vote_counts.most_common(1)[0]
    num_winners = len(
        [count for count in vote_counts.values() if count == winner_counts]
    )

    if num_winners == 1:
        return winner
    else:
        return majority_vote(labels[:-1])
    
class LabeledPoint(NamedTuple):
    point: Vector
    label: str

def knn_classify(k: int, labaled_points: List[LabeledPoint], new_point: Vector) -> str:
    by_distance = sorted(labaled_points, key=lambda lp: distance(lp.point, new_point))

    k_nearest_labels = [lp.label for lp in by_distance[:k]]

    return majority_vote(k_nearest_labels)

if __name__ == "__main__":
    assert raw_majority_vote(['a', 'b', 'c', 'b']) == 'b'

    assert majority_vote(['a', 'b', 'c', 'b', 'a']) == 'b'