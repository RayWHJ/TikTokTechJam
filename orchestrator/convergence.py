from typing import List
from .node import Node

def local_plateau(history: List[float], epsilon: float = 0.002, N: int = 3) -> bool:
    if len(history) < N + 1:
        return False
    best_before = max(history[:-N])
    best_recent = max(history[-N:])
    return (best_recent - best_before) <= epsilon

def global_should_stop(open_nodes: List[Node], remaining_iters: int,
                       global_best: float, margin: float = 0.002,
                       optimistic_gain_per_iter: float = 0.01) -> bool:
    if remaining_iters <= 0 or not open_nodes:
        return True
    best_reachable = max(
        n.local_best_score + optimistic_gain_per_iter * remaining_iters
        for n in open_nodes
    )
    return (best_reachable - global_best) < margin