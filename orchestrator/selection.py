"""Pick which open Node to expand next."""
import math
from typing import List
from .node import Node

def score_node(node: Node, total_visits: int,
               exploration_c: float = 0.7, cost_c: float = 0.05) -> float:
    if node.n_visits == 0:
        return float("inf")  # untried first
    ev = node.local_best_score if node.local_best_score > float("-inf") else 0.0
    exploration = exploration_c * math.sqrt(math.log(max(total_visits, 1)) / node.n_visits)
    cost_penalty = cost_c * (node.wallclock_used_s / 60.0)
    return ev + exploration - cost_penalty

def select(open_nodes: List[Node]) -> Node:
    if not open_nodes:
        raise RuntimeError("no open nodes to select")
    total = sum(n.n_visits for n in open_nodes)
    return max(open_nodes, key=lambda n: score_node(n, total))