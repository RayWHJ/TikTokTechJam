"""Multi-fidelity ranking of partial-run candidates."""
from typing import List, Tuple
from .node import Node

def posterior_estimate(node: Node) -> Tuple[float, float]:
    """Return (mean, std) of the posterior over the node's final score.
    Simple placeholder; swap in a learning-curve model when you have curves."""
    if not node.partial_scores:
        return (0.5, 0.2)  # broad prior for unrun nodes
    mean = sum(node.partial_scores) / len(node.partial_scores)
    n = len(node.partial_scores)
    std = 0.02 / max(n, 1) ** 0.5
    return (mean, std)

def rank(candidates: List[Node], keep: int = 3, wildcard: bool = True) -> List[Node]:
    """Top by posterior mean, plus one wildcard by highest posterior std."""
    if not candidates:
        return []
    scored = [(n, *posterior_estimate(n)) for n in candidates]
    scored.sort(key=lambda t: t[1], reverse=True)

    if wildcard and len(scored) > max(keep - 1, 1):
        top = [t[0] for t in scored[:keep - 1]]
        rest = scored[keep - 1:]
        wild = max(rest, key=lambda t: t[2])
        top.append(wild[0])
        return top
    return [t[0] for t in scored[:keep]]