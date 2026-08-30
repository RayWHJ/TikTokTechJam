from dataclasses import dataclass, field

@dataclass
class Counters:
    proposals: int = 0
    triage_runs: int = 0
    full_runs: int = 0
    semantic_retries: int = 0
    # Candidates that ran clean but reproduced the parent's per-user scores
    # exactly, so the writer was asked again with that fed back. A high count
    # means the writer is annotating files instead of editing the training path.
    no_op_rewrites: int = 0
    # Controlled-ablation runs (MLE-STAR). Billed separately from full_runs
    # because an ablation measures what the pipeline leans on, not whether a
    # candidate is better — but it spends a real valid_search query, so a
    # budget report that can't see it understates what the search cost.
    ablation_runs: int = 0
    scorer_queries: dict = field(default_factory=lambda:
        {"train": 0, "valid_search": 0, "valid_confirm": 0, "test": 0})
    wallclock_s: float = 0.0
    tokens: int = 0

    def bump(self, name, amount=1):
        setattr(self, name, getattr(self, name) + amount)

    def bump_scorer(self, split, amount=1):
        self.scorer_queries[split] = self.scorer_queries.get(split, 0) + amount