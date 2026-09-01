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
    # Iterations where the dedup filter emptied the candidate list and the
    # proposer had to be re-asked with the blocked families enumerated. In the
    # recorded run this happened at least once (iteration 4, 3 hypotheses all
    # dropped) and left NO trace anywhere: no node, no ledger entry, no log
    # line. The only surviving evidence was the arithmetic gap between
    # counters.proposals (8) and sum(n_candidates) (5). Section 2.5 of the
    # problem statement requires per-iteration error and recovery events, so an
    # unlogged widening event is a missing deliverable, not just a blind spot.
    dedup_starved: int = 0
    scorer_queries: dict = field(default_factory=lambda:
        {"train": 0, "valid_search": 0, "valid_confirm": 0, "test": 0})
    # Candidates checked by the 200-row smoke stage, and how many it rejected
    # outright. The pair is the saving, stated: each reject is a 240s triage run
    # plus up to two 240s repair runs that were never spent.
    smoke_runs: int = 0
    smoke_rejects: int = 0
    # Label-permutation control runs (T3.4). One per promotion CANDIDATE, not per
    # candidate: billed separately from full_runs because it measures whether a
    # score is real rather than what the score is.
    permutation_runs: int = 0

    wallclock_s: float = 0.0

    # REAL token accounting, synced from llm_calls.usage.LEDGER — see that module.
    # `tokens` used to be hand-incremented constants (500 a diagnose, 800 a
    # writer call, ...) and every persisted progress.json reports 13200, a number
    # no API ever returned. It is now the honest total, kept under its original
    # name so existing readers of progress.json still resolve.
    tokens: int = 0
    tokens_in: int = 0
    tokens_cached: int = 0
    tokens_out: int = 0
    tokens_reasoning: int = 0
    llm_calls_made: int = 0
    web_searches: int = 0
    # Non-zero means some response carried no usage block, so the totals above
    # are an UNDERCOUNT. Reported rather than hidden: a cost figure that cannot
    # say how complete it is cannot be checked.
    calls_without_usage: int = 0
    estimated_cost_usd: float = 0.0
    # Per-operator breakdown. The fact that decides where to spend: the writer
    # reproduces a whole file at the output rate, the auditor burns ~4,500 input
    # tokens per candidate for a signal that fired on 5 of 5.
    tokens_by_kind: dict = field(default_factory=dict)

    def bump(self, name, amount=1):
        setattr(self, name, getattr(self, name) + amount)

    def sync_usage(self, ledger) -> None:
        """Copy real usage off llm_calls.usage.LEDGER onto these counters.

        Called before every progress.json write rather than once at the end, so
        an overnight run can be inspected mid-flight and so a crash still leaves
        real numbers behind. Pulling rather than pushing keeps the clients free of
        any dependency on the orchestrator.
        """
        t = ledger.totals()
        self.tokens_in = t["tokens_in"]
        self.tokens_cached = t["tokens_cached"]
        self.tokens_out = t["tokens_out"]
        self.tokens_reasoning = t["tokens_reasoning"]
        self.tokens = t["tokens_total"]
        self.llm_calls_made = t["calls"]
        self.web_searches = t["web_searches"]
        self.calls_without_usage = t["calls_without_usage"]
        self.estimated_cost_usd = t["estimated_cost_usd"]
        self.tokens_by_kind = t["by_kind"]

    def bump_scorer(self, split, amount=1):
        self.scorer_queries[split] = self.scorer_queries.get(split, 0) + amount