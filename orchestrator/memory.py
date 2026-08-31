"""Typed evidence store with fingerprint-based dedup."""
import json, os
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, List

Fingerprint = Tuple[str, str, str, str]  # (loss_type, sampler, feature_set, dataset_tier)

@dataclass
class EvidenceEntry:
    fingerprint: Fingerprint
    architecture: str
    loss: str
    sampler: str
    split: str
    seed_count: int
    confidence_interval: Optional[Tuple[float, float]]
    code_hash: str
    evidence_type: str  # invariant | refuted_under_context | failed_implementation | inconclusive
    note: str = ""


class Memory:
    def __init__(self, path="orchestrator/_state/memory.json"):
        self.path = path
        self.entries: List[EvidenceEntry] = []
        self._load()
        if not self.entries:
            self._preseed()
            self.save()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as fh:
                self.entries = [EvidenceEntry(**e) for e in json.load(fh)]

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as fh:
            json.dump([asdict(e) for e in self.entries], fh, indent=2)

    def _preseed(self):
        """Dead ends already measured — two from the starter-kit README, two
        from this repo's own LightGBM evaluation."""
        self.entries.append(EvidenceEntry(
            fingerprint=("pointwise_logloss", "uniform", "cwm_13field", "pure"),
            architecture="FM", loss="pointwise_logloss", sampler="uniform",
            split="valid+test", seed_count=3,
            confidence_interval=(0.593, 0.595),
            code_hash="preseed_cwm13",
            evidence_type="refuted_under_context",
            note="CWM 13-field: primary 0.5940 vs 5-field 0.5950 (noise-level, not better)"))
        self.entries.append(EvidenceEntry(
            fingerprint=("pointwise_logloss", "uniform", "5field_baseline", "pure_k_sweep"),
            architecture="FM", loss="pointwise_logloss", sampler="uniform",
            split="valid+test", seed_count=1,
            confidence_interval=None, code_hash="preseed_capacity",
            evidence_type="refuted_under_context",
            note="k=8/16/32 gave 0.5895/0.5902/0.5887; capacity is not the bottleneck"))
        self.entries.append(EvidenceEntry(
            fingerprint=("lambdarank", "uniform", "lgb_train_aggregates", "pure"),
            architecture="LightGBM", loss="lambdarank", sampler="uniform",
            split="valid+test", seed_count=1,
            confidence_interval=(0.5755, 0.5800),
            code_hash="preseed_lgb_aggregates",
            evidence_type="refuted_under_context",
            note=("GBDT over train-only count aggregates (smoothed video/author/"
                  "user long_view rates, exposure counts, duration, tab, and a "
                  "video x user-activity-decile cross): test primary 0.5755 "
                  "lambdarank / 0.5795 small-capacity / 0.5800 binary, all below "
                  "the FM's 0.5953. Cause: a user x author pair occurs 1.07 times "
                  "in train on average, so per-pair target encoding is one "
                  "observation of noise — FM embeddings share strength across "
                  "users, count features cannot.")))
        self.entries.append(EvidenceEntry(
            fingerprint=("binary_logloss", "uniform", "lgb_plus_oof_fm_score", "pure"),
            architecture="LightGBM+FM", loss="binary_logloss", sampler="uniform",
            split="valid+test", seed_count=1,
            confidence_interval=(0.5797, 0.5797),
            code_hash="preseed_lgb_fm_stack",
            evidence_type="refuted_under_context",
            note=("Stacking a 3-fold out-of-fold FM score into the GBDT's "
                  "features: test primary 0.5797, still below the FM's 0.5953. "
                  "The largest feature gain by 2x was `user_rate`, which is "
                  "CONSTANT WITHIN A USER and so cannot change a within-user "
                  "ranking — a pointwise GBDT spends its capacity on between-user "
                  "variance the metric ignores.")))

    #: Evidence types that make a fingerprint a HARD BLOCK on re-proposal.
    #: Everything else — inconclusive, failed_implementation, timeout, no_op — is
    #: a record of what happened, not a verdict that the mechanism is dead.
    BLOCKING_EVIDENCE = ("refuted_under_context", "invariant")

    def is_duplicate(self, fingerprint: Fingerprint,
                     blocking_only: bool = False) -> Optional[EvidenceEntry]:
        """Find a prior entry with this fingerprint, if any.

        `blocking_only=True` restricts the match to BLOCKING_EVIDENCE. The
        driver's proposal path uses it, and the distinction is load-bearing:
        every scored candidate is recorded here, and most are recorded as
        `inconclusive` because they neither promoted nor refuted anything. With
        an unconditional match, one inconclusive result permanently retired a
        whole mechanism family — including a family the verdict step judged
        `retry_cheaper` (the implementation failed, the idea is untested) or
        `build_on_it` (it worked, keep going). Both are cases where you
        specifically want to propose in that family again.

        Default stays False so the plain "have I seen this fingerprint at all"
        question still has an answer.
        """
        for e in self.entries:
            if tuple(e.fingerprint) != tuple(fingerprint):
                continue
            if blocking_only and e.evidence_type not in self.BLOCKING_EVIDENCE:
                continue
            return e
        return None

    def record(self, entry: EvidenceEntry):
        self.entries.append(entry)
        self.save()

    def by_type(self, evidence_type: str) -> List[EvidenceEntry]:
        return [e for e in self.entries if e.evidence_type == evidence_type]