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
        """Two dead ends the starter-kit README already measured."""
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

    def is_duplicate(self, fingerprint: Fingerprint) -> Optional[EvidenceEntry]:
        for e in self.entries:
            if tuple(e.fingerprint) == tuple(fingerprint):
                return e
        return None

    def record(self, entry: EvidenceEntry):
        self.entries.append(entry)
        self.save()

    def by_type(self, evidence_type: str) -> List[EvidenceEntry]:
        return [e for e in self.entries if e.evidence_type == evidence_type]