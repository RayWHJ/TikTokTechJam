"""Typed evidence store with fingerprint-based dedup."""
import json, os, uuid
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List

Fingerprint = Tuple[str, str, str, str]  # (loss_type, sampler, feature_set, dataset_tier)

#: The fingerprint scheme entries in this store were produced under.
#:
#: Bump this when `driver._fingerprint` changes shape, and stale entries are
#: collected on the next load instead of accumulating forever. That is not
#: hypothetical: the live store held 65 entries of which 61 were prose hashes
#: from a retired scheme — unmatchable by construction, never collected, and
#: linearly scanned on every single lookup.
SCHEME = "family_v2"

#: A retired scheme, named so its entries can be dropped rather than silently
#: kept. `prose_hash_v1` hashed a mechanism's PROSE, so two proposals that would
#: produce substantially the same edit hashed differently and dedup never fired
#: once across 11 candidates.
PROSE_HASH_SCHEME = "prose_hash_v1"

#: run_id marking a hand-authored preseed rather than a measurement this search
#: made. Preseeds block on their own (see `is_blocked`): the corroboration bar
#: exists to stop the search auto-retiring a family off one noisy verdict, and a
#: curated measured fact is not that.
PRESEED_RUN_ID = "preseed"

#: Independent scored refutations required before a fingerprint is a HARD BLOCK.
#:
#: Two, not one. The two bans in the recorded run measured -0.0065 and -0.0050
#: against a paired noise floor of 0.0012, so each is sound evidence about the
#: IMPLEMENTATION it tested. The unsound step is generalising one
#: categorical-field experiment to a whole FAMILY — `sequence_features` matches
#: the bare token "sequence", so it also covers DIN, target attention, history
#: pooling and session features, none of which ever ran. One refutation is now
#: probation, rendered into the prompt as a discount rather than enforced as a
#: filter; two independent ones are a block.
REFUTATIONS_TO_BLOCK = 2


def _infer_scheme(fingerprint) -> str:
    """Which scheme an entry with no `scheme` field was written under.

    Entries predating the field carry no marker, so it is read off the
    fingerprint's own scheme tag. `mechanism_hash` is the retired prose hash;
    everything else (the family branch and the hand-authored structured
    4-tuples) is still matchable today.
    """
    tag = tuple(fingerprint)[0] if fingerprint else ""
    return PROSE_HASH_SCHEME if tag == "mechanism_hash" else SCHEME


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
    #: What happened. Only a REFUTATION blocks — see BLOCKING_EVIDENCE.
    #: "invariant" is what the driver records when a candidate clears the sealed
    #: valid_confirm promotion gate, i.e. it is the record of a SUCCESS, and it
    #: must leave the family proposable so the next iteration can build on it.
    evidence_type: str  # invariant | refuted_under_context | failed_implementation
                        # | timeout | no_op | inconclusive
    note: str = ""
    #: Which run measured this. Makes a ban attributable, and makes "two
    #: INDEPENDENT refutations" checkable rather than a matter of row count.
    #: PRESEED_RUN_ID marks a hand-authored entry.
    run_id: str = ""
    #: The fingerprint scheme this entry's fingerprint was produced under.
    #: Entries whose scheme is not current are dropped on load.
    #:
    #: Defaults to EMPTY, not to SCHEME, so an entry written before this field
    #: existed does not silently claim to be current — `_infer_scheme` reads the
    #: truth off its fingerprint instead. Defaulting to SCHEME would have
    #: relabelled all 61 retired prose hashes in the live store as matchable.
    scheme: str = ""


class Memory:
    def __init__(self, path="orchestrator/_state/memory.json",
                 run_id: str | None = None):
        self.path = path
        #: Stamped onto every entry this process records.
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.entries: List[EvidenceEntry] = []
        self._load()
        # Merge the preseeds on EVERY load, not only into an empty store.
        #
        # `_preseed` used to run `if not self.entries`, so the two LightGBM
        # refutations added after the store was first created exist on no
        # machine that already has a memory.json. Grep the live file for `lgb`
        # and you get zero hits out of 65 entries — which means the single most
        # useful measured fact in the repo (a user x author pair occurs 1.07
        # times in train, so per-pair count features cannot work here) was
        # missing from the thing built to remember it.
        before = len(self.entries)
        self._merge_preseeds()
        if len(self.entries) != before or self._dropped_on_load:
            self.save()

    def _load(self):
        self._dropped_on_load = 0
        if not os.path.exists(self.path):
            return
        with open(self.path) as fh:
            raw = json.load(fh)
        kept = []
        for e in raw:
            entry = EvidenceEntry(**e)
            # A scheme mismatch means this fingerprint can never be matched by
            # anything this process computes, so keeping it costs a linear scan
            # per lookup and buys nothing.
            scheme = entry.scheme or _infer_scheme(entry.fingerprint)
            if scheme != SCHEME:
                self._dropped_on_load += 1
                continue
            entry.scheme = scheme
            kept.append(entry)
        self.entries = kept

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as fh:
            json.dump([asdict(e) for e in self.entries], fh, indent=2)

    def _merge_preseeds(self):
        """Add any preseed whose fingerprint is not already present.

        By fingerprint, so a preseed added to the code later reaches an existing
        store exactly once and never duplicates on repeated loads.
        """
        have = {tuple(e.fingerprint) for e in self.entries}
        for entry in self._preseed_entries():
            if tuple(entry.fingerprint) not in have:
                entry.scheme = entry.scheme or SCHEME
                self.entries.append(entry)
                have.add(tuple(entry.fingerprint))

    def _preseed_entries(self) -> List[EvidenceEntry]:
        """Dead ends already measured — two from the starter-kit README, two
        from this repo's own LightGBM evaluation.

        A pure list rather than a mutation of self.entries, so `_merge_preseeds`
        can diff it against what a store already holds. Every one carries
        run_id=PRESEED_RUN_ID: these are hand-curated measured facts, so each
        blocks on its own without needing the corroboration a search-produced
        verdict does.
        """
        return [EvidenceEntry(
            fingerprint=("pointwise_logloss", "uniform", "cwm_13field", "pure"),
            architecture="FM", loss="pointwise_logloss", sampler="uniform",
            split="valid+test", seed_count=3,
            confidence_interval=(0.593, 0.595),
            code_hash="preseed_cwm13", run_id=PRESEED_RUN_ID,
            evidence_type="refuted_under_context",
            note="CWM 13-field: primary 0.5940 vs 5-field 0.5950 (noise-level, not better)"),
            EvidenceEntry(
            fingerprint=("pointwise_logloss", "uniform", "5field_baseline", "pure_k_sweep"),
            architecture="FM", loss="pointwise_logloss", sampler="uniform",
            split="valid+test", seed_count=1,
            confidence_interval=None, code_hash="preseed_capacity",
            run_id=PRESEED_RUN_ID,
            evidence_type="refuted_under_context",
            note="k=8/16/32 gave 0.5895/0.5902/0.5887; capacity is not the bottleneck"),
            EvidenceEntry(
            fingerprint=("lambdarank", "uniform", "lgb_train_aggregates", "pure"),
            architecture="LightGBM", loss="lambdarank", sampler="uniform",
            split="valid+test", seed_count=1,
            confidence_interval=(0.5755, 0.5800),
            code_hash="preseed_lgb_aggregates", run_id=PRESEED_RUN_ID,
            evidence_type="refuted_under_context",
            note=("GBDT over train-only count aggregates (smoothed video/author/"
                  "user long_view rates, exposure counts, duration, tab, and a "
                  "video x user-activity-decile cross): test primary 0.5755 "
                  "lambdarank / 0.5795 small-capacity / 0.5800 binary, all below "
                  "the FM's 0.5953. Cause: a user x author pair occurs 1.07 times "
                  "in train on average, so per-pair target encoding is one "
                  "observation of noise — FM embeddings share strength across "
                  "users, count features cannot.")),
            EvidenceEntry(
            fingerprint=("binary_logloss", "uniform", "lgb_plus_oof_fm_score", "pure"),
            architecture="LightGBM+FM", loss="binary_logloss", sampler="uniform",
            split="valid+test", seed_count=1,
            confidence_interval=(0.5797, 0.5797),
            code_hash="preseed_lgb_fm_stack", run_id=PRESEED_RUN_ID,
            evidence_type="refuted_under_context",
            note=("Stacking a 3-fold out-of-fold FM score into the GBDT's "
                  "features: test primary 0.5797, still below the FM's 0.5953. "
                  "The largest feature gain by 2x was `user_rate`, which is "
                  "CONSTANT WITHIN A USER and so cannot change a within-user "
                  "ranking — a pointwise GBDT spends its capacity on between-user "
                  "variance the metric ignores."))]

    #: Evidence types that can make a fingerprint a block on re-proposal. ONLY A
    #: REFUTATION. Everything else — inconclusive, failed_implementation,
    #: timeout, no_op, invariant — is a record of what happened, not a verdict
    #: that the mechanism is dead.
    #:
    #: "invariant" used to be in here, and that was a bug reachable only on
    #: success. The driver sets `evidence_type = "invariant"` when a candidate
    #: clears the sealed valid_confirm promotion gate and then writes that value
    #: straight into this store, so the first mechanism that ever WORKED
    #: permanently retired its own family — the exact case the driver's comment
    #: on the promotion path says must stay proposable when the verdict is
    #: `build_on_it`. It never fired in the recorded run only because nothing
    #: ever promoted, which is to say it would have cost the most at the moment
    #: it finally cost anything.
    BLOCKING_EVIDENCE = ("refuted_under_context",)

    def is_duplicate(self, fingerprint: Fingerprint,
                     blocking_only: bool = False) -> Optional[EvidenceEntry]:
        """Find a prior entry with this fingerprint, if any.

        `blocking_only=True` restricts the match to BLOCKING_EVIDENCE, i.e. to
        refutations. The distinction is load-bearing: every scored candidate is
        recorded here, and most are recorded as `inconclusive` because they
        neither promoted nor refuted anything. With an unconditional match, one
        inconclusive result permanently retired a whole mechanism family —
        including a family the verdict step judged `retry_cheaper` (the
        implementation failed, the idea is untested) or `build_on_it` (it worked,
        keep going). Both are cases where you specifically want to propose in
        that family again.

        This is the EVIDENCE question ("have I recorded a refutation at this
        fingerprint"), not the POLICY question ("will the search refuse to
        propose here"). The proposal path asks `is_blocked`, which additionally
        requires the refutation to be corroborated — see that method.

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

    def refutations(self, fingerprint: Fingerprint) -> List[EvidenceEntry]:
        """Blocking-type entries at this fingerprint, deduplicated by
        measurement.

        Keyed on `code_hash`, which the driver sets to the candidate's node id,
        so two records of the SAME measurement count once. Independence is by
        measurement, not by row count — otherwise a store written twice would
        clear the corroboration bar on its own.
        """
        seen, out = set(), []
        for e in self.entries:
            if tuple(e.fingerprint) != tuple(fingerprint):
                continue
            if e.evidence_type not in self.BLOCKING_EVIDENCE:
                continue
            key = (e.code_hash, e.run_id)
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
        return out

    def is_blocked(self, fingerprint: Fingerprint) -> bool:
        """Does the search REFUSE to propose at this fingerprint?

        The POLICY question, deliberately separate from `is_duplicate`'s
        EVIDENCE question. A block needs either

          * a hand-authored preseed (a curated measured fact), or
          * REFUTATIONS_TO_BLOCK independent scored refutations, AT LEAST ONE OF
            WHICH WAS MEASURED BY THIS RUN.

        "Scored" excludes entries with seed_count == 0, which never produced a
        paired delta: those are evidence about the WRITER, not about the
        mechanism, and counting them retires a family that was never tested. In
        the recorded run only 2 of 5 candidates ever produced a paired delta, so
        the distinction decides whether the bar measures anything real.

        WHY THE CURRENT-RUN CLAUSE — this is what makes the ban RUN-SCOPED.
        Requiring two independent refutations was meant to be the bar a ban had to
        clear to survive a process boundary. Measured against the widened proposal
        batch (6-8 per iteration, up to 4 executed), that bar turns out to be
        trivial: a single fresh 8-iteration run writes 28 entries and leaves 7 of
        13 families blocked for the NEXT run, with 5 more on probation. A third
        run would start with almost nothing legal. That is the starvation this
        whole store was rewritten to prevent, one level up — liveness still holds
        because of the frontier floor and the probation node, but the proposal
        space collapses run over run.

        Prior evidence is NOT discarded. It sits as probation
        (`probationary_families`), rendered into the prompt as a discount, and ONE
        corroborating measurement from the current run re-establishes the block.
        So a family that really is dead gets re-blocked after one cheap negative
        result, and a family that was banned off a noisy earlier run gets one more
        fair test instead of being retired forever by a file on disk.

        A preseed still blocks alone: it is a curated measured fact, not one
        run's verdict, and the corroboration bar exists to guard against the
        latter.
        """
        found = self.refutations(fingerprint)
        if any(e.run_id == PRESEED_RUN_ID for e in found):
            return True
        scored = [e for e in found if e.seed_count > 0]
        if len(scored) < REFUTATIONS_TO_BLOCK:
            return False
        return any(e.run_id == self.run_id for e in scored)

    def probationary_families(self) -> List[str]:
        """Mechanism families carrying refuting evidence that does not yet block.

        Rendered into the proposal prompt as a discount rather than enforced as
        a filter, so the proposer can weigh the evidence instead of being
        silently deleted for ignoring it — which is what happened 3 for 3 in
        iteration 4 of the recorded run.
        """
        out = []
        for e in self.entries:
            fp = tuple(e.fingerprint)
            if fp[0] != "mechanism_family" or fp[1] in out:
                continue
            if self.refutations(fp) and not self.is_blocked(fp):
                out.append(fp[1])
        return out

    def record(self, entry: EvidenceEntry):
        # Stamp provenance here rather than at the ~6 construction sites, so an
        # entry can never reach the store anonymous. An explicitly-set run_id
        # (a preseed) is left alone.
        if not entry.run_id:
            entry.run_id = self.run_id
        entry.scheme = entry.scheme or SCHEME
        self.entries.append(entry)
        self.save()

    def by_type(self, evidence_type: str) -> List[EvidenceEntry]:
        return [e for e in self.entries if e.evidence_type == evidence_type]