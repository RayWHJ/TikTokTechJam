"""
Thin, injectable LLM client for codegen/.

Person D's package generates code (writer), repairs it (debug), and writes the
final report (report). Those three call a *code/writing* model — separate from
Person C's reasoning calls. This module is that model client, built so the whole
package is testable **offline, with no API key**:

  * OpenAIBackend — used only when the `OpenAI` SDK is importable AND
    OpenAI_API_KEY is set AND CODEGEN_LLM_BACKEND != "fake".
  * FakeBackend — deterministic, no network. Produces plausible, gate-clean
    output keyed on the `kind` of request. This is the default, so
    `writer.write_fix(...)` and `report.synthesize_report(...)` run end-to-end
    today without any teammate module or credential.

Every public codegen function accepts an optional `client=` so the orchestrator
can inject the real backend later — swapping backends is a one-liner, no code in
writer/debug/report changes.

Env vars:
  CODEGEN_LLM_BACKEND  = "fake" | "openai"   (default: auto-detect)
  CODEGEN_LLM_MODEL    = model id string for the real backend
  OPENAI_API_KEY    = credential for the real backend
"""
from __future__ import annotations
import os, json, textwrap, hashlib

# request "kinds" — the fake backend switches on these; the real backend ignores.
KIND_DIFF = "diff"        # writer: produce a code diff
KIND_DEBUG = "debug"      # debug: repair a crashing/again diff
KIND_SANITY = "sanity"    # debug: judge an implausibly-good result (JSON verdict)
KIND_REPORT = "report"    # report: write the Devpost markdown


class LLMError(RuntimeError):
    """Raised when the real backend is selected but cannot be used/called."""


# --------------------------------------------------------------------------- #
#  Backends                                                                    #
# --------------------------------------------------------------------------- #
class OpenAIBackend:
    """Real backend. Kept deliberately small; only used when explicitly available."""

    def __init__(self, model: str | None = None):
        try:
            import openai  # noqa: F401
        except Exception as e:                       # pragma: no cover - env dependent
            raise LLMError(f"OpenAI SDK not importable: {e}")
        if not os.environ.get("OPENAI_API_KEY"):
            raise LLMError("OPENAI_API_KEY not set")
        import openai
        self._client = openai.OpenAI()
        # Model id is intentionally env-driven so this package pins nothing stale.
        self.model = model or os.environ.get("CODEGEN_LLM_MODEL", "gpt-4o-mini")

    def complete(self, system: str, user: str, kind: str,
                 max_tokens: int = 4000, temperature: float = 0.0) -> str:  # pragma: no cover
        # Temperature intentionally not forwarded — reasoning models (gpt-5,
        # o-series) reject the parameter; the codegen paths call with 0.0
        # anyway, and this keeps the client portable across model families.
        resp = self._client.responses.create(
            model=self.model,
            instructions=system,
            input=user,
            max_output_tokens=max_tokens,
        )
        return resp.output_text


class FakeBackend:
    """Deterministic offline backend. No network, no key. Produces output shaped
    like what the real model would return, so gate/execute/report can be exercised
    end-to-end in tests and demos."""

    def complete(self, system: str, user: str, kind: str,
                 max_tokens: int = 4000, temperature: float = 0.0) -> str:
        if kind == KIND_DIFF:
            return self._fake_diff(system, user)
        if kind == KIND_DEBUG:
            return self._fake_debug(user)
        if kind == KIND_SANITY:
            return self._fake_sanity(user)
        if kind == KIND_REPORT:
            return self._fake_report(user)
        return "# (fake backend: no output for kind=%r)" % kind

    # -- fakes ------------------------------------------------------------- #
    @staticmethod
    def _targets_data(system: str, user: str) -> bool:
        # key off the explicit "File to edit: <name>" line writer.build_writer_user
        # emits — robust, since both system prompts mention the word "feature".
        return "file to edit: data.py" in user.lower()

    def _fake_diff(self, system: str, user: str) -> str:
        """Return a small, contract-CLEAN unified diff so the gate passes.
        Two shapes: a within-user BPR loss patch on baseline.py (model/loss), or
        an extra causal categorical field on data.py (features)."""
        if self._targets_data(system, user):
            return textwrap.dedent('''\
                ```diff
                --- a/data.py
                +++ b/data.py
                @@
                -FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
                +# added a point-in-time-safe temporal field (hour bucket); no
                +# non-causal statistic columns are used as inputs.
                +FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket', 'hour_bucket']
                @@ def raw(x):
                -        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5])))]
                +        return [x[1], x[2], x[3], x[4],
                +                str(int(np.searchsorted(edges, x[5]))),
                +                str(int(x[0]) % 24)]  # hour_bucket, derived from the row's own timestamp
                ```
                ''')
        return textwrap.dedent('''\
            ```diff
            --- a/baseline.py
            +++ b/baseline.py
            @@ class FM
            +    def bpr_step(self, Xpos, Xneg):
            +        """Pairwise BPR update: push score(pos) > score(neg) within a user."""
            +        zp, Ep, Sp = self.logits(Xpos)
            +        zn, En, Sn = self.logits(Xneg)
            +        d = (1.0 / (1.0 + np.exp(np.clip(zp - zn, -30, 30)))).astype(np.float32)
            +        Xb = np.concatenate([Xpos, Xneg], 0)
            +        g = np.concatenate([-d, d]).astype(np.float32) / len(d)
            +        self._apply_logit_grad(Xb, g)
            @@ def run_fm
            -    # pointwise logloss training loop (unchanged)
            +    # NOTE: loss switched from pointwise logloss to within-user BPR;
            +    # model selection still on valid_search only, never test.
            ```
            ''')

    def _fake_debug(self, user: str) -> str:
        return textwrap.dedent('''\
            ```diff
            --- a/baseline.py
            +++ b/baseline.py
            @@
            -        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum(1))
            +        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))  # fix axis bug
            ```
            ''')

    def _fake_sanity(self, user: str) -> str:
        # If the prompt says the score cleared the oracle ceiling or leapt a lot,
        # flag a likely leak; otherwise say it looks real. Deterministic on text.
        leak = any(t in user.lower() for t in
                   ("above oracle", "0.86", "0.9", "0.95", "1.0", "implausible", "leak"))
        return json.dumps({
            "implements_hypothesis": (not leak),
            "leak_suspected": leak,
            "reasoning": ("Score exceeds the oracle ceiling / jumps implausibly; "
                          "likely a label leak or evaluation bug, not the stated mechanism."
                          if leak else
                          "Change matches the stated mechanism and the gain is within a "
                          "plausible range for this edit."),
        })

    def _fake_report(self, user: str) -> str:
        # Pull a few facts out of the embedded run_log JSON if present.
        facts = {}
        try:
            facts = json.loads(user[user.index("{"):user.rindex("}") + 1])
        except Exception:
            pass
        best = facts.get("global_best", {})
        return textwrap.dedent(f'''\
            # Autonomous ML Research Agent — KuaiRand-Pure (Track 2)

            ## Inspiration
            The FM baseline scores **{facts.get("baseline_primary", 0.5946)} primary** on
            test, but trains a *pointwise* objective while the metric is a *ranking*
            metric. We built an agent that diagnoses that mismatch and fixes it under
            strict anti-leakage controls.

            ## What it does
            An ablation-driven loop (diagnose -> ground in literature -> hypothesize ->
            generate code -> gate -> execute -> confirm) with a sealed confirmation
            split and a static pre-execution safety gate.

            ## How we built it
            Four packages: harness (data/eval), llm_calls (reasoning), codegen
            (this — code generation, static gate, sandboxed execution, self-debug),
            orchestrator (search + promotion).

            ## Results
            Best promoted candidate: **{best.get("primary", "n/a")}** on the sealed
            confirmation split ({best.get("mechanism", "ranking-loss refinement")}),
            versus the {facts.get("baseline_primary", 0.5946)} baseline. Full runs:
            {facts.get("counters", {}).get("full_runs", "n/a")}; scorer queries kept
            per-split and honest.

            ## Challenges / What we learned
            Preventing test-label leakage was the central engineering problem; the
            static gate plus physically removing test-named files from the sandbox
            working directory caught it before compute was spent.
            ''')


# --------------------------------------------------------------------------- #
#  Facade                                                                      #
# --------------------------------------------------------------------------- #
class LLMClient:
    """Facade the codegen functions call. Chooses a backend once and forwards."""

    def __init__(self, backend=None):
        self.backend = backend or self._auto_backend()

    @staticmethod
    def _auto_backend():
        choice = os.environ.get("CODEGEN_LLM_BACKEND", "").lower()
        if choice == "fake":
            return FakeBackend()
        if choice == "openai":
            return OpenAIBackend()
        # auto: prefer real only if clearly available, else fake.
        try:
            import openai  # noqa: F401
            if os.environ.get("OPENAI_API_KEY"):
                return OpenAIBackend()
        except Exception:
            pass
        return FakeBackend()

    def complete(self, system: str, user: str, kind: str, **kw) -> str:
        return self.backend.complete(system, user, kind, **kw)

    @property
    def is_fake(self) -> bool:
        return isinstance(self.backend, FakeBackend)


_DEFAULT: LLMClient | None = None

def get_default_client() -> LLMClient:
    """Process-wide default client (lazily built). Pass an explicit client= to any
    codegen function to override for a single call."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = LLMClient()
    return _DEFAULT
