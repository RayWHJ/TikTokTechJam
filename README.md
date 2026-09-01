# Autonomous ML Research Agent for Recommender Systems

## Project Overview

This project is an autonomous machine learning research agent for recommender systems. It automates the full experiment loop:

1. Inspect the current pipeline and results.
2. Propose a focused improvement.
3. Generate a new implementation.
4. Run the candidate in a controlled environment.
5. Evaluate GAUC and nDCG@5.
6. Reflect on the result and select the next action.
7. Keep the best valid checkpoint.

The required benchmark is KuaiRand-Pure. The task predicts the `long_view` label and ranks videos within each user's logged impressions.

The active search strategy follows an AIDE-style code refinement loop. It is supported by additional controls for experiment selection, measurement, failure recovery, convergence, and logging.

## System Components

| Component | Responsibility |
|---|---|
| Harness | Runs candidates, applies limits, validates outputs, collects metrics, and records failures. |
| Codegen | Converts experiment hypotheses into focused code changes. |
| LLM calls | Manages model requests, structured responses, retries, and token accounting. |
| Orchestrator | Controls the research loop, candidate selection, recovery, convergence, and final checkpoint selection. |

These components are separated so that model reasoning, code generation, execution, and evaluation can be tested independently.

## Evaluation Target

| Item | Configuration |
|---|---|
| Required dataset | KuaiRand-Pure |
| Relevance label | `long_view` |
| Ranking scope | Each user's logged impressions |
| Metrics | GAUC and nDCG@5 |
| Primary score | Mean of GAUC and nDCG@5 |
| Baseline | Organizer-provided factorization machine |
| Convergence | No improvement greater than 0.002 for three consecutive iterations |
| Maximum run | 50 iterations or six hours |

KuaiRand-1k and KuaiRand-27k are optional bonus benchmarks.

## Setup and Installation

### Requirements

- Python 3.10 or later
- Git
- KuaiRand-Pure data
- Access to the configured LLM provider
- A valid API key for the selected model

A GPU is not required for the provided baseline.

### Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### Create a Virtual Environment

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

### Configure the Dataset

Download the two KuaiRand-Pure interaction files and keep their original names:

```text
log_standard_4_08_to_4_21_pure.csv
log_standard_4_22_to_5_08_pure.csv
```

Place them in the dataset location expected by `data.py`.

The fixed date split is:

- Training: 8 April 2022 to 21 April 2022
- Validation: 22 April 2022 to 28 April 2022
- Evaluation: 29 April 2022 to 8 May 2022

Do not change the split or use the evaluation data during development.

### Configure the LLM

Export the API key and model configuration required by the LLM-call module before starting a run.

Do not commit API keys, tokens, `.env` files, generated credentials, or private model responses.

## Verify the Installation

Run the automated tests:

```bash
pytest -q
```

Reproduce the official baseline:

```bash
python baseline.py --model fm
```

The baseline run should complete successfully and produce GAUC, nDCG@5, and the combined primary score.

## Steps to Reproduce the Result

### 1. Validate the Baseline

```bash
python baseline.py --model fm
```

Confirm that the baseline result is consistent with `baseline_scores.json`.

### 2. Run the Test Suite

```bash
pytest -q
```

All harness, evaluation, code generation, LLM-call, and orchestration tests should pass before an autonomous run begins.

### 3. Start the Autonomous Research Agent

```bash
python -m orchestrator.driver
```

The orchestrator will:

1. Load the baseline and current candidate.
2. Inspect the experiment history.
3. Choose a focused hypothesis.
4. Request a code change.
5. Validate and execute the candidate.
6. Calculate GAUC, nDCG@5, and the primary score.
7. Record the code diff and outcome.
8. Recover from invalid or failed implementations.
9. Select the next parent candidate.
10. Stop when the convergence or budget rule is reached.

### 4. Review the Run Logs

Each iteration should contain:

- Iteration number
- Parent candidate
- Experiment hypothesis
- Reason for the experiment
- Generated code diff
- GAUC
- nDCG@5
- Primary score
- Runtime
- Token usage
- Candidate status
- Error and recovery events
- Manual interventions

Use the final run summary as the authoritative record of the reproduced result.

### 5. Validate the Final Submission

Generate or select the final prediction file, then validate it:

```bash
python submit.py --check <submission-file.csv>
```

The validator checks:

- Header order
- Row count
- Continuous `row_id` values
- Evaluation-row alignment
- Numeric scores
- Missing or invalid values

Only a validated file should be submitted.

### 6. Confirm the Final Checkpoint

The final checkpoint must be the validation-best candidate available when the run converges. It must not be selected using hidden evaluation results.

Record the following values in the final results summary:

| Metric | Result |
|---|---:|
| GAUC | Generated by the final run |
| nDCG@5 | Generated by the final run |
| Primary score | Mean of GAUC and nDCG@5 |
| Delta over baseline | Final score minus official baseline |
| Iterations | Generated by the final run |
| Manual interventions | Generated by the final run |
| Agent wall-clock | Generated by the final run |
| LLM tokens | Generated by the final run |

## Experiment Integrity

The project applies the following rules:

- Training and model selection use only the training and validation splits.
- The evaluation split is never exposed to the research agent.
- Every candidate uses the same evaluation implementation.
- Invalid candidates cannot replace a valid parent.
- Failed experiments remain visible in the run history.
- Metric comparisons use the same data, seeds, and scoring rules.
- The final checkpoint is selected by validation performance.
- Human interventions are counted and reported.

## Solution Limitations

### Limited Search Budget

The current submission explores only a small part of the possible feature, model, and training space. A better method may remain undiscovered when the run converges early.

### Dependence on LLM Quality

Experiment quality depends on the selected model and prompt. The LLM may propose weak hypotheses, repeat earlier ideas, or generate invalid code.

### Non-Deterministic Generation

The same run may produce different hypotheses and implementations because model responses are not fully deterministic.

### Failure Cost

Invalid candidates still consume time and tokens. Recovery prevents the run from stopping, but it cannot remove the cost of failed attempts.

### Single Required Benchmark

The main development effort focuses on KuaiRand-Pure. Performance on larger KuaiRand variants has not been established.

### Limited Statistical Evidence

A small number of full runs is not enough to measure the variance of the complete autonomous process. Reported results should be interpreted as a demonstrated run, not a guaranteed outcome.

### Local Optimization Risk

The agent may overfit to public validation feedback even when it never accesses the hidden evaluation set.

### Search Strategy

The active runtime uses an AIDE-style refinement loop. More direct ablation-guided experiment selection is not fully active in the submitted run.

## Potential Improvements

### Stronger Experiment Selection

Add ablation-guided selection to identify which pipeline component has the highest improvement potential before generating code.

### Better Search Memory

Store structured information about successful, failed, and redundant ideas. Use this memory to improve candidate diversity and avoid repeated experiments.

### Repeated Runs

Run the full process across multiple seeds and LLM sampling settings. Report the mean, variance, and success rate.

### Cost-Aware Routing

Use smaller models for simple analysis and reserve stronger models for difficult reasoning or code repair.

### Safer Code Generation

Add static analysis, dependency checks, and more targeted patch constraints before candidate execution.

### Better Parent Selection

Improve tree search using uncertainty, novelty, expected gain, and implementation risk instead of relying mainly on the current best candidate.

### Larger Benchmarks

Evaluate the same agent on KuaiRand-1k and KuaiRand-27k to test scalability and generalization.

### Stronger Reproducibility

Record the model version, prompt version, environment, dependency versions, seed, hardware, and dataset checksums for every run.

## Team Member Contributions

The project was divided equally among four team members.

| Team member | Primary responsibility |
|---|---|
| Team Member 1 | Harness |
| Team Member 2 | Code generation |
| Team Member 3 | LLM calls |
| Team Member 4 | Orchestrator |

All four members contributed equally to system design, integration, testing, debugging, documentation, and the final presentation.

## Responsible Use

- Do not commit API keys or secrets.
- Do not use external training data.
- Do not expose the hidden evaluation split to the agent.
- Do not alter the official evaluation metrics.
- Keep all experiment failures and interventions in the run record.
- Validate the final prediction file before submission.
