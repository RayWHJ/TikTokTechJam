"""Ablation strategies for the FM pipeline.

Each strategy is a natural-language `instruction` we hand to the writer,
framed as: "produce the pipeline with component X removed / trivialised."
The writer returns a diff against the CURRENT node's code_dir (not the
pristine baseline), so ablations track what is actually running.

A SMALL delta (parent - parent_without_c near 0) means the pipeline barely
relies on that component — it is the strongest target for refinement,
because there is the most headroom in replacing it.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Ablation:
    name: str          # matches Node.diagnosis["component"] vocabulary
    file: str          # "data.py" or "baseline.py"
    target: str        # value to pass to writer as target_component
    instruction: str   # writer-prompt body describing the removal
    description: str   # human-readable summary for logs


ABLATIONS: Dict[str, Ablation] = {
    "features": Ablation(
        name="features",
        file="data.py",
        target="features",
        instruction=(
            "Reduce the FIELDS list to exactly ['user_id', 'video_id']. "
            "Update the raw() helper inside encode() to return only those "
            "two values, in that order. Preserve every other piece of "
            "encoding logic, the public load() and encode() signatures, "
            "and the return shape (X, y, users) per split. Do not touch "
            "any other module."
        ),
        description="drop FIELDS to (user_id, video_id) only",
    ),
    "regularization": Ablation(
        name="regularization",
        file="baseline.py",
        target="loss",
        instruction=(
            "Set l2=0.0 as the default in FM.__init__. Remove the two "
            "lines inside FM.step that add `self.l2 * self.V` to gV and "
            "`self.l2 * self.W` to gW. Change nothing else — leave the "
            "loss expression, the Adam updates, and every public "
            "signature exactly as they are."
        ),
        description="disable L2 regularization",
    ),
    "capacity": Ablation(
        name="capacity",
        file="baseline.py",
        target="architecture",
        instruction=(
            "In the argparse setup at the bottom of baseline.py, change "
            "`ap.add_argument('--k', type=int, default=16)` so the "
            "default is 4 instead of 16. Also change k=16 in FM.__init__'s "
            "signature to k=4. Change nothing else."
        ),
        description="reduce embedding dim k from 16 to 4",
    ),
}
