"""Strict loading for experiments that must retain every pretrained backbone leaf."""

import dataclasses
import logging

import flax.traverse_util
import numpy as np

from openpi.models import model
from openpi.shared import download

CONTEXT_ROOTS = frozenset(
    {
        "demo_action_proj",
        "demo_state_proj",
        "image_compressor",
        "state_compressor",
        "action_compressor",
    }
)


def is_new_parameter(path: str) -> bool:
    return path.split("/", 1)[0] in CONTEXT_ROOTS or any("lora" in part for part in path.split("/"))


def merge_full_backbone(loaded, template):
    """Require exact backbone keys and shapes, allowing only context/LoRA initialization."""
    flat = flax.traverse_util.flatten_dict(loaded, sep="/")
    ref = flax.traverse_util.flatten_dict(template, sep="/")
    missing = sorted(k for k in ref if k not in flat and not is_new_parameter(k))
    unexpected = sorted(k for k in flat if k not in ref)
    mismatched = sorted(k for k in ref.keys() & flat.keys() if ref[k].shape != flat[k].shape)
    if missing or unexpected or mismatched:
        raise ValueError(
            f"Pretrained backbone mismatch: missing={missing}, unexpected={unexpected}, shape_mismatch={mismatched}"
        )
    result = {k: flat[k].astype(v.dtype) if k in flat else v for k, v in ref.items()}
    logging.info(
        "Full backbone loaded: %d leaves / %d parameters; newly initialized context/LoRA: %d leaves",
        len(flat),
        sum(int(np.prod(v.shape)) for v in flat.values()),
        len(ref) - len(flat),
    )
    return flax.traverse_util.unflatten_dict(result, sep="/")


@dataclasses.dataclass(frozen=True)
class FullBackboneWeightLoader:
    params_path: str

    def load(self, params):
        loaded = model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return merge_full_backbone(loaded, params)
