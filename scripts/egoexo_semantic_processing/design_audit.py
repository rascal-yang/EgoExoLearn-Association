"""CPU mathematical checks for the proposed attention-memory study.

These synthetic checks are not training, benchmark results, or evidence of
semantic grounding. Native model/GPU implementation checks remain pending.
"""
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20261027)
    dtype = torch.float64

    # In M, mean token-to-word similarity equals word similarity to a mean
    # normalized-token vector. The neutral unmatched token adds one to N.
    z = F.normalize(torch.randn(3, 23, 128, dtype=dtype), dim=-1).detach().requires_grad_(True)
    words = F.normalize(torch.randn(7, 128, dtype=dtype), dim=-1)
    logits = torch.einsum('bnd,cd->bnc', z, words).sum(1) / 24
    pooled = (z.sum(1) / 24) @ words.T
    error = float((logits - pooled).abs().max())
    assert error < 1e-12
    gradient, = torch.autograd.grad(F.cross_entropy(logits, torch.tensor([0, 1, 2])), z)
    gradient_difference = float((gradient - gradient[:, :1]).abs().max())
    assert gradient_difference < 1e-12

    # Full queries are retained. For groups of identical key/value inputs,
    # adding log(group size) to the compact attention logits preserves the
    # original multiplicity and exactly reproduces dense attention.
    counts = torch.tensor([1, 3, 5, 7, 2, 4, 6, 4])
    n, d = int(counts.sum()), 32
    memory = torch.randn(2, len(counts), d, dtype=dtype)
    dense = torch.repeat_interleave(memory, counts, dim=1)
    queries = torch.randn(2, n, d, dtype=dtype)
    wk = torch.randn(d, d, dtype=dtype) / math.sqrt(d)
    wv = torch.randn(d, d, dtype=dtype) / math.sqrt(d)
    bk = torch.randn(d, dtype=dtype)
    bv = torch.randn(d, dtype=dtype)

    def attend(x, mass=None):
        scores = (queries @ (x @ wk + bk).transpose(1, 2)) / math.sqrt(d)
        if mass is not None:
            scores = scores + mass.to(dtype).log()[None, None]
        return scores.softmax(-1) @ (x @ wv + bv)

    out = attend(dense)
    compact = attend(memory, counts)
    exact_error = float((out - compact).abs().max())
    without_mass_error = float((out - attend(memory)).abs().max())
    assert exact_error < 1e-12
    assert without_mass_error > 1e-6

    # General merging is approximate even with multiplicity correction.
    perturbed = dense + .1 * torch.randn_like(dense)
    means = torch.stack([part.mean(1) for part in perturbed.split(counts.tolist(), dim=1)], dim=1)
    approximation_error = float((attend(perturbed) - attend(means, counts)).abs().max())
    assert approximation_error > 1e-8

    report = dict(
        scope='Synthetic CPU mathematical audit only; no model training, no GPU, no dataset labels.',
        mean_similarity_equivalence_max_error=error,
        normalized_token_gradient_max_difference=gradient_difference,
        gradient_caveat='Equal gradient is with respect to normalized token features; raw features have different normalization Jacobians.',
        dense_query_count=n,
        compact_memory_count=len(counts),
        represented_mass=int(counts.sum()),
        identical_group_attention_max_error=exact_error,
        missing_multiplicity_attention_max_error=without_mass_error,
        nonidentical_group_attention_approximation_error=approximation_error,
        native_model_wrapper_verified=False,
        training_or_accuracy_result=False,
    )
    root = Path(__file__).resolve().parents[2]
    path = root / 'reports/egoexo_semantic_processing_20261027/CPU数学审计.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
