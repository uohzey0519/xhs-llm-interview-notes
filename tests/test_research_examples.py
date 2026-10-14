"""CPU checks for teaching claims; not model training or benchmark reproduction."""

import itertools
import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from research_updates import curated_answers


def examples(key):
    ns = {}
    for code in re.findall(r"```python\n(.*?)```", curated_answers()[key], re.S):
        exec(compile(code, key, "exec"), ns)
    return ns


class ResearchExamples(unittest.TestCase):
    def test_mla_absorption_nonsquare(self):
        rng = np.random.default_rng(8)
        fn = examples("mla_deepseek")["mla_content_paths"]
        explicit, absorbed, logits, absorbed_logits = fn(
            rng.normal(size=(11, 3)), rng.normal(size=5),
            rng.normal(size=(5, 3)), rng.normal(size=(7, 3)))
        np.testing.assert_allclose(explicit, absorbed, atol=1e-12)
        np.testing.assert_allclose(logits, absorbed_logits, atol=1e-12)
        # A rank-1 factorization cannot reproduce the 2x2 identity.
        self.assertEqual(np.linalg.matrix_rank(np.ones((2, 1)) @ np.ones((1, 2))), 1)

    def test_rope_relative_identity_and_noncommutativity(self):
        rotate = examples("rope_position")["rotate_pairs"]
        q, k, freq = np.array([1., 2., 3., 4.]), np.array([4., 2., 1., 3.]), [1., .01]
        np.testing.assert_allclose(rotate(q, 7, freq) @ rotate(k, 13, freq),
                                   q @ rotate(k, 6, freq), atol=1e-12)
        u, c = np.array([[1., 2.], [0., 3.]]), np.array([.4, .8])
        self.assertFalse(np.allclose(rotate(u @ c, 1, [1.]), u @ rotate(c, 1, [1.])))

    def test_gae_failure_not_all_negative(self):
        gae = examples("reward_advantage_prm")["action_gae"]
        a, _ = gae([0, 0, 0], [.2, .8, .1], [False, False, True], lam=0)
        np.testing.assert_allclose(a, [.6, -.7, -.1])
        a, targets = gae([0, 0, 0], [.2, .8, .1], [False, False, True], lam=1)
        np.testing.assert_allclose(a, [-.2, -.8, -.1])
        np.testing.assert_allclose(targets, 0, atol=1e-15)

    def test_gae_terminal_vs_truncation(self):
        gae = examples("reward_advantage_prm")["action_gae"]
        terminated, _ = gae([0], [.4], [True], bootstrap=.8)
        truncated, _ = gae([0], [.4], [False], bootstrap=.8)
        np.testing.assert_allclose(terminated, [-.4])
        np.testing.assert_allclose(truncated, [.4])
        # Earlier episode must not receive reward from a later episode.
        a, _ = gae([1, 10], [0, 0], [True, True], lam=1)
        np.testing.assert_allclose(a, [1, 10])

    def test_dis_outliers_masks_and_empty(self):
        ns = examples("glm_sao")
        weights = ns["dis_weights"]
        delta = np.array([np.log(.5), 0., np.log(2.), 0., 1000., -1000.])
        with np.errstate(over="raise"):
            w = weights(delta, np.zeros(6), [True, True, True, False, True, True])
        np.testing.assert_array_equal(w, [0, 1, 0, 0, 0, 0])
        self.assertEqual(ns["dis_surrogate"](delta, w * 0, np.ones(6), [True]*6), 0)
        with self.assertRaises(ValueError):
            ns["dis_surrogate"]([0], [0], [0], [False])
        edge = np.array([np.log1p(-.2), np.log1p(.2)])
        np.testing.assert_array_equal(weights(edge, [0, 0], [True, True]), [0, 0])

    def test_dis_frozen_weight_gradient(self):
        ns = examples("glm_sao")
        logp = np.array([-1., -2.])
        w = ns["dis_weights"](logp, logp - np.log([1.1, .9]), [True, True])
        a = np.array([2., -3.])
        for i in range(2):
            direction = np.eye(2)[i] * 1e-6
            fd = (ns["dis_surrogate"](logp + direction, w, a, [True, True])
                  - ns["dis_surrogate"](logp - direction, w, a, [True, True])) / 2e-6
            self.assertAlmostEqual(fd, -w[i] * a[i] / 2, places=8)

    def test_depth_attention_axis_and_uniform_average(self):
        fn = examples("attention_residual")["depth_mix"]
        rng = np.random.default_rng(3)
        values = rng.normal(size=(4, 2, 3, 5))
        out, weights = fn(values, np.zeros(5))
        np.testing.assert_allclose(out, values.mean(axis=0))
        np.testing.assert_allclose(weights.sum(axis=0), 1)
        q = rng.normal(size=5)
        before, _ = fn(values, q)
        changed = values.copy()
        changed[:, :, 0, :] *= 5
        after, _ = fn(changed, q)
        np.testing.assert_allclose(before[:, 1:], after[:, 1:])
        one, _ = fn(values[:1], q)
        np.testing.assert_allclose(one, values[0])

    def test_delta_overwrite_and_interference(self):
        fn = examples("sparse_linear_attention")["delta_update"]
        k1, k2 = np.array([1., 0.]), np.array([.9, np.sqrt(.19)])
        state = fn(np.zeros((2, 1)), k1, np.array([1.]))
        state = fn(state, k2, np.array([0.]))
        np.testing.assert_allclose(k2 @ state, 0, atol=1e-15)
        np.testing.assert_allclose(k1 @ state, [.19])
        zero = fn(state, k1, np.array([2.]), alpha=0, beta=0)
        np.testing.assert_array_equal(zero, np.zeros((2, 1)))

    def test_linear_recurrence_matches_explicit_causal_kernel(self):
        fn = examples("sparse_linear_attention")["causal_linear"]
        rng = np.random.default_rng(4)
        q, k, v = rng.random((9, 4)), rng.random((9, 4)), rng.normal(size=(9, 3))
        recurrent = fn(q, k, v)
        explicit = []
        for t in range(9):
            scores = k[:t+1] @ q[t]
            explicit.append(scores @ v[:t+1] / (scores.sum() + 1e-12))
        np.testing.assert_allclose(recurrent, explicit, atol=1e-12)

    def test_reliability_matches_exhaustive_subsets(self):
        fn = examples("agent_eval_observability")["reliability"]
        for n in range(1, 7):
            for c in range(n + 1):
                trials = [True] * c + [False] * (n - c)
                for k in range(1, n + 1):
                    subsets = list(itertools.combinations(trials, k))
                    r = fn(n, c, k)
                    self.assertAlmostEqual(r["pass_at_k"], sum(map(any, subsets))/len(subsets))
                    self.assertAlmostEqual(r["pass_power_k"], sum(map(all, subsets))/len(subsets))
        with self.assertRaises(ValueError):
            fn(2, 1, 3)

    def test_state_grader_requires_outcome_and_constraints(self):
        fn = examples("agent_eval_observability")["grade_state"]
        self.assertFalse(fn({"saved": True}, {}, [])["task_success"])
        self.assertFalse(fn({"saved": True}, {"saved": True}, ["unauthorized"])["task_success"])
        self.assertTrue(fn({"saved": True}, {"saved": True}, [])["task_success"])


if __name__ == "__main__":
    unittest.main()
