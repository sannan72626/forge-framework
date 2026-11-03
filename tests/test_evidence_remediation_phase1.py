"""
tests/test_evidence_remediation_phase1.py — regression tests for FORGE Evidence
Remediation Phase 1 (task 6):

  (a) checkpoint restoration  — the validation-selected LoRA state is preserved
      and restorable independently of the final-step weights.
  (b) split separation       — validation and final-reporting probes are
      disjoint, deterministic, and hash-verifiable.
  (c) metric aggregation     — J_mean = mean(J_s) where J_s = FQ_NLL_s * MU_ret_s
      (NOT mean(FQ_NLL) * mean(MU_ret)), and J_sd is the *sample* std (ddof=1).
  (d) checkpoint identity    — the official-metric evaluator hard-fails when the
      adapter it is given does not match the diagnostic-selected checkpoint's
      recorded sha256, and succeeds when it does.

These are CPU-only, fast, and exercise the actual production code paths in
scripts/run_experiments.py and scripts/evaluate_tofu_official.py (not
reimplementations), except where noted. scripts/build_provenance.py executes
its provenance-building pipeline at import time (no `__main__` guard) and
reads real outputs/wk8/* data, so it is not imported here; its J_diag fix is
instead guarded by a static source-text check (see TestBuildProvenanceSourceGuard).

Run: python -m pytest tests/test_evidence_remediation_phase1.py -q
"""
import hashlib
import json
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_experiments import (  # noqa: E402
    QAExample,
    _hash_checkpoint_dir,
    _question_id,
    _split_hash,
    make_probe_splits,
)
from scripts.evaluate_tofu_official import (  # noqa: E402
    _hash_adapter_dir,
    verify_checkpoint_identity,
)


def _examples(n, prefix="q"):
    return [QAExample(question=f"{prefix}-{i} what is the answer?", answer=f"a-{i}") for i in range(n)]


# ── (b) split separation ──────────────────────────────────────────────────────

class TestProbeSplitSeparation:
    def test_question_id_is_deterministic(self):
        q = "What year was the author born?"
        assert _question_id(q) == _question_id(q)

    def test_question_id_differs_for_different_questions(self):
        assert _question_id("question A") != _question_id("question B")

    def test_split_hash_is_deterministic_and_order_independent(self):
        ex = _examples(10)
        h1 = _split_hash(ex)
        h2 = _split_hash(list(reversed(ex)))  # same set, different order
        assert h1 == h2, "split hash must depend on membership, not order"

    def test_split_hash_changes_if_membership_changes(self):
        ex = _examples(10)
        h_full = _split_hash(ex)
        h_dropped = _split_hash(ex[:-1])
        assert h_full != h_dropped

    def test_make_probe_splits_is_disjoint(self):
        ex = _examples(120)
        val_ex, val_ids, val_hash, rep_ex, rep_ids, rep_hash = make_probe_splits(ex, n_val=50, n_report=50)
        assert len(val_ex) == 50 and len(rep_ex) == 50
        assert set(val_ids).isdisjoint(set(rep_ids)), "validation/reporting probes must not overlap"
        assert val_hash != rep_hash

    def test_make_probe_splits_is_deterministic_across_calls(self):
        ex = _examples(120)
        r1 = make_probe_splits(ex, n_val=50, n_report=50)
        r2 = make_probe_splits(ex, n_val=50, n_report=50)
        assert r1[2] == r2[2]  # val_hash
        assert r1[5] == r2[5]  # report_hash

    def test_make_probe_splits_raises_when_insufficient_examples(self):
        ex = _examples(10)
        with pytest.raises(ValueError):
            make_probe_splits(ex, n_val=50, n_report=50)

    def test_make_probe_splits_hash_matches_split_hash_helper(self):
        ex = _examples(120)
        val_ex, _, val_hash, _, _, _ = make_probe_splits(ex, n_val=50, n_report=50)
        assert val_hash == _split_hash(val_ex)


# ── (d) checkpoint identity ───────────────────────────────────────────────────

class TestCheckpointIdentity:
    def test_hash_checkpoint_dir_matches_hash_adapter_dir_for_identical_content(self, tmp_path):
        ckpt_dir = tmp_path / "FORGE_selected"
        ckpt_dir.mkdir()
        (ckpt_dir / "adapter_model.safetensors").write_bytes(b"fake-lora-weights-v1")
        assert _hash_checkpoint_dir(ckpt_dir) == _hash_adapter_dir(ckpt_dir)

    def test_hash_differs_for_different_content(self, tmp_path):
        a = tmp_path / "a"; a.mkdir()
        b = tmp_path / "b"; b.mkdir()
        (a / "adapter_model.safetensors").write_bytes(b"weights-A")
        (b / "adapter_model.safetensors").write_bytes(b"weights-B")
        assert _hash_checkpoint_dir(a) != _hash_checkpoint_dir(b)

    def test_hash_checkpoint_dir_returns_none_when_no_weight_file(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        assert _hash_checkpoint_dir(empty_dir) is None

    def test_verify_checkpoint_identity_succeeds_on_matching_hash(self, tmp_path):
        manifest = {"FORGE": {"selected_checkpoint_sha256": "abc123", "selected_step": 40}}
        manifest_path = tmp_path / "checkpoint_manifests.json"
        manifest_path.write_text(json.dumps(manifest))
        step = verify_checkpoint_identity("abc123", str(manifest_path), "FORGE")
        assert step == 40

    def test_verify_checkpoint_identity_hard_fails_on_mismatch(self, tmp_path):
        manifest = {"NPO": {"selected_checkpoint_sha256": "expected-hash", "selected_step": 30}}
        manifest_path = tmp_path / "checkpoint_manifests.json"
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(RuntimeError, match="CHECKPOINT IDENTITY MISMATCH"):
            verify_checkpoint_identity("final-step-hash-not-selected", str(manifest_path), "NPO")

    def test_verify_checkpoint_identity_hard_fails_on_unknown_method(self, tmp_path):
        manifest = {"NPO": {"selected_checkpoint_sha256": "h", "selected_step": 30}}
        manifest_path = tmp_path / "checkpoint_manifests.json"
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(RuntimeError, match="not found"):
            verify_checkpoint_identity("h", str(manifest_path), "FORGE")

    def test_end_to_end_final_checkpoint_is_rejected_selected_checkpoint_is_accepted(self, tmp_path):
        """Simulates the exact Blocker-A scenario from the evidence audit: a
        final-step checkpoint and a validation-selected checkpoint have
        genuinely different weight bytes; the identity check must accept only
        the one the manifest actually names as selected."""
        final_dir = tmp_path / "FORGE_final"; final_dir.mkdir()
        selected_dir = tmp_path / "FORGE_selected"; selected_dir.mkdir()
        (final_dir / "adapter_model.safetensors").write_bytes(b"final-step-100-weights")
        (selected_dir / "adapter_model.safetensors").write_bytes(b"selected-step-40-weights")

        selected_hash = _hash_checkpoint_dir(selected_dir)
        manifest = {"FORGE": {"selected_checkpoint_sha256": selected_hash, "selected_step": 40}}
        manifest_path = tmp_path / "checkpoint_manifests.json"
        manifest_path.write_text(json.dumps(manifest))

        # Evaluating the WRONG (final) checkpoint must hard-fail.
        with pytest.raises(RuntimeError, match="CHECKPOINT IDENTITY MISMATCH"):
            verify_checkpoint_identity(_hash_adapter_dir(final_dir), str(manifest_path), "FORGE")

        # Evaluating the RIGHT (selected) checkpoint must succeed.
        step = verify_checkpoint_identity(_hash_adapter_dir(selected_dir), str(manifest_path), "FORGE")
        assert step == 40


# ── (c) metric aggregation ────────────────────────────────────────────────────

class TestMetricAggregation:
    """Encodes the Task-4 contract: J_s = FQ_NLL_s * MU_ret_s;
    J_mean = mean(J_s); J_sd = sample_std(J_s). This is a *regression guard*
    against the exact bug found in the evidence audit: J_mean computed as
    mean(FQ_NLL) * mean(MU_ret) (product of means) instead of mean of
    per-seed products -- these differ whenever FQ and MU covary across seeds."""

    def test_mean_of_products_differs_from_product_of_means_when_anticorrelated(self):
        # Mirrors the real NPO discrepancy: seed with high FQ/low MU and seeds
        # with low FQ/high MU (negatively correlated across seeds).
        fq = [0.548685, 0.548685, 0.271914]
        mu = [0.25, 0.25, 0.408274]
        j_s = [f * m for f, m in zip(fq, mu)]
        j_mean_correct = statistics.mean(j_s)
        j_mean_buggy = statistics.mean(fq) * statistics.mean(mu)
        assert abs(j_mean_correct - j_mean_buggy) > 0.002, (
            "this synthetic case must reproduce a discrepancy exceeding the "
            "0.002 stop-and-investigate threshold from the evidence audit"
        )

    def test_mean_of_products_equals_product_of_means_only_when_uncorrelated_constant(self):
        # Sanity check of the formula itself, not a repo-code test: when FQ is
        # constant across seeds, mean(FQ*MU) == mean(FQ)*mean(MU) exactly.
        fq = [0.5, 0.5, 0.5]
        mu = [0.3, 0.6, 0.9]
        j_s = [f * m for f, m in zip(fq, mu)]
        assert statistics.mean(j_s) == pytest.approx(statistics.mean(fq) * statistics.mean(mu))

    def test_j_sd_is_sample_std_not_population_std(self):
        j_s = [0.1296, 0.15, 0.11]
        sample_sd = statistics.stdev(j_s)   # ddof=1
        population_sd = statistics.pstdev(j_s)  # ddof=0
        assert sample_sd != pytest.approx(population_sd)
        # Task 4 explicitly specifies sample_std; pstdev must not be substituted.
        assert sample_sd > population_sd

    def test_aggregate_results_j_diag_matches_per_seed_product_contract(self, tmp_path, monkeypatch):
        """Exercises the REAL scripts/aggregate_results.py::load_seed_values
        against synthetic trajectory files shaped like outputs/wk8/.../trajectories.json,
        confirming it reports the per-seed product, not a separately-averaged value."""
        import scripts.aggregate_results as agg

        monkeypatch.setattr(agg, "ROOT", tmp_path)
        seed_dir = tmp_path / "outputs" / "wk8" / "FORGE" / "forget05" / "seed42"
        seed_dir.mkdir(parents=True)
        traj = {"FORGE": [
            {"step": 0, "fq": 0.10, "mu": 0.10},
            {"step": 40, "fq": 0.80, "mu": 0.70},  # argmax(fq*mu) = 0.56
            {"step": 100, "fq": 0.90, "mu": 0.20},  # argmax(fq*mu) = 0.18, must NOT be selected
        ]}
        (seed_dir / "trajectories.json").write_text(json.dumps(traj))

        rows = agg.load_seed_values("FORGE", "forget05")
        assert len(rows) == 1
        row = rows[0]
        assert row["step"] == 40, "must select the true argmax(fq*mu) checkpoint, not the final step"
        assert row["j_diag"] == pytest.approx(0.80 * 0.70)
        assert row["j_diag"] != pytest.approx(0.10 * 0.10)

    def test_aggregate_results_summary_uses_sample_std(self, tmp_path, monkeypatch):
        """Regression test for the pstdev->stdev fix applied in this remediation
        pass: summary_mean_std.csv's 'std' column must be the sample standard
        deviation (ddof=1), matching the J_sd = sample_std(J_s) contract."""
        import csv
        import scripts.aggregate_results as agg

        monkeypatch.setattr(agg, "ROOT", tmp_path)
        monkeypatch.setattr(agg, "METHODS", ["FORGE"])
        monkeypatch.setattr(agg, "SPLITS", ["forget05"])
        monkeypatch.setattr(agg, "SEEDS", [42, 123, 456])

        js = []
        for seed, (fq, mu) in zip([42, 123, 456], [(0.55, 0.25), (0.55, 0.25), (0.27, 0.41)]):
            seed_dir = tmp_path / "outputs" / "wk8" / "FORGE" / "forget05" / f"seed{seed}"
            seed_dir.mkdir(parents=True)
            traj = {"FORGE": [{"step": 40, "fq": fq, "mu": mu}]}
            (seed_dir / "trajectories.json").write_text(json.dumps(traj))
            js.append(fq * mu)

        agg.main()

        with open(tmp_path / "outputs" / "revision_stats" / "summary_mean_std.csv") as f:
            rows = list(csv.DictReader(f))
        j_row = next(r for r in rows if r["metric"] == "j_diag")
        assert float(j_row["std"]) == pytest.approx(statistics.stdev(js), abs=1e-4)
        assert float(j_row["std"]) != pytest.approx(statistics.pstdev(js), abs=1e-4)


class TestBuildProvenanceSourceGuard:
    """scripts/build_provenance.py executes its full provenance pipeline at
    import time (reads outputs/wk8/*, no __main__ guard), so it cannot be
    safely imported in a unit test. These are static source-text regression
    guards against reintroducing the exact product-of-means bug found in the
    evidence audit (manuscript J=0.1391 vs. correct mean-of-products=0.1296
    for NPO)."""

    @pytest.fixture
    def source(self):
        return (ROOT / "scripts" / "build_provenance.py").read_text()

    def test_does_not_compute_j_as_product_of_means(self, source):
        assert "j_mean = fq_mean * mu_mean" not in source, (
            "regression: J_diag must be mean(per-seed FQ*MU products), "
            "not mean(FQ)*mean(MU) -- see outputs/evidence_recovery/EVIDENCE_RECOVERY_REPORT.md"
        )

    def test_computes_j_as_mean_of_per_seed_products(self, source):
        assert "js.append(best[\"fq\"] * best[\"mu\"])" in source
        assert "j_mean = sum(js) / n" in source

    def test_j_diag_add_call_uses_used_seeds_not_raw_slice(self, source):
        assert 'add(f"tofu_phi15_{s}_{m.lower()}_j_diag"' in source
        # the add() call for j_diag must cite used_seeds (the seeds whose files
        # actually existed), not a positional seeds[:n] slice that can silently
        # mislabel which seeds a value came from when some seed files are missing.
        j_diag_call_start = source.index('add(f"tofu_phi15_{s}_{m.lower()}_j_diag"')
        j_diag_call = source[j_diag_call_start:j_diag_call_start + 400]
        assert "used_seeds" in j_diag_call


# ── (a) checkpoint restoration (end-to-end, real train_method()) ────────────
# GPU-gated, consistent with this repo's GPU-first policy (see
# tests/test_npo_simnpo_losses.py) -- exercises the real production function,
# not a reimplementation, with GPT-2 to keep it fast.

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

pytestmark_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU required (repo policy: GPU-first, no CPU fallback for real training)",
)


@pytestmark_gpu
class TestTrainMethodCheckpointRestoration:
    MODEL_NAME = "gpt2"

    @pytest.fixture(scope="class")
    def device(self):
        return torch.device("cuda")

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        tok.pad_token = tok.eos_token
        return tok

    @pytest.fixture(scope="class")
    def ref_model(self, device):
        from transformers import AutoModelForCausalLM
        ref = AutoModelForCausalLM.from_pretrained(self.MODEL_NAME).to(device)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        return ref

    def _model_factory(self):
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(self.MODEL_NAME)

    def _run(self, tmp_path, tokenizer, device, ref_model, n_steps=10, eval_every=5):
        from scripts.run_experiments import make_probe_splits, train_method

        forget_ex = _examples(16, "forget")
        retain_ex = _examples(16, "retain")
        val_f, val_f_ids, val_f_hash, rep_f, rep_f_ids, rep_f_hash = make_probe_splits(forget_ex, n_val=6, n_report=6)
        val_r, val_r_ids, val_r_hash, rep_r, rep_r_ids, rep_r_hash = make_probe_splits(retain_ex, n_val=6, n_report=6)
        val_probes = (val_f, val_r, val_f_ids, val_r_ids, val_f_hash, val_r_hash)
        report_probes = (rep_f, rep_r, rep_f_ids, rep_r_ids, rep_f_hash, rep_r_hash)

        traj, manifest = train_method(
            method_name="GA", base_state=None, ref_model=ref_model,
            forget_ex=forget_ex, retain_ex=retain_ex, tokenizer=tokenizer, device=device,
            model_factory=self._model_factory, n_steps=n_steps, eval_every=eval_every,
            lr=1e-3, batch_size=2, max_length=32, out_dir=tmp_path, lora_rank=4,
            use_lora=True, n_eval=6, model_name=self.MODEL_NAME, loss_kwargs={},
            val_probes=val_probes, report_probes=report_probes,
        )
        return traj, manifest

    def test_both_checkpoints_saved_with_recorded_hashes(self, tmp_path, tokenizer, device, ref_model):
        traj, manifest = self._run(tmp_path, tokenizer, device, ref_model)
        assert manifest["final_checkpoint_sha256"] is not None
        assert manifest["selected_checkpoint_sha256"] is not None
        assert Path(manifest["final_checkpoint_path"]).exists()
        assert Path(manifest["selected_checkpoint_path"]).exists()
        assert manifest["selected_step"] in {m["step"] for m in traj}

    def test_selected_and_final_weights_differ_iff_steps_differ(self, tmp_path, tokenizer, device, ref_model):
        traj, manifest = self._run(tmp_path, tokenizer, device, ref_model)
        same_step = manifest["selected_step"] == manifest["final_step"]
        same_hash = manifest["selected_checkpoint_sha256"] == manifest["final_checkpoint_sha256"]
        assert same_step == same_hash, (
            "selected and final checkpoints must have identical weight hashes "
            "exactly when the selected step IS the final step, and different "
            "hashes whenever training continued past the selected step "
            "(Blocker A from the evidence audit: the old code always saved "
            "only the final-step weights, silently mislabeling them as the "
            "validation-selected checkpoint whenever the two steps differed)"
        )

    def test_report_probe_disjoint_from_validation_probe(self, tmp_path, tokenizer, device, ref_model):
        _, manifest = self._run(tmp_path, tokenizer, device, ref_model)
        val_ids = set(manifest["validation_probe"]["forget_question_ids"])
        rep_ids = set(manifest["report_probe"]["forget_question_ids"])
        assert val_ids.isdisjoint(rep_ids)
        assert manifest["validation_probe"]["forget_split_hash"] != manifest["report_probe"]["forget_split_hash"]

    def test_reported_metrics_computed_on_held_out_probe_at_selected_checkpoint(self, tmp_path, tokenizer, device, ref_model):
        _, manifest = self._run(tmp_path, tokenizer, device, ref_model)
        reported = manifest["reported_metrics"]
        for key in ("fq_nll", "mu_ret", "j_diag"):
            assert key in reported, f"missing '{key}' in reported_metrics -- report_probes was given but not used"
        assert reported["j_diag"] == pytest.approx(reported["fq_nll"] * reported["mu_ret"])

    def test_no_reported_metrics_fabricated_when_report_probes_absent(self, tmp_path, tokenizer, device, ref_model):
        from scripts.run_experiments import make_probe_splits, train_method

        forget_ex = _examples(16, "forget")
        retain_ex = _examples(16, "retain")
        val_f, val_f_ids, val_f_hash, _, _, _ = make_probe_splits(forget_ex, n_val=6, n_report=6)
        val_r, val_r_ids, val_r_hash, _, _, _ = make_probe_splits(retain_ex, n_val=6, n_report=6)
        val_probes = (val_f, val_r, val_f_ids, val_r_ids, val_f_hash, val_r_hash)

        _, manifest = train_method(
            method_name="GA", base_state=None, ref_model=ref_model,
            forget_ex=forget_ex, retain_ex=retain_ex, tokenizer=tokenizer, device=device,
            model_factory=self._model_factory, n_steps=5, eval_every=5,
            lr=1e-3, batch_size=2, max_length=32, out_dir=tmp_path, lora_rank=4,
            use_lora=True, n_eval=6, model_name=self.MODEL_NAME, loss_kwargs={},
            val_probes=val_probes, report_probes=None,
        )
        assert "fq_nll" not in manifest["reported_metrics"], (
            "must not substitute the validation-probe value for a 'reported' "
            "number when no report_probes were given -- that is exactly the "
            "leakage this remediation exists to remove"
        )
        assert "WARNING" in manifest["reported_metrics"]["note"]

    def test_save_all_steps_false_by_default_no_intermediate_dirs(self, tmp_path, tokenizer, device, ref_model):
        """Phase 2C, Task B: save_all_steps defaults to False, so every other
        caller of train_method() (Llama-2-7B, cross-split work) keeps its
        existing on-disk footprint unchanged -- only {method}_final and
        {method}_selected should exist, nothing named {method}_step*."""
        traj, manifest = self._run(tmp_path, tokenizer, device, ref_model)
        step_dirs = sorted(p.name for p in tmp_path.iterdir() if p.is_dir() and "_step" in p.name)
        assert step_dirs == [], f"save_all_steps=False must not create per-step dirs, found: {step_dirs}"
        assert manifest["all_step_manifests"] == []

    def test_save_all_steps_true_saves_every_scheduled_step(self, tmp_path, tokenizer, device, ref_model):
        from scripts.run_experiments import make_probe_splits, train_method, _hash_checkpoint_dir

        forget_ex = _examples(16, "forget")
        retain_ex = _examples(16, "retain")
        val_f, val_f_ids, val_f_hash, rep_f, rep_f_ids, rep_f_hash = make_probe_splits(forget_ex, n_val=6, n_report=6)
        val_r, val_r_ids, val_r_hash, rep_r, rep_r_ids, rep_r_hash = make_probe_splits(retain_ex, n_val=6, n_report=6)
        val_probes = (val_f, val_r, val_f_ids, val_r_ids, val_f_hash, val_r_hash)
        report_probes = (rep_f, rep_r, rep_f_ids, rep_r_ids, rep_f_hash, rep_r_hash)

        n_steps, eval_every = 10, 5
        _, manifest = train_method(
            method_name="GA", base_state=None, ref_model=ref_model,
            forget_ex=forget_ex, retain_ex=retain_ex, tokenizer=tokenizer, device=device,
            model_factory=self._model_factory, n_steps=n_steps, eval_every=eval_every,
            lr=1e-3, batch_size=2, max_length=32, out_dir=tmp_path, lora_rank=4,
            use_lora=True, n_eval=6, model_name=self.MODEL_NAME, loss_kwargs={},
            val_probes=val_probes, report_probes=report_probes,
            save_all_steps=True, run_id="test-run-id", seed=123,
            base_model_revision="test-base-rev", tokenizer_revision="test-tok-rev",
            code_sha256="test-code-sha", config_sha256="test-config-sha",
        )
        expected_steps = {0, 5, 10}  # n_steps=10, eval_every=5 -> checkpoint(0), checkpoint(5), checkpoint(10)
        got_steps = {m["step"] for m in manifest["all_step_manifests"]}
        assert got_steps == expected_steps, f"expected per-step saves at {expected_steps}, got {got_steps}"

        for step_m in manifest["all_step_manifests"]:
            assert step_m["run_id"] == "test-run-id"
            assert step_m["seed"] == 123
            assert step_m["base_model_revision"] == "test-base-rev"
            assert step_m["resumable_for_exact_training_replay"] is False
            on_disk_hash = _hash_checkpoint_dir(Path(step_m["adapter_path"]))
            assert on_disk_hash == step_m["adapter_sha256"], (
                f"step {step_m['step']}: recorded sha256 must match the actual on-disk file"
            )

        # the already-selected and final checkpoints must ALSO exist as their own
        # separate dirs (unaffected by the new per-step saving)
        assert (tmp_path / "GA_final").exists()
        assert (tmp_path / "GA_selected").exists()
