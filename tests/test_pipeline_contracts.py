"""Data-free contracts, extracted from source without Torch or ClearML imports."""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44, 45, 46]


def extract(filename, functions, constants=()):
    path = ROOT / "final_exps" / filename
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0,
    )]
    namespace = {
        "__file__": str(path), "argparse": argparse, "Path": Path,
        "np": np, "pd": pd, "tempfile": tempfile, "shutil": shutil,
        "hashlib": hashlib,
    }
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in constants:
                    namespace[target.id] = ast.literal_eval(node.value)
                    found.add(target.id)
    missing = set(functions) | set(constants)
    if missing - found:
        raise AssertionError(f"Missing source contracts: {sorted(missing - found)}")
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def synthetic_predictions():
    rows = []
    for version in ["raw_4096", "condition_era_90_backfill_4096"]:
        for seed in SEEDS:
            for episode in range(1, 5):
                logit = (episode + seed % 3) / 10
                rows.append({
                    "task": "synthetic", "model": "fixture_model",
                    "model_family": "numeric_sequence", "representation": version,
                    "compression_version": version, "max_len": 4096,
                    "era_gap": np.nan if version == "raw_4096" else 90,
                    "numeric_on": True, "seed": seed, "split": "held_out",
                    "row_id": episode, "subject_id": (episode + 1) // 2,
                    "prediction_time": f"2000-01-0{episode}", "y_true": episode % 2,
                    "logit": logit, "risk_raw": 1 / (1 + np.exp(-logit)),
                    "risk_calibrated": 0.1 + episode / 20 + (seed - 42) / 100,
                })
    return pd.DataFrame(rows)


class BuilderQueueTests(unittest.TestCase):
    def test_cli_queue_overrides_connected_config_and_execution(self):
        ns = extract("01_build_sequence_datasets.py", {"parse_args", "maybe_init_clearml"})
        with mock.patch.object(sys, "argv", [
            "builder", "--run-config", "synthetic.json", "--enable-clearml",
            "--execute-remotely", "--clearml-queue", "synthetic_queue",
        ]):
            args = ns["parse_args"]()
        ns["is_clearml_agent_run"] = lambda: False
        task = mock.Mock()
        fake_task = mock.Mock()
        fake_task.init.return_value = task
        cfg = {"run_set_id": "synthetic", "clearml": {"queue": "config_queue"}}
        with mock.patch.dict(sys.modules, {"clearml": SimpleNamespace(Task=fake_task)}):
            ns["maybe_init_clearml"](args, cfg)
        self.assertEqual(cfg["clearml"]["queue"], "synthetic_queue")
        task.connect.assert_called_once_with(cfg)
        task.execute_remotely.assert_called_once_with(queue_name="synthetic_queue", exit_process=True)

    def test_omitted_queue_keeps_config_default(self):
        ns = extract("01_build_sequence_datasets.py", {"parse_args", "maybe_init_clearml"})
        with mock.patch.object(sys, "argv", ["builder", "--run-config", "synthetic.json"]):
            args = ns["parse_args"]()
        self.assertIsNone(args.clearml_queue)
        ns["is_clearml_agent_run"] = lambda: False
        cfg = {"clearml": {"enabled": False, "queue": "config_queue"}}
        self.assertIsNone(ns["maybe_init_clearml"](args, cfg))
        self.assertEqual(cfg["clearml"]["queue"], "config_queue")


class EnsembleContractTests(unittest.TestCase):
    def setUp(self):
        self.ns = extract("03_analyze_state_or_space.py", {
            "expected_task_versions", "sigmoid_np", "validate_wide_predictions", "make_ensemble",
            "merge_prediction_runs",
        }, {"WIDE_REQUIRED_COLUMNS", "REQUIRED_SEEDS"})
        self.config = {"seeds": SEEDS, "paired_comparisons": [{
            "task": "synthetic", "model_a": "condition_era_90_backfill_4096", "model_b": "raw_4096",
        }]}
        self.pred = synthetic_predictions()

    def validate(self, frame=None, config=None):
        return self.ns["validate_wide_predictions"](
            self.pred if frame is None else frame, self.config if config is None else config,
        )

    def test_valid_reordered_cohort_preserves_probability_means(self):
        valid = self.validate(self.pred.sample(frac=1, random_state=1))
        ensemble = self.ns["make_ensemble"](valid)
        self.assertEqual(len(ensemble), 8)
        self.assertTrue(ensemble.n_seeds.eq(5).all())
        expected = self.pred.groupby(["compression_version", "row_id"]).risk_calibrated.mean()
        actual = ensemble.set_index(["compression_version", "row_id"]).risk_calibrated
        pd.testing.assert_series_equal(actual.sort_index(), expected.sort_index())

    def test_requires_the_exact_final_seed_configuration(self):
        for seeds in [SEEDS[:-1], [1, 2, 3, 4, 5], SEEDS + [42]]:
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                self.validate(config={**self.config, "seeds": seeds})

    def test_missing_seed_is_rejected(self):
        bad = self.pred.loc[~((self.pred.seed == 46) & (self.pred.compression_version == "raw_4096"))]
        with self.assertRaisesRegex(ValueError, "seed coverage"):
            self.validate(bad)

    def test_equal_counts_with_disjoint_seed_cohorts_are_rejected(self):
        bad = self.pred.copy()
        bad.loc[bad.seed == 46, "row_id"] += 100
        with self.assertRaisesRegex(ValueError, "cohort identity"):
            self.validate(bad)
        with self.assertRaises(ValueError):
            self.ns["make_ensemble"](bad)

    def test_episode_metadata_must_agree_across_seeds(self):
        for column, value in [("subject_id", 999), ("y_true", 0), ("prediction_time", "2001-01-01")]:
            bad = self.pred.copy()
            bad.loc[0, column] = value
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "cohort identity"):
                self.validate(bad)

    def test_representation_cohorts_must_agree(self):
        bad = self.pred.copy()
        bad.loc[bad.compression_version == "raw_4096", "row_id"] += 100
        with self.assertRaisesRegex(ValueError, "Representation cohort"):
            self.validate(bad)

    def test_run_metadata_drift_is_rejected(self):
        for column, value in [("model", "other"), ("max_len", 16384), ("representation", "other")]:
            bad = self.pred.copy()
            bad.loc[0, column] = value
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "run metadata"):
                self.validate(bad)

    def test_duplicate_episode_with_changed_subject_is_rejected(self):
        extra = self.pred.iloc[[0]].copy()
        extra["subject_id"] = 999
        bad = pd.concat([self.pred, extra], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.validate(bad)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.ns["make_ensemble"](bad)

    def test_missing_and_fractional_identities_are_rejected(self):
        for column, value in [("prediction_time", None), ("subject_id", np.nan), ("seed", 42.5)]:
            bad = self.pred.copy()
            bad[column] = bad[column].astype(object)
            bad.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                self.validate(bad)

    def test_unexpected_representation_is_rejected(self):
        extra = self.pred[self.pred.compression_version == "raw_4096"].copy()
        extra["compression_version"] = "unexpected"
        with self.assertRaisesRegex(ValueError, "matrix"):
            self.validate(pd.concat([self.pred, extra], ignore_index=True))

    def test_direct_ensemble_cannot_use_one_seed(self):
        with self.assertRaisesRegex(ValueError, "five final seeds"):
            self.ns["make_ensemble"](self.pred[self.pred.seed == 42])

    def test_invalid_merged_cohorts_are_rejected_before_file_write_or_upload(self):
        bad = self.pred.copy()
        bad.loc[bad.seed == 46, "row_id"] += 100
        self.ns["parse_run_tags"] = lambda value: ["synthetic"]
        self.ns["load_prediction_run"] = lambda *args: (bad, {"run_tag": "synthetic"})
        self.ns["assert_duplicate_predictions_consistent"] = mock.Mock()
        self.ns["upload_file"] = mock.Mock()
        with tempfile.TemporaryDirectory(prefix="pipeline_merge_contract_") as temp:
            output = Path(temp) / "combined.csv"
            args = SimpleNamespace(
                prediction_run_tags="synthetic", prediction_results_dir=Path(temp),
                prediction_results_s3_root="", prediction_filename="synthetic.csv",
                predictions=output, skip_combined_upload=False,
                combined_predictions_s3_url="s3://storage.invalid/synthetic.csv",
            )
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "cohort identity"):
                self.ns["merge_prediction_runs"](args, self.config)
            self.assertFalse(output.exists())
            self.ns["upload_file"].assert_not_called()


class PublicExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pipeline_contracts_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ns = extract("07_prepare_public_results.py", {
            "assert_public_checks_pass", "validate_output_directory", "publish_staged_outputs",
            "scan_public_outputs", "sha256_file", "write_sha_manifest", "upload_safe_outputs", "main",
        }, {"FORBIDDEN_COLUMN_NAMES", "FORBIDDEN_OUTPUT_NAME_PARTS"})
        self.output = self.root / "public_results"
        self.args = SimpleNamespace(output_dir=self.output, source_root=None, make_zip=False,
                                    output_s3_prefix="", enable_clearml=False)
        self.ns["parse_args"] = lambda: self.args
        self.ns["collect_inputs"] = mock.Mock(return_value={})
        self.ns["create_tables"] = lambda paths, out: {"synthetic": self.write_pair(out / "tables/table")}
        self.ns["create_figure_1"] = lambda paths, out: self.write_pair(out / "figure_1")
        self.ns["create_figure_2"] = lambda paths, out: self.write_pair(out / "figure_2")
        self.ns["create_results_summary"] = lambda paths, out: self.write_pair(out / "summary")
        self.ns["create_public_checks"] = lambda paths, out: self.write_checks(out, "PASS")
        self.ns["write_readme"] = lambda out: self.write_file(out / "README.md", "synthetic\n")
        self.ns["write_settings"] = lambda out: self.write_file(out / "experiment_settings.json", "{}\n")
        self.ns["init_clearml"] = mock.Mock(return_value=None)
        self.ns["upload_safe_outputs"] = mock.Mock()

    @staticmethod
    def write_file(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def write_pair(self, stem):
        return (self.write_file(stem.with_suffix(".csv"), "count\n1\n"),
                self.write_file(stem.with_suffix(".md"), "synthetic\n"))

    def write_checks(self, out, status):
        stem = out / "checks/publication_safety_and_integrity"
        return (self.write_file(stem.with_suffix(".csv"), f"check,status\nsynthetic,{status}\n"),
                self.write_file(stem.with_suffix(".md"), "synthetic\n"))

    def previous_export(self):
        self.write_file(self.output / "PUBLICATION_SAFETY_MANIFEST.csv", "file,safe\nold,True\n")
        self.write_file(self.output / "experiment_settings.json", "{}\n")
        self.write_file(self.output / "old.txt", "retain these previous results\n")
        return {p.name: p.read_bytes() for p in self.output.iterdir()}

    def assert_previous_untouched(self, snapshot):
        self.assertEqual({p.name: p.read_bytes() for p in self.output.iterdir()}, snapshot)
        self.ns["init_clearml"].assert_not_called()
        self.ns["upload_safe_outputs"].assert_not_called()
        self.assertFalse(list(self.root.glob(".public_results.staging-*")))

    def run_main(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.ns["main"]()

    def test_missing_inputs_preserve_previous_export(self):
        snapshot = self.previous_export()
        self.ns["collect_inputs"].side_effect = FileNotFoundError("synthetic missing input")
        with self.assertRaises(FileNotFoundError):
            self.run_main()
        self.assert_previous_untouched(snapshot)

    def test_failed_integrity_checks_preserve_previous_export_and_never_upload(self):
        snapshot = self.previous_export()
        self.ns["create_public_checks"] = lambda paths, out: self.write_checks(out, "FAIL")
        with self.assertRaisesRegex(RuntimeError, "integrity checks failed"):
            self.run_main()
        self.assert_previous_untouched(snapshot)

    def test_failed_safety_scan_preserves_previous_export(self):
        snapshot = self.previous_export()
        self.ns["scan_public_outputs"] = lambda out: pd.DataFrame({"safe": [False]})
        with self.assertRaisesRegex(RuntimeError, "safety scan failed"):
            self.run_main()
        self.assert_previous_untouched(snapshot)

    def test_success_promotes_checked_files_and_retains_backup(self):
        snapshot = self.previous_export()
        self.run_main()
        self.assertFalse((self.output / "old.txt").exists())
        self.assertTrue((self.output / "SHA256SUMS.txt").is_file())
        backups = list(self.root.glob(".public_results.previous-*/public_results"))
        self.assertEqual(len(backups), 1)
        self.assertEqual({p.name: p.read_bytes() for p in backups[0].iterdir()}, snapshot)
        self.ns["upload_safe_outputs"].assert_called_once_with(self.args, self.output.resolve())

    def test_promoted_artifact_paths_exist_for_mocked_clearml_uploads(self):
        task = mock.Mock()
        self.ns["init_clearml"].return_value = task
        self.run_main()
        self.assertGreater(task.upload_artifact.call_count, 0)
        for call in task.upload_artifact.call_args_list:
            path = Path(call.kwargs["artifact_object"])
            self.assertTrue(path.is_file())
            self.assertIn(self.output.resolve(), path.parents)

    def test_failed_promotion_restores_previous_export(self):
        snapshot = self.previous_export()
        staged = self.root / "staged"
        staged.mkdir()
        original_rename = Path.rename

        def fail_staged(path, target):
            if path == staged:
                raise OSError("synthetic promotion failure")
            return original_rename(path, target)

        with mock.patch.object(Path, "rename", fail_staged), self.assertRaises(OSError):
            self.ns["publish_staged_outputs"](staged, self.output)
        self.assertEqual({p.name: p.read_bytes() for p in self.output.iterdir()}, snapshot)
        self.assertTrue(staged.is_dir())

    def test_unrecognized_directory_and_source_overlap_are_rejected(self):
        self.write_file(self.output / "unrelated.txt", "preserve\n")
        with self.assertRaisesRegex(ValueError, "recognized"):
            self.run_main()
        self.assertTrue((self.output / "unrelated.txt").is_file())
        self.args.source_root = self.root
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.ns["validate_output_directory"](self.args)

    def test_repository_and_protected_output_paths_are_rejected(self):
        for path in [ROOT, ROOT.parent, ROOT / "final_exps/new", ROOT / "configs"]:
            self.args.output_dir = path
            with self.subTest(path=path.name), self.assertRaises(ValueError):
                self.ns["validate_output_directory"](self.args)

    def test_zip_is_staged_and_existing_zip_is_not_overwritten(self):
        self.args.make_zip = True
        self.run_main()
        zip_path = self.root / "public_results.zip"
        before = zip_path.read_bytes()
        snapshot = {p.name: p.read_bytes() for p in self.output.iterdir() if p.is_file()}
        with self.assertRaises(FileExistsError):
            self.run_main()
        self.assertEqual(zip_path.read_bytes(), before)
        self.assertEqual({p.name: p.read_bytes() for p in self.output.iterdir() if p.is_file()}, snapshot)

    def test_empty_or_nonpass_integrity_table_is_rejected(self):
        for content in ["check,status\n", "check,status\nsynthetic,FAIL\n", "check,status\nsynthetic,\n"]:
            path = self.write_file(self.root / "checks.csv", content)
            with self.subTest(content=content), self.assertRaises(RuntimeError):
                self.ns["assert_public_checks_pass"](path)

    def test_direct_storage_upload_is_gated_before_clearml_import(self):
        ns = extract("07_prepare_public_results.py", {"upload_safe_outputs", "assert_public_checks_pass"})
        self.write_checks(self.output, "FAIL")
        args = SimpleNamespace(output_s3_prefix="s3://storage.invalid/synthetic")
        with mock.patch.dict(sys.modules, {"clearml": None}), self.assertRaisesRegex(RuntimeError, "integrity"):
            ns["upload_safe_outputs"](args, self.output)

    def test_real_integrity_checks_reject_empty_or_nonboolean_audits(self):
        ns = extract("07_prepare_public_results.py", {"create_public_checks", "assert_public_checks_pass"})
        frames = {
            "split_status": pd.DataFrame({"status": ["OK"]}),
            "representation_invariants": pd.DataFrame({"passed": [True]}),
            "zero_agreement": pd.DataFrame({
                "n_compared": [1] * 20, "max_abs_logit_difference": [0.] * 20,
                "max_abs_calibrated_risk_difference": [0.] * 20,
            }),
            "ensemble_metrics": pd.DataFrame({"synthetic": range(14)}),
            "copy_probability": pd.DataFrame([
                {"task": task, "compression_version": version, "copy_fraction": fraction}
                for task in ["synthetic_a", "synthetic_b"] for version in ["raw", "era"]
                for fraction in [0., 0.25, 0.5, 1.]
            ]),
        }

        def write_checks(frame, stem):
            csv_path = stem.with_suffix(".csv")
            frame.to_csv(csv_path, index=False)
            return csv_path, stem.with_suffix(".md")

        ns["write_table"] = write_checks
        real_read_csv = pd.read_csv
        for passed in [[], ["False"], ["unknown"], [None]]:
            frames["representation_invariants"] = pd.DataFrame({"passed": passed})
            with self.subTest(passed=passed):
                with mock.patch.object(pd, "read_csv", side_effect=lambda key: frames[key].copy()):
                    csv_path, _ = ns["create_public_checks"]({key: key for key in frames}, self.root)
                self.assertIn("FAIL", set(real_read_csv(csv_path).status))
                with self.assertRaisesRegex(RuntimeError, "representation_invariants"):
                    ns["assert_public_checks_pass"](csv_path)

    def test_history_export_retains_the_actual_backfill_field(self):
        ns = extract("07_prepare_public_results.py", {"create_tables"}, {
            "TASK_LABELS", "VERSION_LABELS", "COMPARISON_LABELS", "METRIC_LABELS",
        })
        frames = {
            "ensemble_metrics": pd.DataFrame({"task": ["synthetic"], "compression_version": ["raw_4096"], "max_len": [4096]}),
            "paired_bootstrap": pd.DataFrame({
                "task": ["synthetic"], "comparison": ["Full4096"], "metric": ["auroc"],
                "higher_is_better": [True], "point_delta_a_minus_b": [0.], "ci_low": [-0.1], "ci_high": [0.1],
            }),
            "copy_robustness": pd.DataFrame({"task": ["synthetic"], "copy_fraction": [1.]}),
            "copy_bootstrap": pd.DataFrame({column: ["synthetic" if column == "task" else 1.] for column in [
                "task", "copy_fraction", "point_delta", "ci_low", "ci_high", "fraction_positive",
                "n_patients", "n_episodes", "n_bootstrap",
            ]}),
            "history_coverage": pd.DataFrame({
                "task": ["synthetic"] * 3, "comparison": ["Full4096"] * 3,
                "history_metric": ["earliest_retained_days_before_prediction", "final_seq_len", "n_backfill_events_added"],
            }),
        }
        captured = {}

        def capture(frame, stem):
            captured[stem.name] = frame
            return stem.with_suffix(".csv"), stem.with_suffix(".md")

        ns["write_table"] = capture
        with mock.patch.object(pd, "read_csv", side_effect=lambda key: frames[key].copy()):
            ns["create_tables"]({key: key for key in frames}, self.root)
        self.assertIn("n_backfill_events_added", set(captured["table_4_history_coverage"].history_metric))


class WrapperContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        cls.bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
        if not cls.bash:
            raise unittest.SkipTest("Bash is unavailable")

    def run_prefix(self, script, boundary, suffix, extra_env=None):
        source = (ROOT / "scripts" / script).read_text(encoding="utf-8")
        prefix = source.split(boundary, 1)[0]
        if prefix == source:
            self.fail("Wrapper prefix boundary is missing")
        prefix = prefix.replace("mkdir -p logs reproducibility", ":")
        env = os.environ.copy()
        for name in ["RUN_TAG", "OUTPUT_DIR", "EHRSHOT_S3_BASE", "CLEARML_PROJECT", "CLEARML_OUTPUT_URI", "REPRO_ENABLE_CLEARML"]:
            env.pop(name, None)
        env.update({"PROJECT_ROOT": ROOT.as_posix(), **(extra_env or {})})
        result = subprocess.run([self.bash, "-c", prefix + "\n" + suffix], env=env,
                                capture_output=True, text=True, encoding="utf-8", timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_training_defaults_and_historical_aliases_match_analysis_tags(self):
        configs = {
            "state_or_space_core_4096_runs.json": "core_4096_wide",
            "state_or_space_context_16384_runs.json": "context_16384_wide",
            "state_or_space_icu_gap_extra_runs.json": "icu_gap_extra_30_180_wide",
            "state_or_space_additional_seeds_45_46_runs.json": "additional_seeds_45_46_all_wide",
        }
        env = {"CLEARML_PROJECT": "synthetic", "CLEARML_OUTPUT_URI": "s3://storage.invalid/synthetic",
               "EHRSHOT_S3_BASE": "s3://storage.invalid/synthetic", "GPU_QUEUE": "synthetic"}
        suffix = 'printf "%s\\n%s\\n%s\\n" "$RUN_TAG" "$OUTPUT_DIR" "$RESULTS_S3_PREFIX"'
        for config, expected in configs.items():
            for alias in [None, expected[:-5], expected]:
                overrides = {**env, "RUN_CONFIG": "configs/" + config}
                if alias is not None:
                    overrides["RUN_TAG"] = alias
                with self.subTest(config=config, alias=alias):
                    lines = self.run_prefix("03_run_training_clearml.sh", "\nmkdir -p ", suffix, overrides).splitlines()
                    self.assertEqual(lines, [expected, "ehrshot_state_or_space_final_sequence_results/" + expected,
                                            "s3://storage.invalid/synthetic/ehrshot_state_or_space_final_sequence_results/" + expected])
        analysis = (ROOT / "scripts/04_run_analysis_clearml.sh").read_text(encoding="utf-8")
        self.assertIn("PREDICTION_RUN_TAGS:-" + ",".join(configs.values()), analysis)
        self.assertIn('"$PREDICTION_RUN_TAGS"', analysis)

    def test_local_package_skip_upload_needs_no_clearml_or_storage_environment(self):
        output = self.run_prefix("07_prepare_reproducibility_package.sh", "\nexport CLEARML_TASK_NO_REUSE",
                                 'printf "%s\\n" "${ARGS[@]}"',
                                 {"REPRO_SOURCE_ROOT": "synthetic_local_source", "SKIP_REPRO_UPLOAD": "1"})
        arguments = output.splitlines()
        self.assertIn("--source-root", arguments)
        self.assertIn("--skip-upload", arguments)
        for flag in ["--enable-clearml", "--clearml-upload-artifacts", "--output-s3-prefix"]:
            self.assertNotIn(flag, arguments)

    def test_bash_syntax_for_authorized_wrappers(self):
        for script in ["03_run_training_clearml.sh", "04_run_analysis_clearml.sh", "07_prepare_reproducibility_package.sh"]:
            result = subprocess.run([self.bash, "-n", (ROOT / "scripts" / script).as_posix()],
                                    capture_output=True, text=True, encoding="utf-8", timeout=20)
            with self.subTest(script=script):
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
