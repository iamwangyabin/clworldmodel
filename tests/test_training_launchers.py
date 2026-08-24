"""CPU-only contracts for reproducible continual-training launchers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class TrainingLauncherTests(unittest.TestCase):
    def _arrow_dry_run(self, *extra_args: str) -> dict:
        with TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "arrow_run"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_arrow_ar50_atari.py",
                    "--seed",
                    "0",
                    "--output-dir",
                    str(output_dir),
                    "--dry-run",
                    *extra_args,
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            launch = json.loads(result.stdout.split("\ncommand:", maxsplit=1)[0])
            return launch

    def _native_r2_dry_run(self, *extra_args: str) -> dict:
        with TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "r2dreamer_arrow_run"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_r2dreamer_arrow_atari.py",
                    "--seed",
                    "0",
                    "--output-dir",
                    str(output_dir),
                    "--dry-run",
                    *extra_args,
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            return json.loads(result.stdout.split("\ncommand:", maxsplit=1)[0])

    def _karrow_dry_run(
        self,
        *extra_args: str,
        script: str = "scripts/run_karrow_ar50_atari.py",
    ) -> tuple[dict, Path]:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        model_dir = root / "dinov3"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps({"hidden_size": 384, "patch_size": 16}),
            encoding="utf-8",
        )
        (model_dir / "model.safetensors").write_bytes(b"test-weights")
        output_dir = root / "karrow_run"
        result = subprocess.run(
            [
                sys.executable,
                script,
                "--seed",
                "0",
                "--dinov3-model-path",
                str(model_dir),
                "--output-dir",
                str(output_dir),
                "--dry-run",
                *extra_args,
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        launch = json.loads(result.stdout.split("\ncommand:", maxsplit=1)[0])
        return launch, output_dir

    def test_arrow_dry_run_records_complete_analysis_snapshot_contract(self) -> None:
        launch = self._arrow_dry_run()
        output_dir = Path(launch["output_dir"])
        self.assertEqual(launch["method"], "ARROW-50")
        self.assertEqual(launch["role"], "primary-method")
        self.assertEqual(launch["fifo_slots"], 512)
        self.assertEqual(launch["ltdm_slots"], 512)
        self.assertEqual(launch["replay_buffer_selection"], {"fifo": 0.5, "ltdm": 0.5})
        determinism = launch["determinism"]
        self.assertEqual(determinism["python_random_seed"], 123456789)
        self.assertEqual(determinism["replay_buffer_selection_rng"], "python_random")
        self.assertTrue(determinism["environment_reset_seeded"])
        self.assertTrue(determinism["action_space_seeded"])
        self.assertTrue(determinism["evaluation_rng_state_restored"])
        self.assertNotEqual(
            determinism["environment_seed_streams"]["collection"], "global_numpy"
        )
        self.assertTrue(determinism["known_nondeterminism"])
        replay_storage = launch["replay_storage_budget"]
        self.assertEqual(replay_storage["dtype"], "float32")
        self.assertEqual(replay_storage["observation_bytes"], 25_769_803_776)
        self.assertEqual(replay_storage["allocated_tensor_bytes"], 25_813_843_968)
        self.assertEqual(replay_storage["actor_comparison_difference_bytes"], 0)
        self.assertEqual(launch["observation_objective"]["name"], "reconstruction")
        self.assertTrue(launch["observation_objective"]["decoder_enabled"])
        self.assertIsNone(launch["r2_dreamer_reference"])
        self.assertEqual(
            launch["analysis_snapshot_semantics"]["task_boundary_epochs"],
            [89, 179, 269, 359, 449, 539],
        )
        self.assertEqual(launch["analysis_snapshot_semantics"]["final_epoch"], 540)
        self.assertFalse(launch["analysis_snapshot_semantics"]["resumable"])

        command = launch["command"]
        self.assertNotIn("--observation-objective", command)
        self.assertNotIn("--evaluate-final", command)
        self.assertFalse(launch["final_evaluation"]["enabled"])
        self.assertEqual(command[command.index("--arrow-replay-ratio") + 1], "50-50")
        self.assertEqual(command[command.index("--log-dir") + 1], str(output_dir.resolve()))
        self.assertEqual(
            command[command.index("--analysis-snapshot-dir") + 1],
            str((output_dir / "analysis_snapshots").resolve()),
        )

    def test_r2_dry_run_is_decoder_free_and_keeps_arrow_50_replay(self) -> None:
        launch = self._arrow_dry_run("--observation-objective", "r2")

        self.assertEqual(launch["method"], "ARROW-R2Rep-50")
        self.assertEqual(launch["role"], "representation-objective-ablation")
        self.assertEqual((launch["fifo_slots"], launch["ltdm_slots"]), (512, 512))
        self.assertEqual(launch["replay_buffer_selection"], {"fifo": 0.5, "ltdm": 0.5})
        objective = launch["observation_objective"]
        self.assertEqual(objective["name"], "r2")
        self.assertFalse(objective["decoder_enabled"])
        self.assertEqual(objective["barlow_loss_scale"], 0.05)
        self.assertEqual(objective["redundancy_scale"], 5e-4)
        self.assertEqual(objective["normalization_eps"], 1e-8)
        self.assertEqual(objective["target_gradient"], "stopped")
        self.assertEqual(
            launch["project_pythonpath_prepend"], str((ROOT / "src").resolve())
        )
        self.assertEqual(
            launch["r2_dreamer_reference"]["commit"],
            "546e4fab8146ea4b14e1d7726bbc1a8a1d50322f",
        )

        command = launch["command"]
        self.assertEqual(command[command.index("--observation-objective") + 1], "r2")
        self.assertEqual(command[command.index("--r2-barlow-loss-scale") + 1], "0.05")
        self.assertEqual(command[command.index("--r2-redundancy-scale") + 1], "0.0005")
        self.assertEqual(command[command.index("--r2-normalization-eps") + 1], "1e-08")

    def test_karrow_dry_run_records_matched_residual_and_cache_contract(self) -> None:
        launch, output_dir = self._karrow_dry_run(
            "--variant", "kan", "--task-prefix-length", "2"
        )

        self.assertEqual(launch["method"], "KARROW-FrozenCore-50-T2Pilot")
        self.assertEqual(launch["training_scope"]["epochs"], 180)
        self.assertEqual(launch["residuals"]["kind"], "kan")
        self.assertEqual(launch["residuals"]["matched_core_parameters"], 32_768)
        self.assertEqual(
            launch["shared_core"]["mode"], "freeze_after_first_task"
        )
        self.assertIn("reward", " ".join(launch["residuals"]["placements"]))
        self.assertIn("continue", " ".join(launch["residuals"]["placements"]))
        self.assertEqual(
            launch["replay"]["feature_cache"]["storage_bytes"], 402_653_184
        )
        self.assertEqual(
            launch["project_pythonpath_prepend"],
            os.pathsep.join((str(ROOT / "src"), str(ROOT))),
        )
        self.assertFalse(launch["observation"]["pixel_decoder"])
        self.assertFalse(output_dir.exists())

    def test_karrow_spatial_v2_uses_patch_posterior_and_accounts_cache(self) -> None:
        launch, output_dir = self._karrow_dry_run(
            "--variant",
            "kan",
            "--task-prefix-length",
            "2",
            "--visual-version",
            "v2",
        )

        self.assertEqual(launch["method"], "KARROW-SpatialFrozenCore-50-T2Pilot")
        self.assertEqual(launch["protocol"], "KARROW-SpatialFrozenCore-v2-Atari")
        self.assertEqual(launch["visual_version"], "v2")
        observation = launch["observation"]
        self.assertEqual(observation["feature_mode"], "patch_grid")
        self.assertEqual(observation["patch_pool_size"], 4)
        self.assertEqual(observation["patch_feature_dim"], 64)
        self.assertEqual(observation["patch_projection"], "task1_pca")
        self.assertEqual(observation["patch_projection_frames"], 512)
        self.assertEqual(
            observation["patch_projection_fit"],
            "closed-form PCA before the first world-model update",
        )
        self.assertEqual(observation["feature_dim"], 1_024)
        self.assertEqual(observation["prediction_state"], "posterior")
        self.assertEqual(
            observation["feature_loss_kind"],
            "batch_standardized_smooth_l1",
        )
        self.assertFalse(observation["first_and_reset_steps_masked"])
        self.assertEqual(
            launch["replay"]["feature_cache"]["storage_bytes"],
            1_073_741_824,
        )
        self.assertFalse(output_dir.exists())

    def test_karrow_v3_records_replay_consolidation_without_extra_updates(self) -> None:
        launch, output_dir = self._karrow_dry_run(
            "--variant",
            "kan",
            "--task-prefix-length",
            "2",
            "--visual-version",
            "v3",
        )

        self.assertEqual(launch["method"], "KARROW-ReplayConsolidated-50-T2Pilot")
        self.assertEqual(launch["protocol"], "KARROW-ReplayConsolidated-v3-Atari")
        self.assertEqual(launch["visual_version"], "v3")
        self.assertEqual(launch["role"], "primary-replay-consolidated-method-pilot")
        consolidation = launch["residual_consolidation"]
        self.assertEqual(consolidation["mode"], "replay_functional")
        self.assertEqual(consolidation["replay_batches"], 16)
        self.assertEqual(consolidation["deterministic_imagination_horizon"], 8)
        self.assertEqual(consolidation["extra_environment_interactions"], 0)
        self.assertEqual(consolidation["extra_gradient_updates"], 0)
        self.assertTrue(consolidation["training_rng_restored_after_estimation"])
        self.assertTrue(consolidation["post_adam_parameter_delta_scaling"])
        self.assertEqual(
            consolidation["boundary_importance_accumulator_peak_bytes"],
            1_048_576,
        )
        self.assertEqual(
            consolidation["post_adam_delta_snapshot_peak_bytes"],
            786_432,
        )
        self.assertTrue(launch["residuals"]["coordinate_map_frozen_after_task_1"])
        self.assertEqual(
            launch["residuals"]["consolidation_state_storage_bytes"],
            3_145_832,
        )
        self.assertEqual(
            launch["shared_core"]["trainable_after_freeze"],
            ["Gaussian RBF coefficients in each residual"],
        )
        self.assertFalse(output_dir.exists())

    def test_karrow_v4_uses_input_aligned_residuals_from_task_1(self) -> None:
        launch, output_dir = self._karrow_dry_run(
            "--variant",
            "kan",
            "--task-prefix-length",
            "2",
            "--visual-version",
            "v4",
        )

        self.assertEqual(launch["method"], "KARROW-InputAligned-50-T2Pilot")
        self.assertEqual(launch["protocol"], "KARROW-InputAligned-v4-Atari")
        self.assertEqual(launch["visual_version"], "v4")
        residuals = launch["residuals"]
        self.assertEqual(residuals["input_mode"], "module_input")
        self.assertTrue(residuals["trained_from_task_1"])
        self.assertEqual(
            residuals["task_1_optimization"],
            "joint base-and-residual optimization",
        )
        self.assertIn("dynamics [z,a,h]", " ".join(residuals["placements"]))
        self.assertIn("actor [z,h]", " ".join(residuals["placements"]))
        self.assertEqual(launch["residual_consolidation"]["mode"], "none")
        self.assertTrue(launch["shared_core"]["task_1_base_trainable"])
        self.assertTrue(launch["shared_core"]["task_1_residual_trainable"])
        self.assertEqual(launch["shared_core"]["freeze_after_completed_task"], 1)
        self.assertFalse(output_dir.exists())

    def test_karrow_records_triton_libcuda_override(self) -> None:
        with TemporaryDirectory() as temporary:
            libcuda_dir = Path(temporary)
            (libcuda_dir / "libcuda.so").touch()
            with mock.patch.dict(
                os.environ,
                {"TRITON_LIBCUDA_PATH": str(libcuda_dir)},
            ):
                launch, _ = self._karrow_dry_run(
                    "--variant",
                    "kan",
                    "--visual-version",
                    "v4",
                )

        self.assertEqual(
            launch["environment"]["TRITON_LIBCUDA_PATH"],
            str(libcuda_dir.resolve()),
        )

    def test_moe_arrow_dry_run_records_task_routing_and_fixed_update_budgets(self) -> None:
        launch, output_dir = self._karrow_dry_run(
            "--task-prefix-length",
            "2",
            script="scripts/run_moe_arrow_atari.py",
        )

        self.assertEqual(launch["method"], "MoE-ARROW-50-T2Pilot")
        self.assertEqual(launch["protocol"], "MoE-ARROW-v1-Atari-TaskAware")
        self.assertEqual(launch["code_id"], "moe_arrow")
        self.assertTrue(launch["task_identity"]["exposed_to_agent"])
        self.assertEqual(launch["world_model"]["router"], "hard_task_id")
        self.assertEqual(launch["world_model"]["allocated_experts"], 6)
        self.assertEqual(
            launch["world_model"]["expert_modules"],
            ["recurrent_dynamics", "latent_prior", "reward_head", "continue_head"],
        )
        self.assertEqual(launch["actor_critic"]["topology"], "per_task_bank")
        self.assertEqual(launch["actor_critic"]["current_task_update_fraction"], 0.5)
        self.assertEqual(launch["observation"]["patch_projection"], "fixed_orthogonal")
        self.assertEqual(launch["observation"]["patch_projection_frames"], 0)
        self.assertFalse(launch["observation"]["pixel_decoder"])
        self.assertEqual(launch["residual_correction"], "none")
        self.assertEqual(launch["replay"]["storage_device"], "cpu")
        self.assertEqual(
            launch["training_scope"]["world_model_updates"], 180_000
        )
        self.assertEqual(
            launch["training_scope"]["actor_critic_updates"], 144_000
        )
        self.assertEqual(launch["extra_gradient_updates"], 0)
        self.assertFalse(output_dir.exists())

    def test_dino_fullbank_dry_run_records_the_corrected_protocol(self) -> None:
        launch, output_dir = self._karrow_dry_run(
            "--task-prefix-length",
            "2",
            "--method",
            "dino-fullbank",
            script="scripts/run_moe_arrow_atari.py",
        )

        self.assertEqual(launch["method"], "DINO-FullBank-ARROW-50-T2Pilot")
        self.assertEqual(
            launch["protocol"], "DINO-FullBank-ARROW-v2-Atari-TaskAware"
        )
        self.assertEqual(launch["code_id"], "dino_fullbank_arrow")
        self.assertEqual(
            launch["world_model"]["expert_modules"],
            [
                "posterior_representation",
                "recurrent_dynamics",
                "latent_prior",
                "feature_predictor",
                "reward_head",
                "continue_head",
            ],
        )
        self.assertEqual(
            launch["world_model"]["shared_modules"], ["frozen DINOv3 encoder"]
        )
        self.assertEqual(
            launch["world_model"]["new_task_initialization"],
            "copy previous complete world-model expert once",
        )
        self.assertEqual(
            launch["actor_critic"]["new_task_initialization"], "fresh independent weights"
        )
        self.assertEqual(launch["actor_critic"]["current_task_update_fraction"], 1.0)
        self.assertEqual(launch["actor_critic"]["old_task_allocation"], "zero")
        self.assertEqual(
            launch["observation"]["objective"],
            "current posterior reconstruction of stopped spatial features",
        )
        self.assertEqual(
            launch["observation"]["feature_loss"],
            "batch_standardized_smooth_l1",
        )
        self.assertEqual(launch["collection"]["new_task_first_epoch_policy"], "random")
        self.assertEqual(launch["extra_gradient_updates"], 0)
        self.assertFalse(output_dir.exists())

    def test_dino_patchbank_dry_run_records_full_patches_and_pixel_decoder(self) -> None:
        launch, output_dir = self._karrow_dry_run(
            "--task-prefix-length",
            "1",
            "--method",
            "dino-patchbank",
            script="scripts/run_moe_arrow_atari.py",
        )

        self.assertEqual(launch["method"], "DINO-PatchBank-ARROW-50-T1Pilot")
        self.assertEqual(
            launch["protocol"], "DINO-PatchBank-ARROW-v3-Atari-TaskAware"
        )
        self.assertEqual(launch["code_id"], "dino_patchbank_arrow")
        self.assertEqual(
            launch["world_model"]["expert_modules"],
            [
                "posterior_representation",
                "recurrent_dynamics",
                "latent_prior",
                "pixel_decoder",
                "reward_head",
                "continue_head",
            ],
        )
        self.assertTrue(launch["world_model"]["pixel_decoder"])
        observation = launch["observation"]
        self.assertEqual(observation["patch_pool_size"], 16)
        self.assertEqual(observation["patch_feature_dim"], 384)
        self.assertEqual(observation["patch_projection"], "none")
        self.assertEqual(observation["feature_dim"], 98_304)
        self.assertEqual(observation["posterior_embedding_dim"], 98_304)
        self.assertEqual(
            observation["posterior_parameters_per_task"], 151_784_960
        )
        self.assertEqual(observation["patch_adapter"]["kind"], "none")
        self.assertEqual(
            observation["patch_adapter"]["trainable_parameters"], 0
        )
        self.assertEqual(observation["replay_feature_mode"], "on_the_fly")
        self.assertEqual(observation["feature_loss"], "not_applicable")
        self.assertTrue(observation["pixel_decoder"])
        self.assertEqual(
            launch["replay"]["feature_cache"]["storage_bytes"],
            0,
        )
        self.assertEqual(
            launch["replay"]["storage_device"],
            "cpu_addressable_file_mmap",
        )
        self.assertEqual(
            launch["replay"]["feature_cache"]["storage_backend"],
            "none",
        )
        self.assertEqual(
            launch["replay"]["feature_cache"]["mode"],
            "on_the_fly_from_sampled_observations",
        )
        self.assertEqual(
            launch["replay"]["base_storage"]["observation_storage_backend"],
            "file_mmap",
        )
        self.assertEqual(
            launch["replay"]["base_storage"]["anonymous_cpu_tensor_bytes"],
            44_040_192,
        )
        self.assertEqual(launch["actor_critic"]["current_task_update_fraction"], 1.0)
        self.assertFalse(output_dir.exists())

    def test_dino_convbank_dry_run_records_shared_4096_adapter(self) -> None:
        launch, output_dir = self._karrow_dry_run(
            "--task-prefix-length",
            "1",
            "--method",
            "dino-convbank",
            script="scripts/run_moe_arrow_atari.py",
        )

        self.assertEqual(
            launch["method"],
            "DINO-ConvBank-ARROW-50-BF16AMP-Uint8Replay-T1Pilot",
        )
        self.assertEqual(
            launch["protocol"],
            "DINO-ConvBank-ARROW-v4-BF16AMP-Uint8Replay-Atari-TaskAware",
        )
        self.assertEqual(launch["code_id"], "dino_convbank_arrow")
        self.assertEqual(launch["precision"]["profile"], "bf16-amp")
        self.assertTrue(launch["precision"]["autocast_enabled"])
        self.assertEqual(launch["precision"]["dinov3_execution_chunk_size"], 512)
        self.assertEqual(
            launch["world_model"]["shared_modules"],
            [
                "frozen DINOv3 encoder",
                "trainable shared DINO patch convolution adapter",
            ],
        )
        self.assertTrue(
            launch["world_model"]["shared_adapter_plastic_across_tasks"]
        )
        self.assertTrue(
            launch["world_model"]["old_task_expert_parameters_frozen"]
        )
        self.assertFalse(launch["world_model"]["old_task_parameters_frozen"])
        self.assertFalse(launch["world_model"]["old_task_functionally_isolated"])
        self.assertEqual(
            launch["world_model"]["expert_modules"],
            [
                "posterior_representation",
                "recurrent_dynamics",
                "latent_prior",
                "pixel_decoder",
                "reward_head",
                "continue_head",
            ],
        )

        observation = launch["observation"]
        self.assertEqual(observation["feature_dim"], 98_304)
        self.assertEqual(observation["posterior_embedding_dim"], 4_096)
        self.assertEqual(observation["posterior_parameters_per_task"], 7_081_472)
        self.assertEqual(
            observation["unadapted_posterior_parameters_per_task"],
            151_784_960,
        )
        adapter = observation["patch_adapter"]
        self.assertEqual(adapter["kind"], "conv_3x3_stride2")
        self.assertEqual(adapter["input_layout"], [16, 16, 384])
        self.assertEqual(adapter["output_layout"], [8, 8, 64])
        self.assertEqual(adapter["kernel_size"], 3)
        self.assertEqual(adapter["stride"], 2)
        self.assertEqual(adapter["padding"], 1)
        self.assertEqual(adapter["normalization"], "channel_layer_norm_eps_1e-3")
        self.assertEqual(adapter["activation"], "silu")
        self.assertEqual(adapter["output_features"], 4_096)
        self.assertEqual(adapter["trainable_parameters"], 221_376)
        self.assertTrue(adapter["shared_across_tasks"])
        self.assertTrue(adapter["trainable"])
        self.assertEqual(
            launch["replay"]["storage_device"],
            "cpu_addressable_file_mmap",
        )
        self.assertEqual(launch["replay"]["feature_cache"]["storage_bytes"], 0)
        self.assertEqual(
            launch["replay"]["feature_cache"]["quantization_dtype"], "bfloat16"
        )
        self.assertEqual(
            launch["replay"]["feature_cache"]["consumer_dtype"], "bfloat16"
        )
        base_storage = launch["replay"]["base_storage"]
        self.assertEqual(base_storage["dtype"], "uint8")
        self.assertEqual(base_storage["observation_bytes"], 6_442_450_944)
        self.assertEqual(base_storage["allocated_tensor_bytes"], 6_486_491_136)
        self.assertEqual(launch["replay"]["sampled_observation_dtype"], "float32")
        self.assertIsNone(launch["hyperparameter_tuning"])
        self.assertFalse(output_dir.exists())

    def test_dino_convbank_task1_tuning_profiles_are_named_and_budget_matched(
        self,
    ) -> None:
        profiles = {
            "aclr5e5": {
                "protocol_suffix": "Task1AcLR5e5",
                "ac_lr": 5e-5,
                "entropy_scale": 3e-4,
            },
            "aclr5e5-ent1e4": {
                "protocol_suffix": "Task1AcLR5e5Ent1e4",
                "ac_lr": 5e-5,
                "entropy_scale": 1e-4,
            },
        }

        for profile, expected in profiles.items():
            with self.subTest(profile=profile):
                launch, output_dir = self._karrow_dry_run(
                    "--task-prefix-length",
                    "1",
                    "--method",
                    "dino-convbank",
                    "--devices",
                    "2",
                    "--task1-tuning-profile",
                    profile,
                    script="scripts/run_moe_arrow_atari.py",
                )

                suffix = expected["protocol_suffix"]
                self.assertEqual(
                    launch["method"],
                    "DINO-ConvBank-ARROW-50-BF16AMP-Uint8Replay-DP2-"
                    f"{suffix}-T1Pilot",
                )
                self.assertEqual(
                    launch["protocol"],
                    "DINO-ConvBank-ARROW-v4-BF16AMP-Uint8Replay-DP2-"
                    f"{suffix}-Atari-TaskAware",
                )
                tuning = launch["hyperparameter_tuning"]
                self.assertEqual(tuning["profile"], profile)
                self.assertEqual(
                    tuning["config_overrides"],
                    {
                        "ac_lr": expected["ac_lr"],
                        "ac_entropy_scale": expected["entropy_scale"],
                    },
                )
                self.assertTrue(tuning["fixed_data_and_update_budgets"])
                self.assertEqual(
                    tuning["acquisition_gate"],
                    {
                        "task": "ALE/MsPacman-v5",
                        "after_completed_epochs": 90,
                        "rollouts": 16,
                        "metric": "raw_return_mean",
                        "minimum": 2000.0,
                        "use_intermediate_peak": False,
                    },
                )
                self.assertEqual(
                    launch["actor_critic"]["learning_rate"], expected["ac_lr"]
                )
                self.assertEqual(
                    launch["actor_critic"]["entropy_scale"],
                    expected["entropy_scale"],
                )
                self.assertEqual(
                    launch["training_scope"]["world_model_updates"], 90_000
                )
                self.assertEqual(
                    launch["training_scope"]["actor_critic_updates"], 72_000
                )
                self.assertFalse(output_dir.exists())

    def test_dino_convbank_task1_tuning_requires_one_task_scope(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "dinov3"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                json.dumps({"hidden_size": 384, "patch_size": 16}),
                encoding="utf-8",
            )
            (model_dir / "model.safetensors").write_bytes(b"test-weights")
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_moe_arrow_atari.py",
                    "--seed",
                    "0",
                    "--dinov3-model-path",
                    str(model_dir),
                    "--method",
                    "dino-convbank",
                    "--task1-tuning-profile",
                    "aclr5e5",
                    "--dry-run",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "requires --task-prefix-length 1",
            result.stderr,
        )

    def test_dino_convbank_explicit_bfloat16_records_required_profile(self) -> None:
        launch, output_dir = self._karrow_dry_run(
            "--task-prefix-length",
            "1",
            "--method",
            "dino-convbank",
            "--precision-profile",
            "bf16-amp",
            script="scripts/run_moe_arrow_atari.py",
        )

        self.assertEqual(
            launch["method"],
            "DINO-ConvBank-ARROW-50-BF16AMP-Uint8Replay-T1Pilot",
        )
        self.assertEqual(
            launch["protocol"],
            "DINO-ConvBank-ARROW-v4-BF16AMP-Uint8Replay-Atari-TaskAware",
        )
        precision = launch["precision"]
        self.assertEqual(precision["profile"], "bf16-amp")
        self.assertTrue(precision["autocast_enabled"])
        self.assertEqual(precision["compute_dtype"], "bfloat16")
        self.assertEqual(precision["parameter_dtype"], "float32")
        self.assertEqual(precision["optimizer_state_dtype"], "float32")
        self.assertFalse(precision["gradient_scaler"])
        self.assertEqual(precision["sensitive_math_dtype"], "float32")
        self.assertEqual(precision["dinov3_execution_chunk_size"], 512)
        self.assertEqual(
            precision["world_model_optimization_batch"],
            {
                "time": 32,
                "sequences": 16,
                "frames": 512,
                "unchanged": True,
            },
        )
        self.assertEqual(launch["training_scope"]["world_model_updates"], 90_000)
        self.assertEqual(launch["training_scope"]["actor_critic_updates"], 72_000)
        self.assertTrue(precision["optimizer_update_budgets_unchanged"])
        feature_source = launch["replay"]["feature_cache"]
        self.assertEqual(feature_source["quantization_dtype"], "bfloat16")
        self.assertEqual(feature_source["consumer_dtype"], "bfloat16")
        self.assertEqual(
            feature_source["quantization_semantics"],
            "encoder output retained without a dtype round trip",
        )
        self.assertFalse(output_dir.exists())

    def test_dino_convbank_two_and_four_gpu_ddp_keep_global_budgets(self) -> None:
        expected_local_batches = {
            2: {"world_model": 8, "actor": 64},
            4: {"world_model": 4, "actor": 32},
        }

        for devices, local in expected_local_batches.items():
            with self.subTest(devices=devices):
                launch, output_dir = self._karrow_dry_run(
                    "--task-prefix-length",
                    "1",
                    "--method",
                    "dino-convbank",
                    "--devices",
                    str(devices),
                    script="scripts/run_moe_arrow_atari.py",
                )

                self.assertEqual(
                    launch["method"],
                    "DINO-ConvBank-ARROW-50-BF16AMP-Uint8Replay-"
                    f"DP{devices}-T1Pilot",
                )
                self.assertEqual(
                    launch["protocol"],
                    "DINO-ConvBank-ARROW-v4-BF16AMP-Uint8Replay-"
                    f"DP{devices}-Atari-TaskAware",
                )
                execution = launch["distributed_execution"]
                self.assertTrue(execution["enabled"])
                self.assertEqual(execution["world_size"], devices)
                self.assertEqual(execution["backend"], "nccl")
                self.assertEqual(execution["replay_owner"], "rank_0")
                self.assertEqual(execution["collection"], "rank_0_only")
                self.assertEqual(
                    execution["world_model_sequences"],
                    {"global": 16, "per_rank": local["world_model"]},
                )
                self.assertEqual(
                    execution["actor_context_sequences"],
                    {"global": 128, "per_rank": local["actor"]},
                )
                self.assertEqual(
                    launch["precision"]["world_model_optimization_batch"][
                        "frames"
                    ],
                    512,
                )
                self.assertEqual(
                    launch["training_scope"]["world_model_updates"], 90_000
                )
                self.assertEqual(
                    launch["training_scope"]["actor_critic_updates"], 72_000
                )
                command = launch["command"]
                self.assertEqual(
                    command[1:6],
                    [
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        "--nproc-per-node",
                        str(devices),
                    ],
                )
                self.assertIn("Code/ARROW_and_DV3/Atari/train.py", command)
                self.assertFalse(output_dir.exists())

    def test_dino_convbank_rejects_fp32_execution(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "dinov3"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                json.dumps({"hidden_size": 384, "patch_size": 16}),
                encoding="utf-8",
            )
            (model_dir / "model.safetensors").write_bytes(b"test-weights")
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_moe_arrow_atari.py",
                    "--seed",
                    "0",
                    "--dinov3-model-path",
                    str(model_dir),
                    "--method",
                    "dino-convbank",
                    "--precision-profile",
                    "fp32-tf32",
                    "--dry-run",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("dino-convbank requires", result.stderr)

    def test_kan_actor_dry_run_changes_only_actor_and_keeps_arrow_50_replay(self) -> None:
        launch = self._arrow_dry_run("--actor-network", "relu_kan")

        self.assertEqual(launch["method"], "ARROW-KANActor-50")
        self.assertEqual(launch["role"], "actor-architecture-ablation")
        self.assertEqual((launch["fifo_slots"], launch["ltdm_slots"]), (512, 512))
        self.assertEqual(launch["replay_buffer_selection"], {"fifo": 0.5, "ltdm": 0.5})
        self.assertEqual(launch["observation_objective"]["name"], "reconstruction")

        actor = launch["actor"]
        self.assertEqual(actor["network"], "relu_kan")
        self.assertEqual(actor["critic_network"], "mlp")
        self.assertEqual(actor["input_features"], 1536)
        self.assertEqual(actor["recurrent_features"], 512)
        self.assertEqual(actor["kan_hidden_features"], 64)
        self.assertEqual(actor["kan_basis_count"], 8)
        self.assertEqual(actor["kan_input_range"], [0.0, 1.0])
        self.assertFalse(actor["kan_grid_trainable"])
        self.assertEqual(actor["trainable_parameters"], 795730)
        self.assertEqual(
            launch["project_pythonpath_prepend"], str((ROOT / "src").resolve())
        )

        command = launch["command"]
        self.assertEqual(command[command.index("--actor-network") + 1], "relu_kan")
        self.assertEqual(command[command.index("--arrow-replay-ratio") + 1], "50-50")

    def test_kan_actor_cannot_be_mixed_with_r2_in_one_run(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_arrow_ar50_atari.py",
                "--actor-network",
                "relu_kan",
                "--observation-objective",
                "r2",
                "--dry-run",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be tested independently", result.stderr)

    def test_two_task_kan_pilot_stops_at_second_boundary(self) -> None:
        launch = self._arrow_dry_run(
            "--actor-network", "relu_kan", "--task-prefix-length", "2"
        )

        self.assertEqual(launch["method"], "ARROW-KANActor-50-T2Pilot")
        self.assertEqual(launch["role"], "actor-architecture-pilot")
        self.assertEqual(launch["training_scope"]["task_prefix_length"], 2)
        self.assertEqual(launch["training_scope"]["epochs"], 180)
        self.assertEqual(
            launch["training_scope"]["tasks"],
            ["ALE/MsPacman-v5", "ALE/Boxing-v5"],
        )
        self.assertEqual(
            launch["analysis_snapshot_semantics"]["task_boundary_epochs"], [89, 179]
        )
        self.assertEqual(launch["analysis_snapshot_semantics"]["final_epoch"], 179)
        self.assertTrue(
            launch["analysis_snapshot_semantics"][
                "final_coincides_with_task_boundary"
            ]
        )
        command = launch["command"]
        self.assertEqual(command[command.index("--epochs") + 1], "180")
        self.assertIn("--evaluate-final", command)
        self.assertEqual(command[command.index("--arrow-replay-ratio") + 1], "50-50")
        self.assertTrue(launch["final_evaluation"]["enabled"])
        self.assertEqual(launch["final_evaluation"]["rollouts_per_task"], 16)
        self.assertFalse(launch["final_evaluation"]["enters_replay"])
        self.assertTrue(
            launch["final_evaluation"]["reports_raw_and_scaled_returns"]
        )

    def test_bounded_kan_t1_pilot_preserves_the_single_task_budget(self) -> None:
        launch = self._arrow_dry_run(
            "--actor-network", "relu_kan_bounded", "--task-prefix-length", "1"
        )

        self.assertEqual(launch["method"], "ARROW-KANActorBounded-50-T1TrainabilityPilot")
        self.assertEqual(launch["role"], "actor-trainability-pilot")
        self.assertEqual(launch["training_scope"]["task_prefix_length"], 1)
        self.assertEqual(launch["training_scope"]["epochs"], 90)
        self.assertEqual(launch["training_scope"]["tasks"], ["ALE/MsPacman-v5"])
        self.assertEqual(launch["analysis_snapshot_semantics"]["task_boundary_epochs"], [89])
        actor = launch["actor"]
        self.assertEqual(actor["network"], "relu_kan_bounded")
        self.assertEqual(actor["kan_hidden_adapter"], "layer_norm_sigmoid")
        self.assertEqual(actor["kan_hidden_adapter_layer_norm_epsilon"], 1e-3)
        self.assertEqual(actor["trainable_parameters"], 795_858)

        command = launch["command"]
        self.assertEqual(command[command.index("--actor-network") + 1], "relu_kan_bounded")
        self.assertEqual(command[command.index("--epochs") + 1], "90")
        self.assertIn("--evaluate-final", command)

    def test_bounded_kan_t1_extension_moves_the_only_task_boundary(self) -> None:
        launch = self._arrow_dry_run(
            "--actor-network",
            "relu_kan_bounded",
            "--task-prefix-length",
            "1",
            "--task-duration-epochs",
            "180",
        )

        self.assertEqual(
            launch["method"], "ARROW-KANActorBounded-50-T1-180EpochTrainabilityPilot"
        )
        self.assertEqual(launch["role"], "actor-trainability-budget-extension")
        scope = launch["training_scope"]
        self.assertEqual(scope["task_prefix_length"], 1)
        self.assertEqual(scope["epochs"], 180)
        self.assertEqual(scope["task_duration_epochs"], 180)
        self.assertEqual(scope["baseline_task_duration_epochs"], 90)
        self.assertEqual(scope["task_duration_epoch_override"], 180)
        self.assertEqual(scope["tasks"], ["ALE/MsPacman-v5"])
        self.assertEqual(
            launch["config_overrides"],
            {"epochs": 180, "esc.kwargs.swap_sched": 180},
        )
        self.assertTrue(
            launch["resolved_training_config"].endswith("resolved_training_config.json")
        )
        self.assertEqual(
            launch["analysis_snapshot_semantics"]["task_boundary_epochs"], [179]
        )
        self.assertEqual(launch["analysis_snapshot_semantics"]["final_epoch"], 179)

        command = launch["command"]
        self.assertEqual(command[command.index("--epochs") + 1], "180")
        self.assertTrue(
            command[command.index("--config") + 1].endswith(
                "resolved_training_config.json"
            )
        )

    def test_adaptive_kan_t1_extension_records_trainable_anchor_protocol(self) -> None:
        launch = self._arrow_dry_run(
            "--actor-network",
            "relu_kan_adaptive",
            "--task-prefix-length",
            "1",
            "--task-duration-epochs",
            "180",
        )

        self.assertEqual(
            launch["method"], "ARROW-KANActorAdaptive-50-T1-180EpochTrainabilityPilot"
        )
        self.assertEqual(launch["role"], "actor-trainability-budget-extension")
        self.assertEqual(launch["training_scope"]["epochs"], 180)
        self.assertEqual(launch["training_scope"]["task_duration_epochs"], 180)
        self.assertEqual(
            launch["training_scope"]["tasks"], ["ALE/MsPacman-v5"]
        )
        self.assertEqual(
            launch["config_overrides"],
            {
                "actor_network": "relu_kan_adaptive",
                "actor_kan_trainable_grid": True,
                "epochs": 180,
                "esc.kwargs.swap_sched": 180,
            },
        )
        self.assertTrue(
            launch["resolved_training_config"].endswith("resolved_training_config.json")
        )
        self.assertEqual(
            launch["analysis_snapshot_semantics"]["task_boundary_epochs"], [179]
        )

        actor = launch["actor"]
        self.assertEqual(actor["network"], "relu_kan_adaptive")
        self.assertTrue(actor["kan_grid_trainable"])
        self.assertEqual(
            actor["kan_anchor_parameterization"], "per_input_start_softplus_width"
        )
        self.assertEqual(actor["kan_anchor_parameters"], 25_600)
        self.assertEqual(actor["trainable_parameters"], 821_458)

        command = launch["command"]
        self.assertEqual(
            command[command.index("--actor-network") + 1], "relu_kan_adaptive"
        )
        self.assertIn("--actor-kan-trainable-grid", command)
        self.assertEqual(command[command.index("--epochs") + 1], "180")
        self.assertTrue(
            command[command.index("--config") + 1].endswith(
                "resolved_training_config.json"
            )
        )

    def test_task_duration_extension_requires_bounded_interface_kan_t1(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_arrow_ar50_atari.py",
                "--actor-network",
                "relu_kan_bounded",
                "--task-prefix-length",
                "2",
                "--task-duration-epochs",
                "180",
                "--dry-run",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires --task-prefix-length 1", result.stderr)

    def test_fastkan_ac_pilot_records_architecture_optimizer_and_step_mapping(self) -> None:
        launch = self._arrow_dry_run(
            "--actor-network",
            "fast_kan_ac",
            "--task-prefix-length",
            "1",
            "--task-duration-epochs",
            "68",
        )

        self.assertEqual(
            launch["method"],
            "ARROW-FastKANAC-KDAligned-50-T1-68EpochTrainabilityPilot",
        )
        self.assertEqual(launch["role"], "actor-critic-kan-dreamer-aligned-pilot")
        self.assertEqual(launch["training_scope"]["epochs"], 68)
        self.assertEqual(launch["training_scope"]["agent_decisions"], 1_114_112)
        self.assertEqual(
            launch["training_scope"]["kan_dreamer_target_environment_steps"],
            1_100_000,
        )
        self.assertEqual((launch["fifo_slots"], launch["ltdm_slots"]), (512, 512))

        actor = launch["actor"]
        self.assertEqual(actor["network"], "fast_kan_ac")
        self.assertEqual(actor["critic_network"], "fast_kan")
        self.assertEqual(actor["kan_hidden_features"], 34)
        self.assertEqual(actor["kan_hidden_layers"], 3)
        self.assertEqual(actor["kan_grid_size"], 8)
        self.assertEqual(actor["kan_input_range"], [-2.0, 2.0])
        self.assertFalse(actor["kan_grid_trainable"])
        self.assertEqual(actor["combined_trainable_parameters"], 1_068_939)

        training = launch["actor_critic_training"]
        self.assertEqual(training["optimizer"], "laprop")
        self.assertEqual(training["learning_rate"], 4e-5)
        self.assertEqual(training["gradient_clipping"]["coefficient"], 0.3)
        self.assertEqual(training["imagination_horizon"], 15)
        self.assertEqual(training["critic_ema_decay"], 0.98)
        self.assertEqual(training["critic_replay_loss_scale"], 0.0)
        self.assertIsNotNone(training["critic_replay_loss_deviation"])
        self.assertEqual(training["imagination_value_target"], "online_critic")
        self.assertEqual(
            training["terminal_bootstrap_state"],
            "legacy_last_pre_transition_state",
        )

        overrides = launch["config_overrides"]
        self.assertEqual(overrides["actor_network"], "fast_kan_ac")
        self.assertEqual(overrides["ac_optimizer"], "laprop")
        self.assertEqual(overrides["epochs"], 68)
        self.assertEqual(overrides["esc.kwargs.swap_sched"], 68)
        command = launch["command"]
        self.assertEqual(command[command.index("--actor-network") + 1], "fast_kan_ac")

    def test_parameter_matched_fastkan_extension_records_repval_and_midpoint(self) -> None:
        launch = self._arrow_dry_run(
            "--actor-network",
            "fast_kan_ac_param_matched",
            "--task-prefix-length",
            "1",
            "--task-duration-epochs",
            "136",
        )

        self.assertEqual(
            launch["method"],
            "ARROW-FastKANAC-ParamMatchedRepVal-50-T1-136EpochTrainabilityPilot",
        )
        self.assertEqual(
            launch["role"],
            "actor-critic-param-matched-replay-value-budget-extension",
        )
        self.assertEqual(launch["training_scope"]["epochs"], 136)
        self.assertEqual(launch["training_scope"]["agent_decisions"], 2_228_224)
        self.assertEqual(launch["training_scope"]["midpoint_completed_epochs"], 68)
        self.assertEqual(
            launch["analysis_snapshot_semantics"]["milestone_completed_epochs"],
            [68],
        )

        actor = launch["actor"]
        self.assertEqual(actor["network"], "fast_kan_ac_param_matched")
        self.assertEqual(actor["kan_hidden_features"], 53)
        self.assertEqual(actor["trainable_parameters"], 793_692)
        self.assertEqual(actor["critic_trainable_parameters"], 906_978)
        self.assertEqual(actor["combined_trainable_parameters"], 1_700_670)
        self.assertEqual(actor["combined_parameter_difference_from_mlp"], -14_291)

        training = launch["actor_critic_training"]
        self.assertEqual(training["critic_replay_loss_scale"], 0.3)
        self.assertIsNone(training["critic_replay_loss_deviation"])
        self.assertEqual(training["actor_advantage_baseline"], "online_critic")
        self.assertEqual(
            training["terminal_bootstrap_state"],
            "legacy_last_pre_transition_state",
        )
        self.assertIn(
            "same four posterior context frames",
            training["critic_replay_loss_semantics"],
        )
        self.assertEqual(
            training["dreamerv3_repval_reference"]["commit"],
            "e3f02248693a79dc8b0ebd62c93683888ddaccfe",
        )

        overrides = launch["config_overrides"]
        self.assertEqual(overrides["fastkan_hidden_features"], 53)
        self.assertEqual(overrides["ac_replay_critic_loss_scale"], 0.3)
        command = launch["command"]
        milestone_index = command.index("--milestone-completed-epoch")
        self.assertEqual(command[milestone_index + 1], "68")

    def test_stable_fastkan_uses_slow_targets_and_terminal_state_bootstrap(self) -> None:
        launch = self._arrow_dry_run(
            "--actor-network",
            "fast_kan_ac_stable",
            "--task-prefix-length",
            "1",
            "--task-duration-epochs",
            "90",
        )

        self.assertEqual(
            launch["method"],
            "ARROW-FastKANAC-StableTargets-50-T1-90EpochTrainabilityPilot",
        )
        self.assertEqual(
            launch["role"],
            "actor-critic-stable-target-correction-pilot",
        )
        self.assertEqual(launch["training_scope"]["epochs"], 90)
        self.assertEqual(launch["training_scope"]["agent_decisions"], 1_474_560)
        self.assertEqual(
            launch["analysis_snapshot_semantics"]["milestone_completed_epochs"],
            [],
        )

        actor = launch["actor"]
        self.assertEqual(actor["network"], "fast_kan_ac_stable")
        self.assertEqual(actor["critic_network"], "fast_kan")
        self.assertEqual(actor["kan_hidden_features"], 53)
        self.assertEqual(actor["combined_trainable_parameters"], 1_700_670)

        training = launch["actor_critic_training"]
        self.assertEqual(training["critic_replay_loss_scale"], 0.3)
        self.assertEqual(training["imagination_value_target"], "ema_slow_critic")
        self.assertEqual(training["actor_advantage_baseline"], "ema_slow_critic")
        self.assertEqual(
            training["terminal_bootstrap_state"],
            "post_transition_imagined_state",
        )

        overrides = launch["config_overrides"]
        self.assertTrue(overrides["ac_use_slow_critic_targets"])
        self.assertTrue(overrides["ac_corrected_imagination_bootstrap"])
        self.assertEqual(overrides["epochs"], 90)
        self.assertEqual(overrides["esc.kwargs.swap_sched"], 90)
        command = launch["command"]
        self.assertEqual(
            command[command.index("--actor-network") + 1],
            "fast_kan_ac_stable",
        )
        self.assertNotIn("--milestone-completed-epoch", command)

    def test_swanlab_mirroring_records_names_but_no_credential(self) -> None:
        launch = self._arrow_dry_run(
            "--actor-network",
            "fast_kan_ac_param_matched",
            "--task-prefix-length",
            "1",
            "--task-duration-epochs",
            "136",
            "--swanlab-project",
            "clworldmodel",
            "--swanlab-experiment-name",
            "fastkan-test",
        )
        logging = launch["metric_logging"]
        self.assertTrue(logging["swanlab_enabled"])
        self.assertEqual(logging["swanlab_project"], "clworldmodel")
        self.assertEqual(logging["swanlab_experiment_name"], "fastkan-test")
        self.assertNotIn("api_key", json.dumps(launch).lower())

    def test_native_r2_dry_run_uses_size12m_geometry_and_arrow_replay(self) -> None:
        launch = self._native_r2_dry_run()

        self.assertEqual(launch["method"], "R2Dreamer-ARROW-50")
        self.assertEqual(launch["role"], "native-r2dreamer-with-arrow-replay")
        self.assertEqual(launch["scope"], "single-task")
        self.assertEqual(launch["status_label"], "pilot")
        self.assertEqual(
            launch["upstream"]["r2dreamer"]["commit"],
            "546e4fab8146ea4b14e1d7726bbc1a8a1d50322f",
        )
        r2 = launch["r2dreamer"]
        self.assertFalse(r2["decoder_enabled"])
        self.assertEqual((r2["embedding_dim"], r2["rssm_feature_dim"]), (1024, 2560))
        self.assertEqual((r2["batch_size"], r2["batch_length"]), (16, 64))
        self.assertEqual(r2["flattened_barlow_samples"], 1024)
        self.assertEqual(r2["optimizer"], "LaProp")

        replay = launch["arrow_replay"]
        self.assertEqual((replay["fifo_slots"], replay["ltdm_slots"]), (512, 512))
        self.assertEqual(replay["buffer_selection"], {"fifo": 0.5, "ltdm": 0.5})
        self.assertEqual(replay["sample_context_steps"], 1)

        budget = launch["budget"]
        self.assertEqual(budget["task_count"], 1)
        self.assertEqual(budget["epochs"], 7)
        self.assertEqual(budget["nominal_world_model_updates_per_epoch"], 2_048)
        self.assertEqual(budget["native_train_ratio"], 128)
        self.assertEqual(budget["nominal_raw_frames_per_epoch"], 65_536)
        self.assertEqual(budget["single_task_target_raw_frames"], 410_000)
        self.assertEqual(budget["source_samples_per_epoch"], 512_000)
        self.assertEqual(budget["r2_samples_per_epoch"], 2_097_152)

        command = launch["command"]
        self.assertIn("--launcher-created-log-dir", command)
        self.assertEqual(command[command.index("--task-count") + 1], "1")
        self.assertEqual(command[command.index("--epochs") + 1], "7")
        self.assertEqual(command[command.index("--native-train-ratio") + 1], "128")

    def test_native_r2_smoke_allows_native_amp_scale_calibration(self) -> None:
        launch = self._native_r2_dry_run("--smoke")

        self.assertEqual(launch["status_label"], "smoke")
        self.assertEqual(launch["budget"]["epochs"], 1)
        self.assertEqual(launch["budget"]["nominal_world_model_updates_per_epoch"], 12)
        self.assertEqual(launch["r2dreamer"]["amp_initial_scale"], 65_536.0)
        command = launch["command"]
        self.assertEqual(
            command[command.index("--world-model-updates-per-epoch") + 1], "12"
        )
        self.assertIn("--require-optimizer-step", command)
        self.assertEqual(
            launch["smoke_checks"]["required_successful_optimizer_steps"], 1
        )

    def test_native_r2_full_single_task_matches_arrow_task_duration(self) -> None:
        launch = self._native_r2_dry_run("--scope", "single-task-full")

        self.assertEqual(launch["scope"], "single-task-full")
        self.assertEqual(launch["status_label"], "full-single-task-pilot")
        budget = launch["budget"]
        self.assertEqual(budget["task_count"], 1)
        self.assertEqual(budget["epochs"], 90)
        self.assertEqual(budget["source_task_switch_epochs"], 90)
        self.assertEqual(budget["single_task_target_raw_frames"], 5_898_240)
        self.assertEqual(budget["total_nominal_raw_frames"], 5_898_240)
        self.assertEqual(
            budget["total_nominal_r2_model_sample_transitions"], 188_743_680
        )

        command = launch["command"]
        self.assertEqual(command[command.index("--task-count") + 1], "1")
        self.assertEqual(command[command.index("--epochs") + 1], "90")


if __name__ == "__main__":
    unittest.main()
