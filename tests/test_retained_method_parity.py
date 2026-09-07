"""Fixed-input parity against the pre-retirement working tree; no environment runs."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/retained_method_parity.json"


def tensor_contracts() -> dict:
    import torch
    from wm import WorldModel

    torch.set_num_threads(1)

    class FixedEncoder(torch.nn.Module):
        output_size = 4096

        def __init__(self):
            super().__init__()
            self.offset = torch.nn.Parameter(torch.zeros(self.output_size))

        def forward(self, images):
            return self.offset.unsqueeze(0).expand(len(images), -1)

    def digest(state):
        result = hashlib.sha256()
        for name, value in state.items():
            result.update(name.encode())
            result.update(str(tuple(value.shape)).encode())
            result.update(value.detach().cpu().contiguous().numpy().tobytes())
        return result.hexdigest()

    result = {}
    for name in ("baseline", "d_family"):
        torch.manual_seed(20260905)
        kwargs = {}
        if name == "d_family":
            kwargs = dict(
                num_task_experts=3, task_shared_prediction_heads=True,
                evolving_shared_core=True, task_projected_image_encoder=True,
                task_symmetric_image_projectors=True, task_projector_bottleneck_features=64,
                task_mechanism_bank=True, task_mechanism_reuse=True,
                task_mechanism_recurrent_width=8, task_mechanism_representation_width=8,
                task_mechanism_transition_width=8, task_mechanism_num_atoms=4,
                task_mechanism_parameterization="adaptive_dense_width",
                task_symmetric_mechanisms=True, image_embedder=FixedEncoder(),
            )
        model = WorldModel(3, (2, 3), 4, 8, cnn_depth=4, mlp_features=8, **kwargs)
        initial_digest = digest(model.state_dict())
        task_id = None
        if name == "d_family":
            model.initialize_task_expert(1, 0)
            model.activate_task_expert(1)
            task_id = 1
            # Exercise nonzero private corrections and old-atom reuse, not only
            # the zero-effect initialization shared by all proposed adapters.
            with torch.no_grad():
                for bank in model.rssm.mechanism_banks().values():
                    for index, mechanism in enumerate(bank.mechanisms):
                        mechanism.up.weight.fill_(0.001 * (index + 1))
                        mechanism.up.bias.fill_(0.002 * (index + 1))
        actions = torch.nn.functional.one_hot(torch.tensor([[0, 1], [2, 3], [1, 0]]), 4).float()
        images = torch.linspace(0, 1, 3 * 2 * 3 * 64 * 64).reshape(3, 2, 3, 64, 64)
        rewards = torch.tensor([0., 1., -1., 2., 0., 1.]).reshape(3, 2, 1)
        continues = torch.ones(3, 2, 1)
        resets = torch.zeros(3, 2, 1)
        resets[0] = 1
        loss, metrics = model.compute_loss(actions, images, rewards, continues, resets, task_id=task_id)
        loss.backward()
        result[name] = {
            "initial_state_sha256": initial_digest,
            "parameters": sum(p.numel() for p in model.parameters()),
            "loss": float(loss.detach()),
            "gradient_norms": {
                key: float(parameter.grad.norm())
                for key, parameter in model.named_parameters()
                if parameter.grad is not None
            },
        }
    return result


class RetainedMethodParityTests(unittest.TestCase):
    def test_initialization_losses_and_gradients_match_pre_retirement(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("requires PyTorch")
        sys.path.insert(0, str(ROOT / "src"))
        sys.path.insert(0, str(ROOT / "third_party/arrow/Code/ARROW_and_DV3/Atari"))
        expected = json.loads(FIXTURE.read_text())["contracts"]
        with torch.random.fork_rng(devices=[]):
            actual = tensor_contracts()
        for name in ("baseline", "d_family"):
            record = expected[name]
            self.assertEqual(actual[name]["initial_state_sha256"], record["initial_state_sha256"], name)
        for name in ("baseline", "d_family"):
            self.assertEqual(actual[name]["parameters"], expected[name]["parameters"])
            torch.testing.assert_close(torch.tensor(actual[name]["loss"]), torch.tensor(expected[name]["loss"]), rtol=1e-6, atol=1e-6)
            self.assertEqual(actual[name]["gradient_norms"].keys(), expected[name]["gradient_norms"].keys())
            for key, value in expected[name]["gradient_norms"].items():
                torch.testing.assert_close(torch.tensor(actual[name]["gradient_norms"][key]), torch.tensor(value), rtol=1e-5, atol=1e-7)
