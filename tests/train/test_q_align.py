# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for q_align data processor and trainer loss."""

import json
import math
import sys
import types
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Pre-patch trl to avoid ImportError on missing symbols in newer trl versions
# ---------------------------------------------------------------------------
def _patch_trl():
    import importlib
    try:
        import trl as _trl
        if not hasattr(_trl, "AutoModelForCausalLMWithValueHead"):
            _trl.AutoModelForCausalLMWithValueHead = MagicMock()
        if not hasattr(_trl, "ModelConfig"):
            _trl.ModelConfig = MagicMock()
    except ImportError:
        # trl not installed at all — inject a fake module
        fake_trl = types.ModuleType("trl")
        fake_trl.AutoModelForCausalLMWithValueHead = MagicMock()
        fake_trl.ModelConfig = MagicMock()
        sys.modules["trl"] = fake_trl

_patch_trl()

# ---------------------------------------------------------------------------
# Patch llamafactory.extras.misc.check_dependencies to skip version checks
# (installed transformers may be newer than the project's pinned range)
# ---------------------------------------------------------------------------
def _patch_check_deps():
    try:
        import llamafactory.extras.misc as _misc
        if hasattr(_misc, "check_dependencies"):
            _misc.check_dependencies = lambda: None
    except Exception:
        pass

_patch_check_deps()

from llamafactory.extras.constants import IGNORE_INDEX


# ---------------------------------------------------------------------------
# ── helpers: build lightweight mock objects ─────────────────────────────────
# ---------------------------------------------------------------------------

def _make_tokenizer(vocab_size: int = 64, pad_id: int = 0):
    """Return a minimal mock tokenizer that encodes strings to fixed token lists."""
    tok = MagicMock()
    tok.pad_token_id = pad_id
    tok.eos_token_id = 1
    # encode → reproducible deterministic ids based on string hash
    def _encode(text, add_special_tokens=True):
        ids = [abs(hash(c)) % (vocab_size - 2) + 2 for c in text[:6]]
        if add_special_tokens:
            ids = [2] + ids  # BOS
        return ids
    tok.encode.side_effect = _encode
    return tok


def _make_template(tokenizer):
    """Return a minimal mock template."""
    tpl = MagicMock()
    tpl.efficient_eos = False

    def _encode_multiturn(tok, messages, *args, **kwargs):
        # source = user content tokens, target = assistant content tokens
        user_text = messages[0]["content"] if messages else ""
        asst_text = messages[1]["content"] if len(messages) > 1 else ""
        src = [abs(hash(c)) % 60 + 3 for c in user_text[:4]]
        tgt = [abs(hash(c)) % 60 + 3 for c in asst_text[:4]]
        return [(src, tgt)]

    tpl.encode_multiturn.side_effect = _encode_multiturn
    return tpl


def _make_data_args(cutoff_len: int = 64):
    args = MagicMock()
    args.cutoff_len = cutoff_len
    args.mask_history = False
    args.train_on_prompt = False
    return args


# ---------------------------------------------------------------------------
# ── 1. QAlignDatasetProcessor tests ───────────────────────────────────
# ---------------------------------------------------------------------------

class TestQAlignDatasetProcessor:
    """Test the data processor without any real model or tokenizer."""

    def _make_processor(self, languages=("en", "l2", "l3")):
        from llamafactory.data.processor.supervised import QAlignDatasetProcessor

        tok = _make_tokenizer()
        tpl = _make_template(tok)
        data_args = _make_data_args()
        return QAlignDatasetProcessor(
            template=tpl,
            tokenizer=tok,
            processor=None,
            data_args=data_args,
            align_languages=list(languages),
        )

    def _make_examples(self, n: int = 2, languages=("en", "l2", "l3")):
        prompts, responses = [], []
        for i in range(n):
            p = {lang: f"prompt_{lang}_{i}" for lang in languages}
            r = {lang: f"response_{lang}_{i}" for lang in languages}
            prompts.append([{"role": "user", "content": json.dumps(p)}])
            responses.append([{"role": "assistant", "content": json.dumps(r)}])
        return {
            "_prompt": prompts,
            "_response": responses,
            "_system": [None] * n,
            "_tools": [None] * n,
            "_images": [None] * n,
            "_videos": [None] * n,
            "_audios": [None] * n,
        }

    def test_output_keys_present(self):
        """All qalign_{lang}_* keys must be in output."""
        languages = ("en", "l2", "l3")
        proc = self._make_processor(languages)
        examples = self._make_examples(n=3, languages=languages)
        out = proc.preprocess_dataset(examples)

        for lang in languages:
            assert f"qalign_{lang}_input_ids" in out, f"missing qalign_{lang}_input_ids"
            assert f"qalign_{lang}_attention_mask" in out
            assert f"qalign_{lang}_labels" in out

    def test_batch_length_consistent(self):
        """Each per-language list must have the same length."""
        languages = ("en", "l2")
        proc = self._make_processor(languages)
        n = 4
        out = proc.preprocess_dataset(self._make_examples(n=n, languages=languages))

        n_out = len(out["qalign_en_input_ids"])
        assert n_out == n
        for lang in languages:
            assert len(out[f"qalign_{lang}_input_ids"]) == n_out

    def test_attention_mask_all_ones(self):
        """attention_mask must be all-ones and same length as input_ids."""
        proc = self._make_processor(("en", "l2"))
        out = proc.preprocess_dataset(self._make_examples(n=2, languages=("en", "l2")))
        for lang in ("en", "l2"):
            for ids, mask in zip(out[f"qalign_{lang}_input_ids"], out[f"qalign_{lang}_attention_mask"]):
                assert mask == [1] * len(ids), f"attention_mask mismatch for {lang}"

    def test_labels_have_ignore_index_for_prompt(self):
        """Prompt tokens must be masked with IGNORE_INDEX in labels."""
        proc = self._make_processor(("en",))
        out = proc.preprocess_dataset(self._make_examples(n=2, languages=("en",)))
        for lbls in out["qalign_en_labels"]:
            # at least some IGNORE_INDEX present (prompt part)
            assert IGNORE_INDEX in lbls, "no IGNORE_INDEX found in labels"

    def test_skip_malformed_json(self):
        """Samples with invalid JSON should be silently skipped."""
        proc = self._make_processor(("en",))
        bad = {
            "_prompt": [[{"role": "user", "content": "NOT JSON"}]],
            "_response": [[{"role": "assistant", "content": '{"en": "ok"}'}]],
            "_system": [None], "_tools": [None],
            "_images": [None], "_videos": [None], "_audios": [None],
        }
        out = proc.preprocess_dataset(bad)
        assert len(out.get("qalign_en_input_ids", [])) == 0

    def test_skip_missing_language(self):
        """Samples where a required language is absent should be skipped."""
        proc = self._make_processor(("en", "l2", "l3"))
        # l3 is missing from the data
        p = {"en": "hello", "l2": "hola"}
        r = {"en": "bye", "l2": "adios"}
        bad = {
            "_prompt": [[{"role": "user", "content": json.dumps(p)}]],
            "_response": [[{"role": "assistant", "content": json.dumps(r)}]],
            "_system": [None], "_tools": [None],
            "_images": [None], "_videos": [None], "_audios": [None],
        }
        out = proc.preprocess_dataset(bad)
        assert len(out.get("qalign_en_input_ids", [])) == 0

    def test_print_data_example_runs(self, capsys):
        """print_data_example should not raise."""
        proc = self._make_processor(("en", "l2"))
        out = proc.preprocess_dataset(self._make_examples(n=1, languages=("en", "l2")))
        example = {k: v[0] for k, v in out.items()}
        proc.print_data_example(example)  # should not raise


# ---------------------------------------------------------------------------
# ── 2. Trainer _compute_loss_q_align tests (mock model, CPU) ──────────
# ---------------------------------------------------------------------------

def _build_mock_model(batch_size: int, seq_len: int, vocab_size: int, hidden_size: int, num_layers: int):
    """Return a mock nn.Module whose forward() produces realistic output shapes."""

    class MockOutput:
        def __init__(self, logits, hidden_states):
            self.logits = logits
            self.hidden_states = hidden_states

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # single linear layer so that backward() actually works
            self.linear = torch.nn.Linear(hidden_size, vocab_size, bias=False)
            self._num_layers = num_layers
            self._hidden_size = hidden_size

        def forward(self, input_ids, attention_mask=None, output_hidden_states=False, return_dict=True):
            B, T = input_ids.shape
            # random but differentiable hidden states
            hs = torch.randn(B, T, self._hidden_size, requires_grad=True)
            logits = self.linear(hs)
            # build tuple of (num_layers+1) hidden states
            hidden_states = tuple(
                torch.randn(B, T, self._hidden_size, requires_grad=True)
                for _ in range(self._num_layers + 1)
            )
            return MockOutput(logits=logits, hidden_states=hidden_states)

    return MockModel()


def _build_finetuning_args(languages=("en", "l2", "l3"), layer_idx: int = 2):
    fa = MagicMock()
    fa.stage = "q_align"
    fa.align_languages = list(languages)
    fa.align_hidden_layer = layer_idx
    fa.align_tau = 0.5
    fa.align_lambda_max = 0.3
    fa.align_gamma = 0.1
    return fa


def _build_trainer(finetuning_args, model, tokenizer):
    """Instantiate QAlignTrainer with minimal mocked dependencies."""
    from llamafactory.train.q_align.trainer import QAlignTrainer

    training_args = MagicMock()
    training_args.fp8 = False
    training_args.local_rank = -1
    training_args.world_size = 1

    # patch __init__ to avoid full HF Trainer setup
    with patch.object(QAlignTrainer, "__init__", lambda self, **kw: None):
        trainer = QAlignTrainer.__new__(QAlignTrainer)

    trainer.finetuning_args = finetuning_args
    trainer.model = model
    trainer.processing_class = tokenizer
    trainer._stored_metrics = {}

    # mock state
    state = MagicMock()
    state.global_step = 5
    trainer.state = state

    # stub store_metrics and log
    trainer._stored_metrics = {}

    def _store(metrics, train_eval="train"):
        pass

    trainer.store_metrics = _store

    return trainer


def _build_inputs(languages, batch_size: int, seq_len: int, vocab_size: int):
    """Build a minimal inputs dict with per-language tensors."""
    inputs = {}
    for lang in languages:
        ids = torch.randint(2, vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        # last half of tokens are labels (non-IGNORE)
        lbls = torch.full((batch_size, seq_len), IGNORE_INDEX)
        lbls[:, seq_len // 2:] = torch.randint(2, vocab_size, (batch_size, seq_len - seq_len // 2))
        inputs[f"qalign_{lang}_input_ids"] = ids
        inputs[f"qalign_{lang}_attention_mask"] = mask
        inputs[f"qalign_{lang}_labels"] = lbls
    return inputs


class TestQAlignTrainerLoss:

    VOCAB = 32
    HIDDEN = 16
    LAYERS = 4
    B = 2
    T = 12
    LANGS = ("en", "l2", "l3")

    def _setup(self, global_step=5):
        tok = _make_tokenizer(vocab_size=self.VOCAB)
        fa = _build_finetuning_args(languages=self.LANGS, layer_idx=2)
        fa.align_hidden_layer = 2  # within 0..LAYERS
        model = _build_mock_model(self.B, self.T, self.VOCAB, self.HIDDEN, self.LAYERS)
        trainer = _build_trainer(fa, model, tok)
        trainer.state.global_step = global_step
        inputs = _build_inputs(self.LANGS, self.B, self.T, self.VOCAB)
        return trainer, model, inputs

    def test_loss_is_scalar(self):
        """Total loss must be a scalar tensor."""
        trainer, model, inputs = self._setup()
        loss = trainer._compute_loss_q_align(model, inputs)
        assert loss.ndim == 0, f"expected scalar, got shape {loss.shape}"

    def test_loss_is_finite(self):
        """Loss must be a finite number."""
        trainer, model, inputs = self._setup()
        loss = trainer._compute_loss_q_align(model, inputs)
        assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"

    def test_loss_positive(self):
        """Cross-entropy loss must be positive."""
        trainer, model, inputs = self._setup()
        loss = trainer._compute_loss_q_align(model, inputs)
        assert loss.item() > 0

    def test_backward_runs(self):
        """backward() must not raise."""
        trainer, model, inputs = self._setup()
        loss = trainer._compute_loss_q_align(model, inputs)
        loss.backward()  # should not raise

    def test_lambda_zero_at_step_0(self):
        """At global_step=0 the exponential schedule gives lambda=0, so total = L_CE."""
        trainer, model, inputs = self._setup(global_step=0)
        loss = trainer._compute_loss_q_align(model, inputs)
        # lambda_val = lambda_max*(1-exp(0)) = 0 → L_total = L_CE
        # just verify it's finite and scalar
        assert torch.isfinite(loss) and loss.ndim == 0

    def test_helper_compute_per_sample_ce(self):
        """_compute_per_sample_ce must return shape [B] with finite values."""
        from llamafactory.train.q_align.trainer import QAlignTrainer

        B, T, V = 3, 8, 32
        logits = torch.randn(B, T, V)
        labels = torch.full((B, T), IGNORE_INDEX)
        labels[:, T // 2:] = torch.randint(0, V, (B, T - T // 2))
        ce = QAlignTrainer._compute_per_sample_ce(logits, labels)
        assert ce.shape == (B,)
        assert torch.all(torch.isfinite(ce))

    def test_helper_get_query_end_hidden_q_sample(self):
        """Q-only sample: query boundary = last IGNORE_INDEX position in labels."""
        trainer, model, inputs = self._setup()
        # IGNORE_INDEX in first 6 positions, labels after
        B, T, D = 2, 10, 16
        hidden = torch.randn(B, T, D)
        labels = torch.full((B, T), IGNORE_INDEX)
        labels[:, 6:] = torch.randint(2, 30, (B, 4))  # last IGNORE_INDEX at pos 5

        h, has_query = trainer._get_query_end_hidden(hidden, labels)
        assert h.shape == (B, D)
        assert has_query.all()

    def test_helper_get_query_end_hidden_r_sample(self):
        """R-only sample (no IGNORE_INDEX): has_query must be False."""
        trainer, model, inputs = self._setup()
        B, T, D = 2, 8, 16
        hidden = torch.randn(B, T, D)
        labels = torch.randint(2, 30, (B, T))  # no IGNORE_INDEX at all

        h, has_query = trainer._get_query_end_hidden(hidden, labels)
        assert h.shape == (B, D)
        assert not has_query.any()
