# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..sft.trainer import CustomSeq2SeqTrainer


logger = logging.get_logger(__name__)


class QAlignTrainer(CustomSeq2SeqTrainer):
    r"""Trainer for the ``q_align`` stage: SFT cross-entropy plus query-boundary alignment loss."""

    @staticmethod
    def _compute_per_sample_ce(
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute per-sample mean cross-entropy loss over non-masked label tokens.

        Args:
            logits: [B, T, V] raw logits from the model.
            labels: [B, T] ground-truth token ids; IGNORE_INDEX positions are masked.

        Returns:
            loss_per_sample: [B] scalar CE for each sample.
        """
        B, T, V = logits.shape
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view(B, T - 1)

        valid_mask = (shift_labels != IGNORE_INDEX).float()
        loss_per_sample = (
            (loss_per_token * valid_mask).sum(dim=1)
            / valid_mask.sum(dim=1).clamp(min=1.0)
        )
        return loss_per_sample

    def _get_query_end_hidden(
        self,
        hidden_state: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Extract the hidden state at the query boundary (Eq. 3).

        The boundary t* = |sp| + |q| is the last query token.  Prompt tokens
        (system prompt + query) carry IGNORE_INDEX in labels, so t* is the last
        IGNORE_INDEX position within the real (non-padded) part of the sequence;
        tail padding also carries IGNORE_INDEX and must be excluded.  Samples
        without any masked prompt token (R-only) have no query boundary and are
        flagged with has_query=False.

        Args:
            hidden_state:   [B, T, D]
            labels:         [B, T]
            attention_mask: [B, T] optional, used to exclude right-padding

        Returns:
            h:         [B, D] hidden states at the query boundary
            has_query: [B] bool tensor, False for R-only samples
        """
        B, T, D = hidden_state.shape
        query_end_pos = torch.zeros(B, dtype=torch.long, device=hidden_state.device)
        has_query = torch.ones(B, dtype=torch.bool, device=hidden_state.device)

        for b in range(B):
            ignore_mask = labels[b] == IGNORE_INDEX
            if attention_mask is not None:
                real_len = int(attention_mask[b].sum().item())
                ignore_mask[real_len:] = False
            if ignore_mask.any():
                query_end_pos[b] = ignore_mask.nonzero(as_tuple=False)[-1].item()
            else:
                has_query[b] = False

        query_end_pos = query_end_pos.clamp(min=0, max=T - 1)
        h = hidden_state[torch.arange(B, device=hidden_state.device), query_end_pos]
        return h, has_query

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        return self._compute_loss_q_align(model, inputs)

    @override
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
        r"""Evaluation step: q_align batches carry only ``qalign_{lang}_*`` keys,
        so the loss is computed directly on the per-language tensors."""
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self._compute_loss_q_align(model, inputs)
        return loss.detach().mean(), None, None

    def _compute_loss_q_align(self, model, inputs) -> torch.Tensor:
        r"""Q-Align: Query-Boundary Multilingual Alignment loss.

        For each parallel sample, the hidden state at the query boundary is extracted
        per language and aligned toward a confidence-weighted anchor:

        - Anchor weight (Eq. 4): α_k = softmax_k(-CE_k / τ), so languages with lower
          cross-entropy act as stronger anchors.
        - Anchor (Eq. 5): h* = Σ_k α_k · h_k (detached; gradients flow only through h_k).
        - Alignment loss (Eq. 6): L_align = (1/K) Σ_k ‖h_k − h*‖₂², averaged uniformly
          over languages and over the alignable samples in the batch.
        R-only samples (no query) are excluded from alignment but still contribute
        to the cross-entropy loss.

        Total loss (Eq. 7):
            L_total = L_CE + λ(t) · L_align
        with λ(t) ramped from 0 to λ_max by the curriculum schedule.
        """
        languages = self.finetuning_args.align_languages
        layer_idx = self.finetuning_args.align_hidden_layer
        tau = self.finetuning_args.align_tau
        lambda_max = self.finetuning_args.align_lambda_max
        gamma = self.finetuning_args.align_gamma
        global_step = self.state.global_step

        train_eval = "train" if torch.is_grad_enabled() else "eval"
        L = len(languages)

        # ------------------------------------------------------------------
        # Gather per-language tensors and pad to uniform max_T
        # ------------------------------------------------------------------
        ids_list = [inputs[f"qalign_{lang}_input_ids"] for lang in languages]
        mask_list = [inputs[f"qalign_{lang}_attention_mask"] for lang in languages]
        labels_list = [inputs[f"qalign_{lang}_labels"] for lang in languages]

        B = ids_list[0].size(0)

        max_T = max(t.size(1) for t in ids_list)
        pad_id = self.processing_class.pad_token_id if self.processing_class.pad_token_id is not None else 0

        def _right_pad(t: torch.Tensor, length: int, value: int) -> torch.Tensor:
            if t.size(1) >= length:
                return t
            return F.pad(t, (0, length - t.size(1)), value=value)

        all_ids = torch.cat([_right_pad(t, max_T, pad_id) for t in ids_list], dim=0)       # [L*B, max_T]
        all_mask = torch.cat([_right_pad(t, max_T, 0) for t in mask_list], dim=0)
        all_labels = torch.cat([_right_pad(t, max_T, IGNORE_INDEX) for t in labels_list], dim=0)

        # ------------------------------------------------------------------
        # Single forward pass (1 forward → 1 backward, DeepSpeed safe)
        # ------------------------------------------------------------------
        out = model(
            input_ids=all_ids,
            attention_mask=all_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        if layer_idx >= len(out.hidden_states):
            raise ValueError(
                f"align_hidden_layer={layer_idx} is out of range: the model returns "
                f"{len(out.hidden_states)} hidden states (indices 0..{len(out.hidden_states) - 1})."
            )

        # Split back into per-language chunks: each [B, max_T, ...]
        logits_chunks = out.logits.split(B, dim=0)
        hidden_chunks = out.hidden_states[layer_idx].split(B, dim=0)
        labels_chunks = all_labels.split(B, dim=0)
        mask_chunks = all_mask.split(B, dim=0)

        # ------------------------------------------------------------------
        # Per-language CE and query-boundary hidden state extraction
        # ------------------------------------------------------------------
        ce_per_lang = []
        h_per_lang = []
        has_query_per_lang = []
        for idx in range(L):
            ce_per_lang.append(
                self._compute_per_sample_ce(logits_chunks[idx], labels_chunks[idx])
            )
            h, hq = self._get_query_end_hidden(
                hidden_chunks[idx], labels_chunks[idx], mask_chunks[idx]
            )
            h_per_lang.append(h)
            has_query_per_lang.append(hq)

        ce_stack = torch.stack(ce_per_lang, dim=1)            # [B, L]
        h_stack = torch.stack(h_per_lang, dim=1)              # [B, L, D]
        has_query_stack = torch.stack(has_query_per_lang, dim=1)  # [B, L]

        # A sample participates in alignment only if ALL its languages have a query
        can_align = has_query_stack.all(dim=1)                # [B]

        # Log alignment participation statistics for the first few steps
        if global_step < 3:
            n_can_align = can_align.sum().item()
            logger.info_rank0(
                f"[QALIGN] step={global_step}, can_align={n_can_align}/{B}"
            )

        # ------------------------------------------------------------------
        # CE loss (all samples, all languages)
        # ------------------------------------------------------------------
        L_CE = ce_stack.mean()

        # ------------------------------------------------------------------
        # Alignment loss (samples with a query boundary in all languages)
        # ------------------------------------------------------------------
        n_align = can_align.float().sum().clamp(min=1.0)

        if can_align.any():
            with torch.no_grad():
                alpha = F.softmax(-ce_stack.detach() / tau, dim=1)             # [B, L]
                h_star = (alpha.unsqueeze(-1) * h_stack.detach()).sum(dim=1)    # [B, D]

            diff = h_stack - h_star.unsqueeze(1)             # [B, L, D]
            align_per_lang = (diff ** 2).sum(dim=-1)         # [B, L]  ||h_k - h*||_2^2 (Eq. 6)
            align_per_sample = align_per_lang.mean(dim=1)    # [B]  (1/K) Σ_k
            L_align = (align_per_sample * can_align.float()).sum() / n_align
        else:
            L_align = torch.tensor(0.0, device=ce_stack.device)

        # ------------------------------------------------------------------
        # Lambda schedule (Eq. 7): exponential warm-up from 0 to lambda_max
        # ------------------------------------------------------------------
        lambda_val = lambda_max * (1.0 - math.exp(-gamma * global_step))

        # ------------------------------------------------------------------
        # Total loss
        # ------------------------------------------------------------------
        L_total = L_CE + lambda_val * L_align

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        metrics = {
            f"{train_eval}_ce_loss": L_CE.detach().item(),
            f"{train_eval}_align_loss": L_align.detach().item(),
            f"{train_eval}_lambda": float(lambda_val),
            f"{train_eval}_n_align": float(can_align.sum().item()),
        }
        self.store_metrics(metrics, train_eval)

        logger.info_rank0(
            "L_CE=%.4f  L_align=%.4f  lambda=%.6f  n_align=%d/%d",
            metrics[f"{train_eval}_ce_loss"],
            metrics[f"{train_eval}_align_loss"],
            metrics[f"{train_eval}_lambda"],
            int(metrics[f"{train_eval}_n_align"]),
            B,
        )

        return L_total
