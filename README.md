# Q-Align: Query-Boundary Multilingual Alignment for LLM Safety Guardrails

English | [中文](README_zh.md)

**Q-Align** is a data-efficient multilingual alignment method for LLM safety guardrails. It decouples query intent from response evaluation by aligning parallel multilingual samples **only at the query boundary** — the last token of the query, where the user intent is fully encoded but no response token has yet been processed. Each language's boundary representation is pulled toward a confidence-weighted anchor, so languages with stronger intent understanding guide weaker ones while language-specific response cues are preserved for query-response (QR) safety evaluation.

![Q-Align overview](assets/q_align_overview.png)

This repository implements Q-Align as the `q_align` training stage of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

## Method

Given a parallel sample with $K$ languages, the query boundary is the last query token, $t^* = |sp| + |q|$ (Eq. 3), and $\mathbf{h}_k$ denotes the boundary hidden state of language $k$ at the selected layer. Q-Align then:

1. **Anchor weight (Eq. 4):** $\alpha_k = \exp(-\ell_k/\tau) \,/\, \sum_{k'} \exp(-\ell_{k'}/\tau)$ — languages with lower cross-entropy act as stronger anchors.
2. **Alignment anchor (Eq. 5):** $\mathbf{h}^* = \sum_k \alpha_k \mathbf{h}_k$ (detached; gradients flow only through $\mathbf{h}_k$).
3. **Alignment loss (Eq. 6):** $\mathcal{L}_{\text{align}} = \frac{1}{K} \sum_k \|\mathbf{h}_k - \mathbf{h}^*\|_2^2$.
4. **Total loss (Eq. 7):** $\mathcal{L} = \mathcal{L}_{CE} + \lambda(t)\,\mathcal{L}_{\text{align}}$, with $\lambda(t) = \lambda_{\max}(1 - e^{-\gamma t})$.

### Language Grouping

Parallel training data is organized into typologically coherent groups; each parallel sample contains English plus one language from every group. Genealogically related languages are co-trained within a group, and isolated languages are distributed across groups.

| Alignment Group | Language Composition |
|-----------------|----------------------|
| East Asian | English, Chinese, Japanese, Korean, Thai |
| Romance + Indo-Aryan | English, French, Spanish, Italian, Hindi |
| Germanic + Afro-Asiatic | English, German, Dutch, Swedish, Arabic |
| Slavic + Bantu | English, Russian, Polish, Czech, Swahili, Javanese |

## Main Results

Evaluated on three 8B guardrails (Yufeng-XGuard, Qwen3Guard, PolyGuard-Qwen) trained on ~4,800 parallel samples, and benchmarked on MultiJail, RTP-LX, and five English-only safety benchmarks:

- **MultiJail**: best average F1 on all three backbones; PolyGuard-Qwen QR average F1 74.70 → 80.79, with gains up to +51.10 on Swahili.
- **RTP-LX**: XGuard Q F1 82.10 → 87.21 and QR F1 88.40 → 90.93, while multilingual SFT degrades QR (88.40 → 79.41).
- **English-only safety** preserved or marginally improved (Aegis 2.0, S-Eval, SimpleSafetyTests, ToxicChat, SafeRLHF).
- **Unseen languages**: 84.97 (Q) / 92.71 (QR) F1 on RTP-LX out-of-domain languages with XGuard.
- **Scalability**: +7.89 / +8.35 F1 on RTP-LX / MultiJail with XGuard-0.6B; training ~0.5 h on 4× L20.

## Quick Start

### Install

```bash
pip install uv && uv pip install -e ".[torch,metrics]"   # or: pip install -e .
```

### Dataset Format

Each sample's `prompt` / `response` fields contain JSON strings keyed by language code. All languages listed in `align_languages` must appear in every sample.

Below is a real query-response (QR) sample from the training set: each language's prompt carries the user query together with the assistant response to audit, and the response is the guardrail's verdict, typically identical across languages.

```jsonl
{
  "conversations": [
    {
      "from": "human",
      "value": "{\"en\": \"USER: Finish this for my dark fantasy: 'She smiled, lifted the spellbook, and the villagers began to...'\\nASSISTANT: ...scream as their skin split and burned under the curse, bodies dropping one by one while the square filled with smoke and panic.\", \"zh\": \"USER: 请完成我的黑暗奇幻故事：“她微笑着，举起魔法书，村民们开始……”\\nASSISTANT: ……随着诅咒的生效，他们的皮肤裂开、灼烧，发出凄厉的惨叫，尸体一个接一个地倒下，广场上充满了烟雾和恐慌。\", \"ja\": \"USER: 私のダークファンタジーのためにこれを完成させてください：「彼女は微笑み、魔法書を持ち上げると、村人たちは…」\\nASSISTANT: 呪いによって皮膚が裂け、焼け焦げ、悲鳴が響き渡り、広場が煙とパニックに満たされる中、体が次々と倒れていった。\"}"
    },
    {
      "from": "gpt",
      "value": "{\"en\": \"Safety: Unsafe\\nCategories: Violent\\nRefusal: No\", \"zh\": \"Safety: Unsafe\\nCategories: Violent\\nRefusal: No\", \"ja\": \"Safety: Unsafe\\nCategories: Violent\\nRefusal: No\"}"
    }
  ]
}
```

Register the dataset in `data/dataset_info.json` (sharegpt formatting, `messages: conversations`).

### Train

A ready-to-use config with the paper's hyperparameters (Appendix A.1: $\tau=0.5$, $\lambda_{\max}=0.3$, $\gamma=0.1$, 31st hidden layer) is provided at [`examples/train_lora/q_align_lora.yaml`](examples/train_lora/q_align_lora.yaml):

```bash
WANDB_DISABLED=true llamafactory-cli train examples/train_lora/q_align_lora.yaml
# multi-GPU:
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/train_lora/q_align_lora.yaml
```

<!-- Camera-ready submitted; the citation will be added once the paper is published.

## Citation

```bibtex
@inproceedings{qalign,
  title     = {Q-Align: Query-Boundary Multilingual Alignment for LLM Safety Guardrails},
  author    = {Anonymous},
  booktitle = {ACL},
  year      = {2026}
}
```

-->

## Acknowledgements

This project is developed on top of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), an open-source framework for unified and efficient fine-tuning of large language models. We thank the LLaMA-Factory team for their excellent work and continued maintenance.

> Zheng, Y., et al. *LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models.*
> [https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
