# IFA_OMEGA

**EN:** A from-scratch *hybrid* language model: Mamba-style SSM blocks + attention every 4th layer + Mixture-of-Experts, in pure PyTorch (no CUDA kernels needed). Phase 1 is text/chat/code; images, video and a cybersecurity layer come later (see roadmap).

**AR:** نموذج ذكاء اصطناعي هجين من الصفر: طبقات SSM (زي Mamba) + طبقة Attention كل 4 طبقات + MoE، مكتوب بـ PyTorch صافي ويشتغل على Colab المجاني. المرحلة 1: نصوص وشات وكود. الصور والفيديو والأمن السيبراني في المراحل الجاية.

## Architecture / المعمارية
```
tokens -> Embedding -> [ HybridLayer x N ] -> RMSNorm -> LM head
HybridLayer = (SSM | Attention) mixer  +  (SwiGLU | MoE) feed-forward
  - Attention at layers 4, 8, ...; SSM elsewhere
  - MoE (top-2 of 4 experts) at layers 2, 4, 6, ...
```
Why hybrid: SSM cost grows linearly with context, attention gives precise recall; MoE adds capacity without more compute per token. / SSM أرخص في السياق الطويل، والـ Attention دقيق في التذكر، والـ MoE بيزود السعة من غير تكلفة حسابية كبيرة.

## Layout
- `ifa_omega/config.py` model settings
- `ifa_omega/layers.py` SSM, Attention, SwiGLU, MoE
- `ifa_omega/model.py` hybrid model
- `scripts/train.py` training loop
- `configs/tiny.yaml` reference config

## Run on Colab / التشغيل على Colab
```python
!git clone https://github.com/alhenawy222-arch/IFA-Adaptive-Cognitive-Intelligence
%cd IFA-Adaptive-Cognitive-Intelligence
!pip install -r requirements.txt
!python scripts/train.py --tokenizer char --steps 500 --layers 6 --d_model 256
```
Runtime > Change runtime type > T4 GPU. The default dataset is tiny Shakespeare; plug in TinyStories at the `TODO(data)` in `scripts/train.py`.

## Notes / ملاحظات
The SSM uses a sequential Python scan: correct but slow. Next step is a parallel scan or `mamba-ssm`. Expect small models (1-50M params) on free compute, not GPT-scale.

## Roadmap
1. Text/chat/code (this repo) 2. Images (encoder) 3. Video 4. Cybersecurity layer (guardrails, prompt-injection defense) + agent/RAG orchestration.
