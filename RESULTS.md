# IFA_OMEGA: first training run / أول تجربة تدريب

## English
**What was run:** `python scripts/train.py --steps 400 --batch 8 --seq 64` on CPU (1 core, PyTorch 2.14 CPU), default dataset (Tiny Shakespeare, 1.1M characters, char-level, vocab 65). Model: 1.73M parameters, 4 layers, d_model 128. Took about 10 minutes. Full log: `results_train_log.txt`.

**Validation loss (cross-entropy, nats):**

| step | 0 | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 |
|---|---|---|---|---|---|---|---|---|---|
| val_loss | 4.215 | 2.794 | 2.452 | 2.398 | 2.100 | 2.244 | 2.061 | 1.907 | 1.997 |

Random guessing over 65 characters is ln(65) = 4.17, so the model starts at chance and drops to about 2.0.

**Samples (after 400 steps):**
```
ROMEO:
What;
Why repes uut learth the kZoak come, or true virtue cauced sWoulrawl.
xe aroad noales, we sin, grad of the krad:

First Citizen:
Od my lord;
Whese you to!. Une ofNBHbraisanry glw v, mces dthe sid here tle'baths wobourm:
```

**What this shows:** the code runs end to end, gradients flow through the SSM, attention and MoE layers, and the loss falls well below chance. The model learned character frequencies, the play layout (names, colons, line breaks) and a few real words.

**What it does not show:** this is a toy run (400 steps of 8x64 tokens, about 205K tokens seen, under one fifth of one pass over the data). The text is not coherent. It says nothing about quality at scale, about whether the hybrid design beats a plain transformer, or about the neuro-symbolic part. Next steps: a GPU (Colab T4), a real dataset (e.g. TinyStories), thousands of steps, and a baseline comparison.

**Checkpoint:** `ifa_omega.pt` is about 7 MB, so it is not committed. Reproduce with the command above.

## العربية
**اللي اتشغّل:** `python scripts/train.py --steps 400 --batch 8 --seq 64` على CPU، على بيانات Tiny Shakespeare (1.1 مليون حرف، تقسيم حرفي، 65 رمز). النموذج 1.73 مليون باراميتر، 4 طبقات. خد حوالي 10 دقايق. اللوج الكامل في `results_train_log.txt`.

**الخسارة على بيانات التحقق:** بدأت 4.215 (يعني تخمين عشوائي، ln65 = 4.17) ونزلت لحوالي 2.0 بعد 400 خطوة (الجدول فوق).

**العينات:** فوق. النص لسه مش مفهوم.

**ده بيثبت إيه:** الكود شغال من أوله لآخره، والتدريب بيمشي في طبقات SSM والانتباه وMoE، والخسارة نزلت كتير تحت العشوائي، والنموذج اتعلم شكل المسرحية (أسماء، نقطتين، أسطر) وكام كلمة حقيقية.

**مش بيثبت إيه:** ده تجربة لعبة صغيرة. مفيش حاجة عن الجودة على نطاق حقيقي، ولا إن التصميم الهجين أحسن من Transformer عادي، ولا عن الجزء العصبي الرمزي. الخطوة الجاية: GPU (Colab T4)، بيانات حقيقية زي TinyStories، آلاف الخطوات، ومقارنة بنموذج أساسي.

**الـCheckpoint:** حجمه حوالي 7 ميجا فمتحطش في الريبو. أعد إنتاجه بنفس الأمر.
