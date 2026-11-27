# IFA_OMEGA training results (scaled run on a free Colab T4)

Run date: 2026-11-26 to 2026-11-27 (Google Colab free tier, Tesla T4, resumed from Drive checkpoints after runtime disconnects).

## Config
- Params: 10.34M (scaled down for a free T4, not the 14M/30M/98M configs)
- Tokenizer: char, vocab 97
- Layers 6, d_model 256, batch 32, seq 256
- Steps: 1500
- Data: TinyStories (ronenldan/TinyStories), train[:1%]
- Script: scripts/train.py from this repo, patched only to save/resume a checkpoint every 200 steps

## Result
- Training reached step 1500 and logged DONE.
- Val loss: 4.658 at step 0, 1.084 at step 200, 0.931 at step 800, 0.857 at step 1200.
- Last full run (steps 1250-1500): 0.839, 0.846, 0.880, 0.852, 0.840, 0.916 (step 1500).
- Val loss is noisy at this size; it flattened at about 0.84-0.92 after step 1100.

## Notes on the log (results/train_log.txt)
- The run was interrupted by Colab disconnects and resumed from checkpoints (steps 201, 801, 1201).
- Steps 1251-1350 appear twice. A duplicate run was started by mistake after the last resume and was stopped. The block 1250 0.839 to 1500 0.916 is the run that finished.
- Raw log is unedited.
