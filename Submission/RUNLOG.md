## Run 1 — baseline
 unmodified train.py/model.py/tokenizer.py, 2000 steps. dev bpb = 2.3718, 1,339,840 params, 130s. Loss decreased steadily with no instability, but tokenizer is raw UTF-8 bytes (vocab 256) — Devanagari characters cost 3 bytes/tokens each, so a large fraction of the corpus's token budget goes to sub-character units. Optimizer is plain Adam, constant LR, no warmup/clipping/decay, flat std=0.05 init.

## Run 2 — BPE tokenizer + arch/optimizer overhaul

**Hypothesis:** Byte-level tokenizer wastes budget on Devanagari (3 bytes/char), 
so BPE should shorten sequences and give more real context per block, lowering bpb.

**Changed:** Tokenizer → BPE (256 merges, vocab 512, verified lossless, 
~2.25 bytes/token). Architecture → n_embd 152, n_layer 6, block_size 256, 
tied weights, scaled residual init (1,792,384 params). Optimizer → AdamW, 
warmup+cosine LR, grad clipping. (Combined-change run, not an ablation.)

**Result:** bpb 2.3718 → 2.2833 (−3.7%), 1,792,384/2,000,000 params, 2000 steps.

**Conclusion:** Clear improvement, under caps. Can't isolate which change 
(tokenizer vs. arch vs. optimizer) contributed most — next step would be 
an ablation isolating the tokenizer alone.

## Run 3 — Ablation: BPE tokenizer only (baseline arch + baseline optimizer)

**Hypothesis:** Run 2 changed tokenizer, architecture, and optimizer together, 
so the 2.3718 → 2.2833 gain couldn't be attributed to any one change. This run 
isolates the tokenizer by pairing it with the original baseline model.py 
(Config: block_size 128, n_layer 4, n_embd 160, tie_weights False) and 
baseline train.py (plain Adam, lr 3e-4, no warmup/clip/decay).

**Changed:** Tokenizer only — BPE (vocab 512) in place of byte-level (vocab 256). 
Everything else reverted to Run 1 baseline.

**Result:** bpb 2.3718 → 2.2826, 1,421,760 params, 2000 steps, 92s.

**Conclusion:** The tokenizer alone accounts for essentially the entire gain 
seen in Run 2 (2.2833). The architecture and optimizer changes in Run 2 
contributed almost nothing on top (difference is within noise, 0.0007). 
Possible reasons: the larger model may need more than 2000 steps to exploit 
its extra capacity, and BPE already shrinks sequences enough that block_size 
128 vs 256 gives less marginal context benefit than expected. Next: sweep 
peak LR (8e-4, 1e-3) on the Run 2 config to check if the arch/optimizer 
changes just need different tuning to show their value within the step budget.

## Run 4 — v2 config (BPE + arch + AdamW/warmup/cosine/clip), peak LR 8e-4

**Hypothesis:** Run 3's ablation showed the arch/optimizer changes added 
nothing over the tokenizer alone at lr=6e-4. One possible reason: the bigger 
model (1,792,384 params vs. 1,421,760) may need a higher peak LR to fully 
exploit its extra capacity within the fixed 2000-step budget. Testing lr=8e-4 
(up from 6e-4) on the same v2 config to check this.

**Changed:** `--lr 8e-4` only, all else identical to Run 2 (BPE tokenizer, 
n_embd 152, n_layer 6, block_size 256, tied weights, scaled init, AdamW, 
100-step warmup + cosine decay to 10% peak, grad clip 1.0).

**Result:** bpb 2.2833 → 2.2762 (Run 2 → Run 4), 1,792,384 params, 2000 steps, 
278s (notably slower wall time than prior runs — likely machine load, not a 
code change, since LR doesn't affect compute cost).

**Conclusion:** Higher peak LR closes and reverses the gap seen in Run 3 — 
this is now the best result so far (2.2762 vs. baseline 2.3718, ≈4.0% 
reduction; vs. tokenizer-only ablation 2.2826, a further small but real gain). 
Supports the hypothesis that the bigger model needed more aggressive LR to 
pay off its extra capacity within 2000 steps. Next: consider lr=1e-3 to check 
if pushing further helps more, budget permitting.

## Run 5 — v2 config, peak LR 1e-3

**Hypothesis:** Run 4 showed lr=8e-4 > lr=6e-4, suggesting the trend continues 
— push peak LR further to see if the bigger model keeps benefiting within the 
2000-step budget.

**Changed:** `--lr 1e-3` only, all else identical to Run 4.

**Result:** bpb 2.2762 → 2.272, 1,792,384 params, 2000 steps, 285s.

**Conclusion:** Marginal further improvement (−0.0042), smaller gain than the 
6e-4→8e-4 step (−0.0071) — diminishing returns, suggesting we're approaching 
the LR ceiling for this step budget before instability would start to hurt. 
This is the best result across all runs: 2.272 vs. baseline 2.3718 (≈4.2% 
reduction).

## Run 6 — batch size 16 (v2 config + lr 1e-3, same as Run 5 otherwise)

**Hypothesis:** Larger batch reduces gradient noise, may let the already-tuned
lr=1e-3 cosine schedule converge more effectively in the fixed 2000-step budget.

**Changed:** `--batch 16` only (was 8), all else identical to Run 5.

**Result:** bpb 2.272 → 2.073 (−8.8%), 1,792,384 params, 2000 steps, 543s
(notably slower than Run 5's 285s — plausibly batch compute cost, but the
~1.9x gap is larger than expected for batch 8→16 alone, so machine load may
also be a factor; not fully isolated).

**Conclusion:** Best result overall: 2.073 vs. baseline 2.3718 (≈12.6%
reduction). Not a clean ablation — didn't test batch 16 at lr 6e-4 or batch 8
at lr 1e-3 in isolation, so can't fully separate the batch effect from the
batch×lr interaction. Given time constraints, adopting this as final config.