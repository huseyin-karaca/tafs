# Evaluation Protocol

**Test-time caching.** In `on_test_start`, initialise `self._test_selections`, `self._test_probs`, `self._test_errors`. In `test_step`, append batched arrays. `evaluate()` concatenates these buffers — **never call the test loader a second time**.

**Seeding.** All randomness flows through `src.core.utils.seeding.seed_everything(seed)`. No bare `torch.manual_seed`, no bare `np.random.seed`.

**Seeds & splits.**
- `seeds: List[int]` in the experiment YAML enumerates the runs.
- Default (`fixed_data_split: false`): each seed re-derives both the train/val/test split and the model init from the same integer. This is the precondition for the Nadeau-Bengio paired t-test.
- `fixed_data_split: true` reverts to "split fixed, only model varies" and **must** log a warning that the t-test assumptions are violated.

**Significance test.** `src.core.stats.paired_t.NadeauBengioCorrectedTTest` with `alpha = 0.05`.

**Aggregation.** Per-child run logs `mean(metric)` and `metric__seed_sem`. Tables in the paper report mean (standard deviation in parentheses).

**No leakage.** Standardisation statistics are fit on the training split only and applied to val/test. No global mean/std.
