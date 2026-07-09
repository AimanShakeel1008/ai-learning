# Chapter 4 Summary — The Machine Learning Workflow

**Phase 2 · Chapter 4 · Summary**
**Position:** Phase 2 (Classical Machine Learning) → Chapter 4 (The Machine Learning Workflow) → after Lessons 11–13.
**Files created:** `lessons/phase-02-classical-machine-learning/chapter-04-the-machine-learning-workflow/chapter-04-summary.md`
**Files updated:** `progress-tracker.md`
**Project change:** none across the whole chapter — Chapter 4 taught the *discipline* of machine learning, not a new feature. The assistant still runs exactly as it did after Chapter 1.

---

## WHAT WE LEARNED, AND WHAT WE ADDED TO THE ASSISTANT

Chapter 4 was about the **workflow** — the repeatable process every machine-learning project follows, no matter which algorithm you eventually use. Before touching a single real algorithm, we learned how to set a project up so its results can be *trusted*.

- **Lesson 11 — the end-to-end pipeline.** We walked the five stages every ML project shares: **data → features → train → evaluate → use**. We turned raw ticket text into numbers with **bag-of-words**, "trained" a simple word-count classifier, held out a **test set**, and measured **accuracy against a majority-class baseline** so a score means something.
- **Lesson 12 — train, validation, and test sets.** We learned *why* we hide data. Testing on the data you trained on is the **cardinal sin** — a memoriser scored a fake 1.00 then collapsed to 0.33 on new tickets. We split data three ways: **train** (learn), **validation** (make choices), **test** (final honest grade, looked at once), and named **data leakage**.
- **Lesson 13 — overfitting, underfitting, and bias-variance.** We learned to *read* those honest scores. **Underfitting** (too simple, high bias) fails everywhere; **overfitting** (too complex, high variance) aces training and fails on new data. The **gap between training and validation error** is the diagnostic, and the **bias-variance U-curve** shows the sweet spot in the middle.

Nothing was added to the customer-support assistant — this chapter is the rulebook we'll follow every time we *do* add a model, starting in Chapter 5.

---

## HOW THE CHAPTER'S IDEAS CONNECT (a plain-language map)

Think of the chapter as one flow, each lesson unlocking the next:

```
  LESSON 11               LESSON 12                 LESSON 13
  the PROCESS      -->     honest SCORING     -->   reading the SCORE
  (5-stage pipeline)      (hide data properly)     (over/underfit, bias-variance)

  data -> features        train | validation |     train error vs validation error
  -> train -> evaluate     test  (each a job)        -> the GAP tells you what's wrong
  -> use                  memoriser proves the      -> U-curve: pick the middle
                          cardinal sin
```

The through-line: **Lesson 11** gives you the machine (a pipeline that produces a model and a score). **Lesson 12** makes sure that score is *honest* by hiding data the model never learns from. **Lesson 13** teaches you what the honest score is *telling you* — and none of Lesson 13's diagnosis is even possible without Lesson 12's held-out split, because you need a training score *and* a validation score to see the gap between them. Three lessons, one idea: build a model, score it honestly, and understand the score.

---

## THE STORY SO FAR (how the assistant has grown)

- **Phase 1 (Chapters 1–3)** gave the assistant its first toy brain (a rule-based, then word-counting, urgency classifier in `ticket_urgency.py`) and gave *us* the tools and math underneath AI: Python libraries, NumPy arrays, pandas tables, plots, linear algebra, calculus/gradient descent, and probability.
- **Chapter 4** added no code to the assistant — on purpose. Instead it installed the **professional habits** we'll use every time we upgrade it: run the full pipeline, split data honestly, and diagnose over/underfitting from the train-vs-validation gap.
- **What's still true:** the assistant is still the small Chapter-1 word-counter. `python main.py` runs it end to end and its self-check `assert`s still pass. The difference is that *we* are now ready to replace that toy with a real, properly-evaluated learned model — which is exactly what Chapter 5 begins.

---

## LESSON TABLE

| Lesson | Core idea | Key tool / term |
|---|---|---|
| **11 — The end-to-end ML pipeline** | Every ML project follows the same five stages: data → features → train → evaluate → use. | Bag-of-words features; hold-out test set; **accuracy vs. majority-class baseline** |
| **12 — Train, validation, and test sets** | Hide data from the model so its score is honest; the cardinal sin is testing on training data. | **Train / validation / test** split; **data leakage** |
| **13 — Overfitting, underfitting, bias-variance** | Read the honest score: both-bad = underfit; great-train-poor-validation = overfit. | **Bias vs. variance**; the **train-vs-validation gap**; the **U-curve** |

---

## IDEAS THAT COME BACK LATER (and where)

- **The train/validation/test split** is the backbone of *every* model we build from Lesson 14 on, and gets a scarce-data upgrade — **cross-validation** — in **Lesson 23**.
- **Overfitting and the bias-variance tradeoff** return with full force in deep learning: **Lesson 33** ("training neural networks well") is largely a toolkit — regularization, dropout, early stopping — for fighting the overfitting we met here.
- **Baselines and honest evaluation** grow into a whole toolbox of **classification metrics** (precision, recall, F1, the confusion matrix) in **Lesson 22**, where we'll see why plain accuracy can lie.
- **Data leakage**, named in Lesson 12, gets its own dedicated treatment alongside scaling and class imbalance in **Lesson 21**.
- **The pipeline shape** (data → features → train → evaluate → use) is exactly the skeleton the scikit-learn `fit`/`predict`/`transform` workflow formalises in **Lesson 26**.

---

## SELF-CHECK (can you answer these?)

1. Name the five stages of the machine-learning pipeline in order, and say what "features" means.
2. Why is a model's **accuracy on its own training data** almost worthless as a measure of quality?
3. What is each of the three splits — **train, validation, test** — actually *for*, and which one may be looked at only once?
4. You see training error 0.02 and validation error 0.40. Which failure is this, what is its technical name, and what are two ways to fix it?
5. As model complexity rises, what happens to **training error** versus **validation error**, and how do you use those two curves to pick the best model?

<details>
<summary>Click to reveal the answers</summary>

1. **Data → features → train → evaluate → use.** *Features* are the numeric signals we turn raw data into so a model can work with it (e.g. bag-of-words counts from ticket text).
2. Because the model was **allowed to learn from those exact examples**, so it can score high by **memorising** rather than understanding — the "cardinal sin." A high training score tells you almost nothing about performance on new, unseen data.
3. **Train** = the examples the model learns its parameters from. **Validation** = held-out data *you* use to make choices during development (which model, which settings). **Test** = the final honest grade, looked at **exactly once**, on data that never influenced the model in any way.
4. This is **overfitting** (tiny training error, much larger validation error, a big gap); its technical name is **high variance**. Fix it by using a **simpler model**, adding **more training data**, or applying techniques that penalise complexity (regularization, early stopping).
5. **Training error keeps falling** as complexity rises (it can always hug the training points tighter). **Validation error falls, bottoms out, then rises** — a **U**. You pick the complexity at the **bottom of the validation U** (lowest validation error), never the lowest training error.

</details>

---

## CHAPTER CHALLENGE (with hidden answer)

**The scenario.** You inherit a support-ticket "priority predictor" from a teammate who left. They proudly documented: *"Trained on all 4,000 historical tickets. Accuracy: 0.97. Ready to ship."* You run it on last week's 300 real tickets and it scores **0.61**. The categories in the historical data are 70% "normal", 20% "high", 10% "urgent".

**Your task:** (1) Diagnose what almost certainly went wrong, using the right technical terms. (2) Explain why the 0.97 was measured wrong *and* why 0.61 might still not be as bad as it looks compared to doing nothing. (3) Lay out, step by step, exactly how you'd rebuild the evaluation so the next number you report is trustworthy — including how you'd choose between two candidate models without cheating.

<details>
<summary>Click to reveal the answer</summary>

**(1) Diagnosis: overfitting, hidden by the cardinal sin of testing on training data.** The 0.97 was measured on the same 4,000 tickets the model trained on, so it's the memoriser's flattering score — it reflects how well the model memorised, not how well it generalises. The drop to 0.61 on genuinely new tickets reveals a large **train-vs-validation gap**: high training performance, much lower real-world performance = **high variance / overfitting**. (We can't be *certain* it's overfitting versus other bugs, but a 0.97 → 0.61 collapse from training data to new data is its textbook signature.)

**(2) Why 0.97 is wrong, and why 0.61 isn't the whole story.** The 0.97 is invalid because it broke the one rule of honest evaluation: the grade must come from data the model never learned from. As for 0.61 — compare it to the **majority-class baseline** from Lesson 11: always guessing "normal" would score about **0.70** on data that's 70% normal. So a model at **0.61 is actually *worse* than doing nothing**. That baseline comparison is what turns a bare accuracy number into a verdict — without it, 0.61 sounds okay; with it, we know this model is broken.

**(3) Rebuilding the evaluation, step by step.**
- **Split first, before any tuning or cleaning:** carve the 4,000 tickets into three piles, e.g. **~2,400 train / 800 validation / 800 test**. Split *before* preprocessing so no information from validation/test leaks into how you prepare the data.
- **Guard against leakage:** make sure no ticket appears in two piles (watch for duplicates), and that the split ratios roughly preserve the 70/20/10 class mix so each pile is representative.
- **Train candidates on train only.** Say you're deciding between Model A and Model B — fit *both* using just the 2,400 training tickets.
- **Choose on validation.** Score both on the 800 validation tickets and keep the winner. All model selection happens here, so the test set stays untouched.
- **Judge once on test.** Take the single chosen model and measure it **one time** on the 800 test tickets. That number is what you report.
- **Report it against the baseline** (0.70 here) and check the **train-vs-validation gap**: if training accuracy is high but validation is low, you're overfitting — simplify the model or get more data before shipping. If both are low and near the baseline, you're underfitting — add better features or a stronger model.
- **If data were scarce,** you'd swap the single validation set for **cross-validation** (Lesson 23) — but the unbreakable rule is unchanged: the final number comes from data the model neither trained on nor was tuned against.

</details>

---

## WHAT IS NEXT

Chapter 4 gave us the workflow and the honesty rules. **Chapter 5 — Core Supervised Algorithms** starts building the real thing: **Lesson 14 — Linear regression**, our first genuine learning algorithm, which fits a line to data using the gradient descent from Lesson 09 — and which we'll evaluate with exactly the train/validation/test discipline and bias-variance eye this chapter drilled in.

---

*Chapter 4 of the curriculum complete · Phase 2 (Classical Machine Learning) · The Machine Learning Workflow · Lessons 11–13 done.*
