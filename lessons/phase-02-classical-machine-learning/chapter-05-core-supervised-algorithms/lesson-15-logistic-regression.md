# Lesson 15 — Logistic Regression: Predicting a Category

**Phase 2 · Chapter 5 · Lesson 15**
**Position:** Phase 2 (Classical Machine Learning) → Chapter 5 (Core Supervised Algorithms) → Lesson 15 of the curriculum (Lesson 2 of 5 in this chapter).
**Files created:**
- `lesson-notebooks/phase-02-classical-machine-learning/chapter-05-core-supervised-algorithms/lesson-15-logistic-regression.ipynb`
**Files updated:**
- `customer-support-assistant/ticket_urgency.py` (hand-written rule → learned logistic-regression classifier, in place)
- `customer-support-assistant/notebooks/ticket_urgency.ipynb` (companion notebook, upgraded to match)
- `customer-support-assistant/main.py` (urgency now prints a learned probability)
- `progress-tracker.md`
**Prerequisites:** Lesson 14 (the weighted sum `w·x + b`, MSE, two-knob gradient descent — logistic regression reuses all of it), Lesson 09 (gradient descent, learning rate), Lesson 10 (probability, and the idea of a calibrated confidence).
**Project change:** yes — this is the lesson that finally **replaces the toy word-counter** in `ticket_urgency.py` with real machine learning.

---

## SECTION 1 — WHY THIS MATTERS

Last lesson the assistant learned to predict a **number** (resolution minutes). But most of the decisions a support team makes are **yes/no**: *Is this ticket urgent? Is this order fraudulent? Will this customer cancel?* Those are **categories**, and predicting a category is called **classification** — the single most common job in everyday machine learning.

The default tool for classification is **logistic regression**. It is not a fancy new machine: it is linear regression with one extra step bolted on. It computes the same weighted sum `w·x + b`, then passes that number through one squashing function to turn it into a **probability** between 0 and 1, and finally makes a yes/no call. Because it is so simple, so fast, and so interpretable, it is the model you reach for *first* for any classification task — the honest baseline everything fancier has to beat.

For our assistant, this is a milestone. The urgency detector has been a hand-written rule since Lesson 01: count "urgent" words minus "calm" words, call it urgent if the score is above zero. Nobody ever checked that zero was the right cutoff, and it could only ever give a flat label. Logistic regression fixes both problems at once — it **learns** the cutoff from labelled data, and it hands back a **calibrated probability** ("this one is 0.92 urgent") that a team can actually triage on.

---

## SECTION 2 — REAL WORLD ANALOGY

Think about a doctor deciding whether a patient has a fever. They don't jump straight to "sick / not sick." First they read a **thermometer** — one number. Then they mentally turn that number into a **confidence**: 36.5°C feels like "almost certainly fine," 39°C feels like "almost certainly a fever," and 37.5°C is a genuine "hmm, maybe — call it 60/40." Only after forming that confidence do they *decide* what to do.

Logistic regression works exactly like this. The **thermometer reading** is the weighted sum `w·x + b` — one number squeezed out of the inputs. The doctor's **smooth sense of "how likely"** is the sigmoid, bending that number into a probability between 0 and 1. And the doctor's **cutoff for acting** ("above this, I treat it as a fever") is the decision threshold. The key idea the analogy carries: there is a smooth middle ground. The model doesn't lurch from "no" straight to "yes" — it slides through "probably not," "coin flip," "probably yes," and *you* choose where to draw the line.

---

## SECTION 3 — THE CONCEPT EXPLAINED

### From a number to a probability: the sigmoid

Linear regression's weighted sum `w·x + b` can be *any* number — a million, minus a million. But a probability has to live between 0 and 1. So we need a gate that takes any number and squashes it into that range. That gate is the **sigmoid** function ("sigmoid" just means "S-shaped").

**Plain definition:** the sigmoid takes any number `z` and returns a value between 0 and 1. A big **positive** `z` comes out near **1**; a big **negative** `z` comes out near **0**; `z = 0` comes out at exactly **0.5**. It bends smoothly between, so you always get a graded confidence, never a hard flip.

**Why it exists:** without it, a model could output "urgency = 7.3" or "urgency = −40," which is meaningless as a probability. The sigmoid is the translator from "raw score" to "chance between 0% and 100%."

**Tiny concrete example:** `sigmoid(0) = 0.5`, `sigmoid(2) ≈ 0.88`, `sigmoid(−2) ≈ 0.12`, `sigmoid(6) ≈ 0.998`.

**The technical form (meaning first, above):**

```
sigmoid(z) = 1 / (1 + e^(-z))
```

Here `e` is just a fixed number (about 2.718) that shows up all over math; `e^(-z)` means "e raised to the power negative-z." When `z` is large and positive, `e^(-z)` is tiny, so the bottom is ≈1 and the whole thing is ≈1. When `z` is large and negative, `e^(-z)` is huge, so the fraction is ≈0. That is the entire mechanism.

```
   P
 1.0 |                     . - - - - - - -
     |                 . '
     |             . '
 0.5 |- - - - - - +
     |         . '
     |     . '
 0.0 | - '
     +-----------|-------------------------  z = w·x + b
                z = 0
```

### The model, the boundary, and the decision

The whole model is one line:

```
P(urgent) = sigmoid(w·x + b)
```

The inside part `w·x + b` is the same weighted sum as linear regression. It has its own name here: the **logit** — the raw score *before* squashing. `w` and `b` are still the two learned numbers (weight and bias).

To make an actual yes/no call, we pick a **threshold**, usually 0.5: if `P ≥ 0.5`, say "urgent," else "not urgent." The input value where the model is exactly 50/50 is the **decision boundary**. Because `sigmoid` equals 0.5 exactly when its input is 0, the boundary sits where `w·x + b = 0`, which rearranges to:

```
decision boundary:  x = −b / w
```

That is a genuinely useful fact — from just the two learned numbers you can read off the exact cutoff the model settled on.

### Scoring a classifier: log loss, not MSE

To train, we first need to score any given model so we can compare. For classification we do **not** reuse mean squared error. We use **log loss** (also called **cross-entropy**).

**Plain meaning:** for each ticket, look at the probability the model gave to the *correct* answer, and penalise it more the further that probability is from 1 — but on a curve that explodes as you approach a confident mistake.

- If the ticket is truly **urgent** (`y = 1`), the penalty is `−log(P)`. If the model said `P = 0.9`, the penalty is `−log(0.9) = 0.105` (tiny). If it confidently said `P = 0.1`, the penalty is `−log(0.1) = 2.303` — over twenty times worse for the *same* raw distance.
- If the ticket is truly **not urgent** (`y = 0`), the penalty is `−log(1 − P)` instead.

**In one sentence:** log loss rewards being confident and right, and punishes being confident and wrong extremely hard — which is precisely the behaviour you want from something that reports a probability. (Why not MSE? MSE would let a confidently-wrong classifier off far too lightly, and it makes the training math badly behaved for probabilities. Log loss is the natural loss for a yes/no with a probability.)

### How it learns: the same gradient descent, almost unchanged

We minimise log loss with gradient descent — the identical recipe from Lessons 09 and 14: measure the slope of the loss for each knob, step each knob a little downhill, repeat.

Here is the genuinely pleasant part. When you pair the **sigmoid** with **log loss**, the messy pieces cancel and the two slopes come out in the *exact same clean shape* as linear regression:

- **Slope for `w`** = average of `(P − actual) × x`
- **Slope for `b`** = average of `(P − actual)`

where `P` is the predicted probability and `actual` is the true 0 or 1. Compare to Lesson 14's slopes — they are identical in form; the only difference is that `P` here is a squashed probability rather than a raw line value. So the training loop you already wrote barely changes.

### Why the messy pieces cancel (worth seeing once)

"The messy pieces cancel" deserves more than a hand-wave, because that cancellation is the whole reason logistic regression is so pleasant to train. Here it is in plain steps.

Remember the **chain rule** from Lesson 14: the slope of the loss with respect to a knob is built by multiplying a *chain* of smaller slopes — if turning knob A moves B, and B moves C, then A moves C at the two rates multiplied. For one data point, the chain from the weight `w` all the way to the loss `L` has three links:

```
w  ──►  z = w·x + b  ──►  P = sigmoid(z)  ──►  L = log loss
```

The two middle links are the ones that look scary on their own:

- **The sigmoid's slope is unusually tidy.** The rate at which `P` changes as `z` changes turns out to be exactly `P × (1 − P)`. Sanity-check it against the S-shape: in the middle where `P ≈ 0.5`, the slope is `0.5 × 0.5 = 0.25` — the curve's steepest point; out at the ends where `P ≈ 0` or `P ≈ 1`, the slope is `≈ 0` — the curve is flat there. That rise-then-flatten is exactly the S.
- **The log loss's slope is where the mess lives.** The derivative of a `log` puts its input in the *denominator* (the slope of `log(P)` is `1/P`). Working out log loss's slope with respect to `P` leaves `P` and `(1 − P)` sitting on the **bottom** of a fraction — the part that would blow up to infinity as `P` approaches 0 or 1.

Now multiply the two links, as the chain rule says. The sigmoid contributes `P × (1 − P)` on **top**; the log loss contributes `P` and `(1 − P)` on the **bottom**. They divide out:

```
   messy top (from sigmoid)        P · (1 − P)
   ────────────────────────  =  ──────────────────  × (the 0/1 label bits)   ⟶   P − y
   messy bottom (from log)        P   and  (1 − P)
```

Everything scary cancels, and what survives is just `P − y` — the predicted probability minus the true 0/1, with nothing messy attached. The last link (how `z` changes as `w` changes) is simply `x`, exactly as in Lesson 14, so for a single point:

```
slope for w  =  (P − y) · x
slope for b  =  (P − y)
```

Average those over all the points and you have the two bullets above.

**A concrete number.** Take a truly-urgent ticket (`y = 1`) that the model already rates `P = 0.9`. Its per-point slope for `b` is just `P − y = 0.9 − 1 = −0.1`: a gentle nudge, because the model is nearly right. Now a *confidently wrong* one — truly urgent (`y = 1`) but rated `P = 0.1`. The slope is `0.1 − 1 = −0.9`: nine times the push, in the direction that raises `P`. The gradient scales smoothly with how wrong the probability is — no special cases, no explosions.

**Why this matters, not just that it's neat.** This cancellation is the reason the sigmoid and log loss are *always* paired. If you instead paired the sigmoid with **mean squared error**, the `P × (1 − P)` factor would **not** cancel — it would stay stuck to the gradient. And near `P ≈ 0` or `P ≈ 1` — exactly where a confident-but-wrong classifier sits — that factor is `≈ 0`, so the whole gradient nearly vanishes and learning *stalls* right when it should be correcting hardest. Sigmoid + log loss were chosen together as a matched pair precisely so the messy factor divides out and every wrong answer produces a clean, proportional push. That is what "chosen together" really buys you.

### One feature now, many later

We use a single input (a count of urgency-signal words) so the picture is a clean 2-D S-curve you can actually see. Real classifiers use many inputs — `P = sigmoid(w₁x₁ + w₂x₂ + … + b)` — one weight per input plus a bias. Nothing else changes: same sigmoid, same log loss, same gradient descent, just more weights to nudge. Turning raw text into many good features is Chapter 6; measuring a classifier honestly with precision and recall is Chapter 7.

---

## SECTION 4 — THE CODE

The lesson notebook (path in the header) builds a logistic-regression classifier from scratch, no ML library doing the learning:

1. **The sigmoid** — written to avoid overflow; printed at a few inputs to see the squash.
2. **The data** — twelve tickets reduced to one number each (count of urgency-signal words) with a 0/1 label, deliberately overlapping so the S-curve is visible.
3. **The model** — `predict_prob(w, b, x)` returns `sigmoid(w·x + b)`; a rough guess model shows how an untrained classifier behaves.
4. **The loss** — `log_loss(w, b)`, an explicit loop, with the confident-wrong penalty made concrete.
5. **The training loop** — gradient descent with the same-shaped slopes, printing the log loss shrinking.
6. **Read it** — probabilities per signal count, and the decision boundary `x = −b / w`.
7. **Plot** — the smooth S-curve with the real 0/1 tickets and the boundary; saved as a PNG.
8. **Decide** — threshold at 0.5, then accuracy vs. the majority-class baseline.

The code is written plainly — ordinary `for` loops with named variables, one idea per cell, a short comment per line — with one clearly-labelled *compact alternative* note where the counting loops have an idiomatic one-line form worth recognizing later.

> **Verify-as-you-go note:** logistic regression, the sigmoid, and log loss are stable, decades-old fundamentals — no fast-moving API here. When we move to scikit-learn (Chapter 9), the same idea is `LogisticRegression().fit(X, y)` and `.predict_proba(X)`; confirm those method names against the current scikit-learn docs when we get there.

---

## SECTION 5 — IF YOU RAN THIS

Open the notebook in VS Code and run the cells top to bottom. Expected output, labelled a **prediction** (your exact decimals will differ; the *pattern* is the point):

1. **Sigmoid cell** →
   ```
   sigmoid( -6) = 0.002
   sigmoid( -2) = 0.119
   sigmoid(  0) = 0.500
   sigmoid(  2) = 0.881
   sigmoid(  6) = 0.998
   ```
2. **Data cell** → a table of the twelve `(signal_count, label)` pairs, then `Number of tickets: 12`.
3. **Guess-model cell** → probabilities from the `w=0.5, b=0` line, all sitting **above 0.5** — so the untrained model wrongly leans "urgent" even for tickets with zero signal words.
4. **Log-loss cell** → `Log loss of the guess model:` a middling number (predicted roughly **0.56**) — mediocre, which is what we now drive down.
5. **Training loop** → about ten progress lines with the log loss falling steadily, ending near:
   ```
   Trained model: P(urgent) = sigmoid(5.0 * x + -5.0)
   Final log loss: ~0.35
   ```
   (The exact `w` and `b` will vary — `w` a few units positive and `b` a similar amount negative, giving a boundary near `x ≈ 1.5`, is the target.)
6. **Read cell** → probabilities climbing from near 0 at `x=0` up toward 1 by `x=4–5`, then `Decision boundary at x = 1.5x signal words` (approximately).
7. **Plot cell** → `Saved: plots/lesson-15-sigmoid-curve.png`, then an S-curve rising from the cluster of 0-labelled dots on the left to the 1-labelled dots on the right, with a dashed vertical boundary line.
8. **Decision cell** →
   ```
   Model accuracy on the 12 tickets: ~0.83
   Majority-class baseline:          0.58
   ```
   The model clears the baseline — it learned something real, not just "always guess the common class."

The whole lesson is in steps 5–6: gradient descent *discovered* both the steepness of the S-curve and where to place the boundary, straight from the labels. Real numbers shift run to run; the story — loss falling, an S-curve rising through a learned boundary — holds.

---

## SECTION 6 — APPLIED TO OUR ASSISTANT

This is the lesson that retires the hand-written urgency rule. `ticket_urgency.py` is rewritten **in place**: it now reduces each ticket to one number (its count of urgency-signal words), trains a logistic regression by the same from-scratch gradient descent on the eight labelled tickets, and exposes two functions — `urgency_probability(text)` (a 0–1 confidence) and `predict_urgency(text)` (the yes/no call at threshold 0.5). `main.py` now prints the learned probability for each ticket instead of the old raw score.

The philosophical shift is the whole point of this chapter: the old detector applied a cutoff a human picked (`score > 0`); the new one **learned** its cutoff from the labels and reports a calibrated confidence. It is still deliberately small — one feature, eight tickets — so every moving part stays visible. Chapter 6 will let it read the whole ticket instead of a fixed word list, and Chapter 7 will score it properly with precision and recall.

**Predicted `python main.py` output** (run from inside `customer-support-assistant/`), labelled a **prediction**:

```
Customer Support Assistant - current capabilities

1) Ticket urgency detection (learned by logistic regression)
   [urgent] (P(urgent)=0.99) My payment failed twice and I need this resolved right now
   [not urgent] (P(urgent)=0.01) Do you have this item in a larger size

2) Estimated resolution time (learned by linear regression)
   ~11.0 min (11 words) My payment failed twice and I need this resolved right now
   ~10.0 min (9 words) Do you have this item in a larger size

Self-check passed.
```

(The exact probabilities in line 1 will vary a little with training, but the pattern is firm: near 1 for the signal-heavy urgent ticket, near 0 for the calm one. The resolution estimates in section 2 are unchanged from Lesson 14.) Running `python ticket_urgency.py` on its own prints the learned model, its decision boundary, the two demo probabilities, and passes its own asserts. `requirements.txt` stays empty — this classifier is pure standard-library Python (`math` only).

---

## SECTION 7 — COMMON MISTAKES AND GOTCHAS

- **Calling it "regression" and expecting a number.** The name is historical. Logistic *regression* is a **classifier** — it predicts a category. It borrows linear regression's weighted sum, then squashes it; the output is a probability, not a quantity.
- **Using mean squared error to train it.** MSE is for numbers. For a probability, use **log loss** — it punishes confident-wrong answers correctly and keeps the gradient well-behaved. Training a classifier on MSE tends to learn slowly and settle badly.
- **Treating the 0.5 threshold as sacred.** 0.5 is only a default. If missing an urgent ticket is far costlier than a false alarm, *lower* the threshold so more tickets get flagged. The probability is the honest output; the threshold is a business choice (precision/recall, Chapter 7).
- **Confusing the probability with truth.** `P(urgent) = 0.8` means "the model is 80% confident," not "this ticket is 80% urgent." A well-*calibrated* model is right about 80% of the time when it says 0.8 — but calibration has to be checked, not assumed.
- **Perfectly separable tiny data → runaway weights.** When the classes don't overlap (as in our eight-ticket toy), the weights can grow very large chasing ever-more-confident probabilities. It still works here, but on real data you add regularisation (a later topic) to keep weights sane.

---

## SECTION 8 — WHEN TO USE THIS, AND TRADEOFFS

**Reach for logistic regression when** you need to predict a category (especially yes/no) and you want a fast, interpretable baseline that also gives you a probability. It is the correct *first* model for almost any classification task: spam / not spam, fraud / legit, will-churn / won't, urgent / not.

**The tradeoffs:**
- **Strengths:** simple, fast, needs little data, hard to overfit with few features, and fully interpretable — each weight reads as "how much this input pushes the odds toward yes," and you get a calibrated probability for free.
- **Limits:** it can only draw a **straight** decision boundary (a line, or a flat sheet in higher dimensions). If the true boundary curves or the classes interlock, a plain logistic regression underfits — you'd engineer features or move to a curve-capable model (trees, Lesson 16; neural nets, Phase 3).
- **It predicts categories, not quantities.** For "how many minutes / how many dollars," that's linear regression (Lesson 14), its number-predicting cousin.

The honest rule: start with logistic regression as your classification baseline. If a straight boundary and a handful of features get the job done, you're finished and you have a model anyone can explain. If its errors stay stubbornly high, that's your signal the pattern is more tangled — and now you know *why* you need something fancier.

---

## SECTION 9 — KEY TAKEAWAYS

- **Logistic regression is linear regression plus a sigmoid: it computes the same weighted sum `w·x + b`, then squashes it into a probability between 0 and 1, turning a number-predictor into a category-predictor.**
- **The model makes a decision by thresholding that probability (usually at 0.5), and its decision boundary — the exact cutoff — sits at `x = −b / w`, readable straight from the two learned numbers.**
- **Classifiers are trained with log loss, not mean squared error, because log loss rewards confident-correct answers and punishes confident-wrong ones hard — the right incentive for a probability.**
- **It learns by the same gradient descent as before, and thanks to the sigmoid-plus-log-loss pairing the slopes have the identical clean shape as linear regression: the average of `(prediction − actual)`, weighted by the input for the weight.**
- **Our assistant's urgency detector is now a genuinely learned classifier that reports a calibrated probability, replacing the hand-written `score > 0` rule it used since Lesson 01.**

---

## SECTION 10 — CHALLENGE WITH HIDDEN ANSWER

**Challenge:** A shop trains a logistic regression to predict whether an order is **fraudulent** from a single number: how many times the customer changed their shipping address in the last hour. After training, the model is:

> `P(fraud) = sigmoid(1.5 × changes − 3)`

Answer these:
1. What is the decision boundary — how many address changes tips the model to "fraud" (using the usual 0.5 threshold)?
2. Compute `P(fraud)` for an order with **4** address changes. Does it get flagged?
3. The fraud team says a missed fraud costs them far more than a false alarm. Should they raise or lower the 0.5 threshold, and what happens to the boundary?

<details>
<summary>Click to reveal the answer</summary>

**1) The decision boundary.**
The model hits 0.5 exactly when the logit is 0, i.e. when `1.5 × changes − 3 = 0`. Solving: `changes = 3 / 1.5 = 2`. So the boundary is **2 address changes** — at 2 changes the model is a 50/50 coin flip; below 2 it leans "legit," above 2 it leans "fraud." (This is the `x = −b / w` rule: `−(−3) / 1.5 = 2`.)

**2) Four address changes.**
Logit = `1.5 × 4 − 3 = 6 − 3 = 3`. So `P(fraud) = sigmoid(3) ≈ 0.95`. Since `0.95 ≥ 0.5`, the order **is flagged as fraud**, with high (95%) confidence.

**3) Raise or lower the threshold?**
**Lower** it (say to 0.3). A missed fraud is the expensive mistake, so we want to catch more suspicious orders even at the cost of more false alarms — a lower threshold flags more orders. Concretely, lowering the threshold to 0.3 moves the *acting* boundary to a smaller number of address changes (you now flag orders the model is only 30% sure about), so more orders get reviewed. The trained model — its `w`, `b`, and its natural 0.5 boundary at 2 changes — doesn't change at all; you're only changing where *you* choose to act on its probability. That trade-off between catching more real fraud (recall) and raising fewer false alarms (precision) is exactly what Chapter 7 makes precise.

</details>

---

## SECTION 11 — WHAT IS NEXT

You now have both classical workhorses: linear regression for numbers, logistic regression for categories — and both draw a *straight* line. Next comes a model that carves the world with **yes/no questions** instead: **Lesson 16 — Decision trees and random forests**, which can bend around curved, interlocking patterns a straight boundary can't, and whose "forest of trees" idea is one of the most reliable tools in all of classical machine learning.

---

*Lesson 15 of the curriculum · Phase 2 (Classical Machine Learning) · Chapter 5 (Core Supervised Algorithms) · Lesson 2 of 5 in this chapter.*
