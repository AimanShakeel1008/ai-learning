# Lesson 14 — Linear Regression: Predicting a Number

**Phase 2 · Chapter 5 · Lesson 14**
**Position:** Phase 2 (Classical Machine Learning) → Chapter 5 (Core Supervised Algorithms) → Lesson 14 of the curriculum (Lesson 1 of 5 in this chapter).
**Files created:**
- `lesson-notebooks/phase-02-classical-machine-learning/chapter-05-core-supervised-algorithms/lesson-14-linear-regression.ipynb`
- `customer-support-assistant/resolution_time.py`
- `customer-support-assistant/notebooks/resolution_time.ipynb`
**Files updated:** `customer-support-assistant/main.py`, `progress-tracker.md`
**Prerequisites:** Lesson 09 (loss / mean squared error, gradient descent — this lesson *is* that machinery aimed at a real task), Lesson 11 (the train→evaluate→use pipeline), Lesson 13 (the fitting failure modes we'll keep watching for).
**Project change:** yes — the assistant gets its first genuinely **learned** model: `resolution_time.py`, which estimates a ticket's resolution time in minutes.

---

## SECTION 1 — WHY THIS MATTERS

Everything the assistant has done so far, a human hand-wrote the rule for. The urgency detector "learns" word counts, but *we* decided that urgent words add and calm words subtract. This lesson crosses a real line: for the first time, we hand the machine some data and let **it** discover the rule, with no human writing down the relationship.

The tool is **linear regression** — the simplest real machine-learning algorithm, and the one every practitioner reaches for first when the answer is a **number** (minutes, price, temperature, demand). It draws the single straight line that best fits your data, then reads predictions off that line. It's simple enough to build from scratch in an afternoon, yet the exact same idea — parameters trained by gradient descent to minimise a loss — is the seed that grows into neural networks and, eventually, large language models.

For our customer-support assistant, it powers a genuinely useful new feature: **estimated resolution time**. Given how long a ticket is, predict how many minutes it'll take an agent to resolve — the kind of number a support team uses to triage a queue and set expectations. And unlike the urgency toy, no human writes the "minutes per word" rule; the model learns it.

---

## SECTION 2 — REAL WORLD ANALOGY

Imagine you're a moving company giving quotes. Over years you've noticed: the more boxes a customer has, the longer the job takes. You don't have a formula — just a gut feel built from hundreds of past jobs.

Now picture every past job as a dot on a wall chart: boxes across the bottom, hours up the side. The dots scatter, but they clearly drift upward — more boxes, more hours. You grab a piece of string and stretch it across the chart so it runs right through the middle of the cloud of dots, as close to as many as possible. That taut string is your **line of best fit**.

When a new customer calls and says "I have 40 boxes," you find 40 along the bottom, go up to the string, and read the hours across — that's your quote. Linear regression is exactly this, made precise: the dots are your data, the string is the model, "running through the middle of the dots" is *training*, and reading a new value off the string is *prediction*. The only thing the math adds is a rule for placing the string in the provably-best spot instead of by eye.

---

## SECTION 3 — THE CONCEPT EXPLAINED

### The model is a straight line

**Plain definition:** linear regression models the answer as a straight-line function of the input. Feed in a number `x`, get out a number `y`, using the line `y = w × x + b`.

A line is pinned down by exactly two numbers, and each has a plain meaning:

- **`w` — the slope.** Its technical name is the **weight**. It answers: *for each extra unit of input, how much does the output change?* If `w = 5`, every extra unit of `x` adds 5 to `y`.
- **`b` — the intercept.** Its technical name is the **bias**. It's the output when `x = 0` — the baseline the slope builds on. (This "bias" means a constant offset, nothing to do with the "high bias" of Lesson 13. Same word, different job — worth keeping straight.)

Together `w` and `b` are the model's **parameters**: the numbers the model *learns*. Here's the key mental shift — **a trained linear-regression model is literally just those two numbers.** Save `w` and `b` and you've saved the whole model.

**Tiny concrete example:** say we land on `w = 4.9` and `b = 5.6` for "minutes to resolve a ticket, from its length in tens of words." A ticket of 30 words is `x = 3`, so the prediction is `4.9 × 3 + 5.6 = 20.3` minutes. A 50-word ticket is `x = 5` → `4.9 × 5 + 5.6 = 30.1` minutes. The line does all the work.

### Measuring how good a line is: the loss

Before we can find the *best* line, we need to score *any* line, so we can compare. We reuse **mean squared error (MSE)** from Lesson 09.

Plain meaning first: for each data point, look at how far the line's prediction is from the true value — that gap is the **error**. Square it (so a miss above and a miss below both count as positive, and a big miss is punished far more than a small one). Average those squared errors over all points. That average is the loss. Symbols second:

```
MSE = average over all points of (prediction − actual)²
```

Worked example — one point at `x = 2`, true value `14`, using the line `w = 3, b = 0`:
- prediction = `3 × 2 + 0 = 6`
- error = `6 − 14 = −8`
- squared error = `64`

A line hugging the dots gives a small MSE; a badly-placed line gives a big one. **Training = finding the `w` and `b` that make this number as small as possible.**

### How it learns: gradient descent, now with two knobs

Lesson 09 turned one knob downhill. Linear regression has **two** knobs (`w` and `b`), so the gradient is **two** slopes — one saying which way to move `w`, one for `b`. For MSE with a straight line these come out clean (plain meaning first):

- **Slope for `w`** = the average of `(prediction − actual) × x`. Read it as: if the line predicts too high especially on the far-right points (large `x`), tilt `w` downward.
- **Slope for `b`** = the average of `(prediction − actual)`. Read it as: if the line sits too high everywhere, lower the whole line.

Then the update rule from Lesson 09, applied to both at once:

```
w  ←  w  −  learning_rate × (slope for w)
b  ←  b  −  learning_rate × (slope for b)
```

The **learning rate** is the step size — small (like 0.01) is slow but safe; too big and the steps overshoot and blow up (the divergence you saw in Lesson 09). Repeat the update a couple thousand times: each pass the line tilts and shifts a hair toward the dots and the loss drops, until it settles at the bottom — the line of best fit.

```
 minutes                           minutes
   |          . .  /                 |          . ./
   |      . ./  .                    |      . ./ .
   |   . /  .           many         |   . /. 
   | ./  .          steps of         | ./ .
   |/______________  gradient  --->  |/______________
   start: flat line     descent      end: best-fit line
   (big loss)                        (smallest loss)
```

### One feature now, many features later

We're using a single input (ticket length), so the model is a line in 2-D. Real problems use many inputs at once — length **and** number of question marks **and** customer age — and then the model is `y = w₁x₁ + w₂x₂ + w₃x₃ + b`, a weight per input plus one bias. The picture becomes a flat sheet in higher dimensions instead of a line, but **nothing else changes**: same MSE loss, same gradient descent, just more weights to nudge. Master the one-input case and the many-input case is only bookkeeping. (Multiple features properly arrive in Chapter 6.)

### A note on scaling

You'll see the code divide lengths by 10 before training. Gradient descent gets unstable when inputs are large: the slopes scale with `x`, so big `x` makes giant steps that overshoot. Shrinking the input to a friendly range (here 1–8 instead of 10–80) keeps the steps sane. This is **feature scaling**, a standard preprocessing step we cover properly in Lesson 21 — for now, just know it's why the code works in "tens of words."

---

## SECTION 4 — THE CODE

The lesson notebook (path in the header) builds the whole thing from scratch, no ML library doing the learning:

1. **The data** — eight past tickets as `(length, minutes)`, plus a scaled copy of the lengths.
2. **The model** — `predict_with(w, b, x)` returns `w × x + b`; we try a bad guess line to feel how wrong an untrained model is.
3. **The loss** — `mean_squared_error(w, b)`, an explicit loop, scoring the guess line.
4. **The gradients** — `compute_gradients(w, b)` returns both slopes; we print them at the flat start so you can see which way is downhill.
5. **The training loop** — 2000 gradient-descent steps, printing the loss shrinking, ending in the trained `w` and `b`.
6. **Use it** — predict minutes for brand-new ticket lengths (including one past the training range, to flag extrapolation).
7. **Plot** — scatter the dots and draw the fitted line; save a PNG.
8. **Check** — compare our gradient-descent line to `np.polyfit`'s exact line; they should nearly match.

The code is written plainly — ordinary `for` loops with named variables, one idea per cell, a short comment per line — so it reads top to bottom on its own.

> **Verify-as-you-go note:** the check cell uses `np.polyfit(x, y, 1)`, which returns coefficients **highest-power-first**, i.e. `[slope, intercept]` for a line. That's standard NumPy, but confirm the argument order and return shape against the current NumPy docs if anything looks off.

---

## SECTION 5 — IF YOU RAN THIS

Open the notebook in VS Code and run the cells top to bottom. Expected output, labelled as a **prediction** (your exact numbers will differ slightly; the *pattern* is the point):

1. **Data cell** → a table of the eight tickets with `length`, `scaled(x)`, `minutes`, then `Number of tickets: 8`.
2. **Guess-line cell** → a table showing the `w=3, b=0` line under-predicting most tickets (errors mostly negative — the line is too shallow and starts at zero).
3. **Loss cell** → `MSE of the guess line (w=3.0, b=0.0):` a large number (predicted roughly **220**) — that shallow, floor-starting line is a bad fit.
4. **Gradients cell** → at `w=0, b=0`, `slope for w:` about **−300** and `slope for b:` about **−55**, both negative, meaning "push both up."
5. **Training loop** → about ten progress lines with the loss falling fast then levelling off, ending near:
   ```
   Trained line: minutes = 4.893 * (length/10) + 5.607
   Final loss: ~2.0
   ```
   (The exact `w`, `b`, and final loss will vary a little; `w` near 4.9 and `b` near 5.6 is the target.)
6. **Predict cell** →
   ```
   A 15-word ticket -> about 12.9 minutes to resolve
   A 55-word ticket -> about 32.5 minutes to resolve
   A 100-word ticket -> about 54.5 minutes to resolve
   ```
7. **Plot cell** → `Saved: plots/lesson-14-line-of-best-fit.png`, then a scatter of eight dots with a red line running straight through their middle.
8. **NumPy check** → our line and NumPy's printed side by side, both about `w=4.89, b=5.61` — two different methods, the same best-fit line.

The whole lesson is in step 5: starting from a flat line on the floor, gradient descent *discovered* the slope and intercept that fit the data, and step 8 confirms it found the genuinely optimal line. Real numbers shift run to run; the story — loss falling, `w`→~4.9, `b`→~5.6 — holds.

---

## SECTION 6 — APPLIED TO OUR ASSISTANT

This lesson gives the assistant its **first learned model**. New file `resolution_time.py` trains a linear regression (by the same from-scratch gradient descent) on past `(length, minutes)` tickets, then exposes `predict_resolution_minutes(length_words)` — an estimate of how long a new ticket will take to resolve. `main.py` now runs this alongside the urgency detector, estimating resolution time for each sample ticket, and asserts that a longer ticket is estimated to take at least as long as a shorter one.

Note what changed philosophically: the urgency detector applies a rule we designed; `resolution_time.py` applies a rule *it learned from data*. That's the whole leap into real machine learning, and every model from here builds on it. It's still deliberately simple — one feature, a tiny dataset, a straight line — so it's the perfect place to have watched every moving part.

**Predicted `python main.py` output** (run from inside `customer-support-assistant/`), labelled a **prediction**:

```
Customer Support Assistant - current capabilities

1) Ticket urgency detection
   [urgent] (score 3) My payment failed twice and I need this resolved right now
   [not urgent] (score -1) Do you have this item in a larger size

2) Estimated resolution time (learned by linear regression)
   ~11.0 min (11 words) My payment failed twice and I need this resolved right now
   ~10.0 min (9 words) Do you have this item in a larger size

Self-check passed.
```

(The urgency scores in line 1 are illustrative — the real numbers come from the existing word-count model; the resolution estimates near 10–11 minutes are what the trained line predicts for 9–11-word tickets.) Running `python resolution_time.py` on its own prints the learned line and passes its own asserts. `requirements.txt` stays empty — this feature is pure standard-library Python.

---

## SECTION 7 — COMMON MISTAKES AND GOTCHAS

- **Forgetting linear regression can only draw a line.** If the true relationship curves (say, resolution time flattens for very long tickets), a straight line will systematically miss — that's underfitting / high bias from Lesson 13. The fix is more features or a curve-capable model, not more gradient-descent steps.
- **A learning rate that's too big.** Large steps overshoot the valley and the loss explodes to `inf`/`nan` instead of shrinking. If you see the loss growing, shrink the learning rate first.
- **Skipping feature scaling.** Train on raw lengths (10–80) with the same learning rate and gradient descent likely diverges. Big inputs → big slopes → giant steps. Scaling the input to a small range is what keeps it stable.
- **Trusting predictions far outside the training range.** The model learned from 10–80-word tickets; asking it about a 500-word ticket is **extrapolation** — the line keeps going, but there's no data out there to say it should. Treat far-out predictions with suspicion.
- **Confusing the two meanings of "bias."** The intercept `b` is called the bias (a constant offset). "High bias" in Lesson 13 means an underfitting model. Same word, unrelated ideas.

---

## SECTION 8 — WHEN TO USE THIS, AND TRADEOFFS

**Reach for linear regression when** the thing you're predicting is a continuous number and you expect a roughly straight-line relationship with your inputs. It's the right *first* model for almost any number-prediction task: fast to train, impossible to overcomplicate, and — its best feature — completely **interpretable**. You can read each weight as "this much output per unit of this input" and explain the model to a non-technical colleague in one sentence.

**The tradeoffs:**
- **Strengths:** simple, fast, needs little data, hard to overfit with few features, and every parameter has a plain meaning.
- **Limits:** it can *only* fit straight-line (linear) relationships. Real curves, thresholds, and interactions between features are beyond a plain line — you'd add engineered features or move to a more flexible model (trees, Lesson 16; neural nets, Phase 3).
- **It predicts numbers, not categories.** For "which bucket does this belong to?" (urgent vs not, spam vs not) you want its cousin **logistic regression** — which is the very next lesson.

The honest rule: start with linear regression as your baseline. If a straight line is good enough, you're done and you have a model anyone can understand. If its errors stay stubbornly high, that's your signal the relationship is more complex — and now you know *why* you need something fancier.

---

## SECTION 9 — KEY TAKEAWAYS

- **Linear regression predicts a number by fitting the single best straight line through the data, and a trained model is nothing more than two learned numbers — a slope (weight) and an intercept (bias).**
- **"Best" is defined by a loss — mean squared error, the average squared gap between prediction and truth — and training means finding the weight and bias that make that loss as small as possible.**
- **It learns by gradient descent from Lesson 09, now nudging two knobs at once: each step measures the slope of the loss for the weight and for the bias, then steps both a little downhill, repeated until the loss stops falling.**
- **The same recipe (parameters + a loss + gradient descent) scales from one input to many and is the exact foundation that neural networks and modern AI are built on — linear regression is the simplest instance of the idea that runs through the whole field.**
- **Our assistant now has its first truly learned capability: `resolution_time.py` estimates a ticket's resolution time from a rule the model discovered from data, not one a human wrote.**

---

## SECTION 10 — CHALLENGE WITH HIDDEN ANSWER

**Challenge:** A shop trains a linear regression to predict a product's **shipping cost in dollars** from its **weight in kilograms**. After training, the model is:

> `cost = 2.5 × weight + 4`

Answer these:
1. In plain words, what do the `2.5` and the `4` each mean?
2. Predict the shipping cost for a 6 kg parcel.
3. The model was trained only on parcels between 1 kg and 10 kg. A customer asks about a 200 kg pallet and the formula says `cost = 2.5 × 200 + 4 = $504`. Why should you distrust that number even though the arithmetic is correct?

<details>
<summary>Click to reveal the answer</summary>

**1) What the two numbers mean.**
- The **`2.5` is the slope (weight):** each extra kilogram adds **$2.50** to the shipping cost. It's the "dollars per kilogram" rate the model learned.
- The **`4` is the intercept (bias):** the baseline cost when weight is `0` — a fixed **$4** that applies before any per-kilogram charge (think handling/packaging). Read together: "$4 to start, plus $2.50 per kg."

**2) The 6 kg prediction.**
`cost = 2.5 × 6 + 4 = 15 + 4 = $19`. Since 6 kg sits comfortably inside the 1–10 kg training range, this is a prediction we can reasonably trust.

**3) Why distrust the $504 for 200 kg.**
Because 200 kg is **far outside the training range** (1–10 kg) — this is **extrapolation**. The model only ever saw small parcels, so it learned a straight-line rate that fit *those*. There's no guarantee the real cost keeps rising at a flat $2.50/kg all the way to 200 kg: heavy freight often jumps to different pricing tiers, needs a pallet or a truck, or is capped — none of which the line can know, because no data out there taught it. The arithmetic is fine; the *assumption that the same straight line still applies* is the shaky part. The fix is to gather data in that range (or use a model/rule built for freight), not to trust the line blindly.

</details>

---

## SECTION 11 — WHAT IS NEXT

You've built a model that predicts a **number**. Next comes its close cousin for predicting a **category**: **Lesson 15 — Logistic regression**, which takes the same weighted-sum idea, squashes it into a probability between 0 and 1, and turns that into a yes/no decision — the workhorse classifier you'll reach for first, and a much more principled replacement for the assistant's hand-written urgency rule.

---

*Lesson 14 of the curriculum · Phase 2 (Classical Machine Learning) · Chapter 5 (Core Supervised Algorithms) · Lesson 1 of 5 in this chapter.*
