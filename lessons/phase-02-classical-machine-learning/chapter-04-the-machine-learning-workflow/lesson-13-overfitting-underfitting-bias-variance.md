# Lesson 13 — Overfitting, Underfitting, and the Bias-Variance Idea

**Phase 2 · Chapter 4 · Lesson 13**
**Position:** Phase 2 (Classical Machine Learning) → Chapter 4 (The Machine Learning Workflow) → Lesson 13 of the curriculum (Lesson 3 of 3 in this chapter).
**Files created:** `lesson-notebooks/phase-02-classical-machine-learning/chapter-04-the-machine-learning-workflow/lesson-13-overfitting-underfitting-bias-variance.ipynb`
**Files updated:** `progress-tracker.md`
**Prerequisites:** Lesson 12 (train/validation/test split — the gap we use here), Lesson 09 (loss / mean squared error, gradient descent), Lesson 10 (noise, seeds, spread), Lesson 07 (plots).
**Project change:** none — this is a theory-and-demo lesson. It names the failure mode (overfitting) that every real model we build from Lesson 14 onward must guard against.

---

## SECTION 1 — WHY THIS MATTERS

Lesson 12 taught you to hide data so you get an honest score. This lesson is about *what that honest score reveals*. When a model does well on its training data but badly on new data, it has **overfit** — it memorised instead of learning. When it does badly on both, it has **underfit** — it was too simple to catch the real pattern at all. Almost every disappointing model in the real world is one of these two.

Here's the sharp point: the training score alone can't tell you which is happening, or even that anything is wrong. A memorising model shows a beautiful training score right up until it fails in production. The only way to *see* the problem is to compare the training score with the held-out (validation) score — the exact split we built last lesson. The size of the **gap** between them is a diagnostic tool you'll use for the rest of your AI life.

For our customer-support assistant, this is the difference between "92% accurate in my notebook" and "92% accurate on next week's real tickets." Only the second one keeps customers happy, and only understanding overfitting gets you there.

---

## SECTION 2 — REAL WORLD ANALOGY

Think about three students preparing for a driving test.

- **The under-prepared student** barely studied and can't do much of anything — they fail the practice drives *and* the real test. They're **underfitting**: too little learned to handle either. In model terms, **high bias**.
- **The genuinely skilled student** learned to actually drive — reading the road, judging speed. They do well on practice drives *and* on the real test, because the skill transfers. This is the **just-right** model.
- **The rote memoriser** memorised the exact route of one specific practice course — every pothole, every turn, in order. On that one course they look flawless. Put them on any new road and they fall apart, because they learned the *course*, not *driving*. They're **overfitting**: memorising the training route, including its quirks. In model terms, **high variance**.

Map it: the practice course = **training data**, a brand-new road = **validation/test data**, "memorising the exact route including potholes" = fitting the **noise**, and "learning to actually drive" = capturing the **signal**. A great training score with a poor new-road score is the memoriser's signature — and exactly what the train-vs-validation gap detects.

---

## SECTION 3 — THE CONCEPT EXPLAINED

### Signal versus noise

**Plain definition — signal:** the real, repeating pattern in the data, the thing that will still be true for the next customer.
**Plain definition — noise:** the random, one-off wobble in each data point that won't repeat — a customer's mood, a fluke, measurement error.

**Why it matters:** every real dataset is *signal + noise* mixed together. A good model should learn the signal and ignore the noise. The trouble is the model can't see a label saying "this part is noise" — it just sees points. If we let it, a flexible-enough model will happily bend to fit the noise too, and that's overfitting.

**Tiny concrete example:** suppose satisfaction really does follow a gentle hump as tickets get longer (`signal = 3 + 1.2x − 0.15x²`). One customer at `x = 4` *should* sit near `3 + 4.8 − 2.4 = 5.4`, but their actual rating came in at `6.1` — that extra `+0.7` is noise, personal to them. A model that twists itself to pass exactly through `6.1` has learned that one customer's mood, which tells it nothing about the next customer.

### Underfitting = too simple = high bias

**Plain definition:** the model is too simple to capture even the real pattern, so it does poorly everywhere — on training data *and* on new data.

**What it looks like in numbers:** training error **high**, validation error **high**, and the two are close together (small gap, both bad).

**Tiny example:** trying to describe a hump-shaped pattern with a straight line. No straight line can bend into a hump, so it's wrong on nearly every point, whether that point was in training or not.

**Technical name:** this is **high bias** — the model is "biased" toward a shape too plain for the data. (Bias here means a systematic, built-in limitation, not prejudice.)

### Overfitting = too complex = high variance

**Plain definition:** the model is so flexible it fits the training points almost perfectly — including their noise — but that memorised detail doesn't carry over, so it does badly on new data.

**What it looks like in numbers:** training error **tiny**, validation error **high**, a **big gap** between them. That gap is the tell.

**Tiny example:** a curve with fifteen bends threaded through sixteen training points passes through nearly every one (tiny training error), but between the points it swings wildly up and down, so the held-out points — which fall *between* the training points — are way off.

**Technical name:** this is **high variance** — "variance" because the fitted curve would swing dramatically if you gave it a slightly different sample of training points. It's chasing the sample, not the truth.

### The one diagnostic: the train-vs-validation gap

Put the two errors side by side and read them like a doctor:

```
   train error   validation error   diagnosis
   -----------   ----------------   -----------------------------
   HIGH          HIGH  (close)      UNDERFIT  (too simple / high bias)
   LOW           LOW   (close)      JUST RIGHT (keep this one)
   LOW (tiny)    HIGH  (big gap)    OVERFIT   (too complex / high variance)
```

This is *why* Lesson 12's split exists. With training data alone you only ever see the left column, which looks great in every row. The held-out score is what exposes overfitting.

### The bias-variance tradeoff, and the U-shape

**Plain definition:** as you make a model more complex, it underfits less (bias falls) but starts to overfit more (variance rises). You can't push both to zero; you trade one against the other and aim for the balance point.

Watch what happens to the two error curves as complexity climbs from simple to complex:

```
 error
   |*                                           . <- validation error (U-shaped)
   | *.                                       .
   |   *..                                 ..
   |      *...        best             ...
   |          *.....    |         ....
   |               *...........................  <- training error (always falling)
   |________________________|__________________
    simple            complexity           complex
                     (sweet spot = lowest VALIDATION error)
```

- **Training error only ever falls** as complexity rises — more flexibility can always hug the training points tighter. On its own it's a liar: it keeps improving even as the model gets worse for real use.
- **Validation error falls, bottoms out, then rises** — a **U**. The left wall of the U is underfitting (too simple), the right wall is overfitting (too complex), and the bottom is the model you want.

**The rule:** pick the complexity at the **lowest validation error**, never the lowest training error.

### What you do about each

- **Underfitting (high bias):** give the model more power — more complexity, better/more features, or train it longer.
- **Overfitting (high variance):** rein it in — a simpler model, **more training data** (harder to memorise a large pile), or techniques that penalise complexity (regularization, dropout, early stopping — all coming in the deep-learning chapters).

More data is the one fix that helps overfitting almost for free: the more examples, the harder it is to memorise them all, so the model is nudged toward learning the general signal instead.

---

## SECTION 4 — THE CODE

The notebook (path in the header) tells the whole story on data where we secretly know the truth, so we can judge every fit fairly:

1. **Make data** — a gentle hump-shaped **signal** plus random **noise**, 24 points.
2. **Split** — hold out every third point as validation (Lesson 12), spread across the whole range.
3. **MSE helper** — the loss from Lesson 09, written as an explicit loop.
4. **Three fits** — polynomials of degree 1 (too simple), 3 (just right), 15 (too complex); print each one's training error, validation error, and the gap.
5. **Draw the three fits** over the data so under/over-fitting is visible; save a PNG.
6. **Sweep degrees 1–15** — record both errors at every complexity, find the lowest-validation degree, and plot the falling training curve against the U-shaped validation curve; save a PNG.

Concept and *why* live in the markdown cells; each code cell has short per-line comments so it reads top to bottom on its own.

> **Verify-as-you-go note:** the fitting uses `np.polyfit(x, y, degree)` (finds the best polynomial coefficients) and `np.polyval(coeffs, x)` (evaluates that polynomial). These are standard NumPy, but confirm the signatures in the current NumPy docs. At high degree, `np.polyfit` may print a `RankWarning` (the fit is numerically unstable — which is itself part of the overfitting story); that warning is expected here, not a bug.

---

## SECTION 5 — IF YOU RAN THIS

Open the notebook in VS Code and run the cells top to bottom. Expected output, labelled as a **prediction** (your exact numbers will differ; the *pattern* is the point):

1. **Data cell** → a table of the first 5 points showing signal, noise, and observed `y`, then `Total points: 24`.
2. **Split cell** → `Training points: 16`, `Validation points: 8`, and the held-out indices `[2, 5, 8, 11, 14, 17, 20, 23]`.
3. **MSE check** → `MSE of a perfect guess: 0.0` and `MSE with one point off by 1: 0.333...`.
4. **Three-fits cell** (predicted shape of the numbers):
   ```
   Degree 1 - TOO SIMPLE (underfit)
      training error (points it learned): ~4.5
      validation error (unseen points)  : ~5.0
      gap (validation - training)        : small

   Degree 3 - JUST RIGHT
      training error (points it learned): ~1.8
      validation error (unseen points)  : ~2.0
      gap (validation - training)        : small

   Degree 15 - TOO COMPLEX (overfit)
      training error (points it learned): ~0.2
      validation error (unseen points)  : ~40+
      gap (validation - training)        : huge
   ```
5. **Three-fits plot** → `Saved: plots/lesson-13-three-fits.png`, then three panels: a flat-ish line missing the hump; a curve tracing the hump; a violently wiggly curve touching training dots but flying past the red validation squares.
6. **Sweep cell** → 15 lines of `degree | train error | validation error`, with training error sliding steadily down and validation error dropping then climbing; then `Best degree by validation error: 3` (or nearby, e.g. 2–4).
7. **Sweep plot** → `Saved: plots/lesson-13-bias-variance-curve.png`: a black training curve heading down toward zero and a red validation curve forming a U, with a green line at the best degree.

The lesson lives in two contrasts: degree 15 has the *best* training error and the *worst* validation error (overfitting in one line), and the validation curve's U tells you the right complexity is in the middle, not at either extreme. Real numbers will shift, but the shapes — training always falling, validation a U — will hold.

---

## SECTION 6 — APPLIED TO OUR ASSISTANT

Nothing to implement in the project today — this is a diagnosis lesson, not a feature. But it sets the standard for every model we build starting next chapter. When we train the assistant's real ticket-category classifier (Lesson 15 onward), we will always print *both* its training score and its validation score and watch the gap: a big gap means it's memorising tickets and we simplify or add data; both scores low and close means we can trust it. The toy word-counter in `ticket_urgency.py` stays exactly as it is, and `python main.py` still runs the project end to end, unchanged.

---

## SECTION 7 — COMMON MISTAKES AND GOTCHAS

- **Trusting a great training score.** It almost always looks good, even for a badly overfit model. Never celebrate a number measured on data the model learned from — check the held-out score.
- **Assuming more complexity is always better.** Bigger/fancier models overfit *more* readily on small data. Past the bottom of the U, added complexity makes real-world performance worse, not better.
- **Confusing the two failures.** Both scores high = underfit (add power). Low train, high validation = overfit (remove power / add data). The fixes are opposite, so diagnosing wrong makes it worse.
- **Reading noise as signal.** If validation error is stuck high no matter what you try, some of what looks like a pattern may just be irreducible noise — no model can predict a coin flip.
- **Tiny validation sets give a jumpy U.** With only 8 held-out points, the exact best degree wobbles. The *shape* (falling train, U-shaped validation) is reliable; the precise winning degree is not, on small data.

---

## SECTION 8 — WHEN TO USE THIS, AND TRADEOFFS

This isn't an optional technique — checking for over/underfitting is something you do for **every** model you ever train. The moment you have a training score and a validation score, you compare them. It costs nothing beyond the split you already made.

The real tradeoff is *where to sit on the complexity dial*:
- **Simpler models** (fewer features, lower capacity) are less likely to overfit, faster, and easier to explain — but risk underfitting a genuinely complex pattern.
- **More complex models** can capture rich patterns but demand more data and more care to avoid overfitting.
- **The honest guide:** start simple, watch the gap, and add complexity only while validation error keeps falling. Stop when it starts to rise.

When data is scarce, a single validation set gives a noisy read on where the U bottoms out; **cross-validation** (Lesson 23) averages several splits for a steadier estimate. The principle never changes: the number that decides is measured on data the model didn't learn from.

---

## SECTION 9 — KEY TAKEAWAYS

- **Underfitting** means the model is too simple and does poorly on both training and new data; its technical name is **high bias**, and the fix is to give the model more power.
- **Overfitting** means the model memorised the training data (including its noise) and does well there but poorly on new data; its technical name is **high variance**, and the fix is a simpler model or more data.
- **The gap between the training score and the validation score is the diagnostic** — small gap with both good is healthy, a large gap with a great training score is overfitting — which is exactly why Lesson 12's held-out split had to exist.
- As complexity rises, **training error always falls but validation error traces a U**, and the best model sits at the bottom of that U, chosen by validation error and never by training error.
- The **bias-variance tradeoff** is the balancing act at the heart of machine learning: too simple and too complex both fail on new data, and every real model is an attempt to find the middle.

---

## SECTION 10 — CHALLENGE WITH HIDDEN ANSWER

**Challenge:** You train two versions of a ticket classifier and measure each on the same train/validation split.

- **Model A:** training accuracy 0.99, validation accuracy 0.71.
- **Model B:** training accuracy 0.74, validation accuracy 0.72.

Three questions: (1) Which model is overfitting, and how can you tell? (2) Is Model B underfitting, and what does its pair of numbers suggest? (3) Which model would you ship, and what would you try next to do better than either?

<details>
<summary>Click to reveal the answer</summary>

**(1) Model A is overfitting.** The giveaway is the huge gap between a near-perfect training score (0.99) and a much lower validation score (0.71). It has largely *memorised* the training tickets — including their noise — so it looks brilliant on data it learned from and mediocre on data it didn't. Big train-vs-validation gap = high variance = overfitting.

**(2) Model B is probably underfitting (mild high bias).** Its two numbers are close (0.74 vs 0.72), so it's *not* overfitting — but both are fairly low. When training and validation scores are both low and close together, the model isn't even fitting the training data well, which points to a model too simple (or features too weak) to capture the real pattern. Note the honesty, though: Model B's validation score (0.72) is actually *slightly better* than Model A's (0.71), because Model B isn't fooling itself.

**(3) Ship Model B, then try to lift both models toward the middle.** Between these two, Model B is the safer bet: what you see is what you get, and its real-world (validation) accuracy is marginally higher, whereas Model A's true performance is the 0.71, not the flattering 0.99. But 0.72 isn't great, so the goal is to escape *both* failure modes:
- To cure Model A's overfitting: give it **more training data**, or make it **simpler / add a complexity penalty** so the gap closes.
- To cure Model B's underfitting: give it **more capacity or better features** so its training score can rise.
The ideal next model has both scores high *and* close together — the bottom of the bias-variance U. Whichever direction you push, you keep watching the same two numbers and the gap between them.

</details>

---

## SECTION 11 — WHAT IS NEXT

That closes Chapter 4 — the machine-learning *workflow* (pipeline → honest splits → diagnosing fit). Next comes the **Chapter 4 summary**, then Chapter 5 opens with **Lesson 14 — Linear regression**, our first real algorithm: a line-of-best-fit that learns its parameters by gradient descent (Lesson 09) and that we'll deliberately steer along the bias-variance tradeoff you just learned to read.

---

*Lesson 13 of the curriculum · Phase 2 (Classical Machine Learning) · Chapter 4 (The Machine Learning Workflow) · Lesson 3 of 3 in this chapter.*
