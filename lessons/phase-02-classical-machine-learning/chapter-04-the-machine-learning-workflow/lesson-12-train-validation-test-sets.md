# Lesson 12 — Training, Validation, and Test Sets

**Phase 2 · Chapter 4 · Lesson 12**
**Position:** Phase 2 (Classical Machine Learning) → Chapter 4 (The Machine Learning Workflow) → Lesson 12 of the curriculum.
**Files created:** `lesson-notebooks/phase-02-classical-machine-learning/chapter-04-the-machine-learning-workflow/lesson-12-train-validation-test-sets.ipynb`
**Files updated:** `progress-tracker.md`
**Prerequisites:** Lesson 11 (the end-to-end pipeline — where we first hid a test set), Lesson 02 (supervised learning), Lesson 06 (pandas).
**Project change:** none — this is a theory-and-demo lesson. It sharpens the discipline behind the learned ticket classifier we'll build for real later in Phase 2.

---

## SECTION 1 — WHY THIS MATTERS

In Lesson 11 we quietly did something crucial: before the model learned anything, we hid 3 tickets and only graded the model on those. This lesson explains *why* that hiding is the difference between a model you can trust and a model that's fooling you.

The whole idea fits in one sentence: **a model is only impressive if it does well on data it has never seen.** A model that scores 100% on the exact examples it studied has told you nothing — it may have simply memorised the answers. The only score that predicts how your customer-support assistant will behave on *tomorrow's* tickets is a score measured on tickets it has never touched.

Get this wrong and every number you report is a lie you're telling yourself. Get it right and you can actually tell whether a change helped. That's why splitting data correctly is the first real skill of machine learning — before any fancy algorithm.

---

## SECTION 2 — REAL WORLD ANALOGY

Think of a student preparing for an exam.

- The **textbook and homework** are the **training set** — the material they study and learn from.
- The **practice exams** are the **validation set** — they take these many times while preparing, to see what's working and decide what to change (study more chapters? switch technique?). Practice exams guide their choices, but the practice score is not their real grade.
- The **final exam** is the **test set** — taken once, at the end. Its whole point is to measure real understanding on questions that never helped them prepare.

Now picture a student who somehow got the final exam's exact questions and answer key a week early, memorised them, and scored 100%. Did they learn anything? You have no idea — the exam was ruined the moment they saw it in advance. That is exactly what "testing on your training data" does to a model.

---

## SECTION 3 — THE CONCEPT EXPLAINED

### The cardinal sin: testing on data you trained on

**Plain definition:** grading a model using the same examples it was allowed to learn from.

**Why it's a trap:** a model can score perfectly by *memorising* — storing "this exact ticket → billing" — without understanding anything general about billing. It aces the memorised examples and then fails on the first genuinely new ticket. Because the flattering score comes for free, beginners constantly mistake it for success.

**Tiny concrete example:** a "model" that is just a lookup table `{"i was charged twice": "billing", ...}` will get **100%** on the tickets in its table and roughly **random** accuracy on anything else. The notebook builds exactly this and shows the two numbers side by side: `1.00` on memorised tickets, `0.33` on new ones.

**Technical name:** this mistake is called **testing on your training data**, and the more general version — any information about the test set sneaking into training or tuning — is called **data leakage**.

### The fix: three piles, each with one job

We split our labelled examples into three separate sets:

```
   ALL LABELLED TICKETS
   ┌───────────────────────────────────────────────┐
   │  TRAIN (biggest)   │  VALIDATION  │   TEST     │
   └───────────────────────────────────────────────┘
        learn from          tune &         final judge,
        this only           choose         look ONCE
```

**1. Training set** — the biggest slice. The model sets its numbers (its **parameters** — the values it learns from data, defined in Lesson 11) using *only* these examples. This is the textbook.

**2. Validation set** — a smaller slice the model never learns from. *You* use it during development to **make choices**: which of two models is better, whether to remove common words, what settings to use. You try an option, check it on the validation set, and keep the winner. This is the practice exam — take it as often as you like.

Why keep it separate from training? Because the instant you start *choosing* things based on a set of data, you're indirectly bending your model to fit that data — so it can no longer give an honest final score. Your tuning has "used it up."

**3. Test set** — locked in a drawer until the very end, and looked at **exactly once**. Its only job is to answer: "now that I'm completely done, how well does this work on data that never influenced it in any way?" If you peek early and start tweaking to raise the test score, you've silently demoted it to a validation set and lost your honest final number.

**Rough ratios people use:** around 60/20/20 or 70/15/15 (train/validation/test). There's no magic number — the more total data you have, the smaller the validation and test *percentages* can be while still holding enough examples to be meaningful.

### Why three piles and not two?

If you make **no choices** during development, train + test is enough (that's all Lesson 11 needed). But the moment you compare "model A vs model B" and pick the winner *on the test set*, that test set has quietly helped you build the model — so it can't be your honest judge anymore. The **validation set exists to absorb all your choosing**, keeping the test set pure. Our notebook makes exactly one choice (keep filler words vs. remove them), so it genuinely needs all three.

### The subtle trap: overusing the validation set

If you try five hundred options and keep whichever squeaks out the best validation score, you slowly start fitting to the *quirks* of the validation set too. That's why the untouched test set is the real safeguard, and why big projects sometimes refresh their validation data. We'll come back to this in Lesson 13.

---

## SECTION 4 — THE CODE

The notebook (saved at the path in the header) runs the whole story:

1. **Data** — 18 labelled tickets, arranged so rows 0–11 are training, 12–14 validation, 15–17 test.
2. **The cardinal sin** — a memoriser that stores exact-text answers, graded on its own training tickets (perfect) and then on new test tickets (collapses).
3. **The three-way split** — confirming the 12 / 3 / 3 sizes.
4. **Two candidate models** — Candidate A keeps every word; Candidate B removes common filler words first. Both trained on the training set only.
5. **Choose on validation** — score both candidates on the validation set, keep the winner.
6. **Final judge on test** — score the chosen model on the test set, one time.

Concept and *why* live in the markdown cells; each code cell has short per-line comments so it reads top to bottom on its own.

> **Verify-as-you-go note:** `value_counts().idxmax()` is used to pick the memoriser's fallback category. With our perfectly balanced 4/4/4 training set the three categories tie, and pandas returns the first one it encountered (`shipping` here). This is a standard-library/pandas detail worth confirming in the current pandas docs if you rely on the exact tie-break.

---

## SECTION 5 — IF YOU RAN THIS

Open the notebook in VS Code and run the cells top to bottom. Expected output, labelled as a **prediction** (real output may differ slightly in formatting):

1. **Data cell** → `Total tickets: 18`, then an 18-row table.
2. **Memoriser cell**:
   ```
   Memoriser on its OWN training tickets: 1.00
   Memoriser on NEW held-out test tickets: 0.33
   ```
3. **Sizes cell** → `12`, `3`, `3`.
4. **Train-both cell** → `Both candidates trained on 12 tickets.`
5. **Choose-on-validation cell**:
   ```
   Candidate A (keep all words)   validation accuracy: 0.67
   Candidate B (remove filler)    validation accuracy: 1.00
   Chosen model: Candidate B (remove filler)
   ```
6. **Final test cell**:
   ```
   Final test accuracy of Candidate B (remove filler): 1.00
   OK  true=shipping guess=shipping | my package has not arrived and delivery is late
   OK  true=billing  guess=billing  | i was charged twice and need a refund
   OK  true=account  guess=account  | i forgot my password and cannot log in
   ```

The takeaway lives in the contrast: the memoriser looked flawless (`1.00`) yet was worthless (`0.33`) on new data, while the properly built and properly chosen model earns its `1.00` honestly on a test set it never influenced.

Remember these numbers are a prediction; your real run will confirm the *pattern* — memorised score high and meaningless, held-out score is the one that counts — even if a value shifts.

---

## SECTION 6 — APPLIED TO OUR ASSISTANT

Nothing to implement in the project today — this is a discipline lesson, not a feature. But it directly shapes what's coming: when we replace the toy word-counter in `ticket_urgency.py` (and add a real learned category classifier) later in Phase 2, we will split the assistant's data into train / validation / test and report only the honest test score. Every "the assistant is 92% accurate" claim we ever make will rest on the rule established here. The project stays exactly as it was and still runs end to end with `python main.py`.

---

## SECTION 7 — COMMON MISTAKES AND GOTCHAS

- **Reporting the training score as if it meant something.** It almost always looks great and almost always overstates reality. Only quote held-out scores.
- **Tuning on the test set.** Every time you change something to nudge the test number up, you've turned your test set into a validation set and lost your honest judge.
- **Splitting *after* you clean or transform data using the whole dataset.** If information from the test rows shaped your preprocessing, that's leakage — split first, then process each part.
- **Letting the same real example land in two splits** (e.g. a duplicated ticket in both train and test). It secretly reintroduces the cardinal sin.
- **Tiny validation/test sets giving noisy scores.** With only 3 test tickets, one wrong answer swings accuracy by 33 points. Real projects use enough held-out examples for the number to be stable.

---

## SECTION 8 — WHEN TO USE THIS, AND TRADEOFFS

Use a train/validation/test split **whenever you build a model whose quality you intend to trust or compare** — which is essentially always. It costs you nothing but a bit of data set aside.

The tradeoffs are about *how* you split, not *whether*:
- **Two-way (train/test)** is fine when you make no tuning choices. Simpler, but doesn't cover model selection honestly.
- **Three-way (train/validation/test)** is the default once you start choosing between options.
- **When data is scarce**, holding out two separate chunks hurts, so people use **cross-validation** (rotating which slice is held out) — that's Lesson 23. It buys honest estimates from limited data at the cost of more compute.

The one non-negotiable across all of these: the final judge is data the model never learned from or was tuned against.

---

## SECTION 9 — KEY TAKEAWAYS

- A model is only impressive if it scores well on data it has **never seen**; a high score on its own training data usually means nothing.
- The **cardinal sin** is testing on your training data — a memoriser proves it by scoring a perfect but worthless 1.00 on memorised examples and near-random on new ones.
- We split labelled data into **train** (learn parameters), **validation** (make development choices), and **test** (the final honest grade, looked at exactly once).
- The **validation set exists so the test set stays pure** — it absorbs all the choosing and tuning you do while building the model.
- Any leak of test-set information into training or tuning is **data leakage**, and it silently fakes great results.

---

## SECTION 10 — CHALLENGE WITH HIDDEN ANSWER

**Challenge:** A teammate proudly reports: "Our new ticket classifier is 99% accurate!" You ask how they measured it, and they say: "I trained it on all 5,000 of our labelled tickets, then checked how many of those same 5,000 it got right." Two questions: (1) What's wrong with this number? (2) They also mention they tried 30 different model settings and kept the one with the best score on that same 5,000. Does that change anything, and what should they have done instead?

<details>
<summary>Click to reveal the answer</summary>

**(1) The 99% is measured on the training data, so it's the cardinal sin.** The model was allowed to learn from all 5,000 tickets and was then graded on those exact same tickets. A model that partly *memorised* them would score high here while being much weaker on new tickets. The number tells us almost nothing about real-world performance — it's the "student who saw the answer key" score. The honest question is: how does it do on tickets it never studied?

**(2) Yes, it makes it worse — now there's tuning leakage on top.** By trying 30 settings and keeping whichever scored best on that same 5,000, they've also *chosen* their model using that data. Even if they had held some of it out, picking the best-of-30 on one held-out set starts fitting to that set's quirks (the "overusing validation" trap). So there's no clean, honest estimate anywhere.

**What they should have done:** split the 5,000 into three piles first — say **3,000 train / 1,000 validation / 1,000 test**. Train each of the 30 settings on the 3,000; compare them on the 1,000 validation tickets and keep the winner; then, exactly once, measure that single chosen model on the 1,000 test tickets and report *that* number. If they want to squeeze more out of limited data, cross-validation (Lesson 23) is the more advanced answer — but the principle is unchanged: the final number must come from data the model neither learned from nor was tuned against.

</details>

---

## SECTION 11 — WHAT IS NEXT

Lesson 13 — **Overfitting, underfitting, and the bias-variance idea**: what it actually looks like when a model memorises instead of learns, and how the *gap* between the training score and the validation/test score (exactly the splits we built here) is the tool we use to catch it.

---

*Lesson 12 of the curriculum · Phase 2 (Classical Machine Learning) · Chapter 4 (The Machine Learning Workflow) · Lesson 2 of 3 in this chapter.*
