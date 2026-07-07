# Lesson 11 — The End-to-End Machine Learning Pipeline

- **Phase / Chapter / Lesson:** Phase 2 — Classical Machine Learning / Chapter 4 — The Machine Learning Workflow / Lesson 11
- **Files created:** `lesson-notebooks/phase-02-classical-machine-learning/chapter-04-the-machine-learning-workflow/lesson-11-end-to-end-ml-pipeline.ipynb`
- **Files updated:** `progress-tracker.md`
- **Project change:** none — this lesson frames the workflow; the real learned classifier gets built once we have proper algorithms (Chapter 5).
- **Prerequisites:** Lesson 01 (rules vs. learning from data), Lesson 02 (supervised learning), Lessons 05–07 (NumPy, pandas, plots), Lessons 08–10 (the math intuition).

---

## Section 1 — Why this matters

Every book, course, and job in machine learning eventually reveals the same secret: there is one workflow, and you run it over and over. **Get data, turn it into features, train a model, check how good it is, then use it.** Recommendation engines, fraud detectors, medical models, and the ticket classifier we're about to build all ride this same five-step track. Learn the track once and every future project stops feeling like a new mystery — it becomes "which algorithm goes in the middle this time."

This lesson is the map for all of Phase 2. Before we spend chapters on individual algorithms (linear regression, decision trees, boosting), you need the container they all drop into. An algorithm is just the "train" step; it means nothing without the data before it and the honest evaluation after it.

For our assistant, this is the moment the toy retires. Until now, ticket urgency was decided by weights *we typed in by hand*. Starting now, we frame the first task where the computer sets its own settings by studying labelled examples: **predicting which category a support ticket belongs to** — shipping, billing, or account.

## Section 2 — Real-world analogy

Think about learning to cook a dish from a recipe box, then cooking it for guests.

- **Data** = the recipe cards you collect: each card is a finished dish paired with how it was made. Lots of examples of "this input produced this result."
- **Features** = the measurable things you pay attention to: grams of flour, minutes in the oven, oven temperature. You can't learn from "it felt right"; you need numbers.
- **Train** = practising in your own kitchen until you get consistent. You adjust based on the recipe cards you have.
- **Evaluate** = cooking the dish for a friend who wasn't there while you practised, and asking honestly how it turned out. If you only ever judge your cooking by the same three practice runs, you learn nothing about new guests.
- **Use** = the night the actual guests arrive and eat what you make.

The one rule people break: they taste-test using the exact same practice runs they trained on, declare themselves a great cook, and then flop in front of real guests. Holding out a fresh dish for an honest judge is the whole game — and it maps exactly onto holding out a **test set** in machine learning.

## Section 3 — The concept explained

Let's define each stage fully, in order.

### Stage 1 — Data

**Data** here means a pile of **labelled examples**. A single example is an input paired with its correct answer. For us: a ticket's text (input) plus the category a human assigned it (answer). The answer is called the **label**. Because every training example carries its answer, this is **supervised learning** (Lesson 02) — the model learns by comparing its guesses to known answers.

Tiny picture:

```
"Where is my package"        -> shipping
"I was charged twice"        -> billing
"I cannot log in"            -> account
```

More good, correctly-labelled data is the single biggest lever on how well a model works — often more than a fancier algorithm (Lesson 03).

### Stage 2 — Features

A model does arithmetic, not reading. It cannot work with the word "late," but it can work with the number 2. So step two is to rewrite each ticket as numbers. A **feature** is just *one thing the model pays attention to* — one measurable property of the example.

The simplest way to turn text into numbers is a **bag of words**. It sounds fancy but it is two small steps: (1) list the words in the ticket, and (2) count how many times each appears. Take the sentence:

> "My order is late, my order"

Counting each distinct word gives:

| word (the feature) | count (the value) |
| --- | --- |
| my | 2 |
| order | 2 |
| is | 1 |
| late | 1 |

That little table **is** the sentence, rewritten as numbers — "my → 2, order → 2, is → 1, late → 1." Let's unpack the three ideas hiding in that:

**"Each distinct word is a feature; its count is the value."** Each different word becomes one feature (one thing to look at), and the value of that feature is simply how many times the word showed up. Words in, counts out.

**"We throw away word order."** Notice the table above forgets the *order* the words came in. "refund my order" and "my order refund" produce the exact same counts, so afterwards we cannot tell them apart. That is where the name comes from: imagine writing each word on a scrap of paper, dropping them all into a **bag**, and shaking it — you still have every word, but the order is gone.

**"Crude, but it works well enough."** *Crude* means rough and simple. Throwing away order really does lose information (e.g. "did **not** get a refund" vs "**did** get a refund" look almost the same). But here is why it still works: if a ticket contains the word "refund" *at all*, it is almost certainly a **billing** ticket; if it contains "password," almost certainly **account**. The mere presence of that one word is a loud clue, and shaking the bag does not weaken that clue one bit. We threw away information that mostly did not matter (order) and kept the information that does (which strong words are present).

One-sentence version: *count how many times each word appears, ignore the order, and even though that is rough, key words like "refund" and "password" are such strong giveaways that the model can still learn to sort tickets from them.*

### Stage 3 — Train (after holding out a test set)

Two moves, in this exact order.

**First, split.** Set aside some examples the model will never study — the **test set**. The rest is the **training set**. Why first? Because the moment the model sees an example, evaluating on it later is cheating: the model could just memorise answers and look perfect while having learned nothing general.

```
12 tickets
   |
   +-- 9 training tickets  -> model learns from these
   +-- 3 test tickets      -> locked away, used only to judge
```

**Then, train.** **Training** means letting the model set its own internal numbers — its **parameters** — from the training data. In our word-count model, training is literally: for each category, tally how often each word shows up across that category's training tickets. Those three count tables *are* the parameters. We didn't type them; the data did. (In later lessons the parameters are weights adjusted by gradient descent — Lesson 09 — but the idea is identical: settings learned from data, not hand-written.)

To **predict**, score each category by summing the training counts of the ticket's words, then pick the highest. Common words ("my", "the") appear under every category and roughly cancel; the distinctive keywords tip the decision.

### Stage 4 — Evaluate

**Evaluation** is measuring quality on the held-out test set — data the model has never seen — so the score reflects how it will behave on *new* tickets, not how well it memorised old ones.

The simplest score for a classifier is **accuracy**: the fraction of test examples it labels correctly.

```
accuracy = (number predicted correctly) / (total number of test examples)
```

Plain meaning first: "out of the tickets I quizzed it on, what share did it get right?" If it gets 3 of 3, accuracy is `3 / 3 = 1.0`, i.e. 100%.

But a bare number can fool you, so always compare against a **baseline** — the score of a trivial strategy. The classic baseline is "always guess the most common category." If your clever model can't beat *always guessing the same thing*, it isn't clever. (Accuracy also hides important failures when classes are imbalanced — the deeper metrics come in Lesson 22. For now, accuracy-versus-baseline is enough.)

### Stage 5 — Use

Finally, point the finished, evaluated model at brand-new, unlabelled input and take its answer as the prediction. Inside the assistant this is a live ticket arriving and the model tagging it `shipping` so it can be routed. No label, no peeking — just input in, category out.

### The loop

Drawn out, the pipeline is a **loop**, not a straight line:

```
   +-----------------------------------------------+
   v                                               |
 DATA --> FEATURES --> TRAIN --> EVALUATE --> (good enough?) --> USE
                                     |  no
                                     +--> back to data / features / model
```

If evaluation disappoints, you go back — collect more data, build better features, or try a stronger algorithm — and re-evaluate. Almost all real ML work is spinning this loop, not the single triumphant training run people imagine.

## Section 4 — The code

The notebook `lesson-11-end-to-end-ml-pipeline.ipynb` runs the ticket-category task through all five stages: it builds a 12-ticket labelled DataFrame, tokenises each ticket into a bag of words, holds out the last 3 tickets as a test set, trains per-category word-count tables from the 9 training tickets, evaluates accuracy against a majority-class baseline, and finally classifies a brand-new ticket.

The model is intentionally the simple word counter you already understand — the point is to see the *pipeline*, not a new algorithm. Everything uses tools from earlier lessons: pandas for the data table (Lesson 06), Python's `collections.Counter` for the tallies, and a plain `max` to pick the winning category. The notebook explains every code cell line by line in the markdown just above it; below are the two functions worth reading slowly.

**The `train` function, line by line** — this is where "learning" actually happens:

```python
def train_word_counts(train_df):     # take the training table
    counts = {}                        # empty box: category name -> its word tally
    for _, row in train_df.iterrows(): # go through the training tickets one by one
        counts.setdefault(row["category"], Counter())   # first time we see a category, start it a fresh empty tally
        counts[row["category"]].update(row["tokens"])   # add this ticket's words into that category's tally
    return counts                      # hand back the three finished tallies
```

Plain version: for each ticket, find its category's tally sheet (make a new blank one if it's a category we haven't seen yet) and drop this ticket's words onto it. After all 9 training tickets, `counts` holds three word-count tables — and those tables *are* the learned model.

**The `predict` function, line by line** — this is how it guesses:

```python
def predict(tokens, model):           # take a ticket's word-list and the learned tallies
    scores = {}                        # one total score per category
    for category, counter in model.items():                 # look at each category's tally
        scores[category] = sum(counter.get(word, 0) for word in tokens)  # add up how often the ticket's words appeared in that category (0 if never seen)
    return max(scores, key=scores.get), scores              # return the highest-scoring category, plus all the scores
```

Plain version: give each category a score by adding up how familiar this ticket's words are to it, then pick the category with the biggest score. `counter.get(word, 0)` is "look up this word's count, and if the word isn't there, use 0 instead of crashing." `max(scores, key=scores.get)` is "of all the categories, give me the one with the largest score."

A library detail to verify against current docs: pandas `Series.value_counts()` sorts by count and, on a **tie** (our three categories each appear 3 times in training), the order among equal counts is not guaranteed, so which category `.idxmax()` returns as the "majority" can vary by pandas version. It doesn't change the lesson — any single category baseline scores 1 of 3 here — but it's why the predicted baseline label below could differ on your machine.

## Section 5 — If you ran this

Here is what each cell would do, with output labelled as a **prediction** (your real output may differ slightly, especially the tie-break noted above):

1. **Data cell** — builds and displays the 12-row DataFrame with `text` and `category` columns (three shipping, three billing, three account, then one of each held for testing).

2. **Features cell** — adds a `tokens` column and shows it. Row 0's tokens would read:

   ```
   ['where', 'is', 'my', 'package', 'it', 'has', 'not', 'arrived', 'yet']
   ```

3. **Split cell** — prints:

   ```
   Training tickets: 9
   Test tickets: 3
   ```

4. **Train cell** — prints the top-4 words each category learned. **Prediction:**

   ```
   shipping -> [('my', 3), ('is', 2), ('has', 2), ('not', 2)]
   billing -> [('i', 3), ('a', 3), ('for', 2), ('my', 2)]
   account -> [('i', 3), ('my', 3), ('cannot', 1), ('log', 1)]
   ```

5. **Predict cell** — classifies the first test ticket, "my package still has not arrived." Shipping scores 9 (my 3 + package 1 + has 2 + not 2 + arrived 1), billing 3, account 4. **Prediction:**

   ```
   ('shipping', {'shipping': 9, 'billing': 3, 'account': 4})
   ```

6. **Evaluate cell** — all three test tickets are classified correctly (shipping, billing, account), so accuracy is 1.0, while always guessing one category scores 1 of 3. **Prediction:**

   ```
   Model accuracy on held-out test:      1.00
   Baseline (always guess 'shipping'): 0.33
   ```

7. **Use cell** — classifies the new ticket "My package has not shipped and tracking shows it is late." Shipping wins with 14 (billing 4, account 5). **Prediction:**

   ```
   Ticket: My package has not shipped and tracking shows it is late
   Predicted category: shipping
   Scores: {'shipping': 14, 'billing': 4, 'account': 5}
   ```

Remember these are traced-by-hand predictions. Real output will match closely, with the one caveat that the baseline's category label depends on the tie-break.

**How to run it:** open the notebook in VS Code and run the cells top to bottom.

## Section 6 — Applied to our assistant

Nothing is added to `customer-support-assistant/` this lesson, on purpose. This lesson *frames* the assistant's next capability — a learned ticket-**category** classifier — but the model at the centre of a real pipeline should be a real learning algorithm, and we don't meet those until Chapter 5 (logistic regression is the natural fit). Bolting the toy word counter into the project now would just duplicate the existing urgency toy under a new name and teach nothing new.

So the project stays exactly as it is: `python main.py` still runs the Lesson 01 urgency classifier and its self-check, unchanged. What changed is the *plan* — you now know the five slots the category classifier will fill, and the notebook has walked the whole loop once with a stand-in model so the real thing has an obvious home.

## Section 7 — Common mistakes and gotchas

- **Testing on training data.** The cardinal sin. If you evaluate on examples the model trained on, a pure memoriser scores 100% and you ship something that fails on real input. Split first, always. (Lesson 12 goes deep on this.)
- **No baseline.** "95% accuracy!" means nothing if 95% of tickets are one category and the model just guesses that every time. Always compare to a trivial baseline before celebrating.
- **Skipping the data and feature work.** Beginners rush to the algorithm. In practice, most of the effort and most of the payoff live in getting clean data and good features — the "train" step is often the quickest part.
- **Treating it as a straight line.** One training run is rarely the end. Expect to loop back to data or features when evaluation underwhelms; that's normal, not failure.
- **Exact-match brittleness.** Bag-of-words only credits words it saw in training, so "parcel" gets no shipping signal if training only had "package." Better features and real models (later in Phase 2) soften this.

## Section 8 — When to use this, and tradeoffs

The five-stage pipeline is the right frame for essentially **every supervised learning task** — any time you have labelled examples and want to predict labels for new ones. It's the default mental model to reach for first.

Tradeoffs and limits: the pipeline itself is universal, but its *cost* isn't. Gathering and labelling data can dominate the whole project. The pipeline also assumes new input looks like your training data; when the world shifts (new product lines, new slang), a once-good model quietly rots and you must loop back — that's monitoring and drift, much later in Lesson 105. And this workflow is for **learning from labelled examples**; problems with no labels (grouping similar customers) use a different shape (unsupervised learning, Chapter 8), and problems solved by clear fixed rules may not need ML at all.

## Section 9 — Key takeaways

- Machine learning is one repeatable workflow — **data → features → train → evaluate → use** — and every algorithm you'll learn is just the interchangeable "train" step in the middle.
- **Features** turn raw input into numbers; a bag of words counts each word and, despite ignoring order, captures enough keyword signal to classify text.
- **Training** means the model sets its own parameters from the training data — here, per-category word-count tables — instead of a human typing them in.
- The non-negotiable rule is to **hold out a test set before training** and evaluate only on it, comparing accuracy against a trivial baseline so a good-looking number can't fool you.
- The pipeline is a **loop**: weak evaluation sends you back to gather more data, engineer better features, or try a stronger model — which is what most real ML work actually is.

## Section 10 — Challenge with hidden answer

**Challenge:** You build a spam-versus-not-spam email classifier. You have 1,000 labelled emails, of which 950 are "not spam" and 50 are "spam." You train on all 1,000, then evaluate on those same 1,000 and report **95% accuracy**. Your teammate is thrilled. Name the *two* separate things wrong with this evaluation, and for each, say what you'd do instead. Then explain what accuracy figure a lazy "predict not spam every time" baseline would score, and what that tells you.

<details>
<summary>Click to reveal the answer</summary>

**Problem 1 — You evaluated on the training data.** The model was allowed to study all 1,000 emails and was then quizzed on those exact 1,000. That measures memory, not the ability to handle *new* email, which is the only thing that matters in production. **Fix:** split first — hold out, say, 200 emails the model never trains on, train on the other 800, and report accuracy only on the held-out 200. (Lesson 12 refines this with a validation set too.)

**Problem 2 — No baseline, and accuracy hides the failure that matters.** Because 950 of 1,000 emails are "not spam," a do-nothing model that labels *everything* "not spam" already scores `950 / 1000 = 95%` accuracy — the same number your teammate is celebrating. So 95% might mean the model learned nothing and simply never flags spam, catching **zero** of the 50 spam emails. **Fix:** always compare against that majority-class baseline, and for imbalanced problems stop trusting accuracy alone — measure how many actual spam emails you catch (that's recall, coming in Lesson 22).

**What the baseline tells you:** since the trivial "always not spam" strategy also hits 95%, your model has demonstrated *no* value over doing nothing. A useful spam filter has to beat 95% *and* actually catch spam. The baseline turned an exciting-looking number into an obvious red flag — which is exactly why you compute it.

</details>

## Section 11 — What is next

Lesson 12 zooms in on the one rule this lesson leaned on hardest — splitting your data — and adds a third piece, the **validation set**, explaining what each split is for and why testing on data you trained on is the cardinal sin of machine learning.

---

*Phase 2 — Classical Machine Learning · Chapter 4 — The Machine Learning Workflow · Lesson 11 of 113. Next: Lesson 12 — Training, validation, and test sets.*
