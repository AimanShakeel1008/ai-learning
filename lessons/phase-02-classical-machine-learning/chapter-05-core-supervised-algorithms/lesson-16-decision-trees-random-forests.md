# Lesson 16 — Decision Trees and Random Forests

- **Position:** Phase 2 (Classical Machine Learning) → Chapter 5 (Core Supervised Algorithms) → Lesson 16
- **Files created:**
  - `lesson-notebooks/phase-02-classical-machine-learning/chapter-05-core-supervised-algorithms/lesson-16-decision-trees-random-forests.ipynb`
  - `customer-support-assistant/ticket_category.py`
  - `customer-support-assistant/notebooks/ticket_category.ipynb`
- **Files updated:** `customer-support-assistant/main.py`
- **Prerequisites:** Lesson 11 (the ML pipeline and the ticket-category task), Lesson 13 (overfitting), Lessons 14–15 (what "training" and "loss" mean)

---

## SECTION 1 — WHY THIS MATTERS

The last two lessons taught models that learn **numbers**: linear regression learned a slope and an intercept, logistic regression learned a weight and a bias. Both draw *straight lines* through the data. But lots of real-world knowledge is not line-shaped — it is **rule-shaped**. "If the ticket mentions a refund, it is a billing issue. Otherwise, if it mentions a package, it is a shipping issue." Humans reason like this constantly.

A **decision tree** is a model that learns rules like that on its own, just by looking at examples. No human writes the rules. The model figures out two things by itself: *which* yes/no questions are worth asking, and *in what order* to ask them.

There is also a stronger version called a **random forest**: instead of one tree, you train many trees and let them vote on the answer. This simple trick works remarkably well. In fact, when your data looks like a spreadsheet (rows of examples, columns of facts about each one), tree-based models are usually the best tool for the job — even today, and often better than neural networks. That is why real ML engineers still use them all the time. Next lesson covers gradient boosting, another tree-based method in the same family.

For our assistant this lesson is a milestone: back in Lesson 11 we promised a model that predicts a ticket's **category** — which team should handle it. Today we build it for real. The assistant now routes tickets to shipping, billing, or account support with a genuinely learned decision tree.

## SECTION 2 — REAL-WORLD ANALOGY

Think of the game **20 Questions**. You are guessing what object your friend is thinking of, and you may only ask yes/no questions.

A bad player asks random questions: "Is it a spoon? Is it a giraffe?" — each answer eliminates almost nothing. A good player asks the question that **splits the remaining possibilities most evenly and cleanly**: "Is it alive?" — whatever the answer, half the world disappears. Then, *given* that answer, they pick the next most informative question, and so on, until only one answer is left.

Map the pieces:

- The object your friend is thinking of → the ticket's true category
- Each yes/no question → one test on the ticket ("does it contain the word *refund*?")
- Picking the most informative question first → choosing the split with the lowest impurity
- Narrowing down step by step → walking down the tree's branches
- The final guess → the leaf's prediction

A random forest is the same game played by a **panel of players**, each of whom learned from a slightly different set of past games and is allowed a slightly different set of questions — then the panel votes on the final answer.

Why is a panel safer than one player? Because of what we can call a *blind spot*: **a wrong rule someone learned without knowing it is wrong, because of the particular examples they happened to learn from**. Say one player only ever saw games where the answer to "Is it alive?" being yes meant the object was an animal — so they learned the rule "alive means animal". That rule is wrong (a tree is alive too), but the player cannot know that, because nothing they saw ever contradicted it. Our ticket tree will do exactly this later in the lesson: it learns "contains *delivery* means shipping", because in its training tickets the word "delivery" only ever appeared in a shipping ticket — and it then misroutes a billing ticket about a wrongly charged delivery fee.

Here is the key point: every player has *some* wrong rule like this, but because each player learned from **different** past games, they have **different** wrong rules. On any given question, maybe one player is fooled — but the others, who never learned that particular wrong rule, are not, and they outvote the fooled one. For the *panel* to get it wrong, most players would have to have learned the *same* wrong rule from *different* experiences. That is far less likely than one player learning it. This is exactly why the forest votes.

## SECTION 3 — THE CONCEPT EXPLAINED

### 3.1 A tree is a flowchart the data writes

A **decision tree** is a flowchart of yes/no questions. To classify a ticket, you start at the top question, answer it, follow the yes-branch or the no-branch, answer the next question, and keep going until you reach an end point that states a category. The standard vocabulary (the tree metaphor is used everywhere):

- **node** — one question box
- **root** — the first question, at the top
- **branch** — the yes-path or no-path leaving a question
- **leaf** — an end point that asks nothing and just gives a prediction
- **depth** — how many questions deep a path goes

Here is the actual tree our assistant learns in this lesson (nobody wrote it — the data chose every question):

```text
contains "refund"?
  yes: predict billing
  no:  contains "package"?
    yes: predict shipping
    no:  contains "delivery"?
      yes: predict shipping
      no:  predict account
```

The only thing a human supplied was the **menu** of questions the tree was *allowed* to ask — a list of candidate words ("refund", "charged", "package", "delivery", "order", "password", "account", "login", "help"). Which of them to use, and in what order, the tree worked out from 12 labelled tickets.

### 3.2 Gini impurity: measuring how mixed a pile is

To pick good questions, the tree needs to *measure messiness*.

**Plain definition:** take a pile of labelled tickets and draw two at random. The chance that their labels **disagree** is the pile's impurity.

**Why it exists:** remember what a question does — it takes one mixed pile of tickets and splits it into two smaller piles. A *good* question leaves those two piles better sorted (more "same label together") than the pile we started with. But "better sorted" is a feeling, and a program cannot compare feelings. So we turn messiness into a **number**. Once messiness is a number, picking the best question becomes a simple job: try every question, measure how much messiness is left after each one, and keep the question that leaves the smallest number.

**Tiny worked example, one small step at a time.** Take a pile of 4 tickets: 2 shipping and 2 billing.

- **Step 1 — find each label's share of the pile.** A *share* just means: the number of tickets with that label, divided by the total number of tickets in the pile.
  - shipping: 2 out of 4 → share = 2/4 = 0.5
  - billing: 2 out of 4 → share = 2/4 = 0.5
- **Step 2 — find the chance that two random picks land on the SAME label.** Imagine you pick one ticket at random, note its label, put it back, and pick again. What is the chance both picks are shipping? The first pick is shipping with chance 0.5, the second pick is shipping with chance 0.5, so both are shipping with chance 0.5 × 0.5 = 0.25. **This is why the share gets squared** — "squared" is just "the chance of getting that label twice in a row."
  - both picks shipping: 0.5 × 0.5 = 0.25
  - both picks billing: 0.5 × 0.5 = 0.25
- **Step 3 — add those up to get the chance the two picks AGREE.** 0.25 + 0.25 = 0.5. So half the time, two random picks from this pile have the same label.
- **Step 4 — flip it to get the chance they DISAGREE.** Chances of "agree" and "disagree" must add up to 1, so: 1 − 0.5 = **0.5**. That is the pile's impurity.

Written as one formula, those four steps are:

> 1 − (1/2)² − (1/2)² = 1 − 0.25 − 0.25 = **0.5**

Now check that the number behaves the way "messiness" should:

- **A perfectly sorted pile** (all 4 tickets shipping): shipping's share is 4/4 = 1. Two random picks are *always* both shipping, so they can never disagree. The formula agrees: 1 − 1² = **0**. Impurity 0 means "nothing left to sort."
- **Our full training pile** (4 shipping, 4 billing, 4 account — each share 1/3): the chance of agreeing is (1/3)² + (1/3)² + (1/3)² = 3/9 = 1/3, so the chance of disagreeing is 1 − 1/3 = **0.667**. High number, very mixed pile — exactly what we would say by eye.

One-sentence summary of all of the above: **Gini impurity answers "if I grab two tickets at random from this pile, how likely are they to have different labels?" — 0 when the pile is perfectly sorted, and bigger the more mixed it is.**

**Technical name:** this is **Gini impurity**. In symbols: impurity = 1 − Σ pᵢ², where pᵢ is the share of the pile belonging to label i, and Σ means "add this up over every label". Squaring a share gives the chance that *two* random draws both land on that label; adding those up gives the chance the two draws *agree*; subtracting from 1 flips it to the chance they *disagree*. Zero means perfectly sorted; bigger means messier.

(You may also meet **entropy**, an alternative messiness measure from information theory. It ranks splits almost identically in practice; Gini is the common default because it is cheaper to compute.)

### 3.3 Scoring a question

A question splits a pile into a **YES pile** and a **NO pile**. Its score is the **weighted average of the two piles' Gini impurities** — each pile counts in proportion to its size. Lower is better.

Worked example on our 12 tickets, for *contains "refund"?*:

- YES pile: 4 tickets, all billing → Gini 0.0
- NO pile: 8 tickets, 4 shipping + 4 account → Gini 0.5
- score = (4/12) × 0.0 + (8/12) × 0.5 = **0.333**

Compare *contains "order"?* — "order" appears in one shipping ticket and one billing ticket:

- YES pile: 2 tickets, 1 shipping + 1 billing → Gini 0.5
- NO pile: 10 tickets, 3 shipping + 3 billing + 4 account → Gini 0.66
- score = (2/12) × 0.5 + (10/12) × 0.66 = **0.633**

So "refund" (score 0.333) is a much better question than "order" (score 0.633) — remember, lower is better. The weighting matters: a question that purifies a tiny pile but leaves a huge messy one has barely helped, and the size-weighting makes the huge pile dominate the score. One edge case: a word that appears in *none* (or *all*) of a pile's tickets splits nothing and is skipped.

### 3.4 Growing the tree: greed plus recursion

The algorithm, in full:

1. Score every question on the menu against the current pile. Keep the best (lowest score).
2. Split the pile into the YES pile and the NO pile.
3. Treat each side as a brand-new, smaller problem, and repeat from step 1.
4. Stop and place a **leaf** when: the pile is **pure** (Gini 0), or the **depth limit** is reached, or **no question splits the pile** at all. A leaf predicts the **majority label** of the tickets that landed in it.

Step 3 is **recursion** — a function calling itself on smaller pieces of the same problem. Step 1 makes the algorithm **greedy**: it picks the question that looks best *right now* and never reconsiders, even if a less-obvious first question would have set up a cleverer second question. Greedy is not guaranteed optimal, but it is fast and works well, and it is how every real tree library grows trees.

#### The full trace: growing our whole tree, node by node

Let's grow the actual tree, doing every calculation in the open. First, number the 12 training tickets and note which menu words each one contains — that is all the tree can see:

| # | Ticket (shortened) | Menu words it contains | Label |
|---|---|---|---|
| T1 | Where is my **package** it has not arrived | package | shipping |
| T2 | My **order** says delivered but no **package** came | package, order | shipping |
| T3 | The **delivery** is late and tracking has not updated | delivery | shipping |
| T4 | My **package** arrived damaged and crushed | package | shipping |
| T5 | I was **charged** twice for my **order** please **refund** me | charged, order, refund | billing |
| T6 | I want a **refund** for this purchase | refund | billing |
| T7 | Why was I **charged** extra fees I want a **refund** | charged, refund | billing |
| T8 | Requesting a **refund** because the discount was not applied | refund | billing |
| T9 | My **login** is not working after the update | login | account |
| T10 | I forgot my **password** and the reset email never comes | password | account |
| T11 | Please **help** me change the email on my **account** | help, account | account |
| T12 | My **account** is locked and I need to reset my **password** | account, password | account |

**NODE A — the root.** Pile: all 12 tickets (4 shipping, 4 billing, 4 account). Gini = 1 − 3 × (1/3)² = **0.667**. Now score all 9 questions. For each question: put the tickets that contain the word in the YES pile, the rest in the NO pile, compute each pile's Gini, and take the size-weighted average.

| Question | YES pile | Gini yes | NO pile | Gini no | Score |
|---|---|---|---|---|---|
| **refund?** | T5 T6 T7 T8 = 4 billing | 0.0 | 8 left: 4 ship + 4 acct | 0.5 | (4/12)×0.0 + (8/12)×0.5 = **0.333** ← winner |
| charged? | T5 T7 = 2 billing | 0.0 | 10 left: 4 ship + 2 bill + 4 acct | 0.64 | (2/12)×0.0 + (10/12)×0.64 = 0.533 |
| package? | T1 T2 T4 = 3 ship | 0.0 | 9 left: 1 ship + 4 bill + 4 acct | 0.593 | (3/12)×0.0 + (9/12)×0.593 = 0.444 |
| delivery? | T3 = 1 ship | 0.0 | 11 left: 3 ship + 4 bill + 4 acct | 0.661 | (1/12)×0.0 + (11/12)×0.661 = 0.606 |
| order? | T2 T5 = 1 ship + 1 bill | 0.5 | 10 left: 3 ship + 3 bill + 4 acct | 0.66 | (2/12)×0.5 + (10/12)×0.66 = 0.633 |
| password? | T10 T12 = 2 acct | 0.0 | 10 left: 4 ship + 4 bill + 2 acct | 0.64 | 0.533 |
| account? | T11 T12 = 2 acct | 0.0 | 10 left: 4 ship + 4 bill + 2 acct | 0.64 | 0.533 |
| login? | T9 = 1 acct | 0.0 | 11 left: 4 ship + 4 bill + 3 acct | 0.661 | 0.606 |
| help? | T11 = 1 acct | 0.0 | 11 left: 4 ship + 4 bill + 3 acct | 0.661 | 0.606 |

One "Gini no" worked fully, so you can check the rest the same way — "charged?", NO pile of 10 tickets (4 shipping, 2 billing, 4 account): shares are 0.4, 0.2, 0.4, so Gini = 1 − 0.4² − 0.2² − 0.4² = 1 − 0.16 − 0.04 − 0.16 = **0.64**.

Notice something important in this table: **almost every question makes a perfectly pure YES pile** (Gini 0.0). Purity alone is not enough — what separates the winner is *size*. "refund" pulls **four** tickets into its pure pile, "delivery" only one, so "refund" leaves far less messiness behind overall. That is exactly what the size-weighting is for. "refund" wins with 0.333. YES side (T5 T6 T7 T8) is pure billing → **leaf: billing**. NO side is not pure → grow a node there.

**NODE B — the "no refund" pile.** Pile: T1 T2 T3 T4 (shipping) + T9 T10 T11 T12 (account). Gini = 1 − (1/2)² − (1/2)² = **0.5**. Re-score the whole menu *against this smaller pile* — the scores all change, because the pile changed:

| Question | YES pile | Gini yes | NO pile | Gini no | Score |
|---|---|---|---|---|---|
| refund? | nobody left contains it | — | — | — | skipped |
| charged? | nobody left contains it | — | — | — | skipped |
| **package?** | T1 T2 T4 = 3 ship | 0.0 | 5 left: T3 + 4 acct | 0.32 | (3/8)×0.0 + (5/8)×0.32 = **0.200** ← winner |
| delivery? | T3 = 1 ship | 0.0 | 7 left: 3 ship + 4 acct | 0.490 | (1/8)×0.0 + (7/8)×0.490 = 0.429 |
| order? | T2 = 1 ship | 0.0 | 7 left: 3 ship + 4 acct | 0.490 | 0.429 |
| password? | T10 T12 = 2 acct | 0.0 | 6 left: 4 ship + 2 acct | 0.444 | (2/8)×0.0 + (6/8)×0.444 = 0.333 |
| account? | T11 T12 = 2 acct | 0.0 | 6 left: 4 ship + 2 acct | 0.444 | 0.333 |
| login? | T9 = 1 acct | 0.0 | 7 left: 4 ship + 3 acct | 0.490 | 0.429 |
| help? | T11 = 1 acct | 0.0 | 7 left: 4 ship + 3 acct | 0.490 | 0.429 |

Two lessons hide in this table. First, "refund" and "charged" are **skipped**: every remaining ticket answers "no", so asking would split nothing (this also shows the tree never reuses a question by accident — once a word's tickets are gone, the question dies naturally). Second, "order" scored a terrible 0.633 at the root but a decent 0.429 here — **the same question has a different value on a different pile**, which is why the tree re-scores everything at every node. "package" wins with 0.200. YES side (T1 T2 T4) is pure shipping → **leaf: shipping**. NO side is not pure → grow again.

**NODE C — the "no refund, no package" pile.** Pile: T3 (shipping) + T9 T10 T11 T12 (account). Gini = 1 − (1/5)² − (4/5)² = 1 − 0.04 − 0.64 = **0.32**. Re-score once more (refund, charged, package, and order are all skipped now — nobody left contains them):

| Question | YES pile | Gini yes | NO pile | Gini no | Score |
|---|---|---|---|---|---|
| **delivery?** | T3 = 1 ship | 0.0 | T9 T10 T11 T12 = 4 acct | 0.0 | (1/5)×0.0 + (4/5)×0.0 = **0.000** ← perfect split |
| password? | T10 T12 = 2 acct | 0.0 | 3 left: T3 + T9 + T11 = 1 ship + 2 acct | 0.444 | (2/5)×0.0 + (3/5)×0.444 = 0.267 |
| account? | T11 T12 = 2 acct | 0.0 | 3 left: T3 + T9 + T10 = 1 ship + 2 acct | 0.444 | 0.267 |
| login? | T9 = 1 acct | 0.0 | 4 left: T3 + 3 acct | 0.375 | (1/5)×0.0 + (4/5)×0.375 = 0.300 |
| help? | T11 = 1 acct | 0.0 | 4 left: T3 + 3 acct | 0.375 | 0.300 |

"delivery" scores a perfect 0.000 — both sides come out pure — so both children become leaves and the growing stops everywhere (stopping rule 1: pure piles).

**The finished tree, with every ticket's landing place:**

```text
NODE A — all 12 tickets (4 ship, 4 bill, 4 acct), Gini 0.667
ask: contains "refund"?   (score 0.333 — best of 9)
├─ yes → T5 T6 T7 T8 — all billing, Gini 0        → LEAF: predict billing
└─ no  → NODE B — T1 T2 T3 T4 T9 T10 T11 T12, Gini 0.5
         ask: contains "package"?   (score 0.200 — best of the 7 usable)
         ├─ yes → T1 T2 T4 — all shipping, Gini 0  → LEAF: predict shipping
         └─ no  → NODE C — T3 T9 T10 T11 T12, Gini 0.32
                  ask: contains "delivery"?   (score 0.000 — a perfect split)
                  ├─ yes → T3 — shipping, Gini 0    → LEAF: predict shipping
                  └─ no  → T9 T10 T11 T12 — all account, Gini 0 → LEAF: predict account
```

Every leaf is pure, so every training ticket is classified correctly: training accuracy 12/12. (And now you can see precisely what the depth-2 version in the next section does: it cuts the tree off at NODE C, turning that whole 5-ticket pile into one leaf that predicts its majority label, `account` — 4 of its 5 tickets — and sacrifices T3.)

### 3.5 The suspicious branch: trees overfit eagerly

Look again at that last split: the "delivery" branch exists to rescue **exactly one training ticket**. That should make you suspicious. Is "delivery → shipping" a real pattern, or did the tree just memorize one ticket's quirk?

This is the tree version of Lesson 13's warning. **Left unlimited, a tree will keep splitting until every leaf is pure — which is memorization by construction.** Every noisy, mislabelled, or one-off ticket gets its own private branch. Perfect training accuracy, poor generalization: the classic high-variance overfit.

The main defense is the **depth limit** (`max_depth`). Rebuild our tree with depth limit 2 and the "delivery" branch is not allowed to exist; the leftover pile (1 shipping + 4 account) becomes a leaf predicting `account`. Training accuracy drops to 11/12 — and that can easily be the *better* model on new tickets, because it stopped carving branches for individuals. Depth is the tree's bias-variance knob: too shallow underfits, too deep overfits. (Real libraries offer more pruning knobs — minimum tickets per leaf, minimum improvement per split — all serving the same purpose: stop growing before you memorize.)

### 3.6 Trees are unstable — the random forest fixes that

Trees have a second weakness: **instability**. Change a couple of training tickets and the best root question can flip — and since every later split depends on the earlier ones, the whole tree can come out different. One tree is one opinion formed by a chain of greedy choices.

The fix: **train many slightly-different trees and let them vote.** This is a **random forest**. The trees are made different on purpose, in two ways:

1. **Bootstrap sampling.** Each tree trains not on the original 12 tickets but on 12 tickets **drawn at random with replacement** — some tickets appear twice, some not at all. Each tree sees a slightly different version of history. (Drawing with replacement is a *bootstrap sample*; training on resamples and combining the results is called **bagging** — "bootstrap aggregating".)
2. **Random question menus.** Each tree may only use a random subset of the candidate questions. This stops every tree from grabbing the same dominant first split and forces the forest to find *different* routes to the answer. (Real forests re-draw the subset at every split; our lesson code draws one menu per tree to stay short — same idea.)

**Why voting works:** every tree makes mistakes, but because each saw different data and different questions, they make *different* mistakes. In the vote, the errors point in different directions and largely cancel, while the true signal — which all of them partly captured — adds up. It is the same reason averaging many noisy measurements gives a better answer than trusting one measurement, and the same reason a panel of judges is more reliable than a single judge. In the bias-variance words from Lesson 13: averaging many high-variance trees removes most of the variance without adding much bias. That is why a forest usually does better than even its best single tree, and why forests need much less careful depth-tuning than a single tree does.

The cost: a forest of 100 trees takes 100 times the computing work, and you lose the single tree's biggest advantage — you can no longer print one small flowchart and *read* the model's reasoning.

## SECTION 4 — THE CODE

Saved: `lesson-notebooks/phase-02-classical-machine-learning/chapter-05-core-supervised-algorithms/lesson-16-decision-trees-random-forests.ipynb`

The notebook builds everything from scratch, in plain Python with no external libraries:

1. 12 labelled tickets (4 shipping, 4 billing, 4 account) and a 9-word question menu
2. `tokenize` / `has_word` — the yes/no questions
3. `gini` — the messiness score, demonstrated on a pure pile, a 50/50 pile, and the full pile
4. `split_quality` — a question's weighted-Gini score, walked through for "refund", "order", and "delivery"
5. `best_word` — try every question, keep the lowest score (the full score table is printed)
6. `build_tree` — recursion plus the three stopping rules; `format_tree` prints the learned flowchart
7. `predict` — walks a ticket down the tree; training accuracy 12/12
8. Four unseen tickets, including one deliberately designed to fool the tree
9. A depth-2 rebuild showing the overfitting knob in action (11/12 on training, one memorized branch gone)
10. `bootstrap_sample`, `build_forest`, `forest_predict` — a 5-tree random forest that votes

How to run it: open the notebook in VS Code and run the cells top to bottom, one at a time. The forest cells use Python's built-in `random` module with a fixed seed, so re-running the whole notebook from the top reproduces the same forest.

## SECTION 5 — IF YOU RAN THIS

What would happen, step by step:

1. The data cell prints 4 tickets per category — a balanced three-way pile.
2. The Gini cell prints `0.0` for the pure pile, `0.5` for the 50/50 pile, `0.667` for the full pile.
3. The scoring cell shows "refund" splitting off a perfectly pure billing pile (score 0.333) while "order" barely helps (0.633).
4. The best-question table ranks all nine words; "refund" wins the root.
5. `build_tree` grows the refund → package → delivery tree shown in Section 3, and `predict` scores 12/12 on training data.
6. The unseen-ticket cell routes three tickets correctly and misroutes the trap ("The delivery fee was charged to the wrong card" → shipping, though it is really billing).
7. The depth-2 rebuild prints the smaller tree, reports the one training ticket it now misses (11/12).
8. The forest cells print a bootstrap sample (with visible repeats), two of the five trees (asking different questions), and majority votes on two test tickets.

**Expected output — a PREDICTION, not verified.** Key cells (the deterministic ones):

```text
Scores at the root (lower = better):
  refund: 0.333
  charged: 0.533
  package: 0.444
  delivery: 0.606
  order: 0.633
  password: 0.533
  account: 0.533
  login: 0.606
  help: 0.606

Best first question: refund
```

```text
The learned tree:
contains "refund"?
  yes: predict billing
  no:  contains "package"?
    yes: predict shipping
    no:  contains "delivery"?
      yes: predict shipping
      no:  predict account
```

```text
  [shipping] My package never arrived and tracking is stuck
  [billing] I was charged for an order I cancelled please refund me
  [account] I cannot reset my password on my account
  [shipping] The delivery fee was charged to the wrong card
```

The forest cells' *exact* printed trees and vote counts depend on the random draws (even with the fixed seed, I have not run this — treat every number above as a prediction). The **decisions** on the two clear test tickets should come out shipping and billing; if a vote count differs from what you expected, that is normal.

## SECTION 6 — APPLIED TO OUR ASSISTANT

This lesson delivers the capability we framed in Lesson 11: **ticket category routing**.

- **New module — `customer-support-assistant/ticket_category.py`:** a decision-tree classifier trained at import time on 12 labelled tickets, exposing `predict_category(text)` → `"shipping"` / `"billing"` / `"account"`. Pure standard library, fully deterministic, and the learned tree can be printed with `format_tree`. Its `__main__` block prints the tree, routes three demo tickets, and `assert`s both the demo routings and 12/12 training accuracy — run `python ticket_category.py` any time as a regression check.
- **New companion notebook — `customer-support-assistant/notebooks/ticket_category.ipynb`:** the same feature, cell by cell, for reading and experimenting.
- **Updated — `customer-support-assistant/main.py`:** now imports `predict_category` and demonstrates capability 3 alongside urgency detection and resolution-time estimation, with three new `assert`s.

The project keeps a **single tree**, not a forest: with 12 training tickets a forest adds randomness without real benefit, and the single tree stays printable and fully traceable. The forest lives in the lesson notebook; a later lesson (scikit-learn, Chapter 9) upgrades this feature in place.

`requirements.txt` stays empty — still standard library only. Predicted output of `python main.py` (run from inside `customer-support-assistant/`), **a prediction — the probability and minute numbers especially will differ a little**:

```text
Customer Support Assistant - current capabilities

1) Ticket urgency detection (learned by logistic regression)
   [urgent] (P(urgent)=0.93) My payment failed twice and I need this resolved right now
   [not urgent] (P(urgent)=0.05) Do you have this item in a larger size

2) Estimated resolution time (learned by linear regression)
   ~11.0 min (11 words) My payment failed twice and I need this resolved right now
   ~10.0 min (9 words) Do you have this item in a larger size

3) Ticket category routing (learned by a decision tree)
   [shipping] Where is my package it has been two weeks
   [billing] I was charged twice and want a refund
   [account] I cannot log in please reset my password

Self-check passed.
```

## SECTION 7 — COMMON MISTAKES AND GOTCHAS

- **Letting a tree grow without limits.** An unlimited tree reaches pure leaves by memorizing individual examples. Perfect training accuracy from a deep tree is a warning sign, not a victory. Always cap depth (or leaf size) and check against held-out data.
- **Reading feature questions as understanding.** The tree asks about *exact words*. "The delivery fee was charged to the wrong card" goes to shipping because of one word. And "delivered" ≠ "delivery" — a ticket saying "not delivered" does not trigger the delivery question at all.
- **Forgetting the default path.** Every "none of the above" ticket slides down the no-branches into one leaf (here: `account`). A gift-wrapping question would be routed to account support — the tree has no "I don't know" option unless you build one.
- **Trusting one tree's structure too much.** Tiny data changes can reshuffle the whole tree. If you retrain and the tree looks different, that is instability, not a bug — and it is exactly why forests exist.
- **Assuming the forest fixes everything.** A forest averages away variance (noise-chasing), not bias. If the features themselves are too weak — like our 9-word menu — a hundred trees vote confidently on the same missing information.

## SECTION 8 — WHEN TO USE THIS, AND TRADEOFFS

**Reach for trees/forests when:** your data is a table of mixed features (numbers, categories, yes/no flags); features live on wildly different scales (trees don't care — no feature scaling needed, unlike Lesson 14's gradient descent); relationships are rule-like rather than line-like; or you need a model a human can read (single tree) or a strong low-effort baseline (forest). On tabular business data, forests and gradient boosting are usually the first serious models to try.

**Costs and limits:** single trees are unstable and overfit eagerly; forests fix both but cost more compute and lose readability. Tree splits are axis-aligned yes/no cuts, so smooth linear trends need many steps to approximate — logistic/linear regression handles those more gracefully with two numbers. And for raw text, word-presence questions are shallow; better text features (Chapter 6) and, eventually, models that read meaning (Phase 4+) go deeper. For calibrated probabilities, logistic regression (Lesson 15) is still the cleaner tool — a leaf's majority share is a cruder confidence than a sigmoid's output.

## SECTION 9 — KEY TAKEAWAYS

- A decision tree classifies by walking an example through a flowchart of yes/no questions, and it learns both which questions to ask and in what order purely from labelled data.
- Gini impurity measures how mixed a pile of labels is (the chance two random draws disagree), and each split greedily minimizes the size-weighted impurity of the two piles it creates.
- An unlimited tree will grow a private branch for every quirky example and reach perfect training accuracy by memorization, so the depth limit is the tree's main bias-variance knob.
- A random forest trains many trees on bootstrap samples with restricted question menus and takes a majority vote, which cancels the trees' individual, differently-directed errors and slashes variance.
- Our assistant now routes tickets to shipping, billing, or account support with a learned, printable decision tree — the ticket-category capability promised in Lesson 11.

## SECTION 10 — CHALLENGE

You are given 6 labelled tickets and a menu of just two candidate questions: *contains "return"?* and *contains "broken"?*

| # | Ticket | Label |
|---|--------|-------|
| 1 | I want to return these shoes | returns |
| 2 | How do I return a gift | returns |
| 3 | Please help me return my order | returns |
| 4 | The zipper is broken after one day | product |
| 5 | My blender arrived broken | product |
| 6 | I want to return this broken toy | returns |

(a) Compute the Gini impurity of the full 6-ticket pile. (b) Score both candidate questions (weighted Gini). (c) Which question should the root ask, and what does the finished tree look like?

<details>
<summary>Click to reveal the answer</summary>

**(a) The full pile.** 4 returns + 2 product. Shares: 4/6 = 2/3 and 2/6 = 1/3.

> Gini = 1 − (2/3)² − (1/3)² = 1 − 4/9 − 1/9 = 4/9 ≈ **0.444**

**(b) Score each question.**

*Contains "return"?* — tickets 1, 2, 3, 6 say yes (note ticket 6 contains both words):

- YES pile: {1, 2, 3, 6} = 4 returns → Gini = 1 − 1² = 0.0
- NO pile: {4, 5} = 2 product → Gini = 0.0
- score = (4/6) × 0.0 + (2/6) × 0.0 = **0.0**

*Contains "broken"?* — tickets 4, 5, 6 say yes:

- YES pile: {4, 5, 6} = 2 product + 1 returns → Gini = 1 − (2/3)² − (1/3)² = 4/9 ≈ 0.444
- NO pile: {1, 2, 3} = 3 returns → Gini = 0.0
- score = (3/6) × 0.444 + (3/6) × 0.0 = **0.222**

**(c) The root asks "return".** Its score (0.0) beats "broken" (0.222). Both resulting piles are already pure, so both children are leaves and the tree is a single question:

```text
contains "return"?
  yes: predict returns
  no:  predict product
```

The interesting ticket is #6: it mentions "broken", but the customer *wants a return* — and the "return" split classifies it correctly, while a root split on "broken" would have polluted its yes-pile with it. The impurity arithmetic detected that automatically: that is exactly what the weighted-Gini score is for.

</details>

## SECTION 11 — WHAT IS NEXT

The forest made its trees independently and averaged them. Lesson 17 — **gradient boosting** — builds trees *in sequence*, each new tree trained to correct the mistakes of the ones before it. That one change produces the family of models (XGBoost, LightGBM) that dominates tabular-data competitions to this day.

---

*Lesson 16 of 113 — Phase 2 (Classical Machine Learning), Chapter 5 (Core Supervised Algorithms), lesson 3 of 5 in this chapter.*
