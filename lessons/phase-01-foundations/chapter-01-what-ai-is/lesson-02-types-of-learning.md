# Lesson 02 — The Main Types of Machine Learning

**Phase 1 — Foundations, Chapter 1 — What AI, Machine Learning, and Deep Learning Really Are, Lesson 2**

**Files created:**
- `lessons/phase-01-foundations/chapter-01-what-ai-is/lesson-02-types-of-learning.md` (this file)
- `lesson-notebooks/phase-01-foundations/chapter-01-what-ai-is/lesson-02-types-of-learning.ipynb`

**Prerequisite lessons:** Lesson 01 — AI, machine learning, deep learning, and generative AI.

---

## 1. Why This Matters

Lesson 1 established that machine learning means "learn patterns from data instead of hand-writing rules." But that one idea splits into three genuinely different *situations*, depending on what kind of data you actually have. Do you already know the right answers for your examples? Do you have a pile of examples with no answers at all? Or do you not have a fixed pile of examples at all, just a chance to act and see what happens?

Getting this right matters immediately, because it decides which tool you reach for. Later in this course you will build a classifier that needs labeled tickets (supervised learning), and eventually an agent that needs to try actions and learn from outcomes (reinforcement learning). If you use the wrong type of learning for the situation you're in, nothing works — it's like trying to teach by handing out an answer key that does not exist.

## 2. Real-World Analogy

Picture three different ways someone might learn to cook a new dish.

**Supervised learning is like learning from a recipe book with photos of the finished dish.** Every recipe (the input) comes with a picture of exactly what it should look like when done (the correct answer, the **label**). You cook, compare your result to the photo, adjust, and repeat. You are always learning *against a known right answer*.

**Unsupervised learning is like being handed a fridge full of random ingredients with no recipes at all**, and being asked to group them into categories that make sense — "these are all dairy," "these are all vegetables" — using only what you notice about them yourself. Nobody tells you the categories exist; you invent them from the patterns you see.

**Reinforcement learning is like learning to cook a brand-new dish purely by tasting your own attempts.** Nobody hands you a recipe or a finished photo. You just try something, taste it (get a reward — good or bad), adjust next time based only on that feedback, and slowly get better through repeated trial and error.

Same overall goal — get good at cooking — but three completely different learning situations, needing three completely different strategies.

## 3. The Concept Explained

### Supervised Learning

**Plain definition:** supervised learning is training a program using examples that already have the correct answer attached. The program's whole job during training is to find a pattern that connects the example to its attached answer, so it can guess the answer for new examples later.

**Why it exists:** many real problems have plenty of past examples where we already know what the "right answer" turned out to be — past emails already marked spam or not, past X-rays a doctor already diagnosed, past tickets a human already sorted into a category. Supervised learning is how a program turns that historical answer key into a general-purpose skill.

**Tiny concrete example:** show a program 6 support tickets, each already tagged `billing`, `shipping`, or `account`. It notices that words like "charged" and "billed" show up mostly in tickets tagged `billing`, while "tracking" and "package" show up mostly in tickets tagged `shipping`. Now, given a brand-new ticket saying "my card was charged twice," it can guess `billing`, even though it never saw that exact sentence before, because it learned the *pattern*, not the specific sentence.

**Technical name:** Supervised Learning. The attached correct answer is called a **label**, and data that has labels attached is called **labeled data**.

### Unsupervised Learning

**Plain definition:** unsupervised learning is finding structure, patterns, or groupings in data that has *no* attached answers at all. Nobody tells the program what the "right" groups are — it has to notice similarity on its own.

**Why it exists:** most real-world data does not come with a neat answer key. Getting a human to label every single example is often slow, expensive, or simply impossible at scale. Unsupervised learning lets you still get value — spotting groups, patterns, or oddities — from raw, unlabeled data.

**Tiny concrete example:** give a program the same 6 support tickets, but with the `billing`/`shipping`/`account` tags hidden. It notices that two of the tickets share the words "was" and "for," and groups just those two together — without ever being told what "billing" means, or that a category called "billing" even exists. It only ever notices "these two are similar to each other," never "these two are the billing ones."

**Technical name:** Unsupervised Learning. A common unsupervised task — sorting examples into groups of similar things — is called **clustering**.

### Reinforcement Learning

**Plain definition:** reinforcement learning is learning by repeatedly taking an action, getting a signal back for how good or bad that action turned out to be, and using that signal to choose better actions next time. There is no upfront labeled dataset at all — the only "data" is the feedback the learner generates for itself by acting.

**Why it exists:** some problems are not "look at this fixed example and guess its answer." They are "make a sequence of decisions in a changing situation, and learn which decisions tend to work out." Nobody has a ready-made answer key for "should the robot turn left or right right now" — the only way to find out is to try it and see what happens.

**Tiny concrete example:** picture a robot in front of two vending machines, `A` and `B`, that each secretly have their own odds of giving a treat. The robot does not know either machine's odds in advance. It tries `A`, gets nothing, tries `B`, gets a treat, and starts favoring `B` — purely because of what happened when it tried each one, not because anyone told it "`B` is the better machine."

**Technical name:** Reinforcement Learning (RL). The decision-maker is called the **agent**, the thing it interacts with is the **environment**, each decision it makes is an **action**, and the feedback signal after each action is the **reward**.

### Which type fits which kind of problem

```
Do you already have examples WITH the correct answer attached?
├── Yes  ->  SUPERVISED LEARNING
│           (e.g. past tickets already tagged with their category)
│
└── No, but you have a pile of examples with NO answers at all
    ├── You want to find groups/patterns hiding in that pile
    │        ->  UNSUPERVISED LEARNING
    │           (e.g. finding that certain tickets tend to cluster together)
    │
    └── You don't have a fixed pile at all — you have a chance to ACT
        and see what happens, repeatedly, in a changing situation
             ->  REINFORCEMENT LEARNING
            (e.g. an agent deciding which action earns the best outcome)
```

Most of what you will build early in this course is supervised learning, because most business data (past tickets, past sales, past customers who did or didn't cancel) already comes with a known outcome attached. Reinforcement learning shows up later in the course (Phase 3 and again in Phase 7, when we cover how modern language models are tuned using human feedback) — it needs a very different setup: something that can act repeatedly and receive a reward, not just a static pile of examples.

## 4. The Code

The notebook for this lesson builds one tiny, from-scratch demo for each of the three types, all using the same six support tickets so they are easy to compare side by side.

Open it here: `lesson-notebooks/phase-01-foundations/chapter-01-what-ai-is/lesson-02-types-of-learning.ipynb`

*(Lesson markdown files hold only explanation and never contain runnable code themselves — the notebook above is where you actually run something.)*

The notebook's markdown cells carry the explanation of each step; the code cells are kept clean, matching this course's format. Broadly, it:

1. **Part 1 (supervised):** trains word-count scores per category from six labeled tickets, then predicts the category of two brand-new tickets by adding up their words' scores.
2. **Part 2 (unsupervised):** ignores the labels entirely, and groups any two tickets that share at least two words, purely from overlapping vocabulary.
3. **Part 3 (reinforcement):** simulates an agent choosing between two vending machines over 8 trials, always picking whichever machine's running-average reward currently looks best.

*(Library note: this lesson uses only Python's standard library — `collections.Counter` and `itertools` — so there is nothing here that changes between versions.)*

## 5. If You Ran This

Walking through the notebook top to bottom:

**Part 1 — supervised prediction cell.** The categorizer is trained on the six labeled tickets, then tested on two new ones.

- The first new ticket, "My refund never showed up on my card," scores `billing` and `account` exactly tied at 4 (both `shipping` at 2). Because the code picks the first category that reaches the highest score while scanning in order (`billing`, `shipping`, `account`), it prints `billing` — which happens to be correct, since this ticket really is about a billing problem.
- The second new ticket, "Where is my order, it has not arrived yet," is genuinely about shipping, but scores `account` highest (6, versus 2 and 2), because common words like "is," "it," and "not" happened to appear more often in the small `account` training examples. The categorizer prints `account` — a wrong prediction, caused entirely by ordinary words contaminating a tiny training set, exactly the same weakness Lesson 1's challenge uncovered.

**Prediction** of the printed lines:

```
My refund never showed up on my card
  predicted category: billing  (scores: {'billing': 4, 'shipping': 2, 'account': 4})
Where is my order, it has not arrived yet
  predicted category: account  (scores: {'billing': 2, 'shipping': 2, 'account': 6})
```

**Part 2 — unsupervised grouping cell.** Comparing every pair of the six tickets by shared vocabulary alone:

```
Ticket 0 and Ticket 1 share 2 words -> grouped together
  My card was charged twice for one order
  I was billed for an item I never received
Ticket 4 and Ticket 5 share 2 words -> grouped together
  I can not log into my account
  Please reset my password, it is not working
```

Notice tickets 2 and 3 (both genuinely about shipping) are never grouped — they share zero words with each other, so this simple method finds two of the three true groupings and misses the third completely.

**Part 3 — reinforcement learning cell.** Machine `A` starts tied with `B` (both assumed 0.5, since neither has been tried), so `A` is tried first (ties go to whichever is checked first) and gets a `0`. From then on `B`'s running average stays above `A`'s zero for the rest of the run, so the agent keeps choosing `B` for all 7 remaining trials and never gives `A` a second chance.

**Prediction** of the printed lines:

```
Trial 1: tried machine A, got reward 0
Trial 2: tried machine B, got reward 1
Trial 3: tried machine B, got reward 1
Trial 4: tried machine B, got reward 0
Trial 5: tried machine B, got reward 1
Trial 6: tried machine B, got reward 1
Trial 7: tried machine B, got reward 0
Trial 8: tried machine B, got reward 1

Machine A: tried 1 times, average reward 0.00
Machine B: tried 7 times, average reward 0.71
```

This is a prediction based on tracing through the code by hand, not a verified run — the learner should open the notebook and run it to confirm.

## 6. Applied to Our Assistant

This lesson is about building the mental map for *which type of learning fits which situation* — it does not introduce a new technique meant to be implemented in the project right now, so there is no new `.py` file or `main.py` change this lesson. The project stays exactly as it was after Lesson 1, still fully runnable with `python main.py` from inside `customer-support-assistant/`.

It is worth noticing, though, that `ticket_urgency.py` — built in Lesson 1 — is already a tiny example of **supervised learning**: it was trained on eight tickets that each came with a known "urgent" or "not urgent" answer attached. As the course continues, you will see this same project eventually gain features from all three types covered today: supervised classifiers (Phase 2), and later, an agent that takes actions and learns from outcomes (Phase 8).

## 7. Common Mistakes and Gotchas

- **Assuming "no labels" always means unsupervised learning is the answer.** Sometimes it means you should go label some data instead, because supervised learning would work far better if you did — unsupervised learning is a fallback for when labeling truly is not feasible, not automatically the better choice.
- **Expecting unsupervised clustering to find the groupings you had in mind.** As Part 2 showed, clustering only finds structure that is actually visible in the data's surface features (here, shared words) — it can miss a grouping that is obvious to a human but invisible in the specific signal the algorithm is using.
- **Thinking reinforcement learning needs a "correct answer" at all.** It never gets one — only a reward signal telling it how good an action turned out to be, which is a fundamentally different, sparser kind of feedback than a label.
- **Only ever exploiting, never exploring.** Our Part 3 agent always chose whichever machine currently looked best, and once `A` looked bad after a single unlucky try, it was never tried again — even though its true long-run odds might have been fine. Real reinforcement learning needs a way to occasionally try the "worse-looking" option anyway, a tradeoff we return to properly in Lesson 39.
- **Mixing up "the type of learning" with "the specific algorithm."** Supervised, unsupervised, and reinforcement learning are *categories* of problem setup — decision trees, k-means, and Q-learning (covered later) are specific *algorithms* that live inside those categories.

## 8. When to Use This, and Tradeoffs

Reach for supervised learning whenever you have — or can reasonably get — examples with known correct answers; it is the most reliable and easiest-to-evaluate of the three, which is why most of Phase 2 focuses on it. Reach for unsupervised learning when you have plenty of data but no answer key, and you want to explore it for groupings or patterns you did not already know to look for — but stay honest that what it finds depends entirely on which features it can see. Reach for reinforcement learning only when your problem is genuinely about a sequence of actions with feedback over time, not a fixed pile of examples — it is powerful but by far the hardest of the three to set up and get working reliably, which is part of why it is introduced later in this course, after the fundamentals are solid.

## 9. Key Takeaways

- Supervised learning trains on examples that already have the correct answer attached, called labels, and is the most common starting point for real business problems.
- Unsupervised learning finds structure or groupings in data that has no labels at all, but what it finds is limited to whatever pattern is actually visible in the data it's given.
- Reinforcement learning learns by taking actions in a changing situation and adjusting based on a reward signal, with no upfront dataset of correct answers.
- Choosing the right type starts with asking what data you actually have: labeled examples, unlabeled examples, or just a chance to act and observe feedback.
- The customer-support assistant will end up using all three types over the course: supervised classifiers first, unsupervised techniques for exploring customer data, and reinforcement-learning-adjacent ideas once we reach agents in Phase 8.

## 10. Challenge

Using the trained word counts from `category_word_counts` in the notebook, work out by hand which category the supervised approach from Part 1 would predict for this brand-new ticket, and show your work:

> "Can I get my password fixed for my account"

Tokenized (lowercased, no punctuation to strip): `can`, `i`, `get`, `my`, `password`, `fixed`, `for`, `my`, `account`

<details>
<summary>Click to reveal the answer</summary>

Recall the three tally sheets built from the six training examples (only the relevant words are shown):

- **billing:** `i`:2, `my`:1, `for`:2
- **shipping:** `my`:1
- **account:** `can`:1, `i`:1, `my`:2, `password`:1, `account`:1

Word by word, remembering `my` appears twice in this ticket:

**billing** = can(0) + i(2) + get(0) + my(1) + password(0) + fixed(0) + for(2) + my(1) + account(0) = **6**

**shipping** = can(0) + i(0) + get(0) + my(1) + password(0) + fixed(0) + for(0) + my(1) + account(0) = **2**

**account** = can(1) + i(1) + get(0) + my(2) + password(1) + fixed(0) + for(0) + my(2) + account(1) = **8**

`account` scores highest (8, versus 6 and 2), so the supervised approach predicts **account** — correctly, since this ticket is clearly about a password and account problem. Unlike the two tickets tested in the notebook, there is no tie and no contamination from common words here, because the words that matter most ("password," "account") only ever appeared in the `account` training examples.

</details>

## 11. What Is Next

Lesson 3 asks why AI works so well *right now*, covering the three ingredients — data, computing power, and better algorithms — plus a short, honest history from early AI up through today's large language models.

---

**Lesson 2 of 113 — Phase 1, Chapter 1, Lesson 2 of 3 in this chapter.**
