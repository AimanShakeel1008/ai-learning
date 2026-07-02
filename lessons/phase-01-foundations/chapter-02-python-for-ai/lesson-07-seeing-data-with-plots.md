# Lesson 07 — Seeing Your Data With Plots

**Phase 1 — Foundations, Chapter 2 — Python for AI, Lesson 4**

**Files created:**
- `lessons/phase-01-foundations/chapter-02-python-for-ai/lesson-07-seeing-data-with-plots.md` (this file)
- `lesson-notebooks/phase-01-foundations/chapter-02-python-for-ai/lesson-07-seeing-data-with-plots.ipynb`

**Prerequisite lessons:** Lesson 06 — pandas for handling data.

---

## 1. Why This Matters

Lesson 06 built the habit of looking at a table before touching it: `.head()`, `.shape`, `.info()`. Those tools describe a table in words and numbers. They still miss things that jump out the moment the same data is drawn as a picture — a handful of outliers, a lopsided category count, two numbers that quietly move together (or, just as usefully, that don't move together at all).

This is not decoration before the "real" work of modeling. Every experienced practitioner looks at their data visually before training anything, because a model trained on data with a hidden problem — one wildly imbalanced category, one outlier skewing an average, a data-entry mistake sitting off in a corner of a scatter plot — will confidently learn the wrong thing, and the resulting bug is far harder to catch after the fact than it would have been in a five-second look at a chart beforehand. Phase 2's first real classifier depends on the exact category-balance question this lesson's bar chart raises.

## 2. Real-World Analogy

Imagine a doctor given a patient's raw vital-sign readings as a printed list of a thousand numbers — versus the same numbers drawn as a heart-rate line over time. Reading the list, a single dangerous spike buried on line 743 is easy to miss. On the line chart, that spike is the first thing anyone's eye lands on. The numbers are identical in both cases; only the picture makes the important thing impossible to miss. That is the entire job of the plots in this lesson: same data, arranged so the eye does the finding instead of a person scanning digits one at a time.

## 3. The Concept Explained

### Why a picture beats a table, mechanically

A printed table asks a reader to hold many numbers in their head at once and compare them mentally — slow, and reliably error-prone past a handful of rows. A plot instead uses position, height, or spread on a page, which human vision parses almost instantly and in parallel: a bar noticeably taller than its neighbors, a dot sitting apart from a cluster of others, a box shifted higher than the box beside it. The underlying data never changes; a plot is a different way of routing the same numbers through a part of the brain that is much faster at this particular kind of comparison.

### matplotlib and pyplot

**Plain definition:** **matplotlib** is Python's standard plotting library — code that turns numbers into pictures like bars, dots, and lines. Its most common interface is called **pyplot**, always imported under the nickname `plt`.

**Why it matters:** without a plotting library, drawing even a simple bar chart by hand would mean manually computing pixel positions — matplotlib handles the drawing, scaling, and labeling, leaving just the choice of what to plot and how to describe it.

**Tiny concrete example:** `plt.hist(data, bins=5)` builds a histogram in one line; `plt.show()` displays whatever has been built up by the plotting calls before it, the conventional way to reveal a finished plot in a notebook.

**Technical name:** `matplotlib.pyplot`, imported as `plt`.

### The histogram — one column's shape

**Plain definition:** a **histogram** splits a column of numbers into equal-width buckets, called **bins**, and draws a bar for each bucket as tall as the count of values that landed inside it.

**Why it matters:** it shows a column's spread at a glance — clustered together, spread evenly, or with an **outlier**, a value sitting far outside where the rest of the data lives.

**Tiny concrete example:** this lesson's twelve ticket-text lengths, sorted, are `30, 37, 37, 38, 38, 39, 39, 40, 43, 46, 47, 59`. Eleven of the twelve sit between 30 and 47; one, `59`, sits well beyond the rest. A histogram of this column draws a tall cluster of bars in the middle of the range and one short, isolated bar out past a gap — the outlier is visible as empty space on the chart, not just as a number that happens to be bigger.

### The bar chart — comparing category counts

**Plain definition:** a **bar chart** draws one bar per category, as tall as how many rows belong to it.

**Why it matters:** it is the fastest way to catch **class imbalance** — one category vastly outnumbering another in the data. A classifier trained on imbalanced data can learn to mostly guess the biggest category and still score deceptively well by a simple accuracy count, while barely learning the smaller categories at all (the full fix comes in Lesson 21; for now, the point is just learning to *see* the imbalance).

**Tiny concrete example:** `df['category'].value_counts()` on this lesson's 12 tickets gives `billing: 6, product: 3, shipping: 3` — billing alone is half the data. Plotted as a bar chart, one bar stands twice as tall as the other two, an imbalance easy to miss reading three numbers in a row but impossible to miss as two very different bar heights.

### The scatter plot — do two numbers move together?

**Plain definition:** a **scatter plot** draws one dot per row, placed left-to-right by one number and up-down by a second number.

**Why it matters:** it reveals whether two numeric columns tend to rise and fall together, move in opposite directions, or show no relationship at all — the last of which is itself a useful, real finding, not a failed experiment.

**Tiny concrete example:** plotting this lesson's ticket `length` against `satisfaction_rating` gives points including `(59, 2)`, `(47, 5)`, and `(30, 2)` — a long ticket with a low rating, a fairly long ticket with the top rating, and a short ticket with a low rating. The dots scatter with no visible upward or downward drift, which means length and satisfaction do not appear related in this data — worth knowing *before* building anything that assumes otherwise.

### The box plot — comparing spread across groups

**Plain definition:** a **box plot** summarizes one column's spread per group, with a box spanning the middle half of the values, a line marking the **median** (the middle value once sorted), and thin "whisker" lines reaching to the lowest and highest values in that group.

**Why it matters:** it compares several groups' spread side by side in one picture, instead of computing min, max, and median by hand for each group separately.

**Tiny concrete example:** this lesson's ratings, grouped by category: `billing` is `2, 3, 2, 4, 3, 2` (median 2.5, narrow), `product` is `4, 5` (median 4.5, narrow and high), `shipping` is `1, 5, 1` (median 1, but stretching from the very bottom to the very top of the scale). Drawn as three boxes side by side, shipping's whiskers visibly stretch the farthest — flagging it as the least consistent customer experience of the three categories, a pattern easy to miss reading eleven ratings in a row but obvious as three shapes side by side.

## 4. The Code

The notebook builds a 12-row support-ticket table (extending Lesson 06's 6-row version) and draws all four plots above from it, using only pandas and matplotlib.

Open it here: `lesson-notebooks/phase-01-foundations/chapter-02-python-for-ai/lesson-07-seeing-data-with-plots.ipynb`

matplotlib needs to be installed to run it. From a terminal, in whatever Python environment the notebook's kernel uses:

```bash
pip install matplotlib
```

*(Verify this install command still matches current matplotlib documentation if anything looks unfamiliar — package install commands are usually very stable, but it costs nothing to check.)*

1. Import pandas and `matplotlib.pyplot as plt`, create a `plots/` folder next to the notebook with `os.makedirs("plots", exist_ok=True)`, then build a 12-ticket DataFrame with `text`, `category`, and `satisfaction_rating` (one still missing, as in Lesson 06), plus `length` and `exclamations` columns computed with the same vectorized `.str` operations from last lesson.
2. Draw a histogram of `length` with `plt.hist(..., bins=5)`, labelled with a title and axis labels, then save it with `plt.savefig("plots/lesson-07-histogram-length.png")` before `plt.show()`.
3. Count tickets per `category` with `.value_counts()`, plot those counts as a bar chart, and save it the same way.
4. Drop the one missing rating with `.dropna(subset=['satisfaction_rating'])` (Lesson 06's tool, used here because the next two plots need real numbers in both places), scatter-plot `length` against `satisfaction_rating`, and save it.
5. Draw a box plot of `satisfaction_rating` grouped `by='category'`, clearing pandas' default title with `plt.suptitle("")` so only the chosen title remains, and save it.

Each `plt.savefig(...)` call must come *before* `plt.show()` — once a plot is shown, matplotlib clears its internal drawing surface, so a `savefig` call placed after `plt.show()` would save a blank image instead of the chart just drawn.

## 5. If You Ran This

Plots are pictures, so the "expected output" below describes the shape of each picture rather than an exact pixel image — treat every description as a **prediction** of the pattern, not a verified render.

**Step 2 — building the table.** Prints the `ticket_id`, `category`, `satisfaction_rating`, and `length` columns for all 12 rows; row 5 (ticket 5) shows `NaN` for its rating, same as Lesson 06's missing-value row.

**Step 2 (plot) — histogram of `length`.**

**Prediction:** a cluster of tall bars covering roughly 37–47 characters, then a gap, then one short, isolated bar out around 59 characters — the one long ticket sitting apart from the other eleven.

**Step 3 — bar chart of category counts.**

**Prediction (printed counts):**

```
category
billing     6
product     3
shipping    3
Name: count, dtype: int64
```

*(The exact column header text above — `Name: count, dtype: int64` — reflects a recent pandas `.value_counts()` convention; verify this against the installed pandas version if it looks different.)*

**Prediction (chart):** three bars — `billing` roughly twice as tall as `product` and `shipping`, which are roughly equal to each other.

**Step 4 — scatter plot of `length` vs. `satisfaction_rating`.**

**Prediction:** eleven dots (the twelfth, missing-rating row dropped) scattered across the plot with no visible upward or downward trend — long tickets appear at both high and low ratings, and so do short ones.

**Step 5 — box plot of `satisfaction_rating` by `category`.**

**Prediction:** three boxes side by side. `billing`'s box sits low-to-middle and narrow (around 2 to 3). `product`'s box sits high and narrow (around 4 to 5). `shipping`'s whiskers stretch from the bottom of the scale to the top (1 to 5) — visibly the widest spread of the three.

This is a prediction based on tracing through matplotlib and pandas' documented behavior by hand, not a verified run — the learner should open the notebook, run `pip install matplotlib` if needed, and run the cells top to bottom to confirm.

## 6. Applied to Our Assistant

Like Lessons 04, 05, and 06, this lesson is about the tool itself — learning to see a table's shape, balance, relationships, and group differences — not a new capability meant to be implemented in the assistant right now. `ticket_urgency.py` is unchanged; there is no new project `.py` file, no change to `main.py`, and no change to `customer-support-assistant/requirements.txt` this lesson, since today's matplotlib import lives only in this lesson's notebook.

The connection forward is direct: once Phase 2 starts building a real classifier on the assistant's ticket data, the very first step after loading the data will be exactly these four plots — checking category balance with a bar chart before training, exactly as demonstrated here, is standard practice precisely because of the imbalance problem explained in Section 3.

## 7. Common Mistakes and Gotchas

- **Skipping straight to modeling without ever plotting the data.** An outlier, an imbalance, or a data-entry error found *after* training wastes far more time than the minute it takes to draw four basic plots first.
- **Choosing too few or too many histogram bins.** Too few bins hides real structure by lumping everything into one or two bars; too many bins makes noise look like meaningful shape. There is no single correct number — trying a couple of different bin counts is normal practice.
- **Reading a scatter plot with no visible pattern as "the code is broken."** No relationship between two columns is a legitimate, useful finding, not an error — it says something real about the data.
- **Forgetting that `.dropna()` (like `.fillna()`) returns a new table.** The scatter and box plot cells in this lesson deliberately use `df_clean = df.dropna(...)`, reassigning the result — using plain `df.dropna(...)` without reassigning would silently leave the original missing value in place.
- **Not labeling axes and titles.** An unlabeled bar chart or scatter plot forces a reader to guess what the numbers even represent — always set `plt.title(...)`, `plt.xlabel(...)`, and `plt.ylabel(...)`.

## 8. When to Use This, and Tradeoffs

Reach for these four plots as a near-automatic first step on any new dataset, before writing a single line of modeling code — the cost is a few minutes; the payoff is catching problems that are far more expensive to discover later. The tradeoff is mostly about which plot fits which question: a histogram and box plot both describe spread, but a histogram looks at one column alone while a box plot compares that column split across groups; a bar chart counts categories, while a scatter plot relates two numeric columns to each other. Picking the wrong one for the question at hand (for example, a scatter plot to compare category counts) simply won't answer it — matching the plot to the question, as laid out in Section 3, is the actual skill here, more than any specific matplotlib syntax.

## 9. Key Takeaways

- A histogram shows one numeric column's spread and flags outliers as bars sitting apart from the rest.
- A bar chart counts rows per category and is the fastest way to catch class imbalance before it silently damages a later model.
- A scatter plot shows whether two numeric columns move together, move oppositely, or show no relationship at all — and "no relationship" is a real, useful answer.
- A box plot compares one column's spread across several groups at once, using a box for the middle half of the values and whiskers reaching to the extremes.
- Looking at data visually before modeling it catches problems — outliers, imbalance, unexpected relationships — that a printed table of the same numbers reliably hides.

## 10. Challenge

A dataset of 1,000 customer orders has an `order_amount` column and a `region` column (`north`, `south`, `east`, `west`). Which one plot from this lesson would best answer each of these two questions, and why?

1. "Are there any orders with a suspiciously huge amount compared to the rest?"
2. "Do orders from one region tend to be larger than orders from another?"

<details>
<summary>Click to reveal the answer</summary>

**Question 1** calls for a **histogram** of `order_amount`. It shows the whole column's spread in one picture, so a handful of unusually huge orders would show up as isolated bars sitting apart from the main cluster — exactly the outlier-spotting job a histogram is built for.

**Question 2** calls for a **box plot** of `order_amount`, grouped `by='region'`. It draws one box per region side by side, so a region with a noticeably higher or lower box (and its median line) immediately stands out against the others — a direct group-to-group spread comparison, which is what a box plot is for and a plain histogram of the whole column (with regions all mixed together) could not show on its own.

</details>

## 11. What Is Next

Lesson 08 begins Chapter 3 — the math actually needed for AI, in plain language — starting with linear algebra intuition: what a vector and a matrix really are, the dot product, and why almost everything in AI comes down to vectors and matrices.

---

**Lesson 7 of 113 — Phase 1, Chapter 2, Lesson 4 of 4 in this chapter.**
