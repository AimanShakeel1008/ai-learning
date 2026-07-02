# Lesson 06 — pandas for Handling Data

**Phase 1 — Foundations, Chapter 2 — Python for AI, Lesson 3**

**Files created:**
- `lessons/phase-01-foundations/chapter-02-python-for-ai/lesson-06-pandas-basics.md` (this file)
- `lesson-notebooks/phase-01-foundations/chapter-02-python-for-ai/lesson-06-pandas-basics.ipynb`

**Prerequisite lessons:** Lesson 05 — NumPy and thinking in arrays.

---

## 1. Why This Matters

Lesson 05 built the array mental model: numbers laid out together, operated on all at once. Real data, though, is rarely just numbers — it is a table: rows of examples, columns of named properties, mixed types, and gaps where something wasn't recorded. **pandas** is the library built for exactly that table shape, on top of the same array ideas from last lesson.

This is not a side skill. Every AI project in this course, starting with Phase 2's first real classifier, begins the same way: load the data into a table, look at it, select the pieces that matter, filter out what doesn't belong, and patch up the gaps — all *before* any model gets touched. Experienced practitioners often say most of the real work in a machine learning project is data work, not model work, and this lesson is the first hands-on look at why that is true.

## 2. Real-World Analogy

Picture a filing cabinet versus a spreadsheet. A filing cabinet (a plain Python list) holds folders in order, but finding "every folder from March" means pulling out and checking each folder by hand, one at a time. A spreadsheet (a pandas DataFrame) has named columns — a "Date" column, a "Category" column — so finding every March row is a matter of asking the whole sheet a question at once ("show me rows where Date starts with March") instead of checking folders one by one. pandas is that spreadsheet, built for Python, with the speed of Lesson 05's arrays running underneath it.

## 3. The Concept Explained

### The DataFrame: pandas' table

**Plain definition:** a **DataFrame** is a table of rows and columns, like a spreadsheet, where every column can hold its own type of data.

**Why it matters:** a NumPy array (Lesson 05) requires every single element to be the same type — great for pure numbers, useless for a support ticket that has a numeric id, a text message, and a text category all in one row. A DataFrame lifts that restriction column by column: one column of whole numbers, one of text, one of decimals, all living in the same table, each column internally still fast like a NumPy array.

**Tiny concrete example:** this lesson's notebook builds a DataFrame of 6 support tickets from a Python dictionary — `{"ticket_id": [1, 2, ...], "text": [...], "category": [...], "satisfaction_rating": [...]}` — where each dictionary key becomes a column name and each value is the list of that column's entries, one per row.

**The technical name:** the whole table is a **DataFrame**; one column (or one row) pulled out on its own is called a **Series** — a labeled 1D array, carrying its row labels along with it.

### The index: row labels

**Plain definition:** the **index** is the label attached to each row, shown on the left whenever a DataFrame prints.

**Why it matters:** a plain Python list only ever knows a row by its position (item 0, item 1, ...). pandas gives every row an actual label, separate from its position, which stays attached to that row even after the table is filtered, sorted, or rearranged — this is exactly what makes `.loc` (label-based lookup) and `.iloc` (position-based lookup) two genuinely different tools, covered below.

**Tiny concrete example:** build a DataFrame without specifying anything about row labels, and pandas assigns `0, 1, 2, 3, 4, 5` automatically — a **RangeIndex**. Later, an index can be set to something meaningful, like `ticket_id`, so a row can be looked up by its actual id instead of counting rows from the top.

### Selecting: one bracket versus two

**Plain definition:** selecting means pulling out only the columns (or rows) needed, out of a larger table.

**Why it matters:** real tables often have far more columns than any one task needs. Selecting just the relevant ones keeps code readable and avoids dragging unrelated data through calculations that don't need it.

**Tiny concrete example:** `df['category']` (single brackets, one name) returns a Series — one column. `df[['text', 'category']]` (double brackets, a list of names) returns a smaller DataFrame — several columns. The outer bracket always means "select from `df`"; whether a plain name or a list goes inside decides whether the answer comes back as a Series or a DataFrame.

For rows, `.iloc[i]` selects by position (like list indexing — the "3rd row," whatever its label happens to be), and `.loc[label]` selects by the row's actual index label (the row *labeled* `3`, wherever it now sits). With the default index these give the same row; they diverge the moment a table's rows have been filtered or its index has been set to something else.

### Filtering with a boolean mask, again

**Plain definition:** filtering a DataFrame keeps only the rows that match some condition.

**Why it matters:** this is the single most common thing done to real data — "show me only the shipping tickets," "show me only the urgent ones." pandas reuses the exact boolean-mask pattern from Lesson 05: build a `True`/`False` answer for every row with a comparison, then use that answer to keep only the matching rows.

**Tiny concrete example:** `df['category'] == 'shipping'` compares every row's category to `'shipping'` at once, returning a Series of `True`/`False`, one per row. `df[mask]` then keeps only the rows marked `True`. No loop, no `if` statement, written by hand.

### Missing values: NaN, isna, fillna, dropna

**Plain definition:** a **missing value** is a gap in the data — something that was never recorded for that row. pandas represents a missing *number* with the special value **NaN** ("Not a Number").

**Why it matters:** real data almost always has gaps — a skipped survey question, a sensor that failed to report, a form field left blank. A model cannot do arithmetic on "nothing," so every gap has to be found and handled — either filled in with a reasonable stand-in value, or the row removed entirely — before the data is usable.

**Tiny concrete example:** this lesson's tickets include one satisfaction rating written as `None` because that customer never rated their ticket. Because a numeric column can't hold "missing" as a whole number, pandas silently upgrades the *entire* `satisfaction_rating` column to decimals (`float64`) to make room for `NaN` — this is why `2` prints as `2.0`. `.isna()` finds the gap (another boolean mask); `.fillna(df['satisfaction_rating'].mean())` fills it with the average of the other ratings (reusing Lesson 05's `.mean()`, rather than guessing a number that was never actually given); `.dropna()` removes the row with the gap entirely. Both `.fillna(...)` and `.dropna()` return a **new** table — the original `df` is untouched unless the result is reassigned back to it.

### Vectorized string operations

**Plain definition:** a **vectorized string operation** applies one text operation to an entire text column at once, using pandas' `.str` accessor.

**Why it matters:** exactly the same speed and no-loop argument from Lesson 05, now for text instead of numbers — a real dataset might have a text column with a million rows, and computing something like "how long is this message" one row at a time in a Python loop would be far slower than letting pandas do it internally.

**Tiny concrete example:** `df['text'].str.len()` computes the character length of every ticket's text in a single call; `df['text'].str.count('!')` counts `!` characters in every ticket the same way. Assigning either to `df['length'] = ...` adds it as a brand-new column.

## 4. The Code

The notebook for this lesson builds a 6-row DataFrame of support tickets, then walks through looking at it, selecting from it, filtering it, cleaning a missing value, and adding a new column — using only pandas (which itself depends on NumPy).

Open it here: `lesson-notebooks/phase-01-foundations/chapter-02-python-for-ai/lesson-06-pandas-basics.ipynb`

To run it, pandas needs to be installed first (NumPy comes along with it automatically, since pandas depends on it). From a terminal, in whatever Python environment the notebook's kernel uses:

```bash
pip install pandas
```

*(Verify this install command still matches current pandas documentation if anything looks unfamiliar — package install commands are usually very stable, but it costs nothing to check.)*

1. Import pandas under its standard nickname, `pd`.
2. Build a 6-row DataFrame from a dictionary of ticket id, text, category, and satisfaction rating (one rating is `None`, since that customer never responded).
3. Look at the table with `.head(3)`, `.shape`, and `.info()`.
4. Select the `category` column alone (a Series), then `text` and `category` together (a DataFrame), noting the single-bracket-versus-double-bracket difference.
5. Select the first row two ways: `.iloc[0]` (by position) and `.loc[0]` (by label).
6. Build a boolean mask for `category == 'shipping'` and use it to filter the table down to just the shipping tickets.
7. Find the missing rating with `.isna()`, fill it with the column's mean using `.fillna(...)`, and drop the row with `.dropna()` — noting none of this changes `df` itself unless reassigned.
8. Add two new columns, `length` and `exclamations`, computed with vectorized `.str.len()` and `.str.count('!')`.

## 5. If You Ran This

Walking through the notebook top to bottom. Real pandas output includes precise column-width spacing that is easy to get slightly wrong by hand — treat the layout below as a close approximation of the real thing, not a pixel-exact copy, and expect small spacing differences when actually run.

**Step 2 — building the table.**

**Prediction:**

```
   ticket_id                                                text  category  satisfaction_rating
0          1  My payment failed twice and I need this resol...   billing                   2.0
1          2           Do you have this item in a larger size?   product                   4.0
2          3        Where is my order, it has not arrived yet!!  shipping                   1.0
3          4            Thanks for the quick delivery, love it!  shipping                   5.0
4          5    The app keeps crashing when I try to check out   product                   NaN
5          6             Can I get a refund for a damaged item?   billing                   3.0
```

(Note row 0's `text` is cut short with `...` — pandas truncates long text columns in *display only* after a default width, commonly 50 characters; the underlying data is not affected. This default is a display setting, `pd.options.display.max_colwidth`, and is worth confirming against the installed pandas version if the exact cutoff ever matters.)

**Step 3 — head, shape, info.**

**Prediction:**

```
   ticket_id                                                text  category  satisfaction_rating
0          1  My payment failed twice and I need this resol...   billing                   2.0
1          2           Do you have this item in a larger size?   product                   4.0
2          3        Where is my order, it has not arrived yet!!  shipping                   1.0
(6, 4)
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6 entries, 0 to 5
Data columns (total 4 columns):
 #   Column               Non-Null Count  Dtype  
---  ------               --------------  -----  
 0   ticket_id            6 non-null      int64  
 1   text                 6 non-null      object 
 2   category             6 non-null      object 
 3   satisfaction_rating  5 non-null      float64
dtypes: float64(1), int64(1), object(2)
memory usage: 320.0+ bytes
```

(The exact `memory usage` figure depends on the environment and pandas version — treat that one line as illustrative, not exact.)

**Step 4 — selecting columns.**

**Prediction:**

```
0     billing
1     product
2    shipping
3    shipping
4     product
5     billing
Name: category, dtype: object
<class 'pandas.core.series.Series'>
                                                text  category
0  My payment failed twice and I need this resol...   billing
1           Do you have this item in a larger size?   product
2        Where is my order, it has not arrived yet!!  shipping
3            Thanks for the quick delivery, love it!  shipping
4    The app keeps crashing when I try to check out   product
5             Can I get a refund for a damaged item?   billing
<class 'pandas.core.frame.DataFrame'>
```

**Step 5 — `.iloc[0]` versus `.loc[0]`.** Both select the same row here, since the index is still the default `0..5`.

**Prediction (printed twice, identically):**

```
ticket_id                                                               1
text                    My payment failed twice and I need this resol...
category                                                          billing
satisfaction_rating                                                  2.0
Name: 0, dtype: object
```

**Step 6 — filtering.**

**Prediction:**

```
0    False
1    False
2     True
3     True
4    False
5    False
Name: category, dtype: bool
   ticket_id                                          text  category  satisfaction_rating
2          3  Where is my order, it has not arrived yet!!  shipping                   1.0
3          4      Thanks for the quick delivery, love it!  shipping                   5.0
```

**Step 7 — missing values.**

**Prediction:**

```
0    2.0
1    4.0
2    1.0
3    5.0
4    NaN
5    3.0
Name: satisfaction_rating, dtype: float64
0    False
1    False
2    False
3    False
4     True
5    False
Name: satisfaction_rating, dtype: bool
0    2.0
1    4.0
2    1.0
3    5.0
4    3.0
5    3.0
Name: satisfaction_rating, dtype: float64
   ticket_id                                                text  category  satisfaction_rating
0          1  My payment failed twice and I need this resol...   billing                   2.0
1          2           Do you have this item in a larger size?   product                   4.0
2          3        Where is my order, it has not arrived yet!!  shipping                   1.0
3          4            Thanks for the quick delivery, love it!  shipping                   5.0
5          6             Can I get a refund for a damaged item?   billing                   3.0
(6, 4)
```

(Row 4, the one with the missing rating, is gone from the `.dropna()` output — but `df.shape` still prints `(6, 4)` afterward, because `.dropna()` returned a new table without changing `df` itself.)

**Step 8 — new columns.**

**Prediction:**

```
                                                text  length  exclamations
0  My payment failed twice and I need this resol...      59             1
1           Do you have this item in a larger size?      39             0
2        Where is my order, it has not arrived yet!!      43             2
3            Thanks for the quick delivery, love it!      39             1
4    The app keeps crashing when I try to check out      46             0
5             Can I get a refund for a damaged item?      38             0
(6, 6)
```

This is a prediction based on tracing through pandas' documented behavior by hand, not a verified run — the learner should open the notebook, run `pip install pandas` if needed, and run the cells top to bottom to confirm.

## 6. Applied to Our Assistant

This lesson, like Lessons 04 and 05, is about the tool itself — the DataFrame mental model — not a new capability meant to be implemented in the assistant right now. `ticket_urgency.py` still stays exactly as it is, a plain-Python word-counting classifier, until Phase 2 rebuilds it on real machine learning. So there is no new project `.py` file, no change to `main.py`, and no change to `customer-support-assistant/requirements.txt` this lesson — that file only gains a line once the *project's own code* imports something new, and today's pandas import lives only in this lesson's notebook.

The connection forward is direct: once Phase 2 starts training a real classifier on many tickets at once, the training data will be loaded and cleaned as a pandas DataFrame exactly like the one built today — `text`, `category`, and derived numeric columns like `length` and `exclamations` are the first hint of what a real feature table for the assistant will look like.

## 7. Common Mistakes and Gotchas

- **Expecting `.fillna(...)` or `.dropna()` to change the table in place.** They return a new table by default; without reassigning (`df = df.dropna()`) or passing `inplace=True`, the original `df` is untouched — a very common source of "why didn't my missing values go away" confusion.
- **Confusing single brackets and double brackets.** `df['col']` is a Series; `df[['col']]` (even with just one name) is a DataFrame. They can look deceptively similar but behave differently in later operations.
- **Assuming `.loc` and `.iloc` are interchangeable.** They agree only when the index happens to match row position, which is true for a fresh default index and stops being true the moment rows are filtered, sorted, or given a custom index.
- **Reading `NaN` as the same thing as the text `"None"`.** `NaN` is a special *numeric* placeholder pandas uses internally for missing numbers; it only appears this way in numeric columns, and is not the literal string `"None"` even though `None` is often what gets typed in to create it.
- **Trusting a printed table's truncated text as the full data.** pandas' default display width cuts long strings off with `...` when printing — the underlying value is untouched; only the on-screen preview is shortened.

## 8. When to Use This, and Tradeoffs

Reach for pandas the moment data has more than one property per example — a support ticket with text, a category, and a rating is already a table, not a flat list. For a single column of plain numbers with nothing else attached, a NumPy array (Lesson 05) is lighter and faster. The tradeoff: pandas adds real overhead and a large API surface compared to a bare NumPy array, but that cost buys labels, mixed types, and the selecting/filtering/cleaning tools this lesson covered — tools that become close to mandatory the moment real, messy data enters the picture, which is true for almost every AI project past the very first toy example.

## 9. Key Takeaways

- A pandas DataFrame is a labeled table of rows and columns where each column can hold its own type, built on top of NumPy arrays underneath.
- Selecting a single column with `df['col']` returns a Series; selecting a list of columns with `df[['col1', 'col2']]` returns a DataFrame — the number of brackets controls which one comes back.
- `.loc` selects rows by their index label and `.iloc` selects rows by position — they agree only when the index still matches row position.
- Filtering a DataFrame reuses the exact boolean-mask pattern from NumPy: build a `True`/`False` Series with a comparison, then index with it to keep only matching rows.
- Missing numeric values show up as `NaN`; `.isna()` finds them, `.fillna(...)` replaces them, and `.dropna()` removes their rows — none of which change the original table unless the result is reassigned.

## 10. Challenge

Given a DataFrame `orders` with columns `order_id`, `amount`, and `is_returned` (`True`/`False`), write, in plain words, the two-step process to get a DataFrame containing only the orders that were **not** returned and had an `amount` greater than `50`. (Hint: a boolean mask can combine two conditions with `&` for "and.")

<details>
<summary>Click to reveal the answer</summary>

Step one: build a combined boolean mask comparing both conditions at once — `mask = (orders['amount'] > 50) & (orders['is_returned'] == False)`. Each condition alone produces its own `True`/`False` Series; wrapping each side in parentheses and combining them with `&` (pandas' vectorized "and," applied element by element, not Python's plain `and` keyword) produces one final mask that is `True` only where *both* conditions hold for that row.

Step two: use that mask to filter — `orders[mask]` — which keeps only the rows where the combined mask is `True`, i.e. orders over 50 that were not returned. The parentheses around each condition matter: without them, Python tries to evaluate `&` before the comparisons, which raises an error.

</details>

## 11. What Is Next

Lesson 07 covers seeing data with plots — why looking at data visually, before modeling it, catches problems numbers alone can hide, and the handful of plot types that matter most.

---

**Lesson 6 of 113 — Phase 1, Chapter 2, Lesson 3 of 4 in this chapter.**
