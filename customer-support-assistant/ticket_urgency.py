"""Decide whether a support ticket is urgent, and how confident we are.

This started (Lesson 01) as a hand-written rule: count "urgent" vs "calm" words
and call it urgent if the score was above zero. Lesson 15 replaces that rule with
a genuinely *learned* classifier: logistic regression. We reduce each ticket to
one number (its count of urgency-signal words), then let gradient descent learn
how strongly that count implies urgency and where the yes/no boundary belongs.
The model outputs a calibrated probability, not just a flat label.
"""

import math

# Past tickets a human already labelled. This is what the model learns from.
TRAINING_TICKETS = [
    ("My order still has not arrived and I need it right now", "urgent"),
    ("This is extremely time sensitive please help right now", "urgent"),
    ("The app crashed and I cannot log in please fix this immediately", "urgent"),
    ("My package was supposed to arrive yesterday and now it says lost", "urgent"),
    ("Just wondering when my package will ship no rush", "not urgent"),
    ("Can you tell me more about your return policy sometime", "not urgent"),
    ("I have a general question about sizing whenever you get a chance", "not urgent"),
    ("Do you offer gift wrapping for orders", "not urgent"),
]

# Words that signal a ticket might be urgent. Counting these is our one feature.
# (Extracting a number from text is feature engineering, covered in Chapter 6;
# the model still has to LEARN how much that number matters and where to cut.)
SIGNAL_WORDS = {
    "now", "immediately", "right", "today", "asap", "urgent",
    "cannot", "help", "fix", "lost", "emergency", "crashed",
}


def tokenize(text):
    lowered = text.lower().replace(",", "").replace(".", "")
    return lowered.split()


def count_signal_words(text):
    words = tokenize(text)
    count = 0
    for word in words:
        if word in SIGNAL_WORDS:
            count += 1
    return count


def sigmoid(z):
    if z < 0:
        return math.exp(z) / (1.0 + math.exp(z))
    return 1.0 / (1.0 + math.exp(-z))


def _build_training_pairs(tickets):
    signal_counts = []
    labels = []
    for text, label in tickets:
        signal_counts.append(count_signal_words(text))
        if label == "urgent":
            labels.append(1)
        else:
            labels.append(0)
    return signal_counts, labels


def _train(signal_counts, labels, learning_rate=0.3, iterations=3000):
    weight = 0.0
    bias = 0.0
    n = len(signal_counts)
    for _ in range(iterations):
        weight_gradient = 0.0
        bias_gradient = 0.0
        for i in range(n):
            x = signal_counts[i]
            actual = labels[i]
            predicted = sigmoid(weight * x + bias)
            error = predicted - actual
            weight_gradient += error * x
            bias_gradient += error
        weight -= learning_rate * (weight_gradient / n)
        bias -= learning_rate * (bias_gradient / n)
    return weight, bias


_SIGNAL_COUNTS, _LABELS = _build_training_pairs(TRAINING_TICKETS)
WEIGHT, BIAS = _train(_SIGNAL_COUNTS, _LABELS)


def urgency_probability(ticket_text):
    x = count_signal_words(ticket_text)
    return sigmoid(WEIGHT * x + BIAS)


def predict_urgency(ticket_text):
    if urgency_probability(ticket_text) >= 0.5:
        return "urgent"
    return "not urgent"


if __name__ == "__main__":
    boundary = -BIAS / WEIGHT
    print(f"Learned model: P(urgent) = sigmoid({WEIGHT:.2f} * signal_words + {BIAS:.2f})")
    print(f"Decision boundary at {boundary:.2f} signal words")
    print()

    demo_tickets = [
        "My payment failed twice and I need this resolved right now",
        "Do you have this item in a larger size",
    ]
    for ticket in demo_tickets:
        probability = urgency_probability(ticket)
        print(f"  P(urgent)={probability:.2f}  [{predict_urgency(ticket)}]  {ticket}")

    # The learned classifier must still get the two clear cases right...
    assert predict_urgency("My payment failed twice and I need this resolved right now") == "urgent"
    assert predict_urgency("Do you have this item in a larger size") == "not urgent"
    # ...output must be a real probability...
    assert 0.0 <= urgency_probability("Do you have this item in a larger size") <= 1.0
    # ...and a signal-heavy ticket must be judged more urgent than a calm one.
    assert urgency_probability("I need this fixed right now") > urgency_probability("just a general question")
    print("Self-check passed.")
