"""Route a support ticket to the team that should handle it.

This is the ticket-category classifier first promised in Lesson 11, built in
Lesson 16 as a decision tree learned from scratch. The tree is given a menu of
yes/no questions ("does the ticket contain the word 'refund'?") and greedily
picks, at every step, the question that best un-mixes the categories (lowest
Gini impurity). The learned model is just a nested set of questions that can
be printed and read like a flowchart.
"""

# Past tickets a human already labelled with the team that handled them.
TRAINING_TICKETS = [
    ("Where is my package it has not arrived", "shipping"),
    ("My order says delivered but no package came", "shipping"),
    ("The delivery is late and tracking has not updated", "shipping"),
    ("My package arrived damaged and crushed", "shipping"),
    ("I was charged twice for my order please refund me", "billing"),
    ("I want a refund for this purchase", "billing"),
    ("Why was I charged extra fees I want a refund", "billing"),
    ("Requesting a refund because the discount was not applied", "billing"),
    ("My login is not working after the update", "account"),
    ("I forgot my password and the reset email never comes", "account"),
    ("Please help me change the email on my account", "account"),
    ("My account is locked and I need to reset my password", "account"),
]

# The menu of questions the tree may ask: "does the ticket contain this word?"
CANDIDATE_WORDS = [
    "refund", "charged", "package", "delivery", "order",
    "password", "account", "login", "help",
]

# Depth limit: how many questions deep the tree may grow before it must stop.
MAX_DEPTH = 3


def tokenize(text):
    lowered = text.lower().replace(",", "").replace(".", "").replace("!", "").replace("?", "")
    return lowered.split()


def has_word(text, word):
    return word in tokenize(text)


def gini(labels):
    total = len(labels)
    counts = {}
    for label in labels:
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    impurity = 1.0
    for label in counts:
        share = counts[label] / total
        impurity -= share * share
    return impurity


def split_quality(tickets, word):
    yes_labels = []
    no_labels = []
    for text, label in tickets:
        if has_word(text, word):
            yes_labels.append(label)
        else:
            no_labels.append(label)
    # A question that puts every ticket on one side tells us nothing.
    if len(yes_labels) == 0 or len(no_labels) == 0:
        return None
    total = len(tickets)
    yes_weight = len(yes_labels) / total
    no_weight = len(no_labels) / total
    return yes_weight * gini(yes_labels) + no_weight * gini(no_labels)


def best_word(tickets):
    best = None
    best_score = None
    for word in CANDIDATE_WORDS:
        score = split_quality(tickets, word)
        if score is None:
            continue
        if best_score is None or score < best_score:
            best = word
            best_score = score
    return best


def majority_label(tickets):
    counts = {}
    for text, label in tickets:
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    winner = None
    winner_count = 0
    for label in counts:
        if counts[label] > winner_count:
            winner = label
            winner_count = counts[label]
    return winner


def build_tree(tickets, depth=0):
    labels = []
    for text, label in tickets:
        labels.append(label)
    if gini(labels) == 0.0 or depth == MAX_DEPTH:
        return {"label": majority_label(tickets)}
    word = best_word(tickets)
    if word is None:
        return {"label": majority_label(tickets)}
    yes_tickets = []
    no_tickets = []
    for text, label in tickets:
        if has_word(text, word):
            yes_tickets.append((text, label))
        else:
            no_tickets.append((text, label))
    return {
        "word": word,
        "yes": build_tree(yes_tickets, depth + 1),
        "no": build_tree(no_tickets, depth + 1),
    }


TREE = build_tree(TRAINING_TICKETS)


def predict_category(ticket_text):
    node = TREE
    while "label" not in node:
        if has_word(ticket_text, node["word"]):
            node = node["yes"]
        else:
            node = node["no"]
    return node["label"]


def format_tree(node, indent=""):
    if "label" in node:
        return "predict " + node["label"]
    yes_part = format_tree(node["yes"], indent + "  ")
    no_part = format_tree(node["no"], indent + "  ")
    return ('contains "' + node["word"] + '"?'
            + "\n" + indent + "  yes: " + yes_part
            + "\n" + indent + "  no:  " + no_part)


if __name__ == "__main__":
    print("Learned decision tree:")
    print(format_tree(TREE))
    print()

    demo_tickets = [
        "Where is my package it has been two weeks",
        "I was charged twice and want a refund",
        "I cannot log in please reset my password",
    ]
    for ticket in demo_tickets:
        print(f"  [{predict_category(ticket)}] {ticket}")

    # One clear ticket from each category must be routed to the right team...
    assert predict_category("Where is my package it has been two weeks") == "shipping"
    assert predict_category("I was charged twice and want a refund") == "billing"
    assert predict_category("I cannot log in please reset my password") == "account"
    # ...and every training ticket must still be classified correctly.
    correct = 0
    for text, label in TRAINING_TICKETS:
        if predict_category(text) == label:
            correct += 1
    assert correct == len(TRAINING_TICKETS)
    print("Self-check passed.")
