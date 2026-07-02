from collections import Counter

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


def tokenize(text):
    return text.lower().replace(",", "").replace(".", "").split()


def _build_word_counts(training_tickets):
    urgent_counts = Counter()
    not_urgent_counts = Counter()
    for text, label in training_tickets:
        words = tokenize(text)
        if label == "urgent":
            urgent_counts.update(words)
        else:
            not_urgent_counts.update(words)
    return urgent_counts, not_urgent_counts


URGENT_WORD_COUNTS, NOT_URGENT_WORD_COUNTS = _build_word_counts(TRAINING_TICKETS)


def urgency_score(ticket_text):
    words = tokenize(ticket_text)
    score = 0
    for word in words:
        score += URGENT_WORD_COUNTS.get(word, 0)
        score -= NOT_URGENT_WORD_COUNTS.get(word, 0)
    return score


def predict_urgency(ticket_text):
    return "urgent" if urgency_score(ticket_text) > 0 else "not urgent"


if __name__ == "__main__":
    demo_tickets = [
        "My payment failed twice and I need this resolved right now",
        "Do you have this item in a larger size",
    ]
    for ticket in demo_tickets:
        print(f"{ticket} -> {predict_urgency(ticket)} (score: {urgency_score(ticket)})")
