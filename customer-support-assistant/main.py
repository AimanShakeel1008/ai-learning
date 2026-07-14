from ticket_urgency import predict_urgency, urgency_probability
from resolution_time import predict_resolution_minutes
from ticket_category import predict_category

SAMPLE_TICKETS = [
    "My payment failed twice and I need this resolved right now",
    "Do you have this item in a larger size",
]

CATEGORY_TICKETS = [
    "Where is my package it has been two weeks",
    "I was charged twice and want a refund",
    "I cannot log in please reset my password",
]


def main():
    print("Customer Support Assistant - current capabilities")
    print()
    print("1) Ticket urgency detection (learned by logistic regression)")
    for ticket in SAMPLE_TICKETS:
        probability = urgency_probability(ticket)
        print(f"   [{predict_urgency(ticket)}] (P(urgent)={probability:.2f}) {ticket}")

    assert predict_urgency(SAMPLE_TICKETS[0]) == "urgent"
    assert predict_urgency(SAMPLE_TICKETS[1]) == "not urgent"

    print()
    print("2) Estimated resolution time (learned by linear regression)")
    for ticket in SAMPLE_TICKETS:
        length_words = len(ticket.split())
        estimate = predict_resolution_minutes(length_words)
        print(f"   ~{estimate} min ({length_words} words) {ticket}")

    # A longer ticket must be estimated to take at least as long as a shorter one.
    short_ticket_words = len(SAMPLE_TICKETS[1].split())
    long_ticket_words = len(SAMPLE_TICKETS[0].split())
    assert predict_resolution_minutes(long_ticket_words) >= predict_resolution_minutes(short_ticket_words)

    print()
    print("3) Ticket category routing (learned by a decision tree)")
    for ticket in CATEGORY_TICKETS:
        print(f"   [{predict_category(ticket)}] {ticket}")

    assert predict_category(CATEGORY_TICKETS[0]) == "shipping"
    assert predict_category(CATEGORY_TICKETS[1]) == "billing"
    assert predict_category(CATEGORY_TICKETS[2]) == "account"

    print()
    print("Self-check passed.")


if __name__ == "__main__":
    main()
