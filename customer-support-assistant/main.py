from ticket_urgency import predict_urgency, urgency_score
from resolution_time import predict_resolution_minutes

SAMPLE_TICKETS = [
    "My payment failed twice and I need this resolved right now",
    "Do you have this item in a larger size",
]


def main():
    print("Customer Support Assistant - current capabilities")
    print()
    print("1) Ticket urgency detection")
    for ticket in SAMPLE_TICKETS:
        print(f"   [{predict_urgency(ticket)}] (score {urgency_score(ticket)}) {ticket}")

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
    print("Self-check passed.")


if __name__ == "__main__":
    main()
