from ticket_urgency import predict_urgency, urgency_score

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
    print("Self-check passed.")


if __name__ == "__main__":
    main()
