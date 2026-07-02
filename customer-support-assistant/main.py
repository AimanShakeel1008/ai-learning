from phase_01_foundations.ticket_urgency import predict_urgency, urgency_score

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


if __name__ == "__main__":
    main()
