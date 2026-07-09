"""Estimate how long a support ticket will take to resolve, in minutes.

This is the assistant's first genuinely *learned* model. Unlike the urgency
word-counter (whose rule a human wrote), here linear regression discovers the
relationship between a ticket's length and its resolution time on its own, by
gradient descent. The trained model is just two numbers: a slope and an
intercept.
"""

# Past tickets we already handled: (length in words, minutes it took to resolve).
TRAINING_EXAMPLES = [
    (10, 11),
    (20, 14),
    (30, 22),
    (40, 24),
    (50, 31),
    (60, 33),
    (70, 42),
    (80, 44),
]

# Raw lengths are large, which makes gradient descent unstable, so we train on
# "tens of words" (length / 10). The slope is therefore minutes per ten words.
FEATURE_SCALE = 10.0


def scale_length(length_words):
    return length_words / FEATURE_SCALE


def predict_with(weight, bias, scaled_length):
    return weight * scaled_length + bias


def mean_squared_error(weight, bias, examples):
    total_squared_error = 0.0
    for length_words, actual_minutes in examples:
        scaled_length = scale_length(length_words)
        predicted = predict_with(weight, bias, scaled_length)
        error = predicted - actual_minutes
        total_squared_error += error * error
    return total_squared_error / len(examples)


def train(examples, learning_rate=0.01, iterations=2000):
    weight = 0.0
    bias = 0.0
    n = len(examples)
    for _ in range(iterations):
        weight_gradient = 0.0
        bias_gradient = 0.0
        for length_words, actual_minutes in examples:
            scaled_length = scale_length(length_words)
            predicted = predict_with(weight, bias, scaled_length)
            error = predicted - actual_minutes
            weight_gradient += error * scaled_length
            bias_gradient += error
        weight_gradient = (2.0 / n) * weight_gradient
        bias_gradient = (2.0 / n) * bias_gradient
        weight -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient
    return weight, bias


WEIGHT, BIAS = train(TRAINING_EXAMPLES)


def predict_resolution_minutes(length_words):
    scaled_length = scale_length(length_words)
    minutes = predict_with(WEIGHT, BIAS, scaled_length)
    return round(minutes, 1)


if __name__ == "__main__":
    print(f"Learned line: minutes = {WEIGHT:.2f} * (length/10) + {BIAS:.2f}")
    for length in [15, 45, 75]:
        print(f"  {length:>3}-word ticket -> ~{predict_resolution_minutes(length)} min")

    # A longer ticket must be predicted to take longer than a shorter one.
    assert predict_resolution_minutes(80) > predict_resolution_minutes(20)
    # A mid-length ticket should land in a sensible range.
    assert 25 <= predict_resolution_minutes(55) <= 40
    print("Self-check passed.")
