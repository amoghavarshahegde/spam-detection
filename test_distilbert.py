
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load our trained DistilBERT model
MODEL_PATH = "models/distilbert_spam"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()


def predict_email(email):

    inputs = tokenizer(
        email,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)

    prediction = torch.argmax(probabilities, dim=1).item()

    confidence = probabilities[0][prediction].item() * 100

    if prediction == 1:
        result = "SPAM"
    else:
        result = "NOT SPAM"

    print("Email:", email)
    print("Prediction:", result)
    print("Confidence:", round(confidence, 2), "%")


# Test emails
predict_email(
    "Congratulations! You won a free iPhone. Click here to claim now!"
)

print("\n----------------------\n")

predict_email(
    "Hello, your college examination timetable has been released."
)
