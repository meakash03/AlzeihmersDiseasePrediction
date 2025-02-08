import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# Load the saved model and tokenizer
model_load_path = './model'
model = RobertaForSequenceClassification.from_pretrained(model_load_path)
tokenizer = RobertaTokenizer.from_pretrained(model_load_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
# Function to load DNA sequences from FASTA files (same as before)
# Function to load DNA sequences from FASTA files

# ... (your code here)

# Function to predict on a new FASTA file
def prediction(sequence):
    predictions = []
    for seq in sequence:
        encoded_sequence = tokenizer(seq, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
        input_ids = encoded_sequence["input_ids"].to(device)
        attention_mask = encoded_sequence["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_label = torch.argmax(logits, dim=1).item()
            predictions.append(predicted_label)
    # Map predicted class to label (e.g., 1 for Alzheimer's, 0 for healthy)
    predicted_label = "Alzheimer's affected" if predictions[0] == 1 else "Alzheimer's unaffected"
    return predicted_label


# # Example usage
# new_fasta_file = "PSEN1_affected_0.fasta"
# prediction = predict_alzheimers(new_fasta_file)
# print("Predicted label:", prediction)

