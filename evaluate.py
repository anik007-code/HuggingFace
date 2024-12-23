from main import model, val_loader, device
import torch
from sklearn.metrics import accuracy_score, classification_report

model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        batch = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())

accuracy = accuracy_score(all_labels, all_predictions)
print(f"Validation Accuracy: {accuracy}")
print(classification_report(all_labels, all_predictions))
