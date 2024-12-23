import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_scheduler
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm
from custom_dataset import train_dataset, val_dataset


def clean_data(text):
    clean = re.sub(r'<[^>]+>', '', text)
    return clean

def read_data(path):
    data = pd.read_csv(path)
    data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})
    data['review'] = data['review'].apply(clean_data)
    return data

paths = read_data("IMDB Dataset.csv")

def train_test_splits(data):
    train_text, val_text, train_label, val_label = train_test_split(
        data['review'], data['sentiment'], test_size=0.2, random_state=42
    )
    return train_text, val_text, train_label, val_label

train_texts, val_texts, train_labels, val_labels = train_test_splits(paths)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=512)


model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_dataset) // 16 * 4
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

epochs = 1
progress_bar = tqdm(range(epochs * len(train_loader)))

for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        batch = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

model.save_pretrained('sentiment_model')
tokenizer.save_pretrained('sentiment_model')

