from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load and preprocess dataset
dataset = load_dataset("yelp_review_full")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
def tokenize(batch): return tokenizer(batch["text"], truncation=True, padding=True)
dataset = dataset.map(tokenize, batched=True)

# Prepare model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)

# Train
args = TrainingArguments("outputs", evaluation_strategy="epoch", per_device_train_batch_size=16)
trainer = Trainer(model=model, args=args, train_dataset=dataset["train"].shuffle(seed=42).select(range(2000)),
                  eval_dataset=dataset["test"].shuffle(seed=42).select(range(500)))
trainer.train()

# Save
model.save_pretrained("model/")
tokenizer.save_pretrained("nlp/tokenizer/")
