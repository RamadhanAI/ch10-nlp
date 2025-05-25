
# Chapter 10 – NLP in Production (AI/ML Book)

This project demonstrates how to fine-tune DistilBERT on a sentiment classification task (Yelp reviews) and deploy it via FastAPI.

## 📦 Features

- Tokenization and training using Hugging Face Transformers
- Model saved locally to `model/` and `tokenizer/`
- FastAPI REST endpoint for inference
- Ready for Docker and CI/CD integration

## 🛠️ Training

```bash
python scripts/train_chat_model.py
🚀 Serve the Model

uvicorn scripts.predict_api:app --reload
Access it at: http://127.0.0.1:8000/docs

🔧 Project Structure

ch10-nlp/
├── scripts/
│   ├── train_chat_model.py      # DistilBERT fine-tuning
│   └── predict_api.py           # FastAPI inference service
├── tokenizer/                   # Saved tokenizer
├── model/                       # Saved model
├── requirements.txt             # Dependencies
└── README.md
🧪 Example Request

POST /predict
{
  "text": "The food was amazing!"
}
