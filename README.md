Chapter 10: NLP in Production — Chat Ticket Classification Pipeline

Welcome to the NLP production pipeline for classifying customer support chat tickets into thematic categories such as Billing, Technical Issues, and Password Resets. This repository contains all code, models, tokenizers, and utilities to train, deploy, and serve a real-world NLP system designed for robust, scalable inference.

ch10-nlp/
├── scripts/
│   ├── train_chat_model.py         # Fine-tune DistilBERT on chat ticket dataset
│   ├── predict_api.py              # FastAPI server for live prediction serving
│   └── utils.py                   # Utility functions for data processing and evaluation
├── nlp/tokenizer/                 # Saved HuggingFace tokenizer artifacts
├── model/chat_distilbert.pt       # Fine-tuned DistilBERT PyTorch model weights
├── requirements.txt               # Python dependencies
├── README.md                     # This file

Setup Instructions

Clone the repository
git clone https://github.com/RamadhanAI/ch10-nlp.git
cd ch10-nlp
Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
Install dependencies
pip install -r requirements.txt
Training the Model

Run the fine-tuning script to train DistilBERT on your customer support chat dataset:

python scripts/train_chat_model.py --data data/chat_tickets.csv --epochs 3 --batch_size 16
Training parameters can be adjusted via command-line arguments. The fine-tuned model will be saved in the model/ directory.

Running the Prediction API

Start the FastAPI server to serve predictions in real time:

uvicorn scripts.predict_api:app --host 0.0.0.0 --port 8000 --reload
Sample Prediction Request
Send a POST request to the /predict endpoint with a JSON payload:

{
  "text": "I am unable to login to my account since yesterday."
}
Response will contain predicted ticket category and confidence score.

Utilities

Data processing: Functions for cleaning and tokenizing raw text data.
Evaluation: Scripts for accuracy, precision, recall, and F1-score calculation.
Interpretability: Tools to visualize important words influencing model decisions.
Model and Tokenizer

Tokenizer artifacts saved using HuggingFace’s save_pretrained() method located in nlp/tokenizer/.
Fine-tuned model weights in PyTorch format stored in model/chat_distilbert.pt.
Ensure you load the tokenizer and model from these exact paths during inference to avoid token mismatch errors.

Troubleshooting

Confirm compatible PyTorch and Transformers versions (see requirements.txt).
Check GPU availability and CUDA driver versions for accelerated training.
For API errors, consult FastAPI and Uvicorn logs.
Contributions

Contributions and suggestions are welcome! Please open issues or submit pull requests for improvements or bug fixes.

