from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('./resume_category_model')
tokenizer = AutoTokenizer.from_pretrained('./resume_category_model')

# Define the category labels (update these based on your actual labels)
categories = [
    "Accountant", "Advocate", "Agriculture", "Apparell", "Arts", 
    "Automobile", "Aviation", "Banking", "BPO", "Business Development",
    "Chef", "Construction", "Consultant", "Designer", "Digital Media",
    "Engineering", "Finance", "Fitness", "Healthcare", "HR",
    "Information Technology", "Public Relations", "Sales", "Teacher"
]

def preprocess_text(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    return inputs

def predict(text):
    inputs = preprocess_text(text)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    return predictions.item()

def get_category_name(label_id):
    return categories[label_id]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_category():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']
    predicted_label_id = predict(text)
    predicted_category = get_category_name(predicted_label_id)
    return jsonify({'predicted_category': predicted_category})

if __name__ == '__main__':
    app.run(debug=True)
