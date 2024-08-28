# Resume-Categoriser
This repository contains a fine-tuned resume classification model designed to categorize resumes into predefined job categories. The project uses a DistilBERT-based model for sequence classification, trained on a dataset of resumes to predict categories such as HR, Designer, Information-Technology, and more.

# Features

Pre-trained Model: Utilizes DistilBERT for efficient sequence classification.
Fine-tuning: Custom-trained on a resume dataset to classify resumes into job categories.
Flask Web App: Provides an interface to input resume text and receive category predictions.
Categories: Classifies resumes into one of the following categories:

Accountant
Advocate
Agriculture
Apparel
Arts
Automobile
Aviation
Banking
Business-Development
Chef
Consultant
Construction
Digital-Media
Designer
Engineering
Finance
Fitness
Healthcare
HR
Information-Technology
Public-Relations
Sales
Teacher


# Technologies Used
Transformers Library: For loading and using the fine-tuned DistilBERT model.
Flask: For creating a simple web interface to interact with the model.
PyTorch: For model inference and handling tensors.
