# Comment-Toxicity-Detection
A machine learning-based web application that detects toxic comments using Natural Language Processing (NLP) techniques and provides real-time predictions through a Streamlit interface.
# Comment Toxicity Detection using Machine Learning and Streamlit

## Project Overview

This project aims to detect toxic comments from user-generated text using Natural Language Processing (NLP) and Machine Learning techniques. The application helps online communities identify harmful content and improve content moderation.

## Problem Statement

Online platforms often face issues such as abusive language, hate speech, and offensive comments. Manual moderation is time-consuming and inefficient. This project provides an automated solution to classify comments as Toxic or Non-Toxic.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* NLP (TF-IDF Vectorization)
* Streamlit
* Joblib

## Project Workflow

1. Data Collection
2. Data Preprocessing
3. Text Cleaning
4. Feature Engineering using TF-IDF
5. Model Training using Logistic Regression
6. Model Evaluation
7. Model Saving using Joblib
8. Streamlit Deployment

## Dataset

The dataset contains user comments and toxicity labels.

Main Columns:

* comment_text
* toxic

## Model Used

* TF-IDF Vectorizer
* Logistic Regression Classifier

## Project Structure

Comment-Toxicity-Detection/

├── app.py

├── train_model.py

├── predict.py

├── toxicity_model.pkl

├── requirements.txt

├── README.md

└── screenshots/

## Installation

Clone the repository:

git clone https://github.com/yourusername/Comment-Toxicity-Detection.git

Install dependencies:

pip install -r requirements.txt

## Run Model Training

python train_model.py

## Run Streamlit Application

streamlit run app.py

## Features

* Real-time toxicity prediction
* Bulk CSV prediction
* Download prediction results
* User-friendly Streamlit interface

## Results

The model successfully classifies toxic and non-toxic comments with high accuracy.

## Future Enhancements

* Deep Learning Models (LSTM/BERT)
* Multi-label Toxicity Classification
* Cloud Deployment
* Real-time API Integration

## Author

Girish

## License

This project is developed for educational and research purposes.
