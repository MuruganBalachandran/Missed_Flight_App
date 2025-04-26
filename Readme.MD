# ✈️ Missed Flight Explanation & Rebooking Assistant

A smart NLP-based tool to classify complaints and assist passengers with suitable rebooking options.

![Output](https://github.com/user-attachments/assets/1aa34cfd-198f-4144-a8cf-0079ae2f058d)

## 📌 Overview

This app helps passengers who have missed their flights by analyzing their complaint messages. It identifies who is at fault (**Passenger**, **Airline**, or **External Factors**), provides a short explanation, and suggests a suitable **rebooking action**.

Built using:

- 🧠 Logistic Regression NLP model  
- 🌐 Streamlit for web UI  
- 🗂️ Custom dataset of complaint texts  

## 🧠 Features

- Classifies complaints into:
  - ✈️ Airline fault
  - 🧍 Passenger fault
  - 🌧️ External factors

- Provides rebooking suggestions like:
  - ✅ Free rebooking (if airline at fault)
  - 💸 Paid rebooking (if passenger at fault)
  - 🆘 Alternative helpdesk advice (for external causes)

## 🛠️ How It Works

1. User types in a reason for missing their flight  
2. The app uses a trained **Logistic Regression** model to predict the **fault type**  
3. A custom response is shown to guide the user for next steps  

## 📂 Folder Structure

missed-flight-app/ │ ├── app.py # Streamlit UI ├── train_model.py # Training script for the model ├── missed_flight_data.csv # Custom dataset (auto-created) ├── missed_flight_model.pkl # Trained classification model ├── vectorizer.pkl # TF-IDF vectorizer ├── requirements.txt # Required Python packages └── README.md # Project documentation

bash
Copy
Edit

## 🚀 Getting Started

### 1. Clone the repo
```
git clone https://github.com/your-username/this-repo

cd missed-flight-app
2. Install dependencies

pip install -r requirements.txt
3. Train the model

python train_model.py
4. Run the app

streamlit run app.py
📊 Example Complaint Inputs
```
Complaint	Fault Type	Suggested Action
I was late due to traffic	External	Helpdesk support
The airline staff delayed boarding	Airline	Free rebooking
I overslept and missed my flight	Passenger	Paid rebooking
🔮 Possible Enhancements
Add sentiment analysis to prioritize urgent cases

Add multilingual support using translation APIs

Replace Logistic Regression with BERT or LLMs

Save complaint logs for retraining and analytics
