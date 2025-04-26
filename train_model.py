import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Step 1: Create the dataset manually
data = {
    "Complaint Text": [
        "Missed flight due to late security check",
        "Airline gate change wasn’t announced",
        "I got to airport late",
        "Heavy traffic caused delay to airport",
        "Airline staff delayed boarding",
        "Passport control took too long",
        "Bad weather delayed my arrival",
        "Flight gate was changed without notice",
        "TSA line was extremely slow",
        "Personal emergency caused delay",
        "Boarding gate closed earlier than announced",
        "Flight delayed but no announcements",
        "Uber driver took wrong route to airport",
        "I overslept and missed my flight",
        "Security staff was undertrained causing huge lines"
    ],
    "Fault Type": [
        "External",  # Security check
        "Airline",   # Gate change
        "Passenger", # Late arrival
        "External",  # Traffic
        "Airline",   # Staff delayed boarding
        "External",  # Passport control
        "External",  # Weather
        "Airline",   # No notice
        "External",  # TSA line
        "Passenger", # Personal issue
        "Airline",   # Boarding gate early close
        "Airline",   # Delay announcements missing
        "External",  # Uber driver wrong
        "Passenger", # Overslept
        "External"   # Security issue
    ]
}

# Step 2: Save to CSV
df = pd.DataFrame(data)
df.to_csv('missed_flight_data.csv', index=False)

# Step 3: Load and prepare data
X = df['Complaint Text']
y = df['Fault Type']

# Step 4: Vectorize the text
vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)

# Step 5: Train the model
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Save model and vectorizer
pickle.dump(model, open('missed_flight_model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("✅ Model and vectorizer saved successfully!")
