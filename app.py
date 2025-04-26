import streamlit as st
import pickle

# Load the trained model and vectorizer
model = pickle.load(open('missed_flight_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Define possible actions
actions = {
    "Passenger": "You may rebook the flight via the passenger support portal. A fee may apply.",
    "Airline": "You are eligible for a free rebooking. Please contact the helpdesk.",
    "External": "Please approach the nearest airline helpdesk for alternate arrangements."
}

# Streamlit App
st.title("✈️ Missed Flight Explanation & Rebooking Assistant")
st.write("Describe what happened, and we'll suggest your next steps!")

# Text input
user_input = st.text_area(
    "Describe your issue:", 
    placeholder="e.g. I missed my flight because of long lines at the security check..."
)

# Button
if st.button("Analyze"):
    if user_input.strip():
        vect_input = vectorizer.transform([user_input])
        prediction = model.predict(vect_input)[0]
        
        st.subheader("🔎 Detected Cause:")
        st.info(prediction)
        
        st.subheader("✅ Suggested Action:")
        st.success(actions[prediction])
    else:
        st.warning("⚠️ Please enter a complaint message to analyze.")
