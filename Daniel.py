import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import json
import os

# ------------------ USER DATA ------------------
# Save users in a JSON file
USER_FILE = "users.json"

# Load existing users
if os.path.exists(USER_FILE):
    with open(USER_FILE, "r") as f:
        USER_CREDENTIALS = json.load(f)
else:
    USER_CREDENTIALS = {}  # Empty dict if no file exists

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# ------------------ LOGIN / SIGN UP ------------------
st.title("🔒 Fake News Detector Login")

mode = st.radio("Select Option", ["Login", "Sign Up"])

username = st.text_input("Username")
password = st.text_input("Password", type="password")

if mode == "Login":
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.success(f"Welcome {username}!")
            st.session_state['logged_in'] = True
        else:
            st.error("❌ Incorrect username or password")
else:  # Sign Up mode
    if st.button("Create Account"):
        if username in USER_CREDENTIALS:
            st.error("❌ Username already exists")
        elif username == "" or password == "":
            st.error("❌ Username and password cannot be empty")
        else:
            USER_CREDENTIALS[username] = password
            # Save to JSON
            with open(USER_FILE, "w") as f:
                json.dump(USER_CREDENTIALS, f)
            st.success("✅ Account created! You can now log in.")

# ------------------ FAKE NEWS DETECTOR ------------------
if st.session_state['logged_in']:
    st.title("📰 Fake News Detector")

    # Load dataset
    data = pd.read_csv('news.csv')  # Make sure this file exists

    # Split features and labels
    X = data['text']
    y = data['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert text to TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    tfidf_test = tfidf_vectorizer.transform(X_test)

    # Train classifier
    classifier = PassiveAggressiveClassifier(max_iter=50)
    classifier.fit(tfidf_train, y_train)

    # Evaluate
    y_pred = classifier.predict(tfidf_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy*100:.2f}%")

    # User input for prediction
    user_input = st.text_area("Enter news to check:")

    if st.button("Check"):
        vector = tfidf_vectorizer.transform([user_input])
        prediction = classifier.predict(vector)[0]
        if prediction == 'FAKE':
            st.error("⚠️ This news is likely FAKE")
        else:
            st.success("✅ This news is likely REAL")
    
    # Logout button
    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()

