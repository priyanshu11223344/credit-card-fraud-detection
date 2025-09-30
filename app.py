import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Load data
data = pd.read_csv('creditcard.csv')

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split data into features and labels
X = data.drop(columns="Class", axis=1)
y = data["Class"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=2)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# Streamlit app
st.title("💳 Credit Card Fraud Detection")
st.write("Enter 30 comma-separated feature values to check if a transaction is legitimate or fraudulent.")

# Input field
input_str = st.text_input("🔢 Enter features (comma-separated)")

if st.button("Predict"):
    try:
        input_list = list(map(float, input_str.strip().split(",")))

        if len(input_list) != X.shape[1]:
            st.error(f"Expected {X.shape[1]} features, but got {len(input_list)}.")
        else:
            input_scaled = scaler.transform([input_list])
            prediction = model.predict(input_scaled)[0]

            if prediction == 0:
                st.success("✅ Legitimate transaction")
            else:
                st.error("⚠ Fraudulent transaction")
    except ValueError:
        st.error("Invalid input. Please enter numeric values separated by commas.")

# Display model performance (optional)
st.sidebar.title("📊 Model Performance")
st.sidebar.write(f"Training Accuracy: {train_acc:.2f}")
st.sidebar.write(f"Testing Accuracy: {test_acc:.2f}")