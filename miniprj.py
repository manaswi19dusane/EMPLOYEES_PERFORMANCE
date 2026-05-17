import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv(r"C:\Users\Manswi R. dusane\Downloads\archive (1)\test_dataset.csv")

# Title
st.title("Employee Performance Evaluation")

# Show dataset
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Plot team-wise productivity
st.subheader("Team-wise Productivity")
team_perf = df.groupby("team")["targeted_productivity"].mean()
st.bar_chart(team_perf)

# Features and target
X = df[["over_time", "incentive", "idle_time", "no_of_workers", "smv"]]
y = df["targeted_productivity"]

# Split data and train model ONCE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)

# Show model performance
y_pred = model.predict(X_test)
st.subheader("Model Performance on Test Data")
st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# Prediction inputs
st.subheader("Predict Productivity")
over_time = st.number_input("Overtime (minutes)", 0, 8000, 1000)
incentive = st.number_input("Incentive", 0, 200, 50)
idle_time = st.number_input("Idle Time", 0.0, 10.0, 0.0)
no_of_workers = st.number_input("Number of Workers", 0, 60, 30)
smv = st.number_input("SMV", 0.0, 40.0, 10.0)

# Predict on button click
if st.button("Predict"):
    pred = model.predict([[over_time, incentive, idle_time, no_of_workers, smv]])
    st.success(f"Predicted Productivity: {pred[0]:.2f}")
