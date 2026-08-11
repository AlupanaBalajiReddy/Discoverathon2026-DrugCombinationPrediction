import joblib
import pandas as pd

print("Loading trained model...")

model = joblib.load("models/random_forest.pkl")

print("Model loaded successfully!")

print(model)