import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load your dataset
df = pd.read_csv("f2.csv")

# Encode categorical columns
encoders = {}
for col in ["Soil_Type", "Crop_Type", "Fertilizer"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features and target
X = df.drop("Fertilizer", axis=1)
y = df["Fertilizer"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
pickle.dump(model, open("fertilizer_model.pkl", "wb"))
pickle.dump(encoders, open("label_encoders.pkl", "wb"))

print("✅ Model and encoders saved successfully!")
