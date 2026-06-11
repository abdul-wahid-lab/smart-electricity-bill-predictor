import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

df = pd.read_csv("electricity_bill_dataset.csv")

X = df.drop("ElectricityBill", axis=1)
y = df["ElectricityBill"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

score = r2_score(y_test, model.predict(X_test))
print(f"R2 Score: {score:.6f}")

joblib.dump(model, "electricity_model.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")
print("Model and features saved.")
print("Features:", X.columns.tolist())
