import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load data
df = pd.read_csv("sleeptime_prediction_dataset.csv")
X = df.drop("SleepTime", axis=1)
y = df["SleepTime"]

# Standarisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model MLPRegressor (Neural Network)
model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Prediksi dan evaluasi
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("=========== Hasil Model =====================")
print(f"MAE      : {mae:.2f}")
print(f"RMSE     : {rmse:.2f}")
print(f"R² Score : {r2:.2f}")
print("============================================\n")

# Contoh input manual user
input_df = pd.DataFrame([[0.3, 0.3, 7.0, 10.0, 300.0, 4.0]],
    columns=["WorkoutTime", "ReadingTime", "PhoneTime", "WorkHours", "CaffeineIntake", "RelaxationTime"])
input_scaled = scaler.transform(input_df)


# Prediksi
predicted_sleep = model.predict(input_scaled)[0]
print(f"Perkiraan waktu tidur: {predicted_sleep:.2f} jam")
