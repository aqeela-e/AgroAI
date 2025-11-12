import pandas as pd
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

df = pd.read_csv("dataagr_clean.csv")
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

np.random.seed(42)
X_train += np.random.normal(0, 0.05, X_train.shape)

model = Sequential([
    Dense(32, activation='relu', input_shape=(X.shape[1],), kernel_regularizer=l2(0.02)),
    BatchNormalization(),
    Dropout(0.5),

    Dense(16, activation='relu', kernel_regularizer=l2(0.02)),
    BatchNormalization(),
    Dropout(0.5),

    Dense(len(label_encoder.classes_), activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=6,          
    min_delta=0.003,     
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Model selesai dilatih hingga epoch ke-{len(history.history['loss'])}!")
print(f"🔍 Akurasi Validasi Akhir: {acc:.4f}")
print(f"📉 Final Loss: {loss:.4f}")

y_pred = model.predict(X_test, verbose=0)
y_pred_labels = np.argmax(y_pred, axis=1)
print("\n🧾 Classification Report:")
print(classification_report(y_test, y_pred_labels, target_names=label_encoder.classes_))

model.save("trained_model.keras")
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

def predict_crop_backend(N, P, K, temperature, humidity, ph, rainfall):
    sample = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled, verbose=0)
    predicted_index = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    confidence = float(np.max(prediction))

    print("\n🌱 HASIL PREDIKSI:")
    print(f"Tanaman yang Direkomendasikan: {predicted_label.upper()}")
    print(f"Tingkat Keyakinan: {confidence:.2%}")

if __name__ == "__main__":
    print("\n🧪 Tes Prediksi Manual")
    try:
        N = float(input("Input Nitrogen (N): "))
        P = float(input("Input Phosphorus (P): "))
        K = float(input("Input Kalium (K): "))
        temperature = float(input("Input Temperature (°C): "))
        humidity = float(input("Input Humidity (%): "))
        ph = float(input("Input pH tanah: "))
        rainfall = float(input("Input Rainfall (mm): "))
        predict_crop_backend(N, P, K, temperature, humidity, ph, rainfall)
    except ValueError:
        print("⚠️ Input salah! Harus berupa angka.")
