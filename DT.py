import numpy as np
import joblib
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import cv2

# Pilih model yang ingin digunakan ('decision_tree' atau 'svm')
MODEL_TYPE = 'decision_tree'  # Ubah menjadi 'svm' jika ingin menggunakan SVM
model = DecisionTreeClassifier(max_depth=10, random_state=42)

# Path ke dataset
dataset_folder = 'training_dataset'

# List untuk menyimpan data
data = []
labels = []

# Looping ke setiap folder warna
for color_name in os.listdir(dataset_folder):
    color_path = os.path.join(dataset_folder, color_name)

    if os.path.isdir(color_path):  # Pastikan hanya membaca folder
        for img_name in os.listdir(color_path):
            img_path = os.path.join(color_path, img_name)

            # Baca gambar dalam format BGR dan konversi ke RGB
            img = cv2.imread(img_path)
            if img is None:
                continue  # Lewati file yang tidak bisa dibaca
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Ambil rata-rata nilai warna (R, G, B)
            avg_color = img.mean(axis=(0, 1))

            # Simpan fitur dan label
            data.append(avg_color)
            labels.append(color_name)

# Konversi ke array numpy
X = np.array(data)
y = np.array(labels)

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Pilih model
if MODEL_TYPE == 'decision_tree':
    model = DecisionTreeClassifier(random_state=42)
elif MODEL_TYPE == 'svm':
    model = SVC(kernel='linear', random_state=42)
else:
    raise ValueError("MODEL_TYPE harus 'decision_tree' atau 'svm'")

# Latih model
model.fit(X_train, y_train)

# Evaluasi model
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
print(f"Akurasi pada data latih: {train_acc * 100:.2f}%")
print(f"Akurasi pada data uji: {test_acc * 100:.2f}%")

# Simpan model dan scaler
joblib.dump(model, 'color_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model dan scaler berhasil disimpan!")

# Muat kembali model dan scaler untuk deteksi warna
model = joblib.load('color_model.pkl')
scaler = joblib.load('scaler.pkl')

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ambil pixel tengah gambar
    height, width, _ = frame.shape
    sample_area = frame[height // 2 - 1:height // 2 + 1, width // 2 - 1:width // 2 + 1]  # Ambil area kecil
    pixel_center = np.mean(sample_area, axis=(0, 1))  # Hitung rata-rata warna

    # Normalisasi pixel sebelum prediksi
    pixel_center_reshaped = np.array(pixel_center).reshape(1, -1)  # Ubah ke bentuk yang sesuai
    pixel_center_scaled = scaler.transform(pixel_center_reshaped)  # Normalisasi dengan scaler yang telah disimpan

    # Prediksi warna
    color_pred = model.predict(pixel_center_scaled)[0]

    # Tampilkan warna pada frame
    cv2.putText(frame, f'Color: {color_pred}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
