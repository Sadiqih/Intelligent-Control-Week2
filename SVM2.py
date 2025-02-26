import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

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

# Pilih model SVM dengan kernel RBF agar lebih fleksibel
model = SVC(kernel='rbf', probability=True, random_state=42)
model.fit(X_train, y_train)

# Evaluasi model
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

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
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = frame.shape

    # Konversi frame ke HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 50, 50), (180, 255, 255))  # Threshold diperbaiki
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            roi = frame[y:y+h, x:x+w]

            # Ambil rata-rata warna dari ROI
            avg_color = roi.mean(axis=(0, 1))

            # Normalisasi sebelum prediksi
            avg_color_scaled = scaler.transform([avg_color])
            color_pred = model.predict(avg_color_scaled)[0]
            prob = model.predict_proba(avg_color_scaled).max() * 100

            # Konversi warna ke integer untuk bounding box
            box_color = tuple(map(int, avg_color))

            # Gambar bounding box dengan warna yang sesuai objek
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(frame, f'{color_pred} ({prob:.2f}%)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    # Konversi kembali ke BGR untuk tampilan OpenCV
    frame_display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Frame', frame_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
