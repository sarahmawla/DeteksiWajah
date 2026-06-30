from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
import time

app = Flask(__name__)
CORS(app) 

# Memuat model pre-trained Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(image):
    # Mengubah gambar ke grayscale karena Haar Cascade membutuhkan format abu-abu
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Proses deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

@app.route('/api/detect', methods=['POST'])
def detect():
    try:
        # Menerima data JSON dari frontend (index.html)
        data = request.json
        img_data = data['image'].split(",")[1]
        
        # Mengubah string Base64 menjadi array NumPy dan mendekodifikasikannya ke gambar BGR (OpenCV)
        nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Memanggil fungsi deteksi
        faces = detect_face(img)
        
        # --- CODE TAMBAHAN UNTUK AUTO-CAPTURE DATASET ---
        # Membuat folder 'dataset' secara otomatis jika belum ada di direktori
        if not os.path.exists('dataset'):
            os.makedirs('dataset')
        # ------------------------------------------------
        
        face_list = []
        for (x, y, w, h) in faces:
            face_list.append({
                'x': int(x), 
                'y': int(y), 
                'width': int(w), 
                'height': int(h)
            })
            
            # --- PROSES PEMOTONGAN DAN PENYIMPANAN DATASET ---
            # Memotong (crop) area gambar yang hanya berisi wajah
            wajah_crop = img[y:y+h, x:x+w]
            # Menamai file menggunakan timestamp milidetik agar nama file selalu unik dan tidak tumpang tindih
            nama_file = f"dataset/wajah_{int(time.time()*1000)}.jpg"
            # Menyimpan potongan wajah ke dalam folder dataset
            cv2.imwrite(nama_file, wajah_crop)
            # -------------------------------------------------
            
        return jsonify({'faces': face_list})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Jalankan server Flask pada port 5000 dengan mode debug aktif
    app.run(debug=True, port=5000)