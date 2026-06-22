from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64

app = Flask(__name__)
CORS(app) 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

@app.route('/api/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        img_data = data['image'].split(",")[1]
        
        nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        faces = detect_face(img)
        
        face_list = []
        for (x, y, w, h) in faces:
            face_list.append({
                'x': int(x), 
                'y': int(y), 
                'width': int(w), 
                'height': int(h)
            })
            
        return jsonify({'faces': face_list})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)