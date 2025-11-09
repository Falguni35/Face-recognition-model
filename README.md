# Face Recognition Model

**Simple face detection / recognition pipeline built with Jupyter notebooks and a saved Keras model.**

---

## Repository contents

* `collect_data.ipynb` — notebook to capture face images for each person (creates labeled folders / dataset).
* `Face_detection.ipynb` — notebook that shows face detection (using Haar cascade) and basic preprocessing.
* `consolidated_data.ipynb` — notebook that prepares the dataset, builds/trains a model and exports `final_model.h5`.
* `recognize.ipynb` — notebook demonstrating how to run real-time recognition using the trained `final_model.h5` and `haarcascade_frontalface_default.xml`.
* `haarcascade_frontalface_default.xml` — OpenCV Haar Cascade for frontal face detection.
* `final_model.h5` — trained Keras model for face recognition.

---

## Quick overview

This repo implements a typical face recognition workflow:

1. **Collect images** for each person using `collect_data.ipynb`.
2. **Preprocess / consolidate** images and labels (`consolidated_data.ipynb`).
3. **Train** a classification model (outputs `final_model.h5`).
4. **Detect faces** in frames using Haar cascades and feed aligned/cropped faces into the model for **recognition** (`recognize.ipynb` / `Face_detection.ipynb`).

---

## Requirements

Tested with Python 3.8+ (recommended). Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate      # on Linux / macOS
.\.venv\Scripts\activate       # on Windows

pip install --upgrade pip
pip install jupyter numpy pandas matplotlib opencv-python tensorflow keras scikit-learn pillow
```

> For GPU training, install the appropriate TensorFlow GPU package and drivers.

---

## How to use

### 1) Collect faces (run `collect_data.ipynb`)

* Launch the webcam capture.
* Detect faces using Haar cascades.
* Save cropped faces in `dataset/<person_name>/image_x.jpg`.

**Goal:** Gather at least 50–200 images per person with variations.

### 2) Prepare and consolidate data (run `consolidated_data.ipynb`)

* Load images from the dataset folders.
* Normalize, resize, and encode labels.
* Split data into training and validation sets.
* Train a CNN and save weights as `final_model.h5`.

### 3) Run recognition (open `recognize.ipynb`)

Example real-time recognition code:

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('final_model.h5')
label_map = {0: 'Alice', 1: 'Bob'}  # adjust per training

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_rgb = cv2.resize(face_img, (128,128))
        face_arr = np.expand_dims(face_rgb.astype('float32') / 255.0, axis=0)
        preds = model.predict(face_arr)
        class_id = np.argmax(preds)
        prob = preds[0][class_id]
        name = label_map.get(class_id, 'Unknown')
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f"{name} {prob:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imshow('Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

---

## Notes & suggestions

* Keep label mapping consistent between training and inference.
* Ensure input shape and normalization match the model.
* Replace Haar cascade with modern detectors (MTCNN, Dlib) for robustness.
* Collect diverse samples for better accuracy.
* Use GPU or lightweight models for real-time performance.
* Always obtain consent when collecting facial images.
* Store large model files using Git LFS if needed.

---

## Folder structure (recommended)

```
Face-recognition-model/
├─ dataset/
│  ├─ Alice/
│  └─ Bob/
├─ notebooks/
│  ├─ collect_data.ipynb
│  ├─ Face_detection.ipynb
│  ├─ consolidated_data.ipynb
│  └─ recognize.ipynb
├─ models/
│  └─ final_model.h5
├─ haarcascade_frontalface_default.xml
├─ requirements.txt
└─ README.md
```

Example `requirements.txt`:

```
jupyter
numpy
pandas
matplotlib
opencv-python
tensorflow
keras
scikit-learn
pillow
```

---

## Troubleshooting

* Haar cascade not loading: check file path.
* Shape errors in `predict`: verify image size and preprocessing.
* Low accuracy: collect more data and tune model.

---

## Improvements

* Add `labels.json` for label mapping.
* Include a CLI (`collect.py`, `train.py`, `recognize.py`).
* Add a license and contribution guidelines.

---
