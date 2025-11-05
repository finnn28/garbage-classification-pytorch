# 🗑️ Garbage Classification using PyTorch & Streamlit

This project implements a **Deep Learning model** for classifying waste images into 6 categories using **PyTorch** and **Transfer Learning (MobileNetV2)**.
It also includes an **interactive Streamlit web app** that allows users to upload an image and see the prediction results instantly.

---

## Features

✅ Automatic dataset split (train, validation, test)
✅ Deep learning model using **MobileNetV2**
✅ Achieved **92% validation accuracy** on test data
✅ Visualized training progress and confusion matrix
✅ Streamlit web-based interface for easy prediction
✅ Displays probabilities for all classes + final conclusion

---

## Classes

The model can classify images into the following 6 waste categories:

1. **Battery**
2. **Glass**
3. **Metal**
4. **Organic**
5. **Paper**
6. **Plastic**

---

##  Project Structure

```
garbageClassification-Pytorch/
│
├── app.py                         # Streamlit web app
├── garbage_classification_pytorch.py  # Training script
├── predict_image.py               # CLI-based prediction script
├── output/
│   └── garbage_cnn_model.pth      # Trained model file
├── data/                          # Dataset folder (not uploaded to GitHub)
├── requirements.txt               # Dependencies list
└── README.md                      # This file
```

---

## Installation & Usage

### Clone the repository

```bash
git clone https://github.com/<username>/garbage-classification-pytorch.git
cd garbage-classification-pytorch
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run locally (Streamlit App)

```bash
streamlit run app.py
```

### Run training script (optional)

```bash
python garbage_classification_pytorch.py
```

---

## Deployment

This project can be deployed easily using **Streamlit Community Cloud**.

1. Push your code to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Select your repo and set main file = `app.py`
4. Click **Deploy**

---

## Model Performance

|              Metric | Result                          |
| ------------------: | ------------------------------- |
| Validation Accuracy | **92.15%**                      |
|  Loss (final epoch) | 0.2174                          |
|               Model | MobileNetV2 (Transfer Learning) |

Confusion Matrix and Accuracy Plot are available in the training results.

---

## Example Prediction

Upload an image through the Streamlit web app, and the model will output:

* Class probabilities for all 6 categories
* Highlight the class with the highest confidence as **final prediction**

---

## Authors

Developed by **Finn Team**
Department of Informatics, [Mikroskil University]
Machine Learning Project — Semester 7

