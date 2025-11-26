import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model():
    model = models.Sequential([
        layers.Input((28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_quick_cnn(X, y, epochs=3):
    """
    使用切割後影像快速訓練一個小型 CNN。
    """
    X = X.astype("float32") / 255.0
    X = X.reshape(-1, 28, 28, 1)
    model = build_cnn_model()
    model.fit(X, y, epochs=epochs, verbose=0)
    return model

def predict_digit(model, img):
    """
    使用訓練好的 model 對單一 28x28 灰階圖做預測。
    """
    X = img.astype("float32") / 255.0
    X = X.reshape(1, 28, 28, 1)
    pred = model.predict(X, verbose=0)[0]
    return pred
