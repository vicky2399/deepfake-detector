import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Paths
real_path = "D:\\DeepFakeProjectNsic\\real"
fake_path = "D:\\DeepFakeProjectNsic\\fake"


# Image size
IMG_SIZE = (128, 128)

def load_images(path, label):
    images = []
    labels = []
    for file in os.listdir(path):
        if file.endswith(".jpg") or file.endswith(".png"):
            img = load_img(os.path.join(path, file), target_size=IMG_SIZE)
            img_array = img_to_array(img) / 255.0  # normalize
            images.append(img_array)
            labels.append(label)
    return images, labels

# Load real and fake images
real_images, real_labels = load_images(real_path, 0)  # 0 = real
fake_images, fake_labels = load_images(fake_path, 1)  # 1 = fake

# Combine data
X = np.array(real_images + fake_images)
y = np.array(real_labels + fake_labels)

print("Images loaded:", X.shape)
print("Labels loaded:", y.shape)
