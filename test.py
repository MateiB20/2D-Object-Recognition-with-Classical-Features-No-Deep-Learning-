import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import joblib

kmeans = joblib.load('bof_kmeans.pkl')
knn = joblib.load('knn_model.pkl')

k = 500
image_folder = 'test_images'
for filename in os.listdir(image_folder):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    full_path = os.path.join(image_folder, filename)
    img = cv.imread(full_path, cv.IMREAD_GRAYSCALE)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    if img is not None:
        orb = cv.ORB_create(nfeatures=2000, fastThreshold=10)
        kp, des = orb.detectAndCompute(img, None)
        test_histo = np.zeros(k)
        if des is not None:
            test_predictions = kmeans.predict(des)
            for p in test_predictions:
                test_histo[p] += 1
            if np.sum(test_histo) > 0:
                test_histo = test_histo / np.sum(test_histo)
        result = knn.predict([test_histo])
        print(f"I think this image ({filename}) is a: {result[0]}")
        img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
        plt.imshow(img2), plt.show()
