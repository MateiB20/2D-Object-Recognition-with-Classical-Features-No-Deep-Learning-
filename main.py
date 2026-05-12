import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
#import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import time

image_folder = 'train_images'
image_paths=[]
train_labels=[]
x_values=[]
knn_accuracies=[]
svm_accuracies=[]
k = 1000
start=time.perf_counter()


def get_spatial_pyramid_histogram(img_shape, coords, descriptors, kmeans_model, k, levels=2):
    h, w = img_shape
    pyramid_histo = []
    for level in range(levels):
        n_cells = 2 ** level
        cell_h, cell_w = h / n_cells, w / n_cells
        for i in range(n_cells):
            for j in range(n_cells):
                idx = np.where((coords[:, 0] >= j * cell_w) & (coords[:, 0] < (j + 1) * cell_w) &
                               (coords[:, 1] >= i * cell_h) & (coords[:, 1] < (i + 1) * cell_h))[0]
                histo = np.zeros(k)
                if len(idx) > 0:
                    preds = kmeans_model.predict(descriptors[idx])
                    for p in preds:
                        histo[p] += 1
                pyramid_histo.extend(histo)
    pyramid_histo = np.array(pyramid_histo)
    return pyramid_histo / (np.sum(pyramid_histo) + 1e-6)



for filename in os.listdir(image_folder):
    image_paths.append(os.path.join(image_folder, filename))
    label = ''
    for f in filename:
        if f.isdigit() or f.isspace():
            break
        label += f
    train_labels.append(label)



orb = cv.ORB_create(nfeatures=1000, fastThreshold=10)
sift = cv.SIFT_create(nfeatures=1000)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
test_sizes=[0.1,0.2,0.3,0.5]
versus=['BoF Only', 'BoF + Spatial Pyramid']
for comparison in versus:
    x_values = []
    knn_accuracies = []
    svm_accuracies = []
    for test_size in test_sizes:
        print(f"{1-test_size}/{test_size} train/test split")
        all_descriptors = []
        img_descriptors = []
        train_paths, test_paths, train_y, test_y = train_test_split(image_paths,train_labels,test_size=test_size,stratify=train_labels)

        train_histograms = []
        for filename in train_paths:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img = cv.imread(filename, 0)
            img = clahe.apply(img)
            if img is None:
                print(f"None image {filename}")
                continue
            kp, des = sift.detectAndCompute(img, None)
            if des is not None and len(des) > 10:
                all_descriptors.extend(des)
                img_descriptors.append(des)
            else:
                print(f"No descriptors image {filename}")
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(np.array(all_descriptors))

        if comparison == 'BoF + Spatial Pyramid':
            for filename in train_paths:
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                img = cv.imread(filename, 0)
                img = clahe.apply(img)
                if img is None:
                    print(f"None image {filename}")
                    continue
                kp, des = sift.detectAndCompute(img, None)
                coords = np.array([p.pt for p in kp])
                histo = get_spatial_pyramid_histogram(img.shape, coords, des, kmeans, k)
                train_histograms.append(histo)
        else:
            for des in img_descriptors:
                histo = np.zeros(k)
                if len(des) > 0:
                    predictions = kmeans.predict(des)
                    for p in predictions:
                        histo[p] += 1
                    if np.sum(histo) > 0:
                        histo = histo / np.sum(histo)
                train_histograms.append(histo)

        train_histograms = np.array(train_histograms)
        knn = KNeighborsClassifier(n_neighbors=3,weights='distance', metric='euclidean')
        knn.fit(train_histograms, train_y)
        svm_model = SVC(kernel='rbf', C=10, gamma='scale')
        svm_model.fit(train_histograms, train_y)

        knn_preds = []
        svm_preds = []
        actual_y = []

        for name, model in [("KNN", knn), ("SVM", svm_model)]:
            print(name)
            for filename in test_paths:
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
                img = clahe.apply(img)
                kp, des = sift.detectAndCompute(img, None)
                coords = np.array([p.pt for p in kp])
                if img is not None:
                    kp, des = sift.detectAndCompute(img, None)
                    if comparison == 'BoF + Spatial Pyramid':
                        test_histo = get_spatial_pyramid_histogram(img.shape, coords, des, kmeans, k)
                    else:
                        test_histo = np.zeros(k)
                        if des is not None:
                            test_predictions = kmeans.predict(des)
                            for p in test_predictions:
                                test_histo[p] += 1
                            if np.sum(test_histo) > 0:
                                test_histo = test_histo / np.sum(test_histo)
                    result = model.predict([test_histo])
                    print(f"I think this image ({filename}) is a: {result[0]}")
                    true_label = test_y[test_paths.index(filename)]
                    if name == 'KNN':
                        knn_preds.append(result[0])
                        actual_y.append(true_label)
                    else:
                        svm_preds.append(result[0])
                    img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
                    plt.imshow(img2), plt.show()

        print(f"\n--- Rezultate pentru split {comparison} {int((1-test_size)*100)}/{int(test_size*100)} ---")
        print(f"KNN Accuracy:  {accuracy_score(actual_y, knn_preds) * 100:.2f}%")
        print(f"SVM Accuracy:  {accuracy_score(actual_y, svm_preds) * 100:.2f}%")
        print()
        report = classification_report(actual_y, knn_preds, digits=3, zero_division=0)
        with open(f"rezultate_proiect_k={k}.txt", "a") as f:
            f.write(f"\nModel: KNN, k={k}, Split: {1 - test_size}/{test_size}\n")
            f.write(report)
            f.write("-" * 40 + "\n")
        report = classification_report(actual_y, svm_preds, digits=3, zero_division=0)
        with open(f"rezultate_proiect_k={k}.txt", "a") as f:
            f.write(f"\nModel: SVM, k={k}, Split: {1 - test_size}/{test_size}\n")
            f.write(report)
            f.write("-" * 40 + "\n")
        x_values.append(1 - test_size)
        knn_accuracies.append(accuracy_score(actual_y, knn_preds))
        svm_accuracies.append(accuracy_score(actual_y, svm_preds))
        cm = confusion_matrix(actual_y, knn_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(actual_y))
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"matrice_de_confuzie_KNN_k={k} {comparison} split {test_size}")
        plt.savefig(f"matrice_de_confuzie_KNN_k={k} {comparison} split {test_size}.png", dpi=300, bbox_inches='tight')
        plt.show()
        cm = confusion_matrix(actual_y, svm_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(actual_y))
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"matrice_de_confuzie_SVM_k={k} {comparison} split {test_size}")
        plt.savefig(f"matrice_de_confuzie_SVM_k={k} {comparison} split {test_size}.png", dpi=300, bbox_inches='tight')
        plt.show()
    plt.figure()
    plt.plot(x_values, knn_accuracies, marker='o', label='KNN Accuracy')
    plt.plot(x_values, svm_accuracies, marker='s', label='SVM Accuracy')
    plt.title(f'Rezultat experiment {comparison} k={k}')
    plt.xlabel('Train Size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'rezultat_experiment_k={k} {comparison}.png')
    plt.show()
    plt.close()
end=time.perf_counter()
with open(f"timpi_k={k}.txt", "a") as f:
    f.write(f"{(end-start)*1000} ms k={k}")