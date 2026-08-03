import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_colors(image_path, n_colors=3):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pixels = image.reshape((-1,3))

    kmeans = KMeans(n_clusters=n_colors, random_state=0)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_.astype(int)

    return colors.tolist()