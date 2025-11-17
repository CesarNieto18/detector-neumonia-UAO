"""
Módulo encargado exclusivamente de la lectura de imágenes:
- Archivos DICOM (.dcm)
- Archivos JPG/JPEG/PNG

Este módulo garantiza cohesión (funciones con responsabilidad única)
y bajo acoplamiento (solo depende de librerías estándar y PIL/pydicom).
"""

import pydicom
import numpy as np
import cv2
from PIL import Image

def read_dicom_file(path):
    try:
        img = pydicom.dcmread(path)
        img_array = img.pixel_array

        img2show = Image.fromarray(img_array)

        img2 = img_array.astype(float)
        img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
        img2 = np.uint8(img2)

        img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

        return img_RGB, img2show

    except Exception as e:
        print(f"Error leyendo archivo DICOM: {e}")
        return None, None


def read_jpg_file(path):
    try:
        img = cv2.imread(path)

        if img is None:
            raise ValueError("No se pudo leer la imagen JPG/PNG.")

        img_array = np.asarray(img)
        img2show = Image.fromarray(img_array)

        img2 = img_array.astype(float)
        img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
        img2 = np.uint8(img2)

        return img2, img2show

    except Exception as e:
        print(f"Error leyendo archivo de imagen: {e}")
        return None, None
