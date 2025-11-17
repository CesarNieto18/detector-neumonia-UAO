
"""
Módulo encargado de realizar el preprocesamiento de imágenes
para el modelo de detección de neumonía.

Incluye:
- Redimensionamiento
- Conversión a escala de grises
- Mejora de contraste con CLAHE
- Normalización
- Ajuste de dimensiones para TensorFlow
"""

import cv2
import numpy as np

def preprocess(array):
    try:
        # Validación básica
        if array is None:
            raise ValueError("La imagen recibida es None.")

        # Redimensionar a 512x512
        array = cv2.resize(array, (512, 512))

        # Convertir a escala de grises
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

        # Mejorar contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        array = clahe.apply(array)

        # Normalización y reshape
        array = array / 255.0
        array = np.expand_dims(array, axis=-1)
        array = np.expand_dims(array, axis=0)

        return array

    except Exception as e:
        print(f"Error en el preprocesamiento de la imagen: {e}")
        return None
