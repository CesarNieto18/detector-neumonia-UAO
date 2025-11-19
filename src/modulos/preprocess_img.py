#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo de preprocesamiento de imágenes médicas
Transforma las imágenes al formato requerido por el modelo:
- Redimensionamiento a 512x512 píxeles
- Conversión a escala de grises
- Ecualización de histograma (CLAHE)
- Normalización de valores [0, 1]
- Conversión a formato de batch (tensor)
"""

import cv2
import numpy as np

def preprocess(array):
    """
    Función principal de preprocesamiento.
    Aplica toda la pipeline de transformaciones a la imagen.
    
    Args:
        array (numpy.ndarray): Imagen original como array numpy
        
    Returns:
        numpy.ndarray: Imagen preprocesada en formato batch (1, 512, 512, 1)
                      o None en caso de error
    """
    try:
        # ✅ MEJORADO: Validación de entrada
        if array is None:
            raise ValueError("El array de entrada es None")
        
        original_shape = array.shape
        print(f"🔧 Preprocesando imagen: {original_shape} -> (512, 512, 1)")
        
        # 1. REDIMENSIONAR a 512x512
        array = cv2.resize(array, (512, 512))
        
        # 2. CONVERTIR a escala de grises (si es necesario)
        if len(array.shape) == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        
        # 3. APLICAR CLAHE para mejora de contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        array = clahe.apply(array)
        
        # 4. NORMALIZAR valores al rango [0, 1]
        array = array.astype(np.float32) / 255.0
        
        # 5. PREPARAR para modelo: agregar dimensiones de batch y canal
        array = np.expand_dims(array, axis=-1)  # Agregar dimensión de canal
        array = np.expand_dims(array, axis=0)   # Agregar dimensión de batch
        
        print(f"✅ Preprocesamiento completado: {original_shape} -> {array.shape}")
        return array
        
    except Exception as e:
        print(f"❌ Error en el preprocesamiento: {e}")
        return None

# ✅ MANTENIDO: Funciones auxiliares para mayor modularidad
def resize_image(array, target_size=(512, 512)):
    """
    Redimensiona una imagen al tamaño objetivo.
    
    Args:
        array (numpy.ndarray): Imagen original
        target_size (tuple): Tamaño objetivo (ancho, alto)
        
    Returns:
        numpy.ndarray: Imagen redimensionada
    """
    return cv2.resize(array, target_size)

def convert_to_grayscale(array):
    """
    Convierte una imagen a escala de grises.
    
    Args:
        array (numpy.ndarray): Imagen original (RGB o BGR)
        
    Returns:
        numpy.ndarray: Imagen en escala de grises
    """
    if len(array.shape) == 3:
        return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    return array

def apply_clahe(array, clip_limit=2.0, tile_grid_size=(4, 4)):
    """
    Aplica ecualización adaptativa del histograma (CLAHE).
    
    Args:
        array (numpy.ndarray): Imagen en escala de grises
        clip_limit (float): Límite de contraste
        tile_grid_size (tuple): Tamaño de la grilla para CLAHE
        
    Returns:
        numpy.ndarray: Imagen con contraste mejorado
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(array)

def normalize_image(array):
    """
    Normaliza los valores de píxeles al rango [0, 1].
    
    Args:
        array (numpy.ndarray): Imagen a normalizar
        
    Returns:
        numpy.ndarray: Imagen normalizada
    """
    return array.astype(np.float32) / 255.0

def add_batch_dimension(array):
    """
    Convierte una imagen al formato de batch para el modelo.
    
    Args:
        array (numpy.ndarray): Imagen individual
        
    Returns:
        numpy.ndarray: Imagen en formato batch (1, height, width, channels)
    """
    # Agregar dimensión de canal si es necesario
    if len(array.shape) == 2:
        array = np.expand_dims(array, axis=-1)
    
    # Agregar dimensión de batch
    array = np.expand_dims(array, axis=0)
    
    return array