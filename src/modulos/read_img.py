#!/usr/bin/env python
# -*- coding: utf-8 -*-

    
"""
Módulo para lectura de imágenes médicas en formatos DICOM, JPG y PNG
"""

import pydicom as dicom
import cv2
import numpy as np
from PIL import Image
import os

def read_dicom_file(path):
    """
    Lee un archivo DICOM y lo convierte a formato RGB para procesamiento.
    
    Args:
        path (str): Ruta del archivo DICOM
        
    Returns:
        tuple: (img_RGB, img2show)
            - img_RGB: Imagen en formato RGB como numpy array (para procesamiento)
            - img2show: Imagen PIL para visualización en interfaz
    """
    try:
        # ✅ MEJORADO: Usar dcmread en lugar de read_file (más moderno)
        dataset = dicom.dcmread(path)
        img_array = dataset.pixel_array
        
        # Crear imagen PIL para visualización
        img2show = Image.fromarray(img_array)
        
        # Normalizar la imagen para procesamiento
        img2 = img_array.astype(float)
        img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
        img2 = np.uint8(img2)
        
        # Convertir a RGB (3 canales)
        img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        
        print(f"✅ DICOM cargado: {os.path.basename(path)} - Tamaño: {img_array.shape}")
        return img_RGB, img2show
        
    except Exception as e:
        print(f"❌ Error leyendo archivo DICOM {path}: {e}")
        return None, None

def read_jpg_file(path):
    """
    Lee un archivo de imagen en formato JPG/PNG y lo procesa.
    
    Args:
        path (str): Ruta del archivo de imagen
        
    Returns:
        tuple: (img_processed, img2show)
            - img_processed: Imagen procesada como numpy array
            - img2show: Imagen PIL para visualización
    """
    try:
        # Leer imagen con OpenCV
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"No se pudo leer la imagen: {path}")
            
        img_array = np.asarray(img)
        img2show = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Normalizar la imagen
        img2 = img_array.astype(float)
        img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
        img2 = np.uint8(img2)
        
        print(f"✅ Imagen cargada: {os.path.basename(path)} - Tamaño: {img_array.shape}")
        return img2, img2show
        
    except Exception as e:
        print(f"❌ Error leyendo archivo de imagen {path}: {e}")
        return None, None

def read_image_file(path):
    """
    Función principal unificada que detecta automáticamente el tipo de archivo
    y llama a la función de lectura apropiada.
    
    Args:
        path (str): Ruta del archivo de imagen
        
    Returns:
        tuple: (img_processed, img2show) o (None, None) en caso de error
    """
    if not path or not os.path.exists(path):
        print(f"❌ Archivo no encontrado: {path}")
        return None, None
    
    # Obtener extensión del archivo
    file_extension = path.lower().split('.')[-1]
    
    print(f"📁 Cargando archivo: {os.path.basename(path)}")
    
    try:
        if file_extension == 'dcm':
            return read_dicom_file(path)
        elif file_extension in ['jpg', 'jpeg', 'png']:
            return read_jpg_file(path)
        else:
            print(f"⚠️ Formato de archivo no soportado: {file_extension}")
            return None, None
            
    except Exception as e:
        print(f"❌ Error inesperado cargando {path}: {e}")
        return None, None
    
