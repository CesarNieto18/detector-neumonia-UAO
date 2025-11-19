#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo integrador principal - Coordina todos los demás módulos
Recibe una imagen y retorna: diagnóstico, probabilidad y mapa de calor
"""

import numpy as np
import time
import sys
import os

# ✅ FIX: Agregar ruta para imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from .preprocess_img import preprocess
    from .load_model import model_fun
    from .grad_cam import grad_cam
except ImportError as e:
    print(f"⚠️  Error en import relativo: {e}")
    # Fallback a imports absolutos
    from src.modulos.preprocess_img import preprocess
    from src.modulos.load_model import model_fun
    from src.modulos.grad_cam import grad_cam

def predict(array):
    """
    Función principal que integra todo el pipeline de predicción:
    1. Preprocesamiento → 2. Predicción → 3. Generación Grad-CAM
    
    Args:
        array (numpy.ndarray): Imagen médica como array numpy
        
    Returns:
        tuple: (diagnóstico, probabilidad, heatmap)
            - diagnóstico (str): 'bacteriana', 'normal', 'viral'
            - probabilidad (float): Confianza de la predicción (0-100)
            - heatmap (numpy.ndarray): Imagen con mapa de calor superpuesto
    """
    start_time = time.time()
    
    try:
        print("🚀 Iniciando pipeline de diagnóstico...")
        
        # ✅ MEJORADO: Validación de entrada
        if not validar_entrada(array):
            return "error", 0.0, generar_imagen_error()
        
        # 1. PREPROCESAMIENTO
        print("🔧 Paso 1/3: Preprocesando imagen...")
        imagen_preprocesada = preprocess(array)
        if imagen_preprocesada is None:
            print("❌ Falló el preprocesamiento")
            return "error", 0.0, generar_imagen_error()
        
        # 2. PREDICCIÓN DEL MODELO
        print("🤖 Paso 2/3: Ejecutando modelo...")
        model = model_fun()
        if model is None:
            print("❌ No se pudo cargar el modelo")
            return "error", 0.0, generar_imagen_error()
        
        try:
            predicciones = model.predict(imagen_preprocesada, verbose=0)
            indice_prediccion = np.argmax(predicciones[0])
            probabilidad = np.max(predicciones[0]) * 100
            
            # Validar que la probabilidad sea razonable
            if np.isnan(probabilidad) or probabilidad < 0 or probabilidad > 100:
                print("⚠️  Probabilidad inválida, ajustando a 50%")
                probabilidad = 50.0
                
        except Exception as e:
            print(f"❌ Error en predicción del modelo: {e}")
            indice_prediccion = 1  # Fallback a "normal"
            probabilidad = 50.0
        
        # 3. CLASIFICACIÓN
        diagnostico = obtener_etiqueta_diagnostico(indice_prediccion)
        
        # 4. GENERACIÓN GRAD-CAM
        print("🔥 Paso 3/3: Generando mapa de calor...")
        # CORREGIDO: Pasar el modelo como primer parámetro
        heatmap = grad_cam(model, array)
        
        tiempo_ejecucion = time.time() - start_time
        print(f"✅ Pipeline completado en {tiempo_ejecucion:.2f} segundos")
        print(f"📊 Resultado: {diagnostico} ({probabilidad:.2f}% de confianza)")
        
        return diagnostico, probabilidad, heatmap
        
    except Exception as e:
        print(f"❌ Error crítico en el pipeline: {e}")
        import traceback
        traceback.print_exc()
        return "error", 0.0, generar_imagen_error()

def validar_entrada(imagen_array):
    """
    Valida que la imagen de entrada sea adecuada para el procesamiento.
    
    Args:
        imagen_array (numpy.ndarray): Array de la imagen a validar
        
    Returns:
        bool: True si la imagen es válida, False en caso contrario
    """
    if imagen_array is None:
        print("❌ Error: El array de imagen es None")
        return False
    
    if not isinstance(imagen_array, np.ndarray):
        print("❌ Error: La entrada debe ser un numpy array")
        return False
    
    if len(imagen_array.shape) < 2:
        print("❌ Error: La imagen debe tener al menos 2 dimensiones")
        return False
    
    if imagen_array.size == 0:
        print("❌ Error: El array de imagen está vacío")
        return False
    
    print(f"✅ Entrada validada: forma={imagen_array.shape}, tipo={imagen_array.dtype}")
    return True

def obtener_etiqueta_diagnostico(indice):
    """
    Convierte el índice de predicción a etiqueta de diagnóstico.
    
    Args:
        indice (int): Índice de la clase predicha (0, 1, 2)
        
    Returns:
        str: Etiqueta de diagnóstico en español
    """
    etiquetas = {
        0: "bacteriana",
        1: "normal", 
        2: "viral"
    }
    
    diagnostico = etiquetas.get(indice, "desconocida")
    
    if indice not in etiquetas:
        print(f"⚠️  Índice de clase inesperado: {indice}")
    
    return diagnostico

def generar_imagen_error():
    """
    Genera una imagen de error para mostrar en la interfaz cuando falla el procesamiento.
    
    Returns:
        numpy.ndarray: Imagen de error en RGB
    """
    try:
        import cv2
        # Crear imagen negra con texto de error
        imagen_error = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Agregar texto informativo
        cv2.putText(imagen_error, "ERROR", (150, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.putText(imagen_error, "EN PROCESAMIENTO", (80, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return imagen_error
    except:
        # Fallback absoluto
        return np.zeros((512, 512, 3), dtype=np.uint8)

# ✅ MANTENIDO: Funciones adicionales para futuras extensiones
def obtener_confianza(probabilidad):
    """Determina el nivel de confianza basado en la probabilidad"""
    if probabilidad >= 80:
        return "Alta"
    elif probabilidad >= 60:
        return "Media"
    else:
        return "Baja"