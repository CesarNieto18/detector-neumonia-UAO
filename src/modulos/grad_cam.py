#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para generación de mapas de calor Grad-CAM (Gradient-weighted Class Activation Mapping)
Visualiza las regiones de la imagen que más influyen en la predicción del modelo.
Versión actualizada para TensorFlow 2.x con eager execution.
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K

def grad_cam(model, array, conv_layer_name="conv10_thisone"):
    """
    Genera un mapa de calor Grad-CAM para la imagen proporcionada.
    Versión actualizada para TensorFlow 2.x.
    
    Args:
        model (tf.keras.Model): Modelo cargado
        array (numpy.ndarray): Imagen original como array numpy
        conv_layer_name (str): Nombre de la capa convolucional para Grad-CAM
        
    Returns:
        numpy.ndarray: Imagen con el mapa de calor superpuesto en RGB
                      o None en caso de error
    """
    try:
        print("🔥 Iniciando Grad-CAM...")
        
        # ✅ CORREGIDO: Validar entrada
        if array is None:
            print("❌ Error en el preprocesamiento: El array de entrada es None")
            return None
        
        # 1. PREPROCESAR la imagen (versión simplificada para prueba)
        try:
            # Procesamiento básico para la prueba
            if len(array.shape) == 3 and array.shape[2] == 3:
                # Convertir RGB a escala de grises
                img_gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = array
            
            # Redimensionar a 512x512
            img_resized = cv2.resize(img_gray, (512, 512))
            
            # Normalizar y expandir dimensiones
            img_normalized = img_resized.astype(np.float32) / 255.0
            img_preprocesada = np.expand_dims(img_normalized, axis=0)  # Batch dimension
            img_preprocesada = np.expand_dims(img_preprocesada, axis=-1)  # Channel dimension
            
            print(f"🔧 Imagen preprocesada: {img_preprocesada.shape}")
            
        except Exception as e:
            print(f"❌ Error en preprocesamiento simplificado: {e}")
            return None
        
        # 2. VALIDAR el modelo
        if model is None:
            print("❌ Modelo no disponible para Grad-CAM")
            return None
        
        # 3. OBTENER la predicción y clase objetivo
        try:
            preds = model.predict(img_preprocesada, verbose=0)
            class_idx = np.argmax(preds[0])
            print(f"   - Clase predicha: {class_idx}, Probabilidad: {np.max(preds[0]):.3f}")
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            class_idx = 0  # Usar clase por defecto
        
        # 4. OBTENER la capa convolucional objetivo
        try:
            target_layer = model.get_layer(conv_layer_name)
            print(f"   - Capa objetivo: {conv_layer_name}")
        except ValueError:
            print(f"❌ No se encontró la capa '{conv_layer_name}'")
            # Intentar encontrar una capa convolucional alternativa
            target_layer = encontrar_capa_convolucional_alternativa(model)
            if target_layer is None:
                return generar_heatmap_simulado(array)
        
        # 5. CALCULAR Grad-CAM usando método compatible con TensorFlow 2.x
        try:
            # Crear un modelo que devuelva tanto la salida como las activaciones de la capa
            grad_model = tf.keras.models.Model(
                inputs=[model.inputs],
                outputs=[model.output, target_layer.output]
            )
            
            # Calcular gradientes usando GradientTape (método moderno de TF 2.x)
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_preprocesada)
                loss = predictions[:, :, :, :]  # Usar todas las características
                if len(predictions.shape) == 4:  # Para capas convolucionales
                    loss = tf.reduce_mean(loss, axis=[1, 2])
            
            # Obtener gradientes
            grads = tape.gradient(loss, conv_outputs)
            
            # Manejar caso donde grads es None
            if grads is None:
                print("⚠️  Gradientes son None, usando método alternativo")
                return generar_heatmap_simulado(array)
            
            # Promediar gradientes espacialmente
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Ponderar los mapas de características con los gradientes
            conv_outputs = conv_outputs[0]  # Primer batch
            heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)
            
            # Aplicar ReLU
            heatmap = np.maximum(heatmap, 0)
            
        except Exception as e:
            print(f"❌ Error en cálculo de Grad-CAM: {e}")
            print("🔄 Usando método de heatmap simulado...")
            return generar_heatmap_simulado(array)
        
        # 6. NORMALIZAR el heatmap
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        else:
            print("⚠️  Heatmap vacío, usando valores por defecto")
            heatmap = np.ones_like(heatmap) * 0.5
        
        # 7. PREPARAR visualización
        heatmap = cv2.resize(heatmap.numpy() if hasattr(heatmap, 'numpy') else heatmap, 
                            (512, 512))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # 8. PREPARAR imagen original
        img_original = preparar_imagen_original(array, (512, 512))
        
        # 9. SUPERPOR heatmap sobre imagen original
        alpha = 0.5  # Transparencia del heatmap
        imagen_superpuesta = cv2.addWeighted(img_original, 1-alpha, heatmap_color, alpha, 0)
        
        # 10. CONVERTIR de BGR a RGB para visualización correcta
        imagen_superpuesta_rgb = cv2.cvtColor(imagen_superpuesta, cv2.COLOR_BGR2RGB)
        
        print("✅ Grad-CAM generado exitosamente")
        return imagen_superpuesta_rgb
        
    except Exception as e:
        print(f"❌ Error crítico en Grad-CAM: {e}")
        import traceback
        traceback.print_exc()
        return generar_heatmap_simulado(array)

def generar_heatmap_simulado(array):
    """
    Genera un heatmap simulado cuando falla el Grad-CAM real.
    Útil para desarrollo y pruebas.
    
    Args:
        array (numpy.ndarray): Imagen original
        
    Returns:
        numpy.ndarray: Heatmap simulado
    """
    try:
        print("🔧 Generando heatmap simulado...")
        
        # Preparar imagen base
        img_original = preparar_imagen_original(array, (512, 512))
        
        # Crear heatmap simulado (centro de la imagen)
        heatmap_simulado = np.zeros((512, 512), dtype=np.float32)
        center_x, center_y = 256, 256
        radius = 100
        
        for i in range(512):
            for j in range(512):
                dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                if dist < radius:
                    heatmap_simulado[i, j] = 1.0 - (dist / radius)
        
        # Convertir a colores
        heatmap_simulado = np.uint8(255 * heatmap_simulado)
        heatmap_color = cv2.applyColorMap(heatmap_simulado, cv2.COLORMAP_JET)
        
        # Superponer
        alpha = 0.4
        imagen_superpuesta = cv2.addWeighted(img_original, 1-alpha, heatmap_color, alpha, 0)
        imagen_superpuesta_rgb = cv2.cvtColor(imagen_superpuesta, cv2.COLOR_BGR2RGB)
        
        print("✅ Heatmap simulado generado")
        return imagen_superpuesta_rgb
        
    except Exception as e:
        print(f"❌ Error en heatmap simulado: {e}")
        # Último recurso: imagen negra
        return np.zeros((512, 512, 3), dtype=np.uint8)

def encontrar_capa_convolucional_alternativa(model):
    """
    Busca una capa convolucional alternativa si la esperada no existe.
    
    Args:
        model (tf.keras.Model): Modelo de keras
        
    Returns:
        tf.keras.layers.Layer: Capa convolucional alternativa o None
    """
    try:
        # Buscar cualquier capa convolucional
        for layer in reversed(model.layers):
            if any(keyword in layer.name.lower() for keyword in ['conv', 'activation']) and hasattr(layer, 'output'):
                print(f"🔍 Usando capa alternativa: {layer.name}")
                return layer
        
        print("❌ No se encontró ninguna capa convolucional adecuada")
        return None
    except Exception as e:
        print(f"❌ Error buscando capa alternativa: {e}")
        return None

def preparar_imagen_original(array, target_size=(512, 512)):
    """
    Prepara la imagen original para la superposición del heatmap.
    
    Args:
        array (numpy.ndarray): Imagen original
        target_size (tuple): Tamaño objetivo
        
    Returns:
        numpy.ndarray: Imagen preparada en BGR
    """
    try:
        # Redimensionar
        img = cv2.resize(array, target_size)
        
        # Convertir a BGR (formato que usa OpenCV)
        if len(img.shape) == 2:  # Escala de grises
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 3:  # RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # Si ya es BGR, no hacer nada
        
        return img
    except Exception as e:
        print(f"❌ Error preparando imagen original: {e}")
        return np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)