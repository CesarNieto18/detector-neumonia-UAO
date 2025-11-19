#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tkinter import *
from tkinter import ttk, font, filedialog
from tkinter.messagebox import askokcancel, showinfo, WARNING
import tkcap
from PIL import ImageTk, Image
import csv
import cv2
import numpy as np
import os

# ✅ CORREGIDO: Importaciones desde  estructura de módulos
from src.modulos.read_img import read_image_file
from src.modulos.integrator import predict
# ✅ CORREGIR imports


# ✅ ELIMINADO: Configuración de TensorFlow duplicada (ya está en load_model.py)
# ✅ ELIMINADO: Importaciones no utilizadas (getpass, pyautogui, img2pdf, time, pydicom, tf, K)

class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")
        
        # Configuración de UI
        self.setup_ui()
        
        # Variables de estado
        self.array = None
        self.reportID = 0
        
        # Iniciar aplicación
        self.root.mainloop()
    
    def setup_ui(self):
        """Configura todos los elementos de la interfaz gráfica"""
        # Fuente en negrita
        fonti = font.Font(weight="bold")
        
        # Configuración de ventana
        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        # ✅ MEJORADO: Labels con nombres más descriptivos
        self.lbl_imagen_original = ttk.Label(self.root, text="Imagen Radiográfica", font=fonti)
        self.lbl_imagen_heatmap = ttk.Label(self.root, text="Imagen con Heatmap", font=fonti)
        self.lbl_resultado = ttk.Label(self.root, text="Resultado:", font=fonti)
        self.lbl_cedula = ttk.Label(self.root, text="Cédula Paciente:", font=fonti)
        self.lbl_titulo = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=fonti,
        )
        self.lbl_probabilidad = ttk.Label(self.root, text="Probabilidad:", font=fonti)

        # Campos de entrada
        self.ID = StringVar()
        self.entry_cedula = ttk.Entry(self.root, textvariable=self.ID, width=10)
        
        # Áreas de texto para imágenes
        self.texto_imagen_original = Text(self.root, width=31, height=15)
        self.texto_imagen_heatmap = Text(self.root, width=31, height=15)
        self.texto_resultado = Text(self.root, width=10, height=1)
        self.texto_probabilidad = Text(self.root, width=10, height=1)

        # ✅ MEJORADO: Botones con tooltips implícitos
        self.btn_predecir = ttk.Button(
            self.root, text="Predecir", state="disabled", command=self.ejecutar_prediccion
        )
        self.btn_cargar_imagen = ttk.Button(
            self.root, text="Cargar Imagen", command=self.cargar_imagen
        )
        self.btn_borrar = ttk.Button(self.root, text="Borrar", command=self.limpiar_campos)
        self.btn_pdf = ttk.Button(self.root, text="PDF", command=self.crear_pdf)
        self.btn_guardar = ttk.Button(
            self.root, text="Guardar", command=self.guardar_resultados
        )

        # Posicionamiento de widgets
        self.posicionar_widgets()
        
        # Enfocar en campo de cédula
        self.entry_cedula.focus_set()

    def posicionar_widgets(self):
        """Posiciona todos los widgets en la ventana"""
        # Labels
        self.lbl_imagen_original.place(x=110, y=65)
        self.lbl_imagen_heatmap.place(x=545, y=65)
        self.lbl_resultado.place(x=500, y=350)
        self.lbl_cedula.place(x=65, y=350)
        self.lbl_titulo.place(x=122, y=25)
        self.lbl_probabilidad.place(x=500, y=400)
        
        # Botones
        self.btn_predecir.place(x=220, y=460)
        self.btn_cargar_imagen.place(x=70, y=460)
        self.btn_borrar.place(x=670, y=460)
        self.btn_pdf.place(x=520, y=460)
        self.btn_guardar.place(x=370, y=460)
        
        # Campos de entrada
        self.entry_cedula.place(x=200, y=350)
        self.texto_resultado.place(x=610, y=350, width=90, height=30)
        self.texto_probabilidad.place(x=610, y=400, width=90, height=30)
        self.texto_imagen_original.place(x=65, y=90)
        self.texto_imagen_heatmap.place(x=500, y=90)

    def cargar_imagen(self):
        """Carga un archivo de imagen DICOM o JPG/PNG"""
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Seleccionar imagen médica",
            filetypes=(
                ("Imágenes DICOM", "*.dcm"),
                ("Imágenes JPEG", "*.jpeg"),
                ("Imágenes JPG", "*.jpg"),
                ("Imágenes PNG", "*.png"),
            ),
        )
        
        if filepath:
            # ✅ MEJORADO: Usar función unificada que detecta automáticamente el tipo
            self.array, img2show = read_image_file(filepath)
            
            if self.array is not None and img2show is not None:
                # Mostrar imagen original redimensionada
                self.img_original = img2show.resize((250, 250), Image.LANCZOS)
                self.img_original_tk = ImageTk.PhotoImage(self.img_original)
                self.texto_imagen_original.image_create(END, image=self.img_original_tk)
                
                # Habilitar botón de predicción
                self.btn_predecir["state"] = "enabled"
                print(f"✅ Imagen cargada: {os.path.basename(filepath)}")
            else:
                showinfo(title="Error", message="No se pudo cargar la imagen seleccionada")

    def ejecutar_prediccion(self):
        """Ejecuta el modelo de predicción y muestra resultados"""
        if self.array is None:
            showinfo(title="Advertencia", message="Primero cargue una imagen")
            return
        
        print("🔮 Ejecutando predicción...")
        
        # ✅ MEJORADO: Manejo de errores en la predicción
        try:
            self.label, self.proba, self.heatmap = predict(self.array)
            
            if self.heatmap is not None:
                # Mostrar heatmap generado
                self.img_heatmap = Image.fromarray(self.heatmap)
                self.img_heatmap = self.img_heatmap.resize((250, 250), Image.LANCZOS)
                self.img_heatmap_tk = ImageTk.PhotoImage(self.img_heatmap)
                self.texto_imagen_heatmap.image_create(END, image=self.img_heatmap_tk)
                
                # Mostrar resultados de predicción
                self.texto_resultado.delete(1.0, END)
                self.texto_resultado.insert(END, self.label)
                
                self.texto_probabilidad.delete(1.0, END)
                self.texto_probabilidad.insert(END, f"{self.proba:.2f}%")
                
                print(f"✅ Predicción completada: {self.label} ({self.proba:.2f}%)")
            else:
                showinfo(title="Error", message="No se pudo generar el mapa de calor")
                
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            showinfo(title="Error", message=f"Error en el procesamiento: {str(e)}")

    def guardar_resultados(self):
        """Guarda los resultados en archivo CSV dentro de detector-neumonia-UAO/ResultadosGuardados"""
        if not hasattr(self, 'label') or self.label is None:
            showinfo(title="Advertencia", message="No hay resultados para guardar")
            return

        try:
            # Carpeta destino (usa os.path.join para evitar problemas de separadores)
            base_dir = os.path.join(os.getcwd(),  "ResultadosGuardados")
            os.makedirs(base_dir, exist_ok=True)

            # Ruta del CSV dentro de la carpeta
            csv_path = os.path.join(base_dir, "historial.csv")
            file_exists = os.path.isfile(csv_path)

            with open(csv_path, "a", newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')

                # Escribir headers si el archivo es nuevo
                if not file_exists:
                    writer.writerow(["Cédula", "Diagnóstico", "Probabilidad", "Fecha"])

                # Escribir datos (manteniendo la fecha como en tu versión original)
                writer.writerow([
                    self.entry_cedula.get(),
                    self.label,
                    f"{self.proba:.2f}%",
                    np.datetime64('now').astype(str)[:19]
                ])

            showinfo(title="Éxito", message=f"Resultados guardados en:\n{csv_path}")

        except Exception as e:
            showinfo(title="Error", message=f"Error guardando resultados: {e}")

    def crear_pdf(self):
        """Genera un PDF del reporte actual y lo guarda en detector-neumonia-UAO/ResultadosGuardados"""
        try:
            # Carpeta destino
            base_dir = os.path.join(os.getcwd(),  "ResultadosGuardados")
            os.makedirs(base_dir, exist_ok=True)

            nombre_jpg = os.path.join(base_dir, f"Reporte_{self.reportID}.jpg")
            nombre_pdf = os.path.join(base_dir, f"Reporte_{self.reportID}.pdf")

            # Capturar pantalla de la aplicación
            cap = tkcap.CAP(self.root)
            cap.capture(nombre_jpg)

            # Convertir a PDF usando PIL
            imagen = Image.open(nombre_jpg)
            imagen = imagen.convert("RGB")
            imagen.save(nombre_pdf)

            self.reportID += 1
            showinfo(title="PDF", message=f"Reporte guardado como:\n{nombre_pdf}")

        except Exception as e:
            showinfo(title="Error", message=f"Error generando PDF: {e}")



    def limpiar_campos(self):
        """Limpia todos los campos de la interfaz"""
        respuesta = askokcancel(
            title="Confirmar",
            message="¿Está seguro de que desea borrar todos los datos?",
            icon=WARNING
        )
        
        if respuesta:
            # Limpiar campos de texto
            self.entry_cedula.delete(0, END)
            self.texto_resultado.delete(1.0, END)
            self.texto_probabilidad.delete(1.0, END)
            self.texto_imagen_original.delete(1.0, END)
            self.texto_imagen_heatmap.delete(1.0, END)
            
            # Resetear variables de estado
            self.array = None
            self.btn_predecir["state"] = "disabled"
            
            showinfo(title="Éxito", message="Todos los datos han sido borrados")

def main():
    """Función principal de la aplicación"""
    app = App()
    return 0

if __name__ == "__main__":
    main()