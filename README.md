# Detector de Neumonía UAO

Este proyecto implementa una aplicación para la detección de neumonía a partir de imágenes DICOM utilizando un modelo de **Deep Learning con TensorFlow/Keras**, acompañado de una interfaz gráfica construida en **Tkinter**.

Incluye:
- Preprocesamiento de imágenes DICOM.
- Carga e inferencia de un modelo entrenado.
- Visualización de resultados y mapa de calor (Grad-CAM).
- Entorno virtual con dependencias administradas vía `requirements.txt`.
- Repositorio colaborativo con integración vía GitHub.

---

## Colaboradores
Este proyecto fue desarrollado por:
- **Cesar Augusto Nieto Russi**
- **Adriana Samira Jasbon Mutis**
- **Andrés Camilo Guerrero Heredia**


Todos los colaboradores fueron agregados oficialmente al repositorio principal.

---

## Requisitos del Proyecto
Estas son las versiones instaladas en el entorno virtual (`venv39`):

### Frameworks principales
- **TensorFlow 2.9.0**
- **Keras 2.9.0**

### Procesamiento numérico y científico
- **NumPy 1.26.4**
- **Pandas 2.3.3**
- **Matplotlib 3.9.4**

### Manejo de imágenes y DICOM
- **opencv-python 4.12.0.88**
- **pydicom 2.4.4**

### Otros paquetes relevantes
- tensorboard 2.9.1
- protobuf 3.19.6
- Pillow 11.3.0

> **Nota:** El proyecto usa NumPy 1.26.4 porque versiones superiores (2.x) son incompatibles con TensorFlow 2.9.

---

## Ejecución del Proyecto
Sigue estos pasos para correr la aplicación localmente.

### 1. Clonar el repositorio
```bash
git clone https://github.com/CesarNieto18/detector-neumonia-UAO.git
cd detector-neumonia-UAO
```

### 2️. Crear entorno virtual
```bash
py -3.9 -m venv venv39
```

### 3️. Activar entorno
```bash
venv39\Scripts\activate
```

### 4️. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 5️. Ejecutar la aplicación
```bash
python detector_neumonia.py
```

---

## 🩻 Uso de la Aplicación
1. Abrir la interfaz.
2. Cargar una imagen **DICOM (.dcm)**.
3. Visualizar la imagen y datos principales.
4. Presionar **PREDECIR**.
5. Mostrar:
   - Resultado: *Neumonía* o *Normal*.
   - Probabilidad asociada.
   - Mapa de calor mediante Grad‑CAM.

---

## Modelo de Deep Learning
El modelo usado fue preentrenado y posteriormente cargado con:
```python
model = tf.keras.models.load_model('modelo_entrenado.h5')
```
Incluye:
- Capas convolucionales.
- MaxPooling.
- Clasificación binaria con activación sigmoide.

Se implementó **Grad‑CAM** para visualización:
```python
grads = K.gradients(output, last_conv_layer.output)[0]
```

---

## Control de Versiones y Colaboración
- Se realizaron *forks* por parte de los compañeros.
- Se enviaron **pull requests (PR)** al repositorio principal.
- Se revisaron, fusionaron y cerraron ambas solicitudes.

---

## Estructura del Proyecto
```
detector-neumonia-UAO/
│── detector_neumonia.py
│── modelo_entrenado.h5
│── requirements.txt
│── README.md
│── /img
└── /data
```

---

## Licencia
UAO Este proyecto es únicamente para fines educativos.
