# Modelo de Red Neuronal para Detección de Neumonía

## 📋 Información General

- **Nombre del modelo**: `conv_MLP_84.h5`
- **Tipo**: Red Neuronal Convolucional (CNN)
- **Propósito**: Clasificación de imágenes radiográficas de tórax
- **Precisión reportada**: 84%

## 🎯 Clases de Clasificación

El modelo clasifica las imágenes en tres categorías:

| Clase | Descripción | Etiqueta |
|-------|-------------|----------|
| 0 | Neumonía Bacteriana | `bacteriana` |
| 1 | Sin Neumonía (Normal) | `normal` |
| 2 | Neumonía Viral | `viral` |

## 🏗️ Arquitectura del Modelo

+----+---------------------+--------------------+-----------------------+---------+
| #  |         Capa        |        Tipo        |         Salida        |  Params |
+----+---------------------+--------------------+-----------------------+---------+
| 0  |       input_9       |     InputLayer     | [(None, 512, 512, 1)] |    0    |
| 1  |        conv1        |       Conv2D       |  (None, 512, 512, 16) |   160   |
| 2  |       bn_conv1      | BatchNormalization |  (None, 512, 512, 16) |    64   |
| 3  |        conv2        |       Conv2D       |  (None, 512, 512, 16) |   2320  |
| 4  |      conv1skip      |       Conv2D       |  (None, 512, 512, 16) |    32   |
| 5  |       bn_conv2      | BatchNormalization |  (None, 512, 512, 16) |    64   |
| 6  |     bn_conv1skp     | BatchNormalization |  (None, 512, 512, 16) |    64   |
| 7  |        add_16       |        Add         |  (None, 512, 512, 16) |    0    |
| 8  |   max_pooling2d_10  |    MaxPooling2D    |  (None, 255, 255, 16) |    0    |
| 9  |        conv3        |       Conv2D       |  (None, 255, 255, 32) |   4640  |
| 10 |       bn_conv3      | BatchNormalization |  (None, 255, 255, 32) |   128   |
| 11 |    activation_16    |     Activation     |  (None, 255, 255, 32) |    0    |
| 12 |        conv4        |       Conv2D       |  (None, 255, 255, 32) |   9248  |
| 13 |      conv2skip      |       Conv2D       |  (None, 255, 255, 32) |   544   |
| 14 |       bn_conv4      | BatchNormalization |  (None, 255, 255, 32) |   128   |
| 15 |     bn_conv2skp     | BatchNormalization |  (None, 255, 255, 32) |   128   |
| 16 |        add_17       |        Add         |  (None, 255, 255, 32) |    0    |
| 17 |    activation_17    |     Activation     |  (None, 255, 255, 32) |    0    |
| 18 |   max_pooling2d_11  |    MaxPooling2D    |  (None, 127, 127, 32) |    0    |
| 19 |        conv5        |       Conv2D       |  (None, 127, 127, 48) |  13872  |
| 20 |       bn_conv5      | BatchNormalization |  (None, 127, 127, 48) |   192   |
| 21 |    activation_18    |     Activation     |  (None, 127, 127, 48) |    0    |
| 22 |        conv6        |       Conv2D       |  (None, 127, 127, 48) |  20784  |
| 23 |      conv3skip      |       Conv2D       |  (None, 127, 127, 48) |   1584  |
| 24 |       bn_conv6      | BatchNormalization |  (None, 127, 127, 48) |   192   |
| 25 |     bn_conv3skp     | BatchNormalization |  (None, 127, 127, 48) |   192   |
| 26 |        add_18       |        Add         |  (None, 127, 127, 48) |    0    |
| 27 |    activation_19    |     Activation     |  (None, 127, 127, 48) |    0    |
| 28 |   max_pooling2d_12  |    MaxPooling2D    |   (None, 63, 63, 48)  |    0    |
| 29 |        conv7        |       Conv2D       |   (None, 63, 63, 64)  |  27712  |
| 30 |       bn_conv7      | BatchNormalization |   (None, 63, 63, 64)  |   256   |
| 31 |    activation_20    |     Activation     |   (None, 63, 63, 64)  |    0    |
| 32 |      dropout_3      |      Dropout       |   (None, 63, 63, 64)  |    0    |
| 33 |        conv8        |       Conv2D       |   (None, 63, 63, 64)  |  36928  |
| 34 |      conv4skip      |       Conv2D       |   (None, 63, 63, 64)  |   3136  |
| 35 |       bn_conv8      | BatchNormalization |   (None, 63, 63, 64)  |   256   |
| 36 |     bn_conv4skp     | BatchNormalization |   (None, 63, 63, 64)  |   256   |
| 37 |        add_19       |        Add         |   (None, 63, 63, 64)  |    0    |
| 38 |    activation_21    |     Activation     |   (None, 63, 63, 64)  |    0    |
| 39 |   max_pooling2d_13  |    MaxPooling2D    |   (None, 31, 31, 64)  |    0    |
| 40 |        conv9        |       Conv2D       |  (None, 31, 31, 128)  |  73856  |
| 41 |       bn_conv9      | BatchNormalization |  (None, 31, 31, 128)  |   512   |
| 42 |    activation_22    |     Activation     |  (None, 31, 31, 128)  |    0    |
| 43 |      dropout_4      |      Dropout       |  (None, 31, 31, 128)  |    0    |
| 44 |    conv10_thisone   |       Conv2D       |  (None, 31, 31, 128)  |  147584 |
| 45 |      conv5skip      |       Conv2D       |  (None, 31, 31, 128)  |   8320  |
| 46 |      bn_conv10      | BatchNormalization |  (None, 31, 31, 128)  |   512   |
| 47 |     bn_conv5skp     | BatchNormalization |  (None, 31, 31, 128)  |   512   |
| 48 |        add_20       |        Add         |  (None, 31, 31, 128)  |    0    |
| 49 |    activation_23    |     Activation     |  (None, 31, 31, 128)  |    0    |
| 50 |   max_pooling2d_14  |    MaxPooling2D    |  (None, 15, 15, 128)  |    0    |
| 51 | average_pooling2d_2 |  AveragePooling2D  |   (None, 8, 8, 128)   |    0    |
| 52 |      flatten_2      |      Flatten       |      (None, 8192)     |    0    |
| 53 |         fc1         |       Dense        |      (None, 1024)     | 8389632 |
| 54 |      dropout_5      |      Dropout       |      (None, 1024)     |    0    |
| 55 |         fc2         |       Dense        |      (None, 1024)     | 1049600 |
| 56 |         fc3         |       Dense        |       (None, 3)       |   3075  |
+----+---------------------+--------------------+-----------------------+---------+
## Especificaciones Técnicas
- **Input shape**: `(1, 512, 512, 1)` (batch, height, width, channels)
- **Output shape**: `(1, 3)` (probabilidades para 3 clases)
- **Función de pérdida**: `categorical_crossentropy`
- **Optimizador**: `adam`
- **Métrica principal**: `accuracy`