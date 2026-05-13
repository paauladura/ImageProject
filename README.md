# Proyecto de Procesamiento de Imágenes

**Autores:** Javier Arroyo | Julia Cano | Paula Durá  
**Asignatura:** Procesamiento de Imágenes  

---

## Descripción

Proyecto de procesamiento de imágenes que abarca análisis exploratorio, data augmentation, clasificación de imágenes, detección de objetos y generación de imágenes. Se trabaja con un dataset de 5 categorías (~150 imágenes por clase) aplicando técnicas de Machine Learning clásico, Deep Learning y modelos preentrenados del estado del arte.

## Dataset

| Categoría | Imágenes originales | Imágenes tras augmentation |
|-----------|:-------------------:|:--------------------------:|
| Animales  | ~150 | 500 |
| Ciudad    | ~150 | 500 |
| Comida    | ~150 | 500 |
| Naturaleza| ~150 | 500 |
| Playa     | ~150 | 500 |
| **Total** | **~750** | **2500** |

## Estructura del proyecto

```
├── 01_EDA_DataAugmentation.ipynb    # Análisis exploratorio + Data Augmentation
├── 02_ImageClassification.ipynb     # Clasificación (SVM, CNN, Transfer Learning)
├── 03_ObjectDetection.ipynb         # Detección de objetos (YOLOv8, CNN localizador)
├── 04_ImageGeneration.ipynb         # Generación de imágenes (cDCGAN, Stable Diffusion)
├── requirements.txt                 # Dependencias del proyecto
└── gen_pretrained/                  # Metadata para generación con Stable Diffusion
    └── metadata.jsonl
```

## Notebooks

### 01 — EDA y Data Augmentation
- Análisis exploratorio: distribución de clases, resoluciones, color, outliers
- Data Augmentation: 9 transformaciones (flip, rotación, brillo, contraste, crop, blur, ruido, combinadas)
- Expansión de ~150 a 500 imágenes por clase

### 02 — Clasificación de Imágenes
- **Baseline ML:** Histogramas HSV + SVM
- **CNN from scratch:** Red convolucional con data augmentation y dropout
- **Transfer Learning:** MobileNetV2 fine-tuned (mejor resultado)
- Métricas: Accuracy, Macro-F1, Confusion Matrix, Sensitivity/Specificity

### 03 — Detección de Objetos
- **YOLOv8 (preentrenado):** Detección multi-objeto en 80 clases COCO
- **CNN Localizador (from scratch):** Regresión de bounding box con pseudo-labels
- Métricas: IoU, confianza, cobertura por categoría

### 04 — Generación de Imágenes
- **cDCGAN (from scratch):** GAN condicional entrenada desde cero (64×64)
- **Stable Diffusion (preentrenado):** Generación text-to-image (512×512)
- Análisis de diversidad y mode collapse

## Instalación

```bash
# Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Registrar kernel de Jupyter
python -m ipykernel install --user --name ImageProject --display-name "Python (ImageProject)"
```

## Requisitos

- Python 3.10+
- TensorFlow 2.x
- PyTorch 2.x
- Transformers (Hugging Face)
- Ultralytics (YOLOv8)
- ~10 GB de espacio (modelos preentrenados se descargan automáticamente)

## Ejecución

Los notebooks están diseñados para ejecutarse **en orden secuencial**:

1. **Notebook 01** genera el dataset aumentado necesario para los siguientes
2. **Notebooks 02-04** pueden ejecutarse independientemente una vez completado el 01

## Tecnologías principales

| Área | Librerías |
|------|-----------|
| ML clásico | scikit-learn |
| Deep Learning | TensorFlow/Keras, PyTorch |
| Detección | Ultralytics (YOLOv8) |
| Generación | Diffusers (Stable Diffusion) |
| Visualización | Matplotlib, Pandas |
