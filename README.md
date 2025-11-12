# 🌍 Multimodal Malnutrition Detection using Deep Learning

> An **AI-based multimodal deep learning system** for early detection of **child malnutrition** by integrating **facial image analysis** using **ResNet CNNs** with **clinical and anthropometric data** through **machine learning fusion**.

---

## 📖 Overview

Malnutrition remains one of the major causes of mortality and developmental delay among children under the age of five, especially in developing countries.  
Traditional assessment methods rely on **manual anthropometric measurements** — which are time-consuming, error-prone, and require skilled healthcare workers.  

This project presents a **multimodal AI framework** that automates malnutrition detection using both **facial image cues** and **clinical data**, enabling **faster, reliable, and accessible health assessments** — even in low-resource settings.

---

## 🧠 Project Highlights

- 🔍 **Multimodal Deep Learning** – Combines image-based and clinical data for accurate prediction.  
- 🧩 **Deep Feature Extraction** – Uses **ResNet-18**, **ResNet-50**, and **ResNet-152** pretrained on **ImageNet** for facial feature embeddings.  
- 🧮 **Clinical Data Processing** – Decision Tree-based imputation for missing values and normalization for tabular data.  
- ⚙️ **Fusion Model** – Merges deep image embeddings with numerical health features before classification.  
- 🩺 **Dual-Level Classification** – Predicts both:
  - **Binary classes:** Healthy / Malnourished  
  - **Multi-class severity:** Mild / Moderate / Severe  
- 📈 **Performance:** Achieved **98.6% accuracy** using the ResNet-50 multimodal model.  
- ⚡ **Optimized Training:** Implemented early stopping, Adam optimizer, and StepLR scheduler for performance stability.  

---

## ⚙️ Tech Stack

| Category | Tools / Frameworks |
|-----------|-------------------|
| **Programming Language** | Python |
| **Deep Learning Framework** | PyTorch, TorchVision |
| **Machine Learning** | scikit-learn |
| **Libraries Used** | NumPy, Pandas, Matplotlib, tqdm |
| **CNN Architectures** | ResNet-18, ResNet-50, ResNet-152 |
| **Data Handling** | DecisionTreeClassifier, OrdinalEncoder, StandardScaler |
| **Environment** | Google Colab |

---

## 🚀 Workflow Summary

### 1️⃣ Data Collection & Preprocessing
- Facial images of children and corresponding clinical data collected.  
- Clinical data cleaned, imputed, and normalized.  
- Image data preprocessed with resizing, normalization, and augmentation.

### 2️⃣ Feature Extraction
- Facial embeddings extracted using pretrained **ResNet CNNs**.  
- Clinical features encoded and scaled numerically.  

### 3️⃣ Feature Fusion
- Deep image embeddings fused with tabular health features.  
- Fusion layer generates unified multimodal feature vectors.

### 4️⃣ Classification
- Fully connected layers classify results into:
  - Binary: Healthy / Malnourished  
  - Multi-class: Mild / Moderate / Severe

### 5️⃣ Evaluation
- Accuracy, Precision, Recall, F1-score calculated.  
- **ResNet-50 fusion model achieved ~98.6% accuracy.**

---

## 🧮 Model Details

| Model | Description | Accuracy |
|--------|--------------|-----------|
| **ResNet-18** | Lightweight CNN with residual blocks | 93.4% |
| **ResNet-50** | Balanced depth and performance | **98.6%** |
| **ResNet-152** | Deep residual CNN with bottleneck layers | 96.97% |

---

## 📊 Results

- **Binary Classification:** ~98% Accuracy  
- **Multi-Class (Severity):** High precision in Moderate & Severe cases  
- **F1-Score:** 0.94 (average across classes)

✅ The multimodal system consistently outperformed unimodal baselines, proving the effectiveness of fusion learning.

---
## 🧭 Future Scope

- 📱 **Mobile Deployment:** Integrate the trained model into an Android or cross-platform mobile application for **real-time facial malnutrition detection**, enabling quick and accessible health screening in low-resource areas.  

- 🧑‍⚕️ **Explainable AI (XAI):** Incorporate **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visually highlight facial regions that influence the model’s predictions, improving transparency and clinical interpretability.  

- 🧮 **Expanded Dataset:** Enlarge and diversify the dataset by including children from **different ethnicities, lighting conditions, and geographical regions** to enhance generalization and robustness of the model.  

- 📊 **Integration with Health Records:** Combine the multimodal model with **Electronic Health Records (EHR)** or other health management systems to provide richer contextual insights and longitudinal tracking of a child’s nutritional status.  

- 🧠 **Transformer Integration:** Extend the model architecture to include **Vision Transformers (ViT)** for enhanced feature representation, or integrate **BERT** if future datasets include **textual medical records or nutrition reports**, enabling a fully multimodal AI ecosystem.  





