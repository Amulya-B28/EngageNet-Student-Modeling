# 🧠 EngageNet: Deep Sequential Modeling of Student Engagement and Performance  
### Using OULAD and Synthetic Indian Data

**EngageNet** is a hybrid deep learning and ensemble-based framework developed to predict student **engagement levels** and **academic performance** across two educational contexts: the widely studied **OULAD (UK)** dataset and a synthetically generated **Indian student dataset**. It leverages both **temporal behavior modeling** and **structured feature learning** to support early identification of at-risk students and drive education analytics.

---

## 🎯 Research Motivation

In India’s diverse and digitally evolving educational ecosystem, traditional models often fail to capture behavioral and infrastructural disparities such as access to internet, tuition support, and device usage. EngageNet aims to:

- Model **temporal learning behavior** through weekly interaction data.
- Combine **demographics**, **academic history**, and **infrastructure access** in a unified prediction pipeline.
- Enable interpretable and generalizable **multi-class classification** of both engagement and academic outcomes.

The proposed pipeline has two distinct stages:
1. **Engagement Prediction** using CNN + BiLSTM + Attention.
2. **Performance Prediction** using a stacked ensemble of ML classifiers, enhanced with deep-learned engagement outputs.

---

## 🗂️ Dataset Overview

### 1️⃣ OULAD Dataset (UK)
- Open University Learning Analytics Dataset (public)
- Features: VLE clickstream, assessments, student info

### 2️⃣ Synthetic Indian Dataset
- Custom dataset modeled on OULAD schema
- Additional features for the Indian context:
  - `region`, `school_board`, `device_type`, `internet_access`, `data_recharge_frequency`, `parental_education`, etc.

### ✨ Feature Categories:
- **Demographics**: `gender`, `age_band`, `region`, `school_board`, `parental_education`, `disability`
- **Academic History**: `studied_credits`, `num_of_prev_attempts`, `has_private_tuition`
- **Behavioral**: `daily_study_hours`, `attendance_rate`, `data_recharge_frequency`
- **Infrastructure**: `internet_access`, `device_type`
- **Targets**:
  - `engagement_level` → {High, Moderate, Low}
  - `performance` → {High, Medium, Low}

---

## 📐 System Architecture

### 1. 📘 **Engagement Prediction Module** (Deep Learning)
- **Input**: Weekly behavioral interaction vectors (reshaped as 2D time-series).
- **Model Architecture**:
  - `Conv1D` layer to extract local temporal patterns.
  - `Bidirectional LSTM` to capture forward and backward dependencies.
  - `Attention Layer` to assign contextual weights to important time steps.
  - `Dense` layer with `softmax` activation for classification.
- **Output**: Probability distribution over engagement levels.

### 2. 📗 **Performance Prediction Module** (Stacked Ensemble)
- **Input**: Combination of:
  - Demographic, academic, behavioral features.
  - **Engagement prediction** scores (softmax outputs).
- **Base Learners**:
  - `XGBoost`
  - `LightGBM`
  - `Random Forest`
  - `Logistic Regression`
- **Meta-Classifier**:
  - `Logistic Regression`
- **Stacking Strategy**: Cross-validated meta-features passed to meta-classifier.

---

## 🔬 Methodology Summary

| Stage          | Task                             | Method                      |
|----------------|----------------------------------|-----------------------------|
| Preprocessing  | Data cleaning and encoding       | `pandas`, `scikit-learn`   |
| Feature Design | Normalize + one-hot encode       | `StandardScaler`, OHE      |
| Engagement     | Sequential behavior modeling     | CNN + BiLSTM + Attention   |
| Performance    | Combined classifier pipeline     | StackingClassifier         |

---

## 📊 Results: Cross-Dataset Comparative Evaluation

To evaluate model generalizability, EngageNet was trained and tested on both OULAD and Indian synthetic datasets.

### 📈 Accuracy Comparison Table

| Dataset                          | Engagement Model Accuracy (%) | Stacked Performance Model Accuracy (%) |
|----------------------------------|-------------------------------|-----------------------------------------|
| **OULAD (UK)**                   | 96.02%                        | **91.10%**                              |
| **Synthetic Data (Indian Context)** | 96.10%                        | **99.93%**                              |

### 🔍 Key Insights

- Engagement prediction is robust across contexts (~96%).
- Indian dataset outperforms OULAD in final performance prediction due to richer contextual features.
- Engagement output significantly boosts downstream accuracy when used as a meta-feature.

---

