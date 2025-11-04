# 💳 Credit Card Fraud Detection

This is my **first Machine Learning project**, where I developed and evaluated two baseline models for **credit card fraud detection**:

1. **Custom (from-scratch) K-Nearest Neighbors (KNN)** implemented manually and optimized using **FAISS** for efficient nearest-neighbor search.  
2. **Decision Tree Classifier** used as a baseline model for comparison and interpretability.


The project focuses on understanding the challenges of **imbalanced data**, the importance of **feature scaling**, and evaluating model performance using appropriate metrics such as **F1-score** and **confusion matrix**.

---

## 📌 Project Overview

Credit card fraud detection is a classic binary classification problem where the fraudulent transactions are extremely rare compared to legitimate ones.  
The dataset used contains over **280,000 transactions**, with only about **0.17%** being fraud cases.

In this project:
- I **implemented a KNN model from scratch**, using **FAISS** for efficient nearest-neighbor searches.
- I trained a **Decision Tree** as a baseline comparison model.
- I handled **class imbalance** using **SMOTE**.
- I explored the effect of **feature scaling** on model performance.
- I evaluated both models using **accuracy**, **F1-score**, and **confusion matrices**.

---

## 🧰 Tools & Technologies

| Category | Libraries / Tools |
|-----------|-------------------|
| **Language** | Python |
| **Data Handling** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Machine Learning (Scikit-learn)** | DecisionTreeClassifier, GridSearchCV, train_test_split, StratifiedKFold, StandardScaler, metrics (accuracy_score, f1_score, confusion_matrix) |
| **Similarity Search** | FAISS |
| **Imbalanced Data Handling** | imbalanced-learn (SMOTE) |

---

## ⚙️ Features Implemented

- **Exploratory Data Analysis (EDA)** - visualizing data distributions and class imbalance.  
- **Custom KNN Implementation** - built from scratch with FAISS acceleration.  
- **Decision Tree Baseline Model** - implemented and visualized with `plot_tree()`.  
- **SMOTE Oversampling** - to balance minority (fraud) class.  
- **Feature Scaling** - standardized numeric features for distance-based models.  
- **Model Evaluation** - compared performance using F1-score, accuracy, and confusion matrices.  
- **Hyperparameter Tuning** - optimized `k` (neighbors) for KNN and `max_depth` for Decision Tree using GridSearchCV.  

---

## Why FAISS Was Used Instead of Scikit-Learn K-NN

The **scikit-learn K-NN** implementation works well for small to medium datasets but becomes **computationally expensive** for large, high-dimensional datasets like this credit card fraud dataset (~284,000 samples × 30 features).

To overcome this, I used **FAISS (Facebook AI Similarity Search)**, a powerful library designed for **fast nearest neighbor searches** on large datasets.

### Benefits of Using FAISS:
1. **Speed:**  
   FAISS is written in C++ with GPU support, providing **significantly faster distance computations** than scikit-learn’s pure Python implementation.  

2. **Scalability:**  
   It handles **hundreds of thousands or even millions of vectors** efficiently, which is ideal for real-world fraud detection datasets.  

3. **Memory Efficiency:**  
   FAISS uses optimized indexing structures to **reduce memory overhead** while maintaining high accuracy.  

4. **Flexibility:**  
   It integrates seamlessly with NumPy arrays, allowing easy use in **custom ML models** like my scratch KNN.

### How It Was Used Here:
In my **scratch KNN**, FAISS handles the **nearest-neighbor search** step (computing distances and retrieving top `k` closest samples), while the rest such as **majority voting**, **label prediction**, and **evaluation** are implemented manually.

This hybrid approach keeps the model educational and customizable, while achieving **practical training and inference speed**.

---

## 📊 Results Summary

| Metric | KNN (Unscaled) | KNN (Scaled) | Decision Tree |
|--------|----------------|--------------|---------------|
| **Accuracy** | 76.76% | **99.87%** | 98.95% |
| **F1-Score** | 0.007 | **0.678** | 0.219 |
| **Correct Fraud Detections** | 47 | 80 | **84** |
| **Missed Fraud Cases** | 51 | 18 | **14** |

**Key Takeaways:**
- Feature scaling dramatically improved KNN performance.  
- Decision Tree is more robust but slightly less balanced on F1-score.  
- SMOTE helped improve recall on the minority class.
