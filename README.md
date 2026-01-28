# Cotton Leaf Disease Classification

This project focuses on **automatic classification of cotton leaf diseases** using a combination of **deep learning feature extraction** and **ensemble machine learning** techniques.

---

## Classes

The model classifies cotton leaf images into four categories:

- Curl Virus  
- Fussarium Wilt  
- Bacterial Blight  
- Healthy  

---

## Approach

1. **Image Preprocessing**
   - Resize images to 224 × 224
   - Normalize pixel values
   - Train-test split (80% / 20%)

2. **Feature Extraction**
   - Pretrained CNN models:
     - VGG16
     - ResNet50
     - DenseNet121
   - Global Average Pooling used to extract features

3. **Feature Fusion**
   - Features from all CNN models are concatenated

4. **Classification**
   - Soft Voting Ensemble using:
     - Logistic Regression
     - Support Vector Machine (RBF)
     - Random Forest

---

## Results

- **Overall Accuracy:** 99.13%
- High precision, recall, and F1-score across all classes
- Very low false positive rates

---

## Evaluation

The following evaluation techniques are used:

- Confusion Matrix
- Classification Report
- ROC Curves and AUC
- Per-class accuracy and specificity

---

## Technologies Used

- Python  
- TensorFlow / Keras  
- Scikit-learn  
- OpenCV  
- NumPy, Pandas  
- Matplotlib, Seaborn  

---

## Project Structure

