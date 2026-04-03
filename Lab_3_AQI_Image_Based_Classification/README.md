# AQI Image Classification using CNN and Transfer Learning  

**Student Name:** Bhatt Deep Manish  
**Student ID:** 202511015  
**Lab Assignment:** Image-Based AQI Classification using CNN and Pretrained Models
**Course:** IT 549 Deep Learning  

---

## Objective  

The objective of this assignment is to demonstrate the use of deep learning techniques for image classification by predicting **AQI_Class** from images of different locations.  

This assignment focuses on building a complete image classification pipeline including data preprocessing, model development, evaluation, and interpretation of results.  

Two approaches are implemented and compared:  
- A Convolutional Neural Network (CNN) trained from scratch  
- A pretrained CNN model using transfer learning  

---

## Dataset Description  

The dataset consists of:  
- **data.csv** – Contains image paths and corresponding AQI class labels  
- **sampled_images/** – Collection of images belonging to different AQI categories  

Only the following fields are used:  
- **image_path** → Input image  
- **AQI_Class** → Target label  

**Dataset Link:**  
https://drive.google.com/drive/folders/1u-sBxgNB67GfhCQ2f7xRkDlF6fgIZZrP?usp=sharing  

---

## Tasks Performed  

### 1. Data Preparation  
- Loaded dataset from CSV file  
- Processed image paths and labels  
- Resized images to a fixed resolution (224 × 224)  
- Normalized pixel values  
- Split dataset into training, validation, and test sets (70/15/15)  

---

### 2. Basic CNN Model  
- Designed and implemented a CNN from scratch  
- Trained model on AQI classification task  
- Evaluated performance on test dataset  

---

### 3. Pretrained CNN Model (Transfer Learning)  
- Used pretrained models such as ResNet / VGG / EfficientNet  
- Modified final classification layer to match AQI classes  
- Applied transfer learning for training  
- Compared performance with basic CNN  

---

### 4. Model Training and Evaluation  
Both models were evaluated using:  
- Accuracy  
- Precision  
- Recall  
- F1-score  

- Generated confusion matrices for performance comparison  

---

### 5. Training Curves  
- Plotted:  
  - Epoch vs Training Loss  
  - Epoch vs Validation Loss  
  - Epoch vs Training Accuracy  
  - Epoch vs Validation Accuracy  

- Analyzed model convergence and overfitting behavior  

---

### 6. Misclassification Analysis  
- Identified 5–10 misclassified images  
- Visualized predictions vs actual labels  
- Analyzed possible reasons such as:  
  - Visual similarity between AQI classes  
  - Poor lighting or image quality  
  - Dataset imbalance  

---

## Technologies Used  

- Python  
- PyTorch / TensorFlow  
- scikit-learn  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- OpenCV / PIL  

---

## Repository Structure  
├── data.csv
├── sampled_images/
├── main.ipynb
├── models/
├── outputs/
└── README.md


---

## How to Run  

1. Clone the repository  
2. Download dataset from the provided Google Drive link  
3. Install required dependencies  
4. Run `main.ipynb` to execute:  
   - Data preprocessing  
   - Model training  
   - Evaluation  
   - Visualization  

---

## Results Summary  

- Compared performance between:  
  - CNN trained from scratch  
  - Pretrained CNN (Transfer Learning)  

- Observations:  
  - Pretrained models generally performed better due to learned feature representations  
  - Faster convergence and improved generalization  
  - Transfer learning significantly improved classification accuracy  

---

## Conclusion  

This assignment demonstrates a complete deep learning pipeline for image classification. It highlights the effectiveness of transfer learning compared to training a CNN from scratch, especially when working with limited datasets.  

The project also emphasizes the importance of proper preprocessing, evaluation metrics, and error analysis in building robust deep learning models.  

---