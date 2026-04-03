# Imitating Fakespot

**Student Name:** Bhatt Deep Manish
**Student ID:** 202511015 
**Lab Assignment:** Text Vectorization and Review Classification  
**Course:** IT 549 Deep Learning

---

## Objective

The objective of this lab is to introduce students to fundamental Natural Language Processing (NLP) techniques for representing textual data numerically and applying machine learning models for classification tasks. Students will work with user reviews from an e-commerce platform spanning multiple product categories and will perform both multiclass and binary classification.

---

## Dataset Description

The dataset contains user reviews from an e-commerce platform covering 10 product categories.  
Each review includes:
- Review text  
- Product category label  
- Review authenticity label (fake or genuine)

**Dataset Source:**  
- fake_reviews_dataset.csv

## Tasks Performed

### 1. Product Category Classification
- Preprocessed review text
- Converted text into numerical vectors using spaCy word embeddings
- Trained a multiclass neural network classifier using PyTorch
- Evaluated the model using accuracy, precision, recall, F1-score, and confusion matrix

### 2. Fake Review Detection
- Trained a binary classification model to classify reviews as fake or genuine
- Evaluated performance using standard classification metrics

### 3. Word Cloud Generation
- Identified frequently occurring words in fake reviews
- Generated a word cloud to visualize influential terms
- Interpreted patterns indicating fake reviews category wise

---

## Technologies Used

- Python  
- spaCy  
- PyTorch  
- scikit-learn  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- WordCloud  

---

## Repository Structure
├── main.ipynb
├── fake_reviews_dataset.csv
└── README.md


---

## How to Run

1. Clone the repository
2. Install required dependencies
3. Run `main.ipynb` to execute preprocessing, training, evaluation, and visualization steps

---

## Conclusion

This assignment demonstrates a complete NLP pipeline including text preprocessing, vectorization using word embeddings, neural network-based classification, performance evaluation, and interpretability through word cloud analysis.

---
