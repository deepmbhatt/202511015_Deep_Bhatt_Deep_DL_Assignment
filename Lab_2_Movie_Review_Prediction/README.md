# GloVe Pretrained Embeddings for Movie Text Prediction  

**Student Name:** Bhatt Deep Manish  
**Student ID:** 202511015  
**Assignment:** GloVe Pretrained Embeddings for Movie Metadata Prediction  
**Course:** IT 549 Deep Learning  

---

## Objective

The objective of this assignment is to demonstrate the use of **pretrained GloVe word embeddings** for two predictive tasks using movie metadata text:

1. **Regression Task:** Predict movie `voting_average` from a single text column.
2. **Multi-Label Classification Task:** Predict movie `genre` labels from a single text column.

Additionally, the assignment includes interpretability analysis to identify:
- Most frequent words per genre  
- Least frequent words per genre  
- Genre-indicative words using TF-IDF + linear models  

The primary goal is to build a complete NLP pipeline using pretrained embeddings and evaluate performance across different text inputs.

---

## Dataset Description

**Dataset Source:**  
Movie Dataset (Kaggle)  
https://www.kaggle.com/datasets/figolm10/movie-dataset  

Only the following columns were used:

- `overview` (text input)
- `tagline` (text input)
- `keywords` (text input)
- `genre` (multi-label target)
- `voting_average` (regression target)

As per assignment instructions, experiments were conducted using **only one text column at a time** (no concatenation).

---

## Tasks Performed

---

## Task 1 – Data Preparation

- Loaded dataset and retained only allowed columns  
- Removed missing or invalid entries  
- Text preprocessing:
  - Converted text to lowercase  
  - Removed URLs, punctuation, and numbers  
  - Tokenization  
  - Optional lemmatization  
- Created reproducible train/validation/test split (70/15/15)  
- Prepared multi-label genre encoding using MultiLabelBinarizer  

---

## Task 2 – GloVe Embedding Pipeline

- Downloaded pretrained **GloVe 100D embeddings**  
- Loaded embeddings into a dictionary  
- Calculated embedding coverage:
  - % of unique dataset tokens present in GloVe  
- Constructed **document embeddings** using:
  - TF-IDF weighted averaging of GloVe word vectors  
- Maintained consistent embedding dimensionality (100D) across all experiments  

---

## Task 3 – Model A: Rating Prediction (Regression)

### Goal
Predict `voting_average` from a single text column.

### Steps
- Selected one text column at a time (`overview`, `tagline`, `keywords`)
- Generated document embeddings
- Trained a neural regression model using PyTorch
- Used MSE loss
- Evaluated using:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)

### Baseline Model
- Predicted global mean rating for all samples
- Compared neural network performance against baseline

### Experiments
Repeated experiments using at least two different single-text inputs and compared performance.

---

## Task 4 – Model B: Genre Prediction (Multi-Label Classification)

### Goal
Predict movie genres from a single text column.

### Steps
- Selected one text column at a time
- Generated TF-IDF weighted GloVe embeddings
- Built multi-label classifier using PyTorch
- Used:
  - Sigmoid activation
  - BCEWithLogitsLoss

### Evaluation Metrics
- Micro-F1 Score  
- Macro-F1 Score  
- Hamming Loss  
- Jaccard Score  

Experiments were repeated using multiple text inputs and results were compared.

---

## Task 5 – Frequent Words per Genre

For each genre:

- Computed **Top 10 most frequent content words**
- Computed **Bottom 10 least frequent words**
  - Applied minimum frequency threshold (≥ 3)

Results were presented in tabular format and interpreted to identify linguistic patterns within genres.

---

## Task 6 – Genre-Indicative Words Using TF-IDF

- Built TF-IDF features
- Trained a linear classifier (Logistic Regression per genre)
- Extracted highest positive-weight words per genre
- Reported top 10 indicative words for each genre
- Provided short interpretation explaining why those words suggest the genre

This step improved model interpretability and demonstrated feature importance analysis.

---

## Technologies Used

- Python  
- GloVe Pretrained Embeddings (100D)  
- PyTorch  
- scikit-learn  
- Pandas  
- NumPy  
- Matplotlib / Seaborn  

---

## Repository Structure

- ├── main.ipynb
- ├── movie_dataset.csv
- └── README.md


---

## How to Run

1. Clone the repository
2. Download GloVe 100D embeddings
3. Place `glove.6B.100d.txt` in the project directory
4. Install required dependencies:

```bash
pip install numpy pandas scikit-learn torch matplotlib seaborn
```

5. Run `main.ipynb` to execute:
- Data preprocessing  
- Embedding construction  
- Regression model training  
- Multi-label classification  
- Interpretability analysis  

---

## Key Learnings

- Practical usage of pretrained word embeddings  
- Importance of embedding coverage  
- TF-IDF weighted averaging vs simple averaging  
- Differences between regression and multi-label classification  
- Baseline comparison for performance validation  
- Interpretable NLP using linear model coefficients  

---

## Conclusion

This assignment demonstrates a complete NLP pipeline using pretrained GloVe embeddings for both regression and multi-label classification tasks. It highlights how pretrained embeddings can effectively represent textual movie metadata and how interpretability techniques can provide insights into genre-specific language patterns.

The experiments show how different text fields (overview, tagline, keywords) influence predictive performance and provide a comprehensive comparison across tasks.

