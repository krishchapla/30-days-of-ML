# 30-days-of-ML

# Day-1

This notebook demonstrates basic **NumPy** operations using Python, ideal for beginners in data science or machine learning.

## 📘 Overview

- Creating NumPy arrays
- Element-wise array operations: addition, subtraction, multiplication, and division
- Scalar operations on arrays
- Built-in NumPy functions: `sum`, `mean`, `std`, etc.
- Simple data import with Pandas (Excel/CSV)

## 🔧 Features

- Learn vectorized operations with NumPy arrays
- Perform arithmetic and statistical operations
- Understand broadcasting and scalar manipulation
- Read a dataset using Pandas (example: COVID-19 dataset)

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Jupyter Notebook

## 🚀 How to Use
  [Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day1.ipynb)




  # 📊 Day 2 - Data Preprocessing & Exploration (30 Days of ML)

This notebook marks **Day 2** of my 30 Days of Machine Learning journey. The focus is on data preprocessing and basic exploratory data analysis (EDA) using Pandas and NumPy.

## 🧠 Key Concepts Covered

- Reading data from Excel using `pandas.read_excel()`
- Inspecting datasets (`head()`, `info()`, `describe()`)
- Handling missing values
- Basic statistical analysis
- Column operations and data filtering

## 🛠️ Tools Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib 

## 🚀 How to Use
  [Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day2.ipynb)



  # 📈 Day 3 – Regression & Classification Models (30 Days of ML)

Day 3 of my Machine Learning journey focuses on implementing **Linear Regression** and **Logistic Regression** using real-world COVID-19 data in India.

## 🧠 What This Notebook Covers

### 🔹 Linear Regression
- Predicting **Recovered Cases** based on **Total Confirmed Cases**
- Train-test split for evaluation
- Visualization of regression line
- Model evaluation using **Mean Squared Error**

### 🔹 Logistic Regression
- Classifying **High Risk** regions based on **Deaths**
- Thresholding TotalConfirmedCases to create binary labels
- Model evaluation using **Accuracy** and **Classification Report**
- Visualization of predictions

## 🛠️ Tools & Libraries
- Python  
- NumPy & Pandas  
- Matplotlib  
- Scikit-learn (LinearRegression, LogisticRegression)

## 🚀 How to Run

 [Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day3.ipynb)


# Day 4 – 30 Days of ML 🚀
Topic: Data Preprocessing & Feature Engineering
📌 What I learned:
Handling missing values using dropna()

One-hot encoding for categorical features

Feature scaling using StandardScaler

Splitting dataset into training and test sets with train_test_split

🛠️ Libraries Used:
pandas

numpy

scikit-learn

📊 Dataset:
COVID-19 dataset (India)

🔍 Workflow Summary:
Dropped null values

Converted categorical data using one-hot encoding

Scaled features to normalize data

Prepared data for modeling by splitting into train/test sets

## 🚀 How to Run

 [Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day4.ipynb)

# 📊 Day 5 - Market Analysis & Data Preprocessing

This notebook demonstrates data cleaning and preprocessing on a real-world **market analysis dataset**. It focuses on preparing data for future machine learning tasks.

## ✅ Key Concepts Covered

- Loaded Excel dataset using `pandas.read_excel()`
- Identified and filled missing values using **mean imputation**
- Applied **One-Hot Encoding** for categorical columns
- Performed **feature scaling** with `StandardScaler`

## 🛠️ Libraries Used

- Python
- Pandas
- NumPy
- Scikit-learn

## 🚀 How to Run

 [Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day5.ipynb)

# 📈 Day 6 - Stock Price Prediction using Linear Regression

On Day 6 of the #30DaysOfML challenge, I built a simple regression model to predict **stock closing prices** using historical market data.

## ✅ Key Highlights

- Loaded Excel data and parsed the 'Date' column
- Filled missing values using `interpolate()`, `ffill()`, and `bfill()` methods
- Selected features and target for prediction
- Built a **Linear Regression** model using `scikit-learn`
- Evaluated performance with **Mean Squared Error** and **R² Score**

## 🛠️ Libraries Used

- pandas
- numpy
- scikit-learn

## 📊 Dataset

- Market analysis data (Excel)

## 🚀 How to Run


[Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day6.ipynb)

# 📊 Day 7 - Understanding Data Types in Machine Learning

In this notebook, I explored the foundational concept of **data types** — a key step before any machine learning model can be built.

## ✅ What I Learned

- Created a small sample dataset using `pandas`
- Identified different types of data:
  - 📈 **Numerical Data** (e.g., integers, floats)
  - 🏷️ **Categorical Data** (e.g., color names)
  - 📄 **Text Data** (e.g., full sentences)
- Performed basic operations:
  - Mean calculation for numerical data
  - Frequency counts for categorical values
  - String length analysis for text data

## 🛠️ Libraries Used

- pandas

## 🚀 How to Run
  
[Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day7.ipynb)

# 📅 Day 8 – Train-Test Split & Cross-Validation in Machine Learning

On Day 8 of #30DaysOfML, I explored one of the most critical parts of model evaluation: how to properly split and validate datasets.

## 📂 Topics Covered

- ✅ Splitting data into training and test sets using `train_test_split`
- ✅ Evaluating models using `cross_val_score`
- ✅ Measuring model performance using **Mean Squared Error (MSE)**

## 🛠️ Libraries Used

- `pandas`, `numpy`
- `scikit-learn`: `train_test_split`, `cross_val_score`, `LinearRegression`

## 📈 Model Used

- **Linear Regression** from `sklearn.linear_model`

## 💡 Key Takeaways

- Always split your dataset to avoid overfitting.
- Use **cross-validation** to get a better estimate of model performance.
- **Mean Squared Error** is a common regression metric.

## ▶️ How to Run

[Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day8.ipynb)

# 📅 Day 9 – Model Evaluation Metrics in Classification

On Day 9 of #30DaysOfML, I explored how to evaluate classification models using key performance metrics.

---

## 📚 Topics Covered

- Accuracy Score
- Precision Score
- Recall Score
- F1 Score

---

## 🧰 Libraries Used

- `pandas`
- `numpy`
- `sklearn.metrics`

---

## 🧠 What I Did

- Created dummy `y_true` and `y_pred` lists
- Calculated:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Compared metrics to understand which fits best depending on the use case

---

## ▶️ How to Run
[Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day9.ipynb)

