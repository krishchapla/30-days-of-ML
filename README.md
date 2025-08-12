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
[Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day_9.ipynb)

# 📅 Day 10 – Decision Tree Classifier

On Day 10 of #30DaysOfML, I implemented a basic **Decision Tree Classifier** using the `zoo` dataset.

---

## 📚 Topics Covered

- Decision Tree Classifier with scikit-learn
- Train/Test Split
- Model Evaluation (Accuracy)
- Tree Visualization with `plot_tree`

---

## 🧰 Libraries Used

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

---

## 🧠 What I Did

- Used `water_need` as a feature and `animal` as the target.
- Trained a `DecisionTreeClassifier`
- Evaluated accuracy on test data
- Visualized the decision tree

---

## ▶️ How to Run

[Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day10.ipynb)

# 📅 Day 11 – Predicting Water Needs for Animals

On Day 11 of **#30DaysOfML**, I worked on predicting **water requirements** for different animals using regression models.

---

## 📚 Topics Covered

- Random Forest Regressor  
- Decision Tree Regressor  
- Train/Test Split  
- Model Evaluation (R² Score, MAE)

---

## 🧰 Libraries Used

- `pandas`  
- `numpy`  
- `scikit-learn`

---

## 🧠 What I Did

- Used `animal` as a feature (after one-hot encoding) to predict `water_need`  
- Trained **RandomForestRegressor** and **DecisionTreeRegressor**  
- Compared model performance on unseen test data

---

## ▶️ How to Run

[Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day11.ipynb)

# 📅 Day 12 – Support Vector Machine Classifier

On Day 12 of #30DaysOfML, I explored the Support Vector Machine (SVM) — a powerful algorithm for classification tasks.

## 📂 Topics Covered

- ✅ Understanding the concept of **hyperplanes** and **margins**
- ✅ Using different **kernels** (Linear, Polynomial, RBF) for classification
- ✅ Visualizing decision boundaries for SVM models
- ✅ Evaluating model performance using accuracy scores

## 🛠️ Libraries Used

- `pandas`, `numpy`
- `scikit-learn`: `SVC`, `train_test_split`, `accuracy_score`
- `matplotlib`

## 📈 Model Used

- **Support Vector Classifier (SVC)** from `sklearn.svm`

## 💡 Key Takeaways

- SVMs are highly effective for both linear and non-linear classification tasks.
- The choice of **kernel** can greatly impact the decision boundary.
- Useful in various domains like **text classification**, **image recognition**, and more.

## ▶️ How to Run

[Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day12.ipynb)

# 📅 Day 13 – Categorical Naive Bayes Classifier

On Day 13 of #30DaysOfML, I explored the **Categorical Naive Bayes** algorithm — perfect for classification tasks with categorical features.

## 📂 Topics Covered
- ✅ Handling categorical data
- ✅ Splitting dataset into training & test sets
- ✅ Training a `CategoricalNB` model from scikit-learn
- ✅ Evaluating with accuracy & classification report

## 🛠️ Libraries Used
- `pandas`, `numpy`
- `scikit-learn`: `CategoricalNB`, `train_test_split`, `accuracy_score`, `classification_report`

## 📈 Model Used
- **Categorical Naive Bayes** (`sklearn.naive_bayes`)

## 💡 Key Takeaways
- Categorical Naive Bayes works best for discrete/categorical inputs.
- Great for text classification, survey data, and categorical feature-heavy datasets.
- Outputs probabilistic predictions.

## ▶️ How to Run
[Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day13.ipynb)

# 📅 Day 14 – Support Vector Machine (SVM) Classifier

On Day 14 of #30DaysOfML, I implemented Support Vector Classifiers (SVC) with Linear and RBF kernels using the dataset in the Day14 notebook.

## 📚 Topics Covered
- SVM basics: hyperplane, margin, support vectors
- Kernel tricks: Linear & RBF
- Train/Test Split
- Training and prediction with `sklearn.svm.SVC`
- Model evaluation: Accuracy, Precision, Recall, F1-score
- Visualizing decision boundaries for linear vs non-linear separation

## 🛠️ Libraries Used
- `pandas`
- `numpy`
- `scikit-learn` (`SVC`, `train_test_split`, `metrics`)
- `matplotlib` / `seaborn` (for plotting)

## 🧠 What I Did
- Preprocessed features and target from the Day14 notebook dataset.
- Trained **Linear SVC** and **RBF SVC** models.
- Used `.predict()` to generate predictions and computed Accuracy, Precision, Recall, and F1-score.
- Plotted decision boundaries to compare linear and RBF separation behavior.
- Clarified differences between SVM (supervised classification) and clustering algorithms (e.g., KMeans) — this notebook implements SVC, not KNN or clustering.

## ▶️ How to Run
[Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day14.ipynb)





