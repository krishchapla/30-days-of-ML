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

# 📅 Day 12 – K-Nearest Neighbors (KNN) Classifier

On Day 12 of #30DaysOfML, I explored the **K-Nearest Neighbors Classifier** — a simple yet powerful algorithm for classification tasks.

## 📂 Topics Covered
- ✅ Understanding the KNN algorithm
- ✅ Choosing the value of **K**
- ✅ Training and testing the KNN model
- ✅ Evaluating model accuracy

## 🛠️ Libraries Used
- pandas
- numpy
- scikit-learn

## 📈 Model Used
- **KNeighborsClassifier** from sklearn.neighbors

## 💡 Key Takeaways
- KNN makes predictions by looking at the **nearest neighbors** in the feature space.
- The choice of **K** can significantly impact accuracy.
- Simple to implement but can be computationally expensive for large datasets.

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

# 📅 Day 15 – Confusion Matrix in Machine Learning

On Day 15 of #30DaysOfML, I explored the **Confusion Matrix** — one of the most important tools for evaluating classification models.

## 📂 Topics Covered
- ✅ Understanding the structure of a confusion matrix  
- ✅ Interpreting **True Positive**, **True Negative**, **False Positive**, and **False Negative**  
- ✅ Calculating the confusion matrix using `sklearn.metrics.confusion_matrix`  
- ✅ Visualizing results for better interpretation  

## 🛠️ Libraries Used
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn` (for metrics and model)

## 📊 Why It Matters
A confusion matrix provides a **complete breakdown** of your model’s predictions, going beyond accuracy to reveal exactly where it’s making mistakes.

## ▶️ How to Run

[Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day15.ipynb)

# 📅 Day 16 – Exploratory Data Analysis (EDA) on Titanic Dataset

Today in #30DaysOfML, I focused on **data preprocessing and visualization** using the Titanic dataset.

## 📂 Topics Covered
- ✅ Loading CSV data with pandas
- ✅ Cleaning and renaming columns
- ✅ Handling missing values (mean, mode, drop)
- ✅ Exploratory visualizations (histograms, countplots, scatterplots, boxplots)
- ✅ Correlation analysis

## 🛠️ Libraries Used
- pandas
- numpy
- matplotlib
- seaborn

## 💡 Key Takeaways
- Data preprocessing is crucial before modeling
- Handling missing values strategically improves dataset quality
- Visualizations help uncover hidden patterns & relationships

## ▶️ How to Run
[Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day16.ipynb)


# 📅 Day 17 – ROC Curve & AUC (Logistic Regression)

On Day 17 of #30DaysOfML, I evaluated a classifier using the **ROC Curve** and **AUC** on the **Breast Cancer** dataset.

## 📂 Topics Covered
- ✅ Train/Test Split
- ✅ Probability scores with `predict_proba`
- ✅ ROC Curve (`sklearn.metrics.roc_curve`)
- ✅ Area Under Curve (AUC)
- ✅ ROC visualization with Matplotlib

## 🛠️ Libraries Used
- numpy
- scikit-learn (LogisticRegression, train_test_split, roc_curve, auc)
- matplotlib

## 🧠 What I Did
- Loaded `load_breast_cancer()` and split into train/test
- Trained **LogisticRegression**
- Computed probability scores and plotted the **ROC curve**
- Calculated **AUC** to summarize classifier performance

## ▶️ How to Run
[Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day17.ipynb)

# 📅 Day 18 – Regularization in Machine Learning (L1 & L2)

On Day 18 of #30DaysOfML, I explored how **Ridge (L2)** and **Lasso (L1)** regularization help improve model generalization and reduce overfitting.

---

## 📚 Topics Covered
- Concept of **regularization** in ML
- Ridge Regression (L2 penalty)
- Lasso Regression (L1 penalty)
- Evaluation with **accuracy comparison** between L1 & L2

---

## 🧰 Libraries Used
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

---

## 🧠 What I Did
- Loaded the **Wine dataset** from `sklearn.datasets`
- Trained models with **Ridge** and **Lasso** regularization
- Plotted **L1 vs L2 accuracy comparison**
- Observed how regularization changes model performance

---

## ▶️ How to Run
[Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day18.ipynb)

# 📅 Day 19 – Decision Tree Regressor in Machine Learning

On Day 19 of #30DaysOfML, I explored how **Decision Trees** can be applied for **regression tasks** using real-world housing data.

---

## 📚 Topics Covered
- Concept of **Decision Tree Regression**
- Splitting continuous data based on feature thresholds
- Evaluating regression models with **MSE** and **R² Score**
- Visualizing predictions vs actual values

---

## 🧰 Libraries Used
- `pandas`, `numpy`
- `matplotlib`
- `scikit-learn`: `DecisionTreeRegressor`, `train_test_split`, `metrics`

---

## 🧠 What I Did
- Loaded the **California Housing dataset**
- Built a **Decision Tree Regressor**
- Evaluated predictions using **Mean Squared Error (MSE)** and **R²**
- Visualized **predicted vs actual values** with a scatter plot

---

## ▶️ How to Run
[Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day19.ipynb)

# 📅 Day 20 – Random Forest Classifier in Machine Learning

## 📂 Topics Covered
- ✅ Introduction to Random Forests (ensemble of Decision Trees 🌲)
- ✅ Training a Random Forest Classifier on the **Breast Cancer dataset**
- ✅ Evaluating performance using:
  - Confusion Matrix
  - Classification Report
- ✅ Comparison of Random Forest vs Decision Tree performance

## 🛠️ Libraries Used
- pandas, numpy
- scikit-learn: RandomForestClassifier, train_test_split, metrics

## 📈 Model Used
- **Random Forest Classifier**

## 💡 Key Takeaways
- Random Forests reduce overfitting compared to single Decision Trees.
- They improve accuracy by combining predictions of multiple trees.
- Widely used in **healthcare, finance, and recommendation systems**.

## ▶️ How to Run
[Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day20.ipynb)

# 📅 Day 21 – Gradient Boosting Classifier

## 📚 Topics Covered
- 🌳 Gradient Boosting vs Bagging/Random Forest
- ⚡ Sequential learning with weak learners
- 📊 Confusion Matrix for classification evaluation
- 🔎 Hyperparameter tuning (n_estimators, learning_rate)

## 🛠️ Libraries Used
- pandas, numpy
- scikit-learn: GradientBoostingClassifier, confusion_matrix
- matplotlib, seaborn

## 🧠 What I Did
- Trained a GradientBoostingClassifier
- Visualized results using a confusion matrix heatmap
- Measured accuracy and classification performance

## ▶️ How to Run
[Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day21.ipynb)

# 📅 Day 22 – XGBoost Classifier with Confusion Matrix

On Day 22 of #30DaysOfML, I implemented an **XGBoost Classifier** on the Titanic dataset and visualized the results using a **Confusion Matrix**.

---

## 📚 Topics Covered

- XGBoost Classifier for tabular data
- Handling categorical + numerical features
- Model evaluation with Accuracy Score
- Confusion Matrix for classification visualization

---

## 🧰 Libraries Used

- `pandas`, `numpy`
- `xgboost`
- `scikit-learn` (train_test_split, accuracy_score, confusion_matrix, metrics)
- `matplotlib`, `seaborn`

---

## 🧠 What I Did

- Preprocessed Titanic dataset (features: Age, Sex, Pclass, etc.)
- Trained an **XGBoost Classifier** for survival prediction  
- Evaluated results using **accuracy score**  
- Plotted **Confusion Matrix** for detailed performance insight  

---

## ▶️ How to Run

[Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day22.ipynb)

# 📅 Day 23 – AdaBoost Classifier in Machine Learning

## 📂 Topics Covered
- ✅ Introduction to AdaBoost (Adaptive Boosting)
- ✅ Using weak learners (Decision Trees) with boosting
- ✅ Training and evaluating AdaBoostClassifier
- ✅ Confusion Matrix for classification performance

## 🛠️ Libraries Used
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `sklearn.ensemble` → `AdaBoostClassifier`
- `sklearn.datasets` → breast cancer dataset
- `sklearn.metrics` → accuracy_score, confusion_matrix

## 📈 Model Used
- **AdaBoost Classifier** (base: Decision Trees)

## 💡 Key Takeaways
- AdaBoost combines multiple weak learners to form a strong classifier.
- Boosting improves accuracy by focusing on misclassified points.
- Confusion Matrix gives deeper insights beyond accuracy.

## ▶️ How to Run
[Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day23.ipynb)

# 📅 Day 24 – Voting Classifier

## 📂 Topics Covered
- ✅ Concept of Voting Classifier
- ✅ Hard vs Soft Voting explained
- ✅ Implementation using sklearn’s VotingClassifier
- ✅ Model evaluation with Confusion Matrix

## 🛠️ Libraries Used
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `sklearn.datasets` → iris dataset
- `sklearn.ensemble` → VotingClassifier
- `sklearn.linear_model` → LogisticRegression
- `sklearn.svm` → SVC
- `sklearn.tree` → DecisionTreeClassifier
- `sklearn.metrics` → confusion_matrix, accuracy_score

## 📈 Model Used
- **Voting Classifier (Hard Voting)** with:
  - Logistic Regression
  - Support Vector Classifier
  - Decision Tree Classifier

## 📊 Visualization
- **Confusion Matrix** to show classification performance

## 💡 Key Takeaways
- Voting combines strengths of multiple models
- Iris dataset → perfect for multi-class classification
- Confusion matrix helps understand misclassifications beyond accuracy

## ▶️ How to Run
[Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day24.ipynb)

# 📅 Day 25 – Bagging Classifier (Ensemble Learning)

## 📂 Topics Covered
- ✅ Introduction to **Bagging (Bootstrap Aggregating)**  
- ✅ Using `BaggingClassifier` with `DecisionTreeClassifier`  
- ✅ Handling bias-variance trade-off with ensembles  
- ✅ Evaluating performance with **accuracy score**  
- ✅ Visualizing predictions using a **bar plot** (Correct vs Incorrect by class)

## 🛠️ Libraries Used
- pandas, numpy, matplotlib  
- scikit-learn: `BaggingClassifier`, `DecisionTreeClassifier`, `train_test_split`, `accuracy_score`  
- sklearn.datasets: `load_wine`

## 📊 Dataset
- **Wine dataset** from `sklearn.datasets`  
  - 178 samples, 13 features, 3 classes

## 📈 Model Used
- **BaggingClassifier** with `DecisionTreeClassifier` as base estimator  

## 💡 Key Takeaways
- Bagging reduces variance and improves stability of weak learners  
- Decision trees benefit significantly from bagging  
- Visualization helps interpret where the model performs well or struggles  

## ▶️ How to Run
[Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day25.ipynb)

# Day 26 of 30DaysOfML 🚀
## Topic: ExtraTrees Classifier (Extremely Randomized Trees)
## Dataset: sklearn.datasets.load_digits

---

# 🔹 Introduction
- ExtraTrees = an ensemble of decision trees with **more randomness** than Random Forest.
- Splits are chosen at **random thresholds**, and by default no bootstrapping is used.
- Goal: reduce variance, speed up training, and improve generalization.

---

# 🔹 Steps Implemented
1. Loaded the **Digits dataset** (8x8 images of handwritten digits).
2. Split into **train (80%)** and **test (20%)** sets.
3. Trained an **ExtraTreesClassifier**.
4. Evaluated with **accuracy score, confusion matrix, and classification report**.

---

# 🔹 Results
- **Accuracy:** 97.78%  
- Confusion matrix shows almost perfect classification across all 10 digits.  
- Precision, recall, and F1-scores are consistently high (≥0.95).  

---

# 🔹 Key Insights
- ExtraTrees trains **faster than RandomForest** while maintaining strong accuracy.  
- Randomized splits → **less overfitting, better generalization**.  
- Works very well for high-dimensional, noisy datasets.  

---
## ▶️ How to Run
[Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day26.ipynb)

# 📌 Next (Day 27): Stacking Classifier – blending multiple models into one powerful learner!

# Day 27 of 30DaysOfML 🚀
## Topic: CatBoost Classifier
## Dataset: sklearn.datasets.load_breast_cancer

---

# 🔹 Introduction
- **CatBoost** (by Yandex) is a powerful gradient boosting algorithm.  
- Designed to handle **categorical features natively**, avoid overfitting, and deliver **state-of-the-art accuracy**.  
- Built-in regularization and symmetric trees → **fast + accurate**.

---

# 🔹 Steps Implemented
1. Loaded the **Breast Cancer dataset** (malignant vs benign).  
2. Split into **train (80%)** and **test (20%)** sets.  
3. Trained a **CatBoostClassifier** with default settings.  
4. Evaluated with **classification report & confusion matrix**.

---

# 🔹 Results
- **Accuracy:** 97%  
- **Precision/Recall/F1-score:** ~0.97 across both classes.  
- **Confusion Matrix:** Only 3 misclassifications out of 114 test samples.  

---

# 🔹 Key Insights
- CatBoost delivers **robust performance** out of the box.  
- Great at handling **imbalanced or categorical-heavy datasets**.  
- Outperforms many traditional ML models in speed + accuracy.  

---

# 📌 Keywords
CatBoost, Gradient Boosting, Ensemble Learning, Sklearn, Breast Cancer Dataset, Python, Machine Learning, Classification, AI, Model Evaluation, Data Science, 30DaysOfML

## ▶️ How to Run
[Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day27.ipynb)

# 🚀 Day 28 - Model Comparison for Classification

## 📌 Overview
This notebook demonstrates how to:
- Generate a synthetic dataset using sklearn
- Perform train/test split
- Train multiple classification models
- Compare their performances

## 🛠️ Models Used
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors

## 📊 Key Steps
1. Data Generation (`make_classification`)
2. Train-Test Split
3. Model Training
4. Performance Comparison

## ▶️ How to Run
[Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day28.ipynb)

# 🌸 Day 29 - Iris Dataset Exploration

## 📌 Overview
This notebook demonstrates the first step in any ML workflow:
**Exploratory Data Analysis (EDA)**.
Using the Iris dataset, we:
- Load data from sklearn
- Convert it into a pandas DataFrame
- Map target integers to species names
- Generate scatter plot matrices to explore relationships

## 📊 Dataset
- Iris Dataset (150 samples)
- Features: sepal length, sepal width, petal length, petal width
- Target: Species (Setosa, Versicolor, Virginica)

## 📈 Visualizations
- Scatter plot matrix of all features
- Color-coded by species
- Insights into feature separability

## ▶️ How to Run
[Code Link](https://github.com/krishchapla/30-days-of-ML/blob/main/Day29.ipynb)


