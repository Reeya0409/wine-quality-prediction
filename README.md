🍷 WINE QUALITY PREDICTION USING MACHINE LEARNING

This project predicts whether a red wine is of **Good** or **Bad** quality based on its chemical properties using Machine Learning.
It includes Exploratory Data Analysis (EDA), model comparison, and a Streamlit web application for real-time predictions.

---

📁 PROJECT STRUCTURE

wine_quality/

* app.py
* requirements.txt
* winequality-red.csv
* Wine Quality Classification by Scikit-Learn_.ipynb
* README.md

---

📌 PROJECT OVERVIEW

* Dataset: Red Wine Quality Dataset
* Task: Wine quality classification

Output Classes

* Good → Quality >= 7

* Bad → Quality < 7

* Best performing model: Random Forest Classifier

* Interface: Streamlit Web Application

---

🧪 DATASET INFORMATION

The dataset contains 12 columns, including the target variable quality.

Features

* Fixed Acidity
* Volatile Acidity
* Citric Acid
* Residual Sugar
* Chlorides
* Free Sulfur Dioxide
* Total Sulfur Dioxide
* Density
* pH
* Sulphates
* Alcohol

Target Variable

* Quality (score between 3 and 8)

Dataset Source

* [https://www.kaggle.com/datasets/yasserh/wine-quality-dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)

---

📊 EXPLORATORY DATA ANALYSIS (EDA)

The following steps were performed in the Jupyter Notebook:

* Checked dataset shape, columns, and null values
* Generated statistical summaries
* Visualized data distributions
* Created correlation heatmaps
* Applied log transformation to reduce skewness
* Identified features influencing wine quality
* Handled class imbalance using SMOTE
* Compared multiple machine learning models

---

🤖 MACHINE LEARNING MODELS USED

* Logistic Regression
* Support Vector Machine (SVM)
* Decision Tree Classifier
* Random Forest Classifier

✅ Random Forest Classifier achieved the highest accuracy (~86%)

---

🚀 STREAMLIT WEB APPLICATION

The Streamlit app allows users to:

* Enter wine chemical properties using sliders
* Predict wine quality (Good or Bad)
* View prediction confidence
* Check model accuracy

---

▶️ HOW TO RUN THE APPLICATION

Step 1: Install dependencies

* pip install -r requirements.txt

Step 2: Run the Streamlit app

* streamlit run app.py

---

🛠️ TECHNOLOGIES USED

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib
* Seaborn

---

📈 MODEL TRAINING SUMMARY

* Data preprocessing using StandardScaler
* Train-test split: 80% training / 20% testing
* Random Forest model trained for classification
* Model accuracy displayed inside the application

---

📚 REFERENCES

* [https://www.kaggle.com/code/yasserh/wine-quality-prediction-comparing-top-ml-models](https://www.kaggle.com/code/yasserh/wine-quality-prediction-comparing-top-ml-models)
* [https://www.kaggle.com/code/mohitgoyal522/wine-quality-data-analysis-and-prediction](https://www.kaggle.com/code/mohitgoyal522/wine-quality-data-analysis-and-prediction)
* [https://www.kaggle.com/code/nikunjmalpani/wine-quality-prediction-imbalanced-data](https://www.kaggle.com/code/nikunjmalpani/wine-quality-prediction-imbalanced-data)
* [https://medium.com/codersarts/wine-quality-prediction-with-machine-learning-2a92567ad2a](https://medium.com/codersarts/wine-quality-prediction-with-machine-learning-2a92567ad2a)

---

🎯 CONCLUSION

This project demonstrates an end-to-end machine learning workflow including data analysis, model selection, and deployment using Streamlit.
It is suitable for beginners and serves as a practical example of ML model deployment.

🥂 HAPPY LEARNING!


