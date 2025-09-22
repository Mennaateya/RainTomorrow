# 🌦️ WeatherAUS - Machine Learning Project  

## 📌 Project Overview
I worked with the **WeatherAUS dataset** (Australian weather data) to answer a very important question:  
**Will it rain tomorrow?** ☔  

This project focuses on end-to-end Machine Learning workflow, from data cleaning to deployment in an interactive **Streamlit web app**.

---

## 🚀 Steps I Followed
- **Data Cleaning & Preprocessing**  
  - Handled missing values using **Iterative Imputer & Simple Imputer**.  
  - Encoded categorical features for model compatibility.  

- **Exploratory Data Analysis (EDA)**  
  - Visualized weather patterns & feature correlations.  

- **Feature Engineering**  
  - Created new features and handled outliers.  

- **Model Training & Evaluation**  
  - Tested multiple algorithms:  
    - Logistic Regression  
    - KNN  
    - Decision Tree  
    - Random Forest  
    - Naive Bayes  
  - Compared results using **accuracy, precision, recall, and F1-score**.  

---

## 📊 Results
- **Decision Tree** and **Random Forest** achieved **100% accuracy** on the test split ⚡.  
- Key influencing factors: **humidity, temperature, wind speed**.  
- Gained insights into the importance of:  
  - Proper handling of missing values.  
  - Impact of scaling & encoding strategies.  
  - Understanding model performance metrics.  

⚠️ Noted the need to avoid **overfitting** when models perform “too perfectly”.

---

## 💻 Streamlit Web App
I built an interactive web app with two main pages:  
1. **Prediction Page** → Users can input weather conditions and get rain predictions.  
2. **Visualization Page** → Explore weather trends through interactive plots (humidity, rainfall, wind, temperature, etc.).  

---

## 🛠 Tech Stack
- Python 🐍  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn, Plotly  
- Streamlit  

---

## 🔮 Future Improvements
- Apply regularization techniques to prevent overfitting.  
- Experiment with advanced models (XGBoost, LightGBM).  
- Deploy Streamlit app on the cloud for wider accessibility.  

---

## 🙌 Feedback
Would love to hear your thoughts or suggestions on improving the model and deployment!  
Feel free to open an issue or contribute 💡  
