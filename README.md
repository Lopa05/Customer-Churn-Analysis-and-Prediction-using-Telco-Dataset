# 📊 Customer Churn Prediction System

##  Overview
Customer churn is a critical problem in industries like telecom, banking, and SaaS — where retaining existing customers is often more profitable than acquiring new ones.  

This project builds a **Customer Churn Prediction System** using **machine learning and deep learning techniques**. It leverages the **Telco Customer Churn dataset** to analyze customer behavior, identify churn risks, and provide actionable insights.  

The project also includes an **interactive Streamlit app** for real-time predictions and visualization.  

---

##  Tools and Frameworks Used
- **Programming Language**: Python 3.11  
- **Data Handling & Processing**:  
  - Pandas, NumPy, zipfile  
- **Data Visualization**:  
  - Matplotlib, Seaborn, Plotly (Express & Graph Objects)  
- **Machine Learning & Deep Learning**:  
  - Scikit-learn (scaling, metrics, evaluation)  
  - PyTorch (neural networks, model training)  
- **Model Training Utilities**:  
  - torch.utils.data (DataLoader, TensorDataset)  
- **Deployment / Web App**:  
  - Streamlit (for interactive dashboards and predictions)  

---

##  Project Workflow
1. **Data Loading & Preprocessing**  
   - Load the Telco Customer Churn dataset (CSV/XLSX).  
   - Handle missing values and categorical encoding.  
   - Scale features with `MinMaxScaler`.  

2. **Exploratory Data Analysis (EDA)**  
   - Visualize churn distribution.  
   - Plot customer demographics and contract types.  
   - Generate correlation heatmaps.  

3. **Feature Engineering**  
   - Convert categorical features into numerical.  
   - Normalize values for neural networks.  

4. **Model Building**  
   - Traditional ML baselines (logistic regression, decision trees).  
   - Deep learning model using **PyTorch**.  

5. **Model Evaluation**  
   - Confusion Matrix  
   - Precision, Recall, F1-score, Accuracy  
   - ROC Curve & AUC  

6. **Deployment**  
   - Streamlit-based web app for interactive churn prediction.  

---

## Dataset
- **Source**: [Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- **Size**: ~7,043 customers, 21 features  
- **Target Variable**: `Churn` (Yes / No)  

---

## 🚀 How to Run
### 1️. Clone this repository
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the notebook 
### 4. Launch streamlit app
```bash
streamlit run app.py
```

##  Results
- Achieved **~80.7% accuracy** on test data.  
- Balanced performance across **precision, recall, and F1-score**.  
- **ROC-AUC** indicates strong separation between churners and non-churners.  

---

##  Key Insights
- **Contract type** and **tenure** are strong predictors of churn.  
- Customers with **month-to-month contracts** and **higher charges** are more likely to churn.  
- Retention efforts should focus on **at-risk groups identified by the model**.  

