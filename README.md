# hr-attrition-prediction-api

An end-to-end machine learning project to predict employee attrition (employee resignation) using HR analytics.

## Features

* Data preprocessing & cleaning
* Handling imbalanced data (SMOTE)
* Machine learning model training
* Model evaluation
* REST API using Flask

---

## Dataset

The dataset used is the **IBM HR Analytics Employee Attrition Dataset** available on Kaggle.

It includes various features such as:

* Employee demographic information
* Job-related data and performance
* Job satisfaction factors
* Attrition history

This dataset is used to build a predictive model for employee attrition.

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Flask

---

## Installation

```bash
git clone https://github.com/eko-hrn/hr-attrition-prediction-api.git
cd hr-attrition-prediction-api
pip install -r requirements.txt
```

---

## Run API

```bash
python app.py
```

---

## API Endpoint

### POST /predict

**Request:**

```json
{
  "Age": 30,
  "MonthlyIncome": 5000,
  "JobLevel": 2
}
```

**Response:**

```json
{
  "prediction": "Yes"
}
```

---

## Model

The model used:

* Random Forest

---

## Author

* Eko Hendrawan
