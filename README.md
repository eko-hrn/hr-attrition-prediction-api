# hr-attrition-prediction-api

End-to-end machine learning project untuk memprediksi employee attrition (resign karyawan) menggunakan HR analytics.

##  Features

* Data preprocessing & cleaning
* Handling imbalanced data (SMOTE)
* Model training (Machine Learning)
* Model evaluation
* REST API menggunakan Flask

---

##  Dataset

Dataset yang digunakan adalah IBM HR Analytics Employee Attrition Dataset yang tersedia di Kaggle
Dataset ini mencakup berbagai fitur seperti:

* Informasi demografis karyawan
* Data pekerjaan dan performa
* Faktor kepuasan kerja
* Riwayat attrition

Dataset digunakan untuk membangun model prediksi employee attrition.


##  Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Flask

---

##  Installation

```bash
git clone https://github.com/eko-hrn/hr-attrition-prediction-api.git
cd hr-attrition-prediction-api
pip install -r requirements.txt
```

---

##  Run API

```bash
python app.py
```

---

##  API Endpoint

### POST /predict

Request:

```json
{
  "Age": 30,
  "MonthlyIncome": 5000,
  "JobLevel": 2
}
```

Response:

```json
{
  "prediction": "Yes"
}
```
---

##  Model

Model yang digunakan:

* Random Forest

---

## Author

* Eko Hendrawan
