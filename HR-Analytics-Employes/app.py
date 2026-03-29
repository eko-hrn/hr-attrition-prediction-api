from flask import Flask, request, jsonify
import pandas as pd
import joblib
import traceback

app = Flask(__name__)

# Load file model
model = joblib.load("model_svm.pkl")
scaler = joblib.load("scaler.pkl")
fitur_model = joblib.load("fitur_model.pkl")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "API Prediksi Attrition aktif",
        "endpoints": {
            "GET /": "Cek status API",
            "POST /predict": "Prediksi 1 atau banyak data karyawan"
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        if data is None:
            return jsonify({
                "error": "Request body kosong. Kirim JSON object atau list of object."
            }), 400

        # Kalau 1 object, ubah jadi list biar seragam
        if isinstance(data, dict):
            data = [data]

        # Harus list
        if not isinstance(data, list):
            return jsonify({
                "error": "Body harus JSON object atau list of object."
            }), 400

        results = []

        for i, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                return jsonify({
                    "error": f"Item ke-{i} harus berupa JSON object."
                }), 400

            # Ubah ke DataFrame
            df_baru = pd.DataFrame([item])

            # One-hot encoding
            df_baru = pd.get_dummies(df_baru)

            df_baru = df_baru.reindex(columns=fitur_model, fill_value=0)

            # Scaling
            df_baru_scaled = scaler.transform(df_baru)

            # Prediksi
            pred = model.predict(df_baru_scaled)[0]
            proba = model.predict_proba(df_baru_scaled)[0]

            results.append({
                "input_index": i,
                "prediction_numeric": int(pred),
                "prediction_label": "Yes" if pred == 1 else "No",
                "probability_no": float(proba[0]),
                "probability_yes": float(proba[1])
            })

        return jsonify({
            "total_input": len(results),
            "predictions": results
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)