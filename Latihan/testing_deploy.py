from pathlib import Path
from time import perf_counter

import joblib
from flask import Flask, jsonify, request

# Simpan gbr_model.joblib di folder yang sama dengan file ini.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "gbr_model.joblib"

app = Flask(__name__)

# Memuat model sekali saat aplikasi dijalankan.
joblib_model = joblib.load(MODEL_PATH)
EXPECTED_FEATURES = getattr(joblib_model, "n_features_in_", None)


@app.get("/")
def home():
    return jsonify({
        "message": "API prediksi harga rumah aktif.",
        "endpoint": "POST /predict",
        "expected_features": EXPECTED_FEATURES,
    })


@app.post("/predict")
def predict():
    start_time = perf_counter()
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict) or "data" not in payload:
        return jsonify({"error": "Kirim body JSON dengan key 'data'."}), 400

    data = payload["data"]
    if not isinstance(data, list) or len(data) == 0:
        return jsonify({"error": "Nilai 'data' harus berupa list yang tidak kosong."}), 400

    # Izinkan satu baris fitur ditulis sebagai [fitur1, fitur2, ...],
    # atau sebagai bentuk 2D [[fitur1, fitur2, ...]].
    if not isinstance(data[0], list):
        data = [data]

    if any(not isinstance(row, list) for row in data):
        return jsonify({"error": "Format 'data' harus berupa array 2D."}), 400

    if EXPECTED_FEATURES is not None:
        wrong_rows = [i + 1 for i, row in enumerate(data) if len(row) != EXPECTED_FEATURES]
        if wrong_rows:
            return jsonify({
                "error": (
                    f"Jumlah fitur harus {EXPECTED_FEATURES} per baris. "
                    f"Jumlah fitur salah pada baris: {wrong_rows}."
                )
            }), 400

    try:
        # Pastikan semua fitur berupa angka.
        numeric_data = [[float(value) for value in row] for row in data]
        prediction = joblib_model.predict(numeric_data).tolist()
    except (TypeError, ValueError) as error:
        return jsonify({"error": f"Data tidak valid untuk model: {error}"}), 400

    latency_ms = (perf_counter() - start_time) * 1000
    # Monitoring dasar: jumlah baris dan waktu proses tercatat di terminal.
    app.logger.info(
        "Prediksi berhasil | jumlah_data=%d | latency=%.2f ms",
        len(numeric_data),
        latency_ms,
    )

    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
