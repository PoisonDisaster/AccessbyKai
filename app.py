from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)

# Muat model dan vectorizer dengan validasi
try:
    model_path = 'models/svm_model.pkl'
    vectorizer_path = 'models/vectorizer.pkl'

    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model berhasil dimuat:", model)

    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    print("Vectorizer berhasil dimuat:", vectorizer)

except Exception as e:
    print(f"Error saat memuat model atau vectorizer: {e}")
    model, vectorizer = None, None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer:
        return jsonify({"error": "Model atau vectorizer tidak berhasil dimuat."})

    try:
        # Ambil data dari JSON
        data = request.get_json()

        # Validasi data JSON
        if not data or 'text' not in data:
            return jsonify({"error": "Data tidak valid atau kosong."})

        text = data['text']

        # Validasi teks kosong
        if not text.strip():
            return jsonify({"error": "Teks tidak boleh kosong."})

        # Debug: Cetak teks asli
        print(f"Teks asli: {text}")

        # Transformasi teks menggunakan vectorizer
        transformed_text = vectorizer.transform([text])

        # Debug: Cetak fitur hasil vektorisasi
        print(f"Fitur setelah vektorisasi: {transformed_text.toarray()}")

        # Prediksi menggunakan model
        prediction = model.predict(transformed_text)

        # Debug: Cetak hasil prediksi mentah
        print(f"Hasil prediksi mentah: {prediction}")

        # Perbaiki logika untuk menangani keluaran string
        if prediction[0].lower() == 'positif':
            sentiment = 'Positif'
        elif prediction[0].lower() == 'negatif':
            sentiment = 'Negatif'
        else:
            sentiment = 'Tidak Diketahui'

        # Kembalikan hasil prediksi
        return jsonify({"sentiment": sentiment})

    except Exception as e:
        # Debug: Tampilkan error
        print(f"Error saat memproses prediksi: {e}")
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
