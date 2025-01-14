from flask import Flask, request, render_template
import pandas as pd
from pycaret.regression import load_model, predict_model

# Inicjalizacja aplikacji Flask
app = Flask(__name__)

# Załaduj wytrenowany model
model = load_model('model/optimized_price_prediction_model')


# Funkcja do przewidywania ceny laptopa
def predict_laptop_price(laptop_data):
    input_data = pd.DataFrame([laptop_data])
    predictions = predict_model(model, data=input_data)
    predicted_price = predictions['prediction_label'][0]
    return predicted_price


# Strona główna
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        laptop_data = {
            "RAM": int(request.form['ram']),
            "Storage": int(request.form['storage']),
            "Screen": float(request.form['screen']),
            "Brand": request.form['brand'],
            "CPU": request.form['cpu'],
            "GPU": request.form['gpu'],
            "Storage type": request.form['storage_type'],
            "Status": request.form['status'],
            "Model": request.form['model']
        }
        predicted_price = predict_laptop_price(laptop_data)
        predicted_price = round(predicted_price, 2)  # Zaokrąglenie do 2 miejsc po przecinku
        return render_template('prediction_result.html', predicted_price=predicted_price, laptop_data=laptop_data)

    # Dane do formularza (przykład, wczytywane z CSV)
    ram_sizes = [2, 4, 8, 16]
    storages = [128, 256, 512, 1024]
    screens = [13, 14, 15.6]
    brands = ['Dell', 'HP', 'Asus', 'Lenovo']
    cpus = ['Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'AMD Ryzen 5']
    gpus = ['Intel UHD', 'NVIDIA GTX 1650', 'NVIDIA RTX 2060']
    storage_types = ['SSD', 'HDD']
    models = ['XPS 15', 'MacBook Pro', 'Asus ROG', 'HP Spectre']

    return render_template('index.html', ram_sizes=ram_sizes, storages=storages, screens=screens,
                           brands=brands, cpus=cpus, gpus=gpus, storage_types=storage_types, models=models)


# Uruchomienie aplikacji przy użyciu Waitress
if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=5000)
