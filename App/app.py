from flask import Flask, render_template, request
import pandas as pd
from pycaret.regression import load_model, predict_model

app = Flask(__name__)

# Load pre-trained model
model = load_model('model/optimized_price_prediction_model')

# Function to read options for each field from the CSV
def get_unique_options(column_name):
    # For simplicity, we assume the column names in the CSV match these values.
    # You should read from a CSV or a database to dynamically populate options.
    data = pd.read_csv('laptops.csv')
    return data[column_name].dropna().unique()

# Function to process and predict the price
def predict_laptop_price(laptop_data):
    input_data = pd.DataFrame([laptop_data])
    predictions = predict_model(model, data=input_data)
    predicted_price = predictions['prediction_label'][0]
    return predicted_price

@app.route('/', methods=['GET', 'POST'])
def home():
    ram_sizes = get_unique_options('RAM')
    storages = get_unique_options('Storage')
    screens = get_unique_options('Screen')
    brands = get_unique_options('Brand')
    cpus = get_unique_options('CPU')
    gpus = get_unique_options('GPU')
    storage_types = get_unique_options('Storage type')
    models = get_unique_options('Model')

    if request.method == 'POST':
        laptop_data = {
            "RAM": int(request.form.get('ram', 0)),
            "Storage": int(request.form.get('storage', 0)),
            "Screen": float(request.form.get('screen', 0.0)),
            "Brand": request.form.get('brand', ''),
            "CPU": request.form.get('cpu', ''),
            "GPU": request.form.get('gpu', ''),
            "Storage type": request.form.get('storage_type', ''),
            "Status": request.form.get('status', ''),
            "Model": request.form.get('model', '')
        }

        if not all(laptop_data.values()):
            return "Error: All fields must be selected", 400  # Return an error if any field is missing

        predicted_price = predict_laptop_price(laptop_data)

        return render_template('prediction_result.html', predicted_price=predicted_price, laptop_data=laptop_data)

    return render_template(
        'index.html',
        ram_sizes=ram_sizes,
        storages=storages,
        screens=screens,
        brands=brands,
        cpus=cpus,
        gpus=gpus,
        storage_types=storage_types,
        models=models
    )

if __name__ == '__main__':
    app.run(debug=True)
