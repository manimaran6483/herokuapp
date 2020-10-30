import numpy as np
import os.path
from PIL import Image
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
import pickle

app = Flask(__name__)
model = pickle.load(open('covid_model.pkl', 'rb'))
UPLOAD_FOLDER = '/uploads'
STATIC_FOLDER = '/static    '

def api(full_path):
    data = image.load_img(full_path, target_size=(50, 50, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255

    # with graph.as_default():
    predicted = model.predict(data)
    return predicted


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def prediction():
    try:
        file = request.files['image']
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)
        indices = {0: 'COVID-19', 1: 'NORMAL', 2: 'PNEUMONIA'}
        result = api(full_name)
        print(result)
        predicted_class = np.asscalar(np.argmax(result, axis=1))
        accuracy = round(result[0][predicted_class] * 100, 2)
        label = indices[predicted_class]
        return render_template('index.html', prediction_text='File Name is {}. \n Result : {}. \nAccuracy of this Model : {}\n'.format(file.filename, label, accuracy))
    except:
        return redirect(url_for('/'))


if __name__ == "__main__":
    app.run(debug=True)
