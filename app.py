from flask import Flask, render_template, request, send_from_directory
import numpy as np
import tensorflow as tf
from PIL import Image
import os

app = Flask(__name__)

# Загрузите модель
model = tf.keras.models.load_model('generator_model.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # Генерация случайного шума для входа в генератор
    noise = np.random.normal(0, 1, (1, 100))
    generated_image = model.predict(noise)

    # Преобразование массива в изображение
    generated_image = (generated_image * 255).astype(np.uint8)
    img = Image.fromarray(generated_image[0])  # Извлечение первого изображения из батча

    # Сохранение изображения
    image_path = 'static/images/generated_image.png'
    img.save(image_path)

    return render_template('index.html', image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)