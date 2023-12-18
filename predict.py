#!/usr/bin/env python
import numpy as np
import tflite_runtime.interpreter as tflite
from io import BytesIO
from urllib import request
from PIL import Image

class_labels = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(
       x, data_format=data_format, mode="caffe"
    )

def predict(url):
    path = download_image(url)
    img = prepare_image(path, (224, 224))
    x = np.array(img, dtype='float32')
    X = np.array([x])

    interpreter = tflite.Interpreter(model_path='efficinentB0_best.tflite')
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    preds = dict(zip(class_labels, preds[0]))
    max_value = max(preds[label] for label in class_labels)
    max_label = [label for label in class_labels if preds[label] == max_value]
    return max_label

def lambda_handler(event, context):
    url = event['url']
    return predict(url)
