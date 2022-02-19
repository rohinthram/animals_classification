from flask import Flask, render_template, request

import json
import numpy as np
from keras.models import load_model
import os
import cv2

import base64
from PIL import Image
from io import BytesIO
import requests, json
import os

import base64

from dotenv import load_dotenv
load_dotenv()


app = Flask(__name__)

classes = list(dict(json.load(open('translation.json'))).values())

def convert_base64_to_image(image_str, return_type='numpy'):
    '''
    Converts a base64 encoded image to Pillow Image or Numpy Array
    
    Args:
        image_str (str): The pure base64 encoded string of the image
        return_type (str): The type of image you want to convert it to. 
                           Choices are [ numpy | pillow ]. Default is numpy.
    Returns:
        PIL.Image or numpy.array: The converted image
    '''
    image = Image.open(BytesIO(base64.b64decode(image_str)))
    if return_type == 'numpy':
        return np.array(image)
    else:
        return image



@app.route('/classify', methods=['POST'])
def predict():
    #print(request)
    
    imgstr = request.data['img']
    img = convert_base64_to_image(imgstr)
    img = cv2.resize(img, (100, 100))
    
    img = np.reshape(img, (1, 100, 100, 3))

    model = load_model('./save/animals_classifier.h5')

    score = list(sorted(list(model.predict(img).tolist())[0]))

    return str(score)


def api_call_cellstarthub(img):
    API_KEY = os.environ.get("API_KEY")

    endpoint = "https://api.cellstrathub.com/rohinthram/sentiment-analyser"
    headers = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
    }
    
    payload = {'img':img}   
    print(payload)
    # make a get request to load the model (needed if calling api after long time)
    # print(requests.get(endpoint, headers=headers).json())

    # Send POST request to get the output
    response = requests.post(endpoint, headers=headers, data=json.dumps(payload)).json()

    #print(response)
    return response

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        img = request.files['img']
        imgstr = base64.b64encode(img.read()).decode('utf-8')

        output = api_call_cellstarthub(imgstr)
        res = output['body']['output']
        res = list(res[1:-1])
        c = res.index(max(res))
        c = classes[c]

        return render_template('home.html', output=c, log=output)

    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)

