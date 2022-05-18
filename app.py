from joblib import dump, load
import cv2
from skimage import feature
import numpy as np

from flask import Flask, request, jsonify, make_response
from flask_restful import Resource, Api
from flask_cors import CORS

import base64
from PIL import Image
import io

app = Flask(__name__)
CORS(app)
api = Api(app)

db_users = {
    "1234": 'f7a50a548fbccdfc1d960d9f3f97ecfe241205b9a35d9fbefd562828f5e34265'
}
spiralModels = load('spiralModels.joblib')
waveModels = load('waveModels.joblib')


def quantify_image(image):
    features = feature.hog(image, orientations=9,
                           pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                           transform_sqrt=True, block_norm="L1")
    return features


def predict(image, typo):
    output = image.copy()
    # output = image
    output = cv2.resize(output, (128, 128))

    output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    output = cv2.resize(output, (200, 200))
    output = cv2.threshold(output, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    #     # # quantify the image and make predictions based on the extracted features
    features = quantify_image(output)
    pred = 0
    if typo == 'spiral':
        pred = spiralModels['Rf']['classifier'].predict([features])
    elif typo == 'wave':
        pred = waveModels['Rf']['classifier'].predict([features])
    else:
        return print('Type Error')

    # return {'type': typo, 'disease':'0' if pred[0]==0 else '1'}
    return typo, 'healthy' if pred[0] == 0 else 'disease'


def string_to_image(base64_string):
    img_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(img_data))


def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


class PredictClass(Resource):
    def post(self):
        # data = request.data
        # print("hello world")
        args = request.args
        user_id = args.get('user_id', default='', type=str)
        access_token = args.get('access_token', default='', type=str)
        # print("access token :", access_token)
        # print("user id :", user_id)

        if not db_users.__contains__(user_id):
            return make_response(jsonify({
                "error": "unauthorized",
                "status": 401,
                "message": "invalid id"
            }), 401)

        if access_token != db_users[user_id]:
            return make_response(jsonify({
                "error": "unauthorized",
                "status": 401,
                "message": "invalid token"
            }), 401)
        # print("hello world -2 ")

        image_file = None
        typo = None
        try:
            image_file = request.form['image']
            typo = request.form["type"]
        except Exception as e:
            print('ERROR :', e)
            return make_response(jsonify({
                "error": "internal server error",
                "status": 500,
                "message": "cannot access form data"
            }), 500)

        # print("hello world -3 ")
        # print("image files :",image_file)
        # print("type :", typo)
        if image_file is None:
            # print("here - image file is missing  ")
            return make_response(jsonify({
                "error": "bad request",
                "status": 400,
                "message": "image is missing"
            }), 400)

        if typo is None:
            # print("here -2 type value is missing here ")
            return make_response(jsonify({
                "error": "bad request",
                "status": 400,
                "message": "image type is missing"
            }), 400)

        if typo != 'spiral' and typo != 'wave':
            # print("here -3 type value is invalid")
            return make_response(jsonify({
                "error": "bad request",
                "status": 400,
                "message": "invalid type"
            }), 400)

        try:
            img = string_to_image(image_file)
            img_RGB = toRGB(img)
            typo, pred = predict(image=img_RGB, typo=typo)
        except Exception as e:
            message = e
            if hasattr(e, 'message'):
                message = e.message
            return make_response(jsonify({
                "massage": message,
                "error": "unknown error",
                "status": 500
            }), 500)
        else:
            return make_response(jsonify({
                "typo": typo,
                "disease": pred
            }), 200)


api.add_resource(PredictClass, '/api')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(debug=True, port=5000)
    # from waitress import serve
    #
    # serve(app, host="0.0.0.0", port=5000)
