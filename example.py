from joblib import dump, load
import cv2
from skimage import feature
import os
from imutils import paths
import numpy as np
from imageio import imread
from io import BytesIO
import uuid

from flask import Flask, request, jsonify, make_response
from flask_restful import Resource, Api
from flask_cors import CORS
import json
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


def predication(image_list):
    # image_list=[[image,type],[image,type]]

    result = []
    # image = None
    for image_id, image, type in image_list:
        # print(image_id, image, type)
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
        if type == 'spiral':
            pred = spiralModels['Rf']['classifier'].predict([features])
        elif type == 'wave':
            pred = waveModels['Rf']['classifier'].predict([features])
        else:
            return print('Type Error')

        r = {'image_id': image_id, 'type': type, 'disease': pred[0]}
        # r = np.tolist([image_id, type, pred[0]])
        result.append(r)
        # result.append(r)
    return result


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
    return typo, '0' if pred[0] == 0 else '1'


def read(testing_paths):
    # if len(testing_paths)>15:
    #     return print("Keep the data less than 15")

    # print(testing_paths)
    images = []
    for i in range(len(testing_paths)):
        if i > 3:
            break
        image = cv2.imread(testing_paths[i])
        images.append([i, image, 'spiral'])

    return images


def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()


def read_image_array(request):
    length = request.form.get('length')
    print("length here", length)


class Prediction(Resource):
    def get(self):
        testingPath = os.path.sep.join(["./spiral", "testing"])
        # print(testingPaths_1)
        testingPaths_1 = list(paths.list_images(testingPath))
        image_list = read(testingPaths_1)
        # print(image_list)
        result_images = predication(image_list)
        # res = result_images.to_json(orient='records')
        # print(testingPath)
        # print(image_list)
        # print(result_images)
        res = json.dumps(result_images, default=myconverter)
        print(res)
        return make_response(jsonify(res))


def string_to_image(base64_string):
    img_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(img_data))


def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


# def save(encoded_data):
#     print(encoded_data)
#     nparr = np.fromstring(str(base64.b64decode(encoded_data)), np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
#     return img

class P(Resource):
    def get(self):
        # data = request.data
        args = request.args
        user_id = args.get('user_id', default='', type=str)
        access_token = args.get('access_token', default='', type=str)

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

        image_file = request.form['image']
        typo = request.form["type"]
        if image_file is None:
            return make_response(jsonify({
                "error": "bad request",
                "status": 400,
                "message": "image is missing"
            }), 400)

        if typo is None:
            return make_response(jsonify({
                "error": "bad request",
                "status": 400,
                "message": "image type is missing"
            }), 400)

        if typo != 'spiral' and typo != 'wave':
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
                "error":"unknown error",
                "status":500
            }),500)
        else:
            return make_response(jsonify({
                "typo": typo,
                "disease": pred
            }), 200)



api.add_resource(Prediction, '/api')
api.add_resource(P, '/test')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(debug=True, port=5000)
    # from waitress import serve
    #
    # serve(app, host="0.0.0.0", port=5000)
