from joblib import dump, load
import cv2
from skimage import feature
import os

from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
api = Api(app)


def quantify_image(self, image):
    features = feature.hog(image, orientations=9,
                           pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                           transform_sqrt=True, block_norm="L1")
    return features


def predication(image_list):
    # image_list=[[image,type],[image,type]]
    spiralModels = load('spiralModels.joblib')
    waveModels = load('waveModels.joblib')

    result = []
    # image = None
    for image_id, image, type in image_list:

        # output = image.copy()
        output = image
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

        result.append([image_id, type, pred[0]])

    return result


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


class Prediction(Resource):
    def get(self):
        testingPaths_1 = os.path.sep.join(["./wave", "testing"])
        image_list = read(testingPaths_1)
        result_images = predication(image_list)
        res = result_images.to_json(orient='records')


api.add_resource(Prediction, '/api')





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(debug=True)

