from collections import defaultdict
import cv2
from darkflow.net.build import TFNet
from flask import Flask
from flask import send_file
from flask import request
import imageio
import json
import numpy as np

options = {
    "model": "cfg/yolov2-tiny-voc.cfg",
    "load": "bin/yolov2-tiny-voc.weights",
    "threshold": 0.1
}
tfnet = TFNet(options)
app = Flask(__name__)

translations = {
    'bycicle': 'bicileta',
    'car': 'carro',
    'person': 'pessoa',
    'truck': 'caminhão'
}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def summarize_detections(detected_objects):
    summary = defaultdict(lambda: 0)
    for detected_object in detected_objects:
        summary[detected_object['label']] += 1

    text_response = 'Nesta imagem foram reconhecidos '
    if len(summary) > 2:
        for label, quantity in summary.items():
            if quantity > 1:
                text_response += '%d objetos do tipo %s, ' % (quantity, translations[label])
            else:
                text_response += '%d objeto do tipo %s, ' % (quantity, translations[label])
        text_response = text_response[:-2]
    else:
        rows = list(summary.items())
        pattern = '%d objetos do tipo %s e %d objetos do tipo %s.'
        values = (rows[0][1], translations[rows[0][0]], rows[1][1], translations[rows[1][0]])
        text_response += pattern % values

    return text_response


def tuple_coord(dict_coord):
    return (dict_coord['x'], dict_coord['y'])


@app.route("/", methods=['POST', 'GET'])
def detect():
    if 'image' in request.files:
        image = imageio.imread(request.files['image'])
        detected_objects = tfnet.return_predict(image)
        for detected in detected_objects:
            cv2.rectangle(image, tuple_coord(detected['topleft']), tuple_coord(detected['bottomright']), (0, 255, 0))

        imageio.imwrite('./output.jpg', image)

        return send_file('./output.jpg')
    else:
        return 'got no image'
