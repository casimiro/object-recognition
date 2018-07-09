from darkflow.net.build import TFNet
from flask import Flask
from flask import request
import json
import numpy as np
from scipy.misc import imread

options = {
    "model": "cfg/yolov2-tiny-voc.cfg",
    "load": "bin/yolov2-tiny-voc.weights",
    "threshold": 0.1
}
tfnet = TFNet(options)
app = Flask(__name__)


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


@app.route("/", methods=['POST', 'GET'])
def detect():
    if 'image' in request.files:
        image = imread(request.files['image'])
        return json.dumps(tfnet.return_predict(image), cls=NumpyEncoder)
    else:
        return 'got no image'
