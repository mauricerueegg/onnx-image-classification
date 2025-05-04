from flask import Flask, request, jsonify
from flask.helpers import send_file
import numpy as np
import onnxruntime
import cv2
import json

app = Flask(__name__,
            static_url_path='/', 
            static_folder='web')

ort_session = onnxruntime.InferenceSession("efficientnet-lite4-11.onnx")

# Dynamisch Input- und Output-Namen holen
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

# Labels laden
labels = json.load(open("labels_map.txt", "r"))

# Bildvorverarbeitung
def pre_process_edgetpu(img, dims):
    output_height, output_width, _ = dims
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')
    img -= [127.0, 127.0, 127.0]
    img /= [128.0, 128.0, 128.0]
    return img

def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    return cv2.resize(img, (w, h), interpolation=inter_pol)

def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    return img[top:bottom, left:right]

@app.route("/")
def indexPage():
    return send_file("web/index.html")    

@app.route("/analyze", methods=["POST"])
def analyze():
    content = request.files.get('0', '').read()
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = pre_process_edgetpu(img, (224, 224, 3))
    img_batch = np.expand_dims(img, axis=0)

    results = ort_session.run([output_name], {input_name: img_batch})[0]
    top_indices = reversed(results[0].argsort()[-5:])
    result_list = [{"class": labels[str(i)], "value": float(results[0][i])} for i in top_indices]

    return jsonify(result_list)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)