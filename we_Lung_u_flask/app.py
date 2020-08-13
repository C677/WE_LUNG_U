# import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torchvision
import torch
import json
import io
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw
import numpy as np
# from flask_bootstrap import Bootstrap

app = Flask(__name__)


# Bootstrap(app)

@app.route('/')
def we_lung_u():
    return render_template('index.html')


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/check', methods=['GET', 'POST'])
def check():
    if request.method == 'POST':
        # 파일이 없을 때
        if request.files['file'] is None:
            return render_template('check.html', title='Check', check_message_test='empty')
        else:
            f = request.files['file']
            f.save('./static/img/' + secure_filename(f.filename))
            return render_template('check.html', title='Check', result='lung cancer',
                                   check_message_test=str(f.filename), LC=80, p=5, t=5, e=10)
    else:
        return render_template('check.html', title='Check', check_message_test='empty')


@app.route('/contact')
def contact():
    return render_template('contact.html', title='Contact')


@app.route('/team')
def team():
    return render_template('team.html')


def get_instance_segmentation_model(num_classes):
    # Load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Replace the classifier with a new one, that has
    # num_classes which is user-defined
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def drawPrediction(img, prediction):
    image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    draw = ImageDraw.Draw(image)

    for element in range(len(prediction[0]["boxes"])):
        boxes = prediction[0]["boxes"][element].cpu().numpy()
        score = np.round(prediction[0]["scores"][element].cpu().numpy(), decimals=4)
        draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], outline="red", width=3)
        draw.text((boxes[0], boxes[1]), text=str(score))
    return image


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        #img = file.read()
        img = Image.open(file)
        model = get_instance_segmentation_model(4)
        model.load_state_dict(torch.load("../model/model_epoch100.pth", map_location=torch.device('cpu')))

        # Put the model in evaluation mode
        # 모델을 추론에만 사용할 것이므로, `eval` 모드로 변경합니다:
        model.eval()
        with torch.no_grad():
            prediction = model([img])
        result = drawPrediction(img, prediction)
        return result


if __name__ == "__main__":
    #  port = int(os.environ.get("PORT", 80))
    try:
        app.run(host="0.0.0.0", port=80, debug=True)
    except Exception as ex:
        print(ex)
