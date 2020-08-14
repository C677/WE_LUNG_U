import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torchvision
import torch
from torchvision import transforms
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
    #image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    draw = ImageDraw.Draw(img)
    results = [0.0, 0.0, 0.0]

    for element in range(len(prediction[0]["boxes"])):
        boxes = prediction[0]["boxes"][element].detach().numpy()
        score = np.round(prediction[0]["scores"][element].detach().numpy() , decimals=4)
        temp = []

        label = int(prediction[0]["labels"][element].detach().numpy())

        if label == 1 : temp.extend(["covid-19", "red"])
        elif label == 2 : temp.extend(["nodule", "blue"])
        elif label == 3 : temp.extend(["cancer", "green"])
        
        if results[label-1] < round(score*100, 2) : results[label-1] = round(score*100, 2)

        if score > 0.5 :
            draw.text((boxes[0], boxes[1]-20), text=temp[0])
            draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], outline=temp[1], width=3)
            draw.text((boxes[0], boxes[1]-10), text=str(round(score*100, 2))+" %")

    img.save("./static/img/predict.png")
    return results


@app.route('/check', methods=['GET', 'POST'])
def check():
    if request.method == 'POST':
        # 파일이 없을 때
        if request.files['file'] is None:
            return render_template('check.html', title='Check', check_message_test='empty')
        else:
            file = request.files['file']
            file.save(secure_filename(file.filename))

            model = get_instance_segmentation_model(4)
            model.load_state_dict(torch.load("model5_epoch500.pth", map_location=torch.device('cpu')))

            img = Image.open(file.filename).convert('RGB')
            copy = torchvision.transforms.ToTensor()(img).unsqueeze(0)

            model.eval()
            prediction = model(copy)
            results = drawPrediction(img, prediction)
            disease = 'You are healty!'

            if sum(results) == 0 : pass
            elif results.index(max(results)) == 0 : disease = 'covid-19'
            elif results.index(max(results)) == 1 : disease = 'nodule'
            elif results.index(max(results)) == 2 : disease = 'cancer'

            os.remove("./"+file.filename)

            return render_template('check.html', title='Check', filepath="predict.png", result=disease,
                c=results[0], n=results[1], LC=results[2])
    else:
        return render_template('check.html', title='Check', check_message_test='empty')


@app.route('/contact')
def contact():
    return render_template('contact.html', title='Contact')


@app.route('/team')
def team():
    return render_template('team.html')

if __name__ == "__main__":
    #  port = int(os.environ.get("PORT", 80))
    try:
        app.run(host="0.0.0.0", port=80, debug=True)
    except Exception as ex:
        print(ex)