from flask import Flask, jsonify,request
import io
import torch.nn as nn
import json
import torch
app = Flask(__name__)
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
use_gpu=False



def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize((400,400)),
                                        transforms.CenterCrop((300,300)),
                                        transforms.ToTensor(),])
                                        # transforms.Normalize(
                                        #     [0.485, 0.456, 0.406],
                                        #     [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)
# with open("./tmd.jpg", 'rb') as f:
#     image_bytes = f.read()
#     tensor = transform_image(image_bytes=image_bytes)
#     print(tensor)
from torchvision import models

# Make sure to pass `pretrained` as `True` to use the pretrained weights:
class VisitNet(nn.Module):
    def __init__(self):
        super(VisitNet, self).__init__()
        model = models.resnet18(True)
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7,7), stride=(2, 2), padding=(3, 3), bias=False)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        self.resnet = model

    def forward(self, x):
        out = self.resnet(x)
        return out
if use_gpu:
    model = VisitNet().cuda()
else:
    model = VisitNet()
# Since we are using our model only for inference, switch to `eval` mode:
checkPoint = torch.load('./checkpoint/transfer_resnet_two_classification_v1.pkl')
# checkPoint = torch.load('resnet18 .pth')
model.load_state_dict(checkPoint)
model.eval()  # 预测模式


imagenet_class_index = json.load(open('./imagenet_class_index.json'))
def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]
    # possibility = torch.sigmoid(outputs).cpu().detach().numpy()
    # _, predicted = torch.max(outputs, 1)  # 获取分类结果
    # classIndex_ = predicted[0]
    # return classIndex_,predicted
    # predicted_idx = str(y_hat.item())
    # return imagenet_class_index[predicted_idx]
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})

if __name__ == '__main__':
    app.run()
#%%
'''
测试代码
'''
# with open("./ECG.jpg", 'rb') as f:
#     image_bytes = f.read()
#     print(get_prediction(image_bytes=image_bytes))
#     # # tensor = transform_image(image_bytes=image_bytes)
#     # print(tensor)