import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import resnet34
from concurrent.futures import ThreadPoolExecutor

def predict_image(file, dir_path, device, data_transform, class_indict, model, json_path, weights_path):
    imgs = os.path.join(dir_path, file)
    assert os.path.exists(imgs), "file: '{}' dose not exist.".format(imgs)
    img = Image.open(imgs).convert("RGB")
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    model = resnet34(num_classes=17).to(device)
    weights_path = "./resNet34.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    print("正在预测图片" + imgs +"\n"+"图片为" + file + "    " + "图片预测为" + class_indict[str(predict_cla)])

    plt.title(print_res)
    plt.savefig("./PredictType/PredictRes{}.png".format(file))

def main(dir_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # dir_path = "./SpliteImg"
    files = os.listdir(dir_path)

    json_path = './class_indices.json'
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    model = resnet34(num_classes=17).to(device)
    weights_path = "./resNet34.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))

    with ThreadPoolExecutor() as executor:
        for file in files:
            executor.submit(predict_image, file, dir_path, device, data_transform, class_indict, model, json_path, weights_path)

if __name__ == '__main__':
    main()