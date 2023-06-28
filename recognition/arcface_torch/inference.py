import argparse

import cv2
import numpy as np
import torch
import glob
from pathlib import Path 

from backbones import get_model

def calc_cosind_d(id_featureA, id_featureB):
    cosine_d = np.sum(id_featureA * id_featureB)
    return cosine_d

@torch.no_grad()
def inference(weight, name, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()
    feat = net(img).numpy()
    return feat

@torch.no_grad()
def calc_image_distance(weight, name, img, target_image_path):
    image_path_list = glob.glob(str(Path(target_image_path) / '*.jpg'))
    feature = inference(weight, name, img)
    for img_path in image_path_list:
        target_feature = inference(weight, name, img_path)
        distance = calc_cosind_d(feature, target_feature)
        print(img, img_path, distance)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--img', type=str, default=None)
    parser.add_argument('--target', type=str, default=None)
    args = parser.parse_args()
    if not args.tareget:
        feature = inference(args.weight, args.network, args.img)
        print(feature)
    elif args.target:
        calc_image_distance(args.weight, args.network, args.img, args.target)
