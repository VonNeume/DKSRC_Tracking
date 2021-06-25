import os
import random
import cv2
import numpy as np
import torch
from tracker.DKSRCTracker import DKSRCTracker
from utils.img_tools import draw_img

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(123)

seq_name = 'data/OTB/Basketball/img/'
img_list = os.listdir(seq_name)
img_list.sort()

tracker = DKSRCTracker()
tracker.init(cv2.imread(os.path.join(seq_name, img_list[0])), np.array([198,214,34,81]))


for i in range(1, len(img_list)):
    img = cv2.imread(os.path.join(seq_name, img_list[i]))

    res = tracker.update(img)

    im = draw_img(img, res, idx=i+1)

    cv2.imwrite("results/{}.jpg".format(i), im)
    cv2.imshow("Test", im)
    cv2.waitKey(10)

