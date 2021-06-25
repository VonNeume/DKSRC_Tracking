import numpy as np
import cv2
import torch

def _crop_and_resize(image, box, out_size, pad_color, padding=0):
    # convert box to corners (0-indexed)
    size = np.round([box[3], box[2]])
    corners = np.concatenate((
        np.round([box[1], box[0]]),
        np.round([box[1], box[0]]) + size))
    corners = np.round(corners).astype(int)

    # pad image if necessary
    pads = np.concatenate((
        -corners[:2], corners[2:] - image.shape[:2]))
    npad = max(0, int(pads.max()))
    if npad > 0:
        image = cv2.copyMakeBorder(
            image, npad, npad, npad, npad,
            cv2.BORDER_ISOLATED, value=pad_color)

    # crop image patch
    corners = (corners + npad).astype(int)
    patch = image[max(0, corners[0]-padding):corners[2]+padding, max(0, corners[1]-padding):corners[3]+padding]

    # resize to out_size
    patch = cv2.resize(patch, (out_size[0], out_size[1]))
    # import matplotlib.pyplot as plt
    # plt.imshow(patch)
    # plt.show()
    return patch

def crop_samples(img, out_size, rects, transform):
    w, h = out_size
    samples = torch.zeros((len(rects), 3, w, h))

    for i in range(len(rects)):
        r = rects[i, :]
        pdcolor = (img[:, :, 0].mean(), img[:, :, 1].mean(), img[:, :, 2].mean(),)
        samples[i, :, :, :] = transform(_crop_and_resize(img, r, out_size, pad_color=pdcolor))

    return samples