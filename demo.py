import sys

sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder


DEVICE = 'cuda'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
    cv2.waitKey()


# read a video file
def read_video(path):
    video = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)

    return frames


def save_flow_video(video_path, flow_up_list):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (frames[0].shape[1], frames[0].shape[0]))

    for flow_up in flow_up_list:
        flow_up = flow_up[0].permute(1, 2, 0).cpu().numpy()
        flow_up = flow_viz.flow_to_image(flow_up)
        video_writer.write(np.concatenate([frames[0], flow_up], axis=0))

    video_writer.release()


def demo_from_frames(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + glob.glob(os.path.join(args.path, '*.jpg'))

        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up)


def demo_from_video(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    video_path = args.path
    frames = read_video(video_path)

    flow_low_list = []
    flow_up_list = []

    with torch.no_grad():
        for i in range(len(frames) - 1):
            image1 = load_image(frames[i])
            image2 = load_image(frames[i + 1])

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            flow_up_list.append(flow_up)
            flow_low_list.append(flow_low)
            # viz(image1, flow_up)

    save_flow_video(video_path, flow_up_list)
    save_flow_video(video_path, flow_low_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="models/raft-things.pth", help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo_from_video(args)
