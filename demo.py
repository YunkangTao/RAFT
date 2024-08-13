import copy
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


# convert opencv frame to pytorch tensor
def cvt_to_tensor(frame):
    return torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None].to(DEVICE)


# read a video file
def read_video(path):
    video = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)

    # concat the first frame to the end of the list
    frames = frames + [frames[0]]

    return frames


def save_flow_video(video_path, flow_list, image_list, binarazition_list, union_mask):
    """
    flow_list: List of flow images, torch.Size([1, 2, 336, 600])
    image_list: List of input images, torch.Size([1, 3, 336, 600])
    binarazition_list: List of binarazition masks, np(336, 600)
    union_mask: Union mask image, np(336, 600)
    """
    union_mask = union_mask * 255  # binarazition_list to [0, 255]
    union_mask_3channel = cv2.cvtColor(union_mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)  # (336, 600, 3)

    assert len(image_list[1:]) == len(flow_list), "The number of frames and flow images must match"

    # get the width and height of flow_list
    height, width = flow_list[0].shape[2:]

    # create the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (width * 2, height * 2))

    for image, flow, mask in zip(image_list[1:], flow_list, binarazition_list):
        """
        image: torch.Size([1, 3, 336, 600])
        flow: torch.Size([1, 2, 336, 600])
        maks: (336, 600)
        union_mask: (336, 600)
        """
        flow = flow[0].permute(1, 2, 0).cpu().numpy()  # (336, 600, 2)
        flow = flow_viz.flow_to_image(flow).astype(np.uint8)  # (336, 600, 3)
        image = image[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # (336, 600, 3)

        mask = mask * 255  # binarazition_list to [0, 255]
        mask_3channel = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)  # (336, 600, 3)

        concated_image = np.concatenate([np.concatenate([image, flow], axis=1), np.concatenate([mask_3channel, union_mask_3channel], axis=1)], axis=0)

        video_writer.write(concated_image)

    video_writer.release()


def demo_from_videos(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    videos_path = glob.glob(os.path.join(args.videos_direction_path, '*.mp4'))

    for video_path in videos_path:
        frames = read_video(video_path)

        flow_up_list = []
        image_list = []
        binarazition_list = []

        with torch.no_grad():
            for i in range(len(frames) - 1):
                image1 = cvt_to_tensor(frames[i])  # torch.Size([1, 3, 336, 596])
                image2 = cvt_to_tensor(frames[i + 1])  # torch.Size([1, 3, 336, 596])

                padder = InputPadder(image1.shape)  # Pads images such that dimensions are divisible by 8
                image1, image2 = padder.pad(image1, image2)
                image_list.append(image1)

                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)  # flow_low torch.Size([1, 2, 42, 75]); flow_up torch.Size([1, 2, 336, 600])

                # binarize flow
                binarazition_list.append(flow_viz.flow_to_mask(copy.deepcopy(flow_up)[0].permute(1, 2, 0).cpu().numpy()), args.banerization_threshold_mode)  # (336, 600)

                flow_up_list.append(flow_up)

            image_list.append(image2)

        union_mask = np.logical_or.reduce(binarazition_list).astype(int)  # (336, 600)
        number_of_zero_in_mask = np.count_nonzero(union_mask == 0)
        percentage_of_zero_in_mask = number_of_zero_in_mask / (union_mask.size)  # percentage of zero in union mask, static pixel percentage
        # join the path of video name and save flow video
        save_flow_up_path = os.path.join(args.results_direction_path, video_path.split('/')[-1].split('.')[0] + f"_{percentage_of_zero_in_mask:f}.mp4")
        save_flow_video(save_flow_up_path, flow_up_list, image_list, binarazition_list, union_mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="models/raft-things.pth", help="restore checkpoint")
    parser.add_argument('--videos_direction_path', default="assets", help="dataset for evaluation")
    parser.add_argument('--results_direction_path', default="results/binary_206", help="directory to save results")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--banerization_threshold_mode', type=str, default='medium_number', choices=['medium_number', 'average_number'], help='banerization threshold')
    args = parser.parse_args()

    demo_from_videos(args)
