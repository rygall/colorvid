from deoldify import device
from deoldify.device_id import DeviceId
device.set(device=DeviceId.GPU0)

from deoldify.visualize import *
plt.style.use('dark_background')
torch.backends.cudnn.benchmark=True
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")

import cv2
from python_color_transfer.color_transfer import ColorTransfer
import numpy as np
from moviepy.editor import ImageSequenceClip

import argparse


def video_breakdown(path):

    # import video
    video = cv2.VideoCapture(path)

    # checks whether frames were extracted 
    success = 1

    # frames array
    frames = []

    while success: 
        # extract frame
        success, frame = video.read() 

        # add frame to frames array
        if success:
            frames.append(frame)

    video.release()
    return frames


def deoldify_smoothed(frames):

    deoldified_frames = []

    for i in range(0, len(frames)-1):

        # grab frame to deoldify
        frame = frames[i]
        
        """DEOLDIFY COLORIZATION"""
        colorizer = get_image_colorizer(artistic=True)
        render_factor=30
        colorized_frame = colorizer.get_transformed_image(frame, render_factor=render_factor, watermarked=False)

        """COLOR TRANSFER FIRST FRAME COLORS TO NEW DEOLDIFIED IMAGE"""
        if i > 0:
            # grab images
            input_image = np.array(colorized_frame)
            ref_image = deoldified_frames[0]

            # get color transfer object
            PT = ColorTransfer()

            # lab mean transfer
            img_arr = PT.lab_transfer(img_arr_in=input_image, img_arr_ref=ref_image)

            # add deoldified frame to array
            deoldified_frames.append(np.array(img_arr))

        # add first frame to array since theres no transfer possible
        else:
            # convert back to numpy array from PIL image
            colorized_frame = np.array(colorized_frame)
            deoldified_frames.append(colorized_frame)

    return deoldified_frames


def save_video(frames):
    clip = ImageSequenceClip(list(frames), fps=20)
    clip.write_videofile('result_videos/output.mp4')
                

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="testing")
    parser = argparse.ArgumentParser(description='test')
    # the hyphen makes the argument optional
    parser.add_argument('--input-video', type=str, default='test_video.mp4', help='specify the name of the input video')
    parser.add_argument('--output-name', type=str, default='output_video.mp4', help='give a file name to the output video')

    # get path from user
    path = '/home/ryan/deoldify_smoothed/test_videos/quickclip.mp4'

    # break video into frames
    frames = video_breakdown(path=path)

    # deoldify frames
    deoldified_frames = deoldify_smoothed(frames=frames)

    # save video
    save_video(deoldified_frames)

