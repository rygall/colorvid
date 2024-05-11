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
from skimage.color import rgb2lab, deltaE_ciede2000
import argparse
import matplotlib.pyplot as plt



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

def calculate_lab_difference(image1, image2):
    # Convert the images to the LAB color space
    image1_lab = rgb2lab(image1)
    image2_lab = rgb2lab(image2)

    # Define a threshold to ignore white pixels (you can adjust this value)
    white_threshold = 1.00

    # Create masks for the white pixels
    mask1 = np.all(image1 > white_threshold, axis=-1)
    mask2 = np.all(image2 > white_threshold, axis=-1)

    # Create a combined mask that ignores white pixels in either image
    mask = np.logical_or(mask1, mask2)

    # Apply the mask to the LAB images
    image1_lab = image1_lab[~mask]
    image2_lab = image2_lab[~mask]

    # Calculate the CIEDE2000 color difference
    return deltaE_ciede2000(image1_lab, image2_lab)

def compare_frames(frame1, frame2, frame_num):

    # Calculate and print the average CIEDE2000 color difference
    avg_ciede2000 = np.mean(calculate_lab_difference(frame1, frame2))
    # Calculate and print the average LAB difference (min:0 , max)
    print("Frame Number: "+ str(frame_num))
    print(f'Average CIEDE2000 Color Difference = {avg_ciede2000}')
    print('-----------------------------------------------')
    return avg_ciede2000
                

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="testing")
    parser = argparse.ArgumentParser(description='test')
    # the hyphen makes the argument optional
    parser.add_argument('--input-video', type=str, default='test_video.mp4', help='specify the name of the input video')
    parser.add_argument('--output-name', type=str, default='output_video.mp4', help='give a file name to the output video')

    # get path from user
    path = 'test_videos/quickclip.mp4'

    # break video into frames
    # if color video
    #    color_frames = video_breakdown(path=path)
    #    convert_to_blackwhite(path)
    #    frames = video_breakdown(path = bwpath)
    #else: 
    frames = video_breakdown(path=path)
    
    # deoldify frames
    deoldified_frames = deoldify_smoothed(frames=frames)

    #Rob Stuff Start
    test_frames_float = np.linspace(0, len(deoldified_frames)-1, 10)

    test_frames_int = [int(round(num)) for num in test_frames_float]

    fig, axs = plt.subplots(10, 2, figsize=(10, 10))  # Change the figure size here

    for index, frame_num in enumerate(test_frames_int):
        compare_frames(frames[frame_num], deoldified_frames[frame_num], frame_num)
        # Create subplots
        axs[index,0].imshow(frames[frame_num])  
        axs[index,1].imshow(deoldified_frames[frame_num]) 
        # Remove the axis
        for ax in axs[index]:
            ax.axis('off')
    plt.show()
    #Rob Stuff

#0,1   2,3   4,5  6,7 8,9
    # save video
    save_video(deoldified_frames)

