from deoldify import device
from deoldify.device_id import DeviceId
device.set(device=DeviceId.GPU0)

from deoldify.visualize import *
plt.style.use('dark_background')
torch.backends.cudnn.benchmark=True
import warnings
warnings.filterwarnings("ignore")

import cv2
from python_color_transfer.color_transfer import ColorTransfer
import numpy as np
from moviepy.editor import ImageSequenceClip, VideoFileClip, clips_array
from skimage.color import rgb2lab, deltaE_ciede2000
import argparse
import matplotlib.pyplot as plt
import pygame
from PIL import Image, ImageStat


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


def video_breakdown_color(path):
    # import video
    video = cv2.VideoCapture(path)

    # checks whether frames were extracted 
    success = 1

    # frames array
    frames = []

    # 

    while success: 
        # extract frame
        success, frame = video.read() 

        # add frame to frames array
        if success:
            # convert frame to greyscale
            grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            grey3 = cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB)
            frames.append(grey3)
    video.release()
    return frames


def deoldify_smoothed(frames):

    deoldified_frames = []
    colorizer = get_image_colorizer(artistic=True)
    render_factor=30

    for i in range(0, len(frames)-1):

        # grab frame to deoldify
        frame = frames[i]
        
        """DEOLDIFY COLORIZATION"""
        colorized_frame = colorizer.get_transformed_image(frame, render_factor=render_factor, watermarked=False)

        """COLOR TRANSFER FIRST FRAME COLORS TO NEW DEOLDIFIED FRAME"""
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


def show_video(video_1_frames, video_2_frames):
    # Load your two videos
    clip1 = ImageSequenceClip(list(video_1_frames), fps=20)
    clip2 = ImageSequenceClip(list(video_2_frames), fps=20)

    # Create a single video clip that places the two videos side by side
    final_clip = clips_array([[clip1, clip2]])

    # Write the result to a file
    final_clip.preview()


def calculate_lab_difference(image1, image2):
    # Convert the images to the LAB color space
    image1_lab = rgb2lab(image1)
    image2_lab = rgb2lab(image2)
    # Calculate the CIEDE2000 color difference
    return deltaE_ciede2000(image1_lab, image2_lab)


def compare_frames(frame1, frame2, frame_num):
    # Calculate and print the average CIEDE2000 color difference
    avg_ciede2000 = np.mean(calculate_lab_difference(frame1, frame2))
    # Calculate and print the average LAB difference (min:0 , max)
    #print("Frame Number: "+ str(frame_num))
    #print(f'Average CIEDE2000 Color Difference = {avg_ciede2000}')
    #print('-----------------------------------------------')
    return avg_ciede2000


def get_user_input():
    path = 'input_videos/'
    dir_list = os.listdir(path)

    i = 1
    for file in dir_list:
        print("("+ str(i) + ") " + file)
        i += 1
    video = input("Enter the number of the video you want to deoldify: ")

    i = 1
    input_video = None
    for file in dir_list:
        if i == int(video):
            input_video = file
        i += 1
    
    return 'input_videos/' + input_video


def detect_color(file, thumb_size=40, MSE_cutoff=22, adjust_color_bias=True):
    # import video
    video = cv2.VideoCapture(file)
    success, frame = video.read() 
    pil_img = Image.fromarray(frame)
    bands = pil_img.getbands()
    if bands == ('R','G','B') or bands== ('R','G','B','A'):
        thumb = pil_img.resize((thumb_size,thumb_size))
        SSE, bias = 0, [0,0,0]
        if adjust_color_bias:
            bias = ImageStat.Stat(thumb).mean[:3]
            bias = [b - sum(bias)/3 for b in bias ]
        for pixel in thumb.getdata():
            mu = sum(pixel)/3
            SSE += sum((pixel[i] - mu - bias[i])*(pixel[i] - mu - bias[i]) for i in [0,1,2])
        MSE = float(SSE)/(thumb_size*thumb_size)
        if MSE <= MSE_cutoff:
            return False
        else:
            return True
    elif len(bands)==1:
        return None
    else:
        return None

if __name__ == "__main__":
    # get path from user
    path = get_user_input()

    # detect if color video input
    color = detect_color(path)

    # breakdown video into frames
    if color:
        print("color detected")
        frames = video_breakdown_color(path=path)
    else: 
        print("greyscale detected")
        frames = video_breakdown(path=path)
    
    # deoldify frames
    deoldified_frames = deoldify_smoothed(frames=frames)

    # show video preview
    show_video(frames, deoldified_frames)

    # testing start
    test_frames_float = np.linspace(0, len(deoldified_frames)-1, 5)

    test_frames_int = [int(round(num)) for num in test_frames_float]

    fig, axs = plt.subplots(5, 3, figsize=(10, 10))  # Change the figure size here

    for index, frame_num in enumerate(test_frames_int):
        frame_dif = compare_frames(frames[frame_num], deoldified_frames[frame_num], frame_num)
        # Create subplots
        #Original Frames
        axs[index,0].imshow(frames[frame_num])  
        axs[index,0].set_title('Original Frame: '+str(frame_num))

        #Color Difference Text
        axs[index, 1].text(0.5, 0.5, 'CIEDE2000 Color Difference:\n '+str(frame_dif), horizontalalignment='center', verticalalignment='center', fontsize=12, transform=axs[index, 1].transAxes)
        axs[index, 1].axis('off')  # Hide axes

        #Colorized Frames
        axs[index,2].imshow(deoldified_frames[frame_num]) 
        axs[index,2].set_title('Colorized Frame: '+str(frame_num))

        # Remove the axis
        for ax in axs[index]:
            ax.axis('off')
    plt.show()
    # testing end

    # save video
    name = os.path.splitext(path)[0]
    result_name = 'result_videos/' + name + "_output.mp4"
    save_video(deoldified_frames)

