import os 
import glob
import cv2
import math
import numpy as np

from os.path import join
from scipy import signal

"""
overlays optical flow field on image
plots vectors seperated by 'gap' number of pixels
filters vectors smaller than thresh are filtered
scale, color parameters are used to change vector appearance
"""
def overlay_optical_flow(image, flow_field, gap = 5, thresh = 0, color = (0,0,255), scale = 1):
    if((flow_field.shape[0], flow_field.shape[1]) != (image.shape[0], image.shape[1])):
        raise Error("Mismatched dimensions for image and flow field")
    mag, ang = cv2.cartToPolar(flow_field[...,0], flow_field[...,1])
    for y in range(0, len(image), gap):
        for x in range(0, len(image[y]), gap):
            if(mag[y][x] > thresh):
                u = int(flow_field[y][x][0] * scale)
                v = int(flow_field[y][x][1] * scale)
                image = cv2.arrowedLine(image, (x, y), (x + u, y+v), color)
    return image


def main():
    input_frames_dir = "data/Crowd_PETS09/S3/High_Level/Time_14-16/View_001/"
    input_opflow_dir = "data/Crowd_PETS09/S3/High_Level/Time_14-16/View_001_Optical_Flows/"
#    print(sorted(glob.glob(input_dir + "*.jpg")))
    frame_files = sorted(glob.glob(input_frames_dir + "*.jpg"))
    flow_files = sorted(glob.glob(input_opflow_dir+"*.npy"))
    line_color = (0, 0, 255)
    i = 0
    file_num = len(flow_files)

    while(i < file_num):
        image = cv2.imread(frame_files[i+1])
        flow_field = np.load(flow_files[i])
        
        image = overlay_optical_flow(image, flow_field, gap = 10, color = (0,0,255), thresh = 5)
    
        cv2.imshow(frame_files[i+1], image)
        cv2.waitKey(1000)
        i += 1

    cv2.destroyAllWindows()





if __name__ == "__main__":
    main()

