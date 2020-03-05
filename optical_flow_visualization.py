import os 
import glob
import cv2
import math
import argparse
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
    parser = argparse.ArgumentParser(description="""calculate and save optical
                             flow between frames""")
    parser.add_argument('frame_dir', action = 'store')
    parser.add_argument('optflow_dir', action = 'store')
    parser.add_argument('--threshold', action = 'store', type = float, 
                            default = 0.5)
    parser.add_argument('--gap', action = 'store', type = int, default = 25)
    parser.add_argument('--scale', action = 'store', type = float, default = 1)

    args = parser.parse_args()

    input_frames_dir = args.frame_dir
    input_opflow_dir = args.optflow_dir
#    print(sorted(glob.glob(input_dir + "*.jpg")))
    frame_files = sorted(glob.glob(input_frames_dir + "*.jpg"))
    flow_files = sorted(glob.glob(input_opflow_dir+"*.npy"))
    line_color = (0, 0, 255)
    i = 0
    file_num = len(flow_files)

    while(i < file_num):
        image = cv2.imread(frame_files[i+1])
        flow_field = np.load(flow_files[i])
        
        image = overlay_optical_flow(image, flow_field, gap = args.gap, 
            color = (0,0,255), thresh = args.threshold, scale = args.scale)
    
        cv2.imshow(frame_files[i+1], image)
        cv2.waitKey(1000)
        i += 1

    cv2.destroyAllWindows()





if __name__ == "__main__":
    main()

