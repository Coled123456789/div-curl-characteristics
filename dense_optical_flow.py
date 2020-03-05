import os 
import argparse
import glob
import cv2
import numpy as np

from os.path import join
from scipy import signal


def main():
    parser = argparse.ArgumentParser(description="""calculate and save optical 
                            flow between frames""")
    parser.add_argument('input_dir', action = 'store')
    parser.add_argument('output_dir', action = 'store')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    files = sorted(glob.glob(input_dir + "*.jpg"))
    win_size = 15
    imga = None
    imgb = cv2.imread(files[1])
    imgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY).astype(int)


    i = 1
    print(len(files))
    while(i < len(files)):
        imga = imgb
        imgb = cv2.imread(files[i])
        imgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY).astype(int)
        flow = cv2.calcOpticalFlowFarneback(imga,imgb,None, 0.5, 1, 15, 3, 5, 1.5, 0)

        output_filename = output_dir+"opt_flow"+format(i - 1, '04')+".npy"
       
        print(output_filename)
        np.save(output_filename, flow)
        i += 1

        """
        output_filename = output_dir+"opt_flow_u"+format(file_num, '03')+".flo"
        print(output_filename)
        cv2.imwrite(output_filename, flow[:,:,0])
        output_filename = output_dir+"opt_flow_v"+format(file_num, '03')+".flo"
        cv2.imwrite(output_filename, flow[:,:,1])
        """



if __name__ == "__main__":
    main()

