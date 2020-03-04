import os 
import glob
import cv2
import math
import numpy as np

from os.path import join
from scipy import signal


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
        print(flow_files[i])
        flow_field = np.load(flow_files[i])
        i += 1
        image_height = flow_field.shape[0]
        image_length = flow_field.shape[1]
        vals = np.array
        thresh = 0.1

        image = cv2.imread(frame_files[i+1])
        cv2.line(image, (0,0), (100,100), color = line_color)
        for x in range(image_height):
            for y in range(image_length):
                u = flow_field[x][y][0]
                v = flow_field[x][y][1]
                #print(type(v))
                
                mag = math.sqrt(u**2 + v**2)
                theta = math.atan(float(v/u))
                if(mag > 0.5):
                    vals = np.append(vals, mag)
                    u = int(u*10)
                    v = int(v*10)

                    #u = 100
                    #v = 100
                    cv2.line(image, (x,y), (int(x+u), int(y+v)), color = line_color)
                    
                    print("POS: ", x, y)
                    print("Cart:", u, v)
                    print("Rad: ", mag, theta)
                    print("==========================")
                    
        
        cv2.imshow(frame_files[i+1], image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(flow_field.shape)
        print(image.shape)





if __name__ == "__main__":
    main()

