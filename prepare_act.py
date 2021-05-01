import cv2
from scipy.misc import imread, imsave
import os
#----------------prepare street view image-------------------#
# input_dir = '/home/wangtyu/datasets/ANU_data_small/streetview/'
# output_dir = '/home/wangtyu/datasets/CVACT/streetview/'

# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# images = os.listdir(input_dir)

# for img in images:
#     signal = imread(input_dir + img)

#     start = int(832 / 4)
#     image = signal[start: start + int(832 / 2), :, :]
#     image = cv2.resize(image, (616, 112), interpolation=cv2.INTER_AREA)
#     imsave(output_dir + img.replace('jpg', 'png'), image)

#---------prepare satellite view image-------------#
input_dir = '/home/wangtyu/ANU_data_small/satview_polish/'
output_dir = '/home/wangtyu/datasets/CVACT/satview_polish/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

images = os.listdir(input_dir)

for img in images:
    print(input_dir + img)
    image = cv2.imread(input_dir + img)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    imsave(output_dir + img.replace('jpg', 'png'), image)
