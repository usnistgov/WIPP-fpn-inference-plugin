# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.
import argparse
import sys, glob, warnings, os
import time
import os

import sys, glob

import PIL
import matplotlib.pyplot as plt
import skimage
import skimage.io
from PIL import Image
from skimage.transform import resize
from skimage.util import img_as_ubyte
from skimage.color import label2rgb


sys.path.insert(1, '../src');
sys.path.insert(1, '../../visualization')
from watershed_infer import *
from os.path import isfile, join
from os import listdir



from skimage.util import img_as_ubyte
from skimage.color import label2rgb

sys.path.insert(1, '../inference/mrcnn/src');
sys.path.insert(1, '../visualization')

from watershed_infer import *
from download_util import *


def main():
    tic = time.perf_counter()

    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    # trying to fix the cuDNN issue
    import tensorflow as tf
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    ##end of fixing

    # import tensorflow as tf
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True

    # time.sleep(30)

    tic = time.perf_counter()
    parser = argparse.ArgumentParser(prog='threshold',
                                     description='Create a binary image from a grayscale image and threshold value')

    # Define arguments
    parser.add_argument('--inputImages', dest='input_images', type=str,
                        help='filepath to the directory containing the images', required=True)
    # parser.add_argument('--thresholdValue', dest='threshold_value', type=int, required=True)
    parser.add_argument('--output', dest='output_folder', type=str, required=True)

    # Parse arguments
    args = parser.parse_args()
    input_images = args.input_images
    # threshold_value = args.threshold_value
    output_folder = args.output_folder

    BLUR_MODEL_URL = 'https://ndownloader.figshare.com/files/22280349?private_link=a3fec498ef6d08ac6973'
    BLUR_MODEL_PATH = 'blurred_border_FPN_pretrained.zip'
    # download_and_unzip_datasets(BLUR_MODEL_URL, BLUR_MODEL_PATH)

    modelwtsfname = "./blurred_border_FPN_pretrained.npy"
    modeljsonfname = "./blurred_border_FPN_pretrained.json"
    gaussian_blur_model = get_model(modeljsonfname, modelwtsfname)

    DISTANCE_MAP_MODEL_URL = 'https://ndownloader.figshare.com/files/22280352?private_link=5b1454e3f3bd23dea56f'
    DISTANCE_MAP_MODEL_PATH = 'distance_map_FPN_pretrained.zip'
    # download_and_unzip_datasets(DISTANCE_MAP_MODEL_URL, DISTANCE_MAP_MODEL_PATH)

    modelwtsfname = "./distance_map_FPN_pretrained.npy"
    modeljsonfname = "./distance_map_FPN_pretrained.json"
    distance_map_model = get_model(modeljsonfname, modelwtsfname)

    config_file_path = './demo.ini'
    with open(config_file_path, 'r') as fin:
        print(fin.read())



    images = listdir(input_images)

    img = np.zeros((len(images), 1078, 1278))

    for i in range(len(img)):
        tic = 0
        toc = 0
        tic = time.perf_counter()

        image = PIL.Image.open(input_images + "/" + images[i])
        width, height = image.size
        print("LLALALALALALALLALALALALALALALALALALALALLA")
        print(width, height)

        # image_resized = img_as_ubyte(resize(np.array(Image.open(input_images + "/" + images[i]).convert("L")), (1078, 1278)))

        image_resized = img_as_ubyte(resize(np.array(Image.open(input_images + "/" + images[i])), (1078, 1278)))
        img[i, :, :] = image_resized

        binary = watershed_infer(img, gaussian_blur_model, distance_map_model, config_file_path)

        print(binary.shape)

        plt.rcParams['figure.figsize'] = [15, 15]

        #skimage.io.imsave(output_folder + "/" + images[i], binary[i].astype(np.uint16), 'tifffile', False,
        #                  tile=(1024, 1024))

        output = skimage.transform.resize(binary[i], (height, width))

        skimage.io.imsave(output_folder + "/" + images[i], output.astype(np.uint16), 'tifffile', False,
                          tile=(1024, 1024))



        toc = time.perf_counter()

        print(f"Processing the image" + images[i] + f"took {toc - tic:0.4f} seconds")

        print('End processing image' + images[i])


if __name__ == "__main__":
    main()
