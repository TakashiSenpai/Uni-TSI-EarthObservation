'''
    Ideas:
        - Generate tiled and downsampled images in the preprocessing step
        - Train a Convolutional Neural Network (CNN) to recognize colors
            * Use a single tiled image as a dataset
            * Filter out vegetation, burnt and agriculture surfaces
            * Recognize rivers and water areas
        - Evaluate total burnt surface over time
'''

import matplotlib.pyplot as plt
import image_processing_tools as ipt
import numpy as np
import time
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

DATA_DIR = '../local/'

if __name__ == '__main__':

    # =================================== #
    # === Pre-process Combined Images === #
    # =================================== #

    '''
    print('Pre-processing dataset...')
    t0 = time.perf_counter()
    datasets = ipt.get_datasets('../../landsat_data/')    
    print(datasets)
    t1 = time.perf_counter()
    print(f'Getting datasets took {t1-t0}s')

    t0 = time.perf_counter()
    image_files = [None] * len(datasets)
    for i, dataset in enumerate(datasets):
        print(f'Generating image {i+1}/{len(datasets)}')
        image = ipt.generate_landsat_rgb_image(dataset, r_index=6, g_index=5, b_index=2)
        file_name = dataset[dataset.rfind('/')+1:-11]
        plt.imsave(f'{DATA_DIR}{file_name}.png', image)
    t1 = time.perf_counter()
    print(f'Generating rgb images took {t1-t0}s')
    '''
   

    # =================== #
    # === Load images === #
    # =================== #
    
    print('Loading the images...')
    t0 = time.perf_counter()
    
    image_files = ipt.get_image_files(DATA_DIR)
    RGB_images  = [None] * len(image_files) 
    HSV_images  = [None] * len(image_files)
    downsampled_images = [None] * len(image_files) 
    scale_factor = 10
    for i, file in enumerate(image_files):
        if i == 1: break
        print(f'Loading image {i+1}/{len(image_files)}')
        pixels = plt.imread(DATA_DIR + file)[:, :, 0:3] # .png images have an alpha channel in 4th position
        if pixels.dtype != np.float32 and pixels.dtype != np.float64:
            pixels = pixels / 255.0
        RGB_images[i] = pixels
        HSV_images[i] = rgb_to_hsv(RGB_images[i])
        downsampled_images[i] = RGB_images[i][::scale_factor, ::scale_factor, :]

    
    t1 = time.perf_counter()
    print(f'Loading the images took {t1-t0:.3f}s')
    '''
    '''
 



    # =================== #
    # === Convolution === #
    # =================== #
    
    '''
    print('Computing the convolution...')
    t0 = time.perf_counter()
    kernel = ipt.gaussian_kernel(3, 3, 1)
    convolution = ipt.fft_convolve(pixels, kernel)
    t1 = time.perf_counter()
    print(f'Taking Fourier transform took {t1-t0:.3f}s')    
    plt.imshow(convolution)
    plt.show()
    '''


    # ============================== #
    # === Manual Color Filtering === #
    # ============================== #

    brown_min = np.array([40 / 255, 40 / 255, 30 / 255])
    brown_max = np.array([60 / 255, 50 / 255, 40 / 255])
    blue_min  = np.array([0.110, 0.130, 0.090])
    blue_max  = np.array([0.125, 0.230, 0.200])
    fire_min  = np.array([0.500, 0.210, 0.135])
    fire_max  = np.array([1.000, 0.270, 0.160])

    pixels = downsampled_images[0]
    brown = ipt.filter_color(pixels, brown_min, brown_max)
    blue  = ipt.filter_color(pixels, blue_min, blue_max)
    fires = ipt.filter_color(pixels, fire_min, fire_max)

    fig, ax = plt.subplots(2,2)
    ax[0,0].imshow(brown)
    ax[0,1].imshow(blue)
    ax[1,0].imshow(fires)
    ax[1,1].imshow(pixels)
    plt.show()

    # ===================== #
    # === Gaussian Blur === # 
    # ===================== #

    # blurred_pixels = ipt.gaussian_blur(pixels)
    # plt.imshow(blurred_pixels)
    # plt.show()

    # ======================== #
    # === RGB DISTRIBUTION === # 
    # ======================== #

    # step = 10
    # fig, axes = plt.subplots(subplot_kw={'projection': '3d'})
    # axes.scatter(pixels[::step,::step,0], pixels[::step,::step,1], pixels[::step,::step,2], marker='.')
    # plt.show()

