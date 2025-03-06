'''
    Ideas:
        - Generate tiled and downsampled images in the preprocessing step
        - Train a Convolutional Neural Network (CNN) to recognize colors
            * Use a single tiled image as a dataset
            * Filter out vegetation, burnt and agriculture surfaces
            * Recognize rivers and water areas
        - Evaluate total burnt surface over time
'''

from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import matplotlib.pyplot as plt
import image_processing_tools as ipt
import numpy as np
import time
import os
import glob

DATA_DIR = '../local/'
LANDSAT_DIR = '../../landsat_data/'

if __name__ == '__main__':

    # =================================== #
    # === Pre-process Combined Images === #
    # =================================== #

    # --- Scan directory containing landsat data ---
    print('\nPre-processing dataset...')
    t0 = time.perf_counter()
    
    datasets = ipt.get_datasets(LANDSAT_DIR)    
    print(f'Found {len(datasets)}') # datasets: \n{datasets}')

    t1 = time.perf_counter()
    print(f'Getting datasets took {t1-t0:.3f}s')

    # --- Generate usable images for analysis ---
    print('\nGenerating RGB images...')
    t0 = time.perf_counter()

    old_dir = os.getcwd()
    os.chdir(DATA_DIR)
    png_files = glob.glob('*.png')
    png_files = [el.replace('.png', '') for el in png_files]
    scale_factor = 10
    for i, dataset in enumerate(datasets):
        file_name = dataset[dataset.rfind('/')+1:-11]
        if file_name in png_files: 
            print(f'Skipping {file_name} as it is already processed')
            continue
        else:
            print(f'Generating image {i+1}/{len(datasets)}')
            image = ipt.generate_landsat_rgb_image(dataset, r_index=6, g_index=5, b_index=2)
            downsampled_image = image[::scale_factor, ::scale_factor, :]
            plt.imsave(f'{file_name}.png', image)
            plt.imsave(f'downsampled_x{scale_factor}_{file_name}.png', downsampled_image)
    os.chdir(old_dir)
    t1 = time.perf_counter()
    print(f'Generating RGB images took {t1-t0:.3f}s')
    '''
    '''
   

    # =================== #
    # === Load images === #
    # =================== #
    
    print('\nLoading the images...')
    t0 = time.perf_counter()
    
    image_files = ipt.get_image_files(DATA_DIR)
    RGB_images  = [] 
    HSV_images  = []
    downsampled_RGB_images = []
    downsampled_HSV_images = []
    for i, file in enumerate(image_files):
        if i == 10: break # load only the first image for now
        print(f'Loading image {i+1}/{len(image_files)}: {file}')
        if 'downsampled' in file:
            pixels = plt.imread(DATA_DIR + file)[:, :, 0:3]
            if pixels.dtype != np.float32 and pixels.dtype != np.float64:
                pixels = pixels / 255.0
            downsampled_RGB_images.append(pixels)
            downsampled_HSV_images.append(rgb_to_hsv(pixels))
        else:
            pixels = plt.imread(DATA_DIR + file)[:, :, 0:3] # .png images have an alpha channel in 4th position
            if pixels.dtype != np.float32 and pixels.dtype != np.float64:
                pixels = pixels / 255.0
            RGB_images.append(pixels)
            HSV_images.append(rgb_to_hsv(pixels))
    
    t1 = time.perf_counter()
    print(f'Loading the images took {t1-t0:.3f}s')
    '''
    '''
 
    # =========================== #
    # === Analyse color space === #
    # =========================== #

    '''
    for i in range(10):
        h = downsampled_HSV_images[i][:,:,0] * 360
        s = downsampled_HSV_images[i][:,:,1]
        v = downsampled_HSV_images[i][:,:,2]

        def hsv_to_cylindrical(H, S, V):
            theta = np.radians(H)  # Convert Hue to radians
            r = S
            z = V
            return theta, r, z

        theta, r, z = hsv_to_cylindrical(h, s, v)

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.scatter(theta, r, c=downsampled_RGB_images[i].reshape(-1, 3), marker='.')
    plt.show()

    fig, ax = plt.subplots(1, 3)
    fig.suptitle('Color space')

    ax[0].scatter(x, z, s=1)
    ax[0].set_xlabel('Hue')
    ax[0].set_ylabel('Saturation')
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)

    ax[1].scatter(y, z, s=1)
    ax[1].set_xlabel('Value')
    ax[1].set_ylabel('Saturation')
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)

    ax[2].scatter(x, y, s=1)
    ax[2].set_xlabel('Hue')
    ax[2].set_ylabel('Value')
    ax[2].set_xlim(0, 1)
    ax[2].set_ylim(0, 1)

    fig, ax = plt.subplots()
    ax.scatter(h, v, c=downsampled_RGB_images[0].reshape(-1, 3), s=1, rasterized=True)
    ax.set_xlabel('Hue')
    ax.set_ylabel('Value')
    plt.show()
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

    pixels = downsampled_RGB_images[5]
    brown = ipt.filter_color(pixels, brown_min, brown_max)
    blue  = ipt.filter_color(pixels, blue_min, blue_max)
    fires = ipt.filter_color(pixels, fire_min, fire_max)

    plt.imsave('filtered_original.png', pixels)
    plt.imsave('filtered_burnt.png', np.expand_dims(brown, axis=-1))
    plt.imsave('filtered_river.png', np.expand_dims(blue, axis=-1))

    fig, ax = plt.subplots(2,2)
    ax[0,0].imshow(brown)
    ax[0,1].imshow(blue)
    ax[1,0].imshow(fires)
    ax[1,1].imshow(pixels)
    plt.show()
    '''
    '''

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

