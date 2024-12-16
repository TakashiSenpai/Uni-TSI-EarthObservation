import numpy as np
import scipy
import matplotlib.pyplot as plt
import glob
import os

MAX_UINT16 = np.iinfo(np.uint16).max

def get_datasets(directory):
    '''
        Scans the specified directory for TIF files and returns
        the set of dataset identifiers.

        Image files have the identifier BX, where X is the band 
        index. The dataset identifiers are stripped of the file
        extension and band index.
    '''
    old_dir = os.getcwd()
    os.chdir(directory)
    tif_files = glob.glob('*.TIF')
    band_files = []
    for file in tif_files:
        if 'B' in file:
            band_files.append(directory + file[:-5])
    datasets = sorted(set(band_files))
    os.chdir(old_dir)
    return datasets

def get_image_files(directory):
    old_dir = os.getcwd()
    os.chdir(directory)
    files = glob.glob('*.png')
    os.chdir(old_dir)
    return files

def generate_landsat_rgb_image(file_name, r_index=1, g_index=2, b_index=3):
    '''
        Create a false color rgb image based on the specified bands.
    '''
    
    r = plt.imread(f'{file_name}{r_index}.TIF')
    g = plt.imread(f'{file_name}{g_index}.TIF')
    b = plt.imread(f'{file_name}{b_index}.TIF')

    r = r / MAX_UINT16
    g = g / MAX_UINT16
    b = b / MAX_UINT16

    width, height = r.shape

    pixels = np.ndarray((width, height, 3))

    pixels[:, :, 0] = r 
    pixels[:, :, 1] = g 
    pixels[:, :, 2] = b 

    return pixels


def gaussian_kernel(width : int, height : int, sigma : float = 1):
    '''
        Constructs a Gaussian kernel in the spatial domain
    '''

    x = [i + 0.5 - width/2  for i in range(width)]
    y = [i + 0.5 - height/2 for i in range(height)]
    x, y = np.meshgrid(x, y)
    
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel) # normalization

    return kernel


def fft_convolve(pixels, kernel):
    '''
        Takes the convolution of an image by using the fast fourier transform
        to speed up the computation
    '''

    image_width, image_height, channels = pixels.shape
    kernel_width, kernel_height = kernel.shape

    # prepare the kernel for convolution: pad with zeros, and center
    padded_kernel = np.zeros((image_width, image_height))
    start_height = (image_height - kernel_height) // 2
    start_width  = (image_width  - kernel_width ) // 2
    padded_kernel[start_width:start_width+kernel_width, start_height:start_height+kernel_height] = kernel
    padded_kernel = np.fft.fftshift(padded_kernel)

    # take the fourier transform 
    fft_pixels = np.fft.fft2(pixels, axes=(0,1))
    fft_kernel = np.fft.fft2(padded_kernel, axes=(0,1))

    # convolution time 
    fft_convolution = np.zeros_like(fft_pixels)
    convolution = np.zeros_like(pixels)
    for i in range(channels):
        fft_convolution[:, :, i] = fft_pixels[:, :, i] * fft_kernel
        convolution[:, :, i] = np.fft.ifft2(fft_convolution[:, :, i]).real
    convolution = (convolution - convolution.min()) / (convolution.max() - convolution.min())
    
    return convolution


# ======================================================================================================

def gen_image():
    band_1 = plt.imread('LC08_L2SP_227065_20240822_20240830_02_T1_SR_B1.TIF')
    band_2 = plt.imread('LC08_L2SP_227065_20240822_20240830_02_T1_SR_B2.TIF')
    band_3 = plt.imread('LC08_L2SP_227065_20240822_20240830_02_T1_SR_B3.TIF')
    band_4 = plt.imread('LC08_L2SP_227065_20240822_20240830_02_T1_SR_B4.TIF')
    band_5 = plt.imread('LC08_L2SP_227065_20240822_20240830_02_T1_SR_B5.TIF')
    band_6 = plt.imread('LC08_L2SP_227065_20240822_20240830_02_T1_SR_B6.TIF')
    band_7 = plt.imread('LC08_L2SP_227065_20240822_20240830_02_T1_SR_B7.TIF')

    r = band_6
    g = band_5
    b = band_2

    # fig, ax = plt.subplots(2,4)
    # ax[0,0].imshow(band_1)
    # ax[0,1].imshow(band_2)
    # ax[0,2].imshow(band_3)
    # ax[0,3].imshow(band_4)
    # ax[1,0].imshow(band_5)
    # ax[1,1].imshow(band_6)
    # ax[1,2].imshow(band_7)

    r = r / 65536
    g = g / 65536
    b = b / 65536

    width, height = r.shape
    print('Image dimensions:', width, height)
    channels = 3

    pixels = np.ndarray((width, height, channels))

    pixels[:, :, 0] = r 
    pixels[:, :, 1] = g 
    pixels[:, :, 2] = b 

    pixels = pixels 

    fig, ax = plt.subplots()
    ax.imshow(pixels)

    plt.imsave('CombinedImage.png', pixels)
    plt.show()

def filter_color(pixels, min, max):
    mask = np.all((pixels >= min) & (pixels <= max), keepdims=True, axis=-1)
    return mask

def gaussian_blur(pixels, kernel_size=(3,3), sigma=1):
    width, height, channels = pixels.shape
    new_pixels = np.zeros_like(pixels)

    # kernel definition
    x = np.arange(-(kernel_size[0] // 2), kernel_size[0] // 2 + 1)
    y = np.arange(-(kernel_size[1] // 2), kernel_size[1] // 2 + 1)
    x, y = np.meshgrid(x, y)
    #h = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    h = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    print(h)

    # pad the edges with zeroes
    padded_image = np.pad(pixels, [(kernel_size[0] // 2,), (kernel_size[1] // 2,), (0,)], mode='constant')
    
    # image processing
    for channel in range(channels):
        for i in range(width):
            for j in range(height):
                # extract the region of interest
                region = padded_image[i:i+kernel_size[0], j:j+kernel_size[1], channel]
                # calculate the convolution with the region
                new_pixels[i, j, channel] = np.sum(region * h) 

    return new_pixels