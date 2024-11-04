import matplotlib.pyplot as plt
import image_processing_tools as ipt
import numpy as np

if __name__ == '__main__':

    # =================================== #
    # === Pre-process Combined Images === #
    # =================================== #

    datasets = ipt.get_datasets()    
    print(datasets)

    for dataset in datasets:
        ipt.generate_landsat_rgb_image(dataset, r_index=1, g_index=2, b_index=3, save=False, show=True)

    # ============================== #
    # === Manual Color Filtering === #
    # ============================== #

    brown_min = np.array([40 / 255, 40 / 255, 30 / 255])
    brown_max = np.array([60 / 255, 50 / 255, 40 / 255])
    blue_min  = np.array([0.110, 0.130, 0.090])
    blue_max  = np.array([0.125, 0.230, 0.200])
    fire_min  = np.array([0.500, 0.210, 0.135])
    fire_max  = np.array([1.000, 0.270, 0.160])

    pixels = plt.imread('CombinedImage.png')
    width, height, channels = pixels.shape

    pixels = pixels[:,:,0:3]
    
    brown = ipt.filter_color(pixels, brown_min, brown_max)
    blue  = ipt.filter_color(pixels, blue_min, blue_max)
    fires = ipt.filter_color(pixels, fire_min, fire_max)

    # fig, ax = plt.subplots(2,2)
    # ax[0,0].imshow(brown)
    # ax[0,1].imshow(blue)
    # ax[1,0].imshow(fires)
    # ax[1,1].imshow(pixels)
    # plt.show()

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

