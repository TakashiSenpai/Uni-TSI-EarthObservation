import matplotlib.pyplot as plt
import numpy as np
import image_processing_tools as ipt

DATA_DIR = 'IMG/landsat_data/'
OUTPUT_DIR = 'IMG/'

 # --------------------------------- #
    # --- Prepare the training data --- #
    # --------------------------------- #

im = plt.imread(DATA_DIR+'downsampled_x10_LC08_L2SP_227065_20240822_20240830.png')[:,:,:-1] # skip alpha channel in pos 4
h, w, c = im.shape

# sample training data from the original image    
im_bg = im[0:3, 0:3, :]
im_burnt = im[692:702, 293:310, :]
im_farm = im[141:148, 491:498, :]
im_forest = im[200:220, 680:700, :]
im_cloud = im[421:428, 505:509, :]


# fig, ax = plt.subplots()
# fig.suptitle('Original Image')
# ax.imshow(im)

fig, ax = plt.subplots(2, 3)
fig.suptitle('Training data')
ax[0,0].imshow(im_burnt)
ax[0,0].set_title('Burnt Area')
ax[0,1].imshow(im_farm)
ax[0,1].set_title('Farm Area')
ax[0,2].imshow(im_forest)
ax[0,2].set_title('Forest Area')
ax[1,0].imshow(im_bg)
ax[1,0].set_title('Background')
ax[1,1].imshow(im_cloud)
ax[1,1].set_title('Cloud')
ax[1,2].axis('off')  # Turn off the axis to make the space empty
plt.savefig('kNN_trainingData.png', dpi=300, bbox_inches='tight')
plt.show()

# Unravel data
data_bg = np.reshape(im_bg, (im_bg.shape[0] * im_bg.shape[1], 3))
data_burnt = np.reshape(im_burnt, (im_burnt.shape[0] * im_burnt.shape[1], 3))
data_farm = np.reshape(im_farm, (im_farm.shape[0] * im_farm.shape[1], 3))
data_forest = np.reshape(im_forest, (im_forest.shape[0] * im_forest.shape[1], 3))
data_cloud = np.reshape(im_cloud, (im_cloud.shape[0] * im_cloud.shape[1], 3))

labels_bg = np.zeros(data_bg.shape[0])
labels_burnt = np.ones(data_burnt.shape[0])
labels_farm = 2 * np.ones(data_farm.shape[0])
labels_forest = 3 * np.ones(data_forest.shape[0])
labels_cloud = 4 * np.ones(data_cloud.shape[0])

train_data = np.append(data_burnt, data_farm, axis=0)
train_data = np.append(train_data, data_forest, axis=0)
train_data = np.append(train_data, data_bg, axis=0)
train_data = np.append(train_data, data_cloud, axis=0)

train_labels = np.append(labels_burnt, labels_farm, axis=0)
train_labels = np.append(train_labels, labels_forest, axis=0)
train_labels = np.append(train_labels, labels_bg, axis=0)
train_labels = np.append(train_labels, labels_cloud)

# sanity check
print(train_data.shape)
print(train_labels.shape)

'''
    class_names = ['burnt_land', 'forest', 'cultivated_land', 'cloud']
    train_images = []
    train_labels = [] # integers between 0 and len(class_names)
'''

class_names = ['burnt_land', 'farm_land', 'forest', 'background', 'cloud']
n_clusters = len(class_names)
cluster_labels = np.zeros((h*w))

#---------------------------------#
# --- Classify the test image --- #
#---------------------------------#

# print('Image read')
# # Reshape the image to be a list of pixels
# pixels_list = im.reshape(-1, 3)

# print('Pixels list created')

# for i in range(len(pixels_list)):
#         min_distance = 1000000000000000000000000000000000
#         for j in range(train_data.shape[0]):
#             distanceSq = np.sum((train_data[j] - pixels_list[i]) ** 2)
#             if distanceSq < min_distance:
#                 min_distance = distanceSq
#                 min_index = j
#         cluster_labels[i] = train_labels[min_index]
#         if i % 100000 == 0:
#             print(i, 'pixels classified')

# print('Pixels classified')

# # Count number of pixels in each cluster
# for i in range(n_clusters):
#     print(f'Cluster {i}: {np.sum(cluster_labels == i)} pixels')


# # Create a mask which filters out the pixels of one cluster
# fig, axes = plt.subplots(1, n_clusters, figsize=(12, 4))
# for i in range(n_clusters):
#     masked_image = pixels_list.copy()  # Create a fresh copy each time
#     mask = cluster_labels == i
#     mask = ~mask  # Invert mask
#     masked_image[mask] = [68/255, 1/255, 84/255]  # Set cluster pixels to black
#     masked_image = masked_image.reshape(h, w, 3)  # Reshape to image

#     # Show in subplot
#     axes[i].imshow(masked_image)
#     axes[i].axis("off")  # Hide axes
#     axes[i].set_title(f"Cluster {i}")
#     plt.imsave(DATA_DIR + f'1nearest_cluster_{i}.png', masked_image)
#     print('Cluster', i, 'done')

# # Adjust layout and show all images
# plt.tight_layout()
# plt.show()

#---------------------------------#
# ----- Classify all images ----- #
#---------------------------------#

image_files = ipt.get_image_files(DATA_DIR)
image_files = [im for im in image_files if 'downsampled' not in im]
image_files = sorted(image_files)
print(f'\nDatasets to analyze:\n{image_files}')

nIms = len(image_files)

burnt_areas  = np.zeros(nIms)
cloud_cover  = np.zeros(nIms)
farm_areas   = np.zeros(nIms)
forest_areas = np.zeros(nIms)
background = np.zeros(nIms)
for i in range(nIms):
    print(image_files[i])
    image = plt.imread(DATA_DIR + image_files[i])
    image = image[:,:,:3]
    height = image.shape[0]
    width = image.shape[1]
    pixels_list = image.reshape(-1, 3)
    cluster_labels = np.zeros(len(pixels_list))
    for k in range(len(pixels_list)):
        min_distance = 1000000000000000000000000000000000
        for j in range(train_data.shape[0]):
            distanceSq = np.sum((train_data[j] - pixels_list[k]) ** 2)
            if distanceSq < min_distance:
                min_distance = distanceSq
                min_index = j
        cluster_labels[k] = train_labels[min_index]
    print(f'Processing image {i+1}/{nIms}...')
    burnt_areas[i]  = np.sum(cluster_labels == 1)
    cloud_cover[i]  =  np.sum(cluster_labels == 4)
    farm_areas[i]   =  np.sum(cluster_labels == 2)
    forest_areas[i] =  np.sum(cluster_labels == 3)
    background[i] = np.sum(cluster_labels == 0)
    total_pixels = burnt_areas[i] + cloud_cover[i] + farm_areas[i] + forest_areas[i]
    print(f'Burnt: {burnt_areas[i]/total_pixels*100:.2f}%, Cloud: {cloud_cover[i]/total_pixels*100:.2f}%, Farm: {farm_areas[i]/total_pixels*100:.2f}%, Forest: {forest_areas[i]/total_pixels*100:.2f}%, Background: {background[i]/total_pixels*100:.2f}%')
    fig, axes = plt.subplots(1, n_clusters, figsize=(12, 4))
    for j in range(n_clusters):
        masked_image = pixels_list.copy()  # Create a fresh copy each time
        mask = cluster_labels == j
        mask = ~mask  # Invert mask
        masked_image[mask] = [0, 0, 0]
        masked_image = masked_image.reshape(height, width, 3)  # Reshape to image

    # Show in subplot
        axes[j].imshow(masked_image)
        axes[j].axis("off")  # Hide axes
        axes[j].set_title(f"Cluster {i}")
        plt.imsave(OUTPUT_DIR + f'kNN_cluster_{j}_image_{i}.png', masked_image)

# Adjust layout and show all images
    plt.tight_layout()
    plt.show()

# Save the results as csv
results = np.vstack((cloud_cover, farm_areas, forest_areas, burnt_areas, background)).T
np.savetxt('kmeans_analysis_stats.csv', results, delimiter=',', header='0, 1, 2, 3, 4', comments='')
print('Results saved as kmeans_analysis_stats.csv')


x = range(nIms)
plt.plot(x, np.float32(cloud_cover)/total_pixels*100, marker='o', label='Cloud cover')
plt.plot(x, np.float32(farm_areas)/total_pixels*100, marker='o', label='Farm area')
plt.plot(x, np.float32(forest_areas)/total_pixels*100, marker='o', label='Forest Area')
plt.plot(x, np.float32(burnt_areas)/total_pixels*100, marker='o', label='Burnt area')
plt.legend()
plt.xlabel('Image index')
plt.ylabel('Area (%)')
plt.savefig('kmeans_analysis_stats.png', dpi=150, bbox_inches='tight')
plt.show()

sum_farm_burnt = np.float32(farm_areas) + np.float32(burnt_areas)
plt.plot(x, sum_farm_burnt/total_pixels*100, linestyle=':', color='black', marker='o', label='Farm + Burnt')
plt.errorbar(x, sum_farm_burnt/total_pixels*100, yerr=np.float32(cloud_cover)/total_pixels*100, fmt='o', linestyle=':', color='black', capsize=5, label='Error (Cloud Cover)')  
plt.legend()
plt.xlabel('Image index')
plt.ylabel('Area (%)')
plt.savefig('kmeans_developmentArea.png', dpi=150, bbox_inches='tight')
plt.show()
print('Done')