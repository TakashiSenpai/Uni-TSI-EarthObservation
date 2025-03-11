import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import image_processing_tools as ipt


DATA_DIR = 'IMG/landsat_data/'
OUTPUT_DIR = 'IMG/'


image = plt.imread(DATA_DIR + 'downsampled_x10_LC08_L2SP_227065_20240822_20240830.png')
image = image[:,:,:3]
height = image.shape[0]
width = image.shape[1]
n_clusters = 5
print('Image read')
# Reshape the image to be a list of pixels
pixels_list = image.reshape(-1, 3)

print('Pixels list created')
inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(pixels_list)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
#plt.savefig('kMeans_elbow_method.png', dpi=300, bbox_inches='tight')
plt.show() 


kmeans = KMeans(n_clusters=n_clusters, algorithm='elkan', n_init=10, random_state=42)
kmeans.fit(pixels_list)
print('Kmeans fit')
# Get the cluster centroids 
centroids = kmeans.cluster_centers_
# print(centroids)

# Count number of pixels in each cluster
for i in range(n_clusters):
    print(f'Cluster {i}: {np.sum(kmeans.labels_ == i)} pixels')


# Create a mask which filters out the pixels of one cluster
fig, axes = plt.subplots(1, n_clusters, figsize=(12, 4))
for i in range(n_clusters):
    masked_image = pixels_list.copy()  # Create a fresh copy each time
    mask = kmeans.labels_ == i
    mask = ~mask  # Invert mask
    masked_image[mask] = [68/255, 1/255, 84/255]  # Set cluster pixels to purple
    masked_image = masked_image.reshape(height, width, 3)  # Reshape to image

    # Show in subplot
    axes[i].imshow(masked_image)
    axes[i].axis("off")  # Hide axes
    axes[i].set_title(f"Cluster {i}")
    #plt.imsave(DATA_DIR + f'kmeans_cluster_{i}.png', masked_image)
    print('Cluster', i, 'done')

# Adjust layout and show all images
plt.tight_layout()
plt.show()


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
    try:
        image = plt.imread(DATA_DIR + image_files[i])
        image = image[:,:,:3]
        height = image.shape[0]
        width = image.shape[1]
    except FileNotFoundError:
        print(f"File {image_files[i]} not found.")
        continue
    except Exception as e:
        print(f"Error reading file {image_files[i]}: {e}")
        continue
    pixels_list = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, algorithm='elkan', n_init=10, random_state=42)
    kmeans.fit(pixels_list)
    print(f'Processing image {i+1}/{nIms}...')
    burnt_areas[i]  = np.sum(kmeans.labels_ == 0)
    cloud_cover[i]  =  np.sum(kmeans.labels_ == 3)
    farm_areas[i]   =  np.sum(kmeans.labels_ == 2)
    forest_areas[i] =  np.sum(kmeans.labels_ == 4)
    background[i] = np.sum(kmeans.labels_ == 1)
    total_pixels = burnt_areas[i] + cloud_cover[i] + farm_areas[i] + forest_areas[i]
    print(f'Burnt: {burnt_areas[i]/total_pixels*100:.2f}%, Cloud: {cloud_cover[i]/total_pixels*100:.2f}%, Farm: {farm_areas[i]/total_pixels*100:.2f}%, Forest: {forest_areas[i]/total_pixels*100:.2f}%, Background: {background[i]/total_pixels*100:.2f}%')
    fig, axes = plt.subplots(1, n_clusters, figsize=(12, 4))
    for j in range(n_clusters):
        masked_image = pixels_list.copy()  # Create a fresh copy each time
        mask = kmeans.labels_ == j
        mask = ~mask  # Invert mask
        masked_image[mask] = [0, 0, 0]
        masked_image = masked_image.reshape(height, width, 3)  # Reshape to image

    # Show in subplot
        axes[j].imshow(masked_image)
        axes[j].axis("off")  # Hide axes
        axes[j].set_title(f"Cluster {i}")
        plt.imsave(OUTPUT_DIR + f'kmeans_cluster_{j}_image_{i}.png', masked_image)

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