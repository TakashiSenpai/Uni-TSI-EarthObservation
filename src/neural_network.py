import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import image_processing_tools as ipt
import os

os.system('cls')

def classify_image(im):

    h, w, c = im.shape
    data_im = np.reshape(im, (h*w, 3))

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(data_im, batch_size=4096)

    class_indices = np.argmax(predictions, axis=1)
    class_indices = np.reshape(class_indices, (h, w))

    classified_farm   = np.zeros((h, w, c), dtype=np.float16)
    classified_burnt  = np.zeros((h, w, c), dtype=np.float16)
    classified_forest = np.zeros((h, w, c), dtype=np.float16)
    classified_bg     = np.zeros((h, w, c), dtype=np.float16)
    classified_river  = np.zeros((h, w, c), dtype=np.float16)
    classified_cloud  = np.zeros((h, w, c), dtype=np.float16)

    classified_bg    [class_indices == 0] = np.ones(3)
    classified_burnt [class_indices == 1] = np.ones(3)
    classified_farm  [class_indices == 2] = np.ones(3)
    classified_forest[class_indices == 3] = np.ones(3)
    classified_river [class_indices == 4] = np.ones(3)
    classified_cloud [class_indices == 5] = np.ones(3)
    
    return classified_burnt, classified_farm, classified_forest, classified_bg, classified_river, classified_cloud

if __name__ == '__main__':

    # --------------------------------- #
    # --- Prepare the training data --- #
    # --------------------------------- #

    image_path = '../local/downsampled_x10_LC08_L2SP_227065_20240822_20240830.png'
    im = plt.imread(image_path)[:,:,:-1] # skip alpha channel in pos 4
    h, w, c = im.shape

    # sample training data from the original image    
    im_bg = im[0:50, 0:50, :]
    im_burnt = im[369:385, 433:460, :]
    im_farm = im[125:150, 480:520, :]
    im_forest = im[200:300, 600:700, :]
    im_river = im[149:157, 551:552, :]
    im_cloud = im[385:393, 509:513, :]

    fig, ax = plt.subplots()
    fig.suptitle('Original Image')
    ax.imshow(im)

    fig, ax = plt.subplots(2, 3)
    fig.suptitle('Training data')
    ax[0,0].imshow(im_burnt)
    ax[0,0].set_title('Burnt Area')
    ax[0,1].imshow(im_farm)
    ax[0,1].set_title('Farm Area')
    ax[1,0].imshow(im_forest)
    ax[1,0].set_title('Forest Area')
    ax[1,1].imshow(im_bg)
    ax[1,1].set_title('Background')
    ax[0,2].imshow(im_river)
    ax[0,2].set_title('River')
    ax[1,2].imshow(im_cloud)
    ax[1,2].set_title('Cloud')
    plt.show()
    
    # Unravel data
    data_bg     = np.reshape(im_bg, (im_bg.shape[0] * im_bg.shape[1], 3))
    data_burnt  = np.reshape(im_burnt, (im_burnt.shape[0] * im_burnt.shape[1], 3))
    data_farm   = np.reshape(im_farm, (im_farm.shape[0] * im_farm.shape[1], 3))
    data_forest = np.reshape(im_forest, (im_forest.shape[0] * im_forest.shape[1], 3))
    data_river  = np.reshape(im_river, (im_river.shape[0] * im_river.shape[1], 3))
    data_cloud  = np.reshape(im_cloud, (im_cloud.shape[0] * im_cloud.shape[1], 3))

    labels_bg     = np.zeros(data_bg.shape[0])
    labels_burnt  = np.ones(data_burnt.shape[0])
    labels_farm   = 2 * np.ones(data_farm.shape[0])
    labels_forest = 3 * np.ones(data_forest.shape[0])
    labels_river  = 4 * np.ones(data_river.shape[0])
    labels_cloud  = 5 * np.ones(data_cloud.shape[0])

    train_data = np.append(data_burnt, data_farm, axis=0)
    train_data = np.append(train_data, data_forest, axis=0)
    train_data = np.append(train_data, data_bg, axis=0)
    train_data = np.append(train_data, data_river, axis=0)
    train_data = np.append(train_data, data_cloud, axis=0)

    train_labels = np.append(labels_burnt, labels_farm, axis=0)
    train_labels = np.append(train_labels, labels_forest, axis=0)
    train_labels = np.append(train_labels, labels_bg, axis=0)
    train_labels = np.append(train_labels, labels_river)
    train_labels = np.append(train_labels, labels_cloud)

    class_names = ['burnt_land', 'farm_land', 'forest', 'background', 'river', 'cloud']

    # --------------------------------- #
    # --- Build and train the model --- #
    # --------------------------------- #

    # Need to specify the input shape as the shape of the input training data
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(class_names))
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    model.fit(train_data, train_labels, epochs=10)

    # ----------------------- #
    # --- Prediction time --- #
    # ----------------------- #
    DATADIR = '../local/'
    OUTDIR = 'C:/temp/'
    image_files = ipt.get_image_files(DATADIR)
    image_files = [im for im in image_files if 'downsampled' not in im]
    print(f'\nDatasets to analyze:\n{image_files}')

    nIms = len(image_files)
    for i, image_path in enumerate(image_files):
        im = plt.imread(DATADIR + image_path)[:,:,:-1]

        print(f'\nClassifying image {i+1}/{nIms}...\n{image_path}')
        classified_burnt, classified_farm, classified_forest, classified_bg, classified_river, classified_cloud = classify_image(im)

        
        # debug
        fig, ax = plt.subplots(2, 3)
        fig.suptitle('Classified Image')
        ax[0, 0].imshow(classified_burnt)
        ax[0, 1].imshow(classified_farm)
        ax[1, 0].imshow(classified_forest)
        ax[1, 1].imshow(classified_bg)
        ax[0, 2].imshow(classified_river)
        ax[1, 2].imshow(classified_cloud)

        ax[0, 0].set_title('Burnt Area')
        ax[0, 1].set_title('Farm Area')
        ax[1, 0].set_title('Forest')
        ax[1, 1].set_title('Background')
        ax[0, 2].set_title('River')
        ax[1, 2].set_title('Cloud')
        
        
        print('Saving image: burnt...')
        plt.imsave(OUTDIR + 'classified_burnt_' + image_path, classified_burnt)
        
        print('Saving image: cloud...')
        plt.imsave(OUTDIR + 'classified_cloud_' + image_path, classified_cloud)
        
        print('Saving image: farm...')
        plt.imsave(OUTDIR + 'classified_farm_' + image_path, classified_farm)
        
        print('Saving image: forest...')
        plt.imsave(OUTDIR + 'classified_forest_' + image_path, classified_forest)

    plt.show()
    
    # ----------------------------- #
    # --- POST-PROCESS ANALYSIS --- #
    # ----------------------------- #

    DATADIR = 'C:/temp/'
    try:
        os.mkdir(DATADIR)
    except:
        pass
    image_files = ipt.get_image_files(DATADIR)

    n_images = int(len(image_files) / 4)
    images_burnt   = [im for im in image_files if 'burnt' in im]
    images_cloud   = [im for im in image_files if 'cloud' in im]
    images_farm    = [im for im in image_files if 'farm' in im]
    images_forest  = [im for im in image_files if 'forest' in im]
    
    burnt_areas  = np.zeros(n_images)
    cloud_cover  = np.zeros(n_images)
    farm_areas   = np.zeros(n_images)
    forest_areas = np.zeros(n_images)
    for i in range(n_images):
        print(f'Processing image {i+1}/{n_images}...')
        im_burnt  = plt.imread(DATADIR + images_burnt[i])
        im_cloud  = plt.imread(DATADIR + images_cloud[i])
        im_farm   = plt.imread(DATADIR + images_farm[i])
        im_forest = plt.imread(DATADIR + images_forest[i])
        burnt_areas[i]  = np.sum(im_burnt)
        cloud_cover[i]  = np.sum(im_cloud)
        farm_areas[i]   = np.sum(im_farm)
        forest_areas[i] = np.sum(im_forest)

    x = range(n_images)
    plt.plot(x, np.float32(cloud_cover), marker='o', label='Cloud cover')
    plt.plot(x, np.float32(farm_areas), marker='o', label='Farm area')
    plt.plot(x, np.float32(forest_areas), marker='o', label='Forest Area')
    plt.plot(x, np.float32(burnt_areas), marker='o', label='Burnt area')
    plt.legend()
    plt.xlabel('Image index')
    plt.ylabel('Pixel count')
    plt.savefig('analysis_stats.png', dpi=150, bbox_inches='tight')
    plt.show()

    os.rmdir(DATADIR)