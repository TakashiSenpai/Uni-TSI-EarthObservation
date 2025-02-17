import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    
    # --------------------------------- #
    # --- Prepare the training data --- #
    # --------------------------------- #

    image_path = '../local/downsampled_x10_LC09_L2SP_227065_20240830_20240831.png'
    im = plt.imread(image_path)[:,:,:-1] # skip alpha channel in pos 4
    h, w, c = im.shape

    # sample training data from the original image    
    im_bg = im[0:50, 0:50, :]
    im_burnt = im[379:395, 417:480, :]
    im_farm = im[125:150, 480:520, :]
    im_forest = im[200:300, 600:700, :]
    im_cloud = im[315:323, 497:498, :]


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
    ax[1,2].imshow(im_cloud)
    ax[1,2].set_title('Cloud')
    #plt.show()

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

    data_im = np.reshape(im, (h*w, 3))
    print(f'Image shape: h = {h}, w = {w}, c = {c}')
    print('Shape of data:', data_im.shape)

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(data_im)
    np.argmax(predictions)
    
    print('Shape of predictions:', predictions.shape)

    classified_im = np.hstack((predictions, np.zeros((predictions.shape[0], 1))))
    print('Shape of classified image', classified_im.shape)

    classified_burnt = np.zeros((h*w, c))
    classified_farm = np.zeros((h*w, c))
    classified_forest = np.zeros((h*w, c))
    classified_bg = np.zeros((h*w, c))
    classified_cloud = np.zeros((h*w, c))

    for i, p in enumerate(predictions):
        class_index = np.argmax(p)
        if class_index==0:
            classified_bg[i, :] = np.ones(3)
        elif class_index==1:
            classified_burnt[i, :] = np.ones(3)
        elif class_index==2:
            classified_farm[i, :] = np.ones(3)
        elif class_index==3:
            classified_forest[i, :] = np.ones(3)
        elif class_index==4:
            classified_cloud[i, :] = np.ones(3)

    #classified_im = np.reshape(classified_im, (h, w, c))
    classified_burnt = np.reshape(classified_burnt, (h, w, c))
    classified_farm = np.reshape(classified_farm, (h, w, c))
    classified_forest = np.reshape(classified_forest, (h, w, c))
    classified_bg = np.reshape(classified_bg, (h, w, c))
    classified_cloud = np.reshape(classified_cloud, (h, w, c))

    fig, ax = plt.subplots(2, 3)
    fig.suptitle('Classified Image')
    ax[0, 0].imshow(classified_burnt)
    ax[0, 1].imshow(classified_farm)
    ax[1, 0].imshow(classified_forest)
    ax[1, 1].imshow(classified_bg)
    ax[1, 2].imshow(classified_cloud)

    ax[0, 0].set_title('Burnt Area')
    ax[0, 1].set_title('Farm Area')
    ax[1, 0].set_title('Forest')
    ax[1, 1].set_title('Background')
    ax[1, 2].set_title('Cloud')

    masked_im = np.multiply(im, classified_burnt)
    masked_im = np.reshape(masked_im, (h, w, c))

    fig, ax = plt.subplots()
    ax.imshow(masked_im)

    plt.show()
