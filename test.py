import tensorflow as tf
import cv2 as cv
import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_data = []
test_data = []
for img in train_images[:5000]:
    resized_img = cv.resize(img, (224, 224))

    train_data.append(resized_img)
train_data = np.array(train_data)
train_data.shape

for img in test_images[:5000]:
    resized_img = cv.resize(img, (224, 224))

    test_data.append(resized_img)
test_data= np.array(test_data)
test_data.shape
# 将像素的值标准化至0到1的区间内。
train_images, test_images = train_data / 255.0, test_data / 255.0



class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # 由于 CIFAR 的标签是 array，
    # 因此您需要额外的索引（index）。
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()



# model = models.Sequential()
# model.add(layers.Conv2D(24, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(96, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(192, (3, 3), activation='relu'))
# model.add(layers.Conv2D(192, (3, 3), activation='relu'))
# model.add(layers.Conv2D(96, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.summary()
#
# model.add(layers.Flatten())
# model.add(layers.Dense(1024, activation='relu'))
# model.add(layers.Dense(1024, activation='relu'))
# model.add(layers.Dense(10))

model = VGG16(
    weights = None,
    classes = 10
)
model.summary()



model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels[:5000], epochs=10,
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)