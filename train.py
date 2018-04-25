from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
from PIL import Image
import numpy as np
import glob, random

x_train, y_train = np.empty((0,28,28,1)), np.empty(0, 'int8')
for i in range(10):
    for fn in glob.glob('digitals/{}/*.jpg'.format(i)):
#         print(fn)
        im = Image.open(fn)
        imnp = np.array(im)
#         print(imnp.shape)
        x_train = np.vstack((x_train, imnp.reshape(1,28,28,1)))
        y_train = np.append(y_train,i)

x_test, y_test = np.empty((0,28,28,1)), np.empty(0, 'int8')
for i in range(x_train.shape[0]//4):
    id = random.randint(0,x_train.shape[0]-1)
    x_test = np.vstack((x_test, x_train[id].reshape(1,28,28,1)))
    y_test = np.append(y_test, y_train[id])        

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
input_shape = (img_rows, img_cols, 1)

train_datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit_generator(train_datagen.flow(x_train, y_train),
          epochs=epochs,
          verbose=1,
#           steps_per_epoch=len(x_train) / 32,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('minst.h6')
