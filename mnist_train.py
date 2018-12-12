import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras import regularizers
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist

train = pd.read_csv("F:/PycharmProjects/ML/input/train.csv")
test = pd.read_csv("F:/PycharmProjects/ML/input/test.csv")

Y_train = train["label"]
X_train = train.drop(labels=["label"], axis=1)

X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

X_train = X_train / 255.0
test = test / 255.0

Y_train = to_categorical(Y_train, num_classes=10)

(train_images_keras, train_labels_keras), (test_images_keras, test_labels_keras) = mnist.load_data()

train_images_keras = train_images_keras.reshape(60000,28,28,1)
test_images_keras = test_images_keras.reshape(10000,28,28,1)

train_images_keras = train_images_keras.astype('float32') / 255
test_images_keras = test_images_keras.astype('float32') / 255

train_labels_keras = to_categorical(train_labels_keras)
test_labels_keras = to_categorical(test_labels_keras)

train_images = np.concatenate((train_images_keras, X_train), axis=0)
train_labels = np.concatenate((train_labels_keras, Y_train), axis=0)

# X_train, X_validate, Y_train, Y_validate = train_test_split(X_train, Y_train, test_size=0.1, random_state=12345)

batch = 512
epochs = 35
lr = 0.001

# Building the model

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization(momentum=0.99, epsilon=1e-3, axis=-1, center=True, beta_initializer='zeros'))

model.add(Conv2D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization(momentum=0.99, epsilon=1e-3, axis=-1, center=True, beta_initializer='zeros'))

model.add(Flatten())

model.add(Dense(128, activation='tanh'))
model.add(Dropout(rate=0.2))

model.add(Dense(256, activation='tanh'))
model.add(Dropout(rate=0.2))

model.add(Dense(10, activation='softmax'))


# compile model
optimizer1 = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
optimizer2 = RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer1, loss='categorical_crossentropy', metrics=['accuracy'])


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

# train model
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

X_train, X_val, Y_train, Y_val = train_test_split(train_images, train_labels, test_size = 0.1, random_state=2)

datagen.fit(X_train)

history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch),
                              epochs=epochs, validation_data=(X_val, Y_val),
                              verbose=2, steps_per_epoch=X_train.shape[0]//batch,
                              callbacks=[learning_rate_reduction])

test_loss, test_acc = model.evaluate(test_images_keras, test_labels_keras)
print("_"*80)
print("Accuracy on test ", test_acc)

def build_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(28, 28, 1)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization(momentum=0.99, epsilon=1e-3, axis=-1, center=True, beta_initializer='zeros'))

    model.add(Conv2D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization(momentum=0.99, epsilon=1e-3, axis=-1, center=True, beta_initializer='zeros'))

    model.add(Flatten())

    model.add(Dense(128, activation='tanh'))
    model.add(Dropout(rate=0.2))

    model.add(Dense(256, activation='tanh'))
    model.add(Dropout(rate=0.2))

    model.add(Dense(10, activation='softmax'))

    # compile model
    optimizer1 = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    optimizer2 = RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer1, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


train_images1 = np.concatenate((train_images,test_images_keras), axis=0)
train_labels1 = np.concatenate((train_labels,test_labels_keras), axis=0)

X_train1, X_val1, Y_train1, Y_val1 = train_test_split(train_images1, train_labels1, test_size = 0.1, random_state=2)

model = build_model()

datagen.fit(X_train1)
history = model.fit_generator(datagen.flow(X_train1,Y_train1, batch_size=batch),
                              epochs = epochs, validation_data = (X_val1,Y_val1),
                              verbose = 2, steps_per_epoch=X_train1.shape[0] // batch,
                              callbacks=[learning_rate_reduction])

# Result
score = model.predict(test)
score = np.argmax(score, axis=1)
score = pd.Series(score, name="Label")

submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), score], axis=1)
submission.to_csv("submission_final.csv", index=False)
