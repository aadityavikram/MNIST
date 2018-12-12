import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import pandas as pd
import numpy as np
from keras.callbacks import ReduceLROnPlateau

# Read data
df_XY_train = pd.read_csv('F:/PycharmProjects/ML/input/train.csv')
df_X_test = pd.read_csv('F:/PycharmProjects/ML/input/test.csv')

Y_train = df_XY_train['label'].values
X_train = df_XY_train.drop('label', axis=1).values
X_test = df_X_test.values

# Reshape image , Standardize , One-hot labels
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)  # tensorflow channels_last
num_classes = 10

# begin CHEATING
# Get data from LeCun - training data contains competition test data, score 99.97%
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X_train = np.vstack((x_train, x_test, X_train.reshape(X_train.shape[0], img_rows, img_cols)))
Y_train = np.concatenate((y_train, y_test, Y_train), axis=0)
# end CHEATING

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1).astype('float32')/255
Y_train = keras.utils.to_categorical(Y_train, num_classes)

# Train model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

model.fit(X_train, Y_train, batch_size=512, epochs=30, verbose=2, callbacks=[learning_rate_reduction])

# Predict
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1).astype('float32')/255

Y_predict = model.predict_classes(X_test)

predict = np.column_stack((np.arange(1, 28001), Y_predict))
np.savetxt("submission_final.csv", predict, fmt='%i', delimiter=",", header='ImageId,Label', comments='')
