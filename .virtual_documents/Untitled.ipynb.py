import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.nn import relu, softmax


model = Sequential()

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", input_shape=(128,128,1), activation=relu))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation=relu))
model.add(MaxPooling2D())
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation=relu))
model.add(MaxPooling2D())
model.add(Dropout(0.3))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation=relu))
model.add(MaxPooling2D())
model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation=relu))
model.add(MaxPooling2D())
model.add(BatchNormalization())

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation=relu))
model.add(UpSampling2D())

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation=relu))
model.add(UpSampling2D())
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", input_shape=(128,128,1), activation=relu))
model.add(UpSampling2D())

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", input_shape=(128,128,1), activation=relu))
model.add(UpSampling2D())

# model.add(Conv2D(filters=3, kernel_size=(3,3), padding="same", activation=relu))
model.add(Conv2D(filters=1, kernel_size=(3,3), padding="same", activation="sigmoid"))


model.summary()


model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])



