from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

IMG_SIZE = 64

# Needs refactoring....
def loadImageSet():
	train_x = np.zeros(shape=(1199, IMG_SIZE, IMG_SIZE, 3))
	train_y = np.zeros(shape=(1199))
	i = 0
	for imgName in glob.glob("C:/Users/jordan/Desktop/Coding Projects/CatDog/data/train/cat/*.jpg"):
		img = Image.open(imgName)
		img = img.resize((IMG_SIZE,IMG_SIZE))
		imgArray = np.array(img)
		imgArray = imgArray[:,:,0:3]
		imgArray = imgArray / 256
		train_x[i] = imgArray
		train_y[i] = 0
		i+=1

	for imgName in glob.glob("C:/Users/jordan/Desktop/Coding Projects/CatDog/data/train/dog/*.jpg"):
		img = Image.open(imgName)
		img = img.resize((IMG_SIZE,IMG_SIZE))
		imgArray = np.array(img)
		imgArray = imgArray[:,:,0:3]
		imgArray = imgArray / 256
		train_x[i] = imgArray
		train_y[i] = 1
		i+=1

	return train_x , train_y


train_x, train_y = loadImageSet()
print(train_x.shape)
train_x = train_x.reshape(-1, IMG_SIZE,IMG_SIZE, 3)
train_y = to_categorical(train_y)


model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_x, train_y, epochs = 20, batch_size = 100, verbose = 1)


q = model.predict( np.array( [train_x[0],] )  )
plt.imshow(train_x[0])
plt.title("Cat: " + str(q[0][0]) + " " + "Dog: " + str(q[0][1]))
plt.show()