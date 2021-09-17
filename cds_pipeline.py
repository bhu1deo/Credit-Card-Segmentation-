import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
from skimage.metrics import structural_similarity
import skimage
import keras
# %matplotlib inline

# Quickly train a LeNet here
from keras.datasets import mnist
from keras.utils import np_utils

# Load dataset as train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Set numeric type to float32 from uint8
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize value to [0, 1]
x_train /= 255
x_test /= 255

# Transform lables to one-hot encoding
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# Reshape the dataset into 4D array
x_train = x_train.reshape(x_train.shape[0], 28,28,1)
x_test = x_test.reshape(x_test.shape[0], 28,28,1)

from keras.models import Sequential
from keras import models, layers
import keras

#Instantiate an empty model
model = Sequential()
# C1 Convolutional Layer
model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(28,28,1), padding="same"))
# S2 Pooling Layer
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
# C3 Convolutional Layer
model.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
# S4 Pooling Layer
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
# C5 Fully Connected Convolutional Layer
model.add(layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
#Flatten the CNN output so that we can connect it with fully connected layers
model.add(layers.Flatten())
# FC6 Fully Connected Layer
model.add(layers.Dense(84, activation='tanh'))
#Output Layer with softmax activation
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])
hist = model.fit(x=x_train,y=y_train, epochs=20, batch_size=128, validation_data=(x_test, y_test), verbose=1)
# model.save('/content/drive/My Drive/mnist_lenet.h5')
# model = keras.models.load_model('/content/drive/My Drive/mnist_lenet.h5')
model.save('mnist_lenet.h5')
model = keras.models.load_model('mnist_lenet.h5')

# from google.colab import files
# files.upload()

img = cv2.imread("card1.jpg")
img = cv2.resize(img,(280,180))             # Reverse Order!!!
img = img[90:125, :, :]
orig_image = img.copy()
orig_image_temp = orig_image.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img, cmap='gray')
plt.show()

img_thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,2)
plt.imshow(img_thresh, cmap='gray')
plt.show()

img_blur = cv2.GaussianBlur(img,(5,5),0)
otsu_thesholded_img = cv2.threshold(img_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
plt.imshow(otsu_thesholded_img, cmap='gray')
plt.show()

bil_img = cv2.bilateralFilter(img, 15, 25, 25)      # Bilateral Filter
edged_img = cv2.Canny(bil_img, 100, 200)  # Canny Edge detector
edged_img = edged_img[5:-5,5:-5]             # Remove the boundary unnecessarily detecting a contour
plt.imshow(edged_img, cmap='gray')
plt.show()

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))           # Struct elements

edged_morph = cv2.morphologyEx(edged_img, cv2.MORPH_CLOSE, rectKernel)          # Close
thresh = cv2.threshold(edged_morph, 0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]       # thresholding
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)       # closing
plt.imshow(thresh, cmap='gray')
plt.show()

# Now find the contours
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)   # All the contours
print("Number of Contours found = " + str(len(contours)))
# Draw all contours
# -1 signifies drawing all contours
# cv2.drawContours(orig_image, contours, -1, (0, 255, 0), 3)
# plt.imshow(orig_image, cmap='gray')

roi = []
coord_x_min = []
coord_x_max = []
coord_y_min = []
coord_y_max = []
for cnt in contours:
  approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
  n = approx.ravel()
  xmin = np.amin(n[0:-1:2])
  xmax = np.amax(n[0:-1:2])
  ymin = np.amin(n[1:-1:2])
  ymax = np.amax(n[1:-1:2])
  if(thresh[:,xmin:xmax].shape[1]<40):
    continue
  coord_x_min.append(xmin)
  coord_x_max.append(xmax)
  coord_y_min.append(ymin)
  coord_y_max.append(ymax)
  roi.append(img[ymin:ymax+5,xmin:xmax+5])

coord_x_max = [i for _, i in sorted(zip(coord_x_min, coord_x_max))]
coord_y_min = [i for _, i in sorted(zip(coord_x_min, coord_y_min))]
coord_y_max = [i for _, i in sorted(zip(coord_x_min, coord_y_max))]
coord_x_min.sort()

roi = []
for i in range(len(coord_x_min)):
  xmin = coord_x_min[i]
  xmax = coord_x_max[i]
  ymin = coord_y_min[i]
  ymax = coord_y_max[i]
  roi.append(img[ymin:ymax+8,xmin:xmax+10])

for r in roi:
  plt.imshow(r,cmap='gray')
  plt.show()


def extract_one_digit(pix_val, labeled_img):
    index_pos = np.where(labeled_img == pix_val)
    index = list(zip(index_pos[0],index_pos[1]))
    new_image = np.zeros(labeled_img.shape)
    for i in index:
        new_image[i[0],i[1]] = 255                      # Contains Zero

    xmin = np.amin(np.where(new_image==255)[0])
    ymin = np.amin(np.where(new_image==255)[1])
    xmax = np.amax(np.where(new_image==255)[0])
    ymax = np.amax(np.where(new_image==255)[1])

    new_image = new_image[max(0,xmin-3):min(new_image.shape[0],xmax+3),max(0,ymin-3):min(new_image.shape[1],ymax+3)]
    if(new_image.shape[0]<10):
      return None
    new_image = cv2.resize(new_image, (28, 28))
    return new_image

pred = ''
extracted_digits = []
for i in range(len(roi)):
  try:
    ret2,thresh_roi_otsu = cv2.threshold(roi[i],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)            # threshold it
    kernel = np.ones((2,2),np.uint8)           # 3x3 gives extremely thick and 1x1 does nothing
    thresh_roi_dilated = cv2.dilate(thresh_roi_otsu,kernel,iterations = 1)         # dilate it
    # plt.imshow(thresh_roi_dilated,cmap='gray')
    # plt.show()
    num_labels, labels_im = cv2.connectedComponents(thresh_roi_dilated)            # Find connected components
    # print(num_labels)
    (a, b) = labels_im.shape                             # shape is stored
    q = []
    for w in range(a):
        for e in range(b):
            if labels_im[w, e] not in q:
                q.append(labels_im[w, e])                  # pixel values

    count = 0
    for j in q:
        if j != 0:                                       # Background not to be used
            count+= 1
            pix_val = j
            digit = extract_one_digit(pix_val, labels_im)
            if(digit is None):
              continue
            # plt.imshow(labels_im,cmap='gray')
            # plt.show()
            extracted_digits.append(digit)

            digit_temp = np.zeros((50,50))
            digit_temp[11:39,11:39] = digit
            # plt.imshow(digit_temp,cmap='gray')
            # plt.show()
            digit_temp = cv2.resize(digit_temp,(28,28))
            digit = digit_temp

            # Add some boundary here
            x = digit.shape[0]
            y = digit.shape[1]

            digit[digit<220] = 0
            digit = np.resize(digit,(1,28,28,1))
            pred_digit = np.argmax(model.predict(digit)[0])

            pred += str(pred_digit)
    pred += ' '
  except:
    print("Error predicting the following image")
    plt.imshow(roi[i],cmap='gray')
    plt.show()
print('Prediction - ' + pred)

def company(pred):
  if pred[0] == '3':
    return 'American Express'
  elif pred[0] == '4':
    return 'Visa'
  elif pred[0] == '5':
    return 'Mastercard'
  else:
    return 'Unknown'

print('Company name - ' + company(pred))
