##################
# Digit Recognizer
# Project by: Rutuja Kaushike (RNK170000)
# For Machine Learning : CS6375.502 F18  by Prof. Anurag Nagar
################## 

#import packages keras
from keras import utils
from keras.layers import Dense, Reshape, Conv2D, Flatten,Dropout, MaxPool2D
from keras.models import Sequential
from keras.optimizers import RMSprop

#import pacakages panda to read file
import pandas as pd
import numpy as np

#import package matplot to plot the grap of accuracy and loss
import matplotlib.pyplot as plt

#import packages from sklearn to split train data into train and validation data
from sklearn.model_selection import train_test_split

#import package sys to read the CSV file path
import sys

#read training and test data CSV files
train_file_path = sys.argv[1]

test_file_path = sys.argv[2]

output_file_path=sys.argv[3]

data_training = pd.read_csv(train_file_path)
data_test = pd.read_csv(test_file_path)

#print frist 10 rows of the train and test data
print(data_training.head(10))
print(data_test.head(10))

#seperate labels and predictors of training data
train_X = data_training.iloc[:, 1:]
train_Y = data_training.iloc[:, 0]

#normalize the data to float and then divide by 255 to make it in the range of 0.0 -1.0
train_X = train_X.astype('float32')
test_data = data_test.astype('float32')

train_X = train_X/255
test_data = test_data/255

#change label f traiing data to vector
train_Y = utils.to_categorical(train_Y, num_classes=10)

#create training and validation data set by splitting the original training data
train_X1,validation_x,train_Y1,validation_y = train_test_split(train_X,train_Y,test_size=0.2,random_state=42)

#apply CNN & Maxpool
model = Sequential()
model.add(Reshape(target_shape=(1, 28, 28), input_shape=(784,)))
model.add(Conv2D(32,kernel_size=(5, 5), padding="same",data_format="channels_first", kernel_initializer="uniform", use_bias=True))
model.add(MaxPool2D(pool_size=(2, 2), data_format="channels_first"))
model.add(Conv2D(64,kernel_size=(5, 5),padding="same",data_format="channels_first", kernel_initializer="uniform", use_bias=True))
model.add(MaxPool2D(pool_size=(2, 2), data_format="channels_first"))

#flatten data again and applying the activation function
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))


#compile the model using RMSprop optimizer, the only parameter to change is learning rate i.e lr
r_prop = RMSprop(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer=r_prop, metrics=['accuracy'])

#train the model using training and validation data, no of epochs and batch size of the epoch is mentioned
train_result = model.fit(train_X1, train_Y1,validation_data =(validation_x,validation_y), epochs=5, batch_size=64,verbose=1)

#test the model by passing the test data to model
len_test = test_data.shape[0]
result = model.predict_classes(test_data)
predict = {"ImageId":range(1, len_test+1), "Label":result}
predict = pd.DataFrame(predict)
predict.to_csv(output_file_path,header = True ,index = False)

#plot the graph for accuracy
#accuracy graph with epoch on X axis and acuracy on Y axis
plt.plot(train_result.history['acc'], color = 'green', label = 'on training')
plt.plot(train_result.history['val_acc'], color = 'red', label = 'on test')
plt.title('Graph for accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
#loss graph with epoch on X axis and loss on Y axis
plt.plot(train_result.history['loss'], color = 'green', label = 'on training')
plt.plot( train_result.history['val_loss'],color ='red', label = "On test")
plt.title('Graph for loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

print("Check the output")

#End of file
