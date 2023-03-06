import Custom_Library as CL
import numpy as np

train_dataset, test_dataset = CL.load_dataH5('train_catvnoncat.h5', 'test_catvnoncat.h5')

x_train = np.array(train_dataset["train_set_x"][:])
x_test = np.array(test_dataset["test_set_x"][:])

y_train = np.array(train_dataset["train_set_y"][:])
y_test = np.array(test_dataset["test_set_y"][:])

y_train = y_train.reshape((1,y_train.shape[0]))
y_test = y_test.reshape((1,y_test.shape[0]))

classes = np.array(test_dataset["list_classes"][:]) 

print("train shape: " + str(x_train.shape))
print("train label shape: " + str(y_train.shape))
print("test shape: " + str(x_test.shape))
print("test label shape: " + str(y_test.shape))
print("class shape: " + str(classes.shape))

num_px = x_train.shape[1]

x_train = x_train.reshape((x_train.shape[0], -1)).T
x_test = x_test.reshape((x_test.shape[0], -1)).T

x_train = x_train/255
x_test = x_test/255

print("train shape: " + str(x_train.shape))
print("test shape: " + str(x_test.shape))

layers_dims = [12288, 32, 16, 8, 1]

params = CL.main_model(x_train, y_train, layers_dims)

print("train accuracy: ")
pred_train = CL.predict(x_train, y_train, params)

print("test accuracy: ")
pred_test = CL.predict(x_test, y_test, params)

CL.predict_image("test.png",num_px,params,classes)
CL.predict_image("test1.png",num_px,params,classes)
CL.predict_image("test2.png",num_px,params,classes)