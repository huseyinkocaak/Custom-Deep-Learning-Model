import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt

def load_dataH5(train_path, test_path):
    train_dataset = h5py.File(train_path, "r")
    test_dataset = h5py.File(test_path, "r")

    return train_dataset, test_dataset

def initialize_params(layer_dims):
    np.random.seed(1)
    params = {}
    L = len(layer_dims)

    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])
        params['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return params

def sigmoid(z):
    return 1/(1+np.exp(-z)), z

def relu(z):
    return np.maximum(0,z), z

def sigmoid_backward(dA, cache):
    z = cache
    s = 1 / (1 + np.exp(-z))
    dZ = dA * s * (1 - s)
    return dZ

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)

    dZ[Z <= 0] = 0

    return dZ

def linear_forward(A, W, b):
    z = W.dot(A) + b
    cache = (A, W, b)
    return z, cache

def linear_forward_with_activation(A_prev, W, b, A):
    if A == "sigmoid":
        z, linear_cache = linear_forward(A_prev, W, b)
        a, activation_cache = sigmoid(z)
    elif A == "relu":
        z, linear_cache = linear_forward(A_prev, W, b)
        a, activation_cache = relu(z)

    return a, (linear_cache, activation_cache)

def model_forward(x, params):
    caches = []
    a = x
    L = len(params) // 2

    for i in range(1, L):
        a_prev = a

        a, cache = linear_forward_with_activation(a_prev, params['W' + str(i)], params['b' + str(i)], "relu")
        caches.append(cache)

    al, cache = linear_forward_with_activation(a, params['W' + str(L)], params['b' + str(L)], "sigmoid")
    caches.append(cache)

    return al, caches

def cost(p, y):
    m = y.shape[1]

    cost = (1. / m) * (-np.dot(y, np.log(p).T) - np.dot(1 - y, np.log(1 - p).T))

    np.squeeze(cost)

    return cost

def linear_backward(dZ, cache):
    a_prev, W, b = cache
    m = a_prev.shape[1]

    dW = 1. / m * np.dot(dZ, a_prev.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def linear_backward_with_activation(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def model_backward(P, Y, caches):
    grads = {}
    L = len(caches)
    Y = Y.reshape(P.shape)

    dP = - (np.divide(Y, P) - np.divide(1 - Y, 1 - P))

    current_cache = caches[L - 1]
    grads["dP" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward_with_activation(dP, current_cache, activation="sigmoid")

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dP_prev_temp, dW_temp, db_temp = linear_backward_with_activation(grads["dP" + str(l + 1)], current_cache, activation="relu")
        grads["dP" + str(l)] = dP_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_params(params, grads, learning_rate):
    L = len(params) // 2

    for l in range(L):
        params["W" + str(l + 1)] = params["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        params["b" + str(l + 1)] = params["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return params

def main_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000):
    np.random.seed(1)

    params = initialize_params(layers_dims)

    for i in range(0, num_iterations):

        AL, caches = model_forward(X, params)

        cost1 = cost(AL, Y)

        grads = model_backward(AL, Y, caches)

        params = update_params(params, grads, learning_rate)

        if (i+1) % 100 == 0:
            print("Cost after iteration {}: {}".format((i+1), np.squeeze(cost1)))

    return params

def predict(X, y, params):
    m = X.shape[1]
    p = np.zeros((1, m))

    prob, caches = model_forward(X, params)

    for i in range(0, prob.shape[1]):
        if prob[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print(str(np.sum((p == y) / m)) + "\n")

    return p

def predict_image(path,num_px,params,classes):
    fileImage = Image.open(path).convert("RGB").resize([num_px, num_px], Image.ANTIALIAS)
    label = [1]

    img = np.array(fileImage)
    plt.imshow(img)
    plt.show()
    
    img = img.reshape(num_px * num_px * 3, 1)
    img = img / 255.
    predicted_image = predict(img, label, params)

    print("model predicts a \"" + classes[int(np.squeeze(predicted_image)),].decode("utf-8") + "\" picture.")