import numpy as np

# f = w * x     -we dont care about bias
# f = 2 * x

X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0

# model prediction
def forward(x):
    return w * x

# loss , MSE (Mean Squared Error)
def loss (y, y_predicted):
    return ((y_predicted - y) ** 2).mean()

# gradient
# MSE = 1/n * (w*x - y) ** 2
# dJ/dw = 1/n * 2x.(w*x - y)
def gradient(x, y, y_predicted):
    return np.dot(2 * x, y_predicted - y).mean()

print(f'Prediction Before Training: f(5) = {forward(5):.3f}')

# training
learning_rate = .01
n_iters = 10

for epoch in range(n_iters):
    # prediciton = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients
    dw = gradient(X, Y, y_pred)

    # update the weights
    w -= learning_rate * dw
    
    if epoch % 1 == 0:
        print(f'epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')
