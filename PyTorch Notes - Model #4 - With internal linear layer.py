# 1) Design model (input size, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop:
#     - forward pass: compute predictions
#     - backward pass: gradients
#     - update weights
import torch
import torch.nn as nn

# f = w * x     -we dont care about bias
# f = 2 * x

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
x_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape

input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)

print(f'Prediction Before Training: f(5) = {model(x_test).item():.3f}')

# training
learning_rate = .01
n_iters = 60

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

for epoch in range(n_iters):
    # prediciton = forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward()    # this will calculate -> dl/dw

    # update the weights
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()
    
    if epoch % 5 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch + 1}: w = {w[0][0]:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(x_test).item():.3f}')
