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

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
    return w * x

print(f'Prediction Before Training: f(5) = {forward(5):.3f}')

# training
learning_rate = .01
n_iters = 60

loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr = learning_rate)

for epoch in range(n_iters):
    # prediciton = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward()    # this will calculate -> dl/dw

    # update the weights
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()
    
    if epoch % 5 == 0:
        print(f'epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')
