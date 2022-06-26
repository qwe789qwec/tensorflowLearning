import numpy as np
import plotly.express as px

n = 10000 # 10000 datum points
X = np.random.uniform(-10,10, n)
noise = np.random.normal(0, 3, n) # Gaussian distribution
true_w, true_b = 7.6, -3.3

y = true_w * X + true_b + noise # y = w * x + b + ε
# px.scatter(y)

def gradient_descent(X, y, w, b, learning_rate):
    dw = 2 * np.sum(X * (w * X - y + b)) # ∂e/∂w
    db = 2 * np.sum(w * X - y + b)       # ∂e/∂b
    w_new = w - learning_rate * dw        # minus sign since we are minizing e
    b_new = b - learning_rate * db
    return w_new, b_new

def get_loss(X,y,w,b):
    return (y - w * X - b).T @ (y - w * X - b)   # square loss,
    # .T and @ denote transpose and matrix multiplication resp.


learning_rate = 0.000001
max_epoch = 100
w, b = 0,0

for epoch in range(1,max_epoch+1):
    w,b = gradient_descent(X, y, w, b, learning_rate)

    if epoch % 5 == 0:
        print(f'{get_loss(X,y,w,b):.0f}')
        print("w=",w,"b=",b)

if b > 0:
    print(f'y = {w:.2f} x + {b:.2f}')
else:
    print(f'y = {w:.2f} x - {-b:.2f}')
