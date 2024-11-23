import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # learning rate
        self.activation_fn = activation  # activation function
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.zeros((1, output_dim))
        
    def activation(self, x):
        if self.activation_fn == 'tanh':        return np.tanh(x)
        elif self.activation_fn == 'relu':      return np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':   return 1 / (1 + np.exp(-x))
        else: 
            raise ValueError("Unknown activation function")

    def activation_derivative(self, x):
        if self.activation_fn == 'tanh':        return 1 - np.tanh(x) ** 2
        elif self.activation_fn == 'relu':      return (x > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        else:
            raise ValueError("Unknown activation function")
        
    def forward(self, X):
        self.X = X 

        self.z1 = X @ self.W1 + self.b1 
        self.a1 = self.activation(self.z1)  

        self.z2 = self.a1 @ self.W2 + self.b2  
        self.out = 1 / (1 + np.exp(-self.z2))  
        return self.out

    def backward(self, X, y):
        n_samples = X.shape[0]
        # Compute loss derivative with respect to output
        delta_out = self.out - y  # shape (n_samples, 1)
        # Gradients for W2 and b2
        dW2 = self.a1.T @ delta_out / n_samples  # shape (hidden_dim, output_dim)
        db2 = np.sum(delta_out, axis=0, keepdims=True) / n_samples  # shape (1, output_dim)
        # Backpropagate to hidden layer
        delta_a1 = delta_out @ self.W2.T  # shape (n_samples, hidden_dim)
        delta_z1 = delta_a1 * self.activation_derivative(self.z1)  # shape (n_samples, hidden_dim)
        # Gradients for W1 and b1
        dW1 = X.T @ delta_z1 / n_samples  # shape (input_dim, hidden_dim)
        db1 = np.sum(delta_z1, axis=0, keepdims=True) / n_samples  # shape (1, hidden_dim)
        # Update weights and biases
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        # Store gradients for visualization
        self.dW1 = dW1
        self.dW2 = dW2

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int)  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        
    # Plot hidden features
    hidden_features = mlp.a1  # shape (n_samples, hidden_dim)
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title('Hidden Layer Feature Space')

    W2 = mlp.W2 
    b2 = mlp.b2  

    xlim = [hidden_features[:, 0].min(), hidden_features[:, 0].max()]
    ylim = [hidden_features[:, 1].min(), hidden_features[:, 1].max()]
    x = np.linspace(xlim[0], xlim[1], 10)
    y_ = np.linspace(ylim[0], ylim[1], 10)
    X_h, Y_h = np.meshgrid(x, y_)
    if mlp.W2[2, 0] != 0:
        Z_h = (-mlp.W2[0, 0]*X_h - mlp.W2[1, 0]*Y_h - mlp.b2[0, 0])/mlp.W2[2, 0]
        ax_hidden.plot_surface(X_h, Y_h, Z_h, alpha=0.5)
    
    ax_hidden.set_xlabel('Neuron 1')
    ax_hidden.set_ylabel('Neuron 2')
    ax_hidden.set_zlabel('Neuron 3')

    # Plot input layer decision boundary
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    mlp.forward(grid)
    Z = mlp.out.reshape(xx.shape)
    ax_input.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.2, colors=['blue', 'red'])
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_title('Input Space Decision Boundary')
    ax_input.set_xlabel('X1')
    ax_input.set_ylabel('X2')

    # Visualize features and gradients as circles and edges
    # The edge thickness visually represents the magnitude of the gradient
    ax_gradient.set_xlim(-0.5, 2.5)
    ax_gradient.set_ylim(-0.5, 1.5)
    ax_gradient.set_aspect('equal')
    ax_gradient.axis('off')

    layer_positions = {'input': [(0, 0.5), (0, 1)],
                       'hidden': [(1, 0), (1, 0.5), (1, 1)],
                       'output': [(2, 0.5)]}

    for pos in layer_positions['input']:
        circle = plt.Circle(pos, radius=0.05, fill=True, color='lightblue', ec='k')
        ax_gradient.add_patch(circle)
    for pos in layer_positions['hidden']:
        circle = plt.Circle(pos, radius=0.05, fill=True, color='lightgreen', ec='k')
        ax_gradient.add_patch(circle)
    for pos in layer_positions['output']:
        circle = plt.Circle(pos, radius=0.05, fill=True, color='lightcoral', ec='k')
        ax_gradient.add_patch(circle)

    max_grad_W1 = np.max(np.abs(mlp.dW1)) + 1e-6
    for i, pos_i in enumerate(layer_positions['input']):
        for j, pos_j in enumerate(layer_positions['hidden']):
            grad_magnitude = np.abs(mlp.dW1[i, j])
            linewidth = grad_magnitude / max_grad_W1 * 5 
            ax_gradient.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 'k-', linewidth=linewidth)

    max_grad_W2 = np.max(np.abs(mlp.dW2)) + 1e-6
    for j, pos_j in enumerate(layer_positions['hidden']):
        pos_o = layer_positions['output'][0]
        grad_magnitude = np.abs(mlp.dW2[j, 0])
        linewidth = grad_magnitude / max_grad_W2 * 5 
        ax_gradient.plot([pos_j[0], pos_o[0]], [pos_j[1], pos_o[1]], 'k-', linewidth=linewidth)

    ax_gradient.set_title('Network Gradient Visualization')

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
