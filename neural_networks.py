import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Xavier initialization for weights
        stddev_W1 = np.sqrt(2 / (input_dim + hidden_dim))
        stddev_W2 = np.sqrt(2 / (hidden_dim + output_dim))

        self.W1 = np.random.randn(input_dim, hidden_dim) * stddev_W1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * stddev_W2
        self.b2 = np.zeros((1, output_dim))

        # Activation function for hidden layer
        self.activation_fn = activation

        # Placeholders for activations
        self.Z1 = None
        self.A1 = None
        self.Z2 = None
        self.A2 = None

    def activation(self, Z):
        if self.activation_fn == 'tanh':
            return np.tanh(Z)
        elif self.activation_fn == 'relu':
            return np.maximum(0, Z)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        else:
            raise ValueError("Unsupported activation function")

    def activation_derivative(self, Z):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(Z) ** 2
        elif self.activation_fn == 'relu':
            return np.where(Z > 0, 1, 0)
        elif self.activation_fn == 'sigmoid':
            sig = 1 / (1 + np.exp(-Z))
            return sig * (1 - sig)
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, X):
        # Forward pass: Input to hidden layer
        self.Z1 = np.dot(X, self.W1) + self.b1  # Linear transformation
        self.A1 = self.activation(self.Z1)      # Non-linear activation

        # Forward pass: Hidden to output layer with tanh activation
        self.Z2 = np.dot(self.A1, self.W2) + self.b2  # Linear transformation
        self.A2 = np.tanh(self.Z2)                    # Tanh activation for output layer

        return self.A2  # Predicted output

    def compute_loss(self, y_pred, y_true):
        # Mean Squared Error Loss
        m = y_true.shape[0]
        loss = (1 / (2 * m)) * np.sum((y_pred - y_true) ** 2)
        return loss

    def backward(self, X, y):
        m = y.shape[0]  # Number of samples

        # Compute derivative of loss w.r.t output activation (A2)
        dA2 = (self.A2 - y) / m  # Derivative of MSE loss w.r.t A2

        # Derivative of tanh activation function at output layer
        dZ2 = dA2 * (1 - self.A2 ** 2)

        # Gradients for weights and biases between hidden and output layer
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Backpropagate through activation function of hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.activation_derivative(self.Z1)

        # Gradients for weights and biases between input and hidden layer
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Store gradients for visualization
        self.dW1 = dW1
        self.dW2 = dW2

        # Update weights and biases using gradient descent
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Keep track of loss for debugging purposes
losses = []

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Calculate the current step number
    current_step = frame * 50  # Since we perform 10 training steps per frame

    # Perform training steps
    for _ in range(50):
        y_pred = mlp.forward(X)
        mlp.backward(X, y)
        loss = mlp.compute_loss(y_pred, y)
        losses.append(loss)

    # Plot hidden features (activations from the hidden layer)
    hidden_features = mlp.A1  # Shape: (n_samples, hidden_dim)
    ax_hidden.scatter(
        hidden_features[:, 0],
        hidden_features[:, 1],
        hidden_features[:, 2],
        c=y.ravel(),
        cmap='bwr',
        alpha=0.7
    )

    # Hyperplane visualization in the hidden space
    W = mlp.W2  # Weights from hidden to output layer
    b = mlp.b2  # Biases of the output layer

    # Create a grid to plot the decision hyperplane
    x_range = np.linspace(np.min(hidden_features[:, 0]), np.max(hidden_features[:, 0]), 10)
    y_range = np.linspace(np.min(hidden_features[:, 1]), np.max(hidden_features[:, 1]), 10)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)

    W_vec = W[:, 0]  # Extract weights as a vector
    b_scalar = b[0, 0]

    # Check if W_vec[2] is not too small to prevent division by zero
    if np.abs(W_vec[2]) > 1e-6:
        Z_grid = (-W_vec[0] * X_grid - W_vec[1] * Y_grid - b_scalar) / W_vec[2]
        ax_hidden.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.3, color='green')
    else:
        # Skip plotting if the plane is undefined
        pass

    # Distorted input space transformed by the hidden layer
    # Generate a grid in input space
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx_grid, yy_grid = np.meshgrid(
        np.linspace(x_min, x_max, 30),
        np.linspace(y_min, y_max, 30)
    )
    grid_points = np.c_[xx_grid.ravel(), yy_grid.ravel()]

    # Transform grid points through the hidden layer
    Z1_grid = np.dot(grid_points, mlp.W1) + mlp.b1
    A1_grid = mlp.activation(Z1_grid)

    # Reshape transformed grid to match the grid shape
    A1_grid_x = A1_grid[:, 0].reshape(xx_grid.shape)
    A1_grid_y = A1_grid[:, 1].reshape(xx_grid.shape)
    A1_grid_z = A1_grid[:, 2].reshape(xx_grid.shape)

    # Plot the transformed grid in the hidden space as a surface
    ax_hidden.plot_surface(
        A1_grid_x,
        A1_grid_y,
        A1_grid_z,
        color='blue',
        alpha=0.3
    )

    ax_hidden.set_xlabel('Hidden Unit 1')
    ax_hidden.set_ylabel('Hidden Unit 2')
    ax_hidden.set_zlabel('Hidden Unit 3')
    ax_hidden.set_title(f'Hidden Layer Feature Space at Step {current_step}')

    # Plot input layer decision boundary
    # Create a dense grid in input space
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Forward pass on grid points to get predictions
    A2_grid = mlp.forward(grid_points)
    A2_grid = A2_grid.reshape(xx.shape)

    # Plot decision boundary where prediction is 0
    ax_input.contourf(
        xx,
        yy,
        A2_grid,
        levels=[-1, 0, 1],
        alpha=0.3,
        colors=['blue', 'red']
    )
    ax_input.contour(
        xx,
        yy,
        A2_grid,
        levels=[0],
        colors='black'
    )
    ax_input.scatter(
        X[:, 0],
        X[:, 1],
        c=y.ravel(),
        cmap='bwr',
        edgecolors='k'
    )
    ax_input.set_title(f'Input Space Decision Boundary at Step {current_step}')
    ax_input.set_xlabel('X1')
    ax_input.set_ylabel('X2')

    # Visualize features and gradients as circles and edges
    # The edge thickness represents the magnitude of the gradient
    # Define neuron positions for visualization

    # Adjusted X positions to fit within 0 to 1 range
    x_coords = [0.1, 0.5, 0.9]  # Adjusted X positions for each layer
    layer_nodes = [2, 3, 1]     # Number of neurons in each layer
    node_positions = {}
    node_labels = {}

    # Vertical positions remain the same
    for idx, layer_size in enumerate(layer_nodes):
        x = x_coords[idx]
        y_positions = np.linspace(0.0, 1.0, layer_size)
        for neuron_idx in range(layer_size):
            y_pos = y_positions[neuron_idx]
            node_positions[(idx, neuron_idx)] = (x, y_pos)
            # Assign labels
            if idx == 0:
                label = f'$x_{{{neuron_idx+1}}}$'
            elif idx == 1:
                label = f'$h_{{{neuron_idx+1}}}$'
            else:
                label = '$y$'
            node_labels[(idx, neuron_idx)] = label

    # Plot neurons as circles and labels
    for (idx, neuron_idx), pos in node_positions.items():
        circle = Circle(pos, 0.035, fill=True, color='blue', zorder=-1)  # Lower z-order
        ax_gradient.add_patch(circle)
        # Add labels
        ax_gradient.text(
            pos[0],
            pos[1] + 0.05,
            node_labels[(idx, neuron_idx)],
            horizontalalignment='center',
            fontsize=10,
            zorder=3  # Higher z-order for labels
        )

    # Adjust the scaling factor for line widths
    scaling_factor = 100

    # Plot edges with thickness proportional to gradient magnitude
    # Edges from input to hidden layer
    for i in range(2):  # Input neurons
        for j in range(3):  # Hidden neurons
            start = node_positions[(0, i)]
            end = node_positions[(1, j)]
            grad_magnitude = np.abs(mlp.dW1[i, j])
            linewidth = grad_magnitude * scaling_factor
            ax_gradient.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                'k-',
                linewidth=linewidth,
                zorder=-2  # Lower z-order
            )

    # Edges from hidden to output layer
    for i in range(3):  # Hidden neurons
        start = node_positions[(1, i)]
        end = node_positions[(2, 0)]  # Only one output neuron
        grad_magnitude = np.abs(mlp.dW2[i, 0])
        linewidth = grad_magnitude * scaling_factor
        ax_gradient.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            'k-',
            linewidth=linewidth,
            zorder=-2  # Lower z-order
        )

    # Adjust plot appearance for gradients
    ax_gradient.set_xlim(-0.2, 1.2)
    ax_gradient.set_ylim(-0.2, 1.2)
    ax_gradient.set_aspect('equal')  # Ensure equal scaling

    # Add gridlines and ticks
    ax_gradient.set_xticks(np.linspace(0, 1, 6))
    ax_gradient.set_yticks(np.linspace(0, 1, 6))
    ax_gradient.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, zorder=2)  # Higher z-order for gridlines

    # Bring spines to front
    for spine in ax_gradient.spines.values():
        spine.set_zorder(2)

    # Add labels for axes
    ax_gradient.set_xlabel('Horizontal Position', fontsize=10)
    ax_gradient.set_ylabel('Vertical Position', fontsize=10)

    # Ensure tick labels are visible
    ax_gradient.tick_params(axis='both', which='major', labelsize=8)

    # Set the title for this visualization
    ax_gradient.set_title(f'Network Gradients Visualization at Step {current_step}', fontsize=12)

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(40, 14))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(
        fig,
        partial(
            update,
            mlp=mlp,
            ax_input=ax_input,
            ax_hidden=ax_hidden,
            ax_gradient=ax_gradient,
            X=X,
            y=y
        ),
        frames=step_num // 50,
        repeat=False
    )

    # Save the animation as a GIF
    ani.save(
        os.path.join(result_dir, "visualize.gif"),
        writer='pillow',
        fps=10,
        dpi=100
    )
    plt.close()

if __name__ == "__main__":
    activation = "sigmoid"
    lr = 0.3
    step_num = 1000
    visualize(activation, lr, step_num)

    # Loss curve for debugging purposes
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss over Time')
    plt.savefig(os.path.join(result_dir, "loss_curve.png"))
    plt.close()
