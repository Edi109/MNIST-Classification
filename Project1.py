import numpy as np
import sklearn.datasets as sk
import sklearn.model_selection as sk_model_selection
import matplotlib.pyplot as plt

# Step 1: Load the breast cancer dataset and split into training and validation sets
dataset = sk.load_breast_cancer()
X, y = dataset.data, dataset.target
X_train, X_val, y_train, y_val = sk_model_selection.train_test_split(X, y, random_state=123)

# Step 2: Standardize the data
mu = np.mean(X_train, axis=0)
s = np.std(X_train, axis=0)
X_train = (X_train - mu) / s
X_val = (X_val - mu) / s

# Step 3: Implement logistic regression using gradient descent
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        h = sigmoid(np.dot(X, theta))
        gradient = np.dot(X.T, (h - y)) / m
        theta -= alpha * gradient
        cost_history[i] = cost_function(X, y, theta)

    return theta, cost_history

# Step 4: Implement logistic regression using stochastic gradient descent
def stochastic_gradient_descent(X, y, theta, alpha, epochs):
    m = len(y)
    cost_history = []

    for epoch in range(epochs):
        cost_epoch = 0.0
        for i in range(m):
            rand_idx = np.random.randint(0, m)
            X_i = X[rand_idx, :].reshape(1, -1)
            y_i = y[rand_idx].reshape(1, -1)
            h = sigmoid(np.dot(X_i, theta))
            gradient = np.dot(X_i.T, (h - y_i))
            theta -= alpha * gradient
            cost_epoch += cost_function(X_i, y_i, theta)
        cost_history.append(cost_epoch / m)

    return theta, cost_history

# Step 5: Test different learning rates for both gradient descent and stochastic gradient descent
learning_rates = [0.01, 0.05, 0.1, 0.5, 1]
iterations = 1000
epochs = 100

best_accuracy_gd = 0
best_learning_rate_gd = None
best_iterations_gd = None

best_accuracy_sgd = 0
best_learning_rate_sgd = None
best_epochs_sgd = None

# Step 6: Plot the cost function value vs. iteration/epoch for each learning rate
plt.figure(figsize=(10, 6))

# Gradient Descent
for lr in learning_rates:
    theta = np.zeros(X_train.shape[1]).reshape(-1, 1)
    theta, cost_history = gradient_descent(X_train, y_train.reshape(-1, 1), theta, lr, iterations)
    plt.plot(range(iterations), cost_history, label=f"GD lr={lr}")

    # Calculate accuracy
    predictions_gd = (sigmoid(np.dot(X_val, theta)) >= 0.5).astype(int)
    accuracy_gd = np.mean(predictions_gd == y_val.reshape(-1, 1)) * 100
    if accuracy_gd > best_accuracy_gd:
        best_accuracy_gd = accuracy_gd
        best_learning_rate_gd = lr
        best_iterations_gd = iterations

# Stochastic Gradient Descent
for lr in learning_rates:
    theta = np.zeros(X_train.shape[1]).reshape(-1, 1)
    theta, cost_history = stochastic_gradient_descent(X_train, y_train.reshape(-1, 1), theta, lr, epochs)
    plt.plot(range(epochs), cost_history, label=f"SGD lr={lr}")

    # Calculate accuracy
    predictions_sgd = (sigmoid(np.dot(X_val, theta)) >= 0.5).astype(int)
    accuracy_sgd = np.mean(predictions_sgd == y_val.reshape(-1, 1)) * 100
    if accuracy_sgd > best_accuracy_sgd:
        best_accuracy_sgd = accuracy_sgd
        best_learning_rate_sgd = lr
        best_epochs_sgd = epochs

plt.title("Cost Function Value vs. Iteration/Epoch")
plt.xlabel("Iteration/Epoch")
plt.ylabel("Cost Function Value")
plt.legend()
plt.show()

print(f"Best learning rate for Gradient Descent: {best_learning_rate_gd}, with accuracy: {best_accuracy_gd:}%")
print(f"Best learning rate for Stochastic Gradient Descent: {best_learning_rate_sgd}, with accuracy: {best_accuracy_sgd:}%")
