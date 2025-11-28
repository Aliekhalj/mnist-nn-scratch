# 🧠 Neural Network from Scratch

This repository contains a **simple neural network implemented entirely from scratch in Python**, using only `NumPy` for numerical operations. The network is trained to recognize **handwritten digits** from the MNIST dataset.  

It’s a great project to understand **forward propagation, backpropagation, and gradient descent** without relying on frameworks like TensorFlow or PyTorch.

---

## 🚀 Features

- Two-layer neural network: input → hidden → output
- ReLU activation for hidden layer
- Softmax activation for output layer
- Cross-entropy loss (via one-hot encoding)
- Training via gradient descent
- Evaluation on a development set
- Visualizing predictions of individual digits

---

## 📊 How It Works

### 1. Forward Propagation

The network computes outputs using:

\[
Z^{[1]} = W^{[1]} X + b^{[1]} \\
A^{[1]} = \text{ReLU}(Z^{[1]}) \\
Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]} \\
A^{[2]} = \text{Softmax}(Z^{[2]})
\]

Where:  

- \(X\) = input features (pixels)  
- \(W^{[1]}, b^{[1]}\) = weights and biases for layer 1  
- \(W^{[2]}, b^{[2]}\) = weights and biases for layer 2  
- \(A^{[2]}\) = predicted probabilities for each digit class

---

### 2. Backward Propagation

Gradients are computed via the chain rule:

\[
dZ^{[2]} = A^{[2]} - Y_{\text{one-hot}} \\
dW^{[2]} = \frac{1}{m} dZ^{[2]} A^{[1]^T} \\
db^{[2]} = \frac{1}{m} \sum dZ^{[2]} \\
dZ^{[1]} = W^{[2]^T} dZ^{[2]} \odot \text{ReLU}'(Z^{[1]}) \\
dW^{[1]} = \frac{1}{m} dZ^{[1]} X^T \\
db^{[1]} = \frac{1}{m} \sum dZ^{[1]}
\]

- \(Y_{\text{one-hot}}\) = one-hot encoded labels  
- \(m\) = number of examples  
- \(\odot\) = element-wise multiplication

---

### 3. Parameter Update

Weights and biases are updated with **gradient descent**:

\[
W = W - \alpha dW \\
b = b - \alpha db
\]

- \(\alpha\) = learning rate  

---

## 🛠 Usage

1. **Load MNIST dataset** (`MNIST.csv`)  
2. **Split data** into training and development sets  
3. **Initialize parameters** with small random values  
4. **Train** with `gradient_descent(X_train, Y_train, alpha, iterations)`  
5. **Evaluate** on the dev set using `make_predictions` and `get_accuracy`  
6. **Visualize** predictions with `test_prediction(index, W1, b1, W2, b2)`

---

## 💡 Example

```python
# Train the network
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

# Evaluate on dev set
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
accuracy = get_accuracy(dev_predictions, Y_dev)
print("Dev Set Accuracy:", accuracy)

# Visualize a few predictions
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
