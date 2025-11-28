# 🧠 Neural Network from Scratch

This project implements a **simple neural network from scratch in Python** (no TensorFlow/PyTorch) to classify handwritten digits from the MNIST dataset. It’s designed to help understand the inner workings of a neural network, including forward propagation, backpropagation, and gradient descent.

---

## 📦 Project Overview

The network architecture:

- **Input layer:** 784 neurons (28x28 pixels flattened)  
- **Hidden layer:** 10 neurons, ReLU activation  
- **Output layer:** 10 neurons, Softmax activation  

### Key Components

1. **ReLU Activation**  
   - `ReLU(x) = max(0, x)`  
   - Used in the hidden layer to introduce non-linearity.

2. **Softmax Activation**  
   - Converts raw scores into probabilities for each digit (0–9).  
   - Each output sums to 1, representing a probability distribution.

3. **One-Hot Encoding**  
   - Converts labels into a vector where the correct class is 1, the rest 0.  
   - Example: label `3` → `[0,0,0,1,0,0,0,0,0,0]`.

4. **Forward Propagation**  
   - Computes outputs of each layer:  
     ```text
     Z1 = W1*X + b1
     A1 = ReLU(Z1)
     Z2 = W2*A1 + b2
     A2 = Softmax(Z2)
     ```
   - Produces predictions and intermediate activations for backpropagation.

5. **Backward Propagation**  
   - Computes gradients to update weights and biases:  
     ```text
     dZ2 = A2 - Y_one_hot
     dW2 = (1/m) * dZ2 * A1.T
     db2 = (1/m) * sum(dZ2)
     dZ1 = W2.T * dZ2 * ReLU_deriv(Z1)
     dW1 = (1/m) * dZ1 * X.T
     db1 = (1/m) * sum(dZ1)
     ```

6. **Parameter Update (Gradient Descent)**  
   ```text
   W = W - alpha * dW
   b = b - alpha * db
- `alpha` is the learning rate controlling step size.

---

## 🏋️ Training

Training is done using gradient descent:

```python
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha=0.10, iterations=500)
```
- Prints accuracy every 10 iterations to monitor learning progress.  
- Updates the weights and biases so the network can recognize digits better over time.

---

## 🔍 Evaluation

### Development Set Accuracy

Checks how well the trained network generalizes to unseen data:

```python
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
get_accuracy(dev_predictions, Y_dev)
```
## 🔍 Visual Inspection

You can visualize individual predictions:

```python
test_prediction(index, W1, b1, W2, b2)
```
- Shows the digit, predicted label, and true label.

---

## ⚙️ How to Run

1. Clone the repository.  
2. Ensure you have the required packages: `numpy`, `pandas`, `matplotlib`.  
3. Place `MNIST.csv` in the project folder.  
4. Run the notebook cells in order.  

---

## 💡 Notes

- The network is simple and educational — not optimized for speed or state-of-the-art accuracy.  
- Normalizing pixel values (dividing by 255) is crucial for stable training.  
- ReLU and Softmax were chosen for simplicity and interpretability.

---

## 🤓 Fun Facts

- All code is written from scratch — no high-level libraries were used.  
- You can tweak `alpha` (learning rate) or the number of iterations to see how training changes.  
- Watching the network learn is like watching a student slowly get be

