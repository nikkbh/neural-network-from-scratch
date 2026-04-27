# Neural Network from Scratch

A minimal implementation of a neural network built from scratch using a custom autograd engine, inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd).

No PyTorch. No TensorFlow. Just Python.

---

## What is this?

This project builds a fully functional neural network engine from the ground up — including automatic differentiation (backpropagation) — without relying on any ML frameworks.

It is based on Karpathy's lesson: **"The spelled-out intro to neural networks and backpropagation."**

---

## My Learning Note:
Parameters: 41 
Now, compare this to the scale of GPT or Claude which have Trillions of parameters, but the underlying loop stays the same:
1) Forward Pass - The inputs (data, weights and a bias) are given to a neuron and summed up and then supplied to a loss function which give the accuracy of the output given the inputs.
2) Backward Pass(Backpropagation) - Then we use backpropagation from the given output loss and generate the gradient for each neuron.
3) Update (Stochasitic Update in this NN's case): Then we nudge each paramters with respect to negative gradient and try to reduce the loss
4) This is then iterated over many passed till our model is having the lowest loss and the output is as desired. 


## Core Concepts

- **`Value`** — a scalar value that tracks its own gradient and the operations that created it
- **`Neuron`** — a single unit with weights, a bias, and a `tanh` activation
- **`Layer`** — a list of neurons
- **`MLP`** (Multi-Layer Perceptron) — a stack of layers forming the full network

---

## How it works

### 1. The `Value` class (AutoGrad Engine)

Every number is wrapped in a `Value` object that remembers:
- Its data (the actual number)
- Its gradient (`grad`) with respect to the loss
- A `_backward` function to propagate gradients

```python
a = Value(2.0)
b = Value(3.0)
c = a * b        # c.data = 6.0
c.backward()     # computes dc/da and dc/db automatically
```

### 2. Forward Pass

Input flows through each layer, neuron by neuron:

```
output = tanh(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
```

### 3. Loss Calculation

Compare predictions to targets using **Mean Squared Error (MSE)**:

```python
loss = sum((pred - target)**2 for pred, target in zip(predictions, targets))
```

### 4. Backward Pass

Call `.backward()` on the loss — gradients flow back through every operation automatically.

### 5. Update Weights

Nudge each weight in the direction that reduces loss:

```python
for p in model.parameters():
    p.data -= learning_rate * p.grad
```

---

## Training Loop

```python
for step in range(100):
    # Forward pass
    predictions = [model(x) for x in X]

    # Loss
    loss = sum((pred - target)**2 for pred, target in zip(predictions, y))

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Update
    for p in model.parameters():
        p.data -= 0.01 * p.grad

    print(f"Step {step} | Loss: {loss.data:.4f}")
```

---

## Key Takeaways

| Concept | What you learn |
|---|---|
| `Value` class | How autograd tracks operations and gradients |
| Chain rule | How `.backward()` propagates through a computation graph |
| Neuron / Layer / MLP | How networks are structured from first principles |
| Gradient descent | How weights are updated to minimize loss |

---

## Reference

- 📺 [Andrej Karpathy — micrograd lecture](https://www.youtube.com/watch?v=VMj-3S1tku0)
- 💻 [micrograd on GitHub](https://github.com/karpathy/micrograd)
