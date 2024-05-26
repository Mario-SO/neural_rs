# 🧠 Simple Neural Network in Rust 🤖

Welcome to the **Simple Neural Network** project implemented in **Rust**! 🦀 This program demonstrates the fundamentals of a neural network with forward and backpropagation from scratch.

The code is inspired from [this video](https://www.youtube.com/watch?v=LA4I3cWkp1E) in which the guy coded the same but in C language

## 🚀 Getting Started

### Prerequisites

Ensure you have [Rust](https://www.rust-lang.org/) installed. You can install Rust using `rustup`:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/Mario-SO/neural_rs/tree/main
    cd neural_rs
    ```

2. **Run the project**:
    ```sh
    cargo run
    ```

## 🎯 Purpose

This simple neural network aims to classify XOR gate outputs 🌓. The network consists of:
-  An input layer with 2 neurons
-  One hidden layer with 2 neurons
-  An output layer with 1 neuron

## 📝 How It Works

### Initialization

-  **Weights and Biases**: Initialized randomly between 0 and 1. 

### Forward Pass

1. **Hidden Layer**: Calculates activations using input data, weights, and biases, then applies the sigmoid function.
2. **Output Layer**: Computes the final output using hidden layer activations, weights, and biases, then applies the sigmoid function.

### Backpropagation

-  Adjusts weights and biases to minimize the error between the predicted and actual outputs using the learning rate and derivatives of the sigmoid function.

## 📊 Training & Accuracy

-  Trains the network over several epochs and prints the accuracy on the training set after training.
-  Displays final weights and biases after training is complete.

## 📈 Example Output

Each training iteration shows:
-  Input Data
-  Predicted Output
-  Expected Output

After training, it prints the final accuracy, weights, and biases.

Example:
```
Input: [0.0, 0.0] -> Predicted output: [0.045] (Expected: [0.0])
Input: [0.0, 1.0] -> Predicted output: [0.912] (Expected: [1.0])
...
Final accuracy: 100.00%
Final hidden weights: [[0.2, 0.3], [0.4, 0.1]]
Final output weights: [[0.5, 0.7]]
...
```

## 🛠️ Key Functions

1. **initialize_weights**: Initializes weights randomly.
2. **sigmoid**: Sigmoid activation function.
3. **sigmoid_derivative**: Derivative of the sigmoid function for backpropagation.

## 🤝 Contributions

Feel free to fork this repository and contribute by submitting a pull request. Please ensure all changes are well tested.

---

Happy coding! 🎉

🌟 **Star this repository** if you found it helpful!