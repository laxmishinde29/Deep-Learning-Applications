import math
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# STEP 1 : Sigmoid Activation Function
# ---------------------------------------------------------
def sigmoid(z):
    """
    Sigmoid Function
    Formula: 1 / (1 + e^(-z))
    """
    return 1 / (1 + math.exp(-z))

# ---------------------------------------------------------
# STEP 2 : Neuron Forward Pass
# ---------------------------------------------------------
def Marvellous_neuron_forward(inputs, weights, bias):

    print("\n----- NEURON CALCULATION START -----\n")

    print("Inputs (x) :", inputs)
    print("Weights (w) :", weights)
    print("Bias (b) :", bias)

    # Weighted sum
    z = sum(w * x for w, x in zip(weights, inputs)) + bias
    print("\nStep 1 : Weighted Sum")
    print("z =", z)

    # Activation
    y_hat = sigmoid(z)
    print("\nStep 2 : Activation Function")
    print("Activation Function : Sigmoid")
    print("Output (ŷ) =", y_hat)

    print("\n----- NEURON CALCULATION END -----\n")

    return z, y_hat

# ---------------------------------------------------------
# STEP 3 : Plot Sigmoid Function
# ---------------------------------------------------------
def plot_sigmoid():

    z_values = np.linspace(-10, 10, 200)
    sigmoid_values = 1 / (1 + np.exp(-z_values))

    plt.figure(figsize=(8, 5))
    plt.plot(z_values, sigmoid_values, label="Sigmoid Function", linewidth=2, color="blue")

    plt.axhline(y=0, color="black", linewidth=0.5)
    plt.axhline(y=1, color="black", linewidth=0.5)
    plt.axvline(x=0, color="gray", linestyle="--")

    plt.title("Sigmoid Activation Function", fontsize=16)
    plt.xlabel("Input (z)", fontsize=14)
    plt.ylabel("Output (Probability)", fontsize=14)

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.show()

# ---------------------------------------------------------
# STEP 4 : Main Function
# ---------------------------------------------------------
def main():

    print("\n========= MARVELLOUS SIGMOID NEURON =========\n")

    inputs = [1.0, 2.0, 3.0]
    weights = [0.6, 0.4, -0.2]
    bias = 0.5

    z, y_hat = Marvellous_neuron_forward(inputs, weights, bias)

    print("Final z :", z)
    print("Final y_hat :", y_hat)

    plot_sigmoid()

# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()