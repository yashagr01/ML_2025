def gradient_descent_until_error_increases(x, y, beta0, beta1, lr):
    n = len(x)

    # Function to compute Mean Squared Error
    def compute_error(beta0, beta1):
        predictions = [beta0 + beta1 * xi for xi in x]
        errors = [(pred - yi) ** 2 for pred, yi in zip(predictions, y)]
        return sum(errors) / n

    current_error = compute_error(beta0, beta1)
    iterations = 0

    while True:
        iterations += 1

        # Compute gradients
        d_beta0 = (-2/n) * sum([(y[i] - (beta0 + beta1 * x[i])) for i in range(n)])
        d_beta1 = (-2/n) * sum([(y[i] - (beta0 + beta1 * x[i])) * x[i] for i in range(n)])

        # Update parameters
        new_beta0 = beta0 - lr * d_beta0
        new_beta1 = beta1 - lr * d_beta1

        # Compute error after update
        new_error = compute_error(new_beta0, new_beta1)

        # Stop if error stops decreasing
        if new_error >= current_error:
            break

        # Accept updates
        beta0, beta1, current_error = new_beta0, new_beta1, new_error

    return iterations, beta0, beta1, current_error


# ---------------------------
# User Input Section
# ---------------------------

# Input x values
x = list(map(float, input("Enter x values separated by space: ").split()))
# Input y values
y = list(map(float, input("Enter y values separated by space: ").split()))

# Check input lengths
if len(x) != len(y):
    print("Error: x and y must have the same number of elements.")
    exit()

beta0 = float(input("Enter initial beta0: "))
beta1 = float(input("Enter initial beta1: "))
lr = float(input("Enter learning rate (e.g., 0.01): "))

# Run gradient descent
iterations, beta0_final, beta1_final, final_error = gradient_descent_until_error_increases(x, y, beta0, beta1, lr)

print("\n============================")
print("Gradient Descent Completed")
print("============================")
print(f"Iterations (until error stopped decreasing): {iterations}")
print(f"Final beta0: {beta0_final}")
print(f"Final beta1: {beta1_final}")
print(f"Final Error: {final_error}")
