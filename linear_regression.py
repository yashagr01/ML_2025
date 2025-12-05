def linear_regression(x, y):
    n = len(x)

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    # Mean-center X
    x_norm = [xi - mean_x for xi in x]

    # Compute covariance & variance using normalized X
    covariance = sum(x_norm[i] * (y[i] - mean_y) for i in range(n))
    variance = sum(x_norm[i] ** 2 for i in range(n))

    m = covariance / variance        # slope
    c = mean_y                       # intercept becomes mean(Y)

    return m, c


# ----- Main Program -----
n = int(input("Enter number of data points: "))

x = []
y = []

print("\nEnter values for X:")
for i in range(n):
    x_val = float(input(f"X[{i+1}]: "))
    x.append(x_val)

print("\nEnter values for Y:")
for i in range(n):
    y_val = float(input(f"Y[{i+1}]: "))
    y.append(y_val)

m, c = linear_regression(x, y)

print("\n--- Linear Regression Equation (mean-normalized X) ---")
print(f"y = {m:.4f}x + {c:.4f}")
