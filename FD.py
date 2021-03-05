# Finite difference approximations to derivatives
import numpy as np
import matplotlib.pyplot as plt

plot_error = True
order_of_accuracy = False

# Define function for f(x) and f'(x)
def f(x):
    return np.arctan(3 * x)

def f_prime(x):
    return 3 / (1 + (3*x)**2)


# Forward difference approximation
def forward_diff(f, dx, x):
    D = (f(x + dx) - f(x)) / dx
    return D

def centred_diff(f, dx, x):
    return (f(x + dx) - f(x - dx)) / (2 * dx)


if plot_error:

    # Test the function
    x0 = np.linspace(-5, 5, 500)
    dx = [0.04, 0.02, 0.01, 0.005]
    f_prime_exact = f_prime(x0)

    # Plot the function and the derivative
    x = np.linspace(-5, 5, 500)
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    ax[0].plot(x, f_prime_exact, label='Exact')
    # ax[0].plot(x, f_prime_approx, label='Forward difference')
    ax[0].set_title('Derivative')

    for step_size in dx:
        f_prime_approx = centred_diff(f, step_size, x0)

        ax[1].plot(x, np.abs(f_prime_exact - f_prime_approx), label=f'dx = {step_size}')
        
        
    ax[1].set_title('Error')
    ax[1].legend()

    plt.show()

if order_of_accuracy:
    # Pick a point
    x0 = 1.
    f_prime_exact = f_prime(x0)
    
    # Choose values for the step size
    dx = np.logspace(-2, -15, 14, base=2)
    
    err = []
    for step_size in dx:
        f_prime_approx = centred_diff(f, step_size, x0)
        err.append(abs(f_prime_exact - f_prime_approx))
    
    fig, ax = plt.subplots()
    ax.plot(np.log(dx), np.log(err), 'rx')
    ax.set(xlabel='Log step size', ylabel='Log error magnitude', title='Error for forward difference')
    
    plt.show()
    
    # Find the line of best fit
    line_coeffs = np.polyfit(np.log(dx), np.log(err), 1)
    print(line_coeffs[0])
