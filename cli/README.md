# ODE Solver Application - Command Line Interface

![License](https://img.shields.io/badge/license-MIT%20with%20Commons%20Clause-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![SciPy](https://img.shields.io/badge/SciPy-1.10.1-brightgreen.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.24.3-brightgreen.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.1-blue.svg)
![NumExpr](https://img.shields.io/badge/NumExpr-2.8.4-blue.svg)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
  - [Scalar ODEs](#scalar-odes)
  - [System of ODEs](#system-of-odes)
- [Examples](#examples)
- [Configuration](#configuration)
- [License](#license)
- [Author](#author)
- [Acknowledgements](#acknowledgements)
- [Contributing](#contributing)
- [Support](#support)

## Introduction

Welcome to the **ODE Solver Application - CLI**! This command-line tool empowers users to solve Ordinary Differential Equations (ODEs) both scalar and systems with ease. Whether you're a student, researcher, or enthusiast, our application provides a robust and flexible interface to input your ODEs, customize parameters, and visualize solutions through comprehensive plots.

Built with powerful libraries like **SciPy** for numerical integration, **NumPy** for efficient computations, **Matplotlib** for plotting, and **NumExpr** for optimized expression evaluation, the ODE Solver CLI is designed for performance and versatility.

## Features

- **Support for Scalar and System ODEs**: Solve single ODEs or systems of ODEs effortlessly.
- **Linear and Non-linear Equations**: Handle both linear and non-linear ODEs with ease.
- **Customizable Parameters**: Specify initial conditions, time ranges, number of steps, and more.
- **Dynamic Visualizations**:
  - **Solution Plots**: Visualize the time evolution of your ODE solutions.
  - **Phase Space Plots**: Explore the phase space with vector fields, nullclines, and flow lines for 2D systems.
- **Performance Optimizations**: Utilize **NumExpr** for efficient evaluation of mathematical expressions.
- **Comprehensive Documentation**: Clear instructions and comments for easy setup and customization.

## Demo

While this is a command-line application, here's a glimpse of the output you can expect:

![ODE Solver CLI Output](https://github.com/fabiobassini/oden/blob/main/demo/demo.png)

*Figure 1: Example output of the ODE Solver CLI showing solution plots.*

## Installation

### Prerequisites

- **Python 3.8+**: Ensure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).
- **pip**: Python package manager. It usually comes bundled with Python.

### Steps

1. Clone the Repository
    ```bash
    git clone https://github.com/fabio-bassini/ode-solver-cli.git
    cd ode-solver-cli

2. Create a Virtual Environment
    It's recommended to use a virtual environment to manage dependencies.
    ```bash
    python3 -m venv venv

3. Activate the virtual environment:

- **Windows**:
    ```bash
    venv\Scripts\activate

- **macOS/Linux**:
    ```bash
    source venv/bin/activate

4. Install Dependencies
    ```bash
    pip install -r requirements.txt

If you don't have a requirements.txt, you can create one with the following content:

    Flask==2.3.2
    numpy==1.24.3
    scipy==1.10.1
    matplotlib==3.7.1
    numexpr==2.8.4

Alternatively, install them individually:

    pip install Flask numpy scipy matplotlib numexpr

5. Usage

The ODE Solver CLI allows you to solve both scalar ODEs and systems of ODEs. Below are detailed instructions on how to use each feature.

### Scalar ODEs
Solve single Ordinary Differential Equations (ODEs) of the form:

**Command Structure**

    python ode_solver.py scalar [options]

**Options**

- --x0 (float): Required. Initial condition 
- --time (float): Required. Maximum simulation time.
- --steps (int): Optional. Number of time steps. (Default: 200)
- --colors (list): Optional. Color(s) for the solution plot.

**For Linear ODEs**:

- --a (float): Required. Coefficient a in the ODE.
- --b_func (str): Required. External force b(t) as a function of t. Example: 'np.cos(t)'.
- --nominal_a (float): Optional. Nominal value of a for comparison.

**For Non-linear ODEs**:

- --f_func (str): Required. Function f(x,t) as an expression of x and t. Example: 'np.sin(x) + t'.

### Examples

**Linear ODE**:
  
    python ode_solver.py scalar --a 2 --x0 1 --time 10 --steps 100 --b_func 'np.cos(t)' --colors 'blue'

**Non-linear ODE**:
    
    python ode_solver.py scalar --x0 1 --time 10 --steps 100 --f_func 'np.sin(x) + t' --colors 'red'

### System of ODEs
Solve systems of Ordinary Differential Equations (ODEs) of the form:

**Command Structure**

    python ode_solver.py system [options]

**Options**

- --X0 (list of floats): Required. Initial conditions X(0). Example: --X0 1 0.
- --time (float): Required. Maximum simulation time.
- --steps (int): Optional. Number of time steps. (Default: 200)
- --colors (list): Optional. Colors for the solution plots.

**For Linear Systems**:

- --A (list of floats): Required. Elements of the coefficient matrix A in row-major order. For a 2x2 matrix, provide 4 elements. Example: --A 1 0 0 1.
- --B_func (str): Required. External force B(t) as a list of expressions. Example: '[0, np.sin(t)]'.

**For Non-linear Systems**:

- --F_func (str): Required. Functions F(X,t) as a list of expressions. Example: '[X[1], -0.1*X[1] - X[0]]'.

**Additional Options for Phase Space Plot**:

- --x_limits (float float): Optional. Limits of the x-axis for the trajectory plot.
- --y_limits (float float): Optional. Limits of the y-axis for the trajectory plot.
- --vector_field_limits (float float float float): Optional. Limits for the vector field axes.
- --vector_grid_size (int): Optional. Grid size for the vector field. (Default: 20)
- --plot_streamlines: Optional. Flag to enable flow lines in the vector field.
- --density (float): Optional. Density of flow lines or vector fields. (Default: 1.0)


### Examples

**Non-linear System with Phase Space Configuration**:

    python ode_solver.py system --X0 0 1 --time 20 --steps 1000 \
    --F_func "[X[1], (0.5 - X[0]**2)*X[1] - X[0]]" \
    --colors 'blue' 'red' --x_limits -3 3 --y_limits -3 3 \
    --vector_field_limits -6 6 -6 6 --vector_grid_size 30 --density 2.0 --plot_streamlines




**Solving a Linear Scalar ODE**:

    python ode_solver.py scalar --a 3 --x0 2 --time 5 --steps 100 --b_func '1.5' --colors 'green'


**Solving a Non-linear System of ODEs**:

    python ode_solver.py system --X0 1 0 --time 10 --steps 500 \
    --F_func "[X[1], -0.1*X[1] - X[0]]" --colors 'purple' 'orange' \
    --vector_field_limits -10 10 -10 10 --vector_grid_size 25 --density 1.5 --plot_streamlines

The solution plots are colored purple and orange. The phase space plot includes a vector field with specified limits, grid size, and density, along with flow lines.

**Solving a Linear System of ODEs**:

    python ode_solver.py system --A 2 3 -1 4 --B_func "[np.sin(t), np.cos(t)]" --X0 1 0 --time 10 --steps 100 --colors blue red

### Configuration

**Backend (ode_solver.py)**
- **Argument Parsing**: Utilizes argparse to handle command-line arguments for different ODE types and configurations.
- **Numerical Integration**: Uses odeint from SciPy for solving ODEs.
- **Expression Evaluation**: Employs numexpr for efficient and secure evaluation of mathematical expressions provided as strings.

**Frontend (Command Line Interface)**
- **Plotting**: Generates plots using Matplotlib.
- **Performance**: Optimizes performance with numexpr for evaluating dynamic functions.
- **Error Handling**: Provides meaningful error messages for invalid inputs or expression errors.


### License

This project is licensed under the MIT License with Commons Clause. See the LICENSE file for details.

### Author

Developed by Fabio Bassini - 2024.

### Acknowledgements

- **Flask** for backend functionality.
- **Plotly.js** for creating interactive plots.
- **Bootstrap Darkly** for the elegant dark theme styling.

### Contributing

Contributions are welcome! Please open an issue first to discuss what you would like to change. For major changes, please open a feature request.

### Support

For support or questions, open a issue