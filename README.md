![License](https://img.shields.io/badge/license-MIT%20with%20Commons%20Clause-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-brightgreen.svg)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3%2B-blue.svg)
![Plotly](https://img.shields.io/badge/Plotly.js-2.18.2-blue.svg)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [License](#license)
- [Author](#author)
- [Acknowledgements](#acknowledgements)
- [Contributing](#contributing)
- [Support](#support)

## Introduction

Welcome to the **ODEN Application**! This web-based tool allows users to solve Ordinary Differential Equations (ODEs) with ease. Whether you're a student, researcher, or enthusiast, our application provides a user-friendly interface to input your ODEs, customize parameters, and visualize solutions through interactive plots.

Built with a modern **dark theme**, the application ensures a comfortable user experience, especially during extended usage periods. Leveraging powerful technologies like **Flask** for the backend and **Plotly.js** for dynamic visualizations, the ODE Solver is both robust and visually appealing.

## Features

- **Support for Scalar and System ODEs**: Solve single ODEs or systems of ODEs effortlessly.
- **Interactive UI**: Intuitive forms to input ODE parameters with helpful tooltips.
- **Dark Theme**: Modern and sleek dark interface for reduced eye strain.
- **Dynamic Visualizations**:
  - **Solution Plots**: View the time evolution of your ODE solutions.
  - **Vector Fields**: Visualize the phase space and vector fields for systems of ODEs.
  - **3D Vector Fields**: For three-variable systems, explore 3D vector fields and flow lines.
- **Responsive Design**: Accessible on various devices, from desktops to mobile phones.
- **Comprehensive Documentation**: Clear instructions and comments for easy setup and customization.

## Demo

![ODE Solver Screenshot](https://github.com/fabiobassini/oden/blob/main/demo/oden.png)

*Figure 1: Screenshot of the ODEN Application showcasing the dark theme and interactive plots.*

## Installation

### Prerequisites

- **Python 3.8+**: Ensure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).
- **pip**: Python package manager. It usually comes bundled with Python.

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/fabiobassini/oden.git
   cd oden

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt

4. Run the application:
    ```bash
    python server.py

5. Access the app by navigating to http://127.0.0.1:5000 in your web browser.

### Usage

1. Launch the app and open the sidebar to configure your ODE.
2. Choose between scalar or system ODE types.
3. Enter the parameters for your ODE:
4. For scalar ODEs: define initial conditions, coefficients, and functions.
5. For system ODEs: define matrices and function vectors.
6. Click "Solve" to generate plots of solutions and vector fields.

### Configuration

- **ODE Parameters**: Customize the initial conditions, matrix coefficients, and external functions.
- **Plot Settings**: Adjust layout options directly in the plotSolution, plotVectorField, and plotVectorField3D functions in app.js for more control over the visual output.

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