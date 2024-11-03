# =============================================================================
# ODE Solver Application - Backend
# -----------------------------------------------------------------------------
# @autor: Fabio Bassini
# @data: 2024-11-03
# @versione: 1.0.0
# @licenza: MIT License with Commons Clause
# @copyright: © 2024 Fabio Bassini
# =============================================================================
# Descrizione:
# Questo file Python implementa il backend per il solver delle Equazioni
# Differenziali Ordinarie (ODE). Utilizza Flask per gestire le richieste
# HTTP e integra Plotly.js per la visualizzazione dei grafici.
# =============================================================================

# MIT License

# Copyright (c) 2024 Fabio Bassini

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# =============================================================================

# Commons Clause License Condition v1.0

# The Software is provided to you by the Licensor under the License, as defined
# below, subject to the following condition.

# Without limiting other conditions in the License, the grant of rights under the
# License will not include, and the License Condition below prohibits, the right
# to Sell the Software.

# "Sell" means providing to third parties, for a fee or other consideration,
# access to the Software as a service, hosting the Software for third parties,
# or otherwise conveying the Software to third parties for a fee or other
# consideration.

# For purposes of the foregoing, “the License” shall mean the MIT License.
# =============================================================================



from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from scipy.integrate import odeint
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/solve", methods=["POST"])
def solve_ode():
    data = request.get_json()
    ode_type = data.get("type")

    # Validazione dei parametri comuni
    try:
        time = float(data.get("time"))
        steps = int(data.get("steps"))
        t_vals = np.linspace(0, time, steps)
    except Exception as e:
        return jsonify({"message": f"Errore nei parametri comuni: {e}"}), 400

    # ODE Scalare
    if ode_type == "scalar":
        # Recupera i parametri specifici
        try:
            x0 = float(data.get("x0"))
            a = float(data.get("a"))
            b_func_str = data.get("b_func")
            f_func_str = data.get("f_func")
        except Exception as e:
            return jsonify({"message": f"Errore nei parametri scalari: {e}"}), 400

        # Definisci le funzioni b(t) e f(x, t)
        try:
            if b_func_str:
                def b_func(t):
                    return eval(b_func_str, {"t": t, "np": np})
            else:
                b_func = lambda t: 0

            if f_func_str:
                def f_func(x, t):
                    return eval(f_func_str, {"x": x, "t": t, "np": np})
            else:
                f_func = lambda x, t: 0

            # Definisci l'ODE
            def ode_func(x, t):
                return a * x + b_func(t) + f_func(x, t)

        except Exception as e:
            return jsonify({"message": f"Errore nella definizione delle funzioni: {e}"}), 400

        try:
            # Risolvi l'ODE
            x_vals = odeint(ode_func, x0, t_vals)
            solution = x_vals.flatten().tolist()

            return jsonify({
                "message": "Soluzione calcolata",
                "solution": solution,
                "t_vals": t_vals.tolist()
            })

        except Exception as e:
            return jsonify({"message": f"Errore durante la risoluzione dell'ODE scalare: {e}"}), 400

    # Sistema di ODE
    elif ode_type == "system":
        X0_str = data.get("X0")
        A_str = data.get("A")
        B_func_str = data.get("B_func")
        F_func_str = data.get("F_func")

        if not X0_str:
            return jsonify({"message": "Condizione iniziale X0 mancante"}), 400

        try:
            X0 = np.array([float(val) for val in X0_str.split()])
        except Exception as e:
            return jsonify({"message": f"Errore nella condizione iniziale X0: {e}"}), 400

        num_vars = len(X0)  # Numero di variabili del sistema

        if F_func_str:
            try:
                def F_func(X, t):
                    return eval(F_func_str, {"X": X, "t": t, "np": np})
                ode_func = lambda X, t: F_func(X, t)
            except Exception as e:
                return jsonify({"message": f"Errore nella funzione F(X, t): {e}"}), 400
        elif A_str and B_func_str:
            try:
                A = np.array([float(val) for val in A_str.split()]).reshape(num_vars, -1)
                def B_func(t):
                    return np.array(eval(B_func_str, {"t": t, "np": np}))
                ode_func = lambda X, t: A @ X + B_func(t)
            except Exception as e:
                return jsonify({"message": f"Errore nei parametri A o B(t): {e}"}), 400
        else:
            return jsonify({"message": "Parametri mancanti per risolvere il sistema di ODE"}), 400

        try:
            # Risolvi l'ODE
            X_vals = odeint(ode_func, X0, t_vals)
            solution = X_vals.tolist()

            vector_field = None
            vector_field_3d = None
            flow_lines_3d = None  # Inizializziamo le linee di flusso

            if num_vars == 2:
                # Definisci una griglia per il campo vettoriale 2D
                grid_size = 30  # Dimensione della griglia per maggiore dettaglio
                x_min, x_max = np.min(X_vals[:, 0]), np.max(X_vals[:, 0])
                y_min, y_max = np.min(X_vals[:, 1]), np.max(X_vals[:, 1])
                x_margin = (x_max - x_min) * 0.3
                y_margin = (y_max - y_min) * 0.3
                x_vals_grid = np.linspace(x_min - x_margin, x_max + x_margin, grid_size)
                y_vals_grid = np.linspace(y_min - y_margin, y_max + y_margin, grid_size)
                X_grid, Y_grid = np.meshgrid(x_vals_grid, y_vals_grid)

                U = np.zeros_like(X_grid)
                V = np.zeros_like(Y_grid)

                for i in range(grid_size):
                    for j in range(grid_size):
                        X_point = np.array([X_grid[i, j], Y_grid[i, j]])
                        t = 0  # Puoi scegliere un tempo specifico o lasciare t=0
                        if F_func_str:
                            F_val = F_func(X_point, t)
                        else:
                            F_val = ode_func(X_point, t)
                        U[i, j] = F_val[0]
                        V[i, j] = F_val[1]

                vector_field = {
                    "X": X_grid.tolist(),
                    "Y": Y_grid.tolist(),
                    "U": U.tolist(),
                    "V": V.tolist()
                }

            elif num_vars == 3:
                # Definisci una griglia per il campo vettoriale 3D
                grid_size = 10  # Dimensione della griglia (ridotto per motivi di performance)
                x_min, x_max = np.min(X_vals[:, 0]), np.max(X_vals[:, 0])
                y_min, y_max = np.min(X_vals[:, 1]), np.max(X_vals[:, 1])
                z_min, z_max = np.min(X_vals[:, 2]), np.max(X_vals[:, 2])
                x_margin = (x_max - x_min) * 0.3
                y_margin = (y_max - y_min) * 0.3
                z_margin = (z_max - z_min) * 0.3
                x_vals_grid = np.linspace(x_min - x_margin, x_max + x_margin, grid_size)
                y_vals_grid = np.linspace(y_min - y_margin, y_max + y_margin, grid_size)
                z_vals_grid = np.linspace(z_min - z_margin, z_max + z_margin, grid_size)
                X_grid, Y_grid, Z_grid = np.meshgrid(x_vals_grid, y_vals_grid, z_vals_grid)

                U = np.zeros_like(X_grid)
                V = np.zeros_like(Y_grid)
                W = np.zeros_like(Z_grid)

                for i in range(grid_size):
                    for j in range(grid_size):
                        for k in range(grid_size):
                            X_point = np.array([X_grid[i, j, k], Y_grid[i, j, k], Z_grid[i, j, k]])
                            t = 0  # Puoi scegliere un tempo specifico o lasciare t=0
                            if F_func_str:
                                F_val = F_func(X_point, t)
                            else:
                                F_val = ode_func(X_point, t)
                            U[i, j, k] = F_val[0]
                            V[i, j, k] = F_val[1]
                            W[i, j, k] = F_val[2]

                vector_field_3d = {
                    "X": X_grid.tolist(),
                    "Y": Y_grid.tolist(),
                    "Z": Z_grid.tolist(),
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }

                # Calcolo delle Linee di Flusso 3D
                # Definiamo alcuni punti iniziali per le linee di flusso
                num_flow_lines = 5  # Numero di linee di flusso da generare
                # Genera punti iniziali distribuiti nella griglia
                flow_line_start_points = [
                    [x_vals_grid[int(grid_size/4)], y_vals_grid[int(grid_size/4)], z_vals_grid[int(grid_size/4)]],
                    [x_vals_grid[int(3*grid_size/4)], y_vals_grid[int(grid_size/4)], z_vals_grid[int(grid_size/4)]],
                    [x_vals_grid[int(grid_size/4)], y_vals_grid[int(3*grid_size/4)], z_vals_grid[int(grid_size/4)]],
                    [x_vals_grid[int(3*grid_size/4)], y_vals_grid[int(3*grid_size/4)], z_vals_grid[int(grid_size/4)]],
                    [x_vals_grid[int(grid_size/2)], y_vals_grid[int(grid_size/2)], z_vals_grid[int(grid_size/2)]]
                ]

                flow_lines_3d = []
                for start_point in flow_line_start_points[:num_flow_lines]:
                    flow_solution = odeint(ode_func, start_point, t_vals)
                    flow_lines_3d.append(flow_solution.tolist())

            return jsonify({
                "message": "Soluzione calcolata",
                "solution": solution,
                "t_vals": t_vals.tolist(),
                "vector_field": vector_field,
                "vector_field_3d": vector_field_3d,  # Campo vettoriale 3D
                "flow_lines_3d": flow_lines_3d  # Linee di flusso 3D
            })
        except Exception as e:
            return jsonify({"message": f"Errore durante la risoluzione del sistema di ODE: {e}"}), 400

    else:
        return jsonify({"message": "Tipo di ODE non supportato"}), 400

if __name__ == "__main__":
    app.run(debug=True)



# [X[1], (0.5 - X[0]**2)*X[1] - X[0]]

# [10*(X[1] - X[0]), X[0]*(28 - X[2]) - X[1], X[0]*X[1] - (8/3)*X[2]]