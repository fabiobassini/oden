# =============================================================================
# ODE Solver Application - Backend
# -----------------------------------------------------------------------------
# @autor: Fabio Bassini
# @data: 2024-11-03
# @versione: 1.0.0
# @licenza: MIT License with Commons Clause
# @copyright: © 2024 Fabio Bassini
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

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import argparse
import numexpr as ne


"""
Programma per risolvere sistemi di equazioni differenziali ordinarie (ODE) scalari o matriciali, lineari e non lineari.

Questo programma permette di:
    - Risolvere equazioni differenziali ordinarie (ODE) scalari della forma:
        - Lineare: dx/dt = a * x + b(t)
        - Non lineare: dx/dt = f(x, t)
    - Risolvere sistemi di equazioni differenziali ordinarie (ODE) in forma matriciale:
        - Lineare: dX/dt = A * X + B(t)
        - Non lineare: dX/dt = F(X, t)
    - Personalizzare la visualizzazione dei risultati, inclusi colori per le curve, grafici delle soluzioni e spazi delle fasi con campi vettoriali.

Utilizzo:
--------
Per eseguire il programma, scegliere tra la modalità 'scalar' per risolvere un'ODE scalare
o 'system' per risolvere un sistema di ODE. I parametri vengono specificati tramite linea di comando.

1. Risolvere un'ODE scalare:
    -------------------------
    Esegui il seguente comando:

    ```bash
    python ode_solver.py scalar [opzioni]
    ```

    Opzioni comuni:
    - `--x0` (float): Condizione iniziale x(0).
    - `--time` (float): Tempo massimo di simulazione.
    - `--steps` (int): Numero di passi di tempo (default: 200).
    - `--colors` (list): (Opzionale) Colore per la soluzione.

    Opzioni per ODE lineare:
    - `--a` (float): Coefficiente di moltiplicazione per x nell'ODE dx/dt = a * x + b(t).
    - `--b_func` (str): Espressione di b(t) come funzione di `t`, ad esempio `'np.sin(t)'`.
    - `--nominal_a` (float): (Opzionale) Valore nominale di `a` per confronto.

    Opzioni per ODE non lineare:
    - `--f_func` (str): Espressione di f(x, t) come funzione di `x` e `t`, ad esempio `'np.sin(x) + t'`.

    Esempi:

    - ODE lineare:
    ```bash
    python ode_solver.py scalar --a 2 --x0 1 --time 10 --steps 100 --b_func 'np.cos(t)' --colors 'blue'
    ```

    - ODE non lineare:
    ```bash
    python ode_solver.py scalar --x0 1 --time 10 --steps 100 --f_func 'np.sin(x) + t' --colors 'red'
    ```

2. Risolvere un sistema di ODE:
    ----------------------------
    Esegui il seguente comando:

    ```bash
    python ode_solver.py system [opzioni]
    ```

    Opzioni comuni:
    - `--X0` (list of float): Condizione iniziale del vettore di stato, ad esempio `--X0 1 0`.
    - `--time` (float): Tempo massimo di simulazione.
    - `--steps` (int): Numero di passi di tempo (default: 200).
    - `--colors` (list): (Opzionale) Lista di colori per le variabili di stato tracciate.

    Opzioni per sistema lineare:
    - `--A` (list of float): Elementi della matrice dei coefficienti A.
    - `--B_func` (str): Lista di espressioni per B(t) in funzione di `t`, ad esempio `'[0, np.sin(t)]'`.

    Opzioni per sistema non lineare:
    - `--F_func` (str): Lista di espressioni per F(X, t) in funzione di `X` e `t`, ad esempio `'[X[1], -0.1*X[1] - X[0]]'`.

    Opzioni aggiuntive per il grafico dello spazio delle fasi:
    - `--x_limits` (float float): Limiti dell'asse x per il grafico della traiettoria.
    - `--y_limits` (float float): Limiti dell'asse y per il grafico della traiettoria.
    - `--vector_field_limits` (float float float float): Limiti dell'asse x e y per il campo vettoriale, ad esempio `--vector_field_limits -5 5 -5 5`.
    - `--vector_grid_size` (int): Dimensione della griglia del campo vettoriale (default: 20).
    - `--plot_streamlines`: Attiva il tracciamento delle linee di flusso nel campo vettoriale.
    - `--density` (float): Densità delle linee di flusso o del campo vettoriale (default: 1.0).

    Esempi:

    - Sistema non lineare con specifica dei limiti e densità del campo vettoriale:
    ```bash
    python ode_solver.py system --X0 0 1 --time 20 --steps 1000 \
    --F_func "[X[1], (0.5 - X[0]**2)*X[1] - X[0]]" \
    --colors 'blue' 'red' --x_limits -3 3 --y_limits -3 3 \
    --vector_field_limits -6 6 -6 6 --vector_grid_size 30 --density 2.0 --plot_streamlines
    ```

Note:
-----
- Il programma supporta la visualizzazione di:
    - Grafici delle soluzioni per posizione, velocità o altre variabili di stato nel tempo.
    - Spazi delle fasi per sistemi bidimensionali, inclusi campi vettoriali, nullclini e linee di flusso.
- Assicurarsi che:
    - Per ODE scalari, specificare o `--a` e `--b_func` per equazioni lineari, oppure `--f_func` per equazioni non lineari.
    - Per sistemi di ODE, specificare o `--A` e `--B_func` per sistemi lineari, oppure `--F_func` per sistemi non lineari.
- `--b_func`, `--f_func` e `--F_func` devono essere sicuri; supportano funzioni standard NumPy come `np.sin`, `np.cos`, `np.exp`.

Autore:
-------
Fabio B

Data:
-----
2024
"""


class ODESystemSolver:
    """
    Classe per risolvere sistemi di equazioni differenziali ordinarie (ODE), lineari e non lineari.

    Gestisce sistemi del tipo:
        - Lineare: dX/dt = A * X + B(t)
        - Non lineare: dX/dt = F(X, t)

    Attributi:
        - A: matrice dei coefficienti A (solo per sistemi lineari).
        - B: funzione che calcola B(t) (solo per sistemi lineari).
        - F_func: funzione che calcola F(X, t) (solo per sistemi non lineari).
        - X0: vettore delle condizioni iniziali.
        - t_vals: array di valori temporali in cui calcolare la soluzione.
    """

    def __init__(self, X0, t_vals, A=None, B=None, F_func=None):
        """
        Inizializza il risolutore con i parametri dati.

        Args:
            - X0 (np.ndarray): Vettore delle condizioni iniziali.
            - t_vals (np.ndarray): Array dei tempi in cui calcolare la soluzione.
            - A (np.ndarray, opzionale): Matrice dei coefficienti A.
            - B (callable, opzionale): Funzione che restituisce B(t).
            - F_func (callable, opzionale): Funzione che restituisce F(X, t).
        """
        self.X0 = X0
        self.t_vals = t_vals
        self.A = A
        self.B = B
        self.F_func = F_func

    def solve(self):
        """
        Risolve il sistema di ODE.

        Returns:
            - X_vals (np.ndarray): Array delle soluzioni per ogni tempo in t_vals.
        """
        if self.F_func is not None:
            # Sistema non lineare
            def ode_func(X, t):
                return self.F_func(X, t)
        else:
            # Sistema lineare
            def ode_func(X, t):
                return self.A @ X + self.B(t)

        X_vals = odeint(ode_func, self.X0, self.t_vals)
        return X_vals

    def plot_solution(self, X_vals, colors=None):
        """
        Traccia le soluzioni delle variabili di stato nel tempo.

        Args:
            - X_vals (np.ndarray): Array delle soluzioni calcolate.
            - colors (list, opzionale): Lista dei colori per le curve.
        """
        plt.figure(figsize=(12, 6))
        num_vars = X_vals.shape[1]

        if colors is None:
            colors = plt.cm.viridis(np.linspace(0, 1, num_vars))

        for i in range(num_vars):
            plt.plot(
                self.t_vals,
                X_vals[:, i],
                label=f'$x_{{{i+1}}}(t)$',
                color=colors[i % len(colors)]
            )

        plt.xlabel('Tempo $t$')
        plt.ylabel('Valori di $x_i(t)$')
        title = 'Soluzione del Sistema di ODE'
        if self.F_func is not None:
            title += ' Non Lineare'
        else:
            title += ' Lineare'
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()




    def plot_phase_space(self, X_vals, x_limits=None, y_limits=None, vector_field_limits=None,
                         vector_grid_size=20, plot_streamlines=False, density=1.0):
        """
        Traccia lo spazio delle fasi con campo vettoriale, linee di flusso e nullclini per sistemi bidimensionali.

        Args:
            - X_vals (np.ndarray): Array delle soluzioni calcolate.
            - x_limits (tuple, opzionale): Limiti dell'asse x per il grafico.
            - y_limits (tuple, opzionale): Limiti dell'asse y per il grafico.
            - vector_field_limits (tuple, opzionale): Limiti dell'asse x e y per il campo vettoriale come (xmin, xmax, ymin, ymax).
            - vector_grid_size (int, opzionale): Numero di punti nella griglia del campo vettoriale.
            - plot_streamlines (bool, opzionale): Se True, traccia le linee di flusso.
            - density (float, opzionale): Densità delle linee di flusso.
        """
        if X_vals.shape[1] != 2:
            raise ValueError("La funzione plot_phase_space supporta solo sistemi bidimensionali.")

        # Determina i limiti per il grafico della traiettoria
        if x_limits is not None:
            x1_min_plot, x1_max_plot = x_limits
        else:
            x1_min_plot, x1_max_plot = np.min(X_vals[:, 0]), np.max(X_vals[:, 0])
            x1_margin = (x1_max_plot - x1_min_plot) * 0.1
            x1_min_plot -= x1_margin
            x1_max_plot += x1_margin

        if y_limits is not None:
            x2_min_plot, x2_max_plot = y_limits
        else:
            x2_min_plot, x2_max_plot = np.min(X_vals[:, 1]), np.max(X_vals[:, 1])
            x2_margin = (x2_max_plot - x2_min_plot) * 0.1
            x2_min_plot -= x2_margin
            x2_max_plot += x2_margin

        # Determina i limiti per il campo vettoriale
        if vector_field_limits is not None:
            x1_min, x1_max, x2_min, x2_max = vector_field_limits
        else:
            x1_min, x1_max = x1_min_plot, x1_max_plot
            x2_min, x2_max = x2_min_plot, x2_max_plot

        # Genera una griglia per il campo vettoriale
        X1, X2 = np.meshgrid(
            np.linspace(x1_min, x1_max, vector_grid_size),
            np.linspace(x2_min, x2_max, vector_grid_size)
        )

        # Calcola le componenti del campo vettoriale
        U = np.zeros_like(X1)
        V = np.zeros_like(X2)
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                X_point = np.array([X1[i, j], X2[i, j]])
                if self.F_func is not None:
                    F_val = self.F_func(X_point, 0)
                else:
                    F_val = self.A @ X_point + self.B(0)
                U[i, j] = F_val[0]
                V[i, j] = F_val[1]

        # Crea il grafico del campo vettoriale e della traiettoria
        plt.figure(figsize=(8, 8))

        # Determina i limiti complessivi del grafico per includere sia la traiettoria che il campo vettoriale
        x1_min_total = min(x1_min_plot, x1_min)
        x1_max_total = max(x1_max_plot, x1_max)
        x2_min_total = min(x2_min_plot, x2_min)
        x2_max_total = max(x2_max_plot, x2_max)

        if plot_streamlines:
            # Traccia le linee di flusso
            plt.streamplot(X1, X2, U, V, color='gray', density=density, linewidth=0.8, arrowsize=1)
        else:
            # Traccia il campo vettoriale
            plt.quiver(X1, X2, U, V, color='gray', alpha=0.5)
        
        # Traccia la traiettoria del sistema
        plt.plot(X_vals[:, 0], X_vals[:, 1], color='b', label='Traiettoria')

        # Traccia le nullclini dove F1=0 e F2=0
        plt.contour(X1, X2, U, levels=[0], colors='r', linestyles='dashed', linewidths=1)
        plt.contour(X1, X2, V, levels=[0], colors='g', linestyles='dashed', linewidths=1)

        plt.xlabel('$x_1(t)$')
        plt.ylabel('$x_2(t)$')
        plt.title('Spazio delle Fasi con Campo Vettoriale, Linee di Flusso e Nullclini')
        plt.grid(True)
        plt.legend()
        # Imposta i limiti degli assi per includere sia la traiettoria che il campo vettoriale
        plt.xlim(x1_min_total, x1_max_total)
        plt.ylim(x2_min_total, x2_max_total)
        plt.show()



class SingleODESolver:
    """
    Classe per risolvere equazioni differenziali ordinarie (ODE) scalari, lineari e non lineari.

    Gestisce equazioni del tipo:
        - Lineare: dx/dt = a * x + b(t)
        - Non lineare: dx/dt = f(x, t)

    Attributi:
        - a: parametro a nell'ODE (solo per ODE lineari).
        - b_func: funzione che calcola b(t) (solo per ODE lineari).
        - f_func: funzione che calcola f(x, t) (solo per ODE non lineari).
        - x0: condizione iniziale.
        - t_vals: array di valori temporali in cui calcolare la soluzione.
    """

    def __init__(self, x0, t_vals, a=None, b_func=None, f_func=None):
        """
        Inizializza il risolutore con i parametri dati.

        Args:
            - x0 (float): Condizione iniziale x(0).
            - t_vals (np.ndarray): Array dei tempi in cui calcolare la soluzione.
            - a (float, opzionale): Parametro a nell'ODE lineare.
            - b_func (callable, opzionale): Funzione che restituisce b(t).
            - f_func (callable, opzionale): Funzione che restituisce f(x, t).
        """
        self.x0 = x0
        self.t_vals = t_vals
        self.a = a
        self.b_func = b_func
        self.f_func = f_func

    def solve(self):
        """
        Risolve l'ODE scalare.

        Returns:
            - x_vals (np.ndarray): Array delle soluzioni per ogni tempo in t_vals.
        """
        if self.f_func is not None:
            # ODE non lineare
            def ode_func(x, t):
                return self.f_func(x, t)
        else:
            # ODE lineare
            def ode_func(x, t):
                return self.a * x + self.b_func(t)

        x_vals = odeint(ode_func, self.x0, self.t_vals)
        return x_vals



    # def plot_solution(self, x_vals, label=None, color=None):
    #     """
    #     Traccia la soluzione dell'ODE scalare nel tempo.

    #     Args:
    #         - x_vals (np.ndarray): Array delle soluzioni calcolate.
    #         - label (str, opzionale): Etichetta per la curva.
    #         - color (str, opzionale): Colore della curva.
    #     """
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(self.t_vals, x_vals, label=label, color=color)
    #     plt.xlabel('Tempo $t$')
    #     plt.ylabel('$x(t)$')
    #     title = 'Soluzione dell\'ODE Scalare'
    #     if self.f_func is not None:
    #         title += ' Non Lineare'
    #     else:
    #         title += ' Lineare'
    #     plt.title(title)
    #     if label:
    #         plt.legend()
    #     plt.grid(True)
    #     plt.show()

    def plot_solution(self, x_vals, label=None, color=None, show_plot=True):
        """
        Traccia la soluzione dell'ODE scalare nel tempo.

        Args:
            - x_vals (np.ndarray): Array delle soluzioni calcolate.
            - label (str, opzionale): Etichetta per la curva.
            - color (str, opzionale): Colore della curva.
            - show_plot (bool, opzionale): Se True, mostra il grafico.
        """
        plt.plot(self.t_vals, x_vals, label=label, color=color)
        plt.xlabel('Tempo $t$')
        plt.ylabel('$x(t)$')
        title = 'Soluzione dell\'ODE Scalare'
        if self.f_func is not None:
            title += ' Non Lineare'
        else:
            title += ' Lineare'
        plt.title(title)
        if label:
            plt.legend()
        plt.grid(True)
        if show_plot:
            plt.show()



def main():
    """
    Funzione principale che gestisce l'analisi degli argomenti e l'esecuzione del programma.

    Permette di risolvere ODE scalari o sistemi di ODE, lineari o non lineari,
    specificando i parametri da linea di comando.

    Aggiunte:
    - È possibile specificare i limiti degli assi x e y per il grafico della traiettoria con `--x_limits` e `--y_limits`.
    - È possibile specificare i limiti del campo vettoriale con `--vector_field_limits`.
    - È possibile controllare la densità del campo vettoriale o delle linee di flusso con l'opzione `--density`.
    """
    parser = argparse.ArgumentParser(description="Risoluzione di ODE.")
    subparsers = parser.add_subparsers(dest='type', help='Tipo di ODE: scalar o system')

    # Parser per ODE scalare
    scalar_parser = subparsers.add_parser('scalar', help='Risoluzione di un\'ODE scalare')
    scalar_parser.add_argument('--x0', type=float, required=True, help="Condizione iniziale x0.")
    scalar_parser.add_argument('--time', type=float, required=True, help="Tempo massimo di simulazione.")
    scalar_parser.add_argument('--steps', type=int, default=200, help="Numero di passi di tempo.")
    scalar_parser.add_argument('--colors', nargs='+', help="Colori per le curve del grafico.")

    # Opzioni per ODE lineare
    scalar_parser.add_argument('--a', type=float, help="Parametro a nell'ODE dx/dt = a x + b(t).")
    scalar_parser.add_argument('--b_func', type=str, help="Forza esterna b(t) come espressione di t.")

    # Opzioni per ODE non lineare
    scalar_parser.add_argument('--f_func', type=str, help="Funzione f(x, t) come espressione di x e t.")

    scalar_parser.add_argument('--nominal_a', type=float, help="Valore nominale del parametro a.")

    # Parser per sistema di ODE
    system_parser = subparsers.add_parser('system', help='Risoluzione di un sistema di ODE')
    system_parser.add_argument('--X0', nargs='+', type=float, required=True, help="Condizione iniziale X0.")
    system_parser.add_argument('--time', type=float, required=True, help="Tempo massimo di simulazione.")
    system_parser.add_argument('--steps', type=int, default=200, help="Numero di passi di tempo.")
    system_parser.add_argument('--colors', nargs='+', help="Colori per le curve del grafico.")

    # Opzioni per sistema lineare
    system_parser.add_argument('--A', nargs='+', type=float, help="Elementi della matrice dei coefficienti A.")
    system_parser.add_argument('--B_func', type=str, help="Forza esterna B(t) come lista di espressioni di t.")

    # Opzioni per sistema non lineare
    system_parser.add_argument('--F_func', type=str, help="Funzioni F(X, t) come lista di espressioni di X e t.")

    # Opzioni aggiuntive per il grafico dello spazio delle fasi
    system_parser.add_argument('--x_limits', nargs=2, type=float, help="Limiti dell'asse x per il grafico della traiettoria.")
    system_parser.add_argument('--y_limits', nargs=2, type=float, help="Limiti dell'asse y per il grafico della traiettoria.")
    system_parser.add_argument('--vector_field_limits', nargs=4, type=float, help="Limiti dell'asse x e y per il campo vettoriale (xmin xmax ymin ymax).")
    system_parser.add_argument('--vector_grid_size', type=int, default=20, help="Dimensione della griglia del campo vettoriale.")
    system_parser.add_argument('--plot_streamlines', action='store_true', help="Traccia le linee di flusso nel campo vettoriale.")
    system_parser.add_argument('--density', type=float, default=1.0, help="Densità delle linee di flusso o del campo vettoriale.")

    args = parser.parse_args()
    t_vals = np.linspace(0, args.time, args.steps)

    if args.type == 'scalar':
        if args.f_func:
            # ODE non lineare
            f_func_str = args.f_func

            def f_func(x, t):
                allowed_funcs = {'np': np, 'sin': np.sin, 'cos': np.cos, 'exp': np.exp, 'log': np.log}
                local_dict = {'x': x, 't': t}
                try:
                    return eval(f_func_str, {"__builtins__": None}, {**allowed_funcs, **local_dict})
                except Exception as e:
                    raise ValueError(f"Errore nell'elaborazione di f(x, t): {e}")

            solver = SingleODESolver(args.x0, t_vals, f_func=f_func)
            x_vals = solver.solve()
            color = args.colors[0] if args.colors else 'b'
            solver.plot_solution(x_vals, label='Soluzione', color=color)

        elif args.a is not None and args.b_func:
            # # ODE lineare
            # b_func_str = args.b_func

            # def b_func(t):
            #     allowed_funcs = {'np': np, 'sin': np.sin, 'cos': np.cos, 'exp': np.exp}
            #     try:
            #         return eval(b_func_str, {"t": t, "__builtins__": None}, allowed_funcs)
            #     except Exception as e:
            #         raise ValueError(f"Errore nell'elaborazione di b(t): {e}")

            # a_values = [args.a]
            # labels = ['Soluzione con $a$ reale']
            # colors = args.colors if args.colors else ['b']

            # if args.nominal_a is not None:
            #     a_values.append(args.nominal_a)
            #     labels.append('Soluzione con $a$ nominale')
            #     if len(colors) < 2:
            #         colors.append('r')

            # for idx, a in enumerate(a_values):
            #     solver = SingleODESolver(args.x0, t_vals, a=a, b_func=b_func)
            #     x_vals = solver.solve()
            #     solver.plot_solution(x_vals, label=labels[idx], color=colors[idx % len(colors)])
                # ODE lineare
            b_func_str = args.b_func

            def b_func(t):
                local_dict = {'t': t}
                try:
                    return ne.evaluate(b_func_str, local_dict)
                except Exception as e:
                    raise ValueError(f"Errore nell'elaborazione di b(t): {e}")

            a_values = [args.a]
            labels = ['Soluzione con $a$ reale']
            colors = args.colors if args.colors else ['b']

            if args.nominal_a is not None:
                a_values.append(args.nominal_a)
                labels.append('Soluzione con $a$ nominale')
                if len(colors) < 2:
                    colors.append('r')

            # Crea una nuova figura prima di iniziare a tracciare
            plt.figure(figsize=(12, 6))

            for idx, a in enumerate(a_values):
                solver = SingleODESolver(args.x0, t_vals, a=a, b_func=b_func)
                x_vals = solver.solve()
                # Traccia le soluzioni sulla stessa figura
                solver.plot_solution(x_vals, label=labels[idx], color=colors[idx % len(colors)], show_plot=False)

            # Mostra il grafico dopo aver tracciato tutte le soluzioni
            plt.show()
        else:
            raise ValueError("Specificare sia '--a' e '--b_func' per ODE lineare o '--f_func' per ODE non lineare.")

    elif args.type == 'system':
        X0 = np.array(args.X0)
        n = len(X0)

        if args.F_func:
            # Sistema non lineare
            F_func_strs = args.F_func.strip()
            if F_func_strs.startswith('[') and F_func_strs.endswith(']'):
                # Rimuove le parentesi quadre
                F_func_strs = F_func_strs[1:-1]
            else:
                raise ValueError("F_func deve essere una lista di espressioni racchiusa tra parentesi quadre.")

            # Divide la stringa in una lista di espressioni
            F_func_list = []
            expr = ''
            bracket_level = 0
            for char in F_func_strs:
                if char == ',' and bracket_level == 0:
                    F_func_list.append(expr.strip())
                    expr = ''
                else:
                    expr += char
                    if char in '([':
                        bracket_level += 1
                    elif char in ')]':
                        bracket_level -= 1
            if expr:
                F_func_list.append(expr.strip())

            if len(F_func_list) != n:
                raise ValueError("Il numero di funzioni in F_func deve corrispondere alla dimensione di X0.")

            def F_func(X, t):
                allowed_funcs = {'np': np, 'sin': np.sin, 'cos': np.cos, 'exp': np.exp, 'log': np.log}
                local_dict = {'X': X, 't': t}
                try:
                    return np.array([
                        eval(expr, {"__builtins__": None}, {**allowed_funcs, **local_dict})
                        for expr in F_func_list
                    ])
                except Exception as e:
                    raise ValueError(f"Errore nell'elaborazione di F(X, t): {e}")

            solver = ODESystemSolver(X0, t_vals, F_func=F_func)
            X_vals = solver.solve()
            colors = args.colors
            solver.plot_solution(X_vals, colors=colors)
            if n == 2:
                x_limits = tuple(args.x_limits) if args.x_limits else None
                y_limits = tuple(args.y_limits) if args.y_limits else None
                if args.vector_field_limits:
                    vector_field_limits = tuple(args.vector_field_limits)
                    if len(vector_field_limits) != 4:
                        raise ValueError("vector_field_limits richiede 4 valori: xmin xmax ymin ymax")
                else:
                    vector_field_limits = None
                vector_grid_size = args.vector_grid_size
                plot_streamlines = args.plot_streamlines
                density = args.density
                solver.plot_phase_space(
                    X_vals,
                    x_limits=x_limits,
                    y_limits=y_limits,
                    vector_field_limits=vector_field_limits,
                    vector_grid_size=vector_grid_size,
                    plot_streamlines=plot_streamlines,
                    density=density
                )
                if vector_grid_size < 30 and density > 1:
                    print("Avviso: per ottenere un effetto significativo con 'density', si consiglia di aumentare 'vector_grid_size'.")
        elif args.A and args.B_func:
            # Sistema lineare
            if len(args.A) != n * n:
                raise ValueError("Il numero di elementi di A deve essere un quadrato perfetto corrispondente alla dimensione di X0.")

            A = np.array(args.A).reshape((n, n))

            B_func_strs = args.B_func.strip()
            if B_func_strs.startswith('[') and B_func_strs.endswith(']'):
                # Rimuove le parentesi quadre
                B_func_strs = B_func_strs[1:-1]
            else:
                raise ValueError("B_func deve essere una lista di espressioni racchiusa tra parentesi quadre.")

            # Divide la stringa in una lista di espressioni
            B_func_list = []
            expr = ''
            bracket_level = 0
            for char in B_func_strs:
                if char == ',' and bracket_level == 0:
                    B_func_list.append(expr.strip())
                    expr = ''
                else:
                    expr += char
                    if char in '([':
                        bracket_level += 1
                    elif char in ')]':
                        bracket_level -= 1
            if expr:
                B_func_list.append(expr.strip())

            if len(B_func_list) != n:
                raise ValueError("Il numero di funzioni in B_func deve corrispondere alla dimensione di X0.")

            def B_func(t):
                allowed_funcs = {'np': np, 'sin': np.sin, 'cos': np.cos, 'exp': np.exp}
                try:
                    return np.array([
                        eval(expr, {"t": t, "__builtins__": None}, allowed_funcs)
                        for expr in B_func_list
                    ])
                except Exception as e:
                    raise ValueError(f"Errore nell'elaborazione di B(t): {e}")

            solver = ODESystemSolver(X0, t_vals, A=A, B=B_func)
            X_vals = solver.solve()
            colors = args.colors
            solver.plot_solution(X_vals, colors=colors)
            if n == 2:
                x_limits = tuple(args.x_limits) if args.x_limits else None
                y_limits = tuple(args.y_limits) if args.y_limits else None
                if args.vector_field_limits:
                    vector_field_limits = tuple(args.vector_field_limits)
                    if len(vector_field_limits) != 4:
                        raise ValueError("vector_field_limits richiede 4 valori: xmin xmax ymin ymax")
                else:
                    vector_field_limits = None
                vector_grid_size = args.vector_grid_size
                plot_streamlines = args.plot_streamlines
                density = args.density
                solver.plot_phase_space(
                    X_vals,
                    x_limits=x_limits,
                    y_limits=y_limits,
                    vector_field_limits=vector_field_limits,
                    vector_grid_size=vector_grid_size,
                    plot_streamlines=plot_streamlines,
                    density=density
                )
        else:
            raise ValueError("Specificare sia '--A' e '--B_func' per sistema lineare o '--F_func' per sistema non lineare.")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
