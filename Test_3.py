# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 13:41:52 2023

@author: naouf
"""

import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

# Définir les équations différentielles
def system_of_equations(t, y):
    dFe2_dt, dFe3_dt, dH2O2_dt, dHO_dt, dHO2_dt, dO2_minus_dt, dSO4_2_dt, dHSO4_dt, dO2_dt, dSO4_minus_dt, dFeSO4_dt, dFeSO4_plus_dt, dFeSO4_2_minus_dt, dFeOH2_plus_dt, dFeOH2_plus2_dt, dFeHO2_2_plus_dt, dFe2_0OH2_4_plus_dt, dFeOH_OH2_plus_dt, dH_plus_dt, dOH_minus_dt, Fe2_0, H2O2_0, Fe3_0_0_0 = y
    
    n = 2
    F = 96500
    ic = 1
    S = 1
    V = 1
    
    
    # Définition des constantes de vitesse
    k1 = 6.3e4
    k2 = 2.0e-6
    k28 = 3.3e4
    k47 = 1.58e5
    k48 = 1.0e7
    k3 = 3.2e5
    k4 = 1.2e3
    k5 = 3.6e2
    k6 = 1.0e4
    k7 = 5.0e4
    k29 = 5.2e6
    k30 = 8.3e2
    k49 = 1.0e7
    k37 = 7.1e6
    k38 = 1.01e7
    k31 = 9.7e4
    k32 = 5.0e-4
    k33 = 1.3e-4
    k8 = 0  # Négligeable
    k9 = 5.0e4
    k10 = 2.29e8
    k16 = 3.89e9
    k17 = 4.47e7
    k51 = 3.47e8
    k39 = 1.4e4
    k40 = 3.5e2
    k45 = 3.0e5
    k46 = 1.4e4
    k34 = 1.2e4
    k50 = 3.5e6
    k11 = 3.0e5
    k12 = 1.0e10
    k18 = 1.0e10
    k19 = 1.0e10
    k52 = 1.0e10
    k20 = 2.9e7
    k21 = 7.62e3
    k22 = 8.0e3
    k13 = 2.0e-6
    k23 = 1.0e7
    k24 = 1.0e4
    k25 = 1.0e4
    k26 = 3.1e4
    k27 = 1.0e7
    k14 = 2.3e-3
    k35 = 2.0e3
    k36 = 1.0e7
    k15 = 2.3e-3
    
    dFe2_dt = (ic * S) / (n * F * V) - k1 * Fe2_0 * H2O2_0 + k2 * Fe3_0 * H2O2_0_0 - k3 * HO_0 * Fe2_0 - k4 * HO2_0_ * Fe2_0 + k5 * HO2_0 * Fe3_0 - k6 * O2_minus_0 * Fe2_0 + k7 * O2_minus_0 * Fe3_0 - k8 * Fe2_0 * O2_0 + k9 * O2_minus_0 * Fe3_0 - k10 * Fe2_0 * SO4_2_0 - k11 * SO4_minus_0 * Fe2_0 + k12 * FeSO4_0 + k13_0 * FeOH2_plus_0 * H2O2_0 + k14 * FeOH2_plus_02_0 + k15 * FeOH_OH2_plus_0
    dFe3_dt = k1 * Fe2_0 * H2O2_0 - k2 * Fe3_0 * H2O2_0 + k3 * HO_0 * Fe2_0 + k4 * HO2_0 * Fe2_0 - k5 * HO2_0 * Fe3_0 + k6 * O2_minus_0 * Fe2_0 - k7 * O2_minus_0 * Fe3_0 + k8 * Fe2_0 * O2_0 - k9 * O2_minus_0 * Fe3_0 + k11 * SO4_minus_0 * Fe2_0 - k16 * Fe3_0 * SO4_2_0 - k17 * Fe3_0 * SO4_2**2 + k18 * FeSO4_plus_0 + k19 * FeSO4_2_minus_0 - k20 * Fe3_0 - k21 * Fe3_0 - k22 * Fe3_0**2 + k23 * FeOH2_plus_0 * H_plus_0 + k24 * FeOH2_plus_0 * H_plus_0**2 + k25 * FeOH2_plus_0 * H_plus_0**2 - k26 * Fe3_0 * H2O2_0 + k27 * FeOH2_plus_0 * H_plus_0
    dH2O2_dt = -k1 * Fe2_0 * H2O2_0 - k2 * Fe3_0 * H2O2_0 - k28 * H2O2_0 * HO_0 + k4 * HO2_0 * Fe2_0 + k6 * O2_minus_0 * Fe2_0 + k29 * HO_0**2 + k30 * HO2_0**2 + k31 * HO2_0 * O2_minus_0 - k32 * HO2_0 * H2O2_0 - k33 * O2_minus_0 * H2O2_0 - k34 * SO4_minus_0 * H2O2_0 -k13 * FeOH2_plus_0 * H2O2_0 - k26 * Fe3_0 * H2O2_0 + k27 * FeOH2_plus_0 * H_plus_0 - k35 * FeOH2_plus_0 * H2O2_0 + k36 * FeOH_OH2_plus_0 * H_plus_0
    dHO_dt = k1 * Fe2_0 * H2O2_0 - k28 * H2O2_0 * HO_0 - k3 * Fe2_0 * HO_0 - k29 * HO_0**2 - k37 * HO_0 * HO2_0 - k38 * HO_0 * O2_minus_0 + k32 * HO2_0 * H2O2_0 + k33 * O2_minus_0 * H2O2_0 - k39 * SO4_2_0 * HO_0 - k40 * HSO4_0 * HO_0 + k45 * SO4_minus_0 + k46 * SO4_minus_0 * OH_minus_0
    dHO2_dt = k2 * Fe3_0 * H2O2_0 + k28 * H2O2_0 * HO_0 - k47 * HO2_0 + k48 * O2_0_minus_0 * H_plus_0 - k4 * HO2_0 * Fe2_0 - k5 * HO2_0 * Fe3_0 - k30 * HO2_0**2 + k49 * O2_minus_0 * H_plus_0 - k37 * HO_0 * HO2_0 - k31 * HO2_0 * O2_minus_0 - k32 * HO2_0 * H2O2_0 + k34 * SO4_minus_0 * H2O2_0 - k50 * SO4_minus_0 * HO2_0 + k13 * FeOH2_plus_0 * H2O2_0 + k14 * FeHO2_2_plus_0 + k15 * FeOH_OH2_plus_0
    dO2_minus_dt = k47 * HO2_0 - k48 * O2_minus_0 * H_plus_0 - k6 * O2_0_minus_0 * Fe2_0 - k7 * O2_minus_0 * Fe3_0 - k49 * O2_minus_0 * H_plus_0 - k38 * HO_0 * O2_minus_0 - k31 * HO2_0 * O2_0_minus_0 - k33 * O2_minus_0 * H2O2_0 + k8 * Fe2_0 * O2_minus_0 - k9 * Fe3_0 * O2_minus_0
    dSO4_2_0_dt = -k10 * Fe2_0 * SO4_2_0 - k16 * Fe3_0_0 * SO4_2_0 - k17 * Fe3_0_0 * SO4_2_0**2 - k51 * H_plus_0 * SO4_2_0 - k39 * SO4_2_0 * HO_0 + k45 * SO4_minus_0_0 + k46 * SO4_minus_0_0 * OH_minus_0 + k34 * SO4_minus_0_0 * H2O2_0_0 + k50 * SO4_minus_0_0 * HO2_0_0 + k11 * SO4_minus_0_0 * Fe2_0 + k12 * FeSO4 + k18 * FeSO4_plus_0_0 + k19 * FeSO4_2_0_minus_0 + k52 * HSO4_0
    dHSO4_0_dt = k51 * H_plus_0 * SO4_2_0 - k40 * HSO4_0 * HO_0 - k52 * HSO4_0  
    dO2_dt = k5 * HO2_0 * Fe3_0 + k7 * O2_minus_0 * Fe3_0 + k30 * HO2_0**2 + k37 * HO_0 * HO2_0 + k38 * HO_0 * O2_0_minus_0 + k31 * HO2_0 * O2_minus_0 + k32 * HO2_0 * H2O2_0 + k33 * O2_minus_0 * H2O2_0 - k8 * Fe2_0 * O2_0 + k9 * Fe3_0 * O2_minus_0 + k50 * SO4_minus_0 * HO2_0
    dSO4_minus_dt = k39 * SO4_2_0 * HO_0 + k40 * HSO4_0 * HO_0 - k45 * SO4_minus_0 - k46 * SO4_minus_0 * HO_0 - k34 * SO4_minus_0 * H2O2_0 - k50 * SO4_minus_0 * HO2_0 - k11 * SO4_minus_0 * Fe2_0
    dFeSO4_dt = k10 * Fe2_0 * SO4_2_0 - k12 * FeSO4_0
    dFeSO4_plus_dt = k16 * Fe3_0 * SO4_2_0 - k18 * FeSO4_plus_0
    dFeSO4_2_minus_dt = k17 * Fe3_0 * SO4_2_0**2 - k19 * FeSO4_2_minus_0
    dFeOH2_plus_dt = k20 * Fe3_0 - k13 * FeOH2_plus_0 * H2O2_0 - k23 * FeOH2_plus_0 * H_plus_0 + k36 * FeOH_OH2_plus_0 * H_plus_0
    dFeOH2_plus_2_dt = k21 * Fe3_0 - k24 * FeOH2_plus_2_0 * (H_plus_0 ** 2)
    dFeHO2_2_plus_dt = -k27 * FeHO2_2_plus_0 * H_plus_0 - k14 * FeHO2_2_plus_0
    dFe2_OH2_4_plus_dt = k22 * Fe3_0**2 - k25 * Fe2_OH2_4_plus_0 * H_plus_0**2
    dFeOH_OH2_plus_dt = k35 * FeOH2_plus_0 * H2O2_0 - k36 * FeOH_OH2_plus_0 * H_plus_0 - k15 * FeOH_OH2_plus_0
    dH_plus_dt = k2 * Fe3_0 * H2O2_0 + k47 * HO2_0 - k48 * O2_minus_0 * H_plus_0 + k5 * HO2_0 * Fe3_0 - k49 * O2_minus_0 * H_plus_0 + k34 * SO4_minus_0 * H2O2_0 + k50 * SO4_minus_0 * HO2_0 - k51 * H_plus_0 * SO4_2_0 + k45 * SO4_minus_0 + k20 * Fe3_0 + k21 * Fe3_0 + k22 * Fe3_0**2 + k13 * FeOH2_plus_0 * H2O2_0 - k23 * FeOH2_plus_0 * H_plus_0 - k24 * FeHO2_2_plus_0 * H_plus_0**2 - k25 * Fe2_OH2_4_plus_0 * H_plus_0**2 + k26 * Fe3_0 * H2O2_0_0 - k27 * FeHO2_2_plus_0 * H_plus_0 - k35 * FeOH2_plus_0 * H2O2_0 - k36 * FeOH_OH2_plus_0 * H_plus_0 + k52 * HSO4_0
    dOH_minus_dt = k1 * Fe2_0 * H2O2_0 + k3 * Fe2_0 * HO_0 + k4 * HO2_0 * Fe2_0 + k6 * O2_minus_0 * Fe2_0 + k38 * HO_0 * O2_minus_0 + k31 * HO2_0 * O2_minus_0 + k33 * O2_minus_0 * H2O2_0 + k39 * SO4_2_0 * HO_0 - k46 * SO4_minus_0 * OH_minus_0 + k13 * FeOH2_plus_0 * H2O2_0 + k15 * FeOH_OH2_plus_0 
    
    return [dFe2_dt, dFe3_dt, dH2O2_dt, dHO_dt, dHO2_dt, dO2_minus_dt, dSO4_2_dt, dHSO4_dt, dO2_dt, dSO4_minus_dt, dFeSO4_dt, dFeSO4_plus_dt, dFeSO4_2_minus_dt, dFeOH2_plus_dt, dFeOH2_plus2_dt, dFeHO2_2_plus_dt, dFe2OH2_4_plus_dt, dFeOH_OH2_plus_dt, dH_plus_dt, dOH_minus_dt]

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.Fe2_0_label = tk.Label(self, text="Fe2_(0))")
        self.Fe2_0_label.grid(row=0, column=0)
        self.Fe2_0_entry = tk.Entry(self)
        self.Fe2_0_entry.grid(row=0, column=1)
        self.Fe2_0_entry.insert(0, "1")

        self.Fe3_0_label = tk.Label(self, text="Fe3_(0)")
        self.Fe3_0_label.grid(row=1, column=0)
        self.Fe3_0_entry = tk.Entry(self)
        self.Fe3_0_entry.grid(row=1, column=1)
        self.Fe3_0_entry.insert(0, "1")
        
        self.H2O2_0_label = tk.Label(self, text="H2O2_(0)")
        self.H2O2_0_label.grid(row=2, column=0)
        self.H2O2_0_entry = tk.Entry(self)
        self.H2O2_0_entry.grid(row=2, column=1)
        self.H2O2_0_entry.insert(0, "1")
        
        self.HO_0_label = tk.Label(self, text="HO_(0))")
        self.HO_0_label.grid(row=3, column=0)
        self.HO_0_entry = tk.Entry(self)
        self.HO_0_entry.grid(row=3, column=1)
        self.HO_0_entry.insert(0, "1")

        self.HO2_0_label = tk.Label(self, text="HO2_(0)")
        self.HO2_0_label.grid(row=4, column=0)
        self.HO2_0_entry = tk.Entry(self)
        self.HO2_0_entry.grid(row=4, column=1)
        self.HO2_0_entry.insert(0, "1")
        
        self.O2_minus_0_label = tk.Label(self, text="O2_minus_(0)")
        self.O2_minus_0_label.grid(row=5, column=0)
        self.O2_minus_0_entry = tk.Entry(self)
        self.O2_minus_0_entry.grid(row=5, column=1)
        self.O2_minus_0_entry.insert(0, "1")
        
        self.SO4_2_0_label = tk.Label(self, text="SO4_2_(0))")
        self.SO4_2_0_label.grid(row=6, column=0)
        self.SO4_2_0_entry = tk.Entry(self)
        self.SO4_2_0_entry.grid(row=6, column=1)
        self.SO4_2_0_entry.insert(0, "1")
        
        self.HSO4_0_label = tk.Label(self, text="HSO4_(0))")
        self.HSO4_0_label.grid(row=7, column=0)
        self.HSO4_0_entry = tk.Entry(self)
        self.HSO4_0_entry.grid(row=7, column=1)
        self.HSO4_0_entry.insert(0, "1")
        
        self.O2_0_label = tk.Label(self, text="O2_(0)")
        self.O2_0_label.grid(row=8, column=0)
        self.O2_0_entry = tk.Entry(self)
        self.O2_0_entry.grid(row=8, column=1)
        self.O2_0_entry.insert(0, "1")
        
        self.SO4_minus_0_label = tk.Label(self, text="SO4_minus_(0))")
        self.SO4_minus_0_label.grid(row=9, column=0)
        self.SO4_minus_0_entry = tk.Entry(self)
        self.SO4_minus_0_entry.grid(row=9, column=1)
        self.SO4_minus_0_entry.insert(0, "1")

        self.FeSO4_0_label = tk.Label(self, text="FeSO4_(0)")
        self.FeSO4_0_label.grid(row=10, column=0)
        self.FeSO4_0_entry = tk.Entry(self)
        self.FeSO4_0_entry.grid(row=10, column=1)
        self.FeSO4_0_entry.insert(0, "1")
        
        self.FeSO4_plus_0_label = tk.Label(self, text="FeSO4_plus_(0)")
        self.FeSO4_plus_0_label.grid(row=11, column=0)
        self.FeSO4_plus_0_entry = tk.Entry(self)
        self.FeSO4_plus_0_entry.grid(row=11, column=1)
        self.FeSO4_plus_0_entry.insert(0, "1")
        
        self.FeSO4_2_minus_0_label = tk.Label(self, text="FeSO4_2_minus_(0))")
        self.FeSO4_2_minus_0_label.grid(row=12, column=0)
        self.FeSO4_2_minus_0_entry = tk.Entry(self)
        self.FeSO4_2_minus_0_entry.grid(row=12, column=1)
        self.FeSO4_2_minus_0_entry.insert(0, "1")

        self.FeOH2_plus_0_label = tk.Label(self, text="FeOH2_plus_(0)")
        self.FeOH2_plus_0_label.grid(row=13, column=0)
        self.FeOH2_plus_0_entry = tk.Entry(self)
        self.FeOH2_plus_0_entry.grid(row=13, column=1)
        self.FeOH2_plus_0_entry.insert(0, "1")
        
        self.FeOH2_plus2_0_label = tk.Label(self, text="FeOH2_plus2_(0)")
        self.FeOH2_plus2_0_label.grid(row=14, column=0)
        self.FeOH2_plus2_0_entry = tk.Entry(self)
        self.FeOH2_plus2_0_entry.grid(row=14, column=1)
        self.FeOH2_plus2_0_entry.insert(0, "1")
        
        self.FeHO2_2_plus_0_label = tk.Label(self, text="FeHO2_2_plus_(0)")
        self.FeHO2_2_plus_0_label.grid(row=15, column=0)
        self.FeHO2_2_plus_0_entry = tk.Entry(self)
        self.FeHO2_2_plus_0_entry.grid(row=15, column=1)
        self.FeHO2_2_plus_0_entry.insert(0, "1")
        
        self.Fe2OH2_4_plus_0_label = tk.Label(self, text="Fe2OH2_4_plus_(0))")
        self.Fe2OH2_4_plus_0_label.grid(row=16, column=0)
        self.Fe2OH2_4_plus_0_entry = tk.Entry(self)
        self.Fe2OH2_4_plus_0_entry.grid(row=16, column=1)
        self.Fe2OH2_4_plus_0_entry.insert(0, "1")

        self.FeOH_OH2_plus_0_label = tk.Label(self, text="FeOH_OH2_plus_(0)")
        self.FeOH_OH2_plus_0_label.grid(row=17, column=0)
        self.FeOH_OH2_plus_0_entry = tk.Entry(self)
        self.FeOH_OH2_plus_0_entry.grid(row=17, column=1)
        self.FeOH_OH2_plus_0_entry.insert(0, "1")
        
        self.H_plus_0_label = tk.Label(self, text="H_plus_(0)")
        self.H_plus_0_label.grid(row=18, column=0)
        self.H_plus_0_entry = tk.Entry(self)
        self.H_plus_0_entry.grid(row=18, column=1)
        self.H_plus_0_entry.insert(0, "1")
        
        self.OH_minus_0_label = tk.Label(self, text="OH_minus_(0)")
        self.OH_minus_0_label.grid(row=19, column=0)
        self.OH_minus_0_entry = tk.Entry(self)
        self.OH_minus_0_entry.grid(row=19, column=1)
        self.OH_minus_0_entry.insert(0, "1")
    
        self.update_button = tk.Button(self, text="Mettre à jour", command=self.on_update_button_clicked)
        self.update_button.grid(row=20, column=0, columnspan=2)
        
        self.export_button = tk.Button(self, text="Exporter en Excel", command=self.export_to_excel)
        self.export_button.grid(row=20, column=1, columnspan=2)

        self.fig, self.ax = plt.subplots()
        self.line1, = self.ax.plot([], [], label="dFe2/dt")
        self.line2, = self.ax.plot([], [], label="dFe3/dt")
        self.line3, = self.ax.plot([], [], label="dH2O2/dt")
        self.ax.set_xlabel("Temps (t)")
        self.ax.set_ylabel("Solutions dx/dt, dz/dt et df/dt")
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=21, column=0, columnspan=2)
        self.update_graph([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def update_graph(self, initial_conditions):
        t_span = (0, 10)
        solution = solve_ivp(system_of_equations, t_span, initial_conditions, t_eval=np.linspace(0, 10, 100))

        self.line1.set_data(solution.t, solution.y[0])
        self.line2.set_data(solution.t, solution.y[1])
        self.line3.set_data(solution.t, solution.y[2])
        self.line4.set_data(solution.t, solution.y[3])
        self.line5.set_data(solution.t, solution.y[4])
        self.line6.set_data(solution.t, solution.y[5])
        self.line7.set_data(solution.t, solution.y[6])
        self.line8.set_data(solution.t, solution.y[7])
        self.line9.set_data(solution.t, solution.y[8])
        self.line10.set_data(solution.t, solution.y[9])
        self.line11.set_data(solution.t, solution.y[10])
        self.line12.set_data(solution.t, solution.y[11])
        self.line13.set_data(solution.t, solution.y[12])
        self.line14.set_data(solution.t, solution.y[13])
        self.line15.set_data(solution.t, solution.y[14])
        self.line16.set_data(solution.t, solution.y[15])
        self.line17.set_data(solution.t, solution.y[16])
        self.line18.set_data(solution.t, solution.y[17])
        self.line19.set_data(solution.t, solution.y[18])
       

        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

        self.solution = solution  # Stocker la solution pour l'exportation

    def export_to_excel(self):
        if not hasattr(self, 'solution'):
            tk.messagebox.showerror("Erreur", "Aucune donnée à exporter. Cliquez sur 'Mettre à jour' pour générer les courbes.")
            return

        data = {
            "Time": self.solution.t,
            "dx/dt": self.solution.y[0],
            "dz/dt": self.solution.y[1],
            "dF/dt": self.solution.y[2]
        }
        df = pd.DataFrame(data)
        df.to_excel("output_data.xlsx", index=False, engine="openpyxl")

        tk.messagebox.showinfo("Succès", "Les données ont été exportées avec succès dans output_data.xlsx")


    def on_update_button_clicked(self):
        Fe2_0 = float(self.Fe2_0_entry.get())
        Fe3_0 = float(self.Fe3_0_entry.get())
        H2O2_0 = float(self.H2O2_0_entry.get())
        HO_0 = float(self.HO_0_entry.get())
        HO2_0 = float(self.HO2_0_entry.get())
        O2_minus_0 = float(self.O2_minus_0_entry.get())
        SO4_2_0 = float(self.SO4_2_0_entry.get())
        HSO4_0 = float(self.HSO4_0_entry.get())
        O2_0 = float(self.O2_0_entry.get())
        SO4_minus_0 = float(self.SO4_minus_0_entry.get())
        FeSO4_0 = float(self.FeSO4_0_entry.get())
        FeSO4_plus_0 = float(self.FeSO4_plus_0_entry.get())
        FeSO4_2_minus_0 = float(self.FeSO4_2_minus_0_entry.get())
        FeOH2_plus_0 = float(self.FeOH2_plus_0_entry.get())
        FeOH2_plus2_0 = float(self.FeOH2_plus2_0_entry.get()) 
        FeHO2_2_plus_0 = float(self.FeHO2_2_plus_0_entry.get())
        Fe2OH2_4_plus_0 = float(self.Fe2OH2_4_plus_0_entry.get())
        FeOH_OH2_plus_0 = float(self.FeOH_OH2_plus_0_entry.get())
        H_plus_0 = float(self.H_plus_0_entry.get())
        OH_minus_0 = float(self.OH_minus_0_entry.get())
        
        self.update_graph([Fe2_0, Fe3_0, H2O2_0, HO_0, HO2_0, O2_minus_0, SO4_2_0, HSO4_0, O2_0, SO4_minus_0, FeSO4_0, FeSO4_plus_0, FeSO4_2_minus_0, FeOH2_plus_0, FeOH2_plus2_0, FeHO2_2_plus_0, Fe2OH2_4_plus_0, FeOH_OH2_plus_0, H_plus_0, OH_minus_0])

root = tk.Tk()
app = Application(master=root)
app.mainloop()