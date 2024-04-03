from scipy.integrate import odeint, solve_ivp, lsoda
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from lmfit import Parameters, minimize, Model, report_fit, conf_interval
from sklearn.metrics import mean_squared_error
import numdifftools
from PIL import Image
from sklearn.metrics import r2_score
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



import scipy.optimize
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as tkr
import scipy.stats as st
from scipy.stats import scoreatpercentile

colors = [ "#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#999999","#F0E442","#0072B2", "#D55E00" ]
palette = sns.color_palette(colors)

sns.set_theme(context='notebook', style='ticks', font='Arial', 
              font_scale=1.7, 
              rc={"lines.linewidth": 1.8, 'axes.linewidth':2, 
                                  "xtick.major.width":2,"ytick.major.width":2}, 
              palette = palette)

def load_data_frame(excel_name, sheet_name):
    """
    Loads the excel file with data sortened in different sheets. Each sheet corresponds to different parameter change. 
    

    """
    a = pd.read_excel(f'{excel_name}.xlsx', sheet_name= f"{sheet_name}", skiprows=(range(0,2)))
    a.columns = ["time", "F", "Ac", "E1", "E2", "E3","Condition"] 
    return a




def sort_condition(df):
    list_df = []
    conditions = df["Condition"].unique()
    for condition in conditions:
        g = df[df["Condition"] == condition].sort_values(by = ["time", "Condition"]).reset_index(drop = True)
        list_df.append(g)

    return list_df, conditions
def load_initial_conditions(df, k0):
    tspan = np.linspace(df["time"][0], float(df["time"].iloc[-1]) + 1000, 1000)
    F = df["F"][0]
    Ac = df["Ac"][0]
    E1 = df["E1"][0]
    E2 = df["E2"][0]
    E3 = df["E3"][0]
    #[dFdt, dWdt, dAcdt, dAndt, dE1dt, dE2dt, dE3dt, dAn2dt]
    initial_conditions = [F, 0 ,  Ac, 0, E1, E2, E3, 0]  # Add extra zeros to match the unpacking in kinetic_plotting

    k0 = k0  # To be changed depending on pH
    k1 = 0.036
    k2 = .1
    k3 = .1
    k4 = .1
    k5 = 0.1
    k6 = 0.1
    k7 = 0.1
    k8 = 0.1
    k9 = 0.1
    k10 = 1
    k11 = 0.001
    k12 = 0.001
    k13 = 0.001


    minimo3 = 1e-4


    params = Parameters()
    params.add('k0', value=k0, vary=False)
    params.add('k1', value=k1,  min=minimo3, max=1)
    params.add('k2', value=k2,  min=minimo3, max=1)
    params.add('k3', value=k3,  min=minimo3, max=1)
    params.add('k4', value=k4,  min=minimo3, max=1)
    params.add('k5', value=k5, min=minimo3, max=1)
    params.add('k6', value=k6, min=minimo3, max=1)
    params.add('k7', value=k7, min=minimo3, max=1)
    params.add('k8', value=k8,  min=minimo3, max=.1)
    params.add('k9', value=k9,  min=minimo3, max=.1)
    params.add('k10', value=k10,  min=minimo3, max=1)
    params.add('k11', value=k11,  min=minimo3, max=1)
    params.add('k12', value=k12,  min=minimo3, max=1)
    params.add('k13', value=k13,  min=minimo3, max=.1)


    return initial_conditions, params, tspan



def ode_model(z, t, k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13):
    
    """
    takes a vector of the initial concentrations that are previously defined. 
    YOu have to provide also initial guesses for the kinetic constants. 
    You must define as much constants as your system requires. Note that the reactions are expressed as differential
    equations
    
    F: Fuel
    Ac: Precuros
    An: Anhydride
    W: Waste
    E1: Cyclomonomer/phenolester
    E2: Linear dimer (non activated)
    E3: cyclodimer (non activated)

    O: O-acylurea of monomer species
    O2: O-acylure of dimer species

    An2: Linear dimer anhydride
    
    Time is considered to be in **minutes**. Concentrations are in **mM**
    
    """

    #O = ((k1*Ac*F) / (k2+k3))
    #O2 = ((k7*E2*F) / (k8+k9))

    F, W, Ac, An, E1, E2, E3, An2 = z


    dFdt = - k0*F - k1*Ac*F - k7*E2*F
    dWdt = + k0*F + k2*((k1*Ac*F) / (k2+k3)) + k3*((k1*Ac*F) / (k2+k3)) + k8*((k7*E2*F) / (k8+k9)) + k9*((k7*E2*F) / (k8+k9))

    dAcdt = - k1*Ac*F + k3*((k1*Ac*F) / (k2+k3)) + k4*An - k6*An*Ac + k11*E1 + 2*k12*E2
    dAndt = + k2*((k1*Ac*F) / (k2+k3)) - k4*An - k5*An - k6*An*Ac 

    dE1dt = + k5*An - k11*E1
    dE2dt = + k6*An*Ac - k7*E2*F + k9*((k7*E2*F) / (k8+k9)) - k12*E2 + k13*E3
    dE3dt = + k10*An2 - k13*E3

    dAn2dt = + k8*((k7*E2*F) / (k8+k9)) - k10*An2

    return [dFdt, dWdt, dAcdt, dAndt, dE1dt, dE2dt, dE3dt, dAn2dt]

def ode_solver(t, initial_conditions, params):

    """
    Solves the ODE system given initial conditions for both initial concentrations and initial guesses for k
    """
    F, W, Ac, An, E1, E2, E3, An2 = initial_conditions
    k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13 = (
        params['k0'],
        params['k1'],
        params['k2'],
        params['k3'],
        params['k4'],
        params['k5'],
        params['k6'],
        params['k7'],
        params['k8'],
        params['k9'],
        params['k10'],
        params['k11'],
        params['k12'],
        params['k13'],
    )

    res = odeint(ode_model, initial_conditions, t, args=(k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13))

    return res


def kinetic_plotting(z, t, params):
    
    """
    This is a complementary function that is similar to ode_model. Is only used for doing the plotting. 
    Time is considered to be in **minutes**. Concentrations are in **mM**
    
    F: Fuel
    Ac: Precuros
    An: Anhydride
    W: Waste
    E1: Cyclomonomer/phenolester
    E2: Linear dimer (non activated)
    E3: cyclodimer (non activated)

    O: O-acylurea of monomer species
    O2: O-acylure of dimer species

    An2: Linear dimer anhydride
    
    Time is considered to be in **minutes**. Concentrations are in **mM**
    
    """

    k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13 = params
    F, W, Ac, An, E1, E2, E3, An2 = z
    
    #O = (k1*Ac*F) / (k2+k3)
    #O2 = (k7*E2*F) / (k8+k9)



    dFdt = - k0*F - k1*Ac*F - k7*E2*F
    dWdt = + k0*F + k2*((k1*Ac*F) / (k2+k3)) + k3*((k1*Ac*F) / (k2+k3)) + k8*((k7*E2*F) / (k8+k9)) + k9*((k7*E2*F) / (k8+k9))

    dAcdt = - k1*Ac*F + k3*((k1*Ac*F) / (k2+k3)) + k4*An - k6*An*Ac + k11*E1 + 2*k12*E2
    dAndt = + k2*((k1*Ac*F) / (k2+k3)) - k4*An - k5*An - k6*An*Ac 

    dE1dt = + k5*An - k11*E1
    dE2dt = + k6*An*Ac - k7*E2*F + k9*((k7*E2*F) / (k8+k9)) - k12*E2 + k13*E3
    dE3dt = + k10*An2 - k13*E3

    dAn2dt = + k8*((k7*E2*F) / (k8+k9)) - k10*An2

    return [dFdt, dWdt, dAcdt, dAndt, dE1dt, dE2dt, dE3dt, dAn2dt]


def get_fitted_curve(initial_conditions, tspan, params):
    """
    Simulates data based on initial conditions and fitted values for the kinetic constants.

    Parameters:
    initial_conditions: Initial conditions.
    tspan: Time window.
    params: Fitted parameters.

    Returns: "F", "W", "Ac", "An", "E1", "E2", "E3", "An2"
    pd.DataFrame: Fitted data.
    """
    y = pd.DataFrame(odeint(kinetic_plotting, initial_conditions, tspan, args=(params,)), columns=["F", "W", "Ac", "An", "E1", "E2", "E3", "An2" ])
    y['min'] = tspan
    return y

def error(params, initial_conditions, t, data):
    sol = ode_solver(t, initial_conditions, params)
    sol_subset = sol[:, [0, 2, 4, 5, 6]]  # Extract columns corresponding to F, Ac, E1, E2, E3
    error = data - sol_subset
    return error

def error_no_EDC(params, initial_conditions, t, data):
    sol = ode_solver(t, initial_conditions, params)
    sol_subset = sol[:, [2, 4, 5, 6]]  # Extract columns corresponding to F, Ac, E1, E2, E3
    error = data[:, [1, 2, 3, 4]] - sol_subset
    return error


def RMSEerror(params, initial_conditions, t, data): # Root-mean-square deviation

    sol = ode_solver(t, initial_conditions, params)
    sol_subset = sol[:, [0, 2, 4, 5, 6]] # Extract columns corresponding to F, Ac, E1, E2, E3
    error = data - sol_subset
    return (np.sqrt(np.mean((error)**2, axis=0)))

def plot_fitted(df, y):
    
    """
    
    Creates a 4 column plots. In each column there is a different reagent. It plots both the original data and
    the fitted
    
    df: data
    y: fitted
    
    
    """
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5,figsize = (15, 3), 
                                    sharey = False, sharex = True)
    sns.scatterplot(data = df, x = 'time', y = 'F', ax = ax1, color = sns.color_palette(palette)[0])
    sns.lineplot(data = y, x = 'min', y = 'F', ax = ax1, alpha = 0.5, color = sns.color_palette(palette)[0])

    sns.scatterplot(data = df, x = 'time', y = 'Ac', ax = ax2, color = sns.color_palette(palette)[1])
    sns.lineplot(data = y, x = 'min', y = 'Ac', ax = ax2, alpha = 0.5, color = sns.color_palette(palette)[1])

    sns.scatterplot(data = df, x = 'time', y = 'E1', ax = ax3, color = sns.color_palette(palette)[2])
    sns.lineplot(data = y, x = 'min', y = 'E1', ax = ax3, alpha = 0.5, color = sns.color_palette(palette)[2])

    sns.scatterplot(data = df, x = 'time', y = 'E2', ax = ax4, color = sns.color_palette(palette)[3])
    sns.lineplot(data = y, x = 'min', y = 'E2', ax = ax4, alpha = 0.5, color = sns.color_palette(palette)[3])


    sns.scatterplot(data = df, x = 'time', y = 'E3', ax = ax5, color = sns.color_palette(palette)[4])
    sns.lineplot(data = y, x = 'min', y = 'E3', ax = ax5, alpha = 0.5, color = sns.color_palette(palette)[4])


    ax1.set(xlabel = 'Time [min]', ylabel = 'EDC [mM]', xticks = (0, 150, 300),xlim = (-20, 320))
    ax2.set(xlabel = 'Time [min]', ylabel = 'Acid [mM]', xticks = (0, 150, 300),xlim = (-20, 320))
    ax3.set(xlabel = 'Time [min]', ylabel = 'E1 [mM]', xticks = (0, 150, 300),xlim = (-20, 320))
    ax4.set(xlabel = 'Time [min]', ylabel = 'E2 [mM]', xticks = (0, 150, 300),xlim = (-20, 320))
    ax5.set(xlabel = 'Time [min]', ylabel = 'E3 [mM]', xticks = (0, 150, 300),xlim = (-20, 320))

    plt.tight_layout()
    
    return fig, (ax1, ax2, ax3, ax4, ax5)

def process_data(ic, excel_name, sheet_name, k0_input, condition_id):
    df = load_data_frame(excel_name, sheet_name)
    dfs, cond = sort_condition(df)
    df1 = dfs[condition_id]
    initial_conditions, params, tspan = load_initial_conditions(df1, k0_input)
    data = df1[['F', "Ac", "E1", "E2", "E3"]].values
    t = sorted(df1['time'])
    return initial_conditions, params, tspan, data, t, df1

