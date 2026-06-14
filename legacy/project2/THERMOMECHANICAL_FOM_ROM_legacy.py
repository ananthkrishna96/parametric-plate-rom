#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
THERMOMECHANICAL_FOM_ROM.py

Python script converted from THERMOMECHANICAL_FOM_ROM.ipynb.
Notebook markdown cells are preserved as comments, and notebook cell boundaries
are preserved using # %% markers for editor/Jupyter compatibility.

Conversion policy:
- Code-cell source is kept in original order.
- Comments inside code cells are preserved.
- Markdown cells are converted to commented text.
- IPython-only magic lines are preserved as comments because they are not valid
  in plain Python execution.
"""

# %% [markdown] Cell 1 | id: 4f6b26d6-548a-4f64-9a10-b043d7a2a24d
# # **Parametrized ROM for Thermomechanical Orthotropic Kirchhoff Plates with Unilateral Foundation Contact**
#
# This notebook presents a **comprehensive FE framework** for **thermomechanical bending of orthotropic Kirchhoff plates** with **unilateral foundation contact** and **penalty-based domain decomposition**, implementing **intrusive/non-intrusive ROMs** via **POD-Galerkin**, **POD-Interpolation (PODI)**, **POD-Gaussian Process Regression (POD-GPR)**, and **POD-Neural Network (POD-NN)** for efficient parametric analysis.
#
# ---
#
# ## 1. Physical Formulation
#
# ### Thermomechanical Orthotropic Kirchhoff Plate Theory
# Thin **structurally orthotropic** plate (thickness $t$) with direct rigidity parametrization, subjected to **mechanical loading** and **thermal bending effects**. Governing PDE for transverse deflection $w(\mathbf{x})$ over Winkler foundation:
#
# $$D_{x}\frac{\partial^4 w}{\partial x^4} + 2(D_{xy} + 2D_s)\frac{\partial^4 w}{\partial x^2 \partial y^2} + D_{y}\frac{\partial^4 w}{\partial y^4} + k_s w = q(x, y) + q_T(x,y)$$
#
# **Parameters:** $D_x, D_y$ (flexural rigidities), $D_{xy}$ (coupling), $D_s$ (torsional), $k_s$ (foundation stiffness), $q(x,y)$ (mechanical loading), $q_T(x,y)$ (thermal loading from temperature field)
#
# ### Unilateral Foundation & Domain Decomposition
# **Foundation energy:** $\psi_{\text{foundation}}(w) = \frac{1}{2} k_s w^2$ (compression, $w < 0$) or $\frac{1}{2} \varepsilon k_s w^2$ (uplift, $w \geq 0$)
#
# **Interface penalty:** $\Pi_{\text{interface}} = \sum_{i<j} \int_{\Gamma_{ij}} \left[ \frac{\alpha}{2} [w]^2 + \frac{\beta}{2} [\partial_{\mathbf{n}} w]^2 \right] d\Gamma$
#
# **Penalty scaling:** $\alpha = \frac{C_{\alpha}(p) D_{\text{eff}}}{h_F^3}$, $\beta = \frac{C_{\beta}(p) D_{\text{eff}}}{h_F}$ with $D_{\text{eff}} = \sqrt{D_x D_y}$
#
# ### Parametrized Weak Form
# Find $w \in V^{\text{dd}}$ such that $\forall v \in V^{\text{dd}}$:
#
# $$\sum_{i=1}^{M} \int_{\Omega_i} \left[ D_x \partial_{xx} w_i \partial_{xx} v_i + D_y \partial_{yy} w_i \partial_{yy} v_i + 2D_{xy} \partial_{xx} w_i \partial_{yy} v_i + 4D_s \partial_{xy} w_i \partial_{xy} v_i \right] d\Omega + \sum_{i=1}^{M} \int_{\Omega_i} k_s \, g(w_i) \, v_i \, d\Omega + \sum_{i<j} \int_{\Gamma_{ij}} \left[ \alpha [w] [v] + \beta [\partial_{\mathbf{n}} w] [\partial_{\mathbf{n}} v] \right] d\Gamma = \sum_{i=1}^{M} \int_{\Omega_i} \big(q + q_T\big) v_i \, d\Omega$$
#
# ---
#
# ## 2. Numerical Implementation
# **Discretization:** $C^0$ Lagrange elements, IPG stabilization, Solution function spaces | **Solution:** Newton method, analytic Jacobian, direct solver | **Thermal coupling:** prescribed $\Delta T(x,y)$ field or full 3D heat conduction solution with through-thickness integration for $q_T$ *(implemented via the through-thickness linear coefficient $T^1=\partial \Delta T/\partial z$, with the common choice $T^1=\Delta T_{\text{through}}/t$)* | **Parameters:** $\boldsymbol{\mu} = [D_x, D_y, D_{xy}, D_s, k_s, f, \Delta T, \alpha_x, \alpha_y]$
#
# ---
#
# ## 3. POD Framework
# From snapshots $\mathbf{S} = [w(\boldsymbol{\mu}_1), \ldots, w(\boldsymbol{\mu}_N)]$: **Energy-based inner product** → **SVD decomposition** → **Basis truncation** → **Reduced approximation:** $w(\mathbf{x}, \boldsymbol{\mu}) \approx \sum_{i=1}^r c_i(\boldsymbol{\mu}) \phi_i(\mathbf{x})$
#
# Snapshots include both **mechanical** and **thermal** contributions to the response.
#
# ---
#
# ## 4. ROM Methodologies
#
# ### Intrusive ROMs
# **POD-Galerkin:** Projects thermomechanical equations onto reduced space $\mathbf{Z}^T \mathcal{F}(\mathbf{Z} \mathbf{c}; \boldsymbol{\mu}) = \mathbf{0}$ with Newton iteration (physics preservation, high accuracy)
#
# ### Non-Intrusive ROMs
# **PODI:** Independent coefficient interpolation via RBF/Linear methods (simple implementation, smooth predictions)
#
# **POD-GPR:** Gaussian Process Regression for parameter-to-coefficient mapping with uncertainty quantification (probabilistic predictions, thermal sensitivity analysis, confidence intervals)
#
# **POD-NN:** Multi-layer feedforward network for parameter-to-coefficient mapping with supervised learning (fast evaluation, automatic features, scalable to thermal parameters)
#
# ---
#
# ## 5. Framework Capabilities
#  **High-fidelity FEM** for parametrized thermomechanical orthotropic plates with foundation contact  **Thermal bending coupling** via prescribed or computed temperature fields  **Domain decomposition** with penalty interface coupling  **Comprehensive ROM suite** (intrusive/non-intrusive)  **Performance benchmarking** (accuracy, speedup, robustness)  **Parametric studies** for combined mechanical–thermal design exploration  **Unified DD-ROM implementation**
#
# ---
#
# ## Appendix: Thermal Load Contributions
#
# For a temperature field $\Delta T(x,y,z)$, define the **through-thickness linear coefficient**:
#
# $$T^1(x,y) = \frac{\partial \Delta T}{\partial z}(x,y,z) \quad \text{(for a linear profile, }T^1=\Delta T_{\text{through}}/t\text{)}.$$
#
# The thermal bending moments (consistent with the $D$-based parametrization used in the solver) are:
#
# $$M_x^T = \big(D_x \alpha_x + D_{xy} \alpha_y\big) T^1, \qquad 
# M_y^T = \big(D_{xy} \alpha_x + D_y \alpha_y\big) T^1, \qquad 
# M_{xy}^T = 0.$$
#
# - **Uniform temperature rise $\Delta T=$ const. (no gradient):**  
#   $T^1 = 0 \;\Rightarrow\; M^T = 0 \;\Rightarrow\; q_T = 0$ (no bending, only in-plane thermal effects)
#
# - **Linear temperature gradient through thickness:**  
#   $\Delta T(z) = \Delta T_0 + \Delta T_1 z$ with $\Delta T_1 = T^1$  
#   ⇒ Thermal moments given by the formulas above; bending scales **linearly** with $T^1$ (and thus with $\Delta T_{\text{through}}$).
#
# - **Equivalent transverse load from thermal moments:**  
#
#   $$q_T(x,y) = -\Bigg(\frac{\partial^2 M_x^T}{\partial x^2} + 2 \frac{\partial^2 M_{xy}^T}{\partial x \partial y} + \frac{\partial^2 M_y^T}{\partial y^2}\Bigg) = -\Bigg(\frac{\partial^2 M_x^T}{\partial x^2} + \frac{\partial^2 M_y^T}{\partial y^2}\Bigg)$$  
#
#   (the last equality uses $M_{xy}^T = 0$ in this framework).
#
# ---

# %% Cell 2 | id: 56d9a609-fd6e-4a8c-aa1c-7c86c6d14fb0
# ─── Standard library ───────────────────────────────────────────────────────────
import os, sys, json, math, time, shutil, pickle, joblib, hashlib, itertools, warnings, random, glob
from time import perf_counter as clock
from pathlib import Path
from collections import defaultdict
from statistics import mean
from types import SimpleNamespace

# ─── CPU/thread control for NumPy/SciPy/PyTorch/FEniCS/Jupyter ─────────────────
for _var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_var] = "1"

# ─── Numerical, sparse algebra & optimization ─────────────────────────────────
import numpy as np
import numpy as _np
import pandas as pd
from numpy.linalg import solve, norm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, eigsh
from scipy.optimize import least_squares, minimize, Bounds
from scipy.stats import qmc
from scipy.interpolate import RBFInterpolator, Rbf, LinearNDInterpolator, NearestNDInterpolator

# ─── FEniCS / mesh generation ─────────────────────────────────────────────────
from dolfin import *
from mshr import *
from ufl import tanh

NEWTON_SOLVER_PARAMETERS = {
    "newton_solver": dict(
        linear_solver="petsc",
        absolute_tolerance=1E-5,
        relative_tolerance=1E-5,
        maximum_iterations=50,
        relaxation_parameter=1.0,
    )
}

# ─── RBNiCS reduced-order backends ─────────────────────────────────────────────
from rbnics import *
from rbnics.backends.dolfin import ProperOrthogonalDecomposition, BasisFunctionsMatrix

# ─── Plotting & tables ────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import (
    ScalarFormatter, FixedLocator, MaxNLocator, FormatStrFormatter, FuncFormatter,
    LogFormatterMathtext, LogLocator, NullFormatter,
)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tabulate import tabulate

try:
    import scienceplots
    plt.style.use(["science", "ieee", "notebook", "grid"])
except Exception:
    pass

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 14, "axes.titlesize": 15, "legend.fontsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "lines.linewidth": 2.0, "lines.markersize": 6,
    "figure.dpi": 150, "savefig.dpi": 300, "axes.grid": True, "grid.linestyle": ":",
    "grid.alpha": 0.7, "legend.frameon": True, "legend.loc": "best", "figure.autolayout": False,
})

# ─── Machine learning & data loaders ──────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF as GP_RBF,
    ConstantKernel as GP_Const,
    Matern as GP_Matern,
    WhiteKernel as GP_WhiteKernel,
)
from sklearn.exceptions import ConvergenceWarning

WhiteKernel = GP_WhiteKernel

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

print("Torch threads:", torch.get_num_threads())

# ─── Utility ──────────────────────────────────────────────────────────────────
try:
    from ipywidgets import interact, FloatSlider, VBox
except Exception:
    interact = FloatSlider = VBox = None

from tqdm import tqdm
from typing import Callable, Optional
from IPython.display import clear_output


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


# ─── FEniCS form compiler options ──────────────────────────────────────────────
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
parameters["ghost_mode"] = "shared_facet"

# %% [markdown] Cell 3 | id: c992b2f9-b063-4ea1-929b-9b5ff9fd988b
# # Full Order Model

# %% [markdown] Cell 4 | id: 899c0ac1-1d0a-4796-ab7a-ea8c753ea5ad
# ## Kirchhoff-Love Plate Solver

# %% Cell 5 | id: fd2078d0-a610-4a13-97e1-9af40f0cdd71
class GeneralMultiphysicsSolver:
    # ------------------------------------------------------------------
    # Initialization and default solver state
    # ------------------------------------------------------------------
    def __init__(self, study_case=1):
        """
        Initialize geometry, loading, mechanical state, thermal state,
        and thermo-ROM default parameters.
        """
        self.study_case = study_case
        """Set default geometry, mesh, penalty parameters, load configurations, boundary conditions, and initialize material parameters."""
        self.length, self.width = 2.0, 1.0; self.size, self.degree = 64, 2  # Geometry and mesh settings
        self.t = 0.05  # Plate thickness
        self.g = Constant(9.81) # gravitational acceleration
        self.rho = Constant(0.0) # Plate density
        self.foundation_tension_factor = 1e-3  # default unilateral
        self.reinforcement_model = 0  # 0-bare | 1-stiffeners | 2-ribs
        self.x0, self.y0, self.patch_size_x, self.patch_size_y = 1.0, 0.5, 0.351, 0.351  # Patch center & size
        self.patches = [  # Multi-patch: list of localized loads
            dict(x0=0.5, y0=0.25, patch_size_x=0.2, patch_size_y=0.2, load_value=-6000),
            dict(x0=0.5, y0=0.75, patch_size_x=0.2, patch_size_y=0.2, load_value=-6000),
            dict(x0=1.5, y0=0.25, patch_size_x=0.2, patch_size_y=0.2, load_value=-6000),
            dict(x0=1.5, y0=0.75, patch_size_x=0.2, patch_size_y=0.2, load_value=-6000),
        ]
        self.x_start, self.x_end = 0.0, self.length                                           # Linear load range
        self.load_value_start, self.load_value_end = 0.0, -6000                               # Linear load values
        self.bc_type = "simply_supported"  # Options: ["simply_supported", "free_edge"]
        self.load_type = "patch"  # Options: ["uniform", "patch", "multi_patch", "linear"]
        self.set_mu(); self.inner_product = None  # Set parameters and inner product

        # ── thermal (PDE-based only) ─────────────────────────────────────────────
        self._thermal_enabled = False
        self._alpha_iso = 1.2e-5
        self._alpha1_in = None
        self._alpha2_in = None
        self._T1_expr_in = None    

        self.alpha1_c = Constant(0.0)
        self.alpha2_c = Constant(0.0)
        self.T1_plate = Constant(0.0)

        # ---------------------------------------------------------------------
        # THERMO-ROM 
        # ---------------------------------------------------------------------
        self.rom_use_thermal = False
        
        # Thermal environment defaults
        self.T_amb_rom = 305.15      # [K]
        self.T_sub_rom = 298.15      # [K]
        self.h_con_rom = 10.0        # [W/m^2/K]
        self.eps_r_rom = 0.90        # [-]
        self.q_s_rom = 700.0         # [W/m^2]
        
        # Bottom thermal contact / gap defaults
        self.h_c_cont_rom = 200.0    # [W/m^2/K]
        self.h_c_gap_rom = 5.0       # [W/m^2/K]
        self.eta_c_rom = 50.0        # [-]
        self.w_contact_rom = -0.1    # [m] prescribed for one-way thermal solve
        
        # Thermal conductivity defaults
        self.kx_rom = 0.35           # [W/m/K]
        self.ky_rom = 0.35           # [W/m/K]
        self.kz_rom = 0.35           # [W/m/K]
        
        # Thermal expansion defaults
        self.alpha1_rom = 1.3e-4     # [1/K]
        self.alpha2_rom = 1.3e-4     # [1/K]

    # ------------------------------------------------------------------
    # Parametric mechanical model definition
    # ------------------------------------------------------------------
    def set_mu(self, mu=None):
        """
        Wrapper method that calls the main parameter setting function
        using the currently configured study case. This method should not be
        changed, as it ensures compatibility with the rest of the script.
        """
        if not hasattr(self, 'study_case'):
            self.study_case = 4
        self.set_parameters_for_case(case=self.study_case, mu=mu)
    
    def set_parameters_for_case(self, case, mu=None):
        """
        Sets the parameter vector 'mu' and updates all rigidity, thickness, and load
        properties based on the selected study case. This version is parameterized
        directly by plate rigidities (Dx, Dy, Dxy, Ds), not material properties.
        """
        self.study_case = case
        
        # --- REVISED Case Definitions: Focused on Rigidity, Thickness, and Geometry ---
        case_definitions = {
            # Core cases for orthotropy and interaction
            1: {"names": ["Dx", "Dy", "Dxy", "Ds", "ks", "f"], 
                "defaults": [9.5e3, 9.5e3, 2.85e3, 3.32e3, 1.0e8, -7000], 
                "description": "Direct Parametrization of All Rigidities & Interaction"},
            2: {"names": ["D_ratio", "ks", "f"], 
                "defaults": [1.0, 1.0e7, -8000], 
                "description": "Parametrize Rigidity Anisotropy & Interaction"},
            3: {"names": ["ks", "f"], 
                "defaults": [1.0e8, -7000], 
                "description": "Fundamental Soil-Structure Interaction"},
            4: {"names": ["D_eff", "R_B (Dy/Dx)", "R_H"],
                "defaults": [5e4, 0.5, 1.0],
                "description": "Parametrize Effective Stiffness and Orthotropic Ratios"},
            5: {"names": ["R_B (Dy/Dx)", "R_H"],
                "defaults": [1.0, 1.0],
                "description": "Parametrize Orthotropic Character for a Fixed Stiffness"},
            # Cases for reinforcement models
            6: {"names": ["H1", "t1"], "defaults": [0.05, 0.02], 
                "description": "Parametrize Stiffener Geometry (Model 1)"},
            7: {"names": ["I_rib", "b"], "defaults": [2.0e-7, 0.02], 
                "description": "Parametrize Rib Geometry (Model 2)"}
        }
        
        if case not in case_definitions:
            raise ValueError(f"Invalid case '{case}'. Must be one of {list(case_definitions.keys())}")
        
        # --- Parameter Vector Setup ---
        default_mu = case_definitions[case]["defaults"]
        self.mu = list(mu) if mu is not None else default_mu
        if len(self.mu) != len(default_mu):
            raise ValueError(f"For case {case}, mu must have {len(default_mu)} elements: {case_definitions[case]['names']}")
        mu_dict = dict(zip(case_definitions[case]["names"], self.mu))
        
        # --- Initialize Default State (using rigidities) ---
        t_val, ks_val, f_val = 0.05, 1.0e7, -10000
        Dx_base_val, Dy_base_val = 1.5e4, 1.0e4
        Dxy_base_val, Ds_base_val = 3.0e3, 4.0e3

        self.reinforcement_model = 0

        # Reinforcement defaults
        E_prime_val, b1_val, b2_val = 2.1e9, 0.1, 0.1
        H1_val, H2_val, t1_val, t2_val = 0.1, 0.1, 0.01, 0.01
        I_rib_val, b_val, C_val = 1e-6, 0.1, 1e4
        
        # --- Update Parameters Based on the Chosen Study Case ---
        if self.study_case == 1:
            Dx_base_val, Dy_base_val = mu_dict["Dx"], mu_dict["Dy"]
            Dxy_base_val, Ds_base_val = mu_dict["Dxy"], mu_dict["Ds"]
            ks_val, f_val = mu_dict["ks"], mu_dict["f"]
        elif self.study_case == 2:
            D_ratio = mu_dict["D_ratio"]
            D_mean = np.sqrt(Dx_base_val * Dy_base_val)
            Dx_base_val, Dy_base_val = D_mean * np.sqrt(D_ratio), D_mean / np.sqrt(D_ratio)
            ks_val, f_val = mu_dict["ks"], mu_dict["f"]
        elif self.study_case == 3:
            ks_val, f_val = mu_dict["ks"], mu_dict["f"]
        elif self.study_case == 4:
            Dx_base_val = mu_dict["D_eff"]
            Dy_base_val = mu_dict["R_B (Dy/Dx)"] * Dx_base_val
            Dxy_base_val = 0.0
            Ds_base_val = 0.5 * mu_dict["R_H"] * np.sqrt(Dx_base_val * Dy_base_val)
        elif self.study_case == 5:
            D_mean = np.sqrt(Dx_base_val * Dy_base_val)
            R_B = mu_dict["R_B (Dy/Dx)"]
            Dx_base_val = D_mean / np.sqrt(R_B)
            Dy_base_val = D_mean * np.sqrt(R_B)
            Dxy_base_val = 0.0
            Ds_base_val = 0.5 * mu_dict["R_H"] * D_mean
        elif self.study_case == 6:
            self.reinforcement_model = 1
            H1_val, t1_val = mu_dict["H1"], mu_dict["t1"]
            H2_val, t2_val = H1_val, t1_val
        elif self.study_case == 7:
            self.reinforcement_model = 2
            I_rib_val, b_val = mu_dict["I_rib"], mu_dict["b"]

        # --- Set Final Solver Parameters as FEniCS Constants ---
        self.t, self.ks, self.f = Constant(t_val), Constant(ks_val), Constant(f_val)
        self.D_x_base, self.D_y_base = Constant(Dx_base_val), Constant(Dy_base_val)
        self.D_xy_base, self.D_s_base = Constant(Dxy_base_val), Constant(Ds_base_val)
        
        # --- Update Reinforcement Geometric Attributes (if applicable) ---
        self.E_prime, self.b1, self.b2 = Constant(E_prime_val), Constant(b1_val), Constant(b2_val)
        self.H1, self.H2, self.t1, self.t2 = Constant(H1_val), Constant(H2_val), Constant(t1_val), Constant(t2_val)
        self.I_rib, self.b, self.C = Constant(I_rib_val), Constant(b_val), Constant(C_val)
        
        self.update_reinforcement_rigidities()
        
    def update_reinforcement_rigidities(self):
        """Update effective rigidities for the selected reinforcement model."""
        if self.reinforcement_model == 0:
            self.D_x, self.D_y, self.D_xy, self.D_s = self.D_x_base, self.D_y_base, self.D_xy_base, self.D_s_base
        elif self.reinforcement_model == 1:
            self.D_x  = self.D_x_base + self.E_prime * self.b1 * (self.H1**3 - self.t**3) / (12 * self.t1)
            self.D_y  = self.D_y_base + self.E_prime * self.b2 * (self.H2**3 - self.t**3) / (12 * self.t2)
            self.D_xy = self.D_xy_base
            self.D_s  = self.D_s_base
        elif self.reinforcement_model == 2:
            self.D_x  = self.E_prime * self.I_rib / self.t
            self.D_y  = (self.E_x * self.t**3) / (12 * (1 - self.b / self.t + (self.b * self.t**3) / (self.H1**3 * self.t)))
            self.D_xy = Constant(0.0)
            self.D_s  = Constant(self.C * self.t**3 / 12 + self.C / (2 * self.t))
                
    # ------------------------------------------------------------------
    # Geometric domain and panel partition
    # ------------------------------------------------------------------
    def define_domain(self, n_vert, n_horiz, plot_subdomains=False):
        """Build the 2D plate mesh, subdomains, boundaries, and interface data."""
        # Define problem domain
        self.n_vert, self.n_horiz = n_vert, n_horiz
        self.N_subdomains = (self.n_vert + 1) * (self.n_horiz + 1) # Calculate the total number of subdomains
        self.boundary_ids = [1, 2, 3, 4] # External Boundary ids
        self.output_dir = (os.makedirs(dir := f"output/nv{self.n_vert}_nh{self.n_horiz}_{self.bc_type}_{self.load_type}", exist_ok=True) or dir)

        # Build initial domain
        domain = Rectangle(Point(0.0, 0.0), Point(self.length, self.width))
        vertical_divisions = [i * (self.length / (n_vert + 1)) for i in range(n_vert + 2)]
        horizontal_divisions = [j * (self.width / (n_horiz + 1)) for j in range(n_horiz + 2)]

        # Set subdomain IDs for each cell
        subdomain_id = 1
        for i in range(len(vertical_divisions) - 1):
            for j in range(len(horizontal_divisions) - 1):
                domain.set_subdomain(
                    subdomain_id,
                    Rectangle(Point(vertical_divisions[i], horizontal_divisions[j]),
                              Point(vertical_divisions[i + 1], horizontal_divisions[j + 1]))
                )
                subdomain_id += 1
        # Generate mesh
        initial_mesh = generate_mesh(domain, self.size)

        # Create subdomains and boundary markers
        subdomains = MeshFunction("size_t", initial_mesh, initial_mesh.topology().dim(), initial_mesh.domains())
        boundaries = MeshFunction("size_t", initial_mesh, initial_mesh.topology().dim() - 1, 0)

        def mark_boundaries(boundaries, conditions):
            # Mark boundaries based on provided conditions
            class Boundary(SubDomain):
                def __init__(self, parent, condition):
                    super().__init__()
                    self.length, self.width, self.condition = parent.length, parent.width, condition
                def inside(self, x, on_boundary):
                    return self.condition(x)
            for marker, condition in conditions:
                Boundary(self, condition).mark(boundaries, marker)

        # Define boundary conditions dynamically
        boundary_conditions = [
            (1, lambda x: near(x[0], 0.0)),  # Left boundary
            (2, lambda x: near(x[1], self.width)),  # Top boundary
            (3, lambda x: near(x[0], self.length)),  # Right boundary
            (4, lambda x: near(x[1], 0.0)),  # Bottom boundary
        ]
        # Add vertical interface boundaries
        for idx, x_coord in enumerate(vertical_divisions[1:-1], start=5):
            boundary_conditions.append((idx, lambda x, xc=x_coord: near(x[0], xc)))
        # Add horizontal interface boundaries
        for idx, y_coord in enumerate(horizontal_divisions[1:-1], start=5 + n_vert):
            boundary_conditions.append((idx, lambda x, yc=y_coord: near(x[1], yc)))
        # Mark boundaries
        mark_boundaries(boundaries, boundary_conditions)

        self.mesh, self.subdomains, self.boundaries = initial_mesh, subdomains, boundaries

        # Plot subdomains
        if plot_subdomains:
            plt.figure(figsize=(6, 4))
            plot_subdomains = plot(self.subdomains, title="Subdomains", cmap=ListedColormap(plt.cm.viridis(np.linspace(0, 1, subdomain_id - 1))))
            plt.colorbar(plot_subdomains, ticks=range(1, subdomain_id), shrink=0.5, label="Subdomain ID")
            plt.xlabel("x [m]"), plt.ylabel("y [m]"), plt.tight_layout(), plt.show()

        # Identify interface pairs and boundary markers
        self.interface_pairs = sorted({
            (min(self.subdomains[c1], self.subdomains[c2]),
             max(self.subdomains[c1], self.subdomains[c2]),
             self.boundaries[facet.index()])
            for facet in facets(self.mesh)
            if self.N_subdomains > 1 and not facet.exterior() and len(facet.entities(2)) == 2
            and (c1 := facet.entities(2)[0]) is not None
            and (c2 := facet.entities(2)[1]) is not None
            and self.subdomains[c1] != self.subdomains[c2]
        }) if self.N_subdomains > 1 else []
        self.subdomain_ids = np.unique(self.subdomains.array()) if self.N_subdomains > 1 else [1]        
        print("\033[1mIdentified interface pairs → [(Ωᵢ, Ωⱼ), Γ꜀]: \033[0m" + ", ".join(f"[({a}, {b}), {m}]" for a, b, m in sorted(self.interface_pairs)))
        return self.mesh, self.subdomains, self.boundaries, self.interface_pairs

    # ------------------------------------------------------------------
    # Thermal state management and 3D heat-to-plate reduction
    # ------------------------------------------------------------------
    def update_thermal_parameters(
        self,
        enable: bool = False,
        alpha_iso: float = 1.2e-5,
        alpha1: float | None = None,
        alpha2: float | None = None,
        T1_expr=None                    
    ):
        """
        Turn thermal bending on/off and set the through-thickness linear coefficient.
        """
        self._thermal_enabled = bool(enable)
        self._alpha_iso = float(alpha_iso)
        self._alpha1_in = alpha1
        self._alpha2_in = alpha2
        self._T1_expr_in = T1_expr

    def _realize_thermal_fields(self):
        """Convert stored thermal inputs to UFL/FEniCS fields."""
        # alphas
        a1 = self._alpha1_in if self._alpha1_in is not None else self._alpha_iso
        a2 = self._alpha2_in if self._alpha2_in is not None else self._alpha_iso
        self.alpha1_c = Constant(float(a1))
        self.alpha2_c = Constant(float(a2))

        # T^1(x,y): priority → UFL given → else zero
        def _is_ufl(v): return hasattr(v, "_cpp_object")
        if self._T1_expr_in is not None and _is_ufl(self._T1_expr_in):
            self.T1_plate = self._T1_expr_in
        else:
            self.T1_plate = Constant(0.0)
    
    def heat_solve(
        self, *,
        # --- conductivity (diag) ---
        kx: float = 1.0, ky: float | None = None, kz: float | None = None,
    
        # --- documented environmental/top BC data ---
        T_amb=None, h_con=None, eps_r: float = 1.0, q_s=None,
    
        # --- documented bottom/substrate data ---
        T_sub=None, h_c_cont=None, h_c_gap: float = 0.0, eta_c: float = 50.0,
    
        # --- deflection field driving hc(w) ---
        w=None,
    
        # --- volumetric source ---
        source=None,
    
        # --- reference temperature for ΔT = T - Tref ---
        T_ref: float | None = None,
    
        # --- mesh/discretization ---
        nx: int | None = None, ny: int | None = None, nz: int = 10, degree: int = 1,
    
        # --- storage ---
        store: bool = True,
    
        # --- legacy args (explicitly rejected to enforce the approved model) ---
        T_top=None, T_bottom=None, q_top=None, q_bottom=None,
        robin_h_top=None, robin_Tinf_top=None, robin_h_bottom=None, robin_Tinf_bottom=None,
    ):
        r"""
        3D steady heat-conduction solve on Ω^3 = Ω × (-h/2, h/2):
    
          -∇·(K∇T) = s  in Ω^3
    
        Γ_top (z=+h/2), q_n = -n·K∇T:
          q_n = h_con (T - T_amb) + eps_r σ (T^4 - T_amb^4) - q_s
    
        Γ_bot (z=-h/2):
          q_n = h_c(w) (T - T_sub),
          h_c(w)=h_c_cont+(h_c_gap-h_c_cont)S(w),  S(w)=0.5(1+tanh(eta_c w))
    
        Γ_lat: adiabatic q_n=0
    
        Returns: dict(mesh, Vt, facets, T, DeltaT, T_ref)
        NOTE: Radiation (T^4) => nonlinear Newton solve.
        """
        # ---- strict model enforcement ----
        if any(v is not None for v in (
            T_top, T_bottom, q_top, q_bottom,
            robin_h_top, robin_Tinf_top, robin_h_bottom, robin_Tinf_bottom
        )):
            raise ValueError(
                "heat_solve: legacy BC arguments (T_top/T_bottom/q_top/q_bottom/robin_*) are not allowed.\n"
                "This solver strictly implements:\n"
                "  top: convection + radiation + absorbed solar flux,\n"
                "  bottom: h_c(w)(T - T_sub),\n"
                "  lateral: adiabatic."
            )
    
        if T_amb is None or h_con is None or q_s is None:
            raise ValueError("heat_solve: top boundary requires T_amb, h_con, q_s.")
        if T_sub is None or h_c_cont is None:
            raise ValueError("heat_solve: bottom boundary requires T_sub, h_c_cont.")
        if not (0.0 < float(eps_r) <= 1.0):
            raise ValueError("heat_solve: eps_r must be in (0, 1].")
        if float(h_c_gap) < 0.0:
            raise ValueError("heat_solve: h_c_gap must be >= 0.")
        eta_val = float(eta_c)
        if eta_val <= 0.0:
            raise ValueError("heat_solve: eta_c must be > 0.")
    
        # ---- helpers ----
        def _is_scalar_like(a):
            return isinstance(a, (int, float, np.floating))
    
        def _is_cpp_obj(a):
            return hasattr(a, "_cpp_object")
    
        def _as_ufl(a):
            return a if _is_cpp_obj(a) else Constant(float(a))
    
        def _function_geom_dim(obj):
            try:
                return obj.function_space().mesh().geometry().dim()
            except Exception:
                return None

        def _plate_space():
            """
            Return a valid 2D plate CG space for lifting w(x,y) into the 3D heat solve.

            Important:
            - Do not blindly reuse self.V_CG, because it may belong to a previous
              mechanical polynomial degree in a convergence loop.
            - The target degree follows the current plate/mechanical degree,
              not the 3D heat polynomial degree.
            """
            if not hasattr(self, "mesh"):
                raise RuntimeError(
                    "heat_solve: a 2D plate field was passed for w, but the 2D plate mesh does not exist yet. "
                    "Call define_domain() first."
                )

            deg = max(1, int(getattr(self, "degree", degree)))

            if hasattr(self, "V_CG"):
                try:
                    el = self.V_CG.ufl_element()
                    same_mesh = self.V_CG.mesh().id() == self.mesh.id()
                    same_family = el.family() in ("Lagrange", "CG")
                    same_degree = int(el.degree()) == deg

                    if same_mesh and same_family and same_degree:
                        return self.V_CG

                except Exception:
                    pass

            return FunctionSpace(self.mesh, "CG", deg)

        def _to_plate_field(obj):
            V2 = _plate_space()

            if obj is None:
                return interpolate(Constant(0.0), V2)

            if _is_scalar_like(obj):
                return interpolate(Constant(float(obj)), V2)

            if isinstance(obj, Function):
                try:
                    same_mesh = obj.function_space().mesh().id() == V2.mesh().id()
                    same_dim = obj.function_space().dim() == V2.dim()

                    if same_mesh and same_dim:
                        out = Function(V2)
                        out.assign(obj)
                        return out

                except Exception:
                    pass

            try:
                return project(obj, V2)
            except Exception:
                return interpolate(obj, V2)
    
        def _lift_2d_to_3d(w2d, expr_degree):
            class _LiftW(UserExpression):
                def __init__(self, w2d_, **kw):
                    super().__init__(**kw)
                    self.w2d = w2d_
    
                def eval(self, values, x):
                    values[0] = float(self.w2d(Point(x[0], x[1])))
    
                def value_shape(self):
                    return ()
            return _LiftW(w2d, degree=int(expr_degree))
    
        # ---- mesh Ω × (-h/2, h/2) ----
        h = float(self.t)
        if nx is None:
            nx = max(2, int(self.size / 6))
        if ny is None:
            ny = max(2, int(self.size / 6))
        Lx, Ly = float(self.length), float(self.width)
        mesh = BoxMesh(Point(0.0, 0.0, -0.5 * h), Point(Lx, Ly, 0.5 * h), nx, ny, nz)
    
        facets = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    
        class _Top(SubDomain):
            def inside(self, x, on_b):
                return on_b and near(x[2], 0.5 * h)
    
        class _Bottom(SubDomain):
            def inside(self, x, on_b):
                return on_b and near(x[2], -0.5 * h)
    
        class _Lateral(SubDomain):
            def inside(self, x, on_b):
                return on_b and (
                    near(x[0], 0.0) or near(x[0], Lx) or near(x[1], 0.0) or near(x[1], Ly)
                )
    
        _Top().mark(facets, 1)
        _Bottom().mark(facets, 2)
        _Lateral().mark(facets, 3)
    
        dx3 = Measure("dx", domain=mesh)
        ds3 = Measure("ds", domain=mesh, subdomain_data=facets)
    
        # ---- FE space ----
        Vt = FunctionSpace(mesh, "CG", degree)
        T = Function(Vt, name="Temperature")
        v = TestFunction(Vt)
    
        # ---- K = diag(kx,ky,kz) ----
        kx_u = Constant(float(kx))
        ky_u = Constant(float(ky if ky is not None else kx))
        kz_u = Constant(float(kz if kz is not None else kx))
        K = as_tensor(((kx_u, 0.0, 0.0), (0.0, ky_u, 0.0), (0.0, 0.0, kz_u)))
    
        # ---- data ----
        sigma_SB = Constant(5.67e-8)
        eps_u = Constant(float(eps_r))
        T_amb_u = _as_ufl(T_amb)
        h_con_u = _as_ufl(h_con)
        q_s_u = _as_ufl(q_s)
        T_sub_u = _as_ufl(T_sub)
        hc_cont_u = _as_ufl(h_c_cont)
        hc_gap_u = _as_ufl(h_c_gap)
        eta_u = Constant(eta_val)
        s_u = Constant(0.0) if source is None else _as_ufl(source)
    
        # ---- w lifting + h_c(w) ----
        # Cases:
        #   - scalar / Constant: use directly
        #   - 3D Function: use directly
        #   - 2D plate field / expression: lift to (x,y,z) -> w(x,y)
        if w is None:
            # Default to near-contact so S(w)≈0 and hc≈hc_cont.
            w_u = Constant(-10.0 / eta_val)
        elif _is_scalar_like(w):
            w_u = Constant(float(w))
        elif isinstance(w, Constant):
            w_u = w
        elif isinstance(w, Function):
            gdim = _function_geom_dim(w)
            if gdim == 3:
                w_u = w
            elif gdim == 2:
                w2d = _to_plate_field(w)
                w_u = _lift_2d_to_3d(w2d, expr_degree=max(2, int(degree) + 1))
            else:
                w2d = _to_plate_field(w)
                w_u = _lift_2d_to_3d(w2d, expr_degree=max(2, int(degree) + 1))
        else:
            # Generic 2D UFL / Expression / UserExpression path.
            w2d = _to_plate_field(w)
            w_u = _lift_2d_to_3d(w2d, expr_degree=max(2, int(degree) + 1))
    
        S_w = 0.5 * (1.0 + tanh(eta_u * w_u))
        hc_w = hc_cont_u + (hc_gap_u - hc_cont_u) * S_w
    
        # ---- weak form ----
        F = (
            inner(K * grad(T), grad(v)) * dx3
            + h_con_u * T * v * ds3(1)
            + eps_u * sigma_SB * (T**4) * v * ds3(1)
            + hc_w * T * v * ds3(2)
            - s_u * v * dx3
            - h_con_u * T_amb_u * v * ds3(1)
            - eps_u * sigma_SB * (T_amb_u**4) * v * ds3(1)
            - q_s_u * v * ds3(1)
            - hc_w * T_sub_u * v * ds3(2)
        )
    
        # ---- nonlinear solve ----
        problem = NonlinearVariationalProblem(F, T, bcs=[], J=derivative(F, T))
        nls = NonlinearVariationalSolver(problem)
        nls.parameters.update(NEWTON_SOLVER_PARAMETERS)
        # prm = nls.parameters["newton_solver"]
        # prm.update(dict(
        #     linear_solver="petsc",
        #     absolute_tolerance=1e-8,
        #     relative_tolerance=1e-8,
        #     maximum_iterations=50,
        #     relaxation_parameter=1.0
        # ))
        nls.solve()
    
        # ---- reference temperature + ΔT ----
        if T_ref is None:
            meas = assemble(Constant(1.0) * dx3)
            T_ref_eff = assemble(T * dx3) / meas
        else:
            T_ref_eff = float(T_ref)
    
        # DeltaT = project(T - Constant(float(T_ref_eff)), Vt)
        # DeltaT.rename("DeltaT", "")
        # ---- temperature increment ΔT = T - T_ref ----
        Tref_fn = interpolate(Constant(float(T_ref_eff)), Vt)
        
        DeltaT = Function(Vt)
        DeltaT.vector().zero()
        DeltaT.vector().axpy(1.0, T.vector())
        DeltaT.vector().axpy(-1.0, Tref_fn.vector())
        DeltaT.vector().apply("insert")
        DeltaT.rename("DeltaT", "")
    
        if store:
            self.mesh3d_heat = mesh
            self.facets_heat = facets
            self.Vt_heat = Vt
            self.T_3d_heat = T
            self.DeltaT_3d_heat = DeltaT
            self.T_ref_heat = float(T_ref_eff)
    
        return dict(mesh=mesh, Vt=Vt, facets=facets, T=T, DeltaT=DeltaT, T_ref=float(T_ref_eff))
    
    def make_T1_from_DeltaT(self, DeltaT3d, *, Nz_quad: int = 8, cg_degree: int = 1):
        r"""
        Project the 3D ΔT(x,y,z) to the plate thermal driver
    
            θ(x,y) = (12/h^3) ∫_{-h/2}^{h/2} z * ΔT(x,y,z) dz,
    
        i.e. the coefficient of the linear-in-z component of ΔT, consistent with the
        centered 3D heat mesh z ∈ [-h/2, +h/2].
        """
        t = float(self.t)
    
        # Robustness: avoid rare "point not inside domain" from tiny roundoff at ±t/2
        # (the correct fix is the centered z-mapping below).
        try:
            DeltaT3d.set_allow_extrapolation(True)
        except Exception:
            pass
    
        class _ThetaFromDT(UserExpression):
            def __init__(self, DT, t, Nz, **kw):
                super().__init__(**kw)
                self.DT = DT
                self.t = float(t)
                self.xi, self.w = _np.polynomial.legendre.leggauss(int(Nz))  # on [-1,1]
    
            def eval(self, values, x):
                t = self.t
                I = 0.0
                # Map ξ ∈ [-1,1] -> z ∈ [-t/2, t/2] via z = (t/2) ξ, dz = (t/2) dξ
                for (xi, w) in zip(self.xi, self.w):
                    z = 0.5 * t * float(xi)  # centered thickness coordinate
                    DTval = float(self.DT(Point(x[0], x[1], z)))
                    I += w * (z * DTval)
                I *= 0.5 * t  # dz scaling
                values[0] = (12.0 / (t**3)) * I
    
            def value_shape(self):
                return ()
    
        V2D = FunctionSpace(self.mesh, "CG", int(cg_degree))
        theta_expr = _ThetaFromDT(
            DeltaT3d,
            t,
            Nz=Nz_quad,
            degree=max(2, int(cg_degree) + 2),
        )
        theta_fn = interpolate(theta_expr, V2D)
    
        # Keep your legacy naming so downstream code remains drop-in.
        theta_fn.rename("T1_plate", "")
        self.T1_from_heat = theta_fn
        return theta_fn
    
    def enable_plate_thermal_from_heat(
        self,
        *,
        alpha1: float | None = None,
        alpha2: float | None = None,
        Nz_quad: int = 8, cg_degree: int = 1,
        **heat_kwargs
    ):
        """
        One-call: solve heat → build T1(x,y) → enable thermal in Kirchhoff FEM.
        """
        heat = self.heat_solve(**heat_kwargs)
        T1 = self.make_T1_from_DeltaT(heat["DeltaT"], Nz_quad=Nz_quad, cg_degree=cg_degree)
        self.update_thermal_parameters(
            enable=True,
            alpha1=(self._alpha_iso if alpha1 is None else float(alpha1)),
            alpha2=(self._alpha_iso if alpha2 is None else float(alpha2)),
            T1_expr=T1
        )
        return T1
    
    def navier_T1_from_heat_callable(
        self,
        *,
        Nz_quad: int = 8,
        cg_degree: int = 1,
        T1_cg_degree: int | None = None,
    
        # convergence-driver aliases
        heat_nx=None,
        heat_ny=None,
        heat_nz=None,
        heat_degree=None,
    
        **heat_kwargs
    ):
        """
        Build a callable T1(x,y) for the Exact/Navier solver from the PDE heat solve.
    
        This helper is intended for the exact/Navier reference branch of the
        thermo-mechanical convergence study.
    
        It accepts both native heat_solve names:
    
            nx, ny, nz, degree
    
        and convergence-driver aliases:
    
            heat_nx, heat_ny, heat_nz, heat_degree
    
        It also safely ignores thermo-mechanical parameters that do not belong
        to heat_solve, such as alpha1, alpha2, rho, and T1_cg_degree.
        """
        hk = dict(heat_kwargs)
    
        # Allow convergence driver alias for through-thickness quadrature.
        if "Nz_quad_T1" in hk:
            Nz_quad = int(hk.pop("Nz_quad_T1"))
    
        # Allow either cg_degree or T1_cg_degree.
        if T1_cg_degree is not None:
            cg_degree = int(T1_cg_degree)
    
        if "T1_cg_degree" in hk:
            cg_degree = int(hk.pop("T1_cg_degree"))
    
        # Remove thermo-mechanical parameters that heat_solve does not accept.
        hk.pop("alpha1", None)
        hk.pop("alpha2", None)
        hk.pop("rho", None)
    
        # Map convergence-driver aliases to heat_solve names.
        if heat_nx is not None:
            hk["nx"] = int(heat_nx)
        if heat_ny is not None:
            hk["ny"] = int(heat_ny)
        if heat_nz is not None:
            hk["nz"] = int(heat_nz)
        if heat_degree is not None:
            hk["degree"] = int(heat_degree)
    
        # Use ROM thermal state as defaults when the caller does not provide them.
        hk.setdefault("kx", self.kx_rom)
        hk.setdefault("ky", self.ky_rom)
        hk.setdefault("kz", self.kz_rom)
    
        hk.setdefault("T_amb", self.T_amb_rom)
        hk.setdefault("h_con", self.h_con_rom)
        hk.setdefault("eps_r", self.eps_r_rom)
        hk.setdefault("q_s", self.q_s_rom)
    
        hk.setdefault("T_sub", self.T_sub_rom)
        hk.setdefault("h_c_cont", self.h_c_cont_rom)
        hk.setdefault("h_c_gap", self.h_c_gap_rom)
        hk.setdefault("eta_c", self.eta_c_rom)
    
        # If no displacement/contact field is supplied, use the one-way ROM contact state.
        hk.setdefault("w", self.w_contact_rom)
    
        # Preserve direct native names if supplied; otherwise use heat_solve defaults.
        if "nx" in hk and hk["nx"] is not None:
            hk["nx"] = int(hk["nx"])
        if "ny" in hk and hk["ny"] is not None:
            hk["ny"] = int(hk["ny"])
        if "nz" in hk and hk["nz"] is not None:
            hk["nz"] = int(hk["nz"])
        if "degree" in hk and hk["degree"] is not None:
            hk["degree"] = int(hk["degree"])
    
        heat = self.heat_solve(**hk)
    
        T1 = self.make_T1_from_DeltaT(
            heat["DeltaT"],
            Nz_quad=int(Nz_quad),
            cg_degree=int(cg_degree),
        )
    
        try:
            T1.set_allow_extrapolation(True)
        except Exception:
            pass
    
        def _T1xy(x, y):
            return float(T1(Point(float(x), float(y))))
    
        return _T1xy

    # ------------------------------------------------------------------
    # Full-order Kirchhoff plate model
    # ------------------------------------------------------------------
    def setup_kirchhoff_problem(self):
        """Assemble the Kirchhoff plate weak form, including thermal terms if enabled."""
        # Solve the Kirchhoff plate bending problem
        print("=" * 65)
        self.V = FiniteElement("Lagrange", triangle, self.degree)  # Define finite element
        self.V_DG = FunctionSpace(self.mesh, "DG", 0); self.V_CG = FunctionSpace(self.mesh, self.V)
        if self.N_subdomains > 1:
            self.W = FunctionSpace(self.mesh, MixedElement([self.V] * self.N_subdomains))  # Mixed function space
            w_ = Function(self.W)
            w_sub = split(w_)  # Separate solutions for each subdomain
        else:
            self.W = FunctionSpace(self.mesh, self.V)  # Single function space
            w_ = Function(self.W)
            w_sub = [w_]  # Treat it as a single-element list for consistency

        w, w_t = TrialFunction(self.W), TestFunction(self.W)  # Trial and test functions

        # Define measures for subdomains and boundaries
        dx = Measure("dx", domain=self.mesh, subdomain_data=self.subdomains)
        ds = Measure("ds", domain=self.mesh, subdomain_data=self.boundaries)
        dS = Measure("dS", domain=self.mesh, subdomain_data=self.boundaries)
        dS_CDG = eval(" + ".join([f"dS({i})" for i in [0, 1, 2, 3, 4]]))

        # Define inner product
        self.inner_product = assemble(inner(w, w_t) * dx + inner(grad(w), grad(w_t)) * dx)
        u_cg, v_cg = TrialFunction(self.V_CG), TestFunction(self.V_CG)
        dx_cg = Measure("dx", domain=self.mesh)
        self.inner_product_CG = assemble(inner(u_cg, v_cg) * dx_cg + inner(grad(u_cg), grad(v_cg)) * dx_cg)

        # Initialize energy components
        A, L_CDG, psi_found, I_penalty = 0, 0, 0, 0
        self.h_avg = (CellDiameter(self.mesh)('+') + CellDiameter(self.mesh)('-')) / 2.0  # Average cell size
        n = FacetNormal(self.mesh)  # Facet normal vector
        D_eff = sqrt(self.D_x * self.D_y)
        self.alphaCDG = (self.degree + 2)**2 * D_eff
        self.stabCDG = self.alphaCDG('+') / self.h_avg

        # ── physics-based interface penalties ────────────────
        self.C_alpha = self.degree; self.C_beta = 12**(self.degree + 1) / self.degree
        self.alpha = self.C_alpha * D_eff / self.h_avg**3  # [N/m²] for displacement penalty
        self.beta = self.C_beta * D_eff / self.h_avg       # [N] for rotation penalty

        thermal_on = bool(self._thermal_enabled)
        if thermal_on:
            self._realize_thermal_fields()
            # Precompute the thermal bending moments (independent of w)
            MxT = (self.D_x * self.alpha1_c + self.D_xy * self.alpha2_c) * self.T1_plate
            MyT = (self.D_xy * self.alpha1_c + self.D_y  * self.alpha2_c) * self.T1_plate
            MxyT = Constant(0.0)

        # Iterate over subdomains to compute strain energy and stabilization terms
        for i in range(1, self.N_subdomains + 1) if self.N_subdomains > 1 else [1]:
            theta_i, k_i = grad(w_sub[i - 1]), variable(sym(grad(grad(w_sub[i - 1]))))
            k_x, k_y, k_xy = k_i[0, 0], k_i[1, 1], k_i[0, 1]
            term1 = 0.5 * self.D_x * k_x**2
            term2 = 0.5 * 2 * self.D_xy * (k_x * k_y)
            term3 = 0.5 * self.D_y * k_y**2
            term4 = 0.5 * 4 * self.D_s * k_xy**2
            psi_M_i = term1 + term2 + term3 + term4
            A += psi_M_i * dx(i) if self.N_subdomains > 1 else psi_M_i * dx  

            if thermal_on:
                A += (MxT * k_x + MyT * k_y + 2.0 * MxyT * k_xy) * (dx(i) if self.N_subdomains > 1 else dx)

            M_i = diff(psi_M_i, k_i)  # Moment tensor
            M_n_i = inner(M_i, outer(n, n))  # Normal component

            # Stabilization term (Only for multiple subdomains)
            L_CDG += (-inner(jump(theta_i, n), avg(M_n_i)) +
                          (0.5 * self.stabCDG) * inner(jump(theta_i, n), jump(theta_i, n))) * dS_CDG

            # Foundation term
            psi_found += 0.5 * self.ks * conditional(w_sub[i - 1] < 0, w_sub[i - 1]**2, self.foundation_tension_factor * w_sub[i - 1]**2) * dx(i) if self.N_subdomains > 1 else \
                         0.5 * self.ks * conditional(w_ < 0, w_**2, self.foundation_tension_factor * w_**2) * dx

        # Define interface conditions (only when multiple subdomains exist)
        if self.N_subdomains > 1:
            for (i, j, boundary_marker) in self.interface_pairs:
                w_jump = w_sub[i - 1]('-') - w_sub[j - 1]('+')  # Displacement jump
                dw_jump = inner(grad(w_sub[i - 1])('-') - grad(w_sub[j - 1])('+'), n('-'))  # Slope jump
                
                I_disp = 0.5 * self.alpha * dot(w_jump, w_jump)  # Displacement penalty
                I_slope = 0.5 * self.beta * dot(dw_jump, dw_jump)  # Slope penalty
                I_penalty += (I_disp + I_slope) * dS(boundary_marker)

        # Define external load expression
        load_expr = (
            self.f if self.load_type == "uniform" else
            Expression(
                '((x[0] >= x0 - 0.5*patch_size_x) && (x[0] <= x0 + 0.5*patch_size_x) && '
                '(x[1] >= y0 - 0.5*patch_size_y) && (x[1] <= y0 + 0.5*patch_size_y)) ? load_value : 0.0',
                degree=2, x0=self.x0, y0=self.y0,
                patch_size_x=self.patch_size_x, patch_size_y=self.patch_size_y,
                load_value=self.f
            ) if self.load_type == "patch" else
            Expression(
                '(x[0] >= 0.0 && x[0] <= L) ? f0 + (f1 - f0) * (x[0] / L) : 0.0',
                degree=2, L=self.length, f0=float(self.load_value_start), f1=float(self.f)
            ) if self.load_type == "linear" else
            sum(
                Expression(
                    '((x[0] >= x0 - 0.5*patch_size_x) && (x[0] <= x0 + 0.5*patch_size_x) && '
                    '(x[1] >= y0 - 0.5*patch_size_y) && (x[1] <= y0 + 0.5*patch_size_y)) ? load_value : 0.0',
                    degree=2,
                    x0=Constant(p["x0"]), y0=Constant(p["y0"]),
                    patch_size_x=Constant(p["patch_size_x"]), patch_size_y=Constant(p["patch_size_y"]),
                    load_value=float(self.f)
                ) for p in self.patches
            ) if self.load_type == "multi_patch" else
            (_ for _ in ()).throw(ValueError(f"Unknown load_type: {self.load_type}"))
        )

        W_ext = sum(load_expr * w_sub[i - 1] * dx(i) for i in range(1, self.N_subdomains + 1)) \
            if self.N_subdomains > 1 else load_expr * w_ * dx

        # ---------------------------------------------------------
        # self-weight: q_self = -rho * g * t
        # ---------------------------------------------------------
        q_weight = -self.rho * self.g * self.t
        
        W_self = (
            sum(q_weight * w_sub[i - 1] * dx(i) for i in range(1, self.N_subdomains + 1))
            if self.N_subdomains > 1
            else q_weight * w_ * dx
        )
        
        W_ext = W_ext + W_self
        
        # Boundary conditions
        bcs = (
            [DirichletBC(self.W.sub(i), Constant(0.0), self.boundaries, bid)
             for i in range(self.N_subdomains) for bid in self.boundary_ids]
            if self.bc_type == "simply_supported" and self.N_subdomains > 1 else
            [DirichletBC(self.W, Constant(0.0), self.boundaries, bid)
             for bid in self.boundary_ids]
            if self.bc_type == "simply_supported" else
            [] if self.bc_type == "free_edge" else
            (_ for _ in ()).throw(ValueError(f"Unknown bc_type: {self.bc_type}"))
        )
        # Variational formulation
        self.F_plate = derivative(A + L_CDG - W_ext, w_, w_t)  # Plate energy
        self.F_foundation = derivative(psi_found, w_, w_t)  # Foundation energy
        self.F_contact = derivative(I_penalty, w_, w_t) if self.N_subdomains > 1 else 0  # Contact energy

        # Problem setup
        self.w_ = w_
        self.bcs = bcs
        self.F = self.F_plate + self.F_foundation + self.F_contact  # Total residual
        self.J = derivative(self.F, w_, w)  # Jacobian matrix

    def offline_solve_kirchhoff_problem(
        self,
        mu,
        return_mode="global",
        initial_guess=None,
        thermal_on=False,
    ):
        """
        Solve the nonlinear Kirchhoff problem for a given mechanical parameter vector.
    
        Parameters
        ----------
        mu : list[float]
            Mechanical parameter vector.
    
        return_mode : {"global", other}
            If "global", return the projected global CG displacement field.
            Otherwise, return the raw mixed/single solution object.
    
        initial_guess : scalar, Function, Expression, or None
            Optional warm start for the nonlinear mechanical solve.
    
        thermal_on : bool
            False:
                force a purely mechanical solve and clear any stale thermal state.
    
            True:
                keep/use the currently stored thermal driver T1 and thermal
                expansion coefficients. This is used by solve_mechanical_given_T1
                and thermo-mechanical workflows.
    
        Notes
        -----
        This explicit thermal_on switch is important for convergence studies.
        Without it, a direct mechanical solve can accidentally inherit a previous
        one-way/coupled thermal state.
        """
        self.set_mu(mu)
    
        if bool(thermal_on):
            # Keep the currently stored T1/alpha state, but ensure it is active.
            self._thermal_enabled = True
        else:
            # Strictly isolate the mechanical branch from any previous thermal solve.
            self.update_thermal_parameters(
                enable=False,
                alpha_iso=self._alpha_iso,
                alpha1=self.alpha1_rom,
                alpha2=self.alpha2_rom,
                T1_expr=None,
            )
    
        self.setup_kirchhoff_problem()
    
        # Optional warm start for the nonlinear mechanical solve.
        if initial_guess is not None:
            if self.N_subdomains == 1:
                try:
                    if isinstance(initial_guess, (int, float, np.floating)):
                        self.w_.assign(interpolate(Constant(float(initial_guess)), self.W))
                    else:
                        try:
                            self.w_.assign(project(initial_guess, self.W))
                        except Exception:
                            self.w_.assign(interpolate(initial_guess, self.W))
                except Exception as exc:
                    print(f"[offline_solve_kirchhoff_problem] initial_guess ignored: {exc}")
            else:
                print(
                    "[offline_solve_kirchhoff_problem] initial_guess currently reused only "
                    "for N_subdomains == 1; ignored."
                )
    
        problem = NonlinearVariationalProblem(self.F, self.w_, self.bcs, self.J)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters.update(NEWTON_SOLVER_PARAMETERS)
        solver.solve()
    
        self.w_contact_solution = Function(self.W)
        self.w_contact_solution.assign(self.w_)
    
        print("Kirchhoff FEM solution computed.")
    
        chi_functions = {
            sid: Function(self.V_DG, name=f"chi_{sid}")
            for sid in self.subdomain_ids
        }
    
        for sid in self.subdomain_ids:
            chi_functions[sid].vector()[:] = (
                (self.subdomains.array() == sid).astype(float)
                if self.N_subdomains > 1
                else 1.0
            )
    
        if self.N_subdomains > 1:
            self.w_contact_subdomains = {
                sid: project(chi * self.w_contact_solution.sub(i), self.V_DG)
                for i, (sid, chi) in enumerate(chi_functions.items())
            }
    
            self.w_contact_global = project(
                sum(
                    chi * self.w_contact_solution.sub(i)
                    for i, (sid, chi) in enumerate(chi_functions.items())
                ),
                self.V_CG,
            )
    
        else:
            self.w_contact_subdomains = {
                1: project(self.w_contact_solution, self.V_DG)
            }
            self.w_contact_global = project(self.w_contact_solution, self.V_CG)
    
        return self.w_contact_global if return_mode == "global" else self.w_contact_solution

    # ------------------------------------------------------------------
    # Thermo-mechanical coupling helpers
    # ------------------------------------------------------------------
    def _copy_function(self, f):
        """Deep-copy a FEniCS Function."""
        out = Function(f.function_space())
        out.assign(f)
        return out
    
    def _as_plate_CG_function(self, value, cg_degree=None):
        """
        Convert scalar / UFL / Function / Expression to a scalar CG field on the 2D plate mesh.
        Rebuild the target CG space if the current mesh/degree does not match.
        """
        if not hasattr(self, "mesh"):
            raise RuntimeError("Call define_domain() first.")
    
        deg = max(1, int(self.degree if cg_degree is None else cg_degree))
    
        V2 = None
        if hasattr(self, "V_CG"):
            try:
                el = self.V_CG.ufl_element()
                same_mesh = self.V_CG.mesh().id() == self.mesh.id()
                same_family = el.family() in ("Lagrange", "CG")
                same_degree = int(el.degree()) == deg
                if same_mesh and same_family and same_degree:
                    V2 = self.V_CG
            except Exception:
                V2 = None
    
        if V2 is None:
            V2 = FunctionSpace(self.mesh, "CG", deg)
    
        if value is None:
            return interpolate(Constant(0.0), V2)
    
        if isinstance(value, (int, float, np.floating)):
            return interpolate(Constant(float(value)), V2)
    
        if isinstance(value, Function):
            try:
                same_mesh = value.function_space().mesh().id() == V2.mesh().id()
                same_dim = value.function_space().dim() == V2.dim()
                if same_mesh and same_dim:
                    out = Function(V2)
                    out.assign(value)
                    return out
            except Exception:
                pass
    
        try:
            return project(value, V2)
        except Exception:
            return interpolate(value, V2)

    def _field_l2_norm(self, f):
        """
        Compute the discrete L2(Omega) norm of a scalar finite element field
        by integration over the field mesh.
        """
        V = f.function_space()
        dx_f = Measure("dx", domain=V.mesh())
        val = assemble(inner(f, f) * dx_f)
        return float(np.sqrt(max(float(val), 0.0)))

    def _as_same_space_function(self, f, V):
        """
        Return f as a Function in V. If f is already in the same space, copy it;
        otherwise project/interpolate it onto V.
        """
        if isinstance(f, Function):
            try:
                same_mesh = f.function_space().mesh().id() == V.mesh().id()
                same_dim = f.function_space().dim() == V.dim()
                if same_mesh and same_dim:
                    out = Function(V)
                    out.assign(f)
                    return out
            except Exception:
                pass

        try:
            return project(f, V)
        except Exception:
            return interpolate(f, V)

    def _relative_l2_error_same_space(self, f_new, f_old, eps=1e-14):
        """
        Compute a relative L2(Omega) update between two scalar FE fields.

        This is the stopping norm used by the coupled thermo-mechanical
        fixed-point iteration. It is mass-matrix / integral based, not a raw
        Euclidean norm of the coefficient vector.
        """
        V = f_new.function_space()
        f_old_V = self._as_same_space_function(f_old, V)

        diff = Function(V)
        diff.vector().zero()
        diff.vector().axpy(1.0, f_new.vector())
        diff.vector().axpy(-1.0, f_old_V.vector())
        diff.vector().apply("insert")

        numerator = self._field_l2_norm(diff)
        denominator = max(self._field_l2_norm(f_new), float(eps))
        return float(numerator / denominator)

    def _relative_error_same_space(self, f_new, f_old, eps=1e-14):
        """
        Backward-compatible alias. Kept so older calls do not silently revert
        to coefficient-vector norms.
        """
        return self._relative_l2_error_same_space(f_new, f_old, eps=eps)

    def _relax_same_space(self, f_old, f_new, omega):
        """Apply under-relaxed update between same-space fields."""
        omega = float(omega)
        out = Function(f_new.function_space())
        arr = (1.0 - omega) * f_old.vector().get_local() + omega * f_new.vector().get_local()
        out.vector().set_local(arr)
        out.vector().apply("insert")
        return out
    # ------------------------------------------------------------------
    # Thermo-ROM parameter state and coupled workflow
    # ------------------------------------------------------------------
    def set_rom_thermal_parameters(
        self,
        *,
        T_amb=None, T_sub=None, h_con=None, eps_r=None, q_s=None,
        h_c_cont=None, h_c_gap=None, eta_c=None, w_contact=None,
        kx=None, ky=None, kz=None,
        alpha1=None, alpha2=None,
        rho=None
    ):
        """
        Set the thermal part mu_th of the split parameter vector
            mu = (mu_mech, mu_th)
    
        This does NOT solve anything; it only updates the solver state
        used by the thermal subproblem and thermo-mechanical coupling.
        """
        if T_amb is not None:
            self.T_amb_rom = float(T_amb)
        if T_sub is not None:
            self.T_sub_rom = float(T_sub)
        if h_con is not None:
            self.h_con_rom = float(h_con)
        if eps_r is not None:
            self.eps_r_rom = float(eps_r)
        if q_s is not None:
            self.q_s_rom = float(q_s)
    
        if h_c_cont is not None:
            self.h_c_cont_rom = float(h_c_cont)
        if h_c_gap is not None:
            self.h_c_gap_rom = float(h_c_gap)
        if eta_c is not None:
            self.eta_c_rom = float(eta_c)
        if w_contact is not None:
            self.w_contact_rom = float(w_contact)
    
        if kx is not None:
            self.kx_rom = float(kx)
        if ky is not None:
            self.ky_rom = float(ky)
        if kz is not None:
            self.kz_rom = float(kz)
    
        if alpha1 is not None:
            self.alpha1_rom = float(alpha1)
        if alpha2 is not None:
            self.alpha2_rom = float(alpha2)
    
        # internal self-weight parameter already handled by the class
        if rho is not None:
            self.rho.assign(float(rho))

    def build_plate_thermal_driver_from_rom_parameters(
        self,
        *,
        nx=None,
        ny=None,
        nz=10,
        degree=1,
        Nz_quad=8,
        T1_cg_degree=1,
        store=True,
        w_for_heat=None
    ):
        """
        Solve the 3D thermal subproblem using the current thermal parameter state
        and build the 2D plate thermal driver T1(x,y) from DeltaT(x,y,z).
    
        If w_for_heat is None, the legacy one-way behavior is used:
            w = self.w_contact_rom
    
        Otherwise, the provided plate field/scalar is used in h_c(w).
    
        The heat mesh is controlled explicitly through nx, ny, nz.  This is required
        for synchronized thermo-mechanical convergence studies.
        """
        w_used = self.w_contact_rom if w_for_heat is None else w_for_heat
    
        heat = self.heat_solve(
            kx=self.kx_rom,
            ky=self.ky_rom,
            kz=self.kz_rom,
    
            T_amb=self.T_amb_rom,
            h_con=self.h_con_rom,
            eps_r=self.eps_r_rom,
            q_s=self.q_s_rom,
    
            T_sub=self.T_sub_rom,
            h_c_cont=self.h_c_cont_rom,
            h_c_gap=self.h_c_gap_rom,
            eta_c=self.eta_c_rom,
            w=w_used,
    
            source=None,
            T_ref=None,
    
            nx=None if nx is None else int(nx),
            ny=None if ny is None else int(ny),
            nz=int(nz),
            degree=int(degree),
            store=bool(store),
        )
    
        T1_fn = self.make_T1_from_DeltaT(
            heat["DeltaT"],
            Nz_quad=int(Nz_quad),
            cg_degree=int(T1_cg_degree),
        )
    
        return heat, T1_fn

    def solve_thermal_given_w(
        self,
        *,
        w=None,
        Nz_quad=8,
        cg_degree=1,
        store=True,
        **heat_kwargs
    ):
        """
        One thermal solve with a specified plate displacement field/scalar w
        entering the bottom contact law h_c(w).
        """
        hk = dict(heat_kwargs)
        hk["w"] = w
        hk["store"] = bool(store)
    
        heat = self.heat_solve(**hk)
        T1_fn = self.make_T1_from_DeltaT(
            heat["DeltaT"],
            Nz_quad=int(Nz_quad),
            cg_degree=int(cg_degree)
        )
        return {"heat": heat, "T1": T1_fn}

    def solve_mechanical_given_T1(
        self,
        mu_mech,
        T1_expr,
        *,
        thermal_on=True,
        alpha_iso=None,
        alpha1=None,
        alpha2=None,
        initial_guess=None,
        return_mode="global",
    ):
        """
        Solve the Kirchhoff problem for a supplied plate thermal driver T1(x,y).
    
        This function is the safe thermo-mechanical mechanical-solve wrapper.
        It explicitly enables the thermal state before calling the KL solver.
        """
        alpha_iso_eff = self._alpha_iso if alpha_iso is None else float(alpha_iso)
        alpha1_eff = alpha_iso_eff if alpha1 is None else float(alpha1)
        alpha2_eff = alpha_iso_eff if alpha2 is None else float(alpha2)
    
        self.update_thermal_parameters(
            enable=bool(thermal_on),
            alpha_iso=alpha_iso_eff,
            alpha1=alpha1_eff,
            alpha2=alpha2_eff,
            T1_expr=(T1_expr if thermal_on else None),
        )
    
        w_ret = self.offline_solve_kirchhoff_problem(
            mu=mu_mech,
            return_mode=return_mode,
            initial_guess=initial_guess,
            thermal_on=bool(thermal_on),
        )
    
        w_global = Function(self.V_CG)
        w_global.assign(self.w_contact_global)
    
        return {
            "w_return": w_ret,
            "w_global": w_global,
            "w_solution": self.w_contact_solution,
        }

    def solve_coupled_thermo_mechanical(
        self,
        mu_mech,
        *,
        heat_kwargs=None,
        use_rom_thermal_parameters=False,
        alpha_iso=None,
        alpha1=None,
        alpha2=None,
        w0=None,
        omega=0.5,
        tol_w=1e-4,
        tol_T1=1e-4,
        max_coupling_iters=20,
    
        # ------------------------------------------------------------------
        # Explicit heat-discretization controls.
        # These are required for clean thermo-mechanical convergence studies.
        # If heat_nx/heat_ny are left as None, heat_solve keeps its legacy
        # fallback nx=ny=max(2,int(self.size/6)).
        # ------------------------------------------------------------------
        heat_nx=None,
        heat_ny=None,
        heat_nz=10,
        heat_degree=1,
        Nz_quad_T1=8,
        T1_cg_degree=1,
    
        store_each_iter=True,
        verbose=True,
        convergence_plot=False,
        convergence_plot_label=None,
        convergence_plot_savebase=None
    ):
        """
        Partitioned thermo-mechanical coupling with relaxed-iterate stopping.
    
        Outer iteration:
            w^(k) -> heat(w^(k)) -> T1^(k+1)
                  -> mechanics(T1^(k+1)) = w_tilde^(k+1)
                  -> relaxed update
                     w^(k+1) = (1 - omega) w^(k) + omega w_tilde^(k+1)
    
        Stopping test:
            compare the accepted relaxed iterate w^(k+1) against w^(k),
            and compare T1^(k+1) against T1^(k), using relative L2(Ω)
            field norms assembled on the plate mesh.
    
        Final reported state:
            the solver reports the last accepted relaxed displacement and the last
            thermal driver computed inside the outer iteration. No extra thermal
            reconciliation solve and no final mechanical cleanup solve are performed
            after convergence.
    
        Important convergence-analysis controls:
            heat_nx, heat_ny, heat_nz, heat_degree, Nz_quad_T1, T1_cg_degree.
    
        These make the 3D heat discretization explicit rather than hidden behind
        the default heat_solve fallback nx=ny=max(2, int(self.size / 6)).
        """
        if not (0.0 < float(omega) <= 1.0):
            raise ValueError("solve_coupled_thermo_mechanical: omega must be in (0, 1].")
    
        if int(max_coupling_iters) < 1:
            raise ValueError("solve_coupled_thermo_mechanical: max_coupling_iters must be >= 1.")
    
        if not hasattr(self, "mesh"):
            raise RuntimeError("solve_coupled_thermo_mechanical: call define_domain() first.")
    
        if heat_nx is not None and int(heat_nx) < 1:
            raise ValueError("solve_coupled_thermo_mechanical: heat_nx must be positive or None.")
    
        if heat_ny is not None and int(heat_ny) < 1:
            raise ValueError("solve_coupled_thermo_mechanical: heat_ny must be positive or None.")
    
        if int(heat_nz) < 1:
            raise ValueError("solve_coupled_thermo_mechanical: heat_nz must be positive.")
    
        if int(heat_degree) < 1:
            raise ValueError("solve_coupled_thermo_mechanical: heat_degree must be >= 1.")
    
        if int(T1_cg_degree) < 1:
            raise ValueError("solve_coupled_thermo_mechanical: T1_cg_degree must be >= 1.")
    
        def _plot_coupling_history(history, label=None, savebase=None):
            if history is None or len(history) == 0:
                return
    
            hist = pd.DataFrame(history).copy()
            it = hist["iter"].to_numpy(dtype=int)
            err_w_arr = hist["err_w"].to_numpy(dtype=float)
            err_T1_arr = (
                hist["err_T1"].to_numpy(dtype=float)
                if "err_T1" in hist.columns
                else np.full_like(err_w_arr, np.nan, dtype=float)
            )
            w_min_arr = hist["w_min"].to_numpy(dtype=float)
            w_max_arr = hist["w_max"].to_numpy(dtype=float)
    
            plt.style.use(["science", "ieee", "notebook", "grid"])
            plt.rcParams["figure.autolayout"] = False
    
            fig = plt.figure(figsize=(12.0, 4.8), facecolor="white")
            gs = GridSpec(
                1, 2, figure=fig,
                width_ratios=[1.0, 1.0],
                left=0.070, right=0.985,
                bottom=0.160, top=0.90,
                wspace=0.28,
            )
            ax0 = fig.add_subplot(gs[0, 0])
            ax1 = fig.add_subplot(gs[0, 1])
    
            suffix = "" if label is None else f" - {label}"
    
            def _style_axis(ax, title, xlabel, ylabel):
                ax.set_facecolor("white")
                ax.set_title(title, fontsize=14, pad=8)
                ax.set_xlabel(xlabel, fontsize=14)
                ax.set_ylabel(ylabel, fontsize=14)
                ax.tick_params(axis="both", labelsize=13, length=3, pad=2)
                ax.grid(True, which="both", linestyle=":", linewidth=0.45, alpha=0.35)
    
            ax0.semilogy(
                it,
                err_w_arr,
                color="black",
                linestyle="-",
                marker="o",
                linewidth=2.2,
                markersize=8.0,
                markerfacecolor="black",
                label=r"$\varepsilon_w$",
            )
    
            if np.isfinite(err_T1_arr[1:]).any():
                ax0.semilogy(
                    it,
                    err_T1_arr,
                    color="red",
                    linestyle="--",
                    marker="s",
                    linewidth=2.2,
                    markersize=7.5,
                    markerfacecolor="red",
                    label=r"$\varepsilon_{T_1}$",
                )
    
            _style_axis(
                ax0,
                f"Coupled convergence{suffix}",
                "Coupling iteration",
                "Relative update",
            )
    
            leg0 = ax0.legend(
                loc="upper right",
                fontsize=13,
                frameon=True,
                borderpad=0.5,
                handlelength=2.2,
                handletextpad=0.6,
                labelspacing=0.4,
            )
            leg0.get_frame().set_edgecolor("black")
            leg0.get_frame().set_linewidth(0.8)
    
            ax1.plot(
                it,
                w_min_arr,
                color="black",
                linestyle="-",
                marker="o",
                linewidth=2.2,
                markersize=6.8,
                markerfacecolor="black",
                label=r"$w_{\min}$",
            )
            ax1.plot(
                it,
                w_max_arr,
                color="red",
                linestyle="--",
                marker="s",
                linewidth=2.2,
                markersize=6.8,
                markerfacecolor="red",
                label=r"$w_{\max}$",
            )
    
            _style_axis(
                ax1,
                f"Response evolution{suffix}",
                "Coupling iteration",
                "Displacement [m]",
            )
    
            yfmt = ScalarFormatter(useMathText=True)
            yfmt.set_scientific(True)
            yfmt.set_powerlimits((0, 0))
            ax1.yaxis.set_major_formatter(yfmt)
            ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax1.yaxis.get_offset_text().set_size(13)
    
            leg1 = ax1.legend(
                loc="center right",
                fontsize=13,
                frameon=True,
                borderpad=0.5,
                handlelength=2.2,
                handletextpad=0.6,
                labelspacing=0.4,
            )
            leg1.get_frame().set_edgecolor("black")
            leg1.get_frame().set_linewidth(0.8)
    
            if savebase is not None:
                save_dir = os.path.dirname(savebase)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
    
                fig.savefig(
                    f"{savebase}.pdf",
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=0.04,
                    facecolor="white",
                )
                fig.savefig(
                    f"{savebase}.png",
                    dpi=600,
                    bbox_inches="tight",
                    pad_inches=0.04,
                    facecolor="white",
                )
    
            plt.show()
    
        def _thermal_solve_for_current_w(w_current, store_flag):
            """
            One thermal subproblem evaluation:
                w_current -> 3D heat solve -> 2D T1(x,y)
    
            This helper deliberately calls heat_solve directly so the coupled solver
            can control heat_nx and heat_ny even if the one-way wrapper has not been
            patched yet.
            """
            if use_rom_thermal_parameters:
                heat = self.heat_solve(
                    kx=self.kx_rom,
                    ky=self.ky_rom,
                    kz=self.kz_rom,
    
                    T_amb=self.T_amb_rom,
                    h_con=self.h_con_rom,
                    eps_r=self.eps_r_rom,
                    q_s=self.q_s_rom,
    
                    T_sub=self.T_sub_rom,
                    h_c_cont=self.h_c_cont_rom,
                    h_c_gap=self.h_c_gap_rom,
                    eta_c=self.eta_c_rom,
    
                    w=w_current,
                    source=None,
                    T_ref=None,
    
                    nx=None if heat_nx is None else int(heat_nx),
                    ny=None if heat_ny is None else int(heat_ny),
                    nz=int(heat_nz),
                    degree=int(heat_degree),
                    store=bool(store_flag),
                )
    
            else:
                if heat_kwargs is None:
                    raise ValueError(
                        "solve_coupled_thermo_mechanical: provide heat_kwargs "
                        "or set use_rom_thermal_parameters=True."
                    )
    
                hk = dict(heat_kwargs)
                hk["w"] = w_current
    
                if heat_nx is not None:
                    hk["nx"] = int(heat_nx)
                else:
                    hk.setdefault("nx", None)
    
                if heat_ny is not None:
                    hk["ny"] = int(heat_ny)
                else:
                    hk.setdefault("ny", None)
    
                hk.setdefault("nz", int(heat_nz))
                hk.setdefault("degree", int(heat_degree))
                hk.setdefault("store", bool(store_flag))
    
                heat = self.heat_solve(**hk)
    
            T1_fn = self.make_T1_from_DeltaT(
                heat["DeltaT"],
                Nz_quad=int(Nz_quad_T1),
                cg_degree=int(T1_cg_degree),
            )
    
            return heat, T1_fn
    
        # ------------------------------------------------------------------
        # Initial iterate and thermal-expansion data
        # ------------------------------------------------------------------
        if use_rom_thermal_parameters:
            w_used = self._as_plate_CG_function(
                self.w_contact_rom if w0 is None else w0
            )
            alpha_iso_eff = self._alpha_iso if alpha_iso is None else float(alpha_iso)
            alpha1_eff = self.alpha1_rom if alpha1 is None else float(alpha1)
            alpha2_eff = self.alpha2_rom if alpha2 is None else float(alpha2)
    
        else:
            if heat_kwargs is None:
                raise ValueError(
                    "solve_coupled_thermo_mechanical: provide heat_kwargs "
                    "or set use_rom_thermal_parameters=True."
                )
    
            w_used = self._as_plate_CG_function(0.0 if w0 is None else w0)
            alpha_iso_eff = self._alpha_iso if alpha_iso is None else float(alpha_iso)
            alpha1_eff = alpha_iso_eff if alpha1 is None else float(alpha1)
            alpha2_eff = alpha_iso_eff if alpha2 is None else float(alpha2)
    
        T1_prev = None
        w_mech_guess = None
    
        converged = False
        stopped_by_max_iters = False
        termination_reason = None
    
        history = []
    
        heat = None
        T1_new = None
        w_raw = None
        w_relaxed = None
        last_w_predictor = None
    
    
        # ------------------------------------------------------------------
        # Outer staggered fixed-point iteration
        # ------------------------------------------------------------------
        for it in range(1, int(max_coupling_iters) + 1):
    
            # 1) Thermal solve with current accepted iterate w^(k)
            heat, T1_new = _thermal_solve_for_current_w(
                w_current=w_used,
                store_flag=bool(store_each_iter),
            )
    
            # 2) Mechanical solve with updated thermal bending driver
            mech = self.solve_mechanical_given_T1(
                mu_mech,
                T1_new,
                thermal_on=True,
                alpha_iso=alpha_iso_eff,
                alpha1=alpha1_eff,
                alpha2=alpha2_eff,
                initial_guess=w_mech_guess,
                return_mode="global",
            )
    
            w_raw = mech["w_global"]
            last_w_predictor = self._copy_function(w_raw)
    
            # 3) Relaxed accepted iterate
            w_relaxed = self._relax_same_space(w_used, w_raw, omega)
    
            # 4) Convergence diagnostics: relative L2(Omega) field updates
            err_w = self._relative_l2_error_same_space(w_relaxed, w_used)
            err_T1 = (
                np.nan
                if T1_prev is None
                else self._relative_l2_error_same_space(T1_new, T1_prev)
            )
    
            w_rel_arr = w_relaxed.vector().get_local()
            w_raw_arr = w_raw.vector().get_local()
    
            heat_dofs = int(heat["Vt"].dim()) if heat is not None and "Vt" in heat else -1
            T1_dofs = int(T1_new.function_space().dim()) if T1_new is not None else -1
    
            history.append({
                "iter": int(it),
                "err_w": float(err_w),
                "err_T1": float(err_T1),
                "stopping_norm": "L2_Omega",
    
                "w_min": float(w_rel_arr.min()),
                "w_max": float(w_rel_arr.max()),
                "w_raw_min": float(w_raw_arr.min()),
                "w_raw_max": float(w_raw_arr.max()),
    
                "T_ref": float(heat["T_ref"]),
                "heat_nx": int(heat_nx) if heat_nx is not None else -1,
                "heat_ny": int(heat_ny) if heat_ny is not None else -1,
                "heat_nz": int(heat_nz),
                "heat_degree": int(heat_degree),
                "heat_dofs": int(heat_dofs),
                "T1_dofs": int(T1_dofs),
            })
    
            if verbose:
                heat_mesh_msg = (
                    f"heat=({heat_nx},{heat_ny},{int(heat_nz)}), "
                    f"pT={int(heat_degree)}, heat DoFs={heat_dofs}"
                )
                print(
                    f"[coupling {it:02d}] "
                    f"err_w = {err_w:.3e}, err_T1 = {err_T1:.3e}, "
                    f"w_min = {w_rel_arr.min():.6e}, "
                    f"w_max = {w_rel_arr.max():.6e}, "
                    f"{heat_mesh_msg}"
                )
    
            # 5) Stopping test after relaxation
            if (it > 1) and (err_w < float(tol_w)) and (err_T1 < float(tol_T1)):
                converged = True
                termination_reason = "converged"
                w_used = self._copy_function(w_relaxed)
                T1_prev = self._copy_function(T1_new)
                break
    
            # Continue with relaxed accepted iterate
            w_used = self._copy_function(w_relaxed)
            T1_prev = self._copy_function(T1_new)
            w_mech_guess = self._copy_function(w_raw)
    
        if not converged:
            stopped_by_max_iters = True
            termination_reason = "max_iters_reached"
    
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Final accepted coupled state
        # ------------------------------------------------------------------
        # The final displacement is the accepted relaxed iterate produced by
        # the outer fixed-point iteration. No post-loop thermal reconciliation
        # and no final mechanical cleanup solve are performed.
        w_final = self._copy_function(w_used)
        T1_final = self._copy_function(T1_new)
    
        # Final state stored on the object
        # ------------------------------------------------------------------
        self.update_thermal_parameters(
            enable=True,
            alpha_iso=alpha_iso_eff,
            alpha1=alpha1_eff,
            alpha2=alpha2_eff,
            T1_expr=T1_final,
        )
    
        self.coupled_heat = heat
        self.coupled_T1 = self._copy_function(T1_final)
        self.coupled_w_global = self._copy_function(w_final)
        self.coupled_w_iterate = self._copy_function(w_used)
    
        self.coupled_w_raw_outer = self._copy_function(
            last_w_predictor if last_w_predictor is not None else w_final
        )
        self.coupled_w_raw_final = self._copy_function(
            last_w_predictor if last_w_predictor is not None else w_final
        )
    
        self.coupled_history = history
        self.coupled_converged = bool(converged)
        self.coupled_stopped_by_max_iters = bool(stopped_by_max_iters)
        self.coupled_termination_reason = termination_reason
        self.coupled_max_coupling_iters = int(max_coupling_iters)
    
        # Store explicit heat-discretization metadata.
        self.coupled_heat_nx = None if heat_nx is None else int(heat_nx)
        self.coupled_heat_ny = None if heat_ny is None else int(heat_ny)
        self.coupled_heat_nz = int(heat_nz)
        self.coupled_heat_degree = int(heat_degree)
        self.coupled_Nz_quad_T1 = int(Nz_quad_T1)
        self.coupled_T1_cg_degree = int(T1_cg_degree)
    
        if convergence_plot:
            _plot_coupling_history(
                history,
                label=convergence_plot_label,
                savebase=convergence_plot_savebase,
            )
    
        final_heat_dofs = int(heat["Vt"].dim()) if heat is not None and "Vt" in heat else -1
        final_T1_dofs = int(self.coupled_T1.function_space().dim())
        final_w_dofs = int(self.coupled_w_global.function_space().dim())
    
        return {
            "heat": heat,
            "T1": self.coupled_T1,
            "w": self.coupled_w_global,
    
            "w_iterate": self.coupled_w_iterate,
            "w_raw_final": self.coupled_w_raw_final,
            "w_raw_outer": self.coupled_w_raw_outer,
    
            "history": history,
            "iters": len(history),
    
            "converged": bool(converged),
            "stopped_by_max_iters": bool(stopped_by_max_iters),
            "termination_reason": termination_reason,
            "max_coupling_iters": int(max_coupling_iters),
    
            "err_w": float(history[-1]["err_w"]),
            "err_T1": float(history[-1]["err_T1"]),
    
    
            # Explicit discretization metadata for convergence tables.
            "heat_nx": None if heat_nx is None else int(heat_nx),
            "heat_ny": None if heat_ny is None else int(heat_ny),
            "heat_nz": int(heat_nz),
            "heat_degree": int(heat_degree),
            "Nz_quad_T1": int(Nz_quad_T1),
            "T1_cg_degree": int(T1_cg_degree),
    
            "w_dofs": int(final_w_dofs),
            "heat_dofs": int(final_heat_dofs),
            "T1_dofs": int(final_T1_dofs),
            "total_coupled_dofs": int(final_w_dofs + final_heat_dofs + final_T1_dofs),
        }

    def solve_rom_sample(
        self,
        mu_mech,
        *,
        thermal_on=False,
        coupled_on=False,
        heat_nx=None,
        heat_ny=None,
        heat_nz=10,
        heat_degree=1,
        Nz_quad_T1=8,
        T1_cg_degree=1,
        coupling_omega=0.5,
        coupling_tol_w=1e-4,
        coupling_tol_T1=1e-4,
        coupling_max_iters=20,
        coupling_verbose=False,
        w0=None,
        return_mode="global",
        return_coupled_dict=False,
    ):
        """
        Solve one ROM/FOM sample with split parametrization:
            mu = (mu_mech, mu_th)
    
        Modes
        -----
        1) thermal_on=False, coupled_on=False
           -> purely mechanical
    
        2) thermal_on=True, coupled_on=False
           -> one-way thermo-mechanical
    
        3) thermal_on=True, coupled_on=True
           -> coupled thermo-mechanical
    
        The heat mesh is controlled explicitly through heat_nx, heat_ny, heat_nz.
    
        Important:
        ----------
        offline_solve_kirchhoff_problem defaults to thermal_on=False in order to
        protect the mechanical branch from stale thermal state. Therefore, the
        one-way branch must explicitly call it with thermal_on=True.
        """
        if coupled_on and not thermal_on:
            raise ValueError("solve_rom_sample: coupled_on=True requires thermal_on=True.")
    
        # ------------------------------------------------------------------
        # Coupled thermo-mechanical mode
        # ------------------------------------------------------------------
        if coupled_on:
            out = self.solve_coupled_thermo_mechanical(
                mu_mech,
                use_rom_thermal_parameters=True,
                alpha1=self.alpha1_rom,
                alpha2=self.alpha2_rom,
                w0=0.0 if w0 is None else w0,
                omega=float(coupling_omega),
                tol_w=float(coupling_tol_w),
                tol_T1=float(coupling_tol_T1),
                max_coupling_iters=int(coupling_max_iters),
    
                heat_nx=None if heat_nx is None else int(heat_nx),
                heat_ny=None if heat_ny is None else int(heat_ny),
                heat_nz=int(heat_nz),
                heat_degree=int(heat_degree),
                Nz_quad_T1=int(Nz_quad_T1),
                T1_cg_degree=int(T1_cg_degree),
    
                store_each_iter=True,
                verbose=bool(coupling_verbose),
            )
    
            return out if return_coupled_dict else out["heat"]
    
        # ------------------------------------------------------------------
        # Mechanical or one-way thermo-mechanical mode
        # ------------------------------------------------------------------
        heat = None
    
        if thermal_on:
            heat, T1_fn = self.build_plate_thermal_driver_from_rom_parameters(
                nx=None if heat_nx is None else int(heat_nx),
                ny=None if heat_ny is None else int(heat_ny),
                nz=int(heat_nz),
                degree=int(heat_degree),
                Nz_quad=int(Nz_quad_T1),
                T1_cg_degree=int(T1_cg_degree),
                store=True,
                w_for_heat=None,
            )
    
            self.update_thermal_parameters(
                enable=True,
                alpha1=self.alpha1_rom,
                alpha2=self.alpha2_rom,
                T1_expr=T1_fn,
            )
    
        else:
            self.update_thermal_parameters(
                enable=False,
                alpha1=self.alpha1_rom,
                alpha2=self.alpha2_rom,
                T1_expr=None,
            )
    
        self.offline_solve_kirchhoff_problem(
            mu=mu_mech,
            return_mode=return_mode,
            thermal_on=bool(thermal_on),
        )
    
        return heat
    # ------------------------------------------------------------------
    # Snapshot generation
    # ------------------------------------------------------------------
    def generate_and_store_snapshots(self, training_set, save_filename="snapshots.npz"):
        """Generate and store mechanical full-order snapshots."""
        root, temp_dir = self.output_dir, os.path.join(self.output_dir, "tmp_snapshot_data")
        os.makedirs(temp_dir, exist_ok=True)
        existing_files = [int(f[9:-4]) for f in os.listdir(temp_dir) if f.startswith("snapshot_") and f.endswith(".npy")] if os.path.exists(temp_dir) else []
        start_index = max(existing_files) if existing_files else 0
        print(f"\n+++ {'Resuming snapshot generation, starting by re-running snapshot #' + str(start_index + 1) if existing_files else 'Starting New Snapshot Generation'} +++")
        print(f"{'Snapshot':<10}{'Mu Values'}\n{'=' * 60}")
        for i in range(start_index, len(training_set)):
            mu = training_set[i]
            try:
                snapshot = self.offline_solve_kirchhoff_problem(mu, return_mode="Solution")
                np.save(os.path.join(temp_dir, f"snapshot_{i}.npy"), snapshot.vector().get_local())
                np.save(os.path.join(temp_dir, f"mu_{i}.npy"), np.array(mu))
                print(f"{i+1:<10} {mu}")
            except Exception as e:
                print(f"--- ERROR on snapshot {i+1}. Stopping. ---\n--- Reason: {e} ---")
                return
        print("\n+++ Generation complete. Consolidating all snapshots. +++")
        snapshots, mus = zip(*[
            (np.load(os.path.join(temp_dir, f"snapshot_{i}.npy")), np.load(os.path.join(temp_dir, f"mu_{i}.npy")))
            for i in range(len(training_set)) if os.path.exists(os.path.join(temp_dir, f"snapshot_{i}.npy"))
        ])
        final_path = os.path.join(root, save_filename)
        np.savez(final_path, snapshots=np.array(snapshots), mus=np.array(mus))
        print(f"\n Saved all {len(snapshots)} snapshots to '{final_path}'")

    def generate_and_store_thermo_snapshots(
        self,
        sample_list,
        decode_mu_fn,
        save_filename="thermo_snapshots.npz",
    
        # --------------------------------------------------------------
        # Explicit thermal discretization controls
        # --------------------------------------------------------------
        heat_nx=None,
        heat_ny=None,
        heat_nz=10,
        heat_degree=1,
        Nz_quad_T1=8,
        T1_cg_degree=1,
    
        coupled_on=True,
        coupling_omega=0.5,
        coupling_tol_w=1e-4,
        coupling_tol_T1=1e-4,
        coupling_max_iters=20,
        save_heat_metadata=True
    ):
        """
        Generate thermo-mechanical ROM/FOM snapshots using split parametrization:
    
            mu = (mu_mech, mu_th)
    
        Parameters
        ----------
        sample_list : list[list[float]]
            Full sampled parameter vectors.
    
        decode_mu_fn : callable
            Function with signature:
                mu_mech, mu_th, meta = decode_mu_fn(sample, solver)
    
        coupled_on : bool
            True  -> generate coupled thermo-mechanical snapshots.
            False -> generate one-way thermo-mechanical snapshots.
    
        heat_nx, heat_ny, heat_nz, heat_degree : int or None
            Explicit 3D heat-mesh and heat-FE controls.
    
            If heat_nx/heat_ny are None, heat_solve keeps its internal legacy
            fallback. For controlled ROM snapshot generation, pass both explicitly.
    
        Nz_quad_T1 : int
            Number of through-thickness quadrature points used to reduce DeltaT
            to the plate thermal driver T1(x,y).
    
        T1_cg_degree : int
            Polynomial degree of the 2D T1(x,y) field on the plate mesh.
    
        Stored arrays
        -------------
        snapshots
            Final displacement snapshots.
    
        parameters
            Full sampled parameter vectors.
    
        mech_parameters
            Mechanical parameter block.
    
        thermal_parameters
            Selected thermal/environmental parameter block.
    
        heat_summaries
            Per-sample thermal metadata.
    
        snapshot_dofs
            Per-sample DoF accounting:
                [w_dofs, heat_dofs, T1_dofs, total_dofs]
    
        heat_discretizations
            Per-sample heat-discretization metadata:
                [heat_nx, heat_ny, heat_nz, heat_degree, Nz_quad_T1, T1_cg_degree]
    
        coupling_summaries
            Only stored when coupled_on=True:
                [iters, converged_flag, max_iters_flag, max_iters_budget,
                 err_w, err_T1]
        """
        if int(heat_nz) < 1:
            raise ValueError("generate_and_store_thermo_snapshots: heat_nz must be positive.")
    
        if int(heat_degree) < 1:
            raise ValueError("generate_and_store_thermo_snapshots: heat_degree must be >= 1.")
    
        if int(T1_cg_degree) < 1:
            raise ValueError("generate_and_store_thermo_snapshots: T1_cg_degree must be >= 1.")
    
        if heat_nx is not None and int(heat_nx) < 1:
            raise ValueError("generate_and_store_thermo_snapshots: heat_nx must be positive or None.")
    
        if heat_ny is not None and int(heat_ny) < 1:
            raise ValueError("generate_and_store_thermo_snapshots: heat_ny must be positive or None.")
    
        snapshots = []
        parameters = []
        mech_parameters = []
        thermal_parameters = []
        metadata = []
    
        heat_summaries = []
        heat_discretizations = []
        snapshot_dofs = []
        coupling_summaries = []
    
        mode_name = "coupled" if coupled_on else "one-way"
    
        print("=" * 80)
        print(f"Generating thermo-mechanical snapshots in {mode_name} mode")
        print(
            "Thermal discretization: "
            f"heat_nx={heat_nx}, heat_ny={heat_ny}, heat_nz={heat_nz}, "
            f"heat_degree={heat_degree}, Nz_quad_T1={Nz_quad_T1}, "
            f"T1_cg_degree={T1_cg_degree}"
        )
        print("=" * 80)
    
        for i, sample in enumerate(sample_list):
            mu_mech, mu_th, meta = decode_mu_fn(sample, self)
    
            # ----------------------------------------------------------
            # Set thermal/environmental parameter block for this sample
            # ----------------------------------------------------------
            self.set_rom_thermal_parameters(**mu_th)
    
            # ----------------------------------------------------------
            # Solve selected thermo-mechanical mode
            # ----------------------------------------------------------
            if coupled_on:
                out = self.solve_rom_sample(
                    mu_mech,
                    thermal_on=True,
                    coupled_on=True,
    
                    heat_nx=None if heat_nx is None else int(heat_nx),
                    heat_ny=None if heat_ny is None else int(heat_ny),
                    heat_nz=int(heat_nz),
                    heat_degree=int(heat_degree),
                    Nz_quad_T1=int(Nz_quad_T1),
                    T1_cg_degree=int(T1_cg_degree),
    
                    coupling_omega=float(coupling_omega),
                    coupling_tol_w=float(coupling_tol_w),
                    coupling_tol_T1=float(coupling_tol_T1),
                    coupling_max_iters=int(coupling_max_iters),
                    coupling_verbose=False,
    
                    return_mode="global",
                    return_coupled_dict=True,
                )
    
                heat = out["heat"]
                T1_field = out["T1"]
                w_field = out["w"]
                w_snapshot = w_field.vector().get_local().copy()
    
                coupling_summaries.append(np.array([
                    float(out.get("iters", np.nan)),
                    1.0 if bool(out.get("converged", False)) else 0.0,
                    1.0 if bool(out.get("stopped_by_max_iters", False)) else 0.0,
                    float(out.get("max_coupling_iters", coupling_max_iters)),
                    float(out.get("err_w", np.nan)),
                    float(out.get("err_T1", np.nan)),
                ], dtype=float))
    
                w_dofs = int(out.get("w_dofs", w_field.function_space().dim()))
                heat_dofs = int(out.get("heat_dofs", heat["Vt"].dim() if heat is not None else 0))
                T1_dofs = int(out.get("T1_dofs", T1_field.function_space().dim()))
                total_dofs = int(out.get("total_coupled_dofs", w_dofs + heat_dofs + T1_dofs))
    
            else:
                heat = self.solve_rom_sample(
                    mu_mech,
                    thermal_on=True,
                    coupled_on=False,
    
                    heat_nx=None if heat_nx is None else int(heat_nx),
                    heat_ny=None if heat_ny is None else int(heat_ny),
                    heat_nz=int(heat_nz),
                    heat_degree=int(heat_degree),
                    Nz_quad_T1=int(Nz_quad_T1),
                    T1_cg_degree=int(T1_cg_degree),
    
                    return_mode="global",
                )
    
                w_field = self.w_contact_global
                T1_field = getattr(self, "T1_from_heat", None)
                w_snapshot = w_field.vector().get_local().copy()
    
                w_dofs = int(w_field.function_space().dim())
                heat_dofs = int(heat["Vt"].dim()) if heat is not None else 0
                T1_dofs = int(T1_field.function_space().dim()) if T1_field is not None else 0
                total_dofs = int(w_dofs + heat_dofs + T1_dofs)
    
            # ----------------------------------------------------------
            # Store displacement snapshot
            # ----------------------------------------------------------
            snapshots.append(w_snapshot)
    
            # ----------------------------------------------------------
            # Store parameter blocks
            # ----------------------------------------------------------
            parameters.append(np.array(sample, dtype=float))
            mech_parameters.append(np.array(mu_mech, dtype=float))
    
            thermal_parameters.append(
                np.array([
                    mu_th.get("q_s", self.q_s_rom),
                    mu_th.get("T_amb", self.T_amb_rom),
                    mu_th.get("T_sub", self.T_sub_rom),
                    mu_th.get("h_c_cont", self.h_c_cont_rom),
                    mu_th.get("h_c_gap", self.h_c_gap_rom),
                    mu_th.get("alpha1", self.alpha1_rom),
                    mu_th.get("alpha2", self.alpha2_rom),
                    mu_th.get("rho", float(self.rho)),
                ], dtype=float)
            )
    
            metadata.append(meta)
    
            # ----------------------------------------------------------
            # Store discretization and DoF metadata
            # ----------------------------------------------------------
            heat_discretizations.append(np.array([
                -1.0 if heat_nx is None else float(heat_nx),
                -1.0 if heat_ny is None else float(heat_ny),
                float(heat_nz),
                float(heat_degree),
                float(Nz_quad_T1),
                float(T1_cg_degree),
            ], dtype=float))
    
            snapshot_dofs.append(np.array([
                float(w_dofs),
                float(heat_dofs),
                float(T1_dofs),
                float(total_dofs),
            ], dtype=float))
    
            if save_heat_metadata and heat is not None:
                heat_summaries.append(np.array([
                    float(heat["T_ref"]),
                    float(heat["Vt"].dim()),
                    -1.0 if heat_nx is None else float(heat_nx),
                    -1.0 if heat_ny is None else float(heat_ny),
                    float(heat_nz),
                    float(heat_degree),
                    float(Nz_quad_T1),
                    float(T1_cg_degree),
                ], dtype=float))
    
            if (i + 1) % 25 == 0 or (i + 1) == len(sample_list):
                print(
                    f"  thermo snapshots: completed {i + 1}/{len(sample_list)} | "
                    f"w DoFs={w_dofs}, heat DoFs={heat_dofs}, "
                    f"T1 DoFs={T1_dofs}, total={total_dofs}"
                )
    
        # --------------------------------------------------------------
        # Save compressed archive
        # --------------------------------------------------------------
        save_dict = dict(
            snapshots=np.asarray(snapshots),
            parameters=np.asarray(parameters),
            mech_parameters=np.asarray(mech_parameters),
            thermal_parameters=np.asarray(thermal_parameters),
            metadata=np.asarray(metadata, dtype=object),
    
            # New metadata arrays
            heat_discretizations=np.asarray(heat_discretizations),
            snapshot_dofs=np.asarray(snapshot_dofs),
        )
    
        if save_heat_metadata:
            save_dict["heat_summaries"] = np.asarray(heat_summaries)
    
        if coupled_on:
            save_dict["coupling_summaries"] = np.asarray(coupling_summaries)
    
        output_path = os.path.join(self.output_dir, save_filename)
    
        np.savez_compressed(
            output_path,
            **save_dict,
        )
    
        print(f"Saved thermo snapshots to: {output_path}")
    
        return output_path

    def generate_nonlinear_snapshots(self, training_set, save_filename="nonlinear_snapshots_converged.npz"):
        """Generate nonlinear foundation snapshots from converged full-order states."""
        root, temp_dir = self.output_dir, os.path.join(self.output_dir, "tmp_nonlinear_converged_data")
        os.makedirs(temp_dir, exist_ok=True)
        # File helper function
        file_path = lambda i, ftype: os.path.join(temp_dir, f"{ftype}_{i}.npy")
        # Find all missing snapshots
        missing = [i for i in range(len(training_set)) 
                if not all(os.path.exists(file_path(i, ftype)) 
                            for ftype in ["nl", "fom", "mu", "labels"])]
        if not missing:
            print("All snapshots already exist. Consolidating.")
            self._consolidate_snapshots(temp_dir, training_set, root, save_filename)
            return
        print(f"Generating {len(missing)} missing snapshots: indices {missing}")
        print("NOTE: Using CONVERGED solutions for nonlinear basis")
        try:
            for i in missing:
                mu = training_set[i]
                # Clean existing files
                for ftype in ["nl", "fom", "mu", "labels"]:
                    if os.path.exists(f := file_path(i, ftype)):
                        os.remove(f)
                # Generate snapshot from converged solution
                try:
                    # Solve to full convergence for this parameter
                    self.set_mu(mu)
                    self.setup_kirchhoff_problem()
                    w_converged = self.offline_solve_kirchhoff_problem(mu, return_mode="Solution")
                    # Verify solution convergence
                    w_vec = w_converged.vector().get_local()
                    if len(w_vec) != self.W.dim():
                        raise RuntimeError(f"Size mismatch: {len(w_vec)} ≠ {self.W.dim()}")
                    # Evaluate nonlinear term at converged state
                    self.w_.assign(w_converged)
                    nl_converged = assemble(self.F_foundation).get_local().copy()
                    # Store single converged snapshot
                    nl_snaps = [nl_converged]
                    fom_snaps = [w_vec.copy()]
                    labels = ['converged']
                    # Save files
                    np.save(file_path(i, "nl"), nl_snaps)
                    np.save(file_path(i, "fom"), fom_snaps)
                    np.save(file_path(i, "mu"), mu)
                    np.save(file_path(i, "labels"), labels)
                    print(f"Snapshot {i+1} generated: μ={mu}")
                except Exception as e:
                    print(f"Snapshot {i+1} failed: {str(e)}")
                    # Instead of continuing, raise the exception to stop execution
                    raise RuntimeError(f"Simulation failed for parameter set {mu}. Stopping execution.") from e
        except Exception as e:
            print(f"Critical error: {str(e)}")
            # Do not consolidate partial results - stop execution completely
            raise
        # Only consolidate if all snapshots were successfully generated
        self._consolidate_snapshots(temp_dir, training_set, root, save_filename)

    def _consolidate_snapshots(self, temp_dir, training_set, root, filename):
        """Consolidate temporary snapshot files into a single archive."""
        nl_all, fom_all, mu_all, labels_all = [], [], [], []
        for i in range(len(training_set)):
            try:
                nl = np.load(os.path.join(temp_dir, f"nl_{i}.npy"))
                fom = np.load(os.path.join(temp_dir, f"fom_{i}.npy"))
                mu = np.load(os.path.join(temp_dir, f"mu_{i}.npy"))
                labels = np.load(os.path.join(temp_dir, f"labels_{i}.npy"))
                nl_all.extend(nl)
                fom_all.extend(fom)
                mu_all.extend([mu] * len(nl))
                labels_all.extend(labels)
            except Exception as e:
                print(f"  Snapshot {i} missing or incomplete: {str(e)}")
        if not nl_all:
            print("No snapshots to consolidate")
            return
        np.savez(os.path.join(root, filename), 
                fom_snapshots=np.array(fom_all),
                nonlinear_snapshots=np.array(nl_all),
                mus=np.array(mu_all),
                labels=np.array(labels_all))
        print(f"\n Consolidated {len(nl_all)} snapshots")
        print(f"  - FOM: {np.array(fom_all).shape}")
        print(f"  - Nonlinear: {np.array(nl_all).shape}")
        print(f"  - Parameters: {np.array(mu_all).shape}")
        print(f"NOTE: Each snapshot represents the nonlinear term at a converged solution")

    # ------------------------------------------------------------------
    # Reduced-order projection and online solvers
    # ------------------------------------------------------------------
    def POD_projection(self, solution, N_basis, Z):
        """Project a full-order solution onto the reduced basis in W."""
        Z_N = Z[:N_basis]
        MZ = [self.inner_product * z.vector() for z in Z_N]
        c = np.linalg.solve(
            np.array([[zi.vector().inner(mzj) for mzj in MZ] for zi in Z_N]),
            np.array([solution.vector().inner(mzj) for mzj in MZ]))
        reduced = Function(VectorFunctionSpace(self.mesh, "R", 0, dim=N_basis))
        reduced.vector().set_local(c)
        lifted = Function(self.W)
        v = lifted.vector(); v.zero()
        [v.axpy(c[j], z.vector()) for j, z in enumerate(Z_N)]
        v.apply("insert")
        return reduced, lifted

    def POD_projection_projected(self, solution, N_basis, Z_projected):
        """Project a full-order solution onto the reduced basis in V_CG."""
        Z_N = Z_projected[:N_basis]
        MZ = [self.inner_product_CG * z.vector() for z in Z_N]
        c = np.linalg.solve(
            [[zi.vector().inner(mzj) for mzj in MZ] for zi in Z_N],
            [solution.vector().inner(mzj) for mzj in MZ])
        reduced = Function(VectorFunctionSpace(self.mesh, "R", 0, dim=N_basis))
        reduced.vector().set_local(c)
        lifted = Function(self.V_CG)
        lifted.vector().zero()
        for j, z in enumerate(Z_N):
            lifted.vector().axpy(c[j], z.vector())
        lifted.vector().apply("insert")
        return reduced, lifted

    # def online_PODG_solver(self, mu, N_basis, Z, abs_tol=1E-8, rel_tol=1E-8, max_iter=25):
    #     """Online POD-Galerkin solver """
    #     self.set_mu(mu)
    #     self.setup_kirchhoff_problem()
    #     Z_N = Z[:N_basis]
    #     basis_vectors = [z.vector().copy() for z in Z_N]
    #     w_r = np.zeros(N_basis)
    #     w_h_m = Function(self.W)
    #     original_w = Function(self.W); original_w.assign(self.w_)
    #     converged = False
    #     print(f"{'Iter':<5}{'Residual Norm':<15}{'Rel Change':<15}")
    #     for iter in range(max_iter):
    #         w_h_m.vector()[:] = 0.0
    #         for i, c in enumerate(w_r): w_h_m.vector().axpy(c, basis_vectors[i])
    #         self.w_.assign(w_h_m)
    #         F_vec = assemble(self.F)
    #         F_r = np.array([basis_vectors[j].inner(F_vec) for j in range(N_basis)])
    #         norm_r = np.linalg.norm(F_r)
    #         if iter == 0:
    #             norm0 = max(norm_r, DOLFIN_EPS)
    #             print(f"{iter:<5}{norm_r:.3e}{' ':>10}{'-':<15}")
    #         if norm_r < abs_tol or norm_r/norm0 < rel_tol:
    #             converged = True
    #             print(f"Converged at iter {iter}")
    #             break
    #         J_mat = assemble(self.J)
    #         J_r = np.zeros((N_basis, N_basis))
    #         for i in range(N_basis):
    #             J_z_i = J_mat * basis_vectors[i]
    #             for j in range(N_basis):
    #                 J_r[j, i] = basis_vectors[j].inner(J_z_i)
    #         try:
    #             dw_r = np.linalg.solve(J_r, -F_r)
    #         except np.linalg.LinAlgError:
    #             print("Singular Jacobian")
    #             break
    #         w_r += dw_r
    #         if iter > 0:
    #             rel_change = np.linalg.norm(dw_r)/np.linalg.norm(w_r)
    #             print(f"{iter:<5}{norm_r:.3e}{' ':>10}{rel_change:.3e}")
    #     self.w_.assign(original_w)
    #     return w_r, w_h_m

    def online_PODG_solver(
        self, mu, N_basis, Z, abs_tol=1e-8, rel_tol=1e-8, max_iter=25, *,
        thermal_on=False, coupled_on=False,
        heat_nx=None, heat_ny=None, heat_nz=10, heat_degree=1,
        Nz_quad_T1=8, T1_cg_degree=1,
        coupling_omega=0.5, coupling_tol_w=1e-4, coupling_tol_T1=1e-4,
        coupling_max_iters=20, coupling_verbose=True,
        w0=None, return_info=False,
    ):
        """
        Paper-2 online POD-Galerkin solver.

        Modes:
        thermal_on=False, coupled_on=False : mechanical-only POD-Galerkin.
        thermal_on=True,  coupled_on=False : one-way thermo-mechanical POD-Galerkin.
        thermal_on=True,  coupled_on=True  : coupled thermo-mechanical POD-Galerkin.

        For Paper-2 active MP/TP vectors, decode outside this method:
            mu_mech, mu_th, meta = _p2_decode_active_sample(P2_CONFIG, active_mu)
            solver.set_rom_thermal_parameters(**mu_th)
            solver.online_PODG_solver(mu_mech, ..., thermal_on=True, coupled_on=True)

        Important:
        This intrusive POD-Galerkin implementation is restricted to monolithic cases.
        Contact/interface cases require a dedicated reduced mixed/interface residual.
        """

        # ------------------------------------------------------------------
        # Checks
        # ------------------------------------------------------------------
        if coupled_on and not thermal_on:
            raise ValueError("online_PODG_solver: coupled_on=True requires thermal_on=True.")
        if not hasattr(self, "mesh"):
            raise RuntimeError("online_PODG_solver: call define_domain() before ROM solve.")
        if int(getattr(self, "N_subdomains", 1)) != 1:
            raise RuntimeError(
                "online_PODG_solver is currently enabled only for monolithic cases. "
                "For contact/interface cases, use POD projection and non-intrusive ROMs, "
                "or derive a dedicated coupled contact reduced residual."
            )
        if int(N_basis) < 1:
            raise ValueError("online_PODG_solver: N_basis must be >= 1.")
        if int(N_basis) > len(Z):
            raise ValueError(f"online_PODG_solver: requested N_basis={N_basis}, but len(Z)={len(Z)}.")
        if not (0.0 < float(coupling_omega) <= 1.0):
            raise ValueError("online_PODG_solver: coupling_omega must be in (0, 1].")
        if int(coupling_max_iters) < 1:
            raise ValueError("online_PODG_solver: coupling_max_iters must be >= 1.")

        N_basis = int(N_basis)

        # ------------------------------------------------------------------
        # Local helpers
        # ------------------------------------------------------------------
        def _basis_vectors():
            Z_N = list(Z[:N_basis])
            if len(Z_N) != N_basis:
                raise RuntimeError("online_PODG_solver: invalid POD basis slice.")
            for j, zj in enumerate(Z_N):
                if zj.function_space().dim() != self.W.dim():
                    raise RuntimeError(
                        f"online_PODG_solver: basis[{j}] dimension "
                        f"{zj.function_space().dim()} does not match current W dimension {self.W.dim()}."
                    )
            return Z_N, [z.vector().copy() for z in Z_N]

        def _lift(c, basis_vecs, name="w_podg"):
            w = Function(self.W, name=name)
            w.vector().zero()
            for j, cj in enumerate(np.asarray(c, dtype=float)):
                w.vector().axpy(float(cj), basis_vecs[j])
            w.vector().apply("insert")
            return w

        def _initial_coeffs(basis_vecs):
            if w0 is None:
                return np.zeros(N_basis, dtype=float)
            try:
                w0_fun = (
                    interpolate(Constant(float(w0)), self.W)
                    if isinstance(w0, (int, float, np.floating))
                    else self._as_same_space_function(w0, self.W)
                )
                rb0, _ = self.POD_projection(w0_fun, N_basis, Z[:N_basis])
                return rb0.vector().get_local().copy()
            except Exception as exc:
                print(f"[online_PODG_solver] w0 projection failed; using zero coefficients. Reason: {exc}")
                return np.zeros(N_basis, dtype=float)

        def _reduced_newton(c_start, label=""):
            _, basis_vecs = _basis_vectors()
            c = np.asarray(c_start, dtype=float).copy()
            if c.size != N_basis:
                raise ValueError(f"Reduced coefficient size {c.size} does not match N_basis={N_basis}.")

            original_w = Function(self.W)
            original_w.assign(self.w_)
            residual_history, converged, norm0 = [], False, None

            if label:
                print(label)
            print(f"{'Iter':<5}{'Residual Norm':<15}{'Rel Change':<15}")

            try:
                for it in range(int(max_iter)):
                    self.w_.assign(_lift(c, basis_vecs, name="w_podg_reduced"))

                    F_vec = assemble(self.F)
                    for bc in getattr(self, "bcs", []):
                        try:
                            bc.apply(F_vec)
                        except Exception:
                            pass

                    F_r = np.array([basis_vecs[j].inner(F_vec) for j in range(N_basis)], dtype=float)
                    norm_r = float(np.linalg.norm(F_r))
                    residual_history.append(norm_r)

                    if it == 0:
                        norm0 = max(norm_r, float(DOLFIN_EPS))
                        print(f"{it:<5}{norm_r:.3e}{' ':>10}{'-':<15}")

                    if norm_r < float(abs_tol) or norm_r / norm0 < float(rel_tol):
                        converged = True
                        print(f"Converged at reduced Newton iteration {it}")
                        break

                    J_mat = assemble(self.J)
                    for bc in getattr(self, "bcs", []):
                        try:
                            bc.apply(J_mat)
                        except Exception:
                            pass

                    J_r = np.zeros((N_basis, N_basis), dtype=float)
                    for i in range(N_basis):
                        Jzi = J_mat * basis_vecs[i]
                        for j in range(N_basis):
                            J_r[j, i] = basis_vecs[j].inner(Jzi)

                    try:
                        dc = np.linalg.solve(J_r, -F_r)
                    except np.linalg.LinAlgError as exc:
                        raise RuntimeError("online_PODG_solver: singular reduced Jacobian.") from exc

                    c += dc
                    rel_change = float(np.linalg.norm(dc) / max(np.linalg.norm(c), float(DOLFIN_EPS)))
                    if it > 0:
                        print(f"{it:<5}{norm_r:.3e}{' ':>10}{rel_change:.3e}")

                    if rel_change < float(rel_tol):
                        converged = True
                        print(f"Reduced step converged at iteration {it}")
                        break

            finally:
                self.w_.assign(original_w)

            w_rom = _lift(c, basis_vecs, name="w_podg_reduced_final")
            info = dict(
                reduced_newton_converged=bool(converged),
                reduced_newton_iterations=int(len(residual_history)),
                reduced_newton_residual_history=residual_history,
                reduced_newton_final_residual=float(residual_history[-1]) if residual_history else np.nan,
            )
            return c, w_rom, info

        def _heat_to_T1(w_current, store=True):
            return self.build_plate_thermal_driver_from_rom_parameters(
                nx=None if heat_nx is None else int(heat_nx),
                ny=None if heat_ny is None else int(heat_ny),
                nz=int(heat_nz),
                degree=int(heat_degree),
                Nz_quad=int(Nz_quad_T1),
                T1_cg_degree=int(T1_cg_degree),
                store=bool(store),
                w_for_heat=w_current,
            )

        def _store_common(mode, c, w_rom, info, heat=None, T1=None):
            if heat is not None:
                self.podg_heat = heat
            if T1 is not None:
                self.podg_T1 = self._copy_function(T1)
            self.podg_w_global = Function(self.W)
            self.podg_w_global.assign(w_rom)
            self.podg_coefficients = np.asarray(c, dtype=float)
            self.podg_info = dict(mode=mode, thermal_on=bool(thermal_on), coupled_on=bool(coupled_on), **info)
            return (self.podg_coefficients, w_rom, self.podg_info) if return_info else (self.podg_coefficients, w_rom)

        # ------------------------------------------------------------------
        # Branch 1: mechanical-only POD-Galerkin
        # ------------------------------------------------------------------
        if not thermal_on:
            print("\n" + "=" * 80)
            print("ONLINE POD-GALERKIN: MECHANICAL-ONLY MODE")
            print("=" * 80)

            self.set_mu(mu)
            self.update_thermal_parameters(enable=False, alpha1=self.alpha1_rom, alpha2=self.alpha2_rom, T1_expr=None)
            self.setup_kirchhoff_problem()
            _, basis_vecs = _basis_vectors()
            c, w_rom, info = _reduced_newton(_initial_coeffs(basis_vecs))
            return _store_common("mechanical", c, w_rom, info)

        # ------------------------------------------------------------------
        # Branch 2: one-way thermo-mechanical POD-Galerkin
        # ------------------------------------------------------------------
        if thermal_on and not coupled_on:
            print("\n" + "=" * 80)
            print("ONLINE POD-GALERKIN: ONE-WAY THERMO-MECHANICAL MODE")
            print("=" * 80)

            self.set_mu(mu)
            self.update_thermal_parameters(enable=False, alpha1=self.alpha1_rom, alpha2=self.alpha2_rom, T1_expr=None)
            self.setup_kirchhoff_problem()
            _, basis_vecs = _basis_vectors()

            c0 = _initial_coeffs(basis_vecs)
            w_for_heat = _lift(c0, basis_vecs, name="w_podg_oneway_heat_driver")
            heat, T1 = _heat_to_T1(w_for_heat, store=True)

            self.update_thermal_parameters(enable=True, alpha1=self.alpha1_rom, alpha2=self.alpha2_rom, T1_expr=T1)
            self.set_mu(mu)
            self.setup_kirchhoff_problem()

            c, w_rom, info = _reduced_newton(c0)
            info.update(dict(
                heat_nx=None if heat_nx is None else int(heat_nx),
                heat_ny=None if heat_ny is None else int(heat_ny),
                heat_nz=int(heat_nz),
                heat_degree=int(heat_degree),
                Nz_quad_T1=int(Nz_quad_T1),
                T1_cg_degree=int(T1_cg_degree),
            ))
            return _store_common("one_way_thermomechanical", c, w_rom, info, heat=heat, T1=T1)

        # ------------------------------------------------------------------
        # Branch 3: coupled thermo-mechanical POD-Galerkin
        # ------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("ONLINE POD-GALERKIN: COUPLED THERMO-MECHANICAL MODE")
        print("=" * 80)
        print("Reduced model: POD-Galerkin plate solve + full 3D heat solve in the coupling loop.")
        print("Allowed case : monolithic only.")
        print("=" * 80)

        self.set_mu(mu)
        self.update_thermal_parameters(enable=False, alpha1=self.alpha1_rom, alpha2=self.alpha2_rom, T1_expr=None)
        self.setup_kirchhoff_problem()
        _, basis_vecs = _basis_vectors()

        c_used = _initial_coeffs(basis_vecs)
        T1_prev, history = None, []
        last_heat = last_T1 = last_raw_coeffs = last_raw_w = None
        converged, stopped_by_max_iters, termination_reason = False, False, "max_iters_reached"

        for it in range(1, int(coupling_max_iters) + 1):
            w_used = _lift(c_used, basis_vecs, name=f"w_podg_coupling_used_{it}")
            heat, T1_new = _heat_to_T1(w_used, store=True)

            self.update_thermal_parameters(enable=True, alpha1=self.alpha1_rom, alpha2=self.alpha2_rom, T1_expr=T1_new)
            self.set_mu(mu)
            self.setup_kirchhoff_problem()
            _, basis_vecs = _basis_vectors()

            c_raw, w_raw, newton_info = _reduced_newton(
                c_used,
                label=f"\n[coupled PODG outer {it:02d}] reduced mechanical solve",
            )

            c_relaxed = (1.0 - float(coupling_omega)) * c_used + float(coupling_omega) * c_raw
            w_relaxed = _lift(c_relaxed, basis_vecs, name=f"w_podg_coupling_relaxed_{it}")

            err_w = self._relative_l2_error_same_space(w_relaxed, w_used)
            err_T1 = np.nan if T1_prev is None else self._relative_l2_error_same_space(T1_new, T1_prev)

            w_arr = w_relaxed.vector().get_local()
            heat_dofs = int(heat["Vt"].dim()) if heat is not None and "Vt" in heat else -1
            T1_dofs = int(T1_new.function_space().dim()) if T1_new is not None else -1

            row = dict(
                iter=int(it),
                err_w=float(err_w),
                err_T1=float(err_T1),
                stopping_norm="L2_Omega",
                w_min=float(w_arr.min()),
                w_max=float(w_arr.max()),
                heat_nx=-1 if heat_nx is None else int(heat_nx),
                heat_ny=-1 if heat_ny is None else int(heat_ny),
                heat_nz=int(heat_nz),
                heat_degree=int(heat_degree),
                heat_dofs=int(heat_dofs),
                T1_dofs=int(T1_dofs),
                reduced_newton_converged=bool(newton_info["reduced_newton_converged"]),
                reduced_newton_iterations=int(newton_info["reduced_newton_iterations"]),
                reduced_newton_final_residual=float(newton_info["reduced_newton_final_residual"]),
            )
            history.append(row)

            if coupling_verbose:
                print(
                    f"[PODG coupling {it:02d}] "
                    f"err_w={err_w:.3e}, err_T1={err_T1:.3e}, "
                    f"w_min={row['w_min']:.6e}, w_max={row['w_max']:.6e}, "
                    f"heat DoFs={heat_dofs}, T1 DoFs={T1_dofs}, "
                    f"Newton iters={row['reduced_newton_iterations']}"
                )

            last_heat = heat
            last_T1 = self._copy_function(T1_new)
            last_raw_coeffs = np.asarray(c_raw, dtype=float).copy()
            last_raw_w = Function(self.W)
            last_raw_w.assign(w_raw)

            c_used = np.asarray(c_relaxed, dtype=float).copy()

            if it > 1 and err_w < float(coupling_tol_w) and err_T1 < float(coupling_tol_T1):
                converged, termination_reason = True, "converged"
                break

            T1_prev = self._copy_function(T1_new)

        stopped_by_max_iters = not converged
        w_final = _lift(c_used, basis_vecs, name="w_podg_coupled_final")

        self.update_thermal_parameters(enable=True, alpha1=self.alpha1_rom, alpha2=self.alpha2_rom, T1_expr=last_T1)
        self.podg_heat = last_heat
        self.podg_T1 = self._copy_function(last_T1)
        self.podg_w_global = Function(self.W)
        self.podg_w_global.assign(w_final)
        self.podg_w_raw_final = last_raw_w
        self.podg_coefficients = np.asarray(c_used, dtype=float)
        self.podg_raw_coefficients = last_raw_coeffs
        self.podg_history = history
        self.podg_coupled_converged = bool(converged)
        self.podg_coupled_stopped_by_max_iters = bool(stopped_by_max_iters)
        self.podg_termination_reason = termination_reason

        info = dict(
            mode="coupled_thermomechanical",
            thermal_on=True,
            coupled_on=True,
            converged=bool(converged),
            stopped_by_max_iters=bool(stopped_by_max_iters),
            termination_reason=termination_reason,
            iters=int(len(history)),
            err_w=float(history[-1]["err_w"]) if history else np.nan,
            err_T1=float(history[-1]["err_T1"]) if history else np.nan,
            heat_nx=None if heat_nx is None else int(heat_nx),
            heat_y=None if heat_ny is None else int(heat_ny),
            heat_ny=None if heat_ny is None else int(heat_ny),
            heat_nz=int(heat_nz),
            heat_degree=int(heat_degree),
            Nz_quad_T1=int(Nz_quad_T1),
            T1_cg_degree=int(T1_cg_degree),
            history=history,
        )
        self.podg_info = info

        print("\n" + "-" * 80)
        print("Coupled thermo-mechanical POD-Galerkin complete")
        print(f"  converged          : {info['converged']}")
        print(f"  termination        : {info['termination_reason']}")
        print(f"  coupling iterations: {info['iters']}")
        print(f"  final err_w        : {info['err_w']:.3e}")
        print(f"  final err_T1       : {info['err_T1']:.3e}")
        print("-" * 80)

        return (self.podg_coefficients, w_final, self.podg_info) if return_info else (self.podg_coefficients, w_final)

    def online_LSPG_solver(self, mu, N_basis, Z, abs_tol=1E-5, rel_tol=1E-5, max_iter=25):
        """Online LSPG reduced solver."""
        """Custom Gauss-Newton LSPG solver optimized for contact problems"""
        self.set_mu(mu); self.setup_kirchhoff_problem()
        Z_N = Z[:N_basis]; basis_vectors = [z.vector().copy() for z in Z_N]
        w_r = np.zeros(N_basis); w_h_m = Function(self.W)
        original_w = Function(self.W); original_w.assign(self.w_)
        print(f"Custom LSPG solver for μ={mu}")
        print(f"{'Iter':<5}{'||R||':<12}{'||Δc||':<12}{'λ':<8}")
        for iteration in range(max_iter):
            w_h_m.vector()[:] = 0.0
            for i, c in enumerate(w_r): w_h_m.vector().axpy(c, basis_vectors[i])
            self.w_.assign(w_h_m)
            for bc in self.bcs: bc.apply(self.w_.vector())
            R = assemble(self.F)
            for bc in self.bcs: bc.apply(R)
            R_vec = R.get_local()
            J_mat = assemble(self.J)
            for bc in self.bcs: bc.apply(J_mat)
            R_reduced = np.array([basis_vectors[i].inner(R) for i in range(N_basis)])
            J_reduced = np.zeros((N_basis, N_basis))
            for i in range(N_basis):
                J_z_i = J_mat * basis_vectors[i]
                for j in range(N_basis): J_reduced[j, i] = basis_vectors[j].inner(J_z_i)
            residual_norm = np.linalg.norm(R_reduced)
            if iteration == 0: initial_residual = residual_norm
            relative_residual = residual_norm / (initial_residual + 1e-12)
            if residual_norm < abs_tol or relative_residual < rel_tol:
                print(f"Converged at iteration {iteration}"); break
            try:
                JTR = -R_reduced; JTJ = J_reduced
                regularization = 1e-8 * np.trace(JTJ) / N_basis
                JTJ += regularization * np.eye(N_basis)
                delta_c = np.linalg.solve(JTJ, JTR)
            except np.linalg.LinAlgError:
                print(f"Singular system at iteration {iteration}"); break
            alpha = 1.0
            for _ in range(5):
                w_r_new = w_r + alpha * delta_c
                w_h_m.vector()[:] = 0.0
                for i, c in enumerate(w_r_new): w_h_m.vector().axpy(c, basis_vectors[i])
                self.w_.assign(w_h_m)
                for bc in self.bcs: bc.apply(self.w_.vector())
                R_new = assemble(self.F)
                for bc in self.bcs: bc.apply(R_new)
                R_new_reduced = np.array([basis_vectors[i].inner(R_new) for i in range(N_basis)])
                new_residual_norm = np.linalg.norm(R_new_reduced)
                if new_residual_norm < residual_norm * 1.1: break
                alpha *= 0.5
            w_r = w_r + alpha * delta_c
            delta_norm = np.linalg.norm(alpha * delta_c)
            print(f"{iteration:<5}{residual_norm:.3e}{' ':<2}{delta_norm:.3e}{' ':<2}{alpha:.2f}")
            if delta_norm < rel_tol * (np.linalg.norm(w_r) + rel_tol):
                print(f"Step size converged at iteration {iteration}"); break
        w_h_m.vector()[:] = 0.0
        for i, c in enumerate(w_r): w_h_m.vector().axpy(c, basis_vectors[i])
        self.w_.assign(w_h_m)
        for bc in self.bcs: bc.apply(self.w_.vector())
        w_h_m.assign(self.w_); self.w_.assign(original_w)
        return w_r, w_h_m

    # ------------------------------------------------------------------
    # Reference and verification models
    # ------------------------------------------------------------------
    def compute_and_visualize_exact_solution(
            self,
            thermal: bool = False,
            mu=None,
            num_terms: int = 29,
            alpha1: float | None = None,
            alpha2: float | None = None,
            T1_const: float = None,
            dT_through: float = None,
            T1: Callable = None,
            foundation_branch: str = "auto",       # "auto", "compression", "uplift", "none"
            ks_eff_override: float | None = None,  # direct override for k_s^eff, if needed
            branch_tol: float = 1e-10,
            branch_check_grid: int = 81,
            exact_degree: int = 8,
            thermal_projection_grid: int = 401,
        ):
        """
        Compute the Navier reference solution for the single-panel SSSS aligned-orthotropic
        Kirchhoff--Love plate on a branch-frozen linear Winkler foundation.
    
        The foundation stiffness entering the Navier denominator is
    
            k_s^eff = beta * k_s,
    
        where beta = 1 for the compression/contact branch and beta = epsilon for the
        uplift-regularized branch. This is an exact Navier solution of the selected
        linear branch-frozen problem, not of a mixed active-set unilateral problem.
    
        Parameters
        ----------
        thermal : bool
            If True, include thermal bending from DeltaT(x,y,z)=T0(x,y)+z*T1(x,y).
            Only T1 enters the KL thermal bending moments.
    
        foundation_branch : {"auto", "compression", "uplift", "none"}
            - "auto": solve/check both global branch-frozen references and accept the
              one whose reconstructed displacement has the corresponding sign.
            - "compression": force k_s^eff = k_s.
            - "uplift": force k_s^eff = epsilon*k_s.
            - "none": force k_s^eff = 0.
    
        ks_eff_override : float or None
            If provided, use this value directly as k_s^eff.
    
        Stores
        ------
        self.Exact_solution             if thermal=False
        self.Exact_solution_thermal     if thermal=True
    
        Also stores branch metadata:
        self.Exact_solution_ks_eff
        self.Exact_solution_foundation_branch
        self.Exact_solution_branch_range
        self.Exact_solution_q0_eff
        """
        # ------------------------------------------------------------------
        # Preconditions
        # ------------------------------------------------------------------
        if mu is not None:
            self.set_mu(mu)
    
        if self.load_type != "uniform":
            return print("[Exact] Only implemented for uniform load.")
    
        if not hasattr(self, "mesh"):
            return print("[Exact] Call define_domain() first.")
    
        if getattr(self, "N_subdomains", 1) > 1:
            return print("[Exact] Only valid for a single-subdomain mesh.")
    
        # Effective transverse load consistent with the plate/FOM:
        # q_eff = f + q_weight = f - rho*g*t
        rho_val, g_val, t = float(self.rho), float(self.g), float(self.t)
        q0_eff = float(self.f) - rho_val * g_val * t
    
        ks = float(self.ks)
        eta = float(self.foundation_tension_factor)
    
        def _ks_from_branch(branch_name: str):
            branch_name = str(branch_name).lower()
            if branch_name in ("compression", "contact"):
                return ks, "compression"
            if branch_name in ("uplift", "gap", "tension"):
                return eta * ks, "uplift"
            if branch_name in ("none", "zero", "no_foundation"):
                return 0.0, "none"
            raise ValueError(
                "foundation_branch must be 'auto', 'compression', 'uplift', or 'none', "
                "or provide ks_eff_override."
            )
    
        # ------------------------------------------------------------------
        # Free-edge constant reference: mechanical only
        # ------------------------------------------------------------------
        if self.bc_type == "free_edge":
            if thermal:
                return print("[Exact] Thermal exact solution currently only implemented for uniform load + SSSS BCs.")
    
            def _free_edge_w0(ks_trial):
                if abs(float(ks_trial)) < 1e-14:
                    return None
                return q0_eff / float(ks_trial)
    
            if ks_eff_override is not None:
                ks_eff = float(ks_eff_override)
                selected_branch = "override"
                w0 = _free_edge_w0(ks_eff)
                if w0 is None:
                    return print("[Exact] Skipped: k_s^eff ≈ 0 for free-edge case.")
            else:
                branch_req = str(foundation_branch).lower()
    
                if branch_req == "auto":
                    candidates = []
    
                    ks_c, _ = _ks_from_branch("compression")
                    w0_c = _free_edge_w0(ks_c)
                    if w0_c is not None and w0_c <= branch_tol:
                        candidates.append(("compression", ks_c, w0_c))
    
                    ks_u, _ = _ks_from_branch("uplift")
                    w0_u = _free_edge_w0(ks_u)
                    if w0_u is not None and w0_u >= -branch_tol:
                        candidates.append(("uplift", ks_u, w0_u))
    
                    if len(candidates) == 1:
                        selected_branch, ks_eff, w0 = candidates[0]
                    elif len(candidates) > 1:
                        selected_branch, ks_eff, w0 = candidates[0]
                        print(
                            "[Exact] Both free-edge branches are sign-consistent within tolerance; "
                            "using compression by convention."
                        )
                    else:
                        return print(
                            "[Exact] No globally self-consistent free-edge branch found. "
                            "Use the nonlinear FOM, or request a branch-frozen reference explicitly."
                        )
                else:
                    ks_eff, selected_branch = _ks_from_branch(branch_req)
                    w0 = _free_edge_w0(ks_eff)
                    if w0 is None:
                        return print("[Exact] Skipped: k_s^eff ≈ 0 for free-edge case.")
    
                    if selected_branch == "compression" and w0 > branch_tol:
                        print(
                            "[Exact] Warning: compression branch was requested, but the free-edge "
                            f"constant field is positive: w={w0:.3e}. This is only a linear reference."
                        )
                    if selected_branch == "uplift" and w0 < -branch_tol:
                        print(
                            "[Exact] Warning: uplift branch was requested, but the free-edge "
                            f"constant field is negative: w={w0:.3e}. This is only a linear reference."
                        )
    
            V_LG = FunctionSpace(self.mesh, FiniteElement("Lagrange", triangle, 2))
            field = interpolate(Constant(w0), V_LG)
            field.rename("Exact Displacement", "")
            self.Exact_solution = field
    
            self.Exact_solution_ks_eff = ks_eff
            self.Exact_solution_foundation_branch = selected_branch
            self.Exact_solution_branch_range = (float(w0), float(w0))
            self.Exact_solution_q0_eff = q0_eff
    
            print(
                f"Exact mechanical solution computed "
                f"(free-edge branch-frozen Winkler: w=q_eff/k_s^eff={w0:.3e} m, "
                f"k_s^eff={ks_eff:.3e}, branch={selected_branch})."
            )
            return field
    
        if self.bc_type != "simply_supported":
            return print(
                "[Exact] Only implemented for uniform load with simply supported BCs, "
                "or free-edge in the purely mechanical case."
            )
    
        # ------------------------------------------------------------------
        # Geometry and rigidities
        # ------------------------------------------------------------------
        a, b = float(self.length), float(self.width)
        Dx, Dy, Dxy, Ds = map(float, (self.D_x, self.D_y, self.D_xy, self.D_s))
        pi = np.pi
    
        # ------------------------------------------------------------------
        # Resolve thermal input before choosing modes
        # ------------------------------------------------------------------
        if thermal:
            if alpha1 is None:
                alpha1_eff = float(
                    self._alpha1_in
                    if getattr(self, "_alpha1_in", None) is not None
                    else self._alpha_iso
                )
            else:
                alpha1_eff = float(alpha1)
    
            if alpha2 is None:
                alpha2_eff = float(
                    self._alpha2_in
                    if getattr(self, "_alpha2_in", None) is not None
                    else self._alpha_iso
                )
            else:
                alpha2_eff = float(alpha2)
    
            if (
                (T1 is None)
                and (T1_const is None)
                and (dT_through is None)
                and getattr(self, "_thermal_enabled", False)
            ):
                self._realize_thermal_fields()
                T1obj = getattr(self, "T1_plate", None)
                if T1obj is not None:
                    try:
                        T1_const = float(T1obj)
                    except Exception:
                        def _T1_callable(x, y):
                            return float(T1obj(Point(x, y)))
                        T1 = _T1_callable
    
            # If user passed through-thickness temperature difference, convert to T1_const.
            if (dT_through is not None) and (T1_const is None):
                T1_const = float(dT_through) / t
        else:
            alpha1_eff = 0.0
            alpha2_eff = 0.0
    
        # If T1 is a general callable, even modes may be present.  Use all modes up to
        # the same maximum index that the odd-only truncation would have reached.
        use_all_modes = bool(thermal and callable(T1) and T1_const is None)
    
        if use_all_modes:
            m = np.arange(1, 2 * int(num_terms), 1)
            n = np.arange(1, 2 * int(num_terms), 1)
        else:
            m = np.arange(1, 2 * int(num_terms), 2)
            n = np.arange(1, 2 * int(num_terms), 2)
    
        M, N = np.meshgrid(m, n, indexing="ij")
    
        # ------------------------------------------------------------------
        # Modal denominator excluding foundation, and mechanical load numerator
        # ------------------------------------------------------------------
        bending_part = (pi**4) * (
            Dx * (M**4) / a**4
            + 2.0 * (Dxy + 2.0 * Ds) * (M**2 * N**2) / (a**2 * b**2)
            + Dy * (N**4) / b**4
        )
    
        odd_mask = ((M % 2) == 1) & ((N % 2) == 1)
    
        # Uniform-load sine coefficients:
        # q_mn = 16*q0/(pi^2*m*n) for odd-odd modes, zero otherwise.
        numer = np.zeros_like(M, dtype=float)
        numer[odd_mask] = (16.0 * q0_eff / pi**2) * (1.0 / (M[odd_mask] * N[odd_mask]))
    
        # ------------------------------------------------------------------
        # Thermal modal contribution
        # ------------------------------------------------------------------
        if thermal:
            if T1_const is not None:
                T1_mn = np.zeros_like(M, dtype=float)
                T1_mn[odd_mask] = (
                    16.0 * float(T1_const) / pi**2
                ) * (1.0 / (M[odd_mask] * N[odd_mask]))
    
            elif callable(T1):
                Nx = Ny = int(thermal_projection_grid)
                X = np.linspace(0.0, a, Nx)
                Y = np.linspace(0.0, b, Ny)
                XX, YY = np.meshgrid(X, Y, indexing="ij")
    
                T1_vals = np.vectorize(T1, otypes=[float])(XX, YY)
    
                sx = np.sin(pi * np.outer(X / a, m))  # shape: (Nx, Nm)
                sy = np.sin(pi * np.outer(Y / b, n))  # shape: (Ny, Nn)
    
                # First integrate in x:
                # T1_vals[:, :, None] * sx[:, None, :] has shape (Nx, Ny, Nm).
                # Integrating over x gives shape (Ny, Nm).
                Tx = np.trapz(T1_vals[:, :, None] * sx[:, None, :], X, axis=0)
    
                # Then integrate in y:
                # Result has shape (Nm, Nn), matching M,N.  No transpose is needed.
                T1_mn = (4.0 / (a * b)) * np.trapz(
                    Tx[:, :, None] * sy[:, None, :], Y, axis=0
                )
    
            else:
                T1_mn = 0.0
    
            if not (np.isscalar(T1_mn) and T1_mn == 0.0):
                M1_mn = (Dx * alpha1_eff + Dxy * alpha2_eff) * T1_mn
                M2_mn = (Dxy * alpha1_eff + Dy * alpha2_eff) * T1_mn
    
                numer += ((M * pi / a) ** 2) * M1_mn + ((N * pi / b) ** 2) * M2_mn
    
        # ------------------------------------------------------------------
        # Branch-frozen foundation selection
        # ------------------------------------------------------------------
        def _field_range_for_ks(ks_trial):
            coeff_trial = numer / (bending_part + float(ks_trial))
    
            ng = max(11, int(branch_check_grid))
            xs = np.linspace(0.0, a, ng)
            ys = np.linspace(0.0, b, ng)
    
            sxg = np.sin(pi * np.outer(xs / a, m))  # (ng, Nm)
            syg = np.sin(pi * np.outer(ys / b, n))  # (ng, Nn)
    
            Wg = sxg @ coeff_trial @ syg.T
            return float(Wg.min()), float(Wg.max()), coeff_trial
    
        if ks_eff_override is not None:
            ks_eff = float(ks_eff_override)
            selected_branch = "override"
            wmin_chk, wmax_chk, coeff = _field_range_for_ks(ks_eff)
    
        else:
            branch_req = str(foundation_branch).lower()
    
            if branch_req == "auto":
                candidates = []
    
                ks_comp, _ = _ks_from_branch("compression")
                wmin_c, wmax_c, coeff_c = _field_range_for_ks(ks_comp)
                if wmax_c <= branch_tol:
                    candidates.append(("compression", ks_comp, wmin_c, wmax_c, coeff_c))
    
                ks_uplift, _ = _ks_from_branch("uplift")
                wmin_u, wmax_u, coeff_u = _field_range_for_ks(ks_uplift)
                if wmin_u >= -branch_tol:
                    candidates.append(("uplift", ks_uplift, wmin_u, wmax_u, coeff_u))
    
                if len(candidates) == 1:
                    selected_branch, ks_eff, wmin_chk, wmax_chk, coeff = candidates[0]
    
                elif len(candidates) > 1:
                    # Usually occurs only for an approximately zero response.
                    selected_branch, ks_eff, wmin_chk, wmax_chk, coeff = candidates[0]
                    print(
                        "[Exact] Both branch-frozen references are sign-consistent within tolerance; "
                        "using compression branch by convention."
                    )
    
                else:
                    return print(
                        "[Exact] No single global branch-frozen stiffness is self-consistent. "
                        "The reconstructed Navier field changes sign or violates both global branch "
                        "assumptions. Therefore no classical constant-stiffness Navier exact solution "
                        "exists for the full unilateral problem. Use the nonlinear FOM, or request a "
                        "linear reference explicitly with foundation_branch='compression' or "
                        "foundation_branch='uplift'."
                    )
    
            else:
                ks_eff, selected_branch = _ks_from_branch(branch_req)
                wmin_chk, wmax_chk, coeff = _field_range_for_ks(ks_eff)
    
                if selected_branch == "compression" and wmax_chk > branch_tol:
                    print(
                        "[Exact] Warning: compression branch was requested, but the branch-frozen "
                        f"Navier field has positive values up to {wmax_chk:.3e}. "
                        "This is a linear branch-frozen reference, not a self-consistent unilateral solution."
                    )
    
                if selected_branch == "uplift" and wmin_chk < -branch_tol:
                    print(
                        "[Exact] Warning: uplift branch was requested, but the branch-frozen "
                        f"Navier field has negative values down to {wmin_chk:.3e}. "
                        "This is a linear branch-frozen reference, not a self-consistent unilateral solution."
                    )
    
        denom = bending_part + ks_eff
        coeff = numer / denom
    
        # ------------------------------------------------------------------
        # Reconstruct and interpolate
        # ------------------------------------------------------------------
        def w_exact(x, y):
            return np.sum(coeff * np.sin(M * pi * x / a) * np.sin(N * pi * y / b))
    
        class _Exact(UserExpression):
            def eval(_, values, xx):
                values[0] = w_exact(xx[0], xx[1])
    
            def value_shape(_):
                return ()
    
        exact_degree_eff = max(2, int(exact_degree))
        V_LG = FunctionSpace(self.mesh, FiniteElement("Lagrange", triangle, exact_degree_eff))
        field = interpolate(_Exact(degree=exact_degree_eff + 2), V_LG)
    
        # Store metadata for traceability/reporting.
        self.Exact_solution_ks_eff = float(ks_eff)
        self.Exact_solution_foundation_branch = selected_branch
        self.Exact_solution_branch_range = (float(wmin_chk), float(wmax_chk))
        self.Exact_solution_q0_eff = float(q0_eff)
    
        if thermal:
            field.rename("Exact (thermal) Displacement", "")
            self.Exact_solution_thermal = field
            print(
                "Exact thermal Navier reference computed "
                f"(branch-frozen k_s^eff={ks_eff:.6e}, branch={selected_branch}, "
                f"w_min={wmin_chk:.3e}, w_max={wmax_chk:.3e})."
            )
        else:
            field.rename("Exact Displacement", "")
            self.Exact_solution = field
            print(
                "Exact mechanical Navier reference computed "
                f"(branch-frozen k_s^eff={ks_eff:.6e}, branch={selected_branch}, "
                f"w_min={wmin_chk:.3e}, w_max={wmax_chk:.3e})."
            )
    
        return field

    def solve_solid_3d_model(
        self,
        thermal: bool = False,
        thermal_solve: bool = False,
        use_heat_result: bool = False,
        mu=None,
    
        # 3D solid mechanical discretization
        nx=None,
        ny=None,
        nz=12,
        degree=2,
    
        # Optional heat discretization if this function is asked to run heat_solve itself
        heat_nx=None,
        heat_ny=None,
        heat_nz=None,
        heat_degree=1,
    
        # False = build independent/refined solid mesh and interpolate DeltaT onto it.
        # True  = reuse the stored 3D heat mesh as the solid mechanics mesh.
        reuse_heat_mesh_for_solid=False,
    
        # Thermal expansion
        alpha_iso=1.2e-5,
        alpha1=None,
        alpha2=None,
        alpha3=None,
    
        # Heat/environment inputs
        kx: float | None = None,
        ky: float | None = None,
        kz: float | None = None,
        T_amb=None,
        h_con=None,
        eps_r: float | None = None,
        q_s=None,
        T_sub=None,
        h_c_cont=None,
        h_c_gap: float | None = None,
        eta_c: float | None = None,
        w_for_heat=None,
        source=None,
        T_ref: float | None = None,
    
        include_mech=True,
        plot_results=False,
    
        # legacy args: explicitly rejected
        T_top=None,
        T_bottom=None,
        q_top=None,
        q_bottom=None,
        robin_h_top=None,
        robin_Tinf_top=None,
        robin_h_bottom=None,
        robin_Tinf_bottom=None,
    ):
        """
        Solve the 3D solid elasticity / thermoelasticity reference model.
    
        Important modelling distinction
        -------------------------------
        For the 3D solid solve, the thermal load is NOT reduced to the plate driver
        T1(x,y).  T1(x,y) is only the Kirchhoff--Love plate reduction of the 3D
        temperature field.  The solid model uses the full 3D temperature-change field
    
            DeltaT(x,y,z) = T(x,y,z) - T_ref
    
        directly in the 3D thermal strain tensor.
    
        Notes
        -----
        1. The solid mechanics mesh may be independent of the heat mesh.  If the two
           meshes differ, the stored full 3D DeltaT field is interpolated onto the solid
           mechanics mesh.
    
        2. The thickness coordinate is centered:
               z in [-t/2, +t/2],
           matching heat_solve() and make_T1_from_DeltaT().
    
        3. For isotropic 3D thermoelasticity, alpha3 defaults to alpha1.  Setting
           alpha3=0 would create an artificial constrained thermal state.
    
        4. This is a 3D elasticity reference model, not a Kirchhoff--Love exact solution.
           For thin plates it should approach the KL response under sufficient 3D
           resolution, but low-order tetrahedral solid meshes can be too stiff in bending.
        """
        # ------------------------------------------------------------------
        # Reject obsolete thermal boundary-condition interface
        # ------------------------------------------------------------------
        if any(v is not None for v in (
            T_top, T_bottom, q_top, q_bottom,
            robin_h_top, robin_Tinf_top, robin_h_bottom, robin_Tinf_bottom
        )):
            raise ValueError(
                "solve_solid_3d_model: legacy thermal BC arguments are no longer supported. "
                "Use T_amb, h_con, eps_r, q_s, T_sub, h_c_cont, h_c_gap, eta_c, and w_for_heat."
            )
    
        if mu is not None:
            self.set_mu(mu)
    
        if not hasattr(self, "mesh"):
            raise RuntimeError("solve_solid_3d_model: call define_domain() before solving the solid model.")
    
        if getattr(self, "N_subdomains", 1) > 1 or self.reinforcement_model != 0:
            print("[Solid Solver] Skipped: single-subdomain, non-reinforced cases only.")
            return None
    
        # ------------------------------------------------------------------
        # Back-calculate isotropic 3D material from plate rigidities.
        # This solid comparison is meaningful only for an isotropic-equivalent plate.
        # ------------------------------------------------------------------
        Dx_val = float(self.D_x)
        Dy_val = float(self.D_y)
        Dxy_val = float(self.D_xy)
        Ds_val = float(self.D_s)
        t = float(self.t)
    
        if not np.isclose(Dx_val, Dy_val, rtol=1e-8, atol=1e-12):
            print(f"[Solid Solver] Skipped: isotropic 3D solid requires Dx≈Dy, got Dx={Dx_val:.3e}, Dy={Dy_val:.3e}.")
            return None
    
        D_eff = 0.5 * (Dx_val + Dy_val)
        if abs(D_eff) < 1e-14:
            print("[Solid Solver] Skipped: D_eff≈0.")
            return None
    
        nu = Dxy_val / D_eff
        if not (-0.95 < nu < 0.49):
            raise ValueError(
                f"solve_solid_3d_model: recovered Poisson ratio nu={nu:.6g} is outside "
                "the stable 3D isotropic range (-0.95, 0.49). Check Dxy/D."
            )
    
        Ds_iso = D_eff * (1.0 - nu) / 2.0
        if not np.isclose(Ds_val, Ds_iso, rtol=1e-2, atol=1e-12):
            print(
                f"[Solid Solver] Warning: Ds={Ds_val:.3e} differs from isotropic value "
                f"Ds_iso={Ds_iso:.3e}. Proceeding, but the solid is not an exact "
                "isotropic-equivalent match to the plate rigidities."
            )
    
        E = D_eff * 12.0 * (1.0 - nu**2) / (t**3)
    
        if E <= 0.0:
            raise ValueError(f"solve_solid_3d_model: recovered E={E:.6e} is not positive.")
    
        print(f"[Solid Solver] Using isotropic-equivalent material: E={E:.3e} Pa, nu={nu:.3f}")
    
        # ------------------------------------------------------------------
        # Resolve discretization
        # ------------------------------------------------------------------
        Lx = float(self.length)
        Ly = float(self.width)
    
        if nx is None:
            nx = max(2, int(self.size / 2))
        if ny is None:
            ny = max(2, int(round(float(nx) * Ly / Lx)))
    
        nx = int(nx)
        ny = int(ny)
        nz = int(nz)
        degree = int(degree)
        heat_degree = int(heat_degree)
    
        if nx < 1 or ny < 1 or nz < 1:
            raise ValueError("solve_solid_3d_model: nx, ny, nz must be positive.")
        if degree < 1:
            raise ValueError("solve_solid_3d_model: degree must be >= 1.")
        if heat_degree < 1:
            raise ValueError("solve_solid_3d_model: heat_degree must be >= 1.")
    
        if degree == 1:
            print(
                "[Solid Solver] Warning: degree=1 tetrahedral elasticity can be bending-stiff "
                "for thin plates. For quantitative KL comparison, prefer degree=2 with a "
                "moderate mesh, or refine the 3D mesh carefully."
            )
    
        if heat_nz is None:
            heat_nz = nz
        heat_nz = int(heat_nz)
    
        if heat_nx is not None:
            heat_nx = int(heat_nx)
        if heat_ny is not None:
            heat_ny = int(heat_ny)
    
        if heat_nx is not None and heat_nx < 1:
            raise ValueError("solve_solid_3d_model: heat_nx must be positive or None.")
        if heat_ny is not None and heat_ny < 1:
            raise ValueError("solve_solid_3d_model: heat_ny must be positive or None.")
        if heat_nz < 1:
            raise ValueError("solve_solid_3d_model: heat_nz must be positive.")
    
        # ------------------------------------------------------------------
        # Thermal field: compute or fetch full 3D DeltaT.
        # No T1 reduction is used in the 3D solid model.
        # ------------------------------------------------------------------
        DeltaT_source = None
        heat_result_used = None
    
        if thermal:
            kx_eff = self.kx_rom if kx is None else float(kx)
            ky_eff = self.ky_rom if ky is None else float(ky)
            kz_eff = self.kz_rom if kz is None else float(kz)
    
            T_amb_eff = self.T_amb_rom if T_amb is None else T_amb
            h_con_eff = self.h_con_rom if h_con is None else h_con
            eps_r_eff = self.eps_r_rom if eps_r is None else float(eps_r)
            q_s_eff = self.q_s_rom if q_s is None else q_s
    
            T_sub_eff = self.T_sub_rom if T_sub is None else T_sub
            h_c_cont_eff = self.h_c_cont_rom if h_c_cont is None else h_c_cont
            h_c_gap_eff = self.h_c_gap_rom if h_c_gap is None else float(h_c_gap)
            eta_c_eff = self.eta_c_rom if eta_c is None else float(eta_c)
    
            have_stored_heat = hasattr(self, "DeltaT_3d_heat") and getattr(self, "DeltaT_3d_heat") is not None
    
            # If thermal_solve=True, force a fresh heat solve.
            # If use_heat_result=True, require an already stored heat result.
            # If neither is true, use stored heat if available; otherwise compute it.
            if thermal_solve:
                need_heat = True
            elif use_heat_result:
                if not have_stored_heat:
                    raise RuntimeError(
                        "solve_solid_3d_model: use_heat_result=True was requested, "
                        "but no stored DeltaT_3d_heat exists. Run heat_solve/one-way/coupled first, "
                        "or set thermal_solve=True."
                    )
                need_heat = False
            else:
                need_heat = not have_stored_heat
    
            if need_heat:
                default_heat_nx = max(2, int(self.size / 2))
                default_heat_ny = max(2, int(round(default_heat_nx * Ly / Lx)))
    
                heat_result_used = self.heat_solve(
                    kx=kx_eff,
                    ky=ky_eff,
                    kz=kz_eff,
    
                    T_amb=T_amb_eff,
                    h_con=h_con_eff,
                    eps_r=eps_r_eff,
                    q_s=q_s_eff,
    
                    T_sub=T_sub_eff,
                    h_c_cont=h_c_cont_eff,
                    h_c_gap=h_c_gap_eff,
                    eta_c=eta_c_eff,
    
                    w=w_for_heat,
                    source=source,
                    T_ref=T_ref,
    
                    nx=heat_nx if heat_nx is not None else default_heat_nx,
                    ny=heat_ny if heat_ny is not None else default_heat_ny,
                    nz=heat_nz,
                    degree=heat_degree,
                    store=True,
                )
                print("[Solid Solver] 3D heat_solve() executed for the solid model.")
    
            if not hasattr(self, "DeltaT_3d_heat") or self.DeltaT_3d_heat is None:
                raise RuntimeError(
                    "solve_solid_3d_model: thermal=True requires a stored full 3D DeltaT_3d_heat."
                )
    
            DeltaT_source = self.DeltaT_3d_heat
    
            try:
                DeltaT_source.set_allow_extrapolation(True)
            except Exception:
                pass
    
            self.solid_heat_result_used = heat_result_used
    
        # ------------------------------------------------------------------
        # Build solid mechanics mesh.
        # Centered z-coordinate matches heat_solve and T1 construction.
        # ------------------------------------------------------------------
        z_bot = -0.5 * t
        z_top = 0.5 * t
    
        if thermal and reuse_heat_mesh_for_solid and hasattr(self, "mesh3d_heat"):
            mesh = self.mesh3d_heat
            facets = self.facets_heat
            print("[Solid Solver] Reusing stored 3D heat mesh for solid mechanics.")
        else:
            mesh = BoxMesh(
                Point(0.0, 0.0, z_bot),
                Point(Lx, Ly, z_top),
                nx,
                ny,
                nz,
            )
    
            facets = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    
            class _Top(SubDomain):
                def inside(self, x, on_b):
                    return on_b and near(x[2], z_top)
    
            class _Bottom(SubDomain):
                def inside(self, x, on_b):
                    return on_b and near(x[2], z_bot)
    
            _Top().mark(facets, 1)
            _Bottom().mark(facets, 2)
    
            print(
                f"[Solid Solver] Built independent solid mesh: "
                f"nx={nx}, ny={ny}, nz={nz}, degree={degree}."
            )
    
        dx = Measure("dx", domain=mesh)
        ds = Measure("ds", domain=mesh, subdomain_data=facets)
    
        # ------------------------------------------------------------------
        # Interpolate the full 3D DeltaT field onto the solid mechanics mesh if needed.
        # ------------------------------------------------------------------
        if thermal:
            same_delta_mesh = False
            try:
                same_delta_mesh = DeltaT_source.function_space().mesh().id() == mesh.id()
            except Exception:
                same_delta_mesh = False
    
            if same_delta_mesh:
                DeltaT = DeltaT_source
            else:
                VDT = FunctionSpace(mesh, "CG", max(1, heat_degree))
    
                class _DeltaTOnSolidMesh(UserExpression):
                    def __init__(self, DT, **kw):
                        super().__init__(**kw)
                        self.DT = DT
    
                    def eval(self, values, x):
                        values[0] = float(
                            self.DT(Point(float(x[0]), float(x[1]), float(x[2])))
                        )
    
                    def value_shape(self):
                        return ()
    
                DeltaT = interpolate(
                    _DeltaTOnSolidMesh(
                        DeltaT_source,
                        degree=max(2, heat_degree + 2),
                    ),
                    VDT,
                )
                DeltaT.rename("DeltaT_on_solid_mesh", "")
    
            self.DeltaT_3d_solid = DeltaT
    
            try:
                DeltaT.set_allow_extrapolation(True)
            except Exception:
                pass
        else:
            DeltaT = None
    
        # ------------------------------------------------------------------
        # 3D isotropic linear thermoelasticity
        # ------------------------------------------------------------------
        mu_lame = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        I3 = Identity(3)
    
        def eps(u):
            return sym(grad(u))
    
        def sigma_mech(u):
            e = eps(u)
            return lam * tr(e) * I3 + 2.0 * mu_lame * e
    
        if thermal:
            alpha1_eff = float(alpha_iso if alpha1 is None else alpha1)
            alpha2_eff = float(alpha_iso if alpha2 is None else alpha2)
            alpha3_eff = alpha1_eff if alpha3 is None else float(alpha3)
    
            print(
                f"[Solid Solver] Thermal expansion: "
                f"alpha1={alpha1_eff:.3e}, "
                f"alpha2={alpha2_eff:.3e}, "
                f"alpha3={alpha3_eff:.3e}"
            )
    
            def eps_th():
                return as_tensor((
                    (alpha1_eff * DeltaT, 0.0, 0.0),
                    (0.0, alpha2_eff * DeltaT, 0.0),
                    (0.0, 0.0, alpha3_eff * DeltaT),
                ))
    
            def sigma_th():
                eT = eps_th()
                return lam * tr(eT) * I3 + 2.0 * mu_lame * eT
    
        else:
            def sigma_th():
                return as_tensor((
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                ))
    
        # ------------------------------------------------------------------
        # FE space and boundary conditions
        # ------------------------------------------------------------------
        V = VectorFunctionSpace(mesh, "Lagrange", degree)
        u = Function(V, name=("u_solid_T" if thermal else "u_solid"))
        v = TestFunction(V)
        du = TrialFunction(V)
    
        def _zero_on_subspace(subspace):
            """
            Build a zero Function on a collapsed scalar subspace.
    
            This avoids the RBNiCS-wrapped DirichletBC internally projecting
            Constant(0.0), which can fail on larger 3D meshes.
            """
            Vc = subspace.collapse()
            z = Function(Vc)
            z.vector().zero()
            z.vector().apply("insert")
            return z
    
        zero_x = _zero_on_subspace(V.sub(0))
        zero_y = _zero_on_subspace(V.sub(1))
        zero_z = _zero_on_subspace(V.sub(2))
    
        if self.bc_type == "simply_supported":
    
            class _LateralBoundary(SubDomain):
                def inside(self, x, on_b):
                    return on_b and (
                        near(x[0], 0.0)
                        or near(x[0], Lx)
                        or near(x[1], 0.0)
                        or near(x[1], Ly)
                    )
    
            class _P00(SubDomain):
                def inside(self, x, on_b):
                    return (
                        near(x[0], 0.0)
                        and near(x[1], 0.0)
                        and near(x[2], z_bot)
                    )
    
            class _P10(SubDomain):
                def inside(self, x, on_b):
                    return (
                        near(x[0], Lx)
                        and near(x[1], 0.0)
                        and near(x[2], z_bot)
                    )
    
            class _P01(SubDomain):
                def inside(self, x, on_b):
                    return (
                        near(x[0], 0.0)
                        and near(x[1], Ly)
                        and near(x[2], z_bot)
                    )
    
            # KL-compatible simply supported displacement condition:
            # vertical displacement is zero on the lateral boundary.
            bc_w = DirichletBC(
                V.sub(2),
                zero_z,
                _LateralBoundary(),
            )
    
            # Minimal in-plane rigid-body suppression.
            # These point constraints should not clamp the in-plane expansion of edges.
            bc_u0 = DirichletBC(V.sub(0), zero_x, _P00(), method="pointwise")
            bc_v0 = DirichletBC(V.sub(1), zero_y, _P00(), method="pointwise")
            bc_ux = DirichletBC(V.sub(0), zero_x, _P10(), method="pointwise")
            bc_vy = DirichletBC(V.sub(1), zero_y, _P01(), method="pointwise")
    
            bcs = [bc_w, bc_u0, bc_v0, bc_ux, bc_vy]
    
        elif self.bc_type == "free_edge":
            # No essential constraints.  For a pure free solid with weak foundation,
            # the foundation normally removes vertical rigid motion.  If ks=0, this
            # case may still be singular.
            bcs = []
    
        else:
            raise ValueError(f"solve_solid_3d_model: unsupported bc_type '{self.bc_type}'.")
    
        # ------------------------------------------------------------------
        # Mechanical top traction corresponding to plate load q=f
        # ------------------------------------------------------------------
        if include_mech:
            if self.load_type == "uniform":
                tz = Constant(float(self.f))
    
            elif self.load_type == "patch":
                tz = Expression(
                    "((x[0]>=x0-hx)&&(x[0]<=x0+hx)&&(x[1]>=y0-hy)&&(x[1]<=y0+hy)) ? load : 0.0",
                    degree=2,
                    x0=float(self.x0),
                    y0=float(self.y0),
                    hx=float(self.patch_size_x) / 2.0,
                    hy=float(self.patch_size_y) / 2.0,
                    load=float(self.f),
                )
    
            elif self.load_type == "multi_patch":
                cond = " || ".join(
                    f"((x[0]>={float(p['x0'])}-{float(p['patch_size_x'])/2.0})"
                    f"&&(x[0]<={float(p['x0'])}+{float(p['patch_size_x'])/2.0})"
                    f"&&(x[1]>={float(p['y0'])}-{float(p['patch_size_y'])/2.0})"
                    f"&&(x[1]<={float(p['y0'])}+{float(p['patch_size_y'])/2.0}))"
                    for p in self.patches
                )
                tz = Expression(f"({cond}) ? {float(self.f)} : 0.0", degree=2)
    
            elif self.load_type == "linear":
                tz = Expression(
                    "(x[0]>=0.0 && x[0]<=L) ? f0 + (f1 - f0) * (x[0]/L) : 0.0",
                    degree=2,
                    L=float(self.length),
                    f0=float(self.load_value_start),
                    f1=float(self.f),
                )
    
            else:
                raise ValueError(f"solve_solid_3d_model: unsupported load_type '{self.load_type}'.")
    
            traction_top = as_vector((0.0, 0.0, tz))
        else:
            traction_top = as_vector((0.0, 0.0, 0.0))
    
        # ------------------------------------------------------------------
        # Winkler foundation on the bottom surface.
        # Same sign convention as the plate energy:
        #     k_s*w in compression branch, epsilon*k_s*w in uplift branch.
        # ------------------------------------------------------------------
        eta_found = Constant(float(self.foundation_tension_factor))
        w_bottom = u[2]
    
        F_found = (
            Constant(float(self.ks))
            * conditional(w_bottom < 0.0, w_bottom, eta_found * w_bottom)
            * v[2]
            * ds(2)
        )
    
        # Body force: rho is mass density [kg/m^3], g is gravitational acceleration.
        body_force = as_vector((0.0, 0.0, -float(self.rho) * float(self.g)))
    
        # ------------------------------------------------------------------
        # Weak form:
        #     ∫ sigma(u):eps(v) dΩ
        #   - ∫ sigma_th:eps(v) dΩ
        #   + foundation
        #   - external body/top traction work
        # ------------------------------------------------------------------
        F = (
            inner(sigma_mech(u) - sigma_th(), eps(v)) * dx
            + F_found
            - dot(body_force, v) * dx
            - dot(traction_top, v) * ds(1)
        )
    
        J = derivative(F, u, du)
    
        problem = NonlinearVariationalProblem(F, u, bcs, J)
        nls = NonlinearVariationalSolver(problem)
        nls.parameters.update(NEWTON_SOLVER_PARAMETERS)
        nls.solve()
    
        print(f"3D Solid {'thermo-mechanical' if thermal else 'mechanical'} solution computed.")
    
        # ------------------------------------------------------------------
        # Store full 3D displacement and 3D vertical component
        # ------------------------------------------------------------------
        if thermal:
            self.u_solid_T = u
            self.mesh3d_T = mesh
            self.facets3d_T = facets
        else:
            self.u_solid = u
            self.mesh3d = mesh
            self.facets3d = facets
    
        Vw3 = FunctionSpace(mesh, "CG", degree)
        w_3d = project(u[2], Vw3)
        w_3d.rename(("w_solid_T" if thermal else "w_solid"), "")
    
        if thermal:
            self.w_solid_T = w_3d
        else:
            self.w_solid = w_3d
    
        # ------------------------------------------------------------------
        # Extract mid-plane vertical displacement onto the existing 2D plate mesh.
        # This is the field to compare with the Kirchhoff plate displacement w(x,y).
        # ------------------------------------------------------------------
        if hasattr(self, "mesh"):
            try:
                w_3d.set_allow_extrapolation(True)
            except Exception:
                pass
    
            class _MidPlaneW(UserExpression):
                def __init__(self, w3d, z_mid, **kw):
                    super().__init__(**kw)
                    self.w3d = w3d
                    self.z_mid = float(z_mid)
    
                def eval(self, values, x):
                    values[0] = float(
                        self.w3d(Point(float(x[0]), float(x[1]), self.z_mid))
                    )
    
                def value_shape(self):
                    return ()
    
            V2D = FunctionSpace(
                self.mesh,
                "CG",
                max(2, int(getattr(self, "degree", 2))),
            )
    
            w_plate = interpolate(
                _MidPlaneW(w_3d, 0.0, degree=max(2, degree + 1)),
                V2D,
            )
    
            if thermal:
                w_plate.rename("Solid mid-plane displacement, thermal", "")
                self.w_solid_plate_T = w_plate
                self.w_solid_plate_T_solved = w_plate
            else:
                w_plate.rename("Solid mid-plane displacement", "")
                self.w_solid_plate = w_plate
    
        # ------------------------------------------------------------------
        # Metadata for debugging and convergence reporting
        # ------------------------------------------------------------------
        final_field = self.w_solid_plate_T if thermal else self.w_solid_plate
    
        self.solid_3d_metadata = dict(
            thermal=bool(thermal),
            nx=int(nx),
            ny=int(ny),
            nz=int(nz),
            degree=int(degree),
            heat_nx=None if heat_nx is None else int(heat_nx),
            heat_ny=None if heat_ny is None else int(heat_ny),
            heat_nz=int(heat_nz),
            heat_degree=int(heat_degree),
            reuse_heat_mesh_for_solid=bool(reuse_heat_mesh_for_solid),
            E=float(E),
            nu=float(nu),
            D_eff=float(D_eff),
            Ds_iso=float(Ds_iso),
            vector_dofs=int(V.dim()),
            midplane_dofs=int(final_field.function_space().dim()),
            used_full_3d_DeltaT=bool(thermal),
            used_T1_reduction=False,
        )
    
        if plot_results:
            plt.figure(figsize=(6, 4))
            m = plot(final_field, cmap="viridis")
            plt.colorbar(m, shrink=0.8)
            plt.title(f"3D solid{' thermo-mechanical' if thermal else ' mechanical'} — mid-plane displacement")
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.gca().set_aspect("equal")
            plt.tight_layout()
            plt.show()
    
        print(
            f"[Solid Solver] Final mid-plane field DoFs = {final_field.function_space().dim()} | "
            f"3D vector DoFs = {V.dim()} | used_T1_reduction=False"
        )
    
        return final_field
        
    # ------------------------------------------------------------------
    # Post-processing and visualization
    # ------------------------------------------------------------------
    def analyze_and_visualize_general(self, plot_results=True):
        """
        Compare and visualize currently available solution fields.
        Postprocessor that adapts automatically to the solution fields currently
        available in the object:
            ─ Kirchhoff FEM         → self.w_contact_global   (always)
            ─ 3-D solid FEM         → self.w_solid_plate      (optional)
            ─ Exact analytical sol. → self.Exact_solution     (optional)
        """
        all_fields = {
            "Kirchhoff FEM": (self.w_contact_global, "C0"),
            "3-D Solid": (getattr(self, "w_solid_plate", None), "C1"),
            "Exact": (getattr(self, "Exact_solution", None), "C2"),
        }
        fields = [(name, fld, col) for name, (fld, col) in all_fields.items() if fld is not None]
    
        kir_field, kir_col = all_fields["Kirchhoff FEM"]
        solid_field, solid_col = all_fields["3-D Solid"]
        exact_field, exact_col = all_fields["Exact"]
    
        bold = lambda s: f"\033[1m{s}\033[0m"
        sci = lambda x, unit="": f"{float(x):.2E}" + (f" {unit}" if unit else "")
        fmtm = lambda x: f"{float(x):.6f} m"
    
        extrema_rows = [
            row
            for name, fld, _ in fields
            for row in (
                [f"Min ({name})", fmtm(fld.vector().get_local().min())],
                [f"Max ({name})", fmtm(fld.vector().get_local().max())],
            )
        ]
        sd_rows = [[f"Ω{sid}", fmtm(self.w_contact_subdomains[sid].vector().min())] for sid in self.subdomain_ids]
    
        load_rows = {
            "uniform": [["Magnitude", sci(self.f, "N/m²")]],
            "patch": [
                ["Center (x0,y0)", f"({self.x0:.2f}, {self.y0:.2f})"],
                ["Size (x,y)", f"{self.patch_size_x:.2f} × {self.patch_size_y:.2f} m²"],
                ["Magnitude", sci(self.f, "N/m²")],
            ],
            "multi_patch": [
                [
                    f"Patch {i+1}",
                    f"({p['x0']:.2f}, {p['y0']:.2f}) | {p['patch_size_x']:.2f}×{p['patch_size_y']:.2f} | {float(self.f):.2f} N/m²",
                ]
                for i, p in enumerate(self.patches)
            ],
            "linear": [
                ["x Range", f"{self.x_start:.2f} → {self.x_end:.2f} m"],
                ["Value Range", f"{self.load_value_start:.2f} → {float(self.f):.2f} N/m²"],
            ],
        }
    
        table_data = [
            [bold("Plate Properties"), ""],
            ["Length", f"{self.length:.2f} m"],
            ["Width", f"{self.width:.2f} m"],
            ["Thickness", f"{float(self.t):.3f} m"],
            ["", ""],
    
            [bold("Penalty Parameters"), ""],
            ["Contact Displacement Penalty constant C₁ (for α)", sci(self.C_alpha)],
            ["Contact Rotation Penalty constant C₂ (for β)", sci(self.C_beta)],
            ["Foundation Stiffness ks", sci(self.ks, "N/m³")],
            ["", ""],
    
            [bold("Flexural & Torsional Rigidities"), ""],
            ["Flexural Rigidity D_x", sci(self.D_x, "N·m")],
            ["Flexural Rigidity D_y", sci(self.D_y, "N·m")],
            ["Coupling Rigidity D_xy", sci(self.D_xy, "N·m")],
            ["Torsional Rigidity D_s", sci(self.D_s, "N·m")],
            ["", ""],
    
            [bold("Load Information"), ""],
            ["Type", self.load_type],
            *load_rows.get(self.load_type, []),
            ["", ""],
            [bold("Extrema"), ""],
            *extrema_rows,
            ["", ""],
            [bold("Sub-domain Min (Kirchhoff)"), ""],
            *sd_rows,
        ]
    
        print("\n" + "=" * 55)
        print(f"{bold('Plate Solution Information'):^55}")
        print("=" * 55)
        print(tabulate(table_data, tablefmt="plain"))
        print("=" * 55)
    
        if not plot_results:
            return
    
        def sample_along(field, X, Y):
            return np.array([field(Point(x, y)) for x, y in zip(X, Y)])
    
        def field_on_grid(field, Xg, Yg):
            Z = np.zeros_like(Xg)
            for j in range(Xg.shape[0]):
                for i in range(Xg.shape[1]):
                    Z[j, i] = field(Point(float(Xg[j, i]), float(Yg[j, i])))
            return Z
    
        def make_four_ticks(vmin, vmax, positive_only=False):
            eps = 1e-14
            if np.isclose(vmin, vmax, atol=eps):
                return [vmin, vmin, vmin, vmin]
            if positive_only:
                if np.isclose(vmax, 0.0, atol=eps):
                    return [0.0, 0.0, 0.0, 0.0]
                return [0.0, vmax / 3.0, 2.0 * vmax / 3.0, vmax]
            return np.linspace(vmin, vmax, 4).tolist()
    
        def scaled_cbar_info(vmin, vmax, positive_only=False, decimals=1):
            ticks = make_four_ticks(vmin, vmax, positive_only=positive_only)
            ref = max(abs(vmin), abs(vmax))
            exponent = 0 if ref < 1e-14 else int(np.floor(np.log10(ref)))
            scale = 1.0 if ref < 1e-14 else 10.0 ** exponent
            labels = [f"{t / scale:.{decimals}f}" for t in ticks]
            return ticks, labels, exponent
    
        def style_map_axis(ax, xlabel=True, ylabel=True):
            ax.set_facecolor("white")
            ax.set_aspect("equal")
            ax.set_xlim(0.0, self.length)
            ax.set_ylim(0.0, self.width)
            ax.set_xlabel("x [m]" if xlabel else "", fontsize=14)
            ax.set_ylabel("y [m]" if ylabel else "", fontsize=14)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    
            def xfmt(x, pos):
                return "" if np.isclose(x, 0.0) else f"{x:.1f}"
    
            ax.xaxis.set_major_formatter(FuncFormatter(xfmt))
            ax.tick_params(axis="x", labelsize=13, length=3, pad=1)
            ax.tick_params(axis="y", labelsize=13, length=3, pad=3)
            ax.grid(False)
    
        def draw_scalar(fig_obj, ax, Z, title, vmin, vmax, positive_only=False, xlabel=True, ylabel=True):
            mappable = ax.contourf(
                Xg, Yg, Z,
                levels=np.linspace(vmin, vmax, 181),
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            style_map_axis(ax, xlabel=xlabel, ylabel=ylabel)
            ax.set_title(title, fontsize=14, pad=8)
    
            ticks, ticklabels, exponent = scaled_cbar_info(
                vmin, vmax, positive_only=positive_only, decimals=1
            )
            cax = make_axes_locatable(ax).append_axes("right", size="4.2%", pad=0.10)
            cbar = fig_obj.colorbar(mappable, cax=cax)
            cbar.set_ticks(ticks)
            cbar.ax.yaxis.set_major_locator(FixedLocator(ticks))
            cbar.ax.set_yticklabels(ticklabels)
            cbar.ax.minorticks_off()
            cbar.ax.tick_params(labelsize=13, length=2, pad=2)
            cbar.outline.set_linewidth(0.6)
            cbar.solids.set_edgecolor("face")
            cbar.ax.set_title(rf"$\times 10^{{{exponent}}}$" if exponent != 0 else "", fontsize=14, pad=6)
            return mappable
    
        def save_dual(fig_obj, base_path, pad_inches=None):
            save_kw = dict(facecolor="white")
            if pad_inches is not None:
                save_kw["pad_inches"] = pad_inches
            fig_obj.savefig(f"{base_path}.pdf", dpi=300, bbox_inches="tight", **save_kw)
            fig_obj.savefig(f"{base_path}.png", dpi=600, bbox_inches="tight", **save_kw)
    
        def draw_line(ax, lw_main, lw_aux):
            ax.set_facecolor("white")
            ax.plot(diag, diag_k, color=kir_col, lw=lw_main, label=r"$w_{\mathrm{fom}}$")
            if diag_s is not None:
                ax.plot(diag, diag_s, color=solid_col, lw=lw_aux, ls="--", label=r"$w_{\mathrm{sol}}$")
            if diag_e is not None:
                ax.plot(diag, diag_e, color=exact_col, lw=lw_aux, ls="-.", label=r"$w_{\mathrm{ext}}$")
    
            ax.set_box_aspect(0.48)
            ax.set_title(r"Main diagonal comparison", fontsize=13, pad=8)
            ax.set_xlabel("Diagonal coordinate [m]", fontsize=12)
            ax.set_ylabel("w [m]", fontsize=12)
            ax.tick_params(axis="both", labelsize=12, length=3, pad=2)
            ax.grid(True, linestyle=":", linewidth=0.45, alpha=0.35)
    
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-2, 2))
            ax.yaxis.set_major_formatter(formatter)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            ax.yaxis.get_offset_text().set_size(12)
    
            leg = ax.legend(
                loc="lower left",
                frameon=True,
                fontsize=13,
                borderpad=0.45,
                handlelength=2.2,
                handletextpad=0.6,
                labelspacing=0.35,
            )
            leg.get_frame().set_edgecolor("black")
            leg.get_frame().set_linewidth(0.8)
            leg.get_frame().set_alpha(0.95)
    
        xs, ys = np.linspace(0.0, self.length, 200), np.linspace(0.0, self.width, 200)
        diag = np.sqrt(xs**2 + ys**2)
    
        V_plot = FunctionSpace(self.mesh, "CG", 3)
        kir_plot = project(kir_field, V_plot)
        solid_plot = project(solid_field, V_plot) if solid_field is not None else None
        exact_plot = project(exact_field, V_plot) if exact_field is not None else None
        err_s_plot = project(abs(kir_plot - solid_plot), V_plot) if solid_plot is not None else None
        err_e_plot = project(abs(kir_plot - exact_plot), V_plot) if exact_plot is not None else None
    
        diag_k = sample_along(kir_plot, xs, ys)
        diag_s = sample_along(solid_plot, xs, ys) if solid_plot is not None else None
        diag_e = sample_along(exact_plot, xs, ys) if exact_plot is not None else None
    
        nx_plot, ny_plot = 260, 150
        xg, yg = np.linspace(0.0, self.length, nx_plot), np.linspace(0.0, self.width, ny_plot)
        Xg, Yg = np.meshgrid(xg, yg)
    
        kir_grid = field_on_grid(kir_plot, Xg, Yg)
        solid_grid = field_on_grid(solid_plot, Xg, Yg) if solid_plot is not None else None
        exact_grid = field_on_grid(exact_plot, Xg, Yg) if exact_plot is not None else None
        err_s_grid = field_on_grid(err_s_plot, Xg, Yg) if err_s_plot is not None else None
        err_e_grid = field_on_grid(err_e_plot, Xg, Yg) if err_e_plot is not None else None
    
        solution_arrays = [kir_grid] + [g for g in (solid_grid, exact_grid) if g is not None]
        sol_vmin, sol_vmax = min(np.min(a) for a in solution_arrays), max(np.max(a) for a in solution_arrays)
    
        plt.rcParams["figure.autolayout"] = False
        fig = plt.figure(figsize=(12.0, 11.8), facecolor="white")
        gs = GridSpec(
            3, 2, figure=fig,
            width_ratios=[1.0, 1.0],
            left=0.055, right=0.982, bottom=0.055, top=0.985,
            hspace=0.035, wspace=0.29,
        )
    
        ax_fom, ax_sol, ax_ext = [fig.add_subplot(gs[i, 0]) for i in range(3)]
        ax_line, ax_err_s, ax_err_e = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    
        draw_scalar(fig, ax_fom, kir_grid, r"FOM ($w_{fom}$ $[m]$)", sol_vmin, sol_vmax, xlabel=True)
        if solid_grid is not None:
            draw_scalar(fig, ax_sol, solid_grid, r"3D solid ($w_{sol}$ $[m]$)", sol_vmin, sol_vmax, xlabel=True)
        else:
            ax_sol.axis("off")
    
        if exact_grid is not None:
            draw_scalar(fig, ax_ext, exact_grid, r"Exact ($w_{ext}$ $[m]$)", sol_vmin, sol_vmax, xlabel=True)
        else:
            ax_ext.axis("off")
    
        draw_line(ax_line, lw_main=3.4, lw_aux=3.1)
        plt.subplots_adjust(right=0.92)
    
        if err_s_grid is not None:
            draw_scalar(
                fig, ax_err_s, err_s_grid,
                r"$|w_{\mathrm{fom}}-w_{\mathrm{sol}}|$ $[m]$",
                0.0, np.max(err_s_grid), positive_only=True, xlabel=True,
            )
        else:
            ax_err_s.axis("off")
    
        if err_e_grid is not None:
            draw_scalar(
                fig, ax_err_e, err_e_grid,
                r"$|w_{\mathrm{fom}}-w_{\mathrm{ext}}|$ $[m]$",
                0.0, np.max(err_e_grid), positive_only=True, xlabel=True,
            )
        else:
            ax_err_e.axis("off")
    
        save_dual(fig, os.path.join(self.output_dir, "summary_plots"))
    
        summary_indiv_dir = os.path.join(self.output_dir, "summary_indiv")
        os.makedirs(summary_indiv_dir, exist_ok=True)
    
        def save_scalar_panel(Z, title, out_name, vmin, vmax, positive_only=False, xlabel=True, ylabel=True):
            fig_i = plt.figure(figsize=(5.8, 4.4), facecolor="white")
            ax_i = fig_i.add_subplot(111)
            draw_scalar(fig_i, ax_i, Z, title, vmin, vmax, positive_only=positive_only, xlabel=xlabel, ylabel=ylabel)
            save_dual(fig_i, os.path.join(summary_indiv_dir, out_name), pad_inches=0.03)
            plt.close(fig_i)
    
        def save_line_panel(out_name):
            fig_i = plt.figure(figsize=(5.4, 4.4), facecolor="white")
            ax_i = fig_i.add_subplot(111)
            draw_line(ax_i, lw_main=3.0, lw_aux=2.7)
            save_dual(fig_i, os.path.join(summary_indiv_dir, out_name), pad_inches=0.03)
            plt.close(fig_i)
    
        scalar_exports = [
            (kir_grid, r"FOM ($w_{fom}$ $[m]$)", "fom", sol_vmin, sol_vmax, False),
            (solid_grid, r"3D solid ($w_{sol}$ $[m]$)", "solid", sol_vmin, sol_vmax, False),
            (exact_grid, r"Exact ($w_{ext}$ $[m]$)", "exact", sol_vmin, sol_vmax, False),
            (err_s_grid, r"$|w_{\mathrm{fom}}-w_{\mathrm{sol}}|$ $[m]$", "err_fom_sol", 0.0, np.max(err_s_grid) if err_s_grid is not None else None, True),
            (err_e_grid, r"$|w_{\mathrm{fom}}-w_{\mathrm{ext}}|$ $[m]$", "err_fom_ext", 0.0, np.max(err_e_grid) if err_e_grid is not None else None, True),
        ]
        for Z, title, name, vmin, vmax, positive_only in scalar_exports:
            if Z is not None:
                save_scalar_panel(Z, title, name, vmin, vmax, positive_only=positive_only, xlabel=True, ylabel=True)
    
        save_line_panel("diag_compare")
        plt.show()
    
        disp_dir = os.path.join(self.output_dir, "displacement_solutions")
        os.makedirs(disp_dir, exist_ok=True)
    
        def save_xdmf(field, name):
            if field is not None:
                with XDMFFile(os.path.join(disp_dir, f"{name}.xdmf")) as f:
                    f.write(field)
    
        for attr, name in [
            ("u_solid", "u_solid"),
            ("w_solid", "w_solid_m"),
            ("w_solid_plate", "w_solid_midplane"),
            ("w_contact_global", "w_kirchhoff"),
            ("Exact_solution", "w_exact"),
        ]:
            save_xdmf(getattr(self, attr, None), name)
    
    def post_processor(self, use_global=True, degree=0, cmap="viridis", plot_results=True):
        """Compute and visualize derived bending moments and stress quantities."""
        assert hasattr(self, "w_contact_solution"), "Run `offline_solve_kirchhoff_problem()` first."
    
        Vmom = FunctionSpace(self.mesh, "DG", degree)
        Vdisp = FunctionSpace(self.mesh, "CG", max(1, degree))
        z_fibre, I_plate = self.t / 2.0, self.t**3 / 12.0
    
        def derive_fields(w_fenics):
            """Return (w_m, Mx, My, Mxy, M1, M2, S1, S2, vM)."""
            w_m = project(w_fenics, Vdisp)
            w_m.rename("w_m", "")

            sk = sym(grad(grad(w_fenics)))
            kxx, kyy, kxy = sk[0, 0], sk[1, 1], sk[0, 1]
            
            # Mechanical curvature moments, using the physical Reddy sign convention.
            Mx_mec_expr = -(self.D_x * kxx + self.D_xy * kyy)
            My_mec_expr = -(self.D_y * kyy + self.D_xy * kxx)
            Mxy_mec_expr = -(2.0 * self.D_s * kxy)
            
            # Total thermoelastic moments.
            # For thermal-off runs, total = mechanical.
            if bool(getattr(self, "_thermal_enabled", False)):
                self._realize_thermal_fields()
            
                Mx_th_expr = (
                    self.D_x * self.alpha1_c
                    + self.D_xy * self.alpha2_c
                ) * self.T1_plate
            
                My_th_expr = (
                    self.D_xy * self.alpha1_c
                    + self.D_y * self.alpha2_c
                ) * self.T1_plate
            
                # Current implementation assumes aligned orthotropy and no alpha_xy term.
                Mxy_th_expr = Constant(0.0)
            
                Mx_expr = Mx_mec_expr - Mx_th_expr
                My_expr = My_mec_expr - My_th_expr
                Mxy_expr = Mxy_mec_expr - Mxy_th_expr
            else:
                Mx_expr = Mx_mec_expr
                My_expr = My_mec_expr
                Mxy_expr = Mxy_mec_expr
            
            Mx = project(Mx_expr, Vmom); Mx.rename("M_x_tot", "")
            My = project(My_expr, Vmom); My.rename("M_y_tot", "")
            Mxy = project(Mxy_expr, Vmom); Mxy.rename("M_xy_tot", "")
    
            M_avg, Rm = 0.5 * (Mx + My), sqrt((0.5 * (Mx - My))**2 + Mxy**2)
            M1 = project(M_avg + Rm, Vmom); M1.rename("M1", "")
            M2 = project(M_avg - Rm, Vmom); M2.rename("M2", "")
    
            sig_x = (Mx * z_fibre) / I_plate
            sig_y = (My * z_fibre) / I_plate
            tau_xy = (Mxy * z_fibre) / I_plate
            s_avg, Rs = 0.5 * (sig_x + sig_y), sqrt((0.5 * (sig_x - sig_y))**2 + tau_xy**2)
    
            S1 = project(s_avg + Rs, Vmom); S1.rename("S1", "")
            S2 = project(s_avg - Rs, Vmom); S2.rename("S2", "")
            vM = project(sqrt(S1**2 + S2**2 - S1 * S2), Vmom); vM.rename("von_Mises", "")
    
            return w_m, Mx, My, Mxy, M1, M2, S1, S2, vM
    
        def plot_2col(fields_tuple, suffix=""):
            if not plot_results:
                return
    
            plt.rcParams["figure.autolayout"] = False

            titles = (
                r"$M_x^{\mathrm{tot}}\,[\mathrm{N\,m}]$",
                r"$M_y^{\mathrm{tot}}\,[\mathrm{N\,m}]$",
                r"$M_{xy}^{\mathrm{tot}}\,[\mathrm{N\,m}]$",
                r"$S_1^{\mathrm{tot}}\,[\mathrm{Pa}]$",
                r"$S_2^{\mathrm{tot}}\,[\mathrm{Pa}]$",
                r"$\sigma_{\mathrm{vM}}^{\mathrm{tot}}\,[\mathrm{Pa}]$",
                r"$w\,[\mathrm{m}]$",
                r"$M_1^{\mathrm{tot}}\,[\mathrm{N\,m}]$",
                r"$M_2^{\mathrm{tot}}\,[\mathrm{N\,m}]$",
            )
            order = [1, 2, 3, 6, 7, 8, 0, 4, 5]
            file_tags = ("Mx", "My", "Mxy", "S1", "S2", "vonMises", "w", "M1", "M2")
    
            V_plot = FunctionSpace(self.mesh, "CG", 3)
            fields_plot = [project(fields_tuple[i], V_plot) for i in order]
    
            nx_plot, ny_plot = 260, 150
            xg = np.linspace(0.0, self.length, nx_plot)
            yg = np.linspace(0.0, self.width, ny_plot)
            Xg, Yg = np.meshgrid(xg, yg)
    
            def field_on_grid(field):
                Z = np.zeros_like(Xg)
                for j in range(ny_plot):
                    for i in range(nx_plot):
                        Z[j, i] = field(Point(float(Xg[j, i]), float(Yg[j, i])))
                return Z
    
            def make_four_ticks(vmin, vmax):
                eps = 1e-14
                if np.isclose(vmin, vmax, atol=eps):
                    return [vmin, vmin, vmin, vmin]
                return np.linspace(vmin, vmax, 4).tolist()
    
            def scaled_cbar_info(vmin, vmax, decimals=1):
                ticks = make_four_ticks(vmin, vmax)
                ref = max(abs(vmin), abs(vmax))
                exponent = 0 if ref < 1e-14 else int(np.floor(np.log10(ref)))
                scale = 1.0 if ref < 1e-14 else 10.0 ** exponent
                labels = [f"{t / scale:.{decimals}f}" for t in ticks]
                return ticks, labels, exponent
    
            def style_axis(ax, xlabel=True, ylabel=True):
                ax.set_facecolor("white")
                ax.set_aspect("equal")
                ax.set_xlim(0.0, self.length)
                ax.set_ylim(0.0, self.width)
                ax.set_xlabel("x [m]" if xlabel else "", fontsize=14)
                ax.set_ylabel("y [m]" if ylabel else "", fontsize=14)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    
                def xfmt(x, pos):
                    return "" if np.isclose(x, 0.0) else f"{x:.1f}"
    
                ax.xaxis.set_major_formatter(FuncFormatter(xfmt))
                ax.tick_params(axis="x", labelsize=13, length=3, pad=1)
                ax.tick_params(axis="y", labelsize=13, length=3, pad=3)
                ax.grid(False)
    
            def draw_scalar(fig_obj, ax, Z, title, xlabel=True, ylabel=True):
                vmin, vmax = float(np.min(Z)), float(np.max(Z))
                if np.isclose(vmin, vmax):
                    vmax = vmin + 1e-12
    
                mappable = ax.contourf(
                    Xg, Yg, Z,
                    levels=np.linspace(vmin, vmax, 181),
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
    
                style_axis(ax, xlabel=xlabel, ylabel=ylabel)
                ax.set_title(title, fontsize=14, pad=8)
    
                ticks, ticklabels, exponent = scaled_cbar_info(vmin, vmax, decimals=1)
                cax = make_axes_locatable(ax).append_axes("right", size="4.2%", pad=0.10)
                cbar = fig_obj.colorbar(mappable, cax=cax)
                cbar.set_ticks(ticks)
                cbar.ax.yaxis.set_major_locator(FixedLocator(ticks))
                cbar.ax.set_yticklabels(ticklabels)
                cbar.ax.minorticks_off()
                cbar.ax.tick_params(labelsize=13, length=2, pad=2)
                cbar.outline.set_linewidth(0.6)
                cbar.solids.set_edgecolor("face")
                cbar.ax.set_title(rf"$\times 10^{{{exponent}}}$" if exponent != 0 else "", fontsize=14, pad=6)
    
            def save_dual(fig_obj, base_path, pad_inches=None):
                save_kw = dict(facecolor="white")
                if pad_inches is not None:
                    save_kw["pad_inches"] = pad_inches
                fig_obj.savefig(f"{base_path}.pdf", dpi=300, bbox_inches="tight", **save_kw)
                fig_obj.savefig(f"{base_path}.png", dpi=600, bbox_inches="tight", **save_kw)
    
            grids = [field_on_grid(fld) for fld in fields_plot]
    
            fig = plt.figure(figsize=(12.0, 19.2), facecolor="white")
            gs = GridSpec(
                5, 2, figure=fig,
                width_ratios=[1.0, 1.0],
                left=0.055, right=0.982,
                bottom=0.045, top=0.985,
                hspace=0.16,
                wspace=0.29,
            )
            axes = [fig.add_subplot(gs[i, j]) for i in range(5) for j in range(2)]
    
            for ax, Z, ttl in zip(axes[:9], grids, titles):
                draw_scalar(fig, ax, Z, ttl, xlabel=True, ylabel=True)
            axes[9].axis("off")
    
            base = f"post_process{suffix}" if suffix else "post_process"
            save_dual(fig, os.path.join(self.output_dir, base))
    
            post_indiv_dir = os.path.join(self.output_dir, "post_indiv")
            os.makedirs(post_indiv_dir, exist_ok=True)
    
            for Z, ttl, tag in zip(grids, titles, file_tags):
                fig_i = plt.figure(figsize=(5.8, 4.4), facecolor="white")
                ax_i = fig_i.add_subplot(111)
                draw_scalar(fig_i, ax_i, Z, ttl, xlabel=True, ylabel=True)
                out_name = f"{tag}{suffix}" if suffix else tag
                save_dual(fig_i, os.path.join(post_indiv_dir, out_name), pad_inches=0.03)
                plt.close(fig_i)
    
            plt.show()
    
        if use_global or self.N_subdomains == 1:
            out_fields = derive_fields(self.w_contact_global)
            if plot_results:
                plot_2col(out_fields)
            return out_fields
    
        keys = ("w", "Mx", "My", "Mxy", "M1", "M2", "S1", "S2", "vM")
        results = {}
    
        for sid in self.subdomain_ids:
            idx = int(sid) - 1
            fields = derive_fields(self.w_contact_solution.sub(idx))
            results[int(sid)] = dict(zip(keys, fields))
    
            if plot_results:
                print(f"\nSub-domain Ω{sid}")
                plot_2col(fields, suffix=f"_Omega{sid}")
    
        return results

# %% [markdown] Cell 6 | id: 33f1f30e-636f-4d7f-a1cf-c5febd518a47
# # Performance Study

# %% Cell 7 | id: 4ee8c613-4b62-431b-9556-b2c998cd62d7
# ============================================================
# Single thermomechanical performance / verification test
# ============================================================

def run_single_test_thermomechanical(
    case_id,
    mode="one_way",              # "mechanical", "one_way", "coupled", "mechanical+one_way", "one_way+coupled", "all"
    season="summer",             # "summer" | "winter"
    load_type="uniform",
    bc_type="simply_supported",
    n_vert=0,
    n_horiz=0,

    # Mechanical / KL discretization
    plate_degree=2,

    # Explicit heat/T1 controls
    heat_nx=None,
    heat_ny=None,
    heat_nz=None,
    heat_degree=None,
    Nz_quad_T1=None,
    T1_cg_degree=None,

    # Exact/Navier controls
    num_terms_exact=60,
    exact_degree=8,
    thermal_projection_grid=401,
    foundation_branch="auto",

    # Coupling controls
    coupling_omega=None,
    coupling_tol_w=None,
    coupling_tol_T1=None,
    coupling_max_iters=None,
    require_coupled_convergence=True,

    # Optional 3D solid comparison
    run_solid=True,
    solid_nx=None,
    solid_ny=None,
    solid_nz=8,
    solid_degree=2,

    plot_subdomains=True,
    plot_results=True,
    convergence_plot=False,
    convergence_plot_label=None,
    convergence_plot_savebase=None,
):
    # ------------------------------------------------------------------
    # Mechanical test parameter sets
    # ------------------------------------------------------------------
    test_mu_values = {
        # Paper 2 convergence-study isotropic setting:
        # (Dx, Dy, Dxy, Ds, ks, f)
        1: [1.0e4, 1.0e4, 0.30e4, 0.35e4, 1.0e3, -10000.0],

        # Optional Paper 2 convergence-study orthotropic setting:
        # (Dx, Dy, Dxy, Ds, ks, f)
        8: [2.0e4, 1.0e4, 0.25e4, 0.30e4, 1.0e3, -10000.0],

        # Older/generic cases retained.
        2: [2.5, 1.0e7, -9000],
        3: [2.0e8, -10000],
        4: [5.0e4, 0.5, 1.0],
        5: [0.5, 1.2],
        6: [0.06, 0.025],
        7: [2.5e-7, 0.025],
    }

    # ------------------------------------------------------------------
    # Thermal / coupled defaults
    # ------------------------------------------------------------------
    cfg = globals().get("CONFIG", dict(
        # Defaults aligned with the final Paper 2 coupled-refinement controls.
        heat_nx=64,
        heat_ny=32,
        heat_nz=16,
        heat_degree=1,
        Nz_quad_T1=20,
        T1_cg_degree=None,   # if None, use plate_degree, matching p_theta = p

        # Paper 2 thermal material/contact data.
        alpha=1.3e-4,
        k_poly=0.35,
        rho=1050.0,
        h_c_cont=200.0,
        h_c_gap=5.0,
        eta_c=50.0,
        w_const=-0.1,

        # Final coupled-solver settings.
        coupling_omega=0.7,
        coupling_tol_w=1e-7,
        coupling_tol_T1=1e-7,
        coupling_max_iters=40,
    ))

    seasons = globals().get("SEASONS", {
        "summer": {
            "label": "Trieste - Summer",
            "T_amb_C": 27.0,
            "T_sub_C": 24.0,
            "h_con": 10.0,
            "eps_r": 0.90,
            "q_s": 700.0,
        },
        "winter": {
            "label": "Trieste - Winter",
            "T_amb_C": 6.0,
            "T_sub_C": 10.0,
            "h_con": 12.0,
            "eps_r": 0.90,
            "q_s": 250.0,
        },
    })

    K_from_C = globals().get("_K_from_C", lambda Tc: float(Tc) + 273.15)

    # ------------------------------------------------------------------
    # Validation / defaults
    # ------------------------------------------------------------------
    if case_id not in test_mu_values:
        print(f"--- TEST FOR CASE {case_id} SKIPPED: no test mu values defined. ---")
        return None

    if season not in seasons:
        print(f"--- TEST SKIPPED: unknown season '{season}'. Use one of {tuple(seasons)}. ---")
        return None

    heat_nx = cfg.get("heat_nx", None) if heat_nx is None else heat_nx
    heat_ny = cfg.get("heat_ny", None) if heat_ny is None else heat_ny
    heat_nz = cfg.get("heat_nz", 16) if heat_nz is None else heat_nz
    heat_degree = cfg.get("heat_degree", 2) if heat_degree is None else heat_degree
    Nz_quad_T1 = cfg.get("Nz_quad_T1", 20) if Nz_quad_T1 is None else Nz_quad_T1
    T1_cg_degree = cfg.get("T1_cg_degree", None) if T1_cg_degree is None else T1_cg_degree

    # In the convergence/performance check, the thermal-driver CG degree is p_theta = p.
    if T1_cg_degree is None:
        T1_cg_degree = int(plate_degree)

    coupling_omega = cfg["coupling_omega"] if coupling_omega is None else coupling_omega
    coupling_tol_w = cfg["coupling_tol_w"] if coupling_tol_w is None else coupling_tol_w
    coupling_tol_T1 = cfg["coupling_tol_T1"] if coupling_tol_T1 is None else coupling_tol_T1
    coupling_max_iters = cfg["coupling_max_iters"] if coupling_max_iters is None else coupling_max_iters

    test_mu = test_mu_values[case_id]
    s = seasons[season]

    test_mu_th = {
        "T_amb": K_from_C(s["T_amb_C"]),
        "T_sub": K_from_C(s["T_sub_C"]),
        "h_con": s["h_con"],
        "eps_r": s["eps_r"],
        "q_s": s["q_s"],
        "h_c_cont": cfg["h_c_cont"],
        "h_c_gap": cfg["h_c_gap"],
        "eta_c": cfg["eta_c"],
        "w_contact": cfg["w_const"],
        "kx": cfg["k_poly"],
        "ky": cfg["k_poly"],
        "kz": cfg["k_poly"],
        "alpha1": cfg["alpha"],
        "alpha2": cfg["alpha"],
        "rho": cfg["rho"],
    }

    # ------------------------------------------------------------------
    # Mode parsing
    # ------------------------------------------------------------------
    def _normalize_mode_name(name):
        key = str(name).lower().replace(" ", "").replace("-", "_")
        key = key.replace("oneway", "one_way")
        return key

    allowed = ("mechanical", "one_way", "coupled")
    key = _normalize_mode_name(mode)

    if key == "all":
        modes = ["mechanical", "one_way", "coupled"]
    else:
        modes = [_normalize_mode_name(m) for m in key.split("+")]

    modes = list(dict.fromkeys(modes))

    if any(m not in allowed for m in modes):
        print(
            f"--- TEST SKIPPED: unknown mode '{mode}'. "
            "Use 'mechanical', 'one_way', 'coupled', 'mechanical+one_way', "
            "'one_way+coupled', or 'all'. ---"
        )
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _fmt_opt_int(v):
        return "None" if v is None else str(int(v))

    def _l2_report(solver, pairs):
        vals = []
        for name, attr in pairs:
            fld = getattr(solver, attr, None)
            if fld is not None:
                vals.append(f"  {name:<24}: {norm(fld, 'L2'):.6e} m")
        if vals:
            print("\nL2 Norms of Displacement Fields:\n" + "\n".join(vals))

    def _alias_thermal_fields_for_analyzer(solver):
        if getattr(solver, "w_solid_plate_T", None) is not None:
            solver.w_solid_plate = solver.w_solid_plate_T
        if getattr(solver, "Exact_solution_thermal", None) is not None:
            solver.Exact_solution = solver.Exact_solution_thermal

    def _install_final_coupled_w_for_reporting(solver, w_final):
        """
        Ensure post-processing/reporting uses the final coupled displacement field.
        This is important because the coupled solver reports the last accepted
        relaxed iterate, not necessarily the last raw mechanical predictor.
        """
        if w_final is None:
            return

        solver.w_contact_global = Function(w_final.function_space())
        solver.w_contact_global.assign(w_final)

        if hasattr(solver, "V_DG"):
            try:
                solver.w_contact_subdomains = {1: project(solver.w_contact_global, solver.V_DG)}
            except Exception:
                pass

    def _safe_T1_callable(T1_field):
        if T1_field is None:
            return None
        try:
            T1_field.set_allow_extrapolation(True)
        except Exception:
            pass
        return lambda x, y, fld=T1_field: float(fld(Point(float(x), float(y))))

    def _exact_possible(solver, thermal=False):
        if solver.load_type != "uniform":
            return False
        if getattr(solver, "N_subdomains", 1) > 1:
            return False
        if thermal and solver.bc_type != "simply_supported":
            return False
        if solver.bc_type not in ("simply_supported", "free_edge"):
            return False
        return True

    def _make_solver():
        solver = GeneralMultiphysicsSolver(study_case=case_id)
        solver.degree = int(plate_degree)
        solver.load_type = load_type
        solver.bc_type = bc_type
        solver.define_domain(
            n_vert=n_vert,
            n_horiz=n_horiz,
            plot_subdomains=plot_subdomains,
        )
        return solver

    def _maybe_solve_solid(solver, thermal=False, alpha1=None, alpha2=None):
        if not run_solid:
            return

        # Use a genuinely refined 3D solid mechanics mesh.
        # Do not accidentally reuse the coarse heat mesh as the solid mechanics mesh.
        solid_nx_eff = solid_nx
        solid_ny_eff = solid_ny

        if solid_nx_eff is None:
            solid_nx_eff = max(16, int(solver.size))
        if solid_ny_eff is None:
            solid_ny_eff = max(8, int(round(float(solid_nx_eff) * solver.width / solver.length)))

        # For an isotropic 3D thermoelastic solid comparison, alpha3 should match alpha1
        # unless a deliberately anisotropic 3D thermal expansion is intended.
        alpha3_eff = None
        if thermal:
            alpha3_eff = alpha1 if alpha1 is not None else solver.alpha1_rom

        solver.solve_solid_3d_model(
            thermal=bool(thermal),
            thermal_solve=False if thermal else False,
            use_heat_result=True if thermal else False,

            nx=int(solid_nx_eff),
            ny=int(solid_ny_eff),
            nz=int(solid_nz),
            degree=int(solid_degree),

            # Critical: avoid reusing the coarse heat mesh for the solid mechanics solve.
            reuse_heat_mesh_for_solid=False,

            alpha1=alpha1,
            alpha2=alpha2,
            alpha3=alpha3_eff,

            # Keep the heat metadata available if the solid solver ever needs to run heat internally.
            heat_nx=heat_nx,
            heat_ny=heat_ny,
            heat_nz=heat_nz,
            heat_degree=heat_degree,
        )

    def _print_heat_summary(heat, T1_field):
        if heat is None:
            return

        print("\nThermal solve summary:")
        print(f"  T_ref                 : {float(heat['T_ref']):.6f} K")
        print(f"  Heat DoFs             : {int(heat['Vt'].dim())}")
        print(f"  heat_nx, heat_ny, nz  : {_fmt_opt_int(heat_nx)}, {_fmt_opt_int(heat_ny)}, {int(heat_nz)}")
        print(f"  heat_degree           : {int(heat_degree)}")
        print(f"  Nz_quad_T1            : {int(Nz_quad_T1)}")
        print(f"  T1_cg_degree          : {int(T1_cg_degree)}")
        print(f"  Thermal driver stored : {'yes' if T1_field is not None else 'no'}")
        if T1_field is not None:
            print(f"  T1 DoFs               : {int(T1_field.function_space().dim())}")

    def _check_coupled_convergence(out):
        if not bool(out.get("converged", False)) or bool(out.get("stopped_by_max_iters", False)):
            msg = (
                "Coupled thermo-mechanical run did not satisfy the accepted stopping criterion:\n"
                f"  converged            = {out.get('converged', None)}\n"
                f"  stopped_by_max_iters = {out.get('stopped_by_max_iters', None)}\n"
                f"  termination_reason   = {out.get('termination_reason', None)}\n"
                f"  iters                = {out.get('iters', None)}\n"
                f"  err_w                = {out.get('err_w', np.nan):.6e}\n"
                f"  err_T1               = {out.get('err_T1', np.nan):.6e}\n"
            )
            if require_coupled_convergence:
                raise RuntimeError(msg)
            print("\nWARNING:\n" + msg)

    # ------------------------------------------------------------------
    # Main mode runner
    # ------------------------------------------------------------------
    def _run_one(current_mode):
        header = (
            f"--- RUNNING THERMO-MECHANICAL SINGLE TEST "
            f"FOR CASE {case_id} | MODE = {current_mode.upper()} ---"
        )
        print(f"\n{header:=^120}")
        print(f"{f'--- MECHANICAL PARAMETERS (mu_mech) = {test_mu} ---':^120}")

        if current_mode != "mechanical":
            print(f"{f'--- THERMAL PARAMETERS ({season}) (mu_th) = {test_mu_th} ---':^120}")
            heat_disc_msg = (
                f"--- HEAT DISC: nx={heat_nx}, ny={heat_ny}, nz={heat_nz}, "
                f"pT={heat_degree}, Nz_quad_T1={Nz_quad_T1}, "
                f"T1_cg_degree={T1_cg_degree} ---"
            )
            print(f"{heat_disc_msg:^120}")

        print(f"{'=' * 120}\n")

        solver = _make_solver()

        # --------------------------------------------------------------
        # Mechanical only
        # --------------------------------------------------------------
        if current_mode == "mechanical":
            solver.offline_solve_kirchhoff_problem(
                test_mu,
                return_mode="global",
                thermal_on=False,
            )

            if _exact_possible(solver, thermal=False):
                solver.compute_and_visualize_exact_solution(
                    thermal=False,
                    mu=test_mu,
                    num_terms=int(num_terms_exact),
                    exact_degree=int(exact_degree),
                    foundation_branch=foundation_branch,
                )

            _maybe_solve_solid(solver, thermal=False)

            _l2_report(solver, [
                ("Kirchhoff FEM", "w_contact_global"),
                ("3-D Solid", "w_solid_plate"),
                ("Exact", "Exact_solution"),
            ])

            solver.analyze_and_visualize_general(plot_results=plot_results)

        # --------------------------------------------------------------
        # One-way thermo-mechanical
        # --------------------------------------------------------------
        elif current_mode == "one_way":
            solver.set_rom_thermal_parameters(**test_mu_th)

            heat = solver.solve_rom_sample(
                test_mu,
                thermal_on=True,
                coupled_on=False,
                heat_nx=None if heat_nx is None else int(heat_nx),
                heat_ny=None if heat_ny is None else int(heat_ny),
                heat_nz=int(heat_nz),
                heat_degree=int(heat_degree),
                Nz_quad_T1=int(Nz_quad_T1),
                T1_cg_degree=int(T1_cg_degree),
                return_mode="global",
            )

            T1h = getattr(solver, "T1_from_heat", None)
            T1call = _safe_T1_callable(T1h)

            if T1call is not None and _exact_possible(solver, thermal=True):
                solver.compute_and_visualize_exact_solution(
                    thermal=True,
                    mu=test_mu,
                    num_terms=int(num_terms_exact),
                    alpha1=solver.alpha1_rom,
                    alpha2=solver.alpha2_rom,
                    T1=T1call,
                    exact_degree=int(exact_degree),
                    thermal_projection_grid=int(thermal_projection_grid),
                    foundation_branch=foundation_branch,
                )

            _maybe_solve_solid(
                solver,
                thermal=True,
                alpha1=solver.alpha1_rom,
                alpha2=solver.alpha2_rom,
            )

            _print_heat_summary(heat, T1h)

            _l2_report(solver, [
                ("Kirchhoff FEM", "w_contact_global"),
                ("3-D Solid (thermal)", "w_solid_plate_T"),
                ("Exact (thermal)", "Exact_solution_thermal"),
            ])

            _alias_thermal_fields_for_analyzer(solver)
            solver.analyze_and_visualize_general(plot_results=plot_results)

        # --------------------------------------------------------------
        # Coupled thermo-mechanical
        # --------------------------------------------------------------
        elif current_mode == "coupled":
            solver.set_rom_thermal_parameters(**test_mu_th)

            plot_label = season if convergence_plot_label is None else convergence_plot_label

            out = solver.solve_coupled_thermo_mechanical(
                mu_mech=test_mu,
                use_rom_thermal_parameters=True,
                alpha1=solver.alpha1_rom,
                alpha2=solver.alpha2_rom,
                w0=0.0,
                omega=float(coupling_omega),
                tol_w=float(coupling_tol_w),
                tol_T1=float(coupling_tol_T1),
                max_coupling_iters=int(coupling_max_iters),

                heat_nx=None if heat_nx is None else int(heat_nx),
                heat_ny=None if heat_ny is None else int(heat_ny),
                heat_nz=int(heat_nz),
                heat_degree=int(heat_degree),
                Nz_quad_T1=int(Nz_quad_T1),
                T1_cg_degree=int(T1_cg_degree),

                store_each_iter=True,
                verbose=False,
                convergence_plot=convergence_plot,
                convergence_plot_label=plot_label,
                convergence_plot_savebase=convergence_plot_savebase,
            )

            _check_coupled_convergence(out)

            # Make sure reports/plots use the final coupled displacement.
            _install_final_coupled_w_for_reporting(solver, out.get("w", None))

            T1c = getattr(solver, "coupled_T1", None)
            T1call = _safe_T1_callable(T1c)

            if T1call is not None and _exact_possible(solver, thermal=True):
                solver.compute_and_visualize_exact_solution(
                    thermal=True,
                    mu=test_mu,
                    num_terms=int(num_terms_exact),
                    alpha1=solver.alpha1_rom,
                    alpha2=solver.alpha2_rom,
                    T1=T1call,
                    exact_degree=int(exact_degree),
                    thermal_projection_grid=int(thermal_projection_grid),
                    foundation_branch=foundation_branch,
                )

            _maybe_solve_solid(
                solver,
                thermal=True,
                alpha1=solver.alpha1_rom,
                alpha2=solver.alpha2_rom,
            )

            print("\nCoupling summary:")
            print(f"  Iterations            : {out['iters']}")
            print(f"  Converged             : {out['converged']}")
            print(f"  Stopped by max iters  : {out['stopped_by_max_iters']}")
            print(f"  Termination reason    : {out['termination_reason']}")
            print(f"  Stopping norm         : L2_Omega")
            print(f"  Final err_w           : {out['err_w']:.6e}")
            print(f"  Final err_T1          : {out['err_T1']:.6e}")
            print(f"  Max iters budget      : {out['max_coupling_iters']}")
            print(f"  w DoFs                : {out.get('w_dofs', -1)}")
            print(f"  Heat DoFs             : {out.get('heat_dofs', -1)}")
            print(f"  T1 DoFs               : {out.get('T1_dofs', -1)}")
            print(f"  Total coupled DoFs    : {out.get('total_coupled_dofs', -1)}")
            print(f"  heat_degree           : {int(heat_degree)}")
            print(f"  Nz_quad_T1            : {int(Nz_quad_T1)}")
            print(f"  T1_cg_degree          : {int(T1_cg_degree)}")

            if out.get("heat") is not None:
                print(f"  Final T_ref           : {float(out['heat']['T_ref']):.6f} K")

            _l2_report(solver, [
                ("Kirchhoff FEM", "w_contact_global"),
                ("3-D Solid (thermal)", "w_solid_plate_T"),
                ("Exact (thermal)", "Exact_solution_thermal"),
            ])

            _alias_thermal_fields_for_analyzer(solver)
            solver.analyze_and_visualize_general(plot_results=plot_results)

        footer = (
            f"--- THERMO-MECHANICAL SINGLE TEST "
            f"FOR CASE {case_id} | MODE = {current_mode.upper()} COMPLETE ---"
        )
        print(f"\n{footer:=^120}\n")
        return solver

    out = {m: _run_one(m) for m in modes}
    return out[modes[0]] if len(modes) == 1 else out


# ------------------------------------------------------------
# Final solver tolerances for this performance / verification run
# ------------------------------------------------------------
NEWTON_SOLVER_PARAMETERS["newton_solver"].update(
    linear_solver="petsc",
    absolute_tolerance=5e-6,
    relative_tolerance=1e-6,
    maximum_iterations=80,
    relaxation_parameter=1.0,
)


# ------------------------------------------------------------
# Execute final single-test check
# ------------------------------------------------------------
single_test_out = run_single_test_thermomechanical(
    case_id=1,
    mode="coupled",
    season="summer",
    load_type="uniform",
    bc_type="simply_supported",
    n_vert=0,
    n_horiz=0,

    plate_degree=2,

    # Match the final fixed heat controls used in the revised convergence study.
    heat_nx=64,
    heat_ny=32,
    heat_nz=16,
    heat_degree=1,
    Nz_quad_T1=20,
    T1_cg_degree=3,

    # Navier exact-reference controls.
    num_terms_exact=80,
    exact_degree=8,
    thermal_projection_grid=401,
    foundation_branch="compression",

    # Match the final coupled convergence-study settings.
    coupling_omega=0.7,
    coupling_tol_w=1e-6,
    coupling_tol_T1=1e-6,
    coupling_max_iters=40,
    require_coupled_convergence=True,

    # Keep disabled for this exact/Navier check.
    run_solid=False,
    solid_nx=16,
    solid_ny=8,
    solid_nz=6,
    solid_degree=2,

    plot_subdomains=False,
    convergence_plot=True,
    convergence_plot_label="summer, coupled",
    convergence_plot_savebase=None,
    plot_results=True,
)

# %% Cell 8 | id: 3374a1a9-e6e2-4ee3-8fae-e06fc4a1e26d


# %% [markdown] Cell 9 | id: be2036fc-f6ed-482d-9a2f-a6e8a41310dc
# # Convergence Study

# %% Cell 10 | id: 3eab90c7-beb5-4af1-b22f-ef725d87be40
# ============================================================
# Paper 2 thermomechanical FOM convergence
# Initial synchronized coupled-refinement study with coarse reference
# ============================================================
# Intended use:
#   Paste/run this AFTER CODE_THERMOMECHANICAL_FOM.txt has defined
#   GeneralMultiphysicsSolver and the required FEniCS imports.
#
# Study meaning:
#   - coupled-only;
#   - no fixed heat mesh;
#   - plate mesh and 3D heat mesh are refined together;
#   - p = 2 uses heat degree pT = 1, with T1 represented on the p = 2 plate space;
#   - p = 3 uses heat degree pT = 2, with T1 represented on the p = 3 plate space;
#   - initial reference is a coupled FOM with plate N = 32, p = 4,
#     heat mesh 32 x 16 x 8, heat degree pT = 3, and T1 degree q = 4.
#
# Important practical note:
#   This is the correct driver structure for a synchronized coupled refinement
#   study.  The N=32, p=4, heat 32 x 16 x 8, pT=3 reference is intentionally
#   chosen as a first practical reference: it is one refinement level finer than
#   the finest tested level N=16, uses one higher plate degree than the p=3 test
#   branch, and keeps the heat degree one order lower than the plate degree.  For
#   final manuscript-level asymptotic evidence, repeat later with a farther
#   reference such as N=64 or N=100 if computationally feasible.
# ============================================================

# ------------------------------------------------------------
# Global study controls
# ------------------------------------------------------------

PAPER2_OUTDIR = "paper2_initial_sync_coupled_convergence_refN32_pT1_pT2_refpT3"
os.makedirs(PAPER2_OUTDIR, exist_ok=True)

MECHANICAL_BC_TYPE = "simply_supported"

# Synchronized coarse refinement levels requested by the user.
# These are the 2D plate mesh-generation resolutions.
PLATE_RESOLUTIONS = [2, 4, 8, 12, 16]

# Degree pairings for the synchronized coupled study.
# The heat degree is intentionally one order lower than the plate degree:
#   p = 2 uses heat degree pT = 1,
#   p = 3 uses heat degree pT = 2.
# T1_cg_degree is kept equal to the plate degree p, because T1 is represented
# on the 2D plate mesh and enters the Kirchhoff--Love plate thermal moments.
DEGREE_PAIRS = [
    dict(p=2, heat_degree=1, T1_cg_degree=2),
    dict(p=3, heat_degree=2, T1_cg_degree=3),
]
P_VALUES = [d["p"] for d in DEGREE_PAIRS]

# Initial coupled numerical reference.
# This is one synchronized refinement level finer than the finest tested level
# plate N=16 / heat 16 x 8 x 4.  The plate reference uses p=4, the heat
# reference uses pT=3, and the extracted T1 field is represented with qT1=4.
REF_SIZE = 32
REF_P = 4
REF_HEAT_NX = 32
REF_HEAT_NY = 16
REF_HEAT_NZ = 8
REF_HEAT_DEGREE = 3
REF_T1_CG_DEGREE = 4
REF_NZ_QUAD_T1 = 24

# Resume/cache behavior.
RESUME_RUNS = True
FORCE_RERUN = False
RUN_FULL_STUDY = True
RUN_REFERENCE_COMPARISON = False

# Optional: choose only ["Isotropic"] first if you want a lighter starting run.
CASES_TO_RUN = ["Isotropic", "Orthotropic"]

# Optional: stop immediately on the first failed/non-converged run.
# If False, the failed row is stored as reported=False and the loop continues.
STOP_ON_FAILURE = False

REFERENCE_CACHE_VERSION = 13
CONVERGENCE_TABLE_VERSION = 13

STUDY_TAG = (
    f"sync_refN{int(REF_SIZE)}_p{int(REF_P)}"
    f"_heat{int(REF_HEAT_NX)}x{int(REF_HEAT_NY)}x{int(REF_HEAT_NZ)}"
    f"_pT{int(REF_HEAT_DEGREE)}"
    f"_levels{'-'.join(str(v) for v in PLATE_RESOLUTIONS)}"
    f"_p2pT1_p3pT2"
    f"_v{int(CONVERGENCE_TABLE_VERSION)}"
)

PLATE_CASES = {
    "Isotropic": dict(
        mu_mech=[1.0e4, 1.0e4, 0.3e4, 0.35e4, 1.0e3, -10000.0]
    ),
    "Orthotropic": dict(
        mu_mech=[2.0e4, 1.0e4, 0.25e4, 0.30e4, 1.0e3, -10000.0]
    ),
}

THERMAL_PARAMS = dict(
    T_amb=27.0 + 273.15,
    T_sub=24.0 + 273.15,
    h_con=10.0,
    eps_r=0.90,
    q_s=700.0,
    h_c_cont=200.0,
    h_c_gap=5.0,
    eta_c=50.0,
    w_contact=-0.1,
    kx=0.35,
    ky=0.35,
    kz=0.35,
    alpha1=1.3e-4,
    alpha2=1.3e-4,
    rho=1050.0,
)

COUPLING_PARAMS = dict(
    coupling_omega=0.7,
    coupling_tol_w=1e-7,
    coupling_tol_T1=1e-7,
    coupling_max_iters=50,
)

TABLE_COLS = """
mode case bc_type p heat_degree T1_cg_degree size h inv_h
heat_nx heat_ny heat_nz Nz_quad_T1
plate_DoFs heat_DoFs theta_DoFs Total_DoFs
sqrt_plate_DoFs sqrt_Total_DoFs
L2_RMS_Error_w_m Rel_L2_Error_w
L2_RMS_Error_theta Rel_L2_Error_theta
coupled_iters coupled_converged stopped_by_max_iters
termination_reason err_w err_theta stopping_norm reported
""".split()

# Use a slightly looser Newton absolute tolerance than the default to keep the
# coupled campaign robust on coarse and high-order runs.  The coupled stopping is
# still controlled by relative L2(Omega) field updates in solve_coupled_thermo_mechanical.
try:
    NEWTON_SOLVER_PARAMETERS["newton_solver"].update(
        linear_solver="petsc",
        absolute_tolerance=5e-5,
        relative_tolerance=1e-5,
        maximum_iterations=80,
        relaxation_parameter=1.0,
    )
except Exception:
    pass


# ------------------------------------------------------------
# Synchronized heat controls
# ------------------------------------------------------------

def _heat_shape_from_plate_size(size):
    """
    Synchronized 3D heat mesh path tied to the plate resolution.

    The ratio follows the previous physical aspect choice used in the project:
        heat_nx : heat_ny : heat_nz approximately 1 : 1/2 : 1/4.

    For the requested coarse levels this gives:
        plate 2  -> heat 2 x 1 x 1
        plate 4  -> heat 4 x 2 x 1
        plate 8  -> heat 8 x 4 x 2
        plate 12 -> heat 12 x 6 x 3
        plate 16 -> heat 16 x 8 x 4
    """
    n = int(size)
    return dict(
        heat_nx=max(1, n),
        heat_ny=max(1, int(round(n / 2.0))),
        heat_nz=max(1, int(round(n / 4.0))),
    )


def _nz_quad_t1_from_heat_controls(heat_nz, heat_degree, *, ref=False):
    """
    Through-thickness Gauss points for T1 extraction.

    This is not the heat FE degree.  It controls the quadrature used in
    make_T1_from_DeltaT.  It is refined with the heat degree and with heat_nz.
    """
    heat_nz = int(heat_nz)
    heat_degree = int(heat_degree)
    if ref:
        return int(max(REF_NZ_QUAD_T1, 2 * heat_degree + 4, heat_nz + 4))
    return int(max(8, 2 * heat_degree + 4, heat_nz + 2))


def degree_pair_for_p(p):
    for item in DEGREE_PAIRS:
        if int(item["p"]) == int(p):
            return dict(item)
    raise ValueError(f"No synchronized degree pair defined for p={p}. Available: {DEGREE_PAIRS}")


def heat_controls_from_plate_size(size, p=None, *, ref=False):
    """
    Return heat controls for synchronized coupled refinement.

    No fixed heat mesh is used here.
    """
    if ref:
        return dict(
            heat_nx=int(REF_HEAT_NX),
            heat_ny=int(REF_HEAT_NY),
            heat_nz=int(REF_HEAT_NZ),
            heat_degree=int(REF_HEAT_DEGREE),
            Nz_quad_T1=int(_nz_quad_t1_from_heat_controls(
                REF_HEAT_NZ, REF_HEAT_DEGREE, ref=True
            )),
            T1_cg_degree=int(REF_T1_CG_DEGREE),
        )

    if p is None:
        raise ValueError("heat_controls_from_plate_size: p must be supplied for non-reference runs.")

    pair = degree_pair_for_p(p)
    shape = _heat_shape_from_plate_size(size)
    heat_degree = int(pair["heat_degree"])

    return dict(
        heat_nx=int(shape["heat_nx"]),
        heat_ny=int(shape["heat_ny"]),
        heat_nz=int(shape["heat_nz"]),
        heat_degree=int(heat_degree),
        Nz_quad_T1=int(_nz_quad_t1_from_heat_controls(shape["heat_nz"], heat_degree, ref=False)),
        T1_cg_degree=int(pair["T1_cg_degree"]),
    )


def print_refinement_path():
    print("\nSYNCHRONIZED COUPLED REFINEMENT PATH")
    print("Reference:")
    ref_hc = heat_controls_from_plate_size(REF_SIZE, ref=True)
    print(
        f"  plate N={REF_SIZE}, p={REF_P}; "
        f"heat=({ref_hc['heat_nx']},{ref_hc['heat_ny']},{ref_hc['heat_nz']}), "
        f"pT={ref_hc['heat_degree']}, qT1={ref_hc['T1_cg_degree']}, "
        f"Nz_quad_T1={ref_hc['Nz_quad_T1']}"
    )
    print("Test levels:")
    for p in P_VALUES:
        for n in PLATE_RESOLUTIONS:
            hc = heat_controls_from_plate_size(n, p=p, ref=False)
            print(
                f"  p={p}, pT={hc['heat_degree']}, plate N={n}: "
                f"heat=({hc['heat_nx']},{hc['heat_ny']},{hc['heat_nz']}), "
                f"qT1={hc['T1_cg_degree']}, Nz_quad_T1={hc['Nz_quad_T1']}"
            )


# ------------------------------------------------------------
# Solver construction and single coupled solve
# ------------------------------------------------------------

def make_solver(size, p, mu_mech, *, load_type="uniform", bc_type=MECHANICAL_BC_TYPE):
    solver = GeneralMultiphysicsSolver(study_case=1)
    solver.size = int(size)
    solver.degree = int(p)
    solver.load_type = load_type
    solver.bc_type = bc_type
    solver.define_domain(n_vert=0, n_horiz=0, plot_subdomains=False)
    solver.set_mu(mu_mech)
    solver.set_rom_thermal_parameters(**THERMAL_PARAMS)
    return solver


def copy_function(f):
    out = Function(f.function_space())
    out.assign(f)
    return out


def solve_paper2_coupled_case(
    *,
    size,
    p,
    mu_mech,
    heat_nx,
    heat_ny,
    heat_nz,
    heat_degree,
    Nz_quad_T1,
    T1_cg_degree,
    bc_type=MECHANICAL_BC_TYPE,
):
    """Solve one coupled thermomechanical FOM case."""
    solver = make_solver(size=size, p=p, mu_mech=mu_mech, bc_type=bc_type)

    out = solver.solve_rom_sample(
        mu_mech,
        thermal_on=True,
        coupled_on=True,
        heat_nx=int(heat_nx),
        heat_ny=int(heat_ny),
        heat_nz=int(heat_nz),
        heat_degree=int(heat_degree),
        Nz_quad_T1=int(Nz_quad_T1),
        T1_cg_degree=int(T1_cg_degree),
        coupling_omega=COUPLING_PARAMS["coupling_omega"],
        coupling_tol_w=COUPLING_PARAMS["coupling_tol_w"],
        coupling_tol_T1=COUPLING_PARAMS["coupling_tol_T1"],
        coupling_max_iters=COUPLING_PARAMS["coupling_max_iters"],
        coupling_verbose=False,
        w0=0.0,
        return_mode="global",
        return_coupled_dict=True,
    )

    return dict(
        solver=solver,
        mode="coupled",
        bc_type=bc_type,
        w=copy_function(out["w"]),
        theta=copy_function(out["T1"]),
        heat=out["heat"],
        history=out["history"],
        converged=bool(out["converged"]),
        stopped_by_max_iters=bool(out["stopped_by_max_iters"]),
        termination_reason=str(out["termination_reason"]),
        err_w=float(out["err_w"]),
        err_T1=float(out["err_T1"]),
        iters=int(out["iters"]),
        w_dofs=int(out["w_dofs"]),
        heat_dofs=int(out["heat_dofs"]),
        theta_dofs=int(out["T1_dofs"]),
        total_dofs=int(out["total_coupled_dofs"]),
        heat_nx=int(heat_nx),
        heat_ny=int(heat_ny),
        heat_nz=int(heat_nz),
        heat_degree=int(heat_degree),
        Nz_quad_T1=int(Nz_quad_T1),
        T1_cg_degree=int(T1_cg_degree),
    )


# ------------------------------------------------------------
# Error computation against refined reference
# ------------------------------------------------------------

def lift_field_to_reference_mesh(field, Vref, degree=5):
    """Interpolate a scalar plate field from its own mesh onto the reference mesh."""
    try:
        field.set_allow_extrapolation(True)
    except Exception:
        pass

    class FieldOnReferenceMesh(UserExpression):
        def __init__(self, f, **kwargs):
            super().__init__(**kwargs)
            self.f = f

        def eval(self, values, x):
            values[0] = float(self.f(Point(float(x[0]), float(x[1]))))

        def value_shape(self):
            return ()

    return interpolate(FieldOnReferenceMesh(field, degree=int(degree)), Vref)


def l2_rms_error_against_ref(field, ref_field, Vref, sqrt_area, degree=5):
    """RMS L2(Omega) error against the refined coupled numerical FOM reference."""
    field_refmesh = lift_field_to_reference_mesh(field, Vref, degree=degree)
    ref_projected = project(ref_field, Vref)
    return float(errornorm(ref_projected, field_refmesh, norm_type="L2", degree_rise=3) / sqrt_area)


def l2_relative_error_against_ref(field, ref_field, Vref, degree=5, eps=1e-14):
    """Relative L2(Omega) error against the refined coupled numerical FOM reference."""
    dx_ref = Measure("dx", domain=Vref.mesh())
    field_refmesh = lift_field_to_reference_mesh(field, Vref, degree=degree)
    ref_projected = project(ref_field, Vref)

    abs_err = float(errornorm(ref_projected, field_refmesh, norm_type="L2", degree_rise=3))
    ref_norm = float(np.sqrt(max(float(assemble(ref_projected * ref_projected * dx_ref)), 0.0)))
    return abs_err / max(ref_norm, float(eps))


# ------------------------------------------------------------
# Checkpoint helpers
# ------------------------------------------------------------

def _safe_filename_part(text):
    text = str(text)
    for ch in [" ", "/", "\\", ":", ";", ",", "(", ")", "[", "]", "{", "}"]:
        text = text.replace(ch, "_")
    return text


def convergence_csv_path(label, bc_type):
    return os.path.join(
        PAPER2_OUTDIR,
        f"paper2_coupled_{_safe_filename_part(label)}_{_safe_filename_part(bc_type)}_convergence_{STUDY_TAG}.csv",
    )


def _as_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return bool(default)
        return bool(int(value))
    if isinstance(value, str):
        value_l = value.strip().lower()
        if value_l in ("true", "t", "1", "yes", "y"):
            return True
        if value_l in ("false", "f", "0", "no", "n", "nan", "none", ""):
            return False
    return bool(default)


def _bool_series(series, default=False):
    return series.apply(lambda v: _as_bool(v, default=default)).astype(bool)


def _row_is_valid_converged_coupled(row):
    try:
        if str(row.get("mode", "")) != "coupled":
            return False
        if not _as_bool(row.get("coupled_converged", False)):
            return False
        if _as_bool(row.get("stopped_by_max_iters", False)):
            return False
        if "reported" in row.index and not _as_bool(row.get("reported", True), default=True):
            return False
        if "stopping_norm" in row.index and str(row.get("stopping_norm", "")) != "L2_Omega":
            return False

        finite_cols = [
            "h",
            "L2_RMS_Error_w_m",
            "Rel_L2_Error_w",
            "L2_RMS_Error_theta",
            "Rel_L2_Error_theta",
            "err_w",
            "err_theta",
        ]
        return all(col not in row.index or np.isfinite(float(row[col])) for col in finite_cols)
    except Exception:
        return False


def valid_converged_coupled_rows(df):
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out = out[out["mode"].astype(str) == "coupled"]

    if "coupled_converged" in out.columns:
        out = out[_bool_series(out["coupled_converged"], default=False)]
    if "stopped_by_max_iters" in out.columns:
        out = out[~_bool_series(out["stopped_by_max_iters"], default=False)]
    if "reported" in out.columns:
        out = out[_bool_series(out["reported"], default=True)]
    if "stopping_norm" in out.columns:
        out = out[out["stopping_norm"].astype(str) == "L2_Omega"]

    for col in [
        "h",
        "L2_RMS_Error_w_m",
        "Rel_L2_Error_w",
        "L2_RMS_Error_theta",
        "Rel_L2_Error_theta",
        "err_w",
        "err_theta",
    ]:
        if col in out.columns:
            out = out[np.isfinite(out[col].to_numpy(dtype=float))]

    return out.reset_index(drop=True)


def load_existing_convergence_rows(out_csv):
    if not (RESUME_RUNS and os.path.exists(out_csv) and not FORCE_RERUN):
        return None, set()

    df = pd.read_csv(out_csv)
    if df.empty:
        return None, set()

    required_key_cols = ["mode", "case", "bc_type", "p", "heat_degree", "size"]
    valid_keys = {
        (
            str(row["mode"]),
            str(row["case"]),
            str(row["bc_type"]),
            int(row["p"]),
            int(row["heat_degree"]),
            int(row["size"]),
        )
        for _, row in df.iterrows()
        if all(col in row.index for col in required_key_cols) and _row_is_valid_converged_coupled(row)
    }

    print(f"\nResuming from existing table: {out_csv}")
    print(f"Existing rows: {len(df)}")
    print(f"Valid converged coupled rows that will be skipped: {len(valid_keys)}")
    return df, valid_keys


def save_convergence_checkpoint(new_records, existing_df, out_csv):
    if not new_records:
        return existing_df if existing_df is not None else pd.DataFrame()

    new_df = pd.DataFrame.from_records(new_records)
    save_df = pd.concat([existing_df, new_df], ignore_index=True) if existing_df is not None and not existing_df.empty else new_df

    save_df = (
        save_df.drop_duplicates(["mode", "case", "bc_type", "p", "heat_degree", "size"], keep="last")
        .sort_values(["mode", "case", "bc_type", "p", "heat_degree", "h"], ascending=[True, True, True, True, True, False])
        .reset_index(drop=True)
    )
    save_df.to_csv(out_csv, index=False)
    print(f"Checkpoint saved: {out_csv}")
    return save_df


# ------------------------------------------------------------
# Reference cache helpers
# ------------------------------------------------------------

def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def reference_cache_key(*, label, bc_type, ref_size, ref_p, mu_mech, ref_heat):
    payload = dict(
        cache_version=int(REFERENCE_CACHE_VERSION),
        convergence_table_version=int(CONVERGENCE_TABLE_VERSION),
        study_tag=str(STUDY_TAG),
        mode="coupled",
        case=str(label),
        bc_type=str(bc_type),
        ref_size=int(ref_size),
        ref_p=int(ref_p),
        mu_mech=[float(v) for v in mu_mech],
        thermal_params=_json_safe(THERMAL_PARAMS),
        ref_heat=_json_safe(ref_heat),
        coupling_params=_json_safe(COUPLING_PARAMS),
        plate_resolutions=[int(v) for v in PLATE_RESOLUTIONS],
        degree_pairs=_json_safe(DEGREE_PAIRS),
        synchronized_refinement=True,
        fixed_heat=False,
        mechanical_bc_type=str(MECHANICAL_BC_TYPE),
        stopping_norm="L2_Omega",
        reported_refinement_branch="coupled_only_synchronized_full_asymptotic",
    )
    payload_json = json.dumps(payload, sort_keys=True, indent=2)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:16], payload


def reference_cache_paths(*, label, bc_type, ref_size, ref_p, cache_key):
    stem = (
        f"paper2_ref_coupled_{_safe_filename_part(label)}"
        f"_{_safe_filename_part(bc_type)}_size{int(ref_size)}_p{int(ref_p)}_{cache_key}"
    )
    return dict(
        h5=os.path.join(PAPER2_OUTDIR, stem + ".h5"),
        json=os.path.join(PAPER2_OUTDIR, stem + ".json"),
    )


def _assert_converged_coupled_record(record, *, context):
    if not bool(record.get("converged", False)):
        raise RuntimeError(
            f"{context} did not converge: "
            f"iters={record.get('iters', None)}, "
            f"err_w={record.get('err_w', np.nan)}, "
            f"err_T1={record.get('err_T1', np.nan)}, "
            f"termination={record.get('termination_reason', None)}"
        )
    if bool(record.get("stopped_by_max_iters", False)):
        raise RuntimeError(
            f"{context} reached max coupling iterations and must not be reported: "
            f"iters={record.get('iters', None)}, "
            f"err_w={record.get('err_w', np.nan)}, "
            f"err_T1={record.get('err_T1', np.nan)}, "
            f"termination={record.get('termination_reason', None)}"
        )


def _reference_meta(ref, w_save, theta_save, payload):
    return dict(
        payload=payload,
        history=ref.get("history", None),
        converged=bool(ref.get("converged", True)),
        stopped_by_max_iters=bool(ref.get("stopped_by_max_iters", False)),
        termination_reason=str(ref.get("termination_reason", "cached_reference")),
        err_w=float(ref.get("err_w", np.nan)),
        err_T1=float(ref.get("err_T1", np.nan)),
        iters=int(ref.get("iters", 1)),
        w_dofs=int(ref.get("w_dofs", w_save.function_space().dim())),
        heat_dofs=int(ref.get("heat_dofs", 0)),
        theta_dofs=int(ref.get("theta_dofs", theta_save.function_space().dim())),
        total_dofs=int(ref.get("total_dofs", 0)),
        heat_nx=int(ref.get("heat_nx", payload["ref_heat"]["heat_nx"])),
        heat_ny=int(ref.get("heat_ny", payload["ref_heat"]["heat_ny"])),
        heat_nz=int(ref.get("heat_nz", payload["ref_heat"]["heat_nz"])),
        heat_degree=int(ref.get("heat_degree", payload["ref_heat"]["heat_degree"])),
        Nz_quad_T1=int(ref.get("Nz_quad_T1", payload["ref_heat"]["Nz_quad_T1"])),
        T1_cg_degree=int(ref.get("T1_cg_degree", payload["ref_heat"]["T1_cg_degree"])),
    )


def save_reference_cache(ref, *, paths, payload, ref_p):
    _assert_converged_coupled_record(ref, context="Coupled reference before cache save")

    mesh = ref["solver"].mesh
    Vw_ref = FunctionSpace(mesh, "CG", int(ref_p))
    Vtheta_ref = FunctionSpace(mesh, "CG", int(ref_p))

    w_save = project(ref["w"], Vw_ref)
    theta_save = project(ref["theta"], Vtheta_ref)

    h5 = HDF5File(mesh.mpi_comm(), paths["h5"], "w")
    h5.write(mesh, "/mesh")
    h5.write(w_save, "/w")
    h5.write(theta_save, "/theta")
    h5.close()

    with open(paths["json"], "w") as f:
        json.dump(_json_safe(_reference_meta(ref, w_save, theta_save, payload)), f, indent=2, sort_keys=True)

    print("Saved reference cache:")
    print(f"  {paths['h5']}")
    print(f"  {paths['json']}")


def load_reference_cache(*, paths, ref_p):
    if not (os.path.exists(paths["h5"]) and os.path.exists(paths["json"])):
        return None

    try:
        mesh = Mesh()
        h5 = HDF5File(MPI.comm_world, paths["h5"], "r")
        h5.read(mesh, "/mesh", False)

        Vw_ref = FunctionSpace(mesh, "CG", int(ref_p))
        Vtheta_ref = FunctionSpace(mesh, "CG", int(ref_p))
        w_ref = Function(Vw_ref)
        theta_ref = Function(Vtheta_ref)

        h5.read(w_ref, "/w")
        h5.read(theta_ref, "/theta")
        h5.close()

        with open(paths["json"], "r") as f:
            meta = json.load(f)
    except Exception as exc:
        print(f"Reference cache could not be loaded and will be rebuilt: {exc}")
        return None

    ref = dict(
        solver=None,
        mesh=mesh,
        w=w_ref,
        theta=theta_ref,
        heat=None,
        history=meta.get("history", None),
        converged=bool(meta.get("converged", True)),
        stopped_by_max_iters=bool(meta.get("stopped_by_max_iters", False)),
        termination_reason=str(meta.get("termination_reason", "loaded_reference_cache")),
        err_w=float(meta.get("err_w", np.nan)),
        err_T1=float(meta.get("err_T1", np.nan)),
        iters=int(meta.get("iters", 1)),
        w_dofs=int(meta.get("w_dofs", w_ref.function_space().dim())),
        heat_dofs=int(meta.get("heat_dofs", 0)),
        theta_dofs=int(meta.get("theta_dofs", theta_ref.function_space().dim())),
        total_dofs=int(meta.get("total_dofs", 0)),
        heat_nx=int(meta.get("heat_nx", -1)),
        heat_ny=int(meta.get("heat_ny", -1)),
        heat_nz=int(meta.get("heat_nz", -1)),
        heat_degree=int(meta.get("heat_degree", -1)),
        Nz_quad_T1=int(meta.get("Nz_quad_T1", -1)),
        T1_cg_degree=int(meta.get("T1_cg_degree", -1)),
    )

    _assert_converged_coupled_record(ref, context="Loaded coupled reference cache")
    print("Loaded cached reference:")
    print(f"  {paths['h5']}")
    return ref


def get_or_build_reference(*, label, bc_type, ref_size, ref_p, mu_mech):
    """Build/load the synchronized high-resolution coupled reference."""
    ref_heat = heat_controls_from_plate_size(ref_size, ref=True)
    cache_key, payload = reference_cache_key(
        label=label,
        bc_type=bc_type,
        ref_size=ref_size,
        ref_p=ref_p,
        mu_mech=mu_mech,
        ref_heat=ref_heat,
    )
    paths = reference_cache_paths(
        label=label,
        bc_type=bc_type,
        ref_size=ref_size,
        ref_p=ref_p,
        cache_key=cache_key,
    )

    if RESUME_RUNS and not FORCE_RERUN:
        cached = load_reference_cache(paths=paths, ref_p=ref_p)
        if cached is not None:
            return cached

    print("\n" + "=" * 100)
    print(
        f"BUILDING SYNCHRONIZED COUPLED REFERENCE: {label}, bc={bc_type}, "
        f"plate N={ref_size}, p={ref_p}, "
        f"heat=({ref_heat['heat_nx']},{ref_heat['heat_ny']},{ref_heat['heat_nz']}), "
        f"pT={ref_heat['heat_degree']}, qT1={ref_heat['T1_cg_degree']}, "
        f"Nz_quad_T1={ref_heat['Nz_quad_T1']}"
    )
    print("=" * 100)

    ref = solve_paper2_coupled_case(
        size=ref_size,
        p=ref_p,
        mu_mech=mu_mech,
        heat_nx=ref_heat["heat_nx"],
        heat_ny=ref_heat["heat_ny"],
        heat_nz=ref_heat["heat_nz"],
        heat_degree=ref_heat["heat_degree"],
        Nz_quad_T1=ref_heat["Nz_quad_T1"],
        T1_cg_degree=ref_heat["T1_cg_degree"],
        bc_type=bc_type,
    )

    _assert_converged_coupled_record(ref, context="Newly built synchronized coupled reference")
    save_reference_cache(ref, paths=paths, payload=payload, ref_p=ref_p)
    return ref


def reference_mesh(ref):
    return ref["solver"].mesh if ref.get("solver", None) is not None else ref["mesh"]


# ------------------------------------------------------------
# Records and convergence gathering
# ------------------------------------------------------------

def _coupled_solution_is_converged(record):
    return (
        str(record.get("mode", "coupled")) == "coupled"
        and bool(record.get("converged", False))
        and not bool(record.get("stopped_by_max_iters", False))
    )


def _success_record(label, bc_type, p, res, sol, errors):
    ew_rms, etheta_rms, ew_rel, etheta_rel = errors
    h_val = float(sol["solver"].mesh.hmax())
    return dict(
        mode="coupled",
        case=str(label),
        bc_type=str(bc_type),
        p=int(p),
        heat_degree=int(sol["heat_degree"]),
        T1_cg_degree=int(sol["T1_cg_degree"]),
        size=int(res),
        h=h_val,
        inv_h=float(1.0 / h_val),
        heat_nx=int(sol["heat_nx"]),
        heat_ny=int(sol["heat_ny"]),
        heat_nz=int(sol["heat_nz"]),
        Nz_quad_T1=int(sol["Nz_quad_T1"]),
        plate_DoFs=int(sol["w_dofs"]),
        heat_DoFs=int(sol["heat_dofs"]),
        theta_DoFs=int(sol["theta_dofs"]),
        Total_DoFs=int(sol["total_dofs"]),
        sqrt_plate_DoFs=float(np.sqrt(sol["w_dofs"])),
        sqrt_Total_DoFs=float(np.sqrt(sol["total_dofs"])),
        L2_RMS_Error_w_m=float(ew_rms),
        L2_RMS_Error_theta=float(etheta_rms),
        Rel_L2_Error_w=float(ew_rel),
        Rel_L2_Error_theta=float(etheta_rel),
        coupled_iters=int(sol["iters"]),
        coupled_converged=bool(sol["converged"]),
        stopped_by_max_iters=bool(sol["stopped_by_max_iters"]),
        termination_reason=str(sol["termination_reason"]),
        err_w=float(sol["err_w"]),
        err_theta=float(sol["err_T1"]),
        stopping_norm="L2_Omega",
        reported=True,
    )


def _failed_record(label, bc_type, p, res, hc, exc):
    return dict(
        mode="coupled",
        case=str(label),
        bc_type=str(bc_type),
        p=int(p),
        heat_degree=int(hc["heat_degree"]),
        T1_cg_degree=int(hc["T1_cg_degree"]),
        size=int(res),
        h=np.nan,
        inv_h=np.nan,
        heat_nx=int(hc["heat_nx"]),
        heat_ny=int(hc["heat_ny"]),
        heat_nz=int(hc["heat_nz"]),
        Nz_quad_T1=int(hc["Nz_quad_T1"]),
        plate_DoFs=np.nan,
        heat_DoFs=np.nan,
        theta_DoFs=np.nan,
        Total_DoFs=np.nan,
        sqrt_plate_DoFs=np.nan,
        sqrt_Total_DoFs=np.nan,
        L2_RMS_Error_w_m=np.nan,
        L2_RMS_Error_theta=np.nan,
        Rel_L2_Error_w=np.nan,
        Rel_L2_Error_theta=np.nan,
        coupled_iters=np.nan,
        coupled_converged=False,
        stopped_by_max_iters=False,
        termination_reason=f"failed_or_not_reported: {exc}",
        err_w=np.nan,
        err_theta=np.nan,
        stopping_norm="L2_Omega",
        reported=False,
    )


def _excluded_record_from_solution(label, bc_type, p, res, sol):
    h_val = float(sol["solver"].mesh.hmax())
    return dict(
        mode="coupled",
        case=str(label),
        bc_type=str(bc_type),
        p=int(p),
        heat_degree=int(sol["heat_degree"]),
        T1_cg_degree=int(sol["T1_cg_degree"]),
        size=int(res),
        h=h_val,
        inv_h=float(1.0 / h_val),
        heat_nx=int(sol["heat_nx"]),
        heat_ny=int(sol["heat_ny"]),
        heat_nz=int(sol["heat_nz"]),
        Nz_quad_T1=int(sol["Nz_quad_T1"]),
        plate_DoFs=int(sol["w_dofs"]),
        heat_DoFs=int(sol["heat_dofs"]),
        theta_DoFs=int(sol["theta_dofs"]),
        Total_DoFs=int(sol["total_dofs"]),
        sqrt_plate_DoFs=float(np.sqrt(sol["w_dofs"])),
        sqrt_Total_DoFs=float(np.sqrt(sol["total_dofs"])),
        L2_RMS_Error_w_m=np.nan,
        L2_RMS_Error_theta=np.nan,
        Rel_L2_Error_w=np.nan,
        Rel_L2_Error_theta=np.nan,
        coupled_iters=int(sol["iters"]),
        coupled_converged=bool(sol["converged"]),
        stopped_by_max_iters=bool(sol["stopped_by_max_iters"]),
        termination_reason=f"not_reported_nonconverged: {sol['termination_reason']}",
        err_w=float(sol["err_w"]),
        err_theta=float(sol["err_T1"]),
        stopping_norm="L2_Omega",
        reported=False,
    )


def gather_full_asymptotic_coupled_convergence_data(
    *,
    mu_mech,
    resolutions=PLATE_RESOLUTIONS,
    degree_pairs=DEGREE_PAIRS,
    ref_size=REF_SIZE,
    ref_p=REF_P,
    label="case",
    bc_type=MECHANICAL_BC_TYPE,
):
    """
    Gather synchronized coupled-refinement data against a high-resolution coupled FOM reference.

    Non-converged coupled rows are saved for traceability but excluded from plots/tables.
    """
    out_csv = convergence_csv_path(label, bc_type)
    existing_df, existing_keys = load_existing_convergence_rows(out_csv)

    ref = get_or_build_reference(
        label=label,
        bc_type=bc_type,
        ref_size=ref_size,
        ref_p=ref_p,
        mu_mech=mu_mech,
    )
    _assert_converged_coupled_record(ref, context=f"Synchronized coupled reference for {label}, bc={bc_type}")

    ref_mesh_obj = reference_mesh(ref)
    Vw_ref = FunctionSpace(ref_mesh_obj, "CG", int(ref_p))
    Vtheta_ref = FunctionSpace(ref_mesh_obj, "CG", int(ref_p))
    dx_ref = Measure("dx", domain=ref_mesh_obj)
    sqrt_area = float(np.sqrt(assemble(Constant(1.0) * dx_ref)))

    w_ref = project(ref["w"], Vw_ref)
    theta_ref = project(ref["theta"], Vtheta_ref)

    for pair in degree_pairs:
        p = int(pair["p"])
        for res in resolutions:
            case_key = ("coupled", label, bc_type, int(p), int(pair["heat_degree"]), int(res))
            if case_key in existing_keys:
                print(
                    f"SKIPPING valid cached row: case={label}, bc={bc_type}, "
                    f"size={res}, p={p}, pT={pair['heat_degree']}"
                )
                continue

            hc = heat_controls_from_plate_size(res, p=p, ref=False)
            print("\n" + "-" * 100)
            print(
                f"COUPLED SYNC | {label} | bc={bc_type} | "
                f"plate size={res}, p={p} | "
                f"heat=({hc['heat_nx']},{hc['heat_ny']},{hc['heat_nz']}), "
                f"pT={hc['heat_degree']}, qT1={hc['T1_cg_degree']}, "
                f"Nz_quad_T1={hc['Nz_quad_T1']}"
            )
            print("-" * 100)

            try:
                sol = solve_paper2_coupled_case(
                    size=res,
                    p=p,
                    mu_mech=mu_mech,
                    heat_nx=hc["heat_nx"],
                    heat_ny=hc["heat_ny"],
                    heat_nz=hc["heat_nz"],
                    heat_degree=hc["heat_degree"],
                    Nz_quad_T1=hc["Nz_quad_T1"],
                    T1_cg_degree=hc["T1_cg_degree"],
                    bc_type=bc_type,
                )

                if not _coupled_solution_is_converged(sol):
                    msg = (
                        f"EXCLUDED NON-CONVERGED RUN: case={label}, size={res}, p={p}, "
                        f"pT={hc['heat_degree']}, iters={sol.get('iters', None)}, "
                        f"err_w={sol.get('err_w', np.nan)}, err_T1={sol.get('err_T1', np.nan)}, "
                        f"termination={sol.get('termination_reason', None)}"
                    )
                    print(msg)
                    if STOP_ON_FAILURE:
                        raise RuntimeError(msg)
                    record = _excluded_record_from_solution(label, bc_type, p, res, sol)

                else:
                    deg = max(5, int(p) + 2, int(hc["T1_cg_degree"]) + 2)
                    errors = (
                        l2_rms_error_against_ref(sol["w"], w_ref, Vw_ref, sqrt_area, degree=deg),
                        l2_rms_error_against_ref(sol["theta"], theta_ref, Vtheta_ref, sqrt_area, degree=deg),
                        l2_relative_error_against_ref(sol["w"], w_ref, Vw_ref, degree=deg),
                        l2_relative_error_against_ref(sol["theta"], theta_ref, Vtheta_ref, degree=deg),
                    )
                    record = _success_record(label, bc_type, p, res, sol, errors)

            except Exception as exc:
                print(f"FAILED OR EXCLUDED: case={label}, bc={bc_type}, size={res}, p={p}: {exc}")
                if STOP_ON_FAILURE:
                    raise
                record = _failed_record(label, bc_type, p, res, hc, exc)

            existing_df = save_convergence_checkpoint([record], existing_df, out_csv)

            if _row_is_valid_converged_coupled(pd.Series(record)):
                existing_keys.add(case_key)

    df = existing_df.copy() if existing_df is not None and not existing_df.empty else pd.DataFrame()

    if not df.empty:
        df = df.sort_values(
            ["mode", "case", "bc_type", "p", "heat_degree", "h"],
            ascending=[True, True, True, True, True, False],
        ).reset_index(drop=True)
        df.to_csv(out_csv, index=False)

    df_valid = valid_converged_coupled_rows(df)

    print(f"\nSaved convergence table: {out_csv}")
    print(f"Total rows in CSV: {len(df)}")
    print(f"Valid converged coupled rows for reporting: {len(df_valid)}")
    return df, ref


# ------------------------------------------------------------
# Tables and rate summaries
# ------------------------------------------------------------

def finest_converged_rows_for_table(df):
    df_valid = valid_converged_coupled_rows(df)
    if df_valid.empty:
        return df_valid
    rows = [grp.sort_values("size", ascending=False).iloc[0] for _, grp in df_valid.groupby(["case", "p", "heat_degree"])]
    return pd.DataFrame(rows).sort_values(["case", "p", "heat_degree"]).reset_index(drop=True)


def print_finest_converged_table(df, title="FINEST CONVERGED COUPLED ROWS"):
    table_df = finest_converged_rows_for_table(df)
    print("\n" + title)
    if table_df.empty:
        print("No valid converged coupled rows available.")
        return table_df

    existing_cols = [c for c in TABLE_COLS if c in table_df.columns]
    print(table_df[existing_cols].to_string(index=False, float_format="%.6e"))
    return table_df


def compute_rate_summary(df, *, ycols=("L2_RMS_Error_w_m", "L2_RMS_Error_theta")):
    df_valid = valid_converged_coupled_rows(df)
    records = []
    if df_valid.empty:
        return pd.DataFrame()

    for (case, p, pT), grp in df_valid.groupby(["case", "p", "heat_degree"]):
        grp = grp.sort_values("h", ascending=False)
        x = grp["h"].to_numpy(float)
        for ycol in ycols:
            y = grp[ycol].to_numpy(float)
            mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
            xx, yy = x[mask], y[mask]
            rate_all = np.nan
            rate_last3 = np.nan
            if len(xx) >= 2:
                rate_all = abs(float(np.polyfit(np.log10(xx), np.log10(yy), 1)[0]))
            if len(xx) >= 3:
                rate_last3 = abs(float(np.polyfit(np.log10(xx[-3:]), np.log10(yy[-3:]), 1)[0]))
            records.append(dict(
                case=str(case),
                p=int(p),
                heat_degree=int(pT),
                quantity=str(ycol),
                n_points=int(len(xx)),
                rate_all=float(rate_all),
                rate_last3=float(rate_last3),
            ))
    return pd.DataFrame.from_records(records)


def print_rate_summary(df):
    rs = compute_rate_summary(df)
    print("\nLOG-LOG RATE SUMMARY")
    if rs.empty:
        print("No valid rate data.")
    else:
        print(rs.to_string(index=False, float_format="%.4f"))
    return rs


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def plot_paper2_synchronized_coupled_convergence(
    df_coupled,
    *,
    ycol="L2_RMS_Error_w_m",
    ylabel=r"RMS $L^2(\Omega)$ error in $w_h$ [m]",
    x_mode="h",
    filename_base=None,
    rate_fit_slice=slice(None),
):
    """
    Synchronized coupled convergence plot.

    Only valid converged coupled rows are plotted.  For x_mode='h', the x-axis is
    shown coarse-to-fine from left to right, i.e. larger h on the left.
    """

    if filename_base is None:
        filename_base = f"paper2_sync_coupled_{ycol}_{STUDY_TAG}"

    df = valid_converged_coupled_rows(df_coupled)
    if df.empty:
        raise ValueError("No valid converged coupled rows available for plotting.")

    try:
        plt.style.use(["science", "ieee", "notebook", "grid"])
    except Exception:
        pass
    plt.rcParams["figure.autolayout"] = False

    x_options = {
        "h": ("h", r"Characteristic plate mesh length $\ell_{\mathcal{T}}$ [m]"),
        "inv_h": ("inv_h", r"Inverse plate mesh size $1/h$"),
        "sqrt_plate_dofs": ("sqrt_plate_DoFs", r"$\sqrt{\mathrm{plate\ DoFs}}$"),
        "plate_dofs": ("plate_DoFs", "Plate degrees of freedom"),
        "sqrt_total_dofs": ("sqrt_Total_DoFs", r"$\sqrt{\mathrm{total\ DoFs}}$"),
        "total_dofs": ("Total_DoFs", "Total coupled degrees of freedom"),
    }
    if x_mode not in x_options:
        raise ValueError(f"x_mode must be one of {list(x_options.keys())}")
    xcol, xlabel = x_options[x_mode]

    style_specs = [
        ("Isotropic", 2, 1, "black", "-", "o", r"Isotropic, $p=2$, $p_T=1$"),
        ("Isotropic", 3, 2, "blue", "--", "s", r"Isotropic, $p=3$, $p_T=2$"),
        ("Orthotropic", 2, 1, "darkorange", "-", "o", r"Orthotropic, $p=2$, $p_T=1$"),
        ("Orthotropic", 3, 2, "purple", "--", "s", r"Orthotropic, $p=3$, $p_T=2$"),
    ]
    styles = {
        (case_name, p, pT): dict(color=color, linestyle=linestyle, marker=marker, label=label)
        for case_name, p, pT, color, linestyle, marker, label in style_specs
    }

    def positive_finite(vals):
        vals = np.asarray(vals, dtype=float)
        return vals[np.isfinite(vals) & (vals > 0.0)]

    def ordered_xy(x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
        x, y = x[mask], y[mask]
        order = np.argsort(x)[::-1] if x_mode == "h" else np.argsort(x)
        return x[order], y[order]

    def estimate_loglog_slope(x, y, fit_slice=slice(None)):
        x, y = ordered_xy(x, y)
        if len(x) < 2:
            return None
        xx, yy = x[fit_slice], y[fit_slice]
        if len(xx) < 2:
            return None
        slope, _ = np.polyfit(np.log10(xx), np.log10(yy), 1)
        return float(abs(slope))

    def add_rate_triangle(ax, x, y, rate, color, *, location="above", start_idx=-2, span_frac=0.42, tri_gap=1.25):
        if rate is None or rate <= 0.0:
            return
        x, y = ordered_xy(x, y)
        if len(x) < 2:
            return
        start_idx = len(x) + start_idx if start_idx < 0 else start_idx
        if start_idx < 0 or start_idx + 1 >= len(x):
            return

        x_prev = float(x[start_idx])
        x_last = float(x[start_idx + 1])
        y_last = float(y[start_idx + 1])
        if x_prev <= 0 or x_last <= 0 or y_last <= 0:
            return

        if x_mode == "h":
            full_ratio = x_prev / x_last
            if full_ratio <= 1:
                return
            x_right = x_last
            x_left = x_last * (full_ratio ** span_frac)
            local_ratio = x_left / x_right
            x_rate = x_right / (local_ratio ** 0.03)
        else:
            full_ratio = x_last / x_prev
            if full_ratio <= 1:
                return
            x_left = x_last / (full_ratio ** span_frac)
            x_right = x_last
            local_ratio = x_right / x_left
            x_rate = x_right * (local_ratio ** 0.03)

        if location == "below":
            y_top = y_last / tri_gap
            y_bottom = y_top / (local_ratio ** rate)
        else:
            y_bottom = y_last * tri_gap
            y_top = y_bottom * (local_ratio ** rate)

        if min(x_left, x_right, y_top, y_bottom) <= 0:
            return

        text_box = dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.12)
        ax.plot([x_left, x_right], [y_top, y_top], color=color, lw=1.0, zorder=5)
        ax.plot([x_right, x_right], [y_top, y_bottom], color=color, lw=1.0, zorder=5)
        ax.plot([x_left, x_right], [y_top, y_bottom], color=color, lw=1.0, zorder=5)
        ax.text(np.sqrt(x_left * x_right), y_top * 1.06, "1", ha="center", va="bottom", fontsize=8, color=color, bbox=text_box, zorder=6)
        ax.text(x_rate, np.sqrt(y_top * y_bottom), f"{rate:.1f}", ha="left", va="center", fontsize=8, color=color, bbox=text_box, zorder=6)

    all_x = positive_finite(df[xcol].to_numpy(float))
    all_y = positive_finite(df[ycol].to_numpy(float))
    if len(all_x) == 0 or len(all_y) == 0:
        raise ValueError("No positive finite data available for plotting.")

    fig, ax = plt.subplots(figsize=(7.0, 5.3), facecolor="white")
    grouped = {}

    for key, style in styles.items():
        case_name, p, pT = key
        grp = df[
            (df["case"].astype(str) == case_name)
            & (df["p"].astype(int) == int(p))
            & (df["heat_degree"].astype(int) == int(pT))
        ].dropna(subset=[xcol, ycol]).copy()

        if grp.empty:
            continue

        grp = grp.sort_values(xcol, ascending=(x_mode != "h"))
        grouped[key] = grp

        ax.loglog(
            grp[xcol].to_numpy(float),
            grp[ycol].to_numpy(float),
            linestyle=style["linestyle"],
            marker=style["marker"],
            color=style["color"],
            markeredgecolor=style["color"],
            markerfacecolor="none",
            linewidth=2.2,
            markersize=8.0,
            markeredgewidth=1.4,
            label=style["label"],
            clip_on=False,
        )

    ax.set_facecolor("white")
    ax.set_title(r"Synchronized coupled refinement study", fontsize=14, pad=8)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=13, length=4, pad=2)
    ax.tick_params(axis="both", which="minor", length=2)
    ax.grid(True, which="both", linestyle=":", linewidth=0.45, alpha=0.35)

    if x_mode == "h":
        ticks = []
        for size in PLATE_RESOLUTIONS:
            vals = positive_finite(df.loc[df["size"].astype(int) == int(size), "h"].to_numpy(float))
            if len(vals):
                ticks.append(float(np.median(vals)))
        if ticks:
            ticks = sorted(set(np.round(ticks, 12)), reverse=True)
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.2f}" if x >= 0.1 else f"{x:.3f}"))
    else:
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=80))
    ax.xaxis.set_minor_formatter(NullFormatter())

    if x_mode == "h":
        ax.set_xlim(np.max(all_x) * 1.20, np.min(all_x) / 1.14)
    else:
        ax.set_xlim(np.min(all_x) / 1.14, np.max(all_x) * 1.20)
    ax.set_ylim(np.min(all_y) / 2.60, np.max(all_y) * 1.85)

    for idx, (key, grp) in enumerate(grouped.items()):
        style = styles[key]
        rate = estimate_loglog_slope(grp[xcol].to_numpy(float), grp[ycol].to_numpy(float), fit_slice=rate_fit_slice)
        location = "above" if idx % 2 == 0 else "below"
        add_rate_triangle(ax, grp[xcol].to_numpy(float), grp[ycol].to_numpy(float), rate, style["color"], location=location)

    leg = ax.legend(
        loc="lower left",
        fontsize=11.2,
        frameon=True,
        borderpad=0.55,
        handlelength=2.2,
        handletextpad=0.55,
        labelspacing=0.35,
    )
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.8)

    os.makedirs(PAPER2_OUTDIR, exist_ok=True)
    for ext, dpi in (("pdf", 300), ("png", 600)):
        fig.savefig(
            os.path.join(PAPER2_OUTDIR, f"{filename_base}.{ext}"),
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
        )
    plt.show()


def history_dataframe_from_ref(ref_or_solution, case_label):
    hist = pd.DataFrame(ref_or_solution.get("history", None)).copy()
    if hist.empty:
        return hist
    hist["case"] = str(case_label)
    return hist


def plot_paper2_coupled_history_iso_ortho_combined(histories, *, filename_base=None):
    if filename_base is None:
        filename_base = f"paper2_sync_coupled_reference_update_history_{STUDY_TAG}"
    if not histories:
        print("No coupled histories available.")
        return

    hist = pd.concat(histories, ignore_index=True)
    if hist.empty:
        print("No coupled histories available.")
        return

    try:
        plt.style.use(["science", "ieee", "notebook", "grid"])
    except Exception:
        pass
    plt.rcParams["figure.autolayout"] = False

    hist_styles = {
        ("Isotropic", "err_w"): dict(color="black", linestyle="-", marker="o", label=r"Isotropic, $\varepsilon_w$"),
        ("Isotropic", "err_T1"): dict(color="red", linestyle="--", marker="s", label=r"Isotropic, $\varepsilon_{T_1}$"),
        ("Orthotropic", "err_w"): dict(color="blue", linestyle="-", marker="o", label=r"Orthotropic, $\varepsilon_w$"),
        ("Orthotropic", "err_T1"): dict(color="purple", linestyle="--", marker="s", label=r"Orthotropic, $\varepsilon_{T_1}$"),
    }

    fig, ax = plt.subplots(figsize=(6.8, 5.2), facecolor="white")

    for case_name in ["Isotropic", "Orthotropic"]:
        grp = hist[hist["case"].astype(str) == case_name].copy()
        if grp.empty:
            continue
        grp = grp.sort_values("iter")
        it = grp["iter"].to_numpy(int)
        for col in ["err_w", "err_T1"]:
            vals = grp[col].to_numpy(float)
            if col == "err_T1" and not np.isfinite(vals[1:]).any():
                continue
            style = hist_styles[(case_name, col)]
            ax.semilogy(
                it,
                vals,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2.2,
                markersize=7.0,
                markerfacecolor=style["color"],
                label=style["label"],
            )

    ax.set_facecolor("white")
    ax.set_title("Coupled fixed-point update history", fontsize=14, pad=8)
    ax.set_xlabel("Coupling iteration", fontsize=14)
    ax.set_ylabel(r"Relative $L^2(\Omega)$ update", fontsize=14)
    ax.tick_params(axis="both", labelsize=13, length=3, pad=2)
    ax.grid(True, which="both", linestyle=":", linewidth=0.45, alpha=0.35)

    leg = ax.legend(loc="upper right", fontsize=11.5, frameon=True, borderpad=0.55)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.8)

    os.makedirs(PAPER2_OUTDIR, exist_ok=True)
    for ext, dpi in (("pdf", 300), ("png", 600)):
        fig.savefig(
            os.path.join(PAPER2_OUTDIR, f"{filename_base}.{ext}"),
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
        )
    plt.show()


# ------------------------------------------------------------
# Optional reference-level diagnostic
# ------------------------------------------------------------

def compare_two_reference_levels(
    mu_mech,
    *,
    size_a=max(PLATE_RESOLUTIONS),
    size_b=REF_SIZE,
    p=REF_P,
    label="reference_check",
    bc_type=MECHANICAL_BC_TYPE,
):
    """
    Optional diagnostic: compare a smaller synchronized p=4/pT=3 coupled solve
    against the main reference.  This is not needed for the main convergence table.
    """
    ref_heat_b = heat_controls_from_plate_size(size_b, ref=True)

    hc_a_shape = _heat_shape_from_plate_size(size_a)
    hc_a = dict(
        heat_nx=hc_a_shape["heat_nx"],
        heat_ny=hc_a_shape["heat_ny"],
        heat_nz=hc_a_shape["heat_nz"],
        heat_degree=int(REF_HEAT_DEGREE),
        Nz_quad_T1=int(_nz_quad_t1_from_heat_controls(hc_a_shape["heat_nz"], REF_HEAT_DEGREE, ref=False)),
        T1_cg_degree=int(REF_T1_CG_DEGREE),
    )

    print("\nBUILDING OPTIONAL SMALLER p=4/pT=3 REFERENCE-LEVEL DIAGNOSTIC")
    sol_a = solve_paper2_coupled_case(
        size=size_a,
        p=p,
        mu_mech=mu_mech,
        heat_nx=hc_a["heat_nx"],
        heat_ny=hc_a["heat_ny"],
        heat_nz=hc_a["heat_nz"],
        heat_degree=hc_a["heat_degree"],
        Nz_quad_T1=hc_a["Nz_quad_T1"],
        T1_cg_degree=hc_a["T1_cg_degree"],
        bc_type=bc_type,
    )
    _assert_converged_coupled_record(sol_a, context=f"Reference diagnostic level A size={size_a}")

    sol_b = get_or_build_reference(
        label=label,
        bc_type=bc_type,
        ref_size=size_b,
        ref_p=p,
        mu_mech=mu_mech,
    )
    _assert_converged_coupled_record(sol_b, context=f"Reference level B size={size_b}")

    mesh_b = reference_mesh(sol_b)
    Vw_b = FunctionSpace(mesh_b, "CG", int(p))
    Vtheta_b = FunctionSpace(mesh_b, "CG", int(p))
    sqrt_area_b = float(np.sqrt(assemble(Constant(1.0) * Measure("dx", domain=mesh_b))))
    deg = max(5, int(p) + 2)

    comparison = dict(
        mode="coupled",
        label=label,
        bc_type=bc_type,
        size_a=int(size_a),
        size_b=int(size_b),
        p=int(p),
        heat_a=hc_a,
        heat_b=ref_heat_b,
        L2_RMS_Error_w_m=l2_rms_error_against_ref(sol_a["w"], sol_b["w"], Vw_b, sqrt_area_b, degree=deg),
        L2_RMS_Error_theta=l2_rms_error_against_ref(sol_a["theta"], sol_b["theta"], Vtheta_b, sqrt_area_b, degree=deg),
        Rel_L2_Error_w=l2_relative_error_against_ref(sol_a["w"], sol_b["w"], Vw_b, degree=deg),
        Rel_L2_Error_theta=l2_relative_error_against_ref(sol_a["theta"], sol_b["theta"], Vtheta_b, degree=deg),
    )

    out_json = os.path.join(
        PAPER2_OUTDIR,
        f"paper2_coupled_{_safe_filename_part(label)}_{_safe_filename_part(bc_type)}_{STUDY_TAG}_reference_comparison.json",
    )
    with open(out_json, "w") as f:
        json.dump(_json_safe(comparison), f, indent=2, sort_keys=True)

    print("\nREFERENCE-LEVEL COMPARISON")
    for key, value in comparison.items():
        print(f"{key}: {value}")
    print(f"Saved reference comparison: {out_json}")
    return sol_a, sol_b, comparison


# ------------------------------------------------------------
# Driver utilities
# ------------------------------------------------------------

def run_smoke_test(case_label="Isotropic", *, bc_type=MECHANICAL_BC_TYPE):
    """
    Very cheap structure test: small reference and only two coarse levels.
    This is not the final study; it checks that the synchronized driver works.
    """
    global REF_SIZE, REF_P, REF_HEAT_NX, REF_HEAT_NY, REF_HEAT_NZ, REF_HEAT_DEGREE, REF_T1_CG_DEGREE, REF_NZ_QUAD_T1

    old = (REF_SIZE, REF_P, REF_HEAT_NX, REF_HEAT_NY, REF_HEAT_NZ, REF_HEAT_DEGREE, REF_T1_CG_DEGREE, REF_NZ_QUAD_T1)
    try:
        REF_SIZE = 16
        REF_P = 3
        REF_HEAT_NX, REF_HEAT_NY, REF_HEAT_NZ = 16, 8, 4
        REF_HEAT_DEGREE = 2
        REF_T1_CG_DEGREE = 3
        REF_NZ_QUAD_T1 = 16

        mu_mech = PLATE_CASES[case_label]["mu_mech"]
        df_cpl, ref_cpl = gather_full_asymptotic_coupled_convergence_data(
            mu_mech=mu_mech,
            resolutions=[2, 4],
            degree_pairs=[dict(p=2, heat_degree=1, T1_cg_degree=2)],
            ref_size=REF_SIZE,
            ref_p=REF_P,
            label=f"{case_label}_smoke",
            bc_type=bc_type,
        )
        print(f"\nCOUPLED SYNCHRONIZED SMOKE {case_label.upper()} ({bc_type})")
        df_valid = valid_converged_coupled_rows(df_cpl)
        if not df_valid.empty:
            print(df_valid[[c for c in TABLE_COLS if c in df_valid.columns]].to_string(index=False, float_format="%.6e"))
        return df_cpl, ref_cpl
    finally:
        (REF_SIZE, REF_P, REF_HEAT_NX, REF_HEAT_NY, REF_HEAT_NZ, REF_HEAT_DEGREE, REF_T1_CG_DEGREE, REF_NZ_QUAD_T1) = old


def run_case(case_label, *, bc_type=MECHANICAL_BC_TYPE):
    """Run the synchronized synchronized coupled refinement study for one material case."""
    mu_mech = PLATE_CASES[case_label]["mu_mech"]
    df_cpl, ref_cpl = gather_full_asymptotic_coupled_convergence_data(
        mu_mech=mu_mech,
        resolutions=PLATE_RESOLUTIONS,
        degree_pairs=DEGREE_PAIRS,
        ref_size=REF_SIZE,
        ref_p=REF_P,
        label=case_label,
        bc_type=bc_type,
    )

    print(f"\nCOUPLED SYNCHRONIZED {case_label.upper()} ({bc_type})")
    df_valid = valid_converged_coupled_rows(df_cpl)
    if df_valid.empty:
        print("No valid converged coupled rows available.")
    else:
        print(df_valid[[c for c in TABLE_COLS if c in df_valid.columns]].to_string(index=False, float_format="%.6e"))
    return df_cpl, ref_cpl


def replot_from_saved_csv(*, bc_type=MECHANICAL_BC_TYPE):
    frames = []
    for case_label in CASES_TO_RUN:
        path = convergence_csv_path(case_label, bc_type)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing coupled convergence CSV: {path}")
        frames.append(pd.read_csv(path))

    df_all = pd.concat(frames, ignore_index=True)
    df_valid = valid_converged_coupled_rows(df_all)

    plot_paper2_synchronized_coupled_convergence(
        df_valid,
        ycol="L2_RMS_Error_w_m",
        ylabel=r"RMS $L^2(\Omega)$ error in $w_h$ [m]",
        x_mode="h",
        filename_base=f"paper2_sync_coupled_w_convergence_mesh_size_{STUDY_TAG}",
        rate_fit_slice=slice(None),
    )

    plot_paper2_synchronized_coupled_convergence(
        df_valid,
        ycol="L2_RMS_Error_theta",
        ylabel=r"RMS $L^2(\Omega)$ error in $T_1$",
        x_mode="h",
        filename_base=f"paper2_sync_coupled_T1_convergence_mesh_size_{STUDY_TAG}",
        rate_fit_slice=slice(None),
    )
    return df_all


def run_full_study(*, bc_type=MECHANICAL_BC_TYPE):
    print_refinement_path()

    case_outputs = []
    histories = []
    for case_label in CASES_TO_RUN:
        df_case, ref_case = run_case(case_label, bc_type=bc_type)
        case_outputs.append((case_label, df_case, ref_case))
        histories.append(history_dataframe_from_ref(ref_case, case_label))

    df_all = pd.concat([item[1] for item in case_outputs], ignore_index=True)
    df_valid = valid_converged_coupled_rows(df_all)

    combined_csv = os.path.join(
        PAPER2_OUTDIR,
        f"paper2_coupled_synchronized_all_cases_{_safe_filename_part(bc_type)}_{STUDY_TAG}.csv",
    )
    df_all.to_csv(combined_csv, index=False)
    print(f"\nSaved combined CSV: {combined_csv}")

    print_finest_converged_table(
        df_valid,
        title="FINEST-MESH CONVERGED SYNCHRONIZED COUPLED ROWS",
    )
    print_rate_summary(df_valid)

    plot_paper2_synchronized_coupled_convergence(
        df_valid,
        ycol="L2_RMS_Error_w_m",
        ylabel=r"RMS $L^2(\Omega)$ error in $w_h$ [m]",
        x_mode="h",
        filename_base=f"paper2_sync_coupled_w_convergence_mesh_size_{STUDY_TAG}",
        rate_fit_slice=slice(None),
    )

    plot_paper2_synchronized_coupled_convergence(
        df_valid,
        ycol="L2_RMS_Error_theta",
        ylabel=r"RMS $L^2(\Omega)$ error in $T_1$",
        x_mode="h",
        filename_base=f"paper2_sync_coupled_T1_convergence_mesh_size_{STUDY_TAG}",
        rate_fit_slice=slice(None),
    )

    plot_paper2_coupled_history_iso_ortho_combined(
        histories,
        filename_base=f"paper2_sync_coupled_reference_update_history_{STUDY_TAG}",
    )

    if RUN_REFERENCE_COMPARISON and "Isotropic" in PLATE_CASES:
        compare_two_reference_levels(
            PLATE_CASES["Isotropic"]["mu_mech"],
            size_a=max(PLATE_RESOLUTIONS),
            size_b=REF_SIZE,
            p=REF_P,
            label=f"Isotropic_sync_refcheck_{max(PLATE_RESOLUTIONS)}_vs_{REF_SIZE}_{bc_type}",
            bc_type=bc_type,
        )

    return df_all, df_valid, case_outputs


# ------------------------------------------------------------
# Execute synchronized coupled study
# ------------------------------------------------------------

# Optional quick structure test before the full run:
# df_smoke, ref_smoke = run_smoke_test("Isotropic", bc_type=MECHANICAL_BC_TYPE)

if RUN_FULL_STUDY:
    df_cpl_all, df_cpl_valid, case_outputs = run_full_study(bc_type=MECHANICAL_BC_TYPE)

# %% Cell 11 | id: 4523c11a-5110-4bbb-bd4c-90f70e2b2474


# %% Cell 12 | id: c64e6001-ecbe-4e19-a77a-2f60ae19d0d6


# %% Cell 13 | id: 0a3e5c0e-fc32-4273-a615-50f7fcb62205


# %% Cell 14 | id: a0d0ec5d-3cbe-4796-9253-6d23fce867d8


# %% Cell 15 | id: 50d29c4c-f53e-4c27-813c-fcb2e6c944c9


# %% Cell 16 | id: e676de86-08fe-4cb9-a85a-09c9ef052569
# ============================================================
# Paper 2 thermomechanical FOM convergence
# Initial synchronized coupled-refinement study with coarse reference
# ============================================================
# Intended use:
#   Paste/run this AFTER CODE_THERMOMECHANICAL_FOM.txt has defined
#   GeneralMultiphysicsSolver and the required FEniCS imports.
#
# Study meaning:
#   - coupled-only;
#   - no fixed heat mesh;
#   - plate mesh and 3D heat mesh are refined together;
#   - p = 2 uses heat degree pT = 2;
#   - p = 3 uses heat degree pT = 3;
#   - initial reference is a coupled FOM with plate N = 32, p = 4,
#     heat mesh 32 x 16 x 8, heat degree pT = 4, and T1 degree q = 4.
#
# Important practical note:
#   This is the correct driver structure for a synchronized coupled refinement
#   study.  The N=32, p=4, heat 32 x 16 x 8, pT=4 reference is intentionally
#   chosen as a first practical reference: it is one refinement level finer than
#   the finest tested level N=16, and uses one higher polynomial degree than the
#   p=3/pT=3 test branch.  For final manuscript-level asymptotic evidence, repeat
#   later with a farther reference such as N=64 or N=100 if computationally feasible.
# ============================================================


# ------------------------------------------------------------
# Global study controls
# ------------------------------------------------------------

PAPER2_OUTDIR = "paper2_initial_sync_coupled_convergence_refN32"
os.makedirs(PAPER2_OUTDIR, exist_ok=True)

MECHANICAL_BC_TYPE = "simply_supported"

# Synchronized coarse refinement levels requested by the user.
# These are the 2D plate mesh-generation resolutions.
PLATE_RESOLUTIONS = [2, 4, 8, 12, 16]

# Degree pairings for the synchronized coupled study:
#   p=2 with heat degree pT=2,
#   p=3 with heat degree pT=3.
# T1_cg_degree is taken equal to p for the tested runs.
DEGREE_PAIRS = [
    dict(p=2, heat_degree=2, T1_cg_degree=2),
    dict(p=3, heat_degree=3, T1_cg_degree=3),
]
P_VALUES = [d["p"] for d in DEGREE_PAIRS]

# Initial coupled numerical reference.
# This is one synchronized refinement level finer than the finest tested level
# plate N=16 / heat 16 x 8 x 4, and uses p=4/pT=4/qT1=4.
REF_SIZE = 32
REF_P = 4
REF_HEAT_NX = 32
REF_HEAT_NY = 16
REF_HEAT_NZ = 8
REF_HEAT_DEGREE = 4
REF_T1_CG_DEGREE = 4
REF_NZ_QUAD_T1 = 24

# Resume/cache behavior.
RESUME_RUNS = True
FORCE_RERUN = False
RUN_FULL_STUDY = True
RUN_REFERENCE_COMPARISON = False

# Optional: choose only ["Isotropic"] first if you want a lighter starting run.
CASES_TO_RUN = ["Isotropic", "Orthotropic"]

# Optional: stop immediately on the first failed/non-converged run.
# If False, the failed row is stored as reported=False and the loop continues.
STOP_ON_FAILURE = False

REFERENCE_CACHE_VERSION = 12
CONVERGENCE_TABLE_VERSION = 12

STUDY_TAG = (
    f"sync_refN{int(REF_SIZE)}_p{int(REF_P)}"
    f"_heat{int(REF_HEAT_NX)}x{int(REF_HEAT_NY)}x{int(REF_HEAT_NZ)}"
    f"_pT{int(REF_HEAT_DEGREE)}"
    f"_levels{'-'.join(str(v) for v in PLATE_RESOLUTIONS)}"
    f"_p2pT2_p3pT3"
    f"_v{int(CONVERGENCE_TABLE_VERSION)}"
)

PLATE_CASES = {
    "Isotropic": dict(
        mu_mech=[1.0e4, 1.0e4, 0.3e4, 0.35e4, 1.0e3, -10000.0]
    ),
    "Orthotropic": dict(
        mu_mech=[2.0e4, 1.0e4, 0.25e4, 0.30e4, 1.0e3, -10000.0]
    ),
}

THERMAL_PARAMS = dict(
    T_amb=27.0 + 273.15,
    T_sub=24.0 + 273.15,
    h_con=10.0,
    eps_r=0.90,
    q_s=700.0,
    h_c_cont=200.0,
    h_c_gap=5.0,
    eta_c=50.0,
    w_contact=-0.1,
    kx=0.35,
    ky=0.35,
    kz=0.35,
    alpha1=1.3e-4,
    alpha2=1.3e-4,
    rho=1050.0,
)

COUPLING_PARAMS = dict(
    coupling_omega=0.7,
    coupling_tol_w=1e-7,
    coupling_tol_T1=1e-7,
    coupling_max_iters=50,
)

TABLE_COLS = """
mode case bc_type p heat_degree T1_cg_degree size h inv_h
heat_nx heat_ny heat_nz Nz_quad_T1
plate_DoFs heat_DoFs theta_DoFs Total_DoFs
sqrt_plate_DoFs sqrt_Total_DoFs
L2_RMS_Error_w_m Rel_L2_Error_w
L2_RMS_Error_theta Rel_L2_Error_theta
coupled_iters coupled_converged stopped_by_max_iters
termination_reason err_w err_theta stopping_norm reported
""".split()

# Use a slightly looser Newton absolute tolerance than the default to keep the
# coupled campaign robust on coarse and high-order runs.  The coupled stopping is
# still controlled by relative L2(Omega) field updates in solve_coupled_thermo_mechanical.
try:
    NEWTON_SOLVER_PARAMETERS["newton_solver"].update(
        linear_solver="petsc",
        absolute_tolerance=5e-5,
        relative_tolerance=1e-5,
        maximum_iterations=80,
        relaxation_parameter=1.0,
    )
except Exception:
    pass


# ------------------------------------------------------------
# Synchronized heat controls
# ------------------------------------------------------------

def _heat_shape_from_plate_size(size):
    """
    Synchronized 3D heat mesh path tied to the plate resolution.

    The ratio follows the previous physical aspect choice used in the project:
        heat_nx : heat_ny : heat_nz approximately 1 : 1/2 : 1/4.

    For the requested coarse levels this gives:
        plate 2  -> heat 2 x 1 x 1
        plate 4  -> heat 4 x 2 x 1
        plate 8  -> heat 8 x 4 x 2
        plate 12 -> heat 12 x 6 x 3
        plate 16 -> heat 16 x 8 x 4
    """
    n = int(size)
    return dict(
        heat_nx=max(1, n),
        heat_ny=max(1, int(round(n / 2.0))),
        heat_nz=max(1, int(round(n / 4.0))),
    )


def _nz_quad_t1_from_heat_controls(heat_nz, heat_degree, *, ref=False):
    """
    Through-thickness Gauss points for T1 extraction.

    This is not the heat FE degree.  It controls the quadrature used in
    make_T1_from_DeltaT.  It is refined with the heat degree and with heat_nz.
    """
    heat_nz = int(heat_nz)
    heat_degree = int(heat_degree)
    if ref:
        return int(max(REF_NZ_QUAD_T1, 2 * heat_degree + 4, heat_nz + 4))
    return int(max(8, 2 * heat_degree + 4, heat_nz + 2))


def degree_pair_for_p(p):
    for item in DEGREE_PAIRS:
        if int(item["p"]) == int(p):
            return dict(item)
    raise ValueError(f"No synchronized degree pair defined for p={p}. Available: {DEGREE_PAIRS}")


def heat_controls_from_plate_size(size, p=None, *, ref=False):
    """
    Return heat controls for synchronized coupled refinement.

    No fixed heat mesh is used here.
    """
    if ref:
        return dict(
            heat_nx=int(REF_HEAT_NX),
            heat_ny=int(REF_HEAT_NY),
            heat_nz=int(REF_HEAT_NZ),
            heat_degree=int(REF_HEAT_DEGREE),
            Nz_quad_T1=int(_nz_quad_t1_from_heat_controls(
                REF_HEAT_NZ, REF_HEAT_DEGREE, ref=True
            )),
            T1_cg_degree=int(REF_T1_CG_DEGREE),
        )

    if p is None:
        raise ValueError("heat_controls_from_plate_size: p must be supplied for non-reference runs.")

    pair = degree_pair_for_p(p)
    shape = _heat_shape_from_plate_size(size)
    heat_degree = int(pair["heat_degree"])

    return dict(
        heat_nx=int(shape["heat_nx"]),
        heat_ny=int(shape["heat_ny"]),
        heat_nz=int(shape["heat_nz"]),
        heat_degree=int(heat_degree),
        Nz_quad_T1=int(_nz_quad_t1_from_heat_controls(shape["heat_nz"], heat_degree, ref=False)),
        T1_cg_degree=int(pair["T1_cg_degree"]),
    )


def print_refinement_path():
    print("\nSYNCHRONIZED COUPLED REFINEMENT PATH")
    print("Reference:")
    ref_hc = heat_controls_from_plate_size(REF_SIZE, ref=True)
    print(
        f"  plate N={REF_SIZE}, p={REF_P}; "
        f"heat=({ref_hc['heat_nx']},{ref_hc['heat_ny']},{ref_hc['heat_nz']}), "
        f"pT={ref_hc['heat_degree']}, qT1={ref_hc['T1_cg_degree']}, "
        f"Nz_quad_T1={ref_hc['Nz_quad_T1']}"
    )
    print("Test levels:")
    for p in P_VALUES:
        for n in PLATE_RESOLUTIONS:
            hc = heat_controls_from_plate_size(n, p=p, ref=False)
            print(
                f"  p={p}, pT={hc['heat_degree']}, plate N={n}: "
                f"heat=({hc['heat_nx']},{hc['heat_ny']},{hc['heat_nz']}), "
                f"qT1={hc['T1_cg_degree']}, Nz_quad_T1={hc['Nz_quad_T1']}"
            )


# ------------------------------------------------------------
# Solver construction and single coupled solve
# ------------------------------------------------------------

def make_solver(size, p, mu_mech, *, load_type="uniform", bc_type=MECHANICAL_BC_TYPE):
    solver = GeneralMultiphysicsSolver(study_case=1)
    solver.size = int(size)
    solver.degree = int(p)
    solver.load_type = load_type
    solver.bc_type = bc_type
    solver.define_domain(n_vert=0, n_horiz=0, plot_subdomains=False)
    solver.set_mu(mu_mech)
    solver.set_rom_thermal_parameters(**THERMAL_PARAMS)
    return solver


def copy_function(f):
    out = Function(f.function_space())
    out.assign(f)
    return out


def solve_paper2_coupled_case(
    *,
    size,
    p,
    mu_mech,
    heat_nx,
    heat_ny,
    heat_nz,
    heat_degree,
    Nz_quad_T1,
    T1_cg_degree,
    bc_type=MECHANICAL_BC_TYPE,
):
    """Solve one coupled thermomechanical FOM case."""
    solver = make_solver(size=size, p=p, mu_mech=mu_mech, bc_type=bc_type)

    out = solver.solve_rom_sample(
        mu_mech,
        thermal_on=True,
        coupled_on=True,
        heat_nx=int(heat_nx),
        heat_ny=int(heat_ny),
        heat_nz=int(heat_nz),
        heat_degree=int(heat_degree),
        Nz_quad_T1=int(Nz_quad_T1),
        T1_cg_degree=int(T1_cg_degree),
        coupling_omega=COUPLING_PARAMS["coupling_omega"],
        coupling_tol_w=COUPLING_PARAMS["coupling_tol_w"],
        coupling_tol_T1=COUPLING_PARAMS["coupling_tol_T1"],
        coupling_max_iters=COUPLING_PARAMS["coupling_max_iters"],
        coupling_verbose=False,
        w0=0.0,
        return_mode="global",
        return_coupled_dict=True,
    )

    return dict(
        solver=solver,
        mode="coupled",
        bc_type=bc_type,
        w=copy_function(out["w"]),
        theta=copy_function(out["T1"]),
        heat=out["heat"],
        history=out["history"],
        converged=bool(out["converged"]),
        stopped_by_max_iters=bool(out["stopped_by_max_iters"]),
        termination_reason=str(out["termination_reason"]),
        err_w=float(out["err_w"]),
        err_T1=float(out["err_T1"]),
        iters=int(out["iters"]),
        w_dofs=int(out["w_dofs"]),
        heat_dofs=int(out["heat_dofs"]),
        theta_dofs=int(out["T1_dofs"]),
        total_dofs=int(out["total_coupled_dofs"]),
        heat_nx=int(heat_nx),
        heat_ny=int(heat_ny),
        heat_nz=int(heat_nz),
        heat_degree=int(heat_degree),
        Nz_quad_T1=int(Nz_quad_T1),
        T1_cg_degree=int(T1_cg_degree),
    )


# ------------------------------------------------------------
# Error computation against refined reference
# ------------------------------------------------------------

def lift_field_to_reference_mesh(field, Vref, degree=5):
    """Interpolate a scalar plate field from its own mesh onto the reference mesh."""
    try:
        field.set_allow_extrapolation(True)
    except Exception:
        pass

    class FieldOnReferenceMesh(UserExpression):
        def __init__(self, f, **kwargs):
            super().__init__(**kwargs)
            self.f = f

        def eval(self, values, x):
            values[0] = float(self.f(Point(float(x[0]), float(x[1]))))

        def value_shape(self):
            return ()

    return interpolate(FieldOnReferenceMesh(field, degree=int(degree)), Vref)


def l2_rms_error_against_ref(field, ref_field, Vref, sqrt_area, degree=5):
    """RMS L2(Omega) error against the refined coupled numerical FOM reference."""
    field_refmesh = lift_field_to_reference_mesh(field, Vref, degree=degree)
    ref_projected = project(ref_field, Vref)
    return float(errornorm(ref_projected, field_refmesh, norm_type="L2", degree_rise=3) / sqrt_area)


def l2_relative_error_against_ref(field, ref_field, Vref, degree=5, eps=1e-14):
    """Relative L2(Omega) error against the refined coupled numerical FOM reference."""
    dx_ref = Measure("dx", domain=Vref.mesh())
    field_refmesh = lift_field_to_reference_mesh(field, Vref, degree=degree)
    ref_projected = project(ref_field, Vref)

    abs_err = float(errornorm(ref_projected, field_refmesh, norm_type="L2", degree_rise=3))
    ref_norm = float(np.sqrt(max(float(assemble(ref_projected * ref_projected * dx_ref)), 0.0)))
    return abs_err / max(ref_norm, float(eps))


# ------------------------------------------------------------
# Checkpoint helpers
# ------------------------------------------------------------

def _safe_filename_part(text):
    text = str(text)
    for ch in [" ", "/", "\\", ":", ";", ",", "(", ")", "[", "]", "{", "}"]:
        text = text.replace(ch, "_")
    return text


def convergence_csv_path(label, bc_type):
    return os.path.join(
        PAPER2_OUTDIR,
        f"paper2_coupled_{_safe_filename_part(label)}_{_safe_filename_part(bc_type)}_convergence_{STUDY_TAG}.csv",
    )


def _as_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return bool(default)
        return bool(int(value))
    if isinstance(value, str):
        value_l = value.strip().lower()
        if value_l in ("true", "t", "1", "yes", "y"):
            return True
        if value_l in ("false", "f", "0", "no", "n", "nan", "none", ""):
            return False
    return bool(default)


def _bool_series(series, default=False):
    return series.apply(lambda v: _as_bool(v, default=default)).astype(bool)


def _row_is_valid_converged_coupled(row):
    try:
        if str(row.get("mode", "")) != "coupled":
            return False
        if not _as_bool(row.get("coupled_converged", False)):
            return False
        if _as_bool(row.get("stopped_by_max_iters", False)):
            return False
        if "reported" in row.index and not _as_bool(row.get("reported", True), default=True):
            return False
        if "stopping_norm" in row.index and str(row.get("stopping_norm", "")) != "L2_Omega":
            return False

        finite_cols = [
            "h",
            "L2_RMS_Error_w_m",
            "Rel_L2_Error_w",
            "L2_RMS_Error_theta",
            "Rel_L2_Error_theta",
            "err_w",
            "err_theta",
        ]
        return all(col not in row.index or np.isfinite(float(row[col])) for col in finite_cols)
    except Exception:
        return False


def valid_converged_coupled_rows(df):
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out = out[out["mode"].astype(str) == "coupled"]

    if "coupled_converged" in out.columns:
        out = out[_bool_series(out["coupled_converged"], default=False)]
    if "stopped_by_max_iters" in out.columns:
        out = out[~_bool_series(out["stopped_by_max_iters"], default=False)]
    if "reported" in out.columns:
        out = out[_bool_series(out["reported"], default=True)]
    if "stopping_norm" in out.columns:
        out = out[out["stopping_norm"].astype(str) == "L2_Omega"]

    for col in [
        "h",
        "L2_RMS_Error_w_m",
        "Rel_L2_Error_w",
        "L2_RMS_Error_theta",
        "Rel_L2_Error_theta",
        "err_w",
        "err_theta",
    ]:
        if col in out.columns:
            out = out[np.isfinite(out[col].to_numpy(dtype=float))]

    return out.reset_index(drop=True)


def load_existing_convergence_rows(out_csv):
    if not (RESUME_RUNS and os.path.exists(out_csv) and not FORCE_RERUN):
        return None, set()

    df = pd.read_csv(out_csv)
    if df.empty:
        return None, set()

    required_key_cols = ["mode", "case", "bc_type", "p", "heat_degree", "size"]
    valid_keys = {
        (
            str(row["mode"]),
            str(row["case"]),
            str(row["bc_type"]),
            int(row["p"]),
            int(row["heat_degree"]),
            int(row["size"]),
        )
        for _, row in df.iterrows()
        if all(col in row.index for col in required_key_cols) and _row_is_valid_converged_coupled(row)
    }

    print(f"\nResuming from existing table: {out_csv}")
    print(f"Existing rows: {len(df)}")
    print(f"Valid converged coupled rows that will be skipped: {len(valid_keys)}")
    return df, valid_keys


def save_convergence_checkpoint(new_records, existing_df, out_csv):
    if not new_records:
        return existing_df if existing_df is not None else pd.DataFrame()

    new_df = pd.DataFrame.from_records(new_records)
    save_df = pd.concat([existing_df, new_df], ignore_index=True) if existing_df is not None and not existing_df.empty else new_df

    save_df = (
        save_df.drop_duplicates(["mode", "case", "bc_type", "p", "heat_degree", "size"], keep="last")
        .sort_values(["mode", "case", "bc_type", "p", "heat_degree", "h"], ascending=[True, True, True, True, True, False])
        .reset_index(drop=True)
    )
    save_df.to_csv(out_csv, index=False)
    print(f"Checkpoint saved: {out_csv}")
    return save_df


# ------------------------------------------------------------
# Reference cache helpers
# ------------------------------------------------------------

def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def reference_cache_key(*, label, bc_type, ref_size, ref_p, mu_mech, ref_heat):
    payload = dict(
        cache_version=int(REFERENCE_CACHE_VERSION),
        convergence_table_version=int(CONVERGENCE_TABLE_VERSION),
        study_tag=str(STUDY_TAG),
        mode="coupled",
        case=str(label),
        bc_type=str(bc_type),
        ref_size=int(ref_size),
        ref_p=int(ref_p),
        mu_mech=[float(v) for v in mu_mech],
        thermal_params=_json_safe(THERMAL_PARAMS),
        ref_heat=_json_safe(ref_heat),
        coupling_params=_json_safe(COUPLING_PARAMS),
        plate_resolutions=[int(v) for v in PLATE_RESOLUTIONS],
        degree_pairs=_json_safe(DEGREE_PAIRS),
        synchronized_refinement=True,
        fixed_heat=False,
        mechanical_bc_type=str(MECHANICAL_BC_TYPE),
        stopping_norm="L2_Omega",
        reported_refinement_branch="coupled_only_synchronized_full_asymptotic",
    )
    payload_json = json.dumps(payload, sort_keys=True, indent=2)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:16], payload


def reference_cache_paths(*, label, bc_type, ref_size, ref_p, cache_key):
    stem = (
        f"paper2_ref_coupled_{_safe_filename_part(label)}"
        f"_{_safe_filename_part(bc_type)}_size{int(ref_size)}_p{int(ref_p)}_{cache_key}"
    )
    return dict(
        h5=os.path.join(PAPER2_OUTDIR, stem + ".h5"),
        json=os.path.join(PAPER2_OUTDIR, stem + ".json"),
    )


def _assert_converged_coupled_record(record, *, context):
    if not bool(record.get("converged", False)):
        raise RuntimeError(
            f"{context} did not converge: "
            f"iters={record.get('iters', None)}, "
            f"err_w={record.get('err_w', np.nan)}, "
            f"err_T1={record.get('err_T1', np.nan)}, "
            f"termination={record.get('termination_reason', None)}"
        )
    if bool(record.get("stopped_by_max_iters", False)):
        raise RuntimeError(
            f"{context} reached max coupling iterations and must not be reported: "
            f"iters={record.get('iters', None)}, "
            f"err_w={record.get('err_w', np.nan)}, "
            f"err_T1={record.get('err_T1', np.nan)}, "
            f"termination={record.get('termination_reason', None)}"
        )


def _reference_meta(ref, w_save, theta_save, payload):
    return dict(
        payload=payload,
        history=ref.get("history", None),
        converged=bool(ref.get("converged", True)),
        stopped_by_max_iters=bool(ref.get("stopped_by_max_iters", False)),
        termination_reason=str(ref.get("termination_reason", "cached_reference")),
        err_w=float(ref.get("err_w", np.nan)),
        err_T1=float(ref.get("err_T1", np.nan)),
        iters=int(ref.get("iters", 1)),
        w_dofs=int(ref.get("w_dofs", w_save.function_space().dim())),
        heat_dofs=int(ref.get("heat_dofs", 0)),
        theta_dofs=int(ref.get("theta_dofs", theta_save.function_space().dim())),
        total_dofs=int(ref.get("total_dofs", 0)),
        heat_nx=int(ref.get("heat_nx", payload["ref_heat"]["heat_nx"])),
        heat_ny=int(ref.get("heat_ny", payload["ref_heat"]["heat_ny"])),
        heat_nz=int(ref.get("heat_nz", payload["ref_heat"]["heat_nz"])),
        heat_degree=int(ref.get("heat_degree", payload["ref_heat"]["heat_degree"])),
        Nz_quad_T1=int(ref.get("Nz_quad_T1", payload["ref_heat"]["Nz_quad_T1"])),
        T1_cg_degree=int(ref.get("T1_cg_degree", payload["ref_heat"]["T1_cg_degree"])),
    )


def save_reference_cache(ref, *, paths, payload, ref_p):
    _assert_converged_coupled_record(ref, context="Coupled reference before cache save")

    mesh = ref["solver"].mesh
    Vw_ref = FunctionSpace(mesh, "CG", int(ref_p))
    Vtheta_ref = FunctionSpace(mesh, "CG", int(ref_p))

    w_save = project(ref["w"], Vw_ref)
    theta_save = project(ref["theta"], Vtheta_ref)

    h5 = HDF5File(mesh.mpi_comm(), paths["h5"], "w")
    h5.write(mesh, "/mesh")
    h5.write(w_save, "/w")
    h5.write(theta_save, "/theta")
    h5.close()

    with open(paths["json"], "w") as f:
        json.dump(_json_safe(_reference_meta(ref, w_save, theta_save, payload)), f, indent=2, sort_keys=True)

    print("Saved reference cache:")
    print(f"  {paths['h5']}")
    print(f"  {paths['json']}")


def load_reference_cache(*, paths, ref_p):
    if not (os.path.exists(paths["h5"]) and os.path.exists(paths["json"])):
        return None

    try:
        mesh = Mesh()
        h5 = HDF5File(MPI.comm_world, paths["h5"], "r")
        h5.read(mesh, "/mesh", False)

        Vw_ref = FunctionSpace(mesh, "CG", int(ref_p))
        Vtheta_ref = FunctionSpace(mesh, "CG", int(ref_p))
        w_ref = Function(Vw_ref)
        theta_ref = Function(Vtheta_ref)

        h5.read(w_ref, "/w")
        h5.read(theta_ref, "/theta")
        h5.close()

        with open(paths["json"], "r") as f:
            meta = json.load(f)
    except Exception as exc:
        print(f"Reference cache could not be loaded and will be rebuilt: {exc}")
        return None

    ref = dict(
        solver=None,
        mesh=mesh,
        w=w_ref,
        theta=theta_ref,
        heat=None,
        history=meta.get("history", None),
        converged=bool(meta.get("converged", True)),
        stopped_by_max_iters=bool(meta.get("stopped_by_max_iters", False)),
        termination_reason=str(meta.get("termination_reason", "loaded_reference_cache")),
        err_w=float(meta.get("err_w", np.nan)),
        err_T1=float(meta.get("err_T1", np.nan)),
        iters=int(meta.get("iters", 1)),
        w_dofs=int(meta.get("w_dofs", w_ref.function_space().dim())),
        heat_dofs=int(meta.get("heat_dofs", 0)),
        theta_dofs=int(meta.get("theta_dofs", theta_ref.function_space().dim())),
        total_dofs=int(meta.get("total_dofs", 0)),
        heat_nx=int(meta.get("heat_nx", -1)),
        heat_ny=int(meta.get("heat_ny", -1)),
        heat_nz=int(meta.get("heat_nz", -1)),
        heat_degree=int(meta.get("heat_degree", -1)),
        Nz_quad_T1=int(meta.get("Nz_quad_T1", -1)),
        T1_cg_degree=int(meta.get("T1_cg_degree", -1)),
    )

    _assert_converged_coupled_record(ref, context="Loaded coupled reference cache")
    print("Loaded cached reference:")
    print(f"  {paths['h5']}")
    return ref


def get_or_build_reference(*, label, bc_type, ref_size, ref_p, mu_mech):
    """Build/load the synchronized high-resolution coupled reference."""
    ref_heat = heat_controls_from_plate_size(ref_size, ref=True)
    cache_key, payload = reference_cache_key(
        label=label,
        bc_type=bc_type,
        ref_size=ref_size,
        ref_p=ref_p,
        mu_mech=mu_mech,
        ref_heat=ref_heat,
    )
    paths = reference_cache_paths(
        label=label,
        bc_type=bc_type,
        ref_size=ref_size,
        ref_p=ref_p,
        cache_key=cache_key,
    )

    if RESUME_RUNS and not FORCE_RERUN:
        cached = load_reference_cache(paths=paths, ref_p=ref_p)
        if cached is not None:
            return cached

    print("\n" + "=" * 100)
    print(
        f"BUILDING SYNCHRONIZED COUPLED REFERENCE: {label}, bc={bc_type}, "
        f"plate N={ref_size}, p={ref_p}, "
        f"heat=({ref_heat['heat_nx']},{ref_heat['heat_ny']},{ref_heat['heat_nz']}), "
        f"pT={ref_heat['heat_degree']}, qT1={ref_heat['T1_cg_degree']}, "
        f"Nz_quad_T1={ref_heat['Nz_quad_T1']}"
    )
    print("=" * 100)

    ref = solve_paper2_coupled_case(
        size=ref_size,
        p=ref_p,
        mu_mech=mu_mech,
        heat_nx=ref_heat["heat_nx"],
        heat_ny=ref_heat["heat_ny"],
        heat_nz=ref_heat["heat_nz"],
        heat_degree=ref_heat["heat_degree"],
        Nz_quad_T1=ref_heat["Nz_quad_T1"],
        T1_cg_degree=ref_heat["T1_cg_degree"],
        bc_type=bc_type,
    )

    _assert_converged_coupled_record(ref, context="Newly built synchronized coupled reference")
    save_reference_cache(ref, paths=paths, payload=payload, ref_p=ref_p)
    return ref


def reference_mesh(ref):
    return ref["solver"].mesh if ref.get("solver", None) is not None else ref["mesh"]


# ------------------------------------------------------------
# Records and convergence gathering
# ------------------------------------------------------------

def _coupled_solution_is_converged(record):
    return (
        str(record.get("mode", "coupled")) == "coupled"
        and bool(record.get("converged", False))
        and not bool(record.get("stopped_by_max_iters", False))
    )


def _success_record(label, bc_type, p, res, sol, errors):
    ew_rms, etheta_rms, ew_rel, etheta_rel = errors
    h_val = float(sol["solver"].mesh.hmax())
    return dict(
        mode="coupled",
        case=str(label),
        bc_type=str(bc_type),
        p=int(p),
        heat_degree=int(sol["heat_degree"]),
        T1_cg_degree=int(sol["T1_cg_degree"]),
        size=int(res),
        h=h_val,
        inv_h=float(1.0 / h_val),
        heat_nx=int(sol["heat_nx"]),
        heat_ny=int(sol["heat_ny"]),
        heat_nz=int(sol["heat_nz"]),
        Nz_quad_T1=int(sol["Nz_quad_T1"]),
        plate_DoFs=int(sol["w_dofs"]),
        heat_DoFs=int(sol["heat_dofs"]),
        theta_DoFs=int(sol["theta_dofs"]),
        Total_DoFs=int(sol["total_dofs"]),
        sqrt_plate_DoFs=float(np.sqrt(sol["w_dofs"])),
        sqrt_Total_DoFs=float(np.sqrt(sol["total_dofs"])),
        L2_RMS_Error_w_m=float(ew_rms),
        L2_RMS_Error_theta=float(etheta_rms),
        Rel_L2_Error_w=float(ew_rel),
        Rel_L2_Error_theta=float(etheta_rel),
        coupled_iters=int(sol["iters"]),
        coupled_converged=bool(sol["converged"]),
        stopped_by_max_iters=bool(sol["stopped_by_max_iters"]),
        termination_reason=str(sol["termination_reason"]),
        err_w=float(sol["err_w"]),
        err_theta=float(sol["err_T1"]),
        stopping_norm="L2_Omega",
        reported=True,
    )


def _failed_record(label, bc_type, p, res, hc, exc):
    return dict(
        mode="coupled",
        case=str(label),
        bc_type=str(bc_type),
        p=int(p),
        heat_degree=int(hc["heat_degree"]),
        T1_cg_degree=int(hc["T1_cg_degree"]),
        size=int(res),
        h=np.nan,
        inv_h=np.nan,
        heat_nx=int(hc["heat_nx"]),
        heat_ny=int(hc["heat_ny"]),
        heat_nz=int(hc["heat_nz"]),
        Nz_quad_T1=int(hc["Nz_quad_T1"]),
        plate_DoFs=np.nan,
        heat_DoFs=np.nan,
        theta_DoFs=np.nan,
        Total_DoFs=np.nan,
        sqrt_plate_DoFs=np.nan,
        sqrt_Total_DoFs=np.nan,
        L2_RMS_Error_w_m=np.nan,
        L2_RMS_Error_theta=np.nan,
        Rel_L2_Error_w=np.nan,
        Rel_L2_Error_theta=np.nan,
        coupled_iters=np.nan,
        coupled_converged=False,
        stopped_by_max_iters=False,
        termination_reason=f"failed_or_not_reported: {exc}",
        err_w=np.nan,
        err_theta=np.nan,
        stopping_norm="L2_Omega",
        reported=False,
    )


def _excluded_record_from_solution(label, bc_type, p, res, sol):
    h_val = float(sol["solver"].mesh.hmax())
    return dict(
        mode="coupled",
        case=str(label),
        bc_type=str(bc_type),
        p=int(p),
        heat_degree=int(sol["heat_degree"]),
        T1_cg_degree=int(sol["T1_cg_degree"]),
        size=int(res),
        h=h_val,
        inv_h=float(1.0 / h_val),
        heat_nx=int(sol["heat_nx"]),
        heat_ny=int(sol["heat_ny"]),
        heat_nz=int(sol["heat_nz"]),
        Nz_quad_T1=int(sol["Nz_quad_T1"]),
        plate_DoFs=int(sol["w_dofs"]),
        heat_DoFs=int(sol["heat_dofs"]),
        theta_DoFs=int(sol["theta_dofs"]),
        Total_DoFs=int(sol["total_dofs"]),
        sqrt_plate_DoFs=float(np.sqrt(sol["w_dofs"])),
        sqrt_Total_DoFs=float(np.sqrt(sol["total_dofs"])),
        L2_RMS_Error_w_m=np.nan,
        L2_RMS_Error_theta=np.nan,
        Rel_L2_Error_w=np.nan,
        Rel_L2_Error_theta=np.nan,
        coupled_iters=int(sol["iters"]),
        coupled_converged=bool(sol["converged"]),
        stopped_by_max_iters=bool(sol["stopped_by_max_iters"]),
        termination_reason=f"not_reported_nonconverged: {sol['termination_reason']}",
        err_w=float(sol["err_w"]),
        err_theta=float(sol["err_T1"]),
        stopping_norm="L2_Omega",
        reported=False,
    )


def gather_full_asymptotic_coupled_convergence_data(
    *,
    mu_mech,
    resolutions=PLATE_RESOLUTIONS,
    degree_pairs=DEGREE_PAIRS,
    ref_size=REF_SIZE,
    ref_p=REF_P,
    label="case",
    bc_type=MECHANICAL_BC_TYPE,
):
    """
    Gather synchronized coupled-refinement data against a high-resolution coupled FOM reference.

    Non-converged coupled rows are saved for traceability but excluded from plots/tables.
    """
    out_csv = convergence_csv_path(label, bc_type)
    existing_df, existing_keys = load_existing_convergence_rows(out_csv)

    ref = get_or_build_reference(
        label=label,
        bc_type=bc_type,
        ref_size=ref_size,
        ref_p=ref_p,
        mu_mech=mu_mech,
    )
    _assert_converged_coupled_record(ref, context=f"Synchronized coupled reference for {label}, bc={bc_type}")

    ref_mesh_obj = reference_mesh(ref)
    Vw_ref = FunctionSpace(ref_mesh_obj, "CG", int(ref_p))
    Vtheta_ref = FunctionSpace(ref_mesh_obj, "CG", int(ref_p))
    dx_ref = Measure("dx", domain=ref_mesh_obj)
    sqrt_area = float(np.sqrt(assemble(Constant(1.0) * dx_ref)))

    w_ref = project(ref["w"], Vw_ref)
    theta_ref = project(ref["theta"], Vtheta_ref)

    for pair in degree_pairs:
        p = int(pair["p"])
        for res in resolutions:
            case_key = ("coupled", label, bc_type, int(p), int(pair["heat_degree"]), int(res))
            if case_key in existing_keys:
                print(
                    f"SKIPPING valid cached row: case={label}, bc={bc_type}, "
                    f"size={res}, p={p}, pT={pair['heat_degree']}"
                )
                continue

            hc = heat_controls_from_plate_size(res, p=p, ref=False)
            print("\n" + "-" * 100)
            print(
                f"COUPLED SYNC | {label} | bc={bc_type} | "
                f"plate size={res}, p={p} | "
                f"heat=({hc['heat_nx']},{hc['heat_ny']},{hc['heat_nz']}), "
                f"pT={hc['heat_degree']}, qT1={hc['T1_cg_degree']}, "
                f"Nz_quad_T1={hc['Nz_quad_T1']}"
            )
            print("-" * 100)

            try:
                sol = solve_paper2_coupled_case(
                    size=res,
                    p=p,
                    mu_mech=mu_mech,
                    heat_nx=hc["heat_nx"],
                    heat_ny=hc["heat_ny"],
                    heat_nz=hc["heat_nz"],
                    heat_degree=hc["heat_degree"],
                    Nz_quad_T1=hc["Nz_quad_T1"],
                    T1_cg_degree=hc["T1_cg_degree"],
                    bc_type=bc_type,
                )

                if not _coupled_solution_is_converged(sol):
                    msg = (
                        f"EXCLUDED NON-CONVERGED RUN: case={label}, size={res}, p={p}, "
                        f"pT={hc['heat_degree']}, iters={sol.get('iters', None)}, "
                        f"err_w={sol.get('err_w', np.nan)}, err_T1={sol.get('err_T1', np.nan)}, "
                        f"termination={sol.get('termination_reason', None)}"
                    )
                    print(msg)
                    if STOP_ON_FAILURE:
                        raise RuntimeError(msg)
                    record = _excluded_record_from_solution(label, bc_type, p, res, sol)

                else:
                    deg = max(5, int(p) + 2, int(hc["T1_cg_degree"]) + 2)
                    errors = (
                        l2_rms_error_against_ref(sol["w"], w_ref, Vw_ref, sqrt_area, degree=deg),
                        l2_rms_error_against_ref(sol["theta"], theta_ref, Vtheta_ref, sqrt_area, degree=deg),
                        l2_relative_error_against_ref(sol["w"], w_ref, Vw_ref, degree=deg),
                        l2_relative_error_against_ref(sol["theta"], theta_ref, Vtheta_ref, degree=deg),
                    )
                    record = _success_record(label, bc_type, p, res, sol, errors)

            except Exception as exc:
                print(f"FAILED OR EXCLUDED: case={label}, bc={bc_type}, size={res}, p={p}: {exc}")
                if STOP_ON_FAILURE:
                    raise
                record = _failed_record(label, bc_type, p, res, hc, exc)

            existing_df = save_convergence_checkpoint([record], existing_df, out_csv)

            if _row_is_valid_converged_coupled(pd.Series(record)):
                existing_keys.add(case_key)

    df = existing_df.copy() if existing_df is not None and not existing_df.empty else pd.DataFrame()

    if not df.empty:
        df = df.sort_values(
            ["mode", "case", "bc_type", "p", "heat_degree", "h"],
            ascending=[True, True, True, True, True, False],
        ).reset_index(drop=True)
        df.to_csv(out_csv, index=False)

    df_valid = valid_converged_coupled_rows(df)

    print(f"\nSaved convergence table: {out_csv}")
    print(f"Total rows in CSV: {len(df)}")
    print(f"Valid converged coupled rows for reporting: {len(df_valid)}")
    return df, ref


# ------------------------------------------------------------
# Tables and rate summaries
# ------------------------------------------------------------

def finest_converged_rows_for_table(df):
    df_valid = valid_converged_coupled_rows(df)
    if df_valid.empty:
        return df_valid
    rows = [grp.sort_values("size", ascending=False).iloc[0] for _, grp in df_valid.groupby(["case", "p", "heat_degree"])]
    return pd.DataFrame(rows).sort_values(["case", "p", "heat_degree"]).reset_index(drop=True)


def print_finest_converged_table(df, title="FINEST CONVERGED COUPLED ROWS"):
    table_df = finest_converged_rows_for_table(df)
    print("\n" + title)
    if table_df.empty:
        print("No valid converged coupled rows available.")
        return table_df

    existing_cols = [c for c in TABLE_COLS if c in table_df.columns]
    print(table_df[existing_cols].to_string(index=False, float_format="%.6e"))
    return table_df


def compute_rate_summary(df, *, ycols=("L2_RMS_Error_w_m", "L2_RMS_Error_theta")):
    df_valid = valid_converged_coupled_rows(df)
    records = []
    if df_valid.empty:
        return pd.DataFrame()

    for (case, p, pT), grp in df_valid.groupby(["case", "p", "heat_degree"]):
        grp = grp.sort_values("h", ascending=False)
        x = grp["h"].to_numpy(float)
        for ycol in ycols:
            y = grp[ycol].to_numpy(float)
            mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
            xx, yy = x[mask], y[mask]
            rate_all = np.nan
            rate_last3 = np.nan
            if len(xx) >= 2:
                rate_all = abs(float(np.polyfit(np.log10(xx), np.log10(yy), 1)[0]))
            if len(xx) >= 3:
                rate_last3 = abs(float(np.polyfit(np.log10(xx[-3:]), np.log10(yy[-3:]), 1)[0]))
            records.append(dict(
                case=str(case),
                p=int(p),
                heat_degree=int(pT),
                quantity=str(ycol),
                n_points=int(len(xx)),
                rate_all=float(rate_all),
                rate_last3=float(rate_last3),
            ))
    return pd.DataFrame.from_records(records)


def print_rate_summary(df):
    rs = compute_rate_summary(df)
    print("\nLOG-LOG RATE SUMMARY")
    if rs.empty:
        print("No valid rate data.")
    else:
        print(rs.to_string(index=False, float_format="%.4f"))
    return rs


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def plot_paper2_synchronized_coupled_convergence(
    df_coupled,
    *,
    ycol="L2_RMS_Error_w_m",
    ylabel=r"RMS $L^2(\Omega)$ error in $w_h$ [m]",
    x_mode="h",
    filename_base=None,
    rate_fit_slice=slice(None),
):
    """
    Synchronized coupled convergence plot.

    Only valid converged coupled rows are plotted.  For x_mode='h', the x-axis is
    shown coarse-to-fine from left to right, i.e. larger h on the left.
    """

    if filename_base is None:
        filename_base = f"paper2_sync_coupled_{ycol}_{STUDY_TAG}"

    df = valid_converged_coupled_rows(df_coupled)
    if df.empty:
        raise ValueError("No valid converged coupled rows available for plotting.")

    try:
        plt.style.use(["science", "ieee", "notebook", "grid"])
    except Exception:
        pass
    plt.rcParams["figure.autolayout"] = False

    x_options = {
        "h": ("h", r"Characteristic plate mesh length $\ell_{\mathcal{T}}$ [m]"),
        "inv_h": ("inv_h", r"Inverse plate mesh size $1/h$"),
        "sqrt_plate_dofs": ("sqrt_plate_DoFs", r"$\sqrt{\mathrm{plate\ DoFs}}$"),
        "plate_dofs": ("plate_DoFs", "Plate degrees of freedom"),
        "sqrt_total_dofs": ("sqrt_Total_DoFs", r"$\sqrt{\mathrm{total\ DoFs}}$"),
        "total_dofs": ("Total_DoFs", "Total coupled degrees of freedom"),
    }
    if x_mode not in x_options:
        raise ValueError(f"x_mode must be one of {list(x_options.keys())}")
    xcol, xlabel = x_options[x_mode]

    style_specs = [
        ("Isotropic", 2, 2, "black", "-", "o", r"Isotropic, $p=2$, $p_T=2$"),
        ("Isotropic", 3, 3, "blue", "--", "s", r"Isotropic, $p=3$, $p_T=3$"),
        ("Orthotropic", 2, 2, "darkorange", "-", "o", r"Orthotropic, $p=2$, $p_T=2$"),
        ("Orthotropic", 3, 3, "purple", "--", "s", r"Orthotropic, $p=3$, $p_T=3$"),
    ]
    styles = {
        (case_name, p, pT): dict(color=color, linestyle=linestyle, marker=marker, label=label)
        for case_name, p, pT, color, linestyle, marker, label in style_specs
    }

    def positive_finite(vals):
        vals = np.asarray(vals, dtype=float)
        return vals[np.isfinite(vals) & (vals > 0.0)]

    def ordered_xy(x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
        x, y = x[mask], y[mask]
        order = np.argsort(x)[::-1] if x_mode == "h" else np.argsort(x)
        return x[order], y[order]

    def estimate_loglog_slope(x, y, fit_slice=slice(None)):
        x, y = ordered_xy(x, y)
        if len(x) < 2:
            return None
        xx, yy = x[fit_slice], y[fit_slice]
        if len(xx) < 2:
            return None
        slope, _ = np.polyfit(np.log10(xx), np.log10(yy), 1)
        return float(abs(slope))

    def add_rate_triangle(ax, x, y, rate, color, *, location="above", start_idx=-2, span_frac=0.42, tri_gap=1.25):
        if rate is None or rate <= 0.0:
            return
        x, y = ordered_xy(x, y)
        if len(x) < 2:
            return
        start_idx = len(x) + start_idx if start_idx < 0 else start_idx
        if start_idx < 0 or start_idx + 1 >= len(x):
            return

        x_prev = float(x[start_idx])
        x_last = float(x[start_idx + 1])
        y_last = float(y[start_idx + 1])
        if x_prev <= 0 or x_last <= 0 or y_last <= 0:
            return

        if x_mode == "h":
            full_ratio = x_prev / x_last
            if full_ratio <= 1:
                return
            x_right = x_last
            x_left = x_last * (full_ratio ** span_frac)
            local_ratio = x_left / x_right
            x_rate = x_right / (local_ratio ** 0.03)
        else:
            full_ratio = x_last / x_prev
            if full_ratio <= 1:
                return
            x_left = x_last / (full_ratio ** span_frac)
            x_right = x_last
            local_ratio = x_right / x_left
            x_rate = x_right * (local_ratio ** 0.03)

        if location == "below":
            y_top = y_last / tri_gap
            y_bottom = y_top / (local_ratio ** rate)
        else:
            y_bottom = y_last * tri_gap
            y_top = y_bottom * (local_ratio ** rate)

        if min(x_left, x_right, y_top, y_bottom) <= 0:
            return

        text_box = dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.12)
        ax.plot([x_left, x_right], [y_top, y_top], color=color, lw=1.0, zorder=5)
        ax.plot([x_right, x_right], [y_top, y_bottom], color=color, lw=1.0, zorder=5)
        ax.plot([x_left, x_right], [y_top, y_bottom], color=color, lw=1.0, zorder=5)
        ax.text(np.sqrt(x_left * x_right), y_top * 1.06, "1", ha="center", va="bottom", fontsize=8, color=color, bbox=text_box, zorder=6)
        ax.text(x_rate, np.sqrt(y_top * y_bottom), f"{rate:.1f}", ha="left", va="center", fontsize=8, color=color, bbox=text_box, zorder=6)

    all_x = positive_finite(df[xcol].to_numpy(float))
    all_y = positive_finite(df[ycol].to_numpy(float))
    if len(all_x) == 0 or len(all_y) == 0:
        raise ValueError("No positive finite data available for plotting.")

    fig, ax = plt.subplots(figsize=(7.0, 5.3), facecolor="white")
    grouped = {}

    for key, style in styles.items():
        case_name, p, pT = key
        grp = df[
            (df["case"].astype(str) == case_name)
            & (df["p"].astype(int) == int(p))
            & (df["heat_degree"].astype(int) == int(pT))
        ].dropna(subset=[xcol, ycol]).copy()

        if grp.empty:
            continue

        grp = grp.sort_values(xcol, ascending=(x_mode != "h"))
        grouped[key] = grp

        ax.loglog(
            grp[xcol].to_numpy(float),
            grp[ycol].to_numpy(float),
            linestyle=style["linestyle"],
            marker=style["marker"],
            color=style["color"],
            markeredgecolor=style["color"],
            markerfacecolor="none",
            linewidth=2.2,
            markersize=8.0,
            markeredgewidth=1.4,
            label=style["label"],
            clip_on=False,
        )

    ax.set_facecolor("white")
    ax.set_title(r"Synchronized coupled refinement study", fontsize=14, pad=8)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=13, length=4, pad=2)
    ax.tick_params(axis="both", which="minor", length=2)
    ax.grid(True, which="both", linestyle=":", linewidth=0.45, alpha=0.35)

    if x_mode == "h":
        ticks = []
        for size in PLATE_RESOLUTIONS:
            vals = positive_finite(df.loc[df["size"].astype(int) == int(size), "h"].to_numpy(float))
            if len(vals):
                ticks.append(float(np.median(vals)))
        if ticks:
            ticks = sorted(set(np.round(ticks, 12)), reverse=True)
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.2f}" if x >= 0.1 else f"{x:.3f}"))
    else:
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=80))
    ax.xaxis.set_minor_formatter(NullFormatter())

    if x_mode == "h":
        ax.set_xlim(np.max(all_x) * 1.20, np.min(all_x) / 1.14)
    else:
        ax.set_xlim(np.min(all_x) / 1.14, np.max(all_x) * 1.20)
    ax.set_ylim(np.min(all_y) / 2.60, np.max(all_y) * 1.85)

    for idx, (key, grp) in enumerate(grouped.items()):
        style = styles[key]
        rate = estimate_loglog_slope(grp[xcol].to_numpy(float), grp[ycol].to_numpy(float), fit_slice=rate_fit_slice)
        location = "above" if idx % 2 == 0 else "below"
        add_rate_triangle(ax, grp[xcol].to_numpy(float), grp[ycol].to_numpy(float), rate, style["color"], location=location)

    leg = ax.legend(
        loc="lower left",
        fontsize=11.2,
        frameon=True,
        borderpad=0.55,
        handlelength=2.2,
        handletextpad=0.55,
        labelspacing=0.35,
    )
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.8)

    os.makedirs(PAPER2_OUTDIR, exist_ok=True)
    for ext, dpi in (("pdf", 300), ("png", 600)):
        fig.savefig(
            os.path.join(PAPER2_OUTDIR, f"{filename_base}.{ext}"),
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
        )
    plt.show()


def history_dataframe_from_ref(ref_or_solution, case_label):
    hist = pd.DataFrame(ref_or_solution.get("history", None)).copy()
    if hist.empty:
        return hist
    hist["case"] = str(case_label)
    return hist


def plot_paper2_coupled_history_iso_ortho_combined(histories, *, filename_base=None):
    if filename_base is None:
        filename_base = f"paper2_sync_coupled_reference_update_history_{STUDY_TAG}"
    if not histories:
        print("No coupled histories available.")
        return

    hist = pd.concat(histories, ignore_index=True)
    if hist.empty:
        print("No coupled histories available.")
        return

    try:
        plt.style.use(["science", "ieee", "notebook", "grid"])
    except Exception:
        pass
    plt.rcParams["figure.autolayout"] = False

    hist_styles = {
        ("Isotropic", "err_w"): dict(color="black", linestyle="-", marker="o", label=r"Isotropic, $\varepsilon_w$"),
        ("Isotropic", "err_T1"): dict(color="red", linestyle="--", marker="s", label=r"Isotropic, $\varepsilon_{T_1}$"),
        ("Orthotropic", "err_w"): dict(color="blue", linestyle="-", marker="o", label=r"Orthotropic, $\varepsilon_w$"),
        ("Orthotropic", "err_T1"): dict(color="purple", linestyle="--", marker="s", label=r"Orthotropic, $\varepsilon_{T_1}$"),
    }

    fig, ax = plt.subplots(figsize=(6.8, 5.2), facecolor="white")

    for case_name in ["Isotropic", "Orthotropic"]:
        grp = hist[hist["case"].astype(str) == case_name].copy()
        if grp.empty:
            continue
        grp = grp.sort_values("iter")
        it = grp["iter"].to_numpy(int)
        for col in ["err_w", "err_T1"]:
            vals = grp[col].to_numpy(float)
            if col == "err_T1" and not np.isfinite(vals[1:]).any():
                continue
            style = hist_styles[(case_name, col)]
            ax.semilogy(
                it,
                vals,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2.2,
                markersize=7.0,
                markerfacecolor=style["color"],
                label=style["label"],
            )

    ax.set_facecolor("white")
    ax.set_title("Coupled fixed-point update history", fontsize=14, pad=8)
    ax.set_xlabel("Coupling iteration", fontsize=14)
    ax.set_ylabel(r"Relative $L^2(\Omega)$ update", fontsize=14)
    ax.tick_params(axis="both", labelsize=13, length=3, pad=2)
    ax.grid(True, which="both", linestyle=":", linewidth=0.45, alpha=0.35)

    leg = ax.legend(loc="upper right", fontsize=11.5, frameon=True, borderpad=0.55)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.8)

    os.makedirs(PAPER2_OUTDIR, exist_ok=True)
    for ext, dpi in (("pdf", 300), ("png", 600)):
        fig.savefig(
            os.path.join(PAPER2_OUTDIR, f"{filename_base}.{ext}"),
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
        )
    plt.show()


# ------------------------------------------------------------
# Optional reference-level diagnostic
# ------------------------------------------------------------

def compare_two_reference_levels(
    mu_mech,
    *,
    size_a=max(PLATE_RESOLUTIONS),
    size_b=REF_SIZE,
    p=REF_P,
    label="reference_check",
    bc_type=MECHANICAL_BC_TYPE,
):
    """
    Optional diagnostic: compare a smaller synchronized p=4/pT=4 coupled solve
    against the main reference.  This is not needed for the main convergence table.
    """
    ref_heat_b = heat_controls_from_plate_size(size_b, ref=True)

    hc_a_shape = _heat_shape_from_plate_size(size_a)
    hc_a = dict(
        heat_nx=hc_a_shape["heat_nx"],
        heat_ny=hc_a_shape["heat_ny"],
        heat_nz=hc_a_shape["heat_nz"],
        heat_degree=int(REF_HEAT_DEGREE),
        Nz_quad_T1=int(_nz_quad_t1_from_heat_controls(hc_a_shape["heat_nz"], REF_HEAT_DEGREE, ref=False)),
        T1_cg_degree=int(REF_T1_CG_DEGREE),
    )

    print("\nBUILDING OPTIONAL SMALLER p=4/pT=4 REFERENCE-LEVEL DIAGNOSTIC")
    sol_a = solve_paper2_coupled_case(
        size=size_a,
        p=p,
        mu_mech=mu_mech,
        heat_nx=hc_a["heat_nx"],
        heat_ny=hc_a["heat_ny"],
        heat_nz=hc_a["heat_nz"],
        heat_degree=hc_a["heat_degree"],
        Nz_quad_T1=hc_a["Nz_quad_T1"],
        T1_cg_degree=hc_a["T1_cg_degree"],
        bc_type=bc_type,
    )
    _assert_converged_coupled_record(sol_a, context=f"Reference diagnostic level A size={size_a}")

    sol_b = get_or_build_reference(
        label=label,
        bc_type=bc_type,
        ref_size=size_b,
        ref_p=p,
        mu_mech=mu_mech,
    )
    _assert_converged_coupled_record(sol_b, context=f"Reference level B size={size_b}")

    mesh_b = reference_mesh(sol_b)
    Vw_b = FunctionSpace(mesh_b, "CG", int(p))
    Vtheta_b = FunctionSpace(mesh_b, "CG", int(p))
    sqrt_area_b = float(np.sqrt(assemble(Constant(1.0) * Measure("dx", domain=mesh_b))))
    deg = max(5, int(p) + 2)

    comparison = dict(
        mode="coupled",
        label=label,
        bc_type=bc_type,
        size_a=int(size_a),
        size_b=int(size_b),
        p=int(p),
        heat_a=hc_a,
        heat_b=ref_heat_b,
        L2_RMS_Error_w_m=l2_rms_error_against_ref(sol_a["w"], sol_b["w"], Vw_b, sqrt_area_b, degree=deg),
        L2_RMS_Error_theta=l2_rms_error_against_ref(sol_a["theta"], sol_b["theta"], Vtheta_b, sqrt_area_b, degree=deg),
        Rel_L2_Error_w=l2_relative_error_against_ref(sol_a["w"], sol_b["w"], Vw_b, degree=deg),
        Rel_L2_Error_theta=l2_relative_error_against_ref(sol_a["theta"], sol_b["theta"], Vtheta_b, degree=deg),
    )

    out_json = os.path.join(
        PAPER2_OUTDIR,
        f"paper2_coupled_{_safe_filename_part(label)}_{_safe_filename_part(bc_type)}_{STUDY_TAG}_reference_comparison.json",
    )
    with open(out_json, "w") as f:
        json.dump(_json_safe(comparison), f, indent=2, sort_keys=True)

    print("\nREFERENCE-LEVEL COMPARISON")
    for key, value in comparison.items():
        print(f"{key}: {value}")
    print(f"Saved reference comparison: {out_json}")
    return sol_a, sol_b, comparison


# ------------------------------------------------------------
# Driver utilities
# ------------------------------------------------------------

def run_smoke_test(case_label="Isotropic", *, bc_type=MECHANICAL_BC_TYPE):
    """
    Very cheap structure test: small reference and only two coarse levels.
    This is not the final study; it checks that the synchronized driver works.
    """
    global REF_SIZE, REF_P, REF_HEAT_NX, REF_HEAT_NY, REF_HEAT_NZ, REF_HEAT_DEGREE, REF_T1_CG_DEGREE, REF_NZ_QUAD_T1

    old = (REF_SIZE, REF_P, REF_HEAT_NX, REF_HEAT_NY, REF_HEAT_NZ, REF_HEAT_DEGREE, REF_T1_CG_DEGREE, REF_NZ_QUAD_T1)
    try:
        REF_SIZE = 16
        REF_P = 3
        REF_HEAT_NX, REF_HEAT_NY, REF_HEAT_NZ = 16, 8, 4
        REF_HEAT_DEGREE = 3
        REF_T1_CG_DEGREE = 3
        REF_NZ_QUAD_T1 = 16

        mu_mech = PLATE_CASES[case_label]["mu_mech"]
        df_cpl, ref_cpl = gather_full_asymptotic_coupled_convergence_data(
            mu_mech=mu_mech,
            resolutions=[2, 4],
            degree_pairs=[dict(p=2, heat_degree=2, T1_cg_degree=2)],
            ref_size=REF_SIZE,
            ref_p=REF_P,
            label=f"{case_label}_smoke",
            bc_type=bc_type,
        )
        print(f"\nCOUPLED SYNCHRONIZED SMOKE {case_label.upper()} ({bc_type})")
        df_valid = valid_converged_coupled_rows(df_cpl)
        if not df_valid.empty:
            print(df_valid[[c for c in TABLE_COLS if c in df_valid.columns]].to_string(index=False, float_format="%.6e"))
        return df_cpl, ref_cpl
    finally:
        (REF_SIZE, REF_P, REF_HEAT_NX, REF_HEAT_NY, REF_HEAT_NZ, REF_HEAT_DEGREE, REF_T1_CG_DEGREE, REF_NZ_QUAD_T1) = old


def run_case(case_label, *, bc_type=MECHANICAL_BC_TYPE):
    """Run the synchronized synchronized coupled refinement study for one material case."""
    mu_mech = PLATE_CASES[case_label]["mu_mech"]
    df_cpl, ref_cpl = gather_full_asymptotic_coupled_convergence_data(
        mu_mech=mu_mech,
        resolutions=PLATE_RESOLUTIONS,
        degree_pairs=DEGREE_PAIRS,
        ref_size=REF_SIZE,
        ref_p=REF_P,
        label=case_label,
        bc_type=bc_type,
    )

    print(f"\nCOUPLED SYNCHRONIZED {case_label.upper()} ({bc_type})")
    df_valid = valid_converged_coupled_rows(df_cpl)
    if df_valid.empty:
        print("No valid converged coupled rows available.")
    else:
        print(df_valid[[c for c in TABLE_COLS if c in df_valid.columns]].to_string(index=False, float_format="%.6e"))
    return df_cpl, ref_cpl


def replot_from_saved_csv(*, bc_type=MECHANICAL_BC_TYPE):
    frames = []
    for case_label in CASES_TO_RUN:
        path = convergence_csv_path(case_label, bc_type)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing coupled convergence CSV: {path}")
        frames.append(pd.read_csv(path))

    df_all = pd.concat(frames, ignore_index=True)
    df_valid = valid_converged_coupled_rows(df_all)

    plot_paper2_synchronized_coupled_convergence(
        df_valid,
        ycol="L2_RMS_Error_w_m",
        ylabel=r"RMS $L^2(\Omega)$ error in $w_h$ [m]",
        x_mode="h",
        filename_base=f"paper2_sync_coupled_w_convergence_mesh_size_{STUDY_TAG}",
        rate_fit_slice=slice(None),
    )

    plot_paper2_synchronized_coupled_convergence(
        df_valid,
        ycol="L2_RMS_Error_theta",
        ylabel=r"RMS $L^2(\Omega)$ error in $T_1$",
        x_mode="h",
        filename_base=f"paper2_sync_coupled_T1_convergence_mesh_size_{STUDY_TAG}",
        rate_fit_slice=slice(None),
    )
    return df_all


def run_full_study(*, bc_type=MECHANICAL_BC_TYPE):
    print_refinement_path()

    case_outputs = []
    histories = []
    for case_label in CASES_TO_RUN:
        df_case, ref_case = run_case(case_label, bc_type=bc_type)
        case_outputs.append((case_label, df_case, ref_case))
        histories.append(history_dataframe_from_ref(ref_case, case_label))

    df_all = pd.concat([item[1] for item in case_outputs], ignore_index=True)
    df_valid = valid_converged_coupled_rows(df_all)

    combined_csv = os.path.join(
        PAPER2_OUTDIR,
        f"paper2_coupled_synchronized_all_cases_{_safe_filename_part(bc_type)}_{STUDY_TAG}.csv",
    )
    df_all.to_csv(combined_csv, index=False)
    print(f"\nSaved combined CSV: {combined_csv}")

    print_finest_converged_table(
        df_valid,
        title="FINEST-MESH CONVERGED SYNCHRONIZED COUPLED ROWS",
    )
    print_rate_summary(df_valid)

    plot_paper2_synchronized_coupled_convergence(
        df_valid,
        ycol="L2_RMS_Error_w_m",
        ylabel=r"RMS $L^2(\Omega)$ error in $w_h$ [m]",
        x_mode="h",
        filename_base=f"paper2_sync_coupled_w_convergence_mesh_size_{STUDY_TAG}",
        rate_fit_slice=slice(None),
    )

    plot_paper2_synchronized_coupled_convergence(
        df_valid,
        ycol="L2_RMS_Error_theta",
        ylabel=r"RMS $L^2(\Omega)$ error in $T_1$",
        x_mode="h",
        filename_base=f"paper2_sync_coupled_T1_convergence_mesh_size_{STUDY_TAG}",
        rate_fit_slice=slice(None),
    )

    plot_paper2_coupled_history_iso_ortho_combined(
        histories,
        filename_base=f"paper2_sync_coupled_reference_update_history_{STUDY_TAG}",
    )

    if RUN_REFERENCE_COMPARISON and "Isotropic" in PLATE_CASES:
        compare_two_reference_levels(
            PLATE_CASES["Isotropic"]["mu_mech"],
            size_a=max(PLATE_RESOLUTIONS),
            size_b=REF_SIZE,
            p=REF_P,
            label=f"Isotropic_sync_refcheck_{max(PLATE_RESOLUTIONS)}_vs_{REF_SIZE}_{bc_type}",
            bc_type=bc_type,
        )

    return df_all, df_valid, case_outputs


# ------------------------------------------------------------
# Execute synchronized coupled study
# ------------------------------------------------------------

# Optional quick structure test before the full run:
# df_smoke, ref_smoke = run_smoke_test("Isotropic", bc_type=MECHANICAL_BC_TYPE)

if RUN_FULL_STUDY:
    df_cpl_all, df_cpl_valid, case_outputs = run_full_study(bc_type=MECHANICAL_BC_TYPE)

# %% Cell 17 | id: 0822f913-a865-44e8-ae4d-7e9e0069a315


# %% Cell 18 | id: 40a5422d-8535-4818-a20d-5b11b2f5076c


# %% Cell 19 | id: 8f4407a2-a935-467a-a413-1b476f4b4191


# %% Cell 20 | id: 459b664b-8508-4e2e-ad82-edb3c2c2364e


# %% Cell 21 | id: b9afdd60-111e-437d-8a9f-66eafa8acc7d


# %% Cell 22 | id: a867fd48-8ecd-4cf6-9f5d-a343bd28186f


# %% Cell 23 | id: 5f869bee-a10e-4cb4-b055-64ec03994c34


# %% Cell 24 | id: 21acf622-e415-470f-b6b0-9a7b2fe5c000


# %% Cell 25 | id: 915e5ecd-086b-4d1a-b8d2-c28b8412cc72


# %% Cell 26 | id: d8fdbf68-8091-4dd1-8c80-68619613d946


# %% Cell 27 | id: 6678dc24-5bda-4a1f-8723-8115e532c3ff


# %% Cell 28 | id: eba055e6-ceb3-4f7b-b2cd-1348e9e2e99b


# %% Cell 29 | id: e290fecb-3368-4198-bf5c-6d310e290c3f


# %% Cell 30 | id: 30fe04d6-666f-4620-915e-def88c1b2512
# ============================================================
# Paper 2 thermomechanical FOM convergence with refined FOM reference
# ============================================================

PAPER2_OUTDIR = "paper2_fom_convergence"
os.makedirs(PAPER2_OUTDIR, exist_ok=True)

MECHANICAL_BC_TYPE = "simply_supported"
RESOLUTIONS_SMOKE = [8, 11, 16]
RESOLUTIONS_FINAL = [8, 11, 16, 23, 32, 45, 64]
RESOLUTIONS_COUPLED_FINAL = [8, 11, 16, 23, 32, 45, 64]
P_VALUES = [2, 3]

# ------------------------------------------------------------
# User-decision block
# ------------------------------------------------------------
RESUME_RUNS = True
FORCE_RERUN = False
RUN_REFERENCE_COMPARISON = True

# Current working reference setting.
REF_SIZE, REF_P = 100, 4

# Current working fixed heat discretization.
HEAT_NX, HEAT_NY, HEAT_NZ = 64, 32, 16
HEAT_DEGREE = 1
NZ_QUAD_T1 = 20

# General coarse/reference comparison size.
REFERENCE_COMPARISON_COARSE_SIZE = max(RESOLUTIONS_COUPLED_FINAL)

REFERENCE_CACHE_VERSION = 2
CONVERGENCE_TABLE_VERSION = 2

FIXED_HEAT_CONTROLS = dict(
    heat_nx=int(HEAT_NX),
    heat_ny=int(HEAT_NY),
    heat_nz=int(HEAT_NZ),
    heat_degree=int(HEAT_DEGREE),
    Nz_quad_T1=int(NZ_QUAD_T1),
)
REF_HEAT_CONTROLS = dict(FIXED_HEAT_CONTROLS)

# Settings-dependent label used in saved CSV/figure/reference-comparison names.
STUDY_TAG = (
    f"refN{int(REF_SIZE)}_p{int(REF_P)}"
    f"_heat{int(HEAT_NX)}x{int(HEAT_NY)}x{int(HEAT_NZ)}"
    f"_pT{int(HEAT_DEGREE)}"
    f"_qT1{int(NZ_QUAD_T1)}"
    f"_v{int(CONVERGENCE_TABLE_VERSION)}"
)

PLATE_CASES = {
    "Isotropic": dict(mu_mech=[1.0e4, 1.0e4, 0.3e4, 0.35e4, 1.0e3, -10000.0]),
    "Orthotropic": dict(mu_mech=[2.0e4, 1.0e4, 0.25e4, 0.30e4, 1.0e3, -10000.0]),
}

THERMAL_PARAMS = dict(
    T_amb=27.0 + 273.15,
    T_sub=24.0 + 273.15,
    h_con=10.0,
    eps_r=0.90,
    q_s=700.0,
    h_c_cont=200.0,
    h_c_gap=5.0,
    eta_c=50.0,
    w_contact=-0.1,
    kx=0.35,
    ky=0.35,
    kz=0.35,
    alpha1=1.3e-4,
    alpha2=1.3e-4,
    rho=1050.0,
)

COUPLING_PARAMS = dict(
    coupling_omega=0.7,
    coupling_tol_w=1e-6,
    coupling_tol_T1=1e-6,
    coupling_max_iters=40,
)

TABLE_COLS = """
mode case bc_type p size h inv_h plate_DoFs heat_DoFs theta_DoFs Total_DoFs
sqrt_plate_DoFs sqrt_Total_DoFs L2_RMS_Error_w_m Rel_L2_Error_w
L2_RMS_Error_theta Rel_L2_Error_theta coupled_iters coupled_converged
stopped_by_max_iters termination_reason err_w err_theta
""".split()

NEWTON_SOLVER_PARAMETERS["newton_solver"].update(
    linear_solver="petsc",
    absolute_tolerance=1e-4,
    relative_tolerance=1e-5,
    maximum_iterations=80,
    relaxation_parameter=1.0,
)


# ------------------------------------------------------------
# Mesh/thermal controls
# ------------------------------------------------------------

def heat_controls_from_size(size, *, ref=False, fixed=True):
    """
    Heat controls for convergence studies.

    fixed=True keeps the 3D heat discretization fixed so that the reported
    refinement behavior reflects the plate/FOM discretization rather than
    simultaneous heat-mesh refinement.
    """
    if ref:
        return dict(REF_HEAT_CONTROLS)

    if fixed:
        return dict(FIXED_HEAT_CONTROLS)

    return dict(
        heat_nx=int(size),
        heat_ny=max(2, int(round(size / 2))),
        heat_nz=max(8, int(round(size / 4))),
        heat_degree=int(HEAT_DEGREE),
        Nz_quad_T1=max(10, min(24, int(round(size / 4)) + 4)),
    )


# ------------------------------------------------------------
# Solver construction and single-case solve
# ------------------------------------------------------------

def make_solver(size, p, mu_mech, *, load_type="uniform", bc_type=MECHANICAL_BC_TYPE):
    solver = GeneralMultiphysicsSolver(study_case=1)
    solver.size = int(size)
    solver.degree = int(p)
    solver.load_type = load_type
    solver.bc_type = bc_type
    solver.define_domain(n_vert=0, n_horiz=0, plot_subdomains=False)
    solver.set_mu(mu_mech)
    solver.set_rom_thermal_parameters(**THERMAL_PARAMS)
    return solver


def copy_function(f):
    out = Function(f.function_space())
    out.assign(f)
    return out


def solve_paper2_case(
    *,
    mode,
    size,
    p,
    mu_mech,
    heat_nx,
    heat_ny,
    heat_nz,
    heat_degree,
    Nz_quad_T1,
    T1_cg_degree,
    bc_type=MECHANICAL_BC_TYPE,
):
    """
    Solve one Paper 2 FOM case.

    The one-way solver path is kept available because the framework still supports
    one-way thermomechanics. However, the revised refinement-study drivers below
    intentionally use mode='coupled' only.
    """
    solver = make_solver(size=size, p=p, mu_mech=mu_mech, bc_type=bc_type)

    heat_kwargs = dict(
        thermal_on=True,
        heat_nx=int(heat_nx),
        heat_ny=int(heat_ny),
        heat_nz=int(heat_nz),
        heat_degree=int(heat_degree),
        Nz_quad_T1=int(Nz_quad_T1),
        T1_cg_degree=int(T1_cg_degree),
        return_mode="global",
    )
    common = dict(solver=solver, mode=mode, bc_type=bc_type)

    if mode == "one_way":
        heat = solver.solve_rom_sample(mu_mech, coupled_on=False, **heat_kwargs)
        w = copy_function(solver.w_contact_global)
        theta = copy_function(solver.T1_from_heat)
        return dict(
            common,
            w=w,
            theta=theta,
            heat=heat,
            history=None,
            converged=True,
            stopped_by_max_iters=False,
            termination_reason="one_way",
            err_w=np.nan,
            err_T1=np.nan,
            iters=1,
            w_dofs=int(w.function_space().dim()),
            heat_dofs=int(heat["Vt"].dim()),
            theta_dofs=int(theta.function_space().dim()),
            total_dofs=int(w.function_space().dim() + heat["Vt"].dim() + theta.function_space().dim()),
            heat_nx=int(heat_nx),
            heat_ny=int(heat_ny),
            heat_nz=int(heat_nz),
            heat_degree=int(heat_degree),
            Nz_quad_T1=int(Nz_quad_T1),
            T1_cg_degree=int(T1_cg_degree),
        )

    if mode == "coupled":
        out = solver.solve_rom_sample(
            mu_mech,
            coupled_on=True,
            coupling_omega=COUPLING_PARAMS["coupling_omega"],
            coupling_tol_w=COUPLING_PARAMS["coupling_tol_w"],
            coupling_tol_T1=COUPLING_PARAMS["coupling_tol_T1"],
            coupling_max_iters=COUPLING_PARAMS["coupling_max_iters"],
            coupling_verbose=False,
            w0=0.0,
            return_coupled_dict=True,
            **heat_kwargs,
        )
        return dict(
            common,
            w=copy_function(out["w"]),
            theta=copy_function(out["T1"]),
            heat=out["heat"],
            history=out["history"],
            converged=bool(out["converged"]),
            stopped_by_max_iters=bool(out["stopped_by_max_iters"]),
            termination_reason=str(out["termination_reason"]),
            err_w=float(out["err_w"]),
            err_T1=float(out["err_T1"]),
            iters=int(out["iters"]),
            w_dofs=int(out["w_dofs"]),
            heat_dofs=int(out["heat_dofs"]),
            theta_dofs=int(out["T1_dofs"]),
            total_dofs=int(out["total_coupled_dofs"]),
            heat_nx=int(heat_nx),
            heat_ny=int(heat_ny),
            heat_nz=int(heat_nz),
            heat_degree=int(heat_degree),
            Nz_quad_T1=int(Nz_quad_T1),
            T1_cg_degree=int(T1_cg_degree),
        )

    raise ValueError("mode must be 'one_way' or 'coupled'.")


# ------------------------------------------------------------
# Error computation against refined reference
# ------------------------------------------------------------

def lift_field_to_reference_mesh(field, Vref, degree=5):
    """Interpolate a scalar plate field from its own mesh to the reference mesh."""
    try:
        field.set_allow_extrapolation(True)
    except Exception:
        pass

    class FieldOnReferenceMesh(UserExpression):
        def __init__(self, f, **kwargs):
            super().__init__(**kwargs)
            self.f = f

        def eval(self, values, x):
            values[0] = float(self.f(Point(float(x[0]), float(x[1]))))

        def value_shape(self):
            return ()

    return interpolate(FieldOnReferenceMesh(field, degree=degree), Vref)


def l2_rms_error_against_ref(field, ref_field, Vref, sqrt_area, degree=5):
    """
    RMS L2 error against the refined numerical FOM reference.

    This is not an exact-solution error.
    """
    field_refmesh = lift_field_to_reference_mesh(field, Vref, degree=degree)
    ref_projected = project(ref_field, Vref)
    return float(errornorm(ref_projected, field_refmesh, norm_type="L2", degree_rise=3) / sqrt_area)


def l2_relative_error_against_ref(field, ref_field, Vref, degree=5, eps=1e-14):
    """Relative L2 error against the refined numerical FOM reference."""
    dx_ref = Measure("dx", domain=Vref.mesh())
    field_refmesh = lift_field_to_reference_mesh(field, Vref, degree=degree)
    ref_projected = project(ref_field, Vref)

    abs_err = float(errornorm(ref_projected, field_refmesh, norm_type="L2", degree_rise=3))
    ref_norm = float(np.sqrt(max(float(assemble(ref_projected * ref_projected * dx_ref)), 0.0)))
    return abs_err / max(ref_norm, float(eps))


# ------------------------------------------------------------
# Checkpoint helpers
# ------------------------------------------------------------

def convergence_csv_path(mode, label, bc_type):
    return os.path.join(
        PAPER2_OUTDIR,
        f"paper2_{mode}_{label}_{bc_type}_convergence_{STUDY_TAG}.csv",
    )


def _as_bool(value, default=False):
    """Robust boolean conversion for bools, numpy bools, and CSV string values."""
    if value is None:
        return bool(default)

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, np.integer)):
        return bool(value)

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return bool(default)
        return bool(int(value))

    if isinstance(value, str):
        value_l = value.strip().lower()
        if value_l in ("true", "t", "1", "yes", "y"):
            return True
        if value_l in ("false", "f", "0", "no", "n", "nan", "none", ""):
            return False

    return bool(default)


def _bool_series(series, default=False):
    """Robust boolean mask for pandas Series that may contain strings from CSV."""
    return series.apply(lambda v: _as_bool(v, default=default)).astype(bool)


def _coupled_solution_is_converged(record):
    """True only if a coupled record/solution reached the coupled stopping criterion."""
    return (
        str(record.get("mode", "coupled")) == "coupled"
        and _as_bool(record.get("converged", record.get("coupled_converged", False)))
        and not _as_bool(record.get("stopped_by_max_iters", False))
    )


def _row_is_valid_converged_coupled(row):
    """True only for coupled rows that are valid for manuscript tables/plots."""
    try:
        if str(row.get("mode", "")) != "coupled":
            return False
        if not _as_bool(row.get("coupled_converged", False)):
            return False
        if _as_bool(row.get("stopped_by_max_iters", False)):
            return False
        if "reported" in row.index and not _as_bool(row.get("reported", True), default=True):
            return False
        if "stopping_norm" in row.index and str(row.get("stopping_norm", "")) != "L2_Omega":
            return False

        finite_cols = [
            "h",
            "L2_RMS_Error_w_m",
            "Rel_L2_Error_w",
            "L2_RMS_Error_theta",
            "Rel_L2_Error_theta",
            "err_w",
            "err_theta",
        ]
        return all(col not in row or np.isfinite(float(row[col])) for col in finite_cols)

    except Exception:
        return False


def load_existing_convergence_rows(out_csv):
    """
    Load old checkpoint rows.

    Only valid converged coupled rows are treated as completed. Failed or
    non-converged rows are kept in the CSV for traceability, but they are not
    allowed to block reruns.
    """
    if not (RESUME_RUNS and os.path.exists(out_csv) and not FORCE_RERUN):
        return None, set()

    df = pd.read_csv(out_csv)
    if df.empty:
        return None, set()

    required_key_cols = ["mode", "case", "bc_type", "p", "size"]
    valid_keys = {
        (str(row["mode"]), str(row["case"]), str(row["bc_type"]), int(row["p"]), int(row["size"]))
        for _, row in df.iterrows()
        if all(col in row.index for col in required_key_cols) and _row_is_valid_converged_coupled(row)
    }

    print(f"\nResuming from existing table: {out_csv}")
    print(f"Existing rows: {len(df)}")
    print(f"Valid converged coupled rows that will be skipped: {len(valid_keys)}")

    return df, valid_keys


def save_convergence_checkpoint(new_records, existing_df, out_csv):
    if not new_records:
        return existing_df if existing_df is not None else pd.DataFrame()

    new_df = pd.DataFrame.from_records(new_records)
    save_df = pd.concat([existing_df, new_df], ignore_index=True) if existing_df is not None and not existing_df.empty else new_df

    save_df = (
        save_df.drop_duplicates(["mode", "case", "bc_type", "p", "size"], keep="last")
        .sort_values(["mode", "case", "bc_type", "p", "h"], ascending=[True, True, True, True, False])
        .reset_index(drop=True)
    )
    save_df.to_csv(out_csv, index=False)
    print(f"Checkpoint saved: {out_csv}")

    return save_df


# ------------------------------------------------------------
# Reference-cache helpers
# ------------------------------------------------------------

def _json_safe(obj):
    """Convert numpy/scalar objects into JSON-safe Python objects."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _safe_filename_part(text):
    text = str(text)
    for ch in [" ", "/", "\\", ":", ";", ",", "(", ")", "[", "]", "{", "}"]:
        text = text.replace(ch, "_")
    return text


def reference_cache_key(*, mode, label, bc_type, ref_size, ref_p, mu_mech, ref_heat, fixed_heat):
    """Build a unique reference-cache key."""
    payload = dict(
        cache_version=int(REFERENCE_CACHE_VERSION),
        convergence_table_version=int(CONVERGENCE_TABLE_VERSION),
        study_tag=str(STUDY_TAG),
        mode=str(mode),
        case=str(label),
        bc_type=str(bc_type),
        ref_size=int(ref_size),
        ref_p=int(ref_p),
        mu_mech=[float(v) for v in mu_mech],
        thermal_params=_json_safe(THERMAL_PARAMS),
        ref_heat=_json_safe(ref_heat),
        fixed_heat=bool(fixed_heat),
        coupling_params=_json_safe(COUPLING_PARAMS) if mode == "coupled" else None,
        mechanical_bc_type=str(MECHANICAL_BC_TYPE),
        stopping_norm="L2_Omega",
        reported_refinement_branch="coupled_only",
    )
    payload_json = json.dumps(payload, sort_keys=True, indent=2)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:16], payload


def reference_cache_paths(*, mode, label, bc_type, ref_size, ref_p, cache_key):
    stem = (
        f"paper2_ref_{_safe_filename_part(mode)}_{_safe_filename_part(label)}"
        f"_{_safe_filename_part(bc_type)}_size{int(ref_size)}_p{int(ref_p)}_{cache_key}"
    )
    return dict(
        h5=os.path.join(PAPER2_OUTDIR, stem + ".h5"),
        json=os.path.join(PAPER2_OUTDIR, stem + ".json"),
    )


def _reference_meta(ref, w_save, theta_save, payload):
    return dict(
        payload=payload,
        history=ref.get("history", None),
        converged=bool(ref.get("converged", True)),
        stopped_by_max_iters=bool(ref.get("stopped_by_max_iters", False)),
        termination_reason=str(ref.get("termination_reason", "cached_reference")),
        err_w=float(ref.get("err_w", np.nan)),
        err_T1=float(ref.get("err_T1", np.nan)),
        iters=int(ref.get("iters", 1)),
        w_dofs=int(ref.get("w_dofs", w_save.function_space().dim())),
        heat_dofs=int(ref.get("heat_dofs", 0)),
        theta_dofs=int(ref.get("theta_dofs", theta_save.function_space().dim())),
        total_dofs=int(ref.get("total_dofs", 0)),
    )


def _assert_converged_coupled_record(record, *, context):
    """Stop immediately if a coupled reference/run did not converge."""
    if not bool(record.get("converged", False)):
        raise RuntimeError(
            f"{context} did not converge: "
            f"iters={record.get('iters', None)}, "
            f"err_w={record.get('err_w', np.nan)}, "
            f"err_T1={record.get('err_T1', np.nan)}, "
            f"termination={record.get('termination_reason', None)}"
        )

    if bool(record.get("stopped_by_max_iters", False)):
        raise RuntimeError(
            f"{context} reached max coupling iterations and must not be reported: "
            f"iters={record.get('iters', None)}, "
            f"err_w={record.get('err_w', np.nan)}, "
            f"err_T1={record.get('err_T1', np.nan)}, "
            f"termination={record.get('termination_reason', None)}"
        )


def save_reference_cache(ref, *, paths, payload, ref_p):
    """Save reference mesh, displacement, thermal driver, and metadata."""
    if payload.get("mode") == "coupled":
        _assert_converged_coupled_record(ref, context="Coupled reference before cache save")

    mesh = ref["solver"].mesh
    Vw_ref = FunctionSpace(mesh, "CG", int(ref_p))
    Vtheta_ref = FunctionSpace(mesh, "CG", int(ref_p))

    w_save = project(ref["w"], Vw_ref)
    theta_save = project(ref["theta"], Vtheta_ref)

    h5 = HDF5File(mesh.mpi_comm(), paths["h5"], "w")
    for name, obj in [("/mesh", mesh), ("/w", w_save), ("/theta", theta_save)]:
        h5.write(obj, name)
    h5.close()

    with open(paths["json"], "w") as f:
        json.dump(_json_safe(_reference_meta(ref, w_save, theta_save, payload)), f, indent=2, sort_keys=True)

    print("Saved reference cache:")
    print(f"  {paths['h5']}")
    print(f"  {paths['json']}")


def load_reference_cache(*, paths, ref_p):
    """Load cached reference mesh and fields."""
    if not (os.path.exists(paths["h5"]) and os.path.exists(paths["json"])):
        return None

    try:
        mesh = Mesh()
        h5 = HDF5File(MPI.comm_world, paths["h5"], "r")
        h5.read(mesh, "/mesh", False)

        Vw_ref = FunctionSpace(mesh, "CG", int(ref_p))
        Vtheta_ref = FunctionSpace(mesh, "CG", int(ref_p))
        w_ref = Function(Vw_ref)
        theta_ref = Function(Vtheta_ref)

        h5.read(w_ref, "/w")
        h5.read(theta_ref, "/theta")
        h5.close()

        with open(paths["json"], "r") as f:
            meta = json.load(f)

    except Exception as exc:
        print(f"Reference cache could not be loaded and will be rebuilt: {exc}")
        return None

    ref = dict(
        solver=None,
        mesh=mesh,
        w=w_ref,
        theta=theta_ref,
        heat=None,
        history=meta.get("history", None),
        converged=bool(meta.get("converged", True)),
        stopped_by_max_iters=bool(meta.get("stopped_by_max_iters", False)),
        termination_reason=str(meta.get("termination_reason", "loaded_reference_cache")),
        err_w=float(meta.get("err_w", np.nan)),
        err_T1=float(meta.get("err_T1", np.nan)),
        iters=int(meta.get("iters", 1)),
        w_dofs=int(meta.get("w_dofs", w_ref.function_space().dim())),
        heat_dofs=int(meta.get("heat_dofs", 0)),
        theta_dofs=int(meta.get("theta_dofs", theta_ref.function_space().dim())),
        total_dofs=int(meta.get("total_dofs", 0)),
    )

    if meta.get("payload", {}).get("mode") == "coupled":
        _assert_converged_coupled_record(ref, context="Loaded coupled reference cache")

    print("Loaded cached reference:")
    print(f"  {paths['h5']}")

    return ref


def get_or_build_reference(*, mode, label, bc_type, ref_size, ref_p, mu_mech, fixed_heat=True):
    """
    Build or load the refined FOM numerical reference.

    The revised reported refinement study uses coupled references only.
    """
    if mode != "coupled":
        raise ValueError(
            "The revised Paper 2 refinement study is coupled-only. "
            "Use mode='coupled' for reference construction."
        )

    ref_heat = heat_controls_from_size(ref_size, ref=True, fixed=fixed_heat)
    cache_key, payload = reference_cache_key(
        mode=mode,
        label=label,
        bc_type=bc_type,
        ref_size=ref_size,
        ref_p=ref_p,
        mu_mech=mu_mech,
        ref_heat=ref_heat,
        fixed_heat=fixed_heat,
    )
    paths = reference_cache_paths(
        mode=mode,
        label=label,
        bc_type=bc_type,
        ref_size=ref_size,
        ref_p=ref_p,
        cache_key=cache_key,
    )

    if RESUME_RUNS and not FORCE_RERUN:
        cached = load_reference_cache(paths=paths, ref_p=ref_p)
        if cached is not None:
            return cached

    print("\n" + "=" * 90)
    print(f"BUILDING COUPLED REFERENCE: {label}, bc={bc_type}, N_Omega={ref_size}, p={ref_p}")
    print("=" * 90)

    ref = solve_paper2_case(
        mode=mode,
        size=ref_size,
        p=ref_p,
        mu_mech=mu_mech,
        heat_nx=ref_heat["heat_nx"],
        heat_ny=ref_heat["heat_ny"],
        heat_nz=ref_heat["heat_nz"],
        heat_degree=ref_heat["heat_degree"],
        Nz_quad_T1=ref_heat["Nz_quad_T1"],
        T1_cg_degree=ref_p,
        bc_type=bc_type,
    )

    _assert_converged_coupled_record(ref, context="Newly built coupled reference")
    save_reference_cache(ref, paths=paths, payload=payload, ref_p=ref_p)
    return ref


def reference_mesh(ref):
    """Return the mesh from either a live reference solve or a cached reference."""
    return ref["solver"].mesh if ref.get("solver", None) is not None else ref["mesh"]


# ------------------------------------------------------------
# Convergence gathering
# ------------------------------------------------------------

def _success_record(mode, label, bc_type, p, res, sol, errors):
    ew_rms, etheta_rms, ew_rel, etheta_rel = errors
    return dict(
        mode=str(mode),
        case=str(label),
        bc_type=str(bc_type),
        p=int(p),
        size=int(res),
        h=float(sol["solver"].mesh.hmax()),
        inv_h=float(1.0 / sol["solver"].mesh.hmax()),
        plate_DoFs=int(sol["w_dofs"]),
        heat_DoFs=int(sol["heat_dofs"]),
        theta_DoFs=int(sol["theta_dofs"]),
        Total_DoFs=int(sol["total_dofs"]),
        sqrt_plate_DoFs=float(np.sqrt(sol["w_dofs"])),
        sqrt_Total_DoFs=float(np.sqrt(sol["total_dofs"])),
        heat_nx=int(sol["heat_nx"]),
        heat_ny=int(sol["heat_ny"]),
        heat_nz=int(sol["heat_nz"]),
        heat_degree=int(sol["heat_degree"]),
        Nz_quad_T1=int(sol["Nz_quad_T1"]),
        T1_cg_degree=int(sol["T1_cg_degree"]),
        L2_RMS_Error_w_m=float(ew_rms),
        L2_RMS_Error_theta=float(etheta_rms),
        Rel_L2_Error_w=float(ew_rel),
        Rel_L2_Error_theta=float(etheta_rel),
        coupled_iters=int(sol["iters"]),
        coupled_converged=bool(sol["converged"]),
        stopped_by_max_iters=bool(sol["stopped_by_max_iters"]),
        termination_reason=str(sol["termination_reason"]),
        err_w=float(sol["err_w"]),
        err_theta=float(sol["err_T1"]),
        stopping_norm="L2_Omega",
        reported=True,
    )


def _failed_record(mode, label, bc_type, p, res, hc, exc):
    return dict(
        mode=str(mode),
        case=str(label),
        bc_type=str(bc_type),
        p=int(p),
        size=int(res),
        h=np.nan,
        inv_h=np.nan,
        plate_DoFs=np.nan,
        heat_DoFs=np.nan,
        theta_DoFs=np.nan,
        Total_DoFs=np.nan,
        sqrt_plate_DoFs=np.nan,
        sqrt_Total_DoFs=np.nan,
        heat_nx=int(hc["heat_nx"]),
        heat_ny=int(hc["heat_ny"]),
        heat_nz=int(hc["heat_nz"]),
        heat_degree=int(hc["heat_degree"]),
        Nz_quad_T1=int(hc["Nz_quad_T1"]),
        T1_cg_degree=int(p),
        L2_RMS_Error_w_m=np.nan,
        L2_RMS_Error_theta=np.nan,
        Rel_L2_Error_w=np.nan,
        Rel_L2_Error_theta=np.nan,
        coupled_iters=np.nan,
        coupled_converged=False,
        stopped_by_max_iters=False,
        termination_reason=f"failed_or_not_reported: {exc}",
        err_w=np.nan,
        err_theta=np.nan,
        stopping_norm="L2_Omega",
        reported=False,
    )


def _excluded_record_from_solution(mode, label, bc_type, p, res, sol):
    """Save a non-converged coupled run for traceability, but mark it as not reportable."""
    return dict(
        mode=str(mode),
        case=str(label),
        bc_type=str(bc_type),
        p=int(p),
        size=int(res),
        h=float(sol["solver"].mesh.hmax()),
        inv_h=float(1.0 / sol["solver"].mesh.hmax()),
        plate_DoFs=int(sol["w_dofs"]),
        heat_DoFs=int(sol["heat_dofs"]),
        theta_DoFs=int(sol["theta_dofs"]),
        Total_DoFs=int(sol["total_dofs"]),
        sqrt_plate_DoFs=float(np.sqrt(sol["w_dofs"])),
        sqrt_Total_DoFs=float(np.sqrt(sol["total_dofs"])),
        heat_nx=int(sol["heat_nx"]),
        heat_ny=int(sol["heat_ny"]),
        heat_nz=int(sol["heat_nz"]),
        heat_degree=int(sol["heat_degree"]),
        Nz_quad_T1=int(sol["Nz_quad_T1"]),
        T1_cg_degree=int(sol["T1_cg_degree"]),
        L2_RMS_Error_w_m=np.nan,
        L2_RMS_Error_theta=np.nan,
        Rel_L2_Error_w=np.nan,
        Rel_L2_Error_theta=np.nan,
        coupled_iters=int(sol["iters"]),
        coupled_converged=bool(sol["converged"]),
        stopped_by_max_iters=bool(sol["stopped_by_max_iters"]),
        termination_reason=f"not_reported_nonconverged: {sol['termination_reason']}",
        err_w=float(sol["err_w"]),
        err_theta=float(sol["err_T1"]),
        stopping_norm="L2_Omega",
        reported=False,
    )


def valid_converged_coupled_rows(df):
    """
    Keep only rows admissible for manuscript tables and plots.

    Excludes:
        - one-way rows,
        - failed rows,
        - non-converged coupled rows,
        - rows stopped by max iterations,
        - rows with missing/non-finite error values.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out = out[out["mode"].astype(str) == "coupled"]

    if "coupled_converged" in out.columns:
        out = out[_bool_series(out["coupled_converged"], default=False)]
    if "stopped_by_max_iters" in out.columns:
        out = out[~_bool_series(out["stopped_by_max_iters"], default=False)]
    if "reported" in out.columns:
        out = out[_bool_series(out["reported"], default=True)]
    if "stopping_norm" in out.columns:
        out = out[out["stopping_norm"].astype(str) == "L2_Omega"]

    for col in [
        "h",
        "L2_RMS_Error_w_m",
        "Rel_L2_Error_w",
        "L2_RMS_Error_theta",
        "Rel_L2_Error_theta",
        "err_w",
        "err_theta",
    ]:
        if col in out.columns:
            out = out[np.isfinite(out[col].to_numpy(dtype=float))]

    return out.reset_index(drop=True)


def gather_paper2_convergence_data(
    *,
    mode,
    mu_mech,
    resolutions=RESOLUTIONS_COUPLED_FINAL,
    p_values=P_VALUES,
    ref_size=REF_SIZE,
    ref_p=REF_P,
    label="case",
    bc_type=MECHANICAL_BC_TYPE,
    fixed_heat=True,
):
    """
    Gather coupled-only convergence data using a cached refined FOM reference.

    Non-converged coupled runs are saved as failed/non-reported rows, but they
    are excluded from all manuscript tables and plots.
    """
    if mode != "coupled":
        raise ValueError(
            "The revised Paper 2 refinement study must be coupled-only. "
            "Do not call gather_paper2_convergence_data with mode='one_way'."
        )

    out_csv = convergence_csv_path(mode, label, bc_type)
    existing_df, existing_keys = load_existing_convergence_rows(out_csv)

    ref = get_or_build_reference(
        mode=mode,
        label=label,
        bc_type=bc_type,
        ref_size=ref_size,
        ref_p=ref_p,
        mu_mech=mu_mech,
        fixed_heat=fixed_heat,
    )
    _assert_converged_coupled_record(ref, context=f"Coupled reference for {label}, bc={bc_type}")

    ref_mesh_obj = reference_mesh(ref)
    Vw_ref = FunctionSpace(ref_mesh_obj, "CG", int(ref_p))
    Vtheta_ref = FunctionSpace(ref_mesh_obj, "CG", int(ref_p))
    dx_ref = Measure("dx", domain=ref_mesh_obj)
    sqrt_area = float(np.sqrt(assemble(Constant(1.0) * dx_ref)))

    w_ref = project(ref["w"], Vw_ref)
    theta_ref = project(ref["theta"], Vtheta_ref)

    for res in resolutions:
        for p in p_values:
            if int(res) == int(ref_size) and int(p) == int(ref_p):
                continue

            case_key = (mode, label, bc_type, int(p), int(res))
            if case_key in existing_keys:
                print(f"SKIPPING valid cached row: mode={mode}, case={label}, bc={bc_type}, size={res}, p={p}")
                continue

            hc = heat_controls_from_size(res, ref=False, fixed=fixed_heat)
            print("\n" + "-" * 90)
            print(
                f"COUPLED | {label} | bc={bc_type} | size={res}, p={p}, "
                f"heat=({hc['heat_nx']},{hc['heat_ny']},{hc['heat_nz']}), "
                f"pT={hc['heat_degree']}, Nz_quad_T1={hc['Nz_quad_T1']}"
            )
            print("-" * 90)

            try:
                sol = solve_paper2_case(
                    mode=mode,
                    size=res,
                    p=p,
                    mu_mech=mu_mech,
                    heat_nx=hc["heat_nx"],
                    heat_ny=hc["heat_ny"],
                    heat_nz=hc["heat_nz"],
                    heat_degree=hc["heat_degree"],
                    Nz_quad_T1=hc["Nz_quad_T1"],
                    T1_cg_degree=p,
                    bc_type=bc_type,
                )

                if not _coupled_solution_is_converged(sol):
                    print(
                        f"EXCLUDED NON-CONVERGED RUN: case={label}, size={res}, p={p}, "
                        f"iters={sol.get('iters', None)}, err_w={sol.get('err_w', np.nan)}, "
                        f"err_T1={sol.get('err_T1', np.nan)}, termination={sol.get('termination_reason', None)}"
                    )
                    record = _excluded_record_from_solution(mode, label, bc_type, p, res, sol)

                else:
                    deg = max(5, int(p) + 2)
                    errors = (
                        l2_rms_error_against_ref(sol["w"], w_ref, Vw_ref, sqrt_area, degree=deg),
                        l2_rms_error_against_ref(sol["theta"], theta_ref, Vtheta_ref, sqrt_area, degree=deg),
                        l2_relative_error_against_ref(sol["w"], w_ref, Vw_ref, degree=deg),
                        l2_relative_error_against_ref(sol["theta"], theta_ref, Vtheta_ref, degree=deg),
                    )
                    record = _success_record(mode, label, bc_type, p, res, sol, errors)

            except Exception as exc:
                print(f"FAILED OR EXCLUDED: mode={mode}, case={label}, bc={bc_type}, size={res}, p={p}: {exc}")
                record = _failed_record(mode, label, bc_type, p, res, hc, exc)

            existing_df = save_convergence_checkpoint([record], existing_df, out_csv)

            if _row_is_valid_converged_coupled(pd.Series(record)):
                existing_keys.add(case_key)

    df = existing_df.copy() if existing_df is not None and not existing_df.empty else pd.DataFrame()

    if not df.empty:
        df = df.sort_values(
            ["mode", "case", "bc_type", "p", "h"],
            ascending=[True, True, True, True, False],
        ).reset_index(drop=True)

    df.to_csv(out_csv, index=False)
    df_valid = valid_converged_coupled_rows(df)

    print(f"\nSaved convergence table: {out_csv}")
    print(f"Total rows in CSV: {len(df)}")
    print(f"Valid converged coupled rows for reporting: {len(df_valid)}")

    return df, ref


# ------------------------------------------------------------
# Table helper
# ------------------------------------------------------------

def finest_converged_rows_for_table(df):
    """Return the finest valid coupled row for each case and p."""
    df_valid = valid_converged_coupled_rows(df)
    if df_valid.empty:
        return df_valid

    rows = [grp.sort_values("size", ascending=False).iloc[0] for _, grp in df_valid.groupby(["case", "p"])]
    return pd.DataFrame(rows).sort_values(["case", "p"]).reset_index(drop=True)


def print_finest_converged_table(df, title="FINEST CONVERGED COUPLED ROWS"):
    table_df = finest_converged_rows_for_table(df)

    print("\n" + title)
    if table_df.empty:
        print("No valid converged coupled rows available.")
        return table_df

    print(table_df[TABLE_COLS].to_string(index=False, float_format="%.6e"))
    return table_df


# ------------------------------------------------------------
# Coupled-only convergence plotting: revised Fig. 3
# ------------------------------------------------------------

def plot_paper2_coupled_convergence_iso_ortho(
    df_coupled,
    *,
    ycol="L2_RMS_Error_w_m",
    ylabel=r"$L^2$ RMS error in $w_h$ [m]",
    x_mode="h",
    filename_base=None,
    rate_fit_slice=slice(-2, None),
):
    """
    Revised Paper 2 Fig. 3:
    coupled-only displacement convergence for isotropic/orthotropic cases.

    Only valid converged coupled rows are plotted.

    Note:
    For x_mode='h', the x-axis is intentionally shown from coarse to fine,
    i.e. larger h on the left and smaller h on the right.

    Rate triangles:
    - rates are computed from the log-log slope selected by rate_fit_slice;
      the default slice(-2, None) uses the last two finest mesh points.
      Use slice(None) only if a global fitted rate over all refinement data is desired.
    - all triangles are kept near the finest segment;
    - triangles are compact to avoid overlap;
    - for each polynomial degree p, the higher-error curve gets the upper triangle
      and the lower-error curve gets the lower triangle;
    - the label "1" is always placed above the horizontal triangle edge.
    """

    if filename_base is None:
        filename_base = f"paper2_isotropic_orthotropic_simply_supported_coupled_w_convergence_mesh_size_{STUDY_TAG}"

    df = valid_converged_coupled_rows(df_coupled)
    if df.empty:
        raise ValueError("No valid converged coupled rows available for Fig. 3.")

    plt.style.use(["science", "ieee", "notebook", "grid"])
    plt.rcParams["figure.autolayout"] = False

    x_options = {
        "h": ("h", r"Characteristic mesh length $\ell_{\mathcal{T}}$ [m]"),
        "inv_h": ("inv_h", r"Inverse mesh size $1/h$"),
        "sqrt_plate_dofs": ("sqrt_plate_DoFs", r"$\sqrt{\mathrm{plate\ DoFs}}$"),
        "plate_dofs": ("plate_DoFs", "Plate degrees of freedom"),
        "sqrt_total_dofs": ("sqrt_Total_DoFs", r"$\sqrt{\mathrm{total\ DoFs}}$"),
        "total_dofs": ("Total_DoFs", "Total degrees of freedom"),
    }
    if x_mode not in x_options:
        raise ValueError(f"x_mode must be one of {list(x_options.keys())}")

    xcol, xlabel = x_options[x_mode]

    style_specs = [
        ("Isotropic", 2, "black", "-", "o", r"Isotropic, $p=2$"),
        ("Isotropic", 3, "blue", "--", "s", r"Isotropic, $p=3$"),
        ("Orthotropic", 2, "darkorange", "-", "o", r"Orthotropic, $p=2$"),
        ("Orthotropic", 3, "purple", "--", "s", r"Orthotropic, $p=3$"),
    ]
    styles = {
        (case_name, p): dict(color=color, linestyle=linestyle, marker=marker, label=label)
        for case_name, p, color, linestyle, marker, label in style_specs
    }

    def positive_finite(vals):
        vals = np.asarray(vals, dtype=float)
        return vals[np.isfinite(vals) & (vals > 0.0)]

    def ordered_xy(x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
        x, y = x[mask], y[mask]
        order = np.argsort(x)[::-1] if x_mode == "h" else np.argsort(x)
        return x[order], y[order]

    def estimate_loglog_slope(x, y, fit_slice=slice(None)):
        x, y = ordered_xy(x, y)
        if len(x) < 2:
            return None

        xx, yy = x[fit_slice], y[fit_slice]
        if len(xx) < 2:
            return None

        slope, _ = np.polyfit(np.log10(xx), np.log10(yy), 1)
        return float(abs(slope))

    def fine_error_value(grp):
        x, y = ordered_xy(grp[xcol].to_numpy(float), grp[ycol].to_numpy(float))
        return np.nan if len(y) == 0 else float(y[-1])

    def build_triangle_specs(grouped):
        """
        Automatic compact triangle placement.

        For each polynomial degree:
            higher-error curve -> triangle above,
            lower-error curve  -> triangle below.

        All triangles stay near the finest segment. The p=3 triangles are made
        slightly smaller than p=2 because the p=3 curves are closer together.
        """
        specs = {}

        for p in sorted({key[1] for key in grouped}):
            keys_p = [key for key in grouped if key[1] == p]
            if not keys_p:
                continue

            keys_p = sorted(keys_p, key=lambda key: fine_error_value(grouped[key]))

            if len(keys_p) == 1:
                specs[keys_p[0]] = dict(
                    location="above",
                    start_idx=-2,
                    span_frac=0.34,
                    tri_gap=1.22,
                    rate_side_pad=0.020,
                    one_label_gap=1.055,
                    rate_label_shift=1.00,
                )
                continue

            lower_key = keys_p[0]
            upper_key = keys_p[-1]

            if int(p) == 2:
                specs[upper_key] = dict(
                    location="above",
                    start_idx=-2,
                    span_frac=0.36,
                    tri_gap=1.24,
                    rate_side_pad=0.020,
                    one_label_gap=1.055,
                    rate_label_shift=1.00,
                )
                specs[lower_key] = dict(
                    location="below",
                    start_idx=-2,
                    span_frac=0.36,
                    tri_gap=1.28,
                    rate_side_pad=0.020,
                    one_label_gap=1.055,
                    rate_label_shift=1.00,
                )
            else:
                specs[upper_key] = dict(
                    location="above",
                    start_idx=-2,
                    span_frac=0.28,
                    tri_gap=1.20,
                    rate_side_pad=0.018,
                    one_label_gap=1.060,
                    rate_label_shift=1.00,
                )
                specs[lower_key] = dict(
                    location="below",
                    start_idx=-2,
                    span_frac=0.28,
                    tri_gap=1.26,
                    rate_side_pad=0.018,
                    one_label_gap=1.060,
                    rate_label_shift=1.00,
                )

        return specs

    def add_rate_triangle(
        ax,
        x,
        y,
        rate,
        color,
        *,
        start_idx=-2,
        span_frac=0.34,
        tri_gap=1.22,
        rate_side_pad=0.020,
        one_label_gap=1.055,
        rate_label_shift=1.00,
        lw=1.00,
        fontsize=8,
        location="above",
    ):
        if rate is None or rate <= 0.0:
            return

        x, y = ordered_xy(x, y)
        if len(x) < 2:
            return

        start_idx = len(x) + start_idx if start_idx < 0 else start_idx
        if start_idx < 0 or start_idx + 1 >= len(x):
            return

        x_prev = float(x[start_idx])
        x_last = float(x[start_idx + 1])
        y_last = float(y[start_idx + 1])

        if x_prev <= 0.0 or x_last <= 0.0 or y_last <= 0.0:
            return

        if x_mode == "h":
            full_ratio = x_prev / x_last
            if full_ratio <= 1.0:
                return

            x_right = x_last
            x_left = x_last * (full_ratio ** span_frac)
            local_ratio = x_left / x_right
            x_rate = x_right / (local_ratio ** rate_side_pad)

        else:
            full_ratio = x_last / x_prev
            if full_ratio <= 1.0:
                return

            x_left = x_last / (full_ratio ** span_frac)
            x_right = x_last
            local_ratio = x_right / x_left
            x_rate = x_right * (local_ratio ** rate_side_pad)

        if location == "below":
            # Triangle is below the curve.
            # y_top is still the horizontal edge of the triangle.
            y_top = y_last / tri_gap
            y_bottom = y_top / (local_ratio ** rate)
        else:
            # Triangle is above the curve.
            # y_top is the horizontal edge of the triangle.
            y_bottom = y_last * tri_gap
            y_top = y_bottom * (local_ratio ** rate)

        if min(x_left, x_right, y_top, y_bottom) <= 0.0:
            return

        # Important fix:
        # Put the "1" label above the horizontal edge for both upper and lower triangles.
        one_label_y = y_top * one_label_gap
        one_label_va = "bottom"

        y_mid = np.sqrt(y_top * y_bottom) * float(rate_label_shift)

        ymin, ymax = ax.get_ylim()
        log_ymin, log_ymax = np.log10(ymin), np.log10(ymax)
        low = np.log10(min(y_top, y_bottom))
        high = np.log10(max(y_top, y_bottom, one_label_y))

        if low < log_ymin + 0.08:
            scale = 10 ** (log_ymin + 0.08) / min(y_top, y_bottom)
            y_top *= scale
            y_bottom *= scale
            one_label_y *= scale
            y_mid *= scale

        if high > log_ymax - 0.08:
            scale = 10 ** (log_ymax - 0.08) / max(y_top, y_bottom, one_label_y)
            y_top *= scale
            y_bottom *= scale
            one_label_y *= scale
            y_mid *= scale

        text_box = dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.12)

        ax.plot([x_left, x_right], [y_top, y_top], color=color, lw=lw, zorder=5)
        ax.plot([x_right, x_right], [y_top, y_bottom], color=color, lw=lw, zorder=5)
        ax.plot([x_left, x_right], [y_top, y_bottom], color=color, lw=lw, zorder=5)

        ax.text(
            np.sqrt(x_left * x_right),
            one_label_y,
            "1",
            ha="center",
            va=one_label_va,
            fontsize=fontsize,
            color=color,
            bbox=text_box,
            clip_on=True,
            zorder=6,
        )

        ax.text(
            x_rate,
            y_mid,
            f"{rate:.1f}",
            ha="left",
            va="center",
            fontsize=fontsize,
            color=color,
            bbox=text_box,
            clip_on=True,
            zorder=6,
        )

    all_x = positive_finite(df[xcol].to_numpy(float))
    all_y = positive_finite(df[ycol].to_numpy(float))
    if len(all_x) == 0 or len(all_y) == 0:
        raise ValueError("No positive finite data available for Fig. 3.")

    fig, ax = plt.subplots(figsize=(6.8, 5.2), facecolor="white")
    grouped = {}

    for key, style in styles.items():
        case_name, p = key
        grp = df[
            (df["case"].astype(str) == case_name)
            & (df["p"].astype(int) == int(p))
        ].dropna(subset=[xcol, ycol]).copy()

        if grp.empty:
            continue

        grp = grp.sort_values(xcol, ascending=(x_mode != "h"))
        grouped[key] = grp

        ax.loglog(
            grp[xcol].to_numpy(float),
            grp[ycol].to_numpy(float),
            linestyle=style["linestyle"],
            marker=style["marker"],
            color=style["color"],
            markeredgecolor=style["color"],
            markerfacecolor="none",
            linewidth=2.2,
            markersize=8.0,
            markeredgewidth=1.4,
            label=style["label"],
            clip_on=False,
        )

    ax.set_facecolor("white")
    ax.set_title(r"Partitioned coupled refinement study", fontsize=14, pad=8)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=13, length=4, pad=2)
    ax.tick_params(axis="both", which="minor", length=2)
    ax.grid(True, which="both", linestyle=":", linewidth=0.45, alpha=0.35)

    if x_mode == "h":
        ticks = []
        for size in [8, 16, 32, 64]:
            vals = positive_finite(df.loc[df["size"].astype(int) == size, "h"].to_numpy(float))
            if len(vals):
                ticks.append(float(np.median(vals)))

        if ticks:
            ticks = sorted(set(np.round(ticks, 12)), reverse=True)
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, pos: f"{x:.2f}" if x >= 0.1 else f"{x:.3f}")
            )
    else:
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=80))
    ax.xaxis.set_minor_formatter(NullFormatter())

    # Important:
    # For h-refinement, show coarse-to-fine from left to right:
    # left  = largest h, approximately 0.25
    # right = smallest h, approximately 0.031
    if x_mode == "h":
        ax.set_xlim(np.max(all_x) * 1.20, np.min(all_x) / 1.14)
    else:
        ax.set_xlim(np.min(all_x) / 1.14, np.max(all_x) * 1.20)

    ax.set_ylim(np.min(all_y) / 2.60, np.max(all_y) * 1.65)

    triangle_specs = build_triangle_specs(grouped)

    for key, grp in grouped.items():
        style = styles[key]
        spec = triangle_specs.get(
            key,
            dict(
                location="above",
                start_idx=-2,
                span_frac=0.34,
                tri_gap=1.22,
                rate_side_pad=0.020,
                one_label_gap=1.055,
                rate_label_shift=1.00,
            ),
        )

        rate = estimate_loglog_slope(
            grp[xcol].to_numpy(float),
            grp[ycol].to_numpy(float),
            fit_slice=rate_fit_slice,
        )

        add_rate_triangle(
            ax,
            grp[xcol].to_numpy(float),
            grp[ycol].to_numpy(float),
            rate,
            style["color"],
            location=spec["location"],
            start_idx=spec["start_idx"],
            span_frac=spec["span_frac"],
            tri_gap=spec["tri_gap"],
            rate_side_pad=spec["rate_side_pad"],
            one_label_gap=spec["one_label_gap"],
            rate_label_shift=spec["rate_label_shift"],
            lw=1.00,
            fontsize=8,
        )

    leg = ax.legend(
        loc="lower left",
        fontsize=12,
        frameon=True,
        borderpad=0.55,
        handlelength=2.2,
        handletextpad=0.55,
        labelspacing=0.35,
    )
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.8)

    os.makedirs(PAPER2_OUTDIR, exist_ok=True)
    for ext, dpi in (("pdf", 300), ("png", 600)):
        fig.savefig(
            os.path.join(PAPER2_OUTDIR, f"{filename_base}.{ext}"),
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
        )

    plt.show()

# ------------------------------------------------------------
# Coupled update-history plotting: revised Fig. 4
# ------------------------------------------------------------

def history_dataframe_from_ref(ref_or_solution, case_label):
    """Convert a coupled reference/solution history into a DataFrame for Fig. 4."""
    hist = pd.DataFrame(ref_or_solution.get("history", None)).copy()
    if hist.empty:
        return hist

    hist["case"] = str(case_label)
    return hist


def load_saved_coupled_reference_history(case_label, *, bc_type=MECHANICAL_BC_TYPE):
    """
    Load coupled-reference fixed-point history from saved reference-cache JSON.
    No FOM solve is executed.
    """

    pattern = os.path.join(
        PAPER2_OUTDIR,
        (
            f"paper2_ref_coupled_{_safe_filename_part(case_label)}"
            f"_{_safe_filename_part(bc_type)}"
            f"_size{int(REF_SIZE)}_p{int(REF_P)}_*.json"
        ),
    )
    candidates = sorted(glob.glob(pattern))

    if not candidates:
        raise FileNotFoundError(f"No coupled reference-cache JSON found for {case_label}: {pattern}")

    for path in candidates:
        with open(path, "r") as f:
            meta = json.load(f)

        payload = meta.get("payload", {})
        is_match = (
            payload.get("mode") == "coupled"
            and payload.get("case") == case_label
            and payload.get("bc_type") == bc_type
            and int(payload.get("ref_size", -1)) == int(REF_SIZE)
            and int(payload.get("ref_p", -1)) == int(REF_P)
            and payload.get("ref_heat", {}) == _json_safe(REF_HEAT_CONTROLS)
            and payload.get("stopping_norm") == "L2_Omega"
            and payload.get("reported_refinement_branch") == "coupled_only"
        )
        hist = meta.get("history", None)

        if is_match and hist:
            df = pd.DataFrame(hist).copy()
            df["case"] = case_label
            df["source_file"] = os.path.basename(path)
            return df

    raise FileNotFoundError(f"No valid coupled history found for {case_label} in {len(candidates)} JSON file(s).")


def plot_paper2_coupled_history_iso_ortho_combined(
    histories,
    *,
    filename_base=None,
):
    """
    Revised Paper 2 Fig. 4:
    fixed-point relative-update histories only.

    The old response-evolution panel is intentionally removed.
    """
    if filename_base is None:
        filename_base = f"paper2_isotropic_orthotropic_simply_supported_coupled_reference_update_history_{STUDY_TAG}"

    if not histories:
        print("No coupled histories available.")
        return

    hist = pd.concat(histories, ignore_index=True)
    if hist.empty:
        print("No coupled histories available.")
        return

    plt.style.use(["science", "ieee", "notebook", "grid"])
    plt.rcParams["figure.autolayout"] = False

    hist_styles = {
        ("Isotropic", "err_w"): dict(color="black", linestyle="-", marker="o", label=r"Isotropic, $\varepsilon_w$"),
        ("Isotropic", "err_T1"): dict(color="red", linestyle="--", marker="s", label=r"Isotropic, $\varepsilon_{\theta}$"),
        ("Orthotropic", "err_w"): dict(color="blue", linestyle="-", marker="o", label=r"Orthotropic, $\varepsilon_w$"),
        ("Orthotropic", "err_T1"): dict(color="purple", linestyle="--", marker="s", label=r"Orthotropic, $\varepsilon_{\theta}$"),
    }

    fig, ax = plt.subplots(figsize=(6.8, 5.2), facecolor="white")

    for case_name in ["Isotropic", "Orthotropic"]:
        grp = hist[hist["case"].astype(str) == case_name].copy()
        if grp.empty:
            continue

        grp = grp.sort_values("iter")
        it = grp["iter"].to_numpy(int)

        for col in ["err_w", "err_T1"]:
            vals = grp[col].to_numpy(float)
            if col == "err_T1" and not np.isfinite(vals[1:]).any():
                continue

            style = hist_styles[(case_name, col)]
            ax.semilogy(
                it,
                vals,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2.2,
                markersize=7.0,
                markerfacecolor=style["color"],
                label=style["label"],
            )

    ax.set_facecolor("white")
    ax.set_title("Coupled fixed-point update history", fontsize=14, pad=8)
    ax.set_xlabel("Coupling iteration", fontsize=14)
    ax.set_ylabel(r"Relative $L^2(\Omega)$ update", fontsize=14)
    ax.tick_params(axis="both", labelsize=13, length=3, pad=2)
    ax.grid(True, which="both", linestyle=":", linewidth=0.45, alpha=0.35)

    leg = ax.legend(
        loc="upper right",
        fontsize=11.5,
        frameon=True,
        borderpad=0.55,
        handlelength=2.1,
        handletextpad=0.55,
        labelspacing=0.35,
    )
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.8)

    os.makedirs(PAPER2_OUTDIR, exist_ok=True)
    for ext, dpi in (("pdf", 300), ("png", 600)):
        fig.savefig(
            os.path.join(PAPER2_OUTDIR, f"{filename_base}.{ext}"),
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
        )

    plt.show()
    
# ------------------------------------------------------------
# Optional reference-level comparison
# ------------------------------------------------------------

def compare_two_reference_levels(
    mode,
    mu_mech,
    *,
    size_a=REFERENCE_COMPARISON_COARSE_SIZE,
    size_b=REF_SIZE,
    p=REF_P,
    label="reference_check",
    bc_type=MECHANICAL_BC_TYPE,
    fixed_heat=True,
    cache_label_a=None,
    cache_label_b=None,
):
    """
    Optional cached comparison between two coupled refined reference levels.

    This is not the main manuscript convergence table. It is a diagnostic to
    check the separation between the finest tested mesh and the reference.
    """
    if mode != "coupled":
        raise ValueError("Reference-level comparison for the revised study must be coupled-only.")

    cache_label_a = cache_label_a or f"{label}_size{int(size_a)}"
    cache_label_b = cache_label_b or f"{label}_size{int(size_b)}"

    sol_a = get_or_build_reference(
        mode=mode,
        label=cache_label_a,
        bc_type=bc_type,
        ref_size=size_a,
        ref_p=p,
        mu_mech=mu_mech,
        fixed_heat=fixed_heat,
    )
    sol_b = get_or_build_reference(
        mode=mode,
        label=cache_label_b,
        bc_type=bc_type,
        ref_size=size_b,
        ref_p=p,
        mu_mech=mu_mech,
        fixed_heat=fixed_heat,
    )

    _assert_converged_coupled_record(sol_a, context=f"Reference level A size={size_a}")
    _assert_converged_coupled_record(sol_b, context=f"Reference level B size={size_b}")

    mesh_b = reference_mesh(sol_b)
    Vw_b = FunctionSpace(mesh_b, "CG", int(p))
    Vtheta_b = FunctionSpace(mesh_b, "CG", int(p))
    sqrt_area_b = float(np.sqrt(assemble(Constant(1.0) * Measure("dx", domain=mesh_b))))
    deg = max(5, int(p) + 2)

    comparison = dict(
        mode=mode,
        label=label,
        bc_type=bc_type,
        size_a=int(size_a),
        size_b=int(size_b),
        p=int(p),
        L2_RMS_Error_w_m=l2_rms_error_against_ref(sol_a["w"], sol_b["w"], Vw_b, sqrt_area_b, degree=deg),
        L2_RMS_Error_theta=l2_rms_error_against_ref(sol_a["theta"], sol_b["theta"], Vtheta_b, sqrt_area_b, degree=deg),
        Rel_L2_Error_w=l2_relative_error_against_ref(sol_a["w"], sol_b["w"], Vw_b, degree=deg),
        Rel_L2_Error_theta=l2_relative_error_against_ref(sol_a["theta"], sol_b["theta"], Vtheta_b, degree=deg),
    )

    out_json = os.path.join(
        PAPER2_OUTDIR,
        f"paper2_{mode}_{_safe_filename_part(label)}_{bc_type}_{STUDY_TAG}_reference_comparison.json",
    )

    with open(out_json, "w") as f:
        json.dump(_json_safe(comparison), f, indent=2, sort_keys=True)

    print("\nREFERENCE-LEVEL COMPARISON")
    for key, value in comparison.items():
        print(f"{key}: {value}")
    print(f"Saved reference comparison: {out_json}")

    return sol_a, sol_b, comparison


# ------------------------------------------------------------
# Driver utilities: coupled-only
# ------------------------------------------------------------

def run_smoke_test(case_label="Isotropic", *, bc_type=MECHANICAL_BC_TYPE):
    """Coupled-only smoke test. This avoids accidentally reintroducing one-way refinement rows."""
    mu_mech = PLATE_CASES[case_label]["mu_mech"]

    df_cpl, ref_cpl = gather_paper2_convergence_data(
        mode="coupled",
        mu_mech=mu_mech,
        resolutions=RESOLUTIONS_SMOKE,
        p_values=[2],
        ref_size=32,
        ref_p=3,
        label=f"{case_label}_smoke",
        bc_type=bc_type,
        fixed_heat=True,
    )

    print(f"\nCOUPLED SMOKE {case_label.upper()} ({bc_type})")
    print(valid_converged_coupled_rows(df_cpl)[TABLE_COLS].to_string(index=False, float_format="%.6e"))

    return df_cpl, ref_cpl


def run_case(case_label, *, bc_type=MECHANICAL_BC_TYPE):
    """Run the revised Paper 2 refinement study for one material case. Coupled-only."""
    mu_mech = PLATE_CASES[case_label]["mu_mech"]

    df_cpl, ref_cpl = gather_paper2_convergence_data(
        mode="coupled",
        mu_mech=mu_mech,
        resolutions=RESOLUTIONS_COUPLED_FINAL,
        p_values=P_VALUES,
        ref_size=REF_SIZE,
        ref_p=REF_P,
        label=case_label,
        bc_type=bc_type,
        fixed_heat=True,
    )

    print(f"\nCOUPLED {case_label.upper()} ({bc_type})")
    df_valid = valid_converged_coupled_rows(df_cpl)

    if df_valid.empty:
        print("No valid converged coupled rows available.")
    else:
        print(df_valid[TABLE_COLS].to_string(index=False, float_format="%.6e"))

    return df_cpl, ref_cpl


def replot_coupled_from_saved_csv(*, bc_type=MECHANICAL_BC_TYPE):
    """Replot revised Fig. 3 from saved coupled CSVs only."""
    frames = []
    for case_label in ["Isotropic", "Orthotropic"]:
        path = convergence_csv_path("coupled", case_label, bc_type)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing coupled convergence CSV: {path}")
        frames.append(pd.read_csv(path))

    df_all = pd.concat(frames, ignore_index=True)
    plot_paper2_coupled_convergence_iso_ortho(
        df_all,
        ycol="L2_RMS_Error_w_m",
        ylabel=r"$L^2$ RMS error in $w_h$ [m]",
        x_mode="h",
        filename_base=f"paper2_isotropic_orthotropic_simply_supported_coupled_w_convergence_mesh_size_{STUDY_TAG}",
        rate_fit_slice=slice(-2, None),
    )

    return df_all


def replot_coupled_history_from_saved_json(*, bc_type=MECHANICAL_BC_TYPE):
    """Replot revised Fig. 4 from saved coupled reference-cache JSON histories only."""
    histories = [
        load_saved_coupled_reference_history("Isotropic", bc_type=bc_type),
        load_saved_coupled_reference_history("Orthotropic", bc_type=bc_type),
    ]
    plot_paper2_coupled_history_iso_ortho_combined(
        histories,
        filename_base=f"paper2_isotropic_orthotropic_simply_supported_coupled_reference_update_history_{STUDY_TAG}",
    )

    return histories


# ------------------------------------------------------------
# Execute revised coupled-only study
# ------------------------------------------------------------

# Optional quick test:
# df_cpl_smoke, ref_cpl_smoke = run_smoke_test(
#     "Isotropic",
#     bc_type=MECHANICAL_BC_TYPE,
# )

df_cpl_iso, ref_cpl_iso = run_case("Isotropic", bc_type=MECHANICAL_BC_TYPE)
df_cpl_ortho, ref_cpl_ortho = run_case("Orthotropic", bc_type=MECHANICAL_BC_TYPE)

df_cpl_all = pd.concat([df_cpl_iso, df_cpl_ortho], ignore_index=True)
df_cpl_valid = valid_converged_coupled_rows(df_cpl_all)

print_finest_converged_table(
    df_cpl_valid,
    title="FINEST-MESH CONVERGED COUPLED ROWS FOR MANUSCRIPT TABLE",
)

plot_paper2_coupled_convergence_iso_ortho(
    df_cpl_valid,
    ycol="L2_RMS_Error_w_m",
    ylabel=r"$L^2$ RMS error in $w_h$ [m]",
    x_mode="h",
    filename_base=f"paper2_isotropic_orthotropic_simply_supported_coupled_w_convergence_mesh_size_{STUDY_TAG}",
    rate_fit_slice=slice(-2, None),
)

histories_for_fig4 = [
    history_dataframe_from_ref(ref_cpl_iso, "Isotropic"),
    history_dataframe_from_ref(ref_cpl_ortho, "Orthotropic"),
]

plot_paper2_coupled_history_iso_ortho_combined(
    histories_for_fig4,
    filename_base=f"paper2_isotropic_orthotropic_simply_supported_coupled_reference_update_history_{STUDY_TAG}",
)

if RUN_REFERENCE_COMPARISON:
    ref_coarse_cpl, ref_reference_cpl, ref_compare = compare_two_reference_levels(
        "coupled",
        PLATE_CASES["Isotropic"]["mu_mech"],
        size_a=REFERENCE_COMPARISON_COARSE_SIZE,
        size_b=REF_SIZE,
        p=REF_P,
        label=f"Isotropic_coupled_{REFERENCE_COMPARISON_COARSE_SIZE}_vs_{REF_SIZE}_{MECHANICAL_BC_TYPE}",
        bc_type=MECHANICAL_BC_TYPE,
        fixed_heat=True,
        cache_label_a=f"Isotropic_coupled_refcheck_{REFERENCE_COMPARISON_COARSE_SIZE}_{MECHANICAL_BC_TYPE}",
        cache_label_b="Isotropic",
    )

# %% Cell 31 | id: c0375b9d-d1ac-498b-9fc0-1535009eeb04


# %% Cell 32 | id: 53c933d4-baea-4564-9b8a-2937179d0bae


# %% Cell 33 | id: 7a85bfa5-dc51-4c1e-a6d6-1ae8783b9584


# %% Cell 34 | id: 2511ebde-4230-467e-beec-0af77393ea42
# ============================================================
# Paper 2 thermomechanical FOM convergence with refined FOM reference
# ============================================================

PAPER2_OUTDIR = "paper2_fom_convergence"
os.makedirs(PAPER2_OUTDIR, exist_ok=True)

MECHANICAL_BC_TYPE = "simply_supported"
RESOLUTIONS_SMOKE = [8, 11, 16]
RESOLUTIONS_FINAL = [8, 11, 16, 23, 32, 45, 64]
RESOLUTIONS_COUPLED_FINAL = [8, 11, 16, 23, 32, 45, 64]
P_VALUES = [2, 3]

# ------------------------------------------------------------
# User-decision block
# ------------------------------------------------------------
RESUME_RUNS = True
FORCE_RERUN = False
RUN_REFERENCE_COMPARISON = True

# Current working reference setting.
REF_SIZE, REF_P = 100, 4

# Current working fixed heat discretization.
HEAT_NX, HEAT_NY, HEAT_NZ = 64, 32, 16
HEAT_DEGREE = 2
NZ_QUAD_T1 = 20

# General coarse/reference comparison size.
REFERENCE_COMPARISON_COARSE_SIZE = max(RESOLUTIONS_COUPLED_FINAL)

REFERENCE_CACHE_VERSION = 3
CONVERGENCE_TABLE_VERSION = 3

FIXED_HEAT_CONTROLS = dict(
    heat_nx=int(HEAT_NX),
    heat_ny=int(HEAT_NY),
    heat_nz=int(HEAT_NZ),
    heat_degree=int(HEAT_DEGREE),
    Nz_quad_T1=int(NZ_QUAD_T1),
)
REF_HEAT_CONTROLS = dict(FIXED_HEAT_CONTROLS)

# Settings-dependent label used in saved CSV/figure/reference-comparison names.
STUDY_TAG = (
    f"refN{int(REF_SIZE)}_p{int(REF_P)}"
    f"_heat{int(HEAT_NX)}x{int(HEAT_NY)}x{int(HEAT_NZ)}"
    f"_pT{int(HEAT_DEGREE)}"
    f"_qT1{int(NZ_QUAD_T1)}"
    f"_v{int(CONVERGENCE_TABLE_VERSION)}"
)

PLATE_CASES = {
    "Isotropic": dict(mu_mech=[1.0e4, 1.0e4, 0.3e4, 0.35e4, 1.0e3, -10000.0]),
    "Orthotropic": dict(mu_mech=[2.0e4, 1.0e4, 0.25e4, 0.30e4, 1.0e3, -10000.0]),
}

THERMAL_PARAMS = dict(
    T_amb=27.0 + 273.15,
    T_sub=24.0 + 273.15,
    h_con=10.0,
    eps_r=0.90,
    q_s=700.0,
    h_c_cont=200.0,
    h_c_gap=5.0,
    eta_c=50.0,
    w_contact=-0.1,
    kx=0.35,
    ky=0.35,
    kz=0.35,
    alpha1=1.3e-4,
    alpha2=1.3e-4,
    rho=1050.0,
)

COUPLING_PARAMS = dict(
    coupling_omega=0.7,
    coupling_tol_w=1e-6,
    coupling_tol_T1=1e-6,
    coupling_max_iters=40,
)

TABLE_COLS = """
mode case bc_type p size h inv_h plate_DoFs heat_DoFs theta_DoFs Total_DoFs
sqrt_plate_DoFs sqrt_Total_DoFs L2_RMS_Error_w_m Rel_L2_Error_w
L2_RMS_Error_theta Rel_L2_Error_theta coupled_iters coupled_converged
stopped_by_max_iters termination_reason err_w err_theta
""".split()

NEWTON_SOLVER_PARAMETERS["newton_solver"].update(
    linear_solver="petsc",
    absolute_tolerance=1e-4,
    relative_tolerance=1e-5,
    maximum_iterations=80,
    relaxation_parameter=1.0,
)


# ------------------------------------------------------------
# Mesh/thermal controls
# ------------------------------------------------------------

def heat_controls_from_size(size, *, ref=False, fixed=True):
    """
    Heat controls for convergence studies.

    fixed=True keeps the 3D heat discretization fixed so that the reported
    refinement behavior reflects the plate/FOM discretization rather than
    simultaneous heat-mesh refinement.
    """
    if ref:
        return dict(REF_HEAT_CONTROLS)

    if fixed:
        return dict(FIXED_HEAT_CONTROLS)

    return dict(
        heat_nx=int(size),
        heat_ny=max(2, int(round(size / 2))),
        heat_nz=max(8, int(round(size / 4))),
        heat_degree=int(HEAT_DEGREE),
        Nz_quad_T1=max(10, min(24, int(round(size / 4)) + 4)),
    )


# ------------------------------------------------------------
# Solver construction and single-case solve
# ------------------------------------------------------------

def make_solver(size, p, mu_mech, *, load_type="uniform", bc_type=MECHANICAL_BC_TYPE):
    solver = GeneralMultiphysicsSolver(study_case=1)
    solver.size = int(size)
    solver.degree = int(p)
    solver.load_type = load_type
    solver.bc_type = bc_type
    solver.define_domain(n_vert=0, n_horiz=0, plot_subdomains=False)
    solver.set_mu(mu_mech)
    solver.set_rom_thermal_parameters(**THERMAL_PARAMS)
    return solver


def copy_function(f):
    out = Function(f.function_space())
    out.assign(f)
    return out


def solve_paper2_case(
    *,
    mode,
    size,
    p,
    mu_mech,
    heat_nx,
    heat_ny,
    heat_nz,
    heat_degree,
    Nz_quad_T1,
    T1_cg_degree,
    bc_type=MECHANICAL_BC_TYPE,
):
    """
    Solve one Paper 2 FOM case.

    The one-way solver path is kept available because the framework still supports
    one-way thermomechanics. However, the revised refinement-study drivers below
    intentionally use mode='coupled' only.
    """
    solver = make_solver(size=size, p=p, mu_mech=mu_mech, bc_type=bc_type)

    heat_kwargs = dict(
        thermal_on=True,
        heat_nx=int(heat_nx),
        heat_ny=int(heat_ny),
        heat_nz=int(heat_nz),
        heat_degree=int(heat_degree),
        Nz_quad_T1=int(Nz_quad_T1),
        T1_cg_degree=int(T1_cg_degree),
        return_mode="global",
    )
    common = dict(solver=solver, mode=mode, bc_type=bc_type)

    if mode == "one_way":
        heat = solver.solve_rom_sample(mu_mech, coupled_on=False, **heat_kwargs)
        w = copy_function(solver.w_contact_global)
        theta = copy_function(solver.T1_from_heat)
        return dict(
            common,
            w=w,
            theta=theta,
            heat=heat,
            history=None,
            converged=True,
            stopped_by_max_iters=False,
            termination_reason="one_way",
            err_w=np.nan,
            err_T1=np.nan,
            iters=1,
            w_dofs=int(w.function_space().dim()),
            heat_dofs=int(heat["Vt"].dim()),
            theta_dofs=int(theta.function_space().dim()),
            total_dofs=int(w.function_space().dim() + heat["Vt"].dim() + theta.function_space().dim()),
            heat_nx=int(heat_nx),
            heat_ny=int(heat_ny),
            heat_nz=int(heat_nz),
            heat_degree=int(heat_degree),
            Nz_quad_T1=int(Nz_quad_T1),
            T1_cg_degree=int(T1_cg_degree),
        )

    if mode == "coupled":
        out = solver.solve_rom_sample(
            mu_mech,
            coupled_on=True,
            coupling_omega=COUPLING_PARAMS["coupling_omega"],
            coupling_tol_w=COUPLING_PARAMS["coupling_tol_w"],
            coupling_tol_T1=COUPLING_PARAMS["coupling_tol_T1"],
            coupling_max_iters=COUPLING_PARAMS["coupling_max_iters"],
            coupling_verbose=False,
            w0=0.0,
            return_coupled_dict=True,
            **heat_kwargs,
        )
        return dict(
            common,
            w=copy_function(out["w"]),
            theta=copy_function(out["T1"]),
            heat=out["heat"],
            history=out["history"],
            converged=bool(out["converged"]),
            stopped_by_max_iters=bool(out["stopped_by_max_iters"]),
            termination_reason=str(out["termination_reason"]),
            err_w=float(out["err_w"]),
            err_T1=float(out["err_T1"]),
            iters=int(out["iters"]),
            w_dofs=int(out["w_dofs"]),
            heat_dofs=int(out["heat_dofs"]),
            theta_dofs=int(out["T1_dofs"]),
            total_dofs=int(out["total_coupled_dofs"]),
            heat_nx=int(heat_nx),
            heat_ny=int(heat_ny),
            heat_nz=int(heat_nz),
            heat_degree=int(heat_degree),
            Nz_quad_T1=int(Nz_quad_T1),
            T1_cg_degree=int(T1_cg_degree),
        )

    raise ValueError("mode must be 'one_way' or 'coupled'.")


# ------------------------------------------------------------
# Error computation against refined reference
# ------------------------------------------------------------

def lift_field_to_reference_mesh(field, Vref, degree=5):
    """Interpolate a scalar plate field from its own mesh to the reference mesh."""
    try:
        field.set_allow_extrapolation(True)
    except Exception:
        pass

    class FieldOnReferenceMesh(UserExpression):
        def __init__(self, f, **kwargs):
            super().__init__(**kwargs)
            self.f = f

        def eval(self, values, x):
            values[0] = float(self.f(Point(float(x[0]), float(x[1]))))

        def value_shape(self):
            return ()

    return interpolate(FieldOnReferenceMesh(field, degree=degree), Vref)


def l2_rms_error_against_ref(field, ref_field, Vref, sqrt_area, degree=5):
    """
    RMS L2 error against the refined numerical FOM reference.

    This is not an exact-solution error.
    """
    field_refmesh = lift_field_to_reference_mesh(field, Vref, degree=degree)
    ref_projected = project(ref_field, Vref)
    return float(errornorm(ref_projected, field_refmesh, norm_type="L2", degree_rise=3) / sqrt_area)


def l2_relative_error_against_ref(field, ref_field, Vref, degree=5, eps=1e-14):
    """Relative L2 error against the refined numerical FOM reference."""
    dx_ref = Measure("dx", domain=Vref.mesh())
    field_refmesh = lift_field_to_reference_mesh(field, Vref, degree=degree)
    ref_projected = project(ref_field, Vref)

    abs_err = float(errornorm(ref_projected, field_refmesh, norm_type="L2", degree_rise=3))
    ref_norm = float(np.sqrt(max(float(assemble(ref_projected * ref_projected * dx_ref)), 0.0)))
    return abs_err / max(ref_norm, float(eps))


# ------------------------------------------------------------
# Checkpoint helpers
# ------------------------------------------------------------

def convergence_csv_path(mode, label, bc_type):
    return os.path.join(
        PAPER2_OUTDIR,
        f"paper2_{mode}_{label}_{bc_type}_convergence_{STUDY_TAG}.csv",
    )


def _as_bool(value, default=False):
    """Robust boolean conversion for bools, numpy bools, and CSV string values."""
    if value is None:
        return bool(default)

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, np.integer)):
        return bool(value)

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return bool(default)
        return bool(int(value))

    if isinstance(value, str):
        value_l = value.strip().lower()
        if value_l in ("true", "t", "1", "yes", "y"):
            return True
        if value_l in ("false", "f", "0", "no", "n", "nan", "none", ""):
            return False

    return bool(default)


def _bool_series(series, default=False):
    """Robust boolean mask for pandas Series that may contain strings from CSV."""
    return series.apply(lambda v: _as_bool(v, default=default)).astype(bool)


def _coupled_solution_is_converged(record):
    """True only if a coupled record/solution reached the coupled stopping criterion."""
    return (
        str(record.get("mode", "coupled")) == "coupled"
        and _as_bool(record.get("converged", record.get("coupled_converged", False)))
        and not _as_bool(record.get("stopped_by_max_iters", False))
    )


def _row_is_valid_converged_coupled(row):
    """True only for coupled rows that are valid for manuscript tables/plots."""
    try:
        if str(row.get("mode", "")) != "coupled":
            return False
        if not _as_bool(row.get("coupled_converged", False)):
            return False
        if _as_bool(row.get("stopped_by_max_iters", False)):
            return False
        if "reported" in row.index and not _as_bool(row.get("reported", True), default=True):
            return False
        if "stopping_norm" in row.index and str(row.get("stopping_norm", "")) != "L2_Omega":
            return False

        finite_cols = [
            "h",
            "L2_RMS_Error_w_m",
            "Rel_L2_Error_w",
            "L2_RMS_Error_theta",
            "Rel_L2_Error_theta",
            "err_w",
            "err_theta",
        ]
        return all(col not in row or np.isfinite(float(row[col])) for col in finite_cols)

    except Exception:
        return False


def load_existing_convergence_rows(out_csv):
    """
    Load old checkpoint rows.

    Only valid converged coupled rows are treated as completed. Failed or
    non-converged rows are kept in the CSV for traceability, but they are not
    allowed to block reruns.
    """
    if not (RESUME_RUNS and os.path.exists(out_csv) and not FORCE_RERUN):
        return None, set()

    df = pd.read_csv(out_csv)
    if df.empty:
        return None, set()

    required_key_cols = ["mode", "case", "bc_type", "p", "size"]
    valid_keys = {
        (str(row["mode"]), str(row["case"]), str(row["bc_type"]), int(row["p"]), int(row["size"]))
        for _, row in df.iterrows()
        if all(col in row.index for col in required_key_cols) and _row_is_valid_converged_coupled(row)
    }

    print(f"\nResuming from existing table: {out_csv}")
    print(f"Existing rows: {len(df)}")
    print(f"Valid converged coupled rows that will be skipped: {len(valid_keys)}")

    return df, valid_keys


def save_convergence_checkpoint(new_records, existing_df, out_csv):
    if not new_records:
        return existing_df if existing_df is not None else pd.DataFrame()

    new_df = pd.DataFrame.from_records(new_records)
    save_df = pd.concat([existing_df, new_df], ignore_index=True) if existing_df is not None and not existing_df.empty else new_df

    save_df = (
        save_df.drop_duplicates(["mode", "case", "bc_type", "p", "size"], keep="last")
        .sort_values(["mode", "case", "bc_type", "p", "h"], ascending=[True, True, True, True, False])
        .reset_index(drop=True)
    )
    save_df.to_csv(out_csv, index=False)
    print(f"Checkpoint saved: {out_csv}")

    return save_df


# ------------------------------------------------------------
# Reference-cache helpers
# ------------------------------------------------------------

def _json_safe(obj):
    """Convert numpy/scalar objects into JSON-safe Python objects."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _safe_filename_part(text):
    text = str(text)
    for ch in [" ", "/", "\\", ":", ";", ",", "(", ")", "[", "]", "{", "}"]:
        text = text.replace(ch, "_")
    return text


def reference_cache_key(*, mode, label, bc_type, ref_size, ref_p, mu_mech, ref_heat, fixed_heat):
    """Build a unique reference-cache key."""
    payload = dict(
        cache_version=int(REFERENCE_CACHE_VERSION),
        convergence_table_version=int(CONVERGENCE_TABLE_VERSION),
        study_tag=str(STUDY_TAG),
        mode=str(mode),
        case=str(label),
        bc_type=str(bc_type),
        ref_size=int(ref_size),
        ref_p=int(ref_p),
        mu_mech=[float(v) for v in mu_mech],
        thermal_params=_json_safe(THERMAL_PARAMS),
        ref_heat=_json_safe(ref_heat),
        fixed_heat=bool(fixed_heat),
        coupling_params=_json_safe(COUPLING_PARAMS) if mode == "coupled" else None,
        mechanical_bc_type=str(MECHANICAL_BC_TYPE),
        stopping_norm="L2_Omega",
        reported_refinement_branch="coupled_only",
    )
    payload_json = json.dumps(payload, sort_keys=True, indent=2)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:16], payload


def reference_cache_paths(*, mode, label, bc_type, ref_size, ref_p, cache_key):
    stem = (
        f"paper2_ref_{_safe_filename_part(mode)}_{_safe_filename_part(label)}"
        f"_{_safe_filename_part(bc_type)}_size{int(ref_size)}_p{int(ref_p)}_{cache_key}"
    )
    return dict(
        h5=os.path.join(PAPER2_OUTDIR, stem + ".h5"),
        json=os.path.join(PAPER2_OUTDIR, stem + ".json"),
    )


def _reference_meta(ref, w_save, theta_save, payload):
    return dict(
        payload=payload,
        history=ref.get("history", None),
        converged=bool(ref.get("converged", True)),
        stopped_by_max_iters=bool(ref.get("stopped_by_max_iters", False)),
        termination_reason=str(ref.get("termination_reason", "cached_reference")),
        err_w=float(ref.get("err_w", np.nan)),
        err_T1=float(ref.get("err_T1", np.nan)),
        iters=int(ref.get("iters", 1)),
        w_dofs=int(ref.get("w_dofs", w_save.function_space().dim())),
        heat_dofs=int(ref.get("heat_dofs", 0)),
        theta_dofs=int(ref.get("theta_dofs", theta_save.function_space().dim())),
        total_dofs=int(ref.get("total_dofs", 0)),
    )


def _assert_converged_coupled_record(record, *, context):
    """Stop immediately if a coupled reference/run did not converge."""
    if not bool(record.get("converged", False)):
        raise RuntimeError(
            f"{context} did not converge: "
            f"iters={record.get('iters', None)}, "
            f"err_w={record.get('err_w', np.nan)}, "
            f"err_T1={record.get('err_T1', np.nan)}, "
            f"termination={record.get('termination_reason', None)}"
        )

    if bool(record.get("stopped_by_max_iters", False)):
        raise RuntimeError(
            f"{context} reached max coupling iterations and must not be reported: "
            f"iters={record.get('iters', None)}, "
            f"err_w={record.get('err_w', np.nan)}, "
            f"err_T1={record.get('err_T1', np.nan)}, "
            f"termination={record.get('termination_reason', None)}"
        )


def save_reference_cache(ref, *, paths, payload, ref_p):
    """Save reference mesh, displacement, thermal driver, and metadata."""
    if payload.get("mode") == "coupled":
        _assert_converged_coupled_record(ref, context="Coupled reference before cache save")

    mesh = ref["solver"].mesh
    Vw_ref = FunctionSpace(mesh, "CG", int(ref_p))
    Vtheta_ref = FunctionSpace(mesh, "CG", int(ref_p))

    w_save = project(ref["w"], Vw_ref)
    theta_save = project(ref["theta"], Vtheta_ref)

    h5 = HDF5File(mesh.mpi_comm(), paths["h5"], "w")
    for name, obj in [("/mesh", mesh), ("/w", w_save), ("/theta", theta_save)]:
        h5.write(obj, name)
    h5.close()

    with open(paths["json"], "w") as f:
        json.dump(_json_safe(_reference_meta(ref, w_save, theta_save, payload)), f, indent=2, sort_keys=True)

    print("Saved reference cache:")
    print(f"  {paths['h5']}")
    print(f"  {paths['json']}")


def load_reference_cache(*, paths, ref_p):
    """Load cached reference mesh and fields."""
    if not (os.path.exists(paths["h5"]) and os.path.exists(paths["json"])):
        return None

    try:
        mesh = Mesh()
        h5 = HDF5File(MPI.comm_world, paths["h5"], "r")
        h5.read(mesh, "/mesh", False)

        Vw_ref = FunctionSpace(mesh, "CG", int(ref_p))
        Vtheta_ref = FunctionSpace(mesh, "CG", int(ref_p))
        w_ref = Function(Vw_ref)
        theta_ref = Function(Vtheta_ref)

        h5.read(w_ref, "/w")
        h5.read(theta_ref, "/theta")
        h5.close()

        with open(paths["json"], "r") as f:
            meta = json.load(f)

    except Exception as exc:
        print(f"Reference cache could not be loaded and will be rebuilt: {exc}")
        return None

    ref = dict(
        solver=None,
        mesh=mesh,
        w=w_ref,
        theta=theta_ref,
        heat=None,
        history=meta.get("history", None),
        converged=bool(meta.get("converged", True)),
        stopped_by_max_iters=bool(meta.get("stopped_by_max_iters", False)),
        termination_reason=str(meta.get("termination_reason", "loaded_reference_cache")),
        err_w=float(meta.get("err_w", np.nan)),
        err_T1=float(meta.get("err_T1", np.nan)),
        iters=int(meta.get("iters", 1)),
        w_dofs=int(meta.get("w_dofs", w_ref.function_space().dim())),
        heat_dofs=int(meta.get("heat_dofs", 0)),
        theta_dofs=int(meta.get("theta_dofs", theta_ref.function_space().dim())),
        total_dofs=int(meta.get("total_dofs", 0)),
    )

    if meta.get("payload", {}).get("mode") == "coupled":
        _assert_converged_coupled_record(ref, context="Loaded coupled reference cache")

    print("Loaded cached reference:")
    print(f"  {paths['h5']}")

    return ref


def get_or_build_reference(*, mode, label, bc_type, ref_size, ref_p, mu_mech, fixed_heat=True):
    """
    Build or load the refined FOM numerical reference.

    The revised reported refinement study uses coupled references only.
    """
    if mode != "coupled":
        raise ValueError(
            "The revised Paper 2 refinement study is coupled-only. "
            "Use mode='coupled' for reference construction."
        )

    ref_heat = heat_controls_from_size(ref_size, ref=True, fixed=fixed_heat)
    cache_key, payload = reference_cache_key(
        mode=mode,
        label=label,
        bc_type=bc_type,
        ref_size=ref_size,
        ref_p=ref_p,
        mu_mech=mu_mech,
        ref_heat=ref_heat,
        fixed_heat=fixed_heat,
    )
    paths = reference_cache_paths(
        mode=mode,
        label=label,
        bc_type=bc_type,
        ref_size=ref_size,
        ref_p=ref_p,
        cache_key=cache_key,
    )

    if RESUME_RUNS and not FORCE_RERUN:
        cached = load_reference_cache(paths=paths, ref_p=ref_p)
        if cached is not None:
            return cached

    print("\n" + "=" * 90)
    print(f"BUILDING COUPLED REFERENCE: {label}, bc={bc_type}, N_Omega={ref_size}, p={ref_p}")
    print("=" * 90)

    ref = solve_paper2_case(
        mode=mode,
        size=ref_size,
        p=ref_p,
        mu_mech=mu_mech,
        heat_nx=ref_heat["heat_nx"],
        heat_ny=ref_heat["heat_ny"],
        heat_nz=ref_heat["heat_nz"],
        heat_degree=ref_heat["heat_degree"],
        Nz_quad_T1=ref_heat["Nz_quad_T1"],
        T1_cg_degree=ref_p,
        bc_type=bc_type,
    )

    _assert_converged_coupled_record(ref, context="Newly built coupled reference")
    save_reference_cache(ref, paths=paths, payload=payload, ref_p=ref_p)
    return ref


def reference_mesh(ref):
    """Return the mesh from either a live reference solve or a cached reference."""
    return ref["solver"].mesh if ref.get("solver", None) is not None else ref["mesh"]


# ------------------------------------------------------------
# Convergence gathering
# ------------------------------------------------------------

def _success_record(mode, label, bc_type, p, res, sol, errors):
    ew_rms, etheta_rms, ew_rel, etheta_rel = errors
    return dict(
        mode=str(mode),
        case=str(label),
        bc_type=str(bc_type),
        p=int(p),
        size=int(res),
        h=float(sol["solver"].mesh.hmax()),
        inv_h=float(1.0 / sol["solver"].mesh.hmax()),
        plate_DoFs=int(sol["w_dofs"]),
        heat_DoFs=int(sol["heat_dofs"]),
        theta_DoFs=int(sol["theta_dofs"]),
        Total_DoFs=int(sol["total_dofs"]),
        sqrt_plate_DoFs=float(np.sqrt(sol["w_dofs"])),
        sqrt_Total_DoFs=float(np.sqrt(sol["total_dofs"])),
        heat_nx=int(sol["heat_nx"]),
        heat_ny=int(sol["heat_ny"]),
        heat_nz=int(sol["heat_nz"]),
        heat_degree=int(sol["heat_degree"]),
        Nz_quad_T1=int(sol["Nz_quad_T1"]),
        T1_cg_degree=int(sol["T1_cg_degree"]),
        L2_RMS_Error_w_m=float(ew_rms),
        L2_RMS_Error_theta=float(etheta_rms),
        Rel_L2_Error_w=float(ew_rel),
        Rel_L2_Error_theta=float(etheta_rel),
        coupled_iters=int(sol["iters"]),
        coupled_converged=bool(sol["converged"]),
        stopped_by_max_iters=bool(sol["stopped_by_max_iters"]),
        termination_reason=str(sol["termination_reason"]),
        err_w=float(sol["err_w"]),
        err_theta=float(sol["err_T1"]),
        stopping_norm="L2_Omega",
        reported=True,
    )


def _failed_record(mode, label, bc_type, p, res, hc, exc):
    return dict(
        mode=str(mode),
        case=str(label),
        bc_type=str(bc_type),
        p=int(p),
        size=int(res),
        h=np.nan,
        inv_h=np.nan,
        plate_DoFs=np.nan,
        heat_DoFs=np.nan,
        theta_DoFs=np.nan,
        Total_DoFs=np.nan,
        sqrt_plate_DoFs=np.nan,
        sqrt_Total_DoFs=np.nan,
        heat_nx=int(hc["heat_nx"]),
        heat_ny=int(hc["heat_ny"]),
        heat_nz=int(hc["heat_nz"]),
        heat_degree=int(hc["heat_degree"]),
        Nz_quad_T1=int(hc["Nz_quad_T1"]),
        T1_cg_degree=int(p),
        L2_RMS_Error_w_m=np.nan,
        L2_RMS_Error_theta=np.nan,
        Rel_L2_Error_w=np.nan,
        Rel_L2_Error_theta=np.nan,
        coupled_iters=np.nan,
        coupled_converged=False,
        stopped_by_max_iters=False,
        termination_reason=f"failed_or_not_reported: {exc}",
        err_w=np.nan,
        err_theta=np.nan,
        stopping_norm="L2_Omega",
        reported=False,
    )


def _excluded_record_from_solution(mode, label, bc_type, p, res, sol):
    """Save a non-converged coupled run for traceability, but mark it as not reportable."""
    return dict(
        mode=str(mode),
        case=str(label),
        bc_type=str(bc_type),
        p=int(p),
        size=int(res),
        h=float(sol["solver"].mesh.hmax()),
        inv_h=float(1.0 / sol["solver"].mesh.hmax()),
        plate_DoFs=int(sol["w_dofs"]),
        heat_DoFs=int(sol["heat_dofs"]),
        theta_DoFs=int(sol["theta_dofs"]),
        Total_DoFs=int(sol["total_dofs"]),
        sqrt_plate_DoFs=float(np.sqrt(sol["w_dofs"])),
        sqrt_Total_DoFs=float(np.sqrt(sol["total_dofs"])),
        heat_nx=int(sol["heat_nx"]),
        heat_ny=int(sol["heat_ny"]),
        heat_nz=int(sol["heat_nz"]),
        heat_degree=int(sol["heat_degree"]),
        Nz_quad_T1=int(sol["Nz_quad_T1"]),
        T1_cg_degree=int(sol["T1_cg_degree"]),
        L2_RMS_Error_w_m=np.nan,
        L2_RMS_Error_theta=np.nan,
        Rel_L2_Error_w=np.nan,
        Rel_L2_Error_theta=np.nan,
        coupled_iters=int(sol["iters"]),
        coupled_converged=bool(sol["converged"]),
        stopped_by_max_iters=bool(sol["stopped_by_max_iters"]),
        termination_reason=f"not_reported_nonconverged: {sol['termination_reason']}",
        err_w=float(sol["err_w"]),
        err_theta=float(sol["err_T1"]),
        stopping_norm="L2_Omega",
        reported=False,
    )


def valid_converged_coupled_rows(df):
    """
    Keep only rows admissible for manuscript tables and plots.

    Excludes:
        - one-way rows,
        - failed rows,
        - non-converged coupled rows,
        - rows stopped by max iterations,
        - rows with missing/non-finite error values.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out = out[out["mode"].astype(str) == "coupled"]

    if "coupled_converged" in out.columns:
        out = out[_bool_series(out["coupled_converged"], default=False)]
    if "stopped_by_max_iters" in out.columns:
        out = out[~_bool_series(out["stopped_by_max_iters"], default=False)]
    if "reported" in out.columns:
        out = out[_bool_series(out["reported"], default=True)]
    if "stopping_norm" in out.columns:
        out = out[out["stopping_norm"].astype(str) == "L2_Omega"]

    for col in [
        "h",
        "L2_RMS_Error_w_m",
        "Rel_L2_Error_w",
        "L2_RMS_Error_theta",
        "Rel_L2_Error_theta",
        "err_w",
        "err_theta",
    ]:
        if col in out.columns:
            out = out[np.isfinite(out[col].to_numpy(dtype=float))]

    return out.reset_index(drop=True)


def gather_paper2_convergence_data(
    *,
    mode,
    mu_mech,
    resolutions=RESOLUTIONS_COUPLED_FINAL,
    p_values=P_VALUES,
    ref_size=REF_SIZE,
    ref_p=REF_P,
    label="case",
    bc_type=MECHANICAL_BC_TYPE,
    fixed_heat=True,
):
    """
    Gather coupled-only convergence data using a cached refined FOM reference.

    Non-converged coupled runs are saved as failed/non-reported rows, but they
    are excluded from all manuscript tables and plots.
    """
    if mode != "coupled":
        raise ValueError(
            "The revised Paper 2 refinement study must be coupled-only. "
            "Do not call gather_paper2_convergence_data with mode='one_way'."
        )

    out_csv = convergence_csv_path(mode, label, bc_type)
    existing_df, existing_keys = load_existing_convergence_rows(out_csv)

    ref = get_or_build_reference(
        mode=mode,
        label=label,
        bc_type=bc_type,
        ref_size=ref_size,
        ref_p=ref_p,
        mu_mech=mu_mech,
        fixed_heat=fixed_heat,
    )
    _assert_converged_coupled_record(ref, context=f"Coupled reference for {label}, bc={bc_type}")

    ref_mesh_obj = reference_mesh(ref)
    Vw_ref = FunctionSpace(ref_mesh_obj, "CG", int(ref_p))
    Vtheta_ref = FunctionSpace(ref_mesh_obj, "CG", int(ref_p))
    dx_ref = Measure("dx", domain=ref_mesh_obj)
    sqrt_area = float(np.sqrt(assemble(Constant(1.0) * dx_ref)))

    w_ref = project(ref["w"], Vw_ref)
    theta_ref = project(ref["theta"], Vtheta_ref)

    for res in resolutions:
        for p in p_values:
            if int(res) == int(ref_size) and int(p) == int(ref_p):
                continue

            case_key = (mode, label, bc_type, int(p), int(res))
            if case_key in existing_keys:
                print(f"SKIPPING valid cached row: mode={mode}, case={label}, bc={bc_type}, size={res}, p={p}")
                continue

            hc = heat_controls_from_size(res, ref=False, fixed=fixed_heat)
            print("\n" + "-" * 90)
            print(
                f"COUPLED | {label} | bc={bc_type} | size={res}, p={p}, "
                f"heat=({hc['heat_nx']},{hc['heat_ny']},{hc['heat_nz']}), "
                f"pT={hc['heat_degree']}, Nz_quad_T1={hc['Nz_quad_T1']}"
            )
            print("-" * 90)

            try:
                sol = solve_paper2_case(
                    mode=mode,
                    size=res,
                    p=p,
                    mu_mech=mu_mech,
                    heat_nx=hc["heat_nx"],
                    heat_ny=hc["heat_ny"],
                    heat_nz=hc["heat_nz"],
                    heat_degree=hc["heat_degree"],
                    Nz_quad_T1=hc["Nz_quad_T1"],
                    T1_cg_degree=p,
                    bc_type=bc_type,
                )

                if not _coupled_solution_is_converged(sol):
                    print(
                        f"EXCLUDED NON-CONVERGED RUN: case={label}, size={res}, p={p}, "
                        f"iters={sol.get('iters', None)}, err_w={sol.get('err_w', np.nan)}, "
                        f"err_T1={sol.get('err_T1', np.nan)}, termination={sol.get('termination_reason', None)}"
                    )
                    record = _excluded_record_from_solution(mode, label, bc_type, p, res, sol)

                else:
                    deg = max(5, int(p) + 2)
                    errors = (
                        l2_rms_error_against_ref(sol["w"], w_ref, Vw_ref, sqrt_area, degree=deg),
                        l2_rms_error_against_ref(sol["theta"], theta_ref, Vtheta_ref, sqrt_area, degree=deg),
                        l2_relative_error_against_ref(sol["w"], w_ref, Vw_ref, degree=deg),
                        l2_relative_error_against_ref(sol["theta"], theta_ref, Vtheta_ref, degree=deg),
                    )
                    record = _success_record(mode, label, bc_type, p, res, sol, errors)

            except Exception as exc:
                print(f"FAILED OR EXCLUDED: mode={mode}, case={label}, bc={bc_type}, size={res}, p={p}: {exc}")
                record = _failed_record(mode, label, bc_type, p, res, hc, exc)

            existing_df = save_convergence_checkpoint([record], existing_df, out_csv)

            if _row_is_valid_converged_coupled(pd.Series(record)):
                existing_keys.add(case_key)

    df = existing_df.copy() if existing_df is not None and not existing_df.empty else pd.DataFrame()

    if not df.empty:
        df = df.sort_values(
            ["mode", "case", "bc_type", "p", "h"],
            ascending=[True, True, True, True, False],
        ).reset_index(drop=True)

    df.to_csv(out_csv, index=False)
    df_valid = valid_converged_coupled_rows(df)

    print(f"\nSaved convergence table: {out_csv}")
    print(f"Total rows in CSV: {len(df)}")
    print(f"Valid converged coupled rows for reporting: {len(df_valid)}")

    return df, ref


# ------------------------------------------------------------
# Table helper
# ------------------------------------------------------------

def finest_converged_rows_for_table(df):
    """Return the finest valid coupled row for each case and p."""
    df_valid = valid_converged_coupled_rows(df)
    if df_valid.empty:
        return df_valid

    rows = [grp.sort_values("size", ascending=False).iloc[0] for _, grp in df_valid.groupby(["case", "p"])]
    return pd.DataFrame(rows).sort_values(["case", "p"]).reset_index(drop=True)


def print_finest_converged_table(df, title="FINEST CONVERGED COUPLED ROWS"):
    table_df = finest_converged_rows_for_table(df)

    print("\n" + title)
    if table_df.empty:
        print("No valid converged coupled rows available.")
        return table_df

    print(table_df[TABLE_COLS].to_string(index=False, float_format="%.6e"))
    return table_df


# ------------------------------------------------------------
# Coupled-only convergence plotting: revised Fig. 3
# ------------------------------------------------------------

def plot_paper2_coupled_convergence_iso_ortho(
    df_coupled,
    *,
    ycol="L2_RMS_Error_w_m",
    ylabel=r"$L^2$ RMS error in $w_h$ [m]",
    x_mode="h",
    filename_base=None,
    rate_fit_slice=slice(-2, None),
):
    """
    Revised Paper 2 Fig. 3:
    coupled-only displacement convergence for isotropic/orthotropic cases.

    Only valid converged coupled rows are plotted.

    Note:
    For x_mode='h', the x-axis is intentionally shown from coarse to fine,
    i.e. larger h on the left and smaller h on the right.

    Rate triangles:
    - rates are computed from the log-log slope selected by rate_fit_slice;
      the default slice(-2, None) uses the last two finest mesh points.
      Use slice(None) only if a global fitted rate over all refinement data is desired.
    - all triangles are kept near the finest segment;
    - triangles are compact to avoid overlap;
    - for each polynomial degree p, the higher-error curve gets the upper triangle
      and the lower-error curve gets the lower triangle;
    - the label "1" is always placed above the horizontal triangle edge.
    """

    if filename_base is None:
        filename_base = f"paper2_isotropic_orthotropic_simply_supported_coupled_w_convergence_mesh_size_{STUDY_TAG}"

    df = valid_converged_coupled_rows(df_coupled)
    if df.empty:
        raise ValueError("No valid converged coupled rows available for Fig. 3.")

    plt.style.use(["science", "ieee", "notebook", "grid"])
    plt.rcParams["figure.autolayout"] = False

    x_options = {
        "h": ("h", r"Characteristic mesh length $\ell_{\mathcal{T}}$ [m]"),
        "inv_h": ("inv_h", r"Inverse mesh size $1/h$"),
        "sqrt_plate_dofs": ("sqrt_plate_DoFs", r"$\sqrt{\mathrm{plate\ DoFs}}$"),
        "plate_dofs": ("plate_DoFs", "Plate degrees of freedom"),
        "sqrt_total_dofs": ("sqrt_Total_DoFs", r"$\sqrt{\mathrm{total\ DoFs}}$"),
        "total_dofs": ("Total_DoFs", "Total degrees of freedom"),
    }
    if x_mode not in x_options:
        raise ValueError(f"x_mode must be one of {list(x_options.keys())}")

    xcol, xlabel = x_options[x_mode]

    style_specs = [
        ("Isotropic", 2, "black", "-", "o", r"Isotropic, $p=2$"),
        ("Isotropic", 3, "blue", "--", "s", r"Isotropic, $p=3$"),
        ("Orthotropic", 2, "darkorange", "-", "o", r"Orthotropic, $p=2$"),
        ("Orthotropic", 3, "purple", "--", "s", r"Orthotropic, $p=3$"),
    ]
    styles = {
        (case_name, p): dict(color=color, linestyle=linestyle, marker=marker, label=label)
        for case_name, p, color, linestyle, marker, label in style_specs
    }

    def positive_finite(vals):
        vals = np.asarray(vals, dtype=float)
        return vals[np.isfinite(vals) & (vals > 0.0)]

    def ordered_xy(x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
        x, y = x[mask], y[mask]
        order = np.argsort(x)[::-1] if x_mode == "h" else np.argsort(x)
        return x[order], y[order]

    def estimate_loglog_slope(x, y, fit_slice=slice(None)):
        x, y = ordered_xy(x, y)
        if len(x) < 2:
            return None

        xx, yy = x[fit_slice], y[fit_slice]
        if len(xx) < 2:
            return None

        slope, _ = np.polyfit(np.log10(xx), np.log10(yy), 1)
        return float(abs(slope))

    def fine_error_value(grp):
        x, y = ordered_xy(grp[xcol].to_numpy(float), grp[ycol].to_numpy(float))
        return np.nan if len(y) == 0 else float(y[-1])

    def build_triangle_specs(grouped):
        """
        Automatic compact triangle placement.

        For each polynomial degree:
            higher-error curve -> triangle above,
            lower-error curve  -> triangle below.

        All triangles stay near the finest segment. The p=3 triangles are made
        slightly smaller than p=2 because the p=3 curves are closer together.
        """
        specs = {}

        for p in sorted({key[1] for key in grouped}):
            keys_p = [key for key in grouped if key[1] == p]
            if not keys_p:
                continue

            keys_p = sorted(keys_p, key=lambda key: fine_error_value(grouped[key]))

            if len(keys_p) == 1:
                specs[keys_p[0]] = dict(
                    location="above",
                    start_idx=-2,
                    span_frac=0.34,
                    tri_gap=1.22,
                    rate_side_pad=0.020,
                    one_label_gap=1.055,
                    rate_label_shift=1.00,
                )
                continue

            lower_key = keys_p[0]
            upper_key = keys_p[-1]

            if int(p) == 2:
                specs[upper_key] = dict(
                    location="above",
                    start_idx=-2,
                    span_frac=0.36,
                    tri_gap=1.24,
                    rate_side_pad=0.020,
                    one_label_gap=1.055,
                    rate_label_shift=1.00,
                )
                specs[lower_key] = dict(
                    location="below",
                    start_idx=-2,
                    span_frac=0.36,
                    tri_gap=1.28,
                    rate_side_pad=0.020,
                    one_label_gap=1.055,
                    rate_label_shift=1.00,
                )
            else:
                specs[upper_key] = dict(
                    location="above",
                    start_idx=-2,
                    span_frac=0.28,
                    tri_gap=1.20,
                    rate_side_pad=0.018,
                    one_label_gap=1.060,
                    rate_label_shift=1.00,
                )
                specs[lower_key] = dict(
                    location="below",
                    start_idx=-2,
                    span_frac=0.28,
                    tri_gap=1.26,
                    rate_side_pad=0.018,
                    one_label_gap=1.060,
                    rate_label_shift=1.00,
                )

        return specs

    def add_rate_triangle(
        ax,
        x,
        y,
        rate,
        color,
        *,
        start_idx=-2,
        span_frac=0.34,
        tri_gap=1.22,
        rate_side_pad=0.020,
        one_label_gap=1.055,
        rate_label_shift=1.00,
        lw=1.00,
        fontsize=8,
        location="above",
    ):
        if rate is None or rate <= 0.0:
            return

        x, y = ordered_xy(x, y)
        if len(x) < 2:
            return

        start_idx = len(x) + start_idx if start_idx < 0 else start_idx
        if start_idx < 0 or start_idx + 1 >= len(x):
            return

        x_prev = float(x[start_idx])
        x_last = float(x[start_idx + 1])
        y_last = float(y[start_idx + 1])

        if x_prev <= 0.0 or x_last <= 0.0 or y_last <= 0.0:
            return

        if x_mode == "h":
            full_ratio = x_prev / x_last
            if full_ratio <= 1.0:
                return

            x_right = x_last
            x_left = x_last * (full_ratio ** span_frac)
            local_ratio = x_left / x_right
            x_rate = x_right / (local_ratio ** rate_side_pad)

        else:
            full_ratio = x_last / x_prev
            if full_ratio <= 1.0:
                return

            x_left = x_last / (full_ratio ** span_frac)
            x_right = x_last
            local_ratio = x_right / x_left
            x_rate = x_right * (local_ratio ** rate_side_pad)

        if location == "below":
            # Triangle is below the curve.
            # y_top is still the horizontal edge of the triangle.
            y_top = y_last / tri_gap
            y_bottom = y_top / (local_ratio ** rate)
        else:
            # Triangle is above the curve.
            # y_top is the horizontal edge of the triangle.
            y_bottom = y_last * tri_gap
            y_top = y_bottom * (local_ratio ** rate)

        if min(x_left, x_right, y_top, y_bottom) <= 0.0:
            return

        # Important fix:
        # Put the "1" label above the horizontal edge for both upper and lower triangles.
        one_label_y = y_top * one_label_gap
        one_label_va = "bottom"

        y_mid = np.sqrt(y_top * y_bottom) * float(rate_label_shift)

        ymin, ymax = ax.get_ylim()
        log_ymin, log_ymax = np.log10(ymin), np.log10(ymax)
        low = np.log10(min(y_top, y_bottom))
        high = np.log10(max(y_top, y_bottom, one_label_y))

        if low < log_ymin + 0.08:
            scale = 10 ** (log_ymin + 0.08) / min(y_top, y_bottom)
            y_top *= scale
            y_bottom *= scale
            one_label_y *= scale
            y_mid *= scale

        if high > log_ymax - 0.08:
            scale = 10 ** (log_ymax - 0.08) / max(y_top, y_bottom, one_label_y)
            y_top *= scale
            y_bottom *= scale
            one_label_y *= scale
            y_mid *= scale

        text_box = dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.12)

        ax.plot([x_left, x_right], [y_top, y_top], color=color, lw=lw, zorder=5)
        ax.plot([x_right, x_right], [y_top, y_bottom], color=color, lw=lw, zorder=5)
        ax.plot([x_left, x_right], [y_top, y_bottom], color=color, lw=lw, zorder=5)

        ax.text(
            np.sqrt(x_left * x_right),
            one_label_y,
            "1",
            ha="center",
            va=one_label_va,
            fontsize=fontsize,
            color=color,
            bbox=text_box,
            clip_on=True,
            zorder=6,
        )

        ax.text(
            x_rate,
            y_mid,
            f"{rate:.1f}",
            ha="left",
            va="center",
            fontsize=fontsize,
            color=color,
            bbox=text_box,
            clip_on=True,
            zorder=6,
        )

    all_x = positive_finite(df[xcol].to_numpy(float))
    all_y = positive_finite(df[ycol].to_numpy(float))
    if len(all_x) == 0 or len(all_y) == 0:
        raise ValueError("No positive finite data available for Fig. 3.")

    fig, ax = plt.subplots(figsize=(6.8, 5.2), facecolor="white")
    grouped = {}

    for key, style in styles.items():
        case_name, p = key
        grp = df[
            (df["case"].astype(str) == case_name)
            & (df["p"].astype(int) == int(p))
        ].dropna(subset=[xcol, ycol]).copy()

        if grp.empty:
            continue

        grp = grp.sort_values(xcol, ascending=(x_mode != "h"))
        grouped[key] = grp

        ax.loglog(
            grp[xcol].to_numpy(float),
            grp[ycol].to_numpy(float),
            linestyle=style["linestyle"],
            marker=style["marker"],
            color=style["color"],
            markeredgecolor=style["color"],
            markerfacecolor="none",
            linewidth=2.2,
            markersize=8.0,
            markeredgewidth=1.4,
            label=style["label"],
            clip_on=False,
        )

    ax.set_facecolor("white")
    ax.set_title(r"Partitioned coupled refinement study", fontsize=14, pad=8)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=13, length=4, pad=2)
    ax.tick_params(axis="both", which="minor", length=2)
    ax.grid(True, which="both", linestyle=":", linewidth=0.45, alpha=0.35)

    if x_mode == "h":
        ticks = []
        for size in [8, 16, 32, 64]:
            vals = positive_finite(df.loc[df["size"].astype(int) == size, "h"].to_numpy(float))
            if len(vals):
                ticks.append(float(np.median(vals)))

        if ticks:
            ticks = sorted(set(np.round(ticks, 12)), reverse=True)
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, pos: f"{x:.2f}" if x >= 0.1 else f"{x:.3f}")
            )
    else:
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=80))
    ax.xaxis.set_minor_formatter(NullFormatter())

    # Important:
    # For h-refinement, show coarse-to-fine from left to right:
    # left  = largest h, approximately 0.25
    # right = smallest h, approximately 0.031
    if x_mode == "h":
        ax.set_xlim(np.max(all_x) * 1.20, np.min(all_x) / 1.14)
    else:
        ax.set_xlim(np.min(all_x) / 1.14, np.max(all_x) * 1.20)

    ax.set_ylim(np.min(all_y) / 2.60, np.max(all_y) * 1.65)

    triangle_specs = build_triangle_specs(grouped)

    for key, grp in grouped.items():
        style = styles[key]
        spec = triangle_specs.get(
            key,
            dict(
                location="above",
                start_idx=-2,
                span_frac=0.34,
                tri_gap=1.22,
                rate_side_pad=0.020,
                one_label_gap=1.055,
                rate_label_shift=1.00,
            ),
        )

        rate = estimate_loglog_slope(
            grp[xcol].to_numpy(float),
            grp[ycol].to_numpy(float),
            fit_slice=rate_fit_slice,
        )

        add_rate_triangle(
            ax,
            grp[xcol].to_numpy(float),
            grp[ycol].to_numpy(float),
            rate,
            style["color"],
            location=spec["location"],
            start_idx=spec["start_idx"],
            span_frac=spec["span_frac"],
            tri_gap=spec["tri_gap"],
            rate_side_pad=spec["rate_side_pad"],
            one_label_gap=spec["one_label_gap"],
            rate_label_shift=spec["rate_label_shift"],
            lw=1.00,
            fontsize=8,
        )

    leg = ax.legend(
        loc="lower left",
        fontsize=12,
        frameon=True,
        borderpad=0.55,
        handlelength=2.2,
        handletextpad=0.55,
        labelspacing=0.35,
    )
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.8)

    os.makedirs(PAPER2_OUTDIR, exist_ok=True)
    for ext, dpi in (("pdf", 300), ("png", 600)):
        fig.savefig(
            os.path.join(PAPER2_OUTDIR, f"{filename_base}.{ext}"),
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
        )

    plt.show()

# ------------------------------------------------------------
# Coupled update-history plotting: revised Fig. 4
# ------------------------------------------------------------

def history_dataframe_from_ref(ref_or_solution, case_label):
    """Convert a coupled reference/solution history into a DataFrame for Fig. 4."""
    hist = pd.DataFrame(ref_or_solution.get("history", None)).copy()
    if hist.empty:
        return hist

    hist["case"] = str(case_label)
    return hist


def load_saved_coupled_reference_history(case_label, *, bc_type=MECHANICAL_BC_TYPE):
    """
    Load coupled-reference fixed-point history from saved reference-cache JSON.
    No FOM solve is executed.
    """

    pattern = os.path.join(
        PAPER2_OUTDIR,
        (
            f"paper2_ref_coupled_{_safe_filename_part(case_label)}"
            f"_{_safe_filename_part(bc_type)}"
            f"_size{int(REF_SIZE)}_p{int(REF_P)}_*.json"
        ),
    )
    candidates = sorted(glob.glob(pattern))

    if not candidates:
        raise FileNotFoundError(f"No coupled reference-cache JSON found for {case_label}: {pattern}")

    for path in candidates:
        with open(path, "r") as f:
            meta = json.load(f)

        payload = meta.get("payload", {})
        is_match = (
            payload.get("mode") == "coupled"
            and payload.get("case") == case_label
            and payload.get("bc_type") == bc_type
            and int(payload.get("ref_size", -1)) == int(REF_SIZE)
            and int(payload.get("ref_p", -1)) == int(REF_P)
            and payload.get("ref_heat", {}) == _json_safe(REF_HEAT_CONTROLS)
            and payload.get("stopping_norm") == "L2_Omega"
            and payload.get("reported_refinement_branch") == "coupled_only"
        )
        hist = meta.get("history", None)

        if is_match and hist:
            df = pd.DataFrame(hist).copy()
            df["case"] = case_label
            df["source_file"] = os.path.basename(path)
            return df

    raise FileNotFoundError(f"No valid coupled history found for {case_label} in {len(candidates)} JSON file(s).")


def plot_paper2_coupled_history_iso_ortho_combined(
    histories,
    *,
    filename_base=None,
):
    """
    Revised Paper 2 Fig. 4:
    fixed-point relative-update histories only.

    The old response-evolution panel is intentionally removed.
    """
    if filename_base is None:
        filename_base = f"paper2_isotropic_orthotropic_simply_supported_coupled_reference_update_history_{STUDY_TAG}"

    if not histories:
        print("No coupled histories available.")
        return

    hist = pd.concat(histories, ignore_index=True)
    if hist.empty:
        print("No coupled histories available.")
        return

    plt.style.use(["science", "ieee", "notebook", "grid"])
    plt.rcParams["figure.autolayout"] = False

    hist_styles = {
        ("Isotropic", "err_w"): dict(color="black", linestyle="-", marker="o", label=r"Isotropic, $\varepsilon_w$"),
        ("Isotropic", "err_T1"): dict(color="red", linestyle="--", marker="s", label=r"Isotropic, $\varepsilon_{\theta}$"),
        ("Orthotropic", "err_w"): dict(color="blue", linestyle="-", marker="o", label=r"Orthotropic, $\varepsilon_w$"),
        ("Orthotropic", "err_T1"): dict(color="purple", linestyle="--", marker="s", label=r"Orthotropic, $\varepsilon_{\theta}$"),
    }

    fig, ax = plt.subplots(figsize=(6.8, 5.2), facecolor="white")

    for case_name in ["Isotropic", "Orthotropic"]:
        grp = hist[hist["case"].astype(str) == case_name].copy()
        if grp.empty:
            continue

        grp = grp.sort_values("iter")
        it = grp["iter"].to_numpy(int)

        for col in ["err_w", "err_T1"]:
            vals = grp[col].to_numpy(float)
            if col == "err_T1" and not np.isfinite(vals[1:]).any():
                continue

            style = hist_styles[(case_name, col)]
            ax.semilogy(
                it,
                vals,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2.2,
                markersize=7.0,
                markerfacecolor=style["color"],
                label=style["label"],
            )

    ax.set_facecolor("white")
    ax.set_title("Coupled fixed-point update history", fontsize=14, pad=8)
    ax.set_xlabel("Coupling iteration", fontsize=14)
    ax.set_ylabel(r"Relative $L^2(\Omega)$ update", fontsize=14)
    ax.tick_params(axis="both", labelsize=13, length=3, pad=2)
    ax.grid(True, which="both", linestyle=":", linewidth=0.45, alpha=0.35)

    leg = ax.legend(
        loc="upper right",
        fontsize=11.5,
        frameon=True,
        borderpad=0.55,
        handlelength=2.1,
        handletextpad=0.55,
        labelspacing=0.35,
    )
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.8)

    os.makedirs(PAPER2_OUTDIR, exist_ok=True)
    for ext, dpi in (("pdf", 300), ("png", 600)):
        fig.savefig(
            os.path.join(PAPER2_OUTDIR, f"{filename_base}.{ext}"),
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
        )

    plt.show()
    
# ------------------------------------------------------------
# Optional reference-level comparison
# ------------------------------------------------------------

def compare_two_reference_levels(
    mode,
    mu_mech,
    *,
    size_a=REFERENCE_COMPARISON_COARSE_SIZE,
    size_b=REF_SIZE,
    p=REF_P,
    label="reference_check",
    bc_type=MECHANICAL_BC_TYPE,
    fixed_heat=True,
    cache_label_a=None,
    cache_label_b=None,
):
    """
    Optional cached comparison between two coupled refined reference levels.

    This is not the main manuscript convergence table. It is a diagnostic to
    check the separation between the finest tested mesh and the reference.
    """
    if mode != "coupled":
        raise ValueError("Reference-level comparison for the revised study must be coupled-only.")

    cache_label_a = cache_label_a or f"{label}_size{int(size_a)}"
    cache_label_b = cache_label_b or f"{label}_size{int(size_b)}"

    sol_a = get_or_build_reference(
        mode=mode,
        label=cache_label_a,
        bc_type=bc_type,
        ref_size=size_a,
        ref_p=p,
        mu_mech=mu_mech,
        fixed_heat=fixed_heat,
    )
    sol_b = get_or_build_reference(
        mode=mode,
        label=cache_label_b,
        bc_type=bc_type,
        ref_size=size_b,
        ref_p=p,
        mu_mech=mu_mech,
        fixed_heat=fixed_heat,
    )

    _assert_converged_coupled_record(sol_a, context=f"Reference level A size={size_a}")
    _assert_converged_coupled_record(sol_b, context=f"Reference level B size={size_b}")

    mesh_b = reference_mesh(sol_b)
    Vw_b = FunctionSpace(mesh_b, "CG", int(p))
    Vtheta_b = FunctionSpace(mesh_b, "CG", int(p))
    sqrt_area_b = float(np.sqrt(assemble(Constant(1.0) * Measure("dx", domain=mesh_b))))
    deg = max(5, int(p) + 2)

    comparison = dict(
        mode=mode,
        label=label,
        bc_type=bc_type,
        size_a=int(size_a),
        size_b=int(size_b),
        p=int(p),
        L2_RMS_Error_w_m=l2_rms_error_against_ref(sol_a["w"], sol_b["w"], Vw_b, sqrt_area_b, degree=deg),
        L2_RMS_Error_theta=l2_rms_error_against_ref(sol_a["theta"], sol_b["theta"], Vtheta_b, sqrt_area_b, degree=deg),
        Rel_L2_Error_w=l2_relative_error_against_ref(sol_a["w"], sol_b["w"], Vw_b, degree=deg),
        Rel_L2_Error_theta=l2_relative_error_against_ref(sol_a["theta"], sol_b["theta"], Vtheta_b, degree=deg),
    )

    out_json = os.path.join(
        PAPER2_OUTDIR,
        f"paper2_{mode}_{_safe_filename_part(label)}_{bc_type}_{STUDY_TAG}_reference_comparison.json",
    )

    with open(out_json, "w") as f:
        json.dump(_json_safe(comparison), f, indent=2, sort_keys=True)

    print("\nREFERENCE-LEVEL COMPARISON")
    for key, value in comparison.items():
        print(f"{key}: {value}")
    print(f"Saved reference comparison: {out_json}")

    return sol_a, sol_b, comparison


# ------------------------------------------------------------
# Driver utilities: coupled-only
# ------------------------------------------------------------

def run_smoke_test(case_label="Isotropic", *, bc_type=MECHANICAL_BC_TYPE):
    """Coupled-only smoke test. This avoids accidentally reintroducing one-way refinement rows."""
    mu_mech = PLATE_CASES[case_label]["mu_mech"]

    df_cpl, ref_cpl = gather_paper2_convergence_data(
        mode="coupled",
        mu_mech=mu_mech,
        resolutions=RESOLUTIONS_SMOKE,
        p_values=[2],
        ref_size=32,
        ref_p=3,
        label=f"{case_label}_smoke",
        bc_type=bc_type,
        fixed_heat=True,
    )

    print(f"\nCOUPLED SMOKE {case_label.upper()} ({bc_type})")
    print(valid_converged_coupled_rows(df_cpl)[TABLE_COLS].to_string(index=False, float_format="%.6e"))

    return df_cpl, ref_cpl


def run_case(case_label, *, bc_type=MECHANICAL_BC_TYPE):
    """Run the revised Paper 2 refinement study for one material case. Coupled-only."""
    mu_mech = PLATE_CASES[case_label]["mu_mech"]

    df_cpl, ref_cpl = gather_paper2_convergence_data(
        mode="coupled",
        mu_mech=mu_mech,
        resolutions=RESOLUTIONS_COUPLED_FINAL,
        p_values=P_VALUES,
        ref_size=REF_SIZE,
        ref_p=REF_P,
        label=case_label,
        bc_type=bc_type,
        fixed_heat=True,
    )

    print(f"\nCOUPLED {case_label.upper()} ({bc_type})")
    df_valid = valid_converged_coupled_rows(df_cpl)

    if df_valid.empty:
        print("No valid converged coupled rows available.")
    else:
        print(df_valid[TABLE_COLS].to_string(index=False, float_format="%.6e"))

    return df_cpl, ref_cpl


def replot_coupled_from_saved_csv(*, bc_type=MECHANICAL_BC_TYPE):
    """Replot revised Fig. 3 from saved coupled CSVs only."""
    frames = []
    for case_label in ["Isotropic", "Orthotropic"]:
        path = convergence_csv_path("coupled", case_label, bc_type)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing coupled convergence CSV: {path}")
        frames.append(pd.read_csv(path))

    df_all = pd.concat(frames, ignore_index=True)
    plot_paper2_coupled_convergence_iso_ortho(
        df_all,
        ycol="L2_RMS_Error_w_m",
        ylabel=r"$L^2$ RMS error in $w_h$ [m]",
        x_mode="h",
        filename_base=f"paper2_isotropic_orthotropic_simply_supported_coupled_w_convergence_mesh_size_{STUDY_TAG}",
        rate_fit_slice=slice(-2, None),
    )

    return df_all


def replot_coupled_history_from_saved_json(*, bc_type=MECHANICAL_BC_TYPE):
    """Replot revised Fig. 4 from saved coupled reference-cache JSON histories only."""
    histories = [
        load_saved_coupled_reference_history("Isotropic", bc_type=bc_type),
        load_saved_coupled_reference_history("Orthotropic", bc_type=bc_type),
    ]
    plot_paper2_coupled_history_iso_ortho_combined(
        histories,
        filename_base=f"paper2_isotropic_orthotropic_simply_supported_coupled_reference_update_history_{STUDY_TAG}",
    )

    return histories


# ------------------------------------------------------------
# Execute revised coupled-only study
# ------------------------------------------------------------

# Optional quick test:
# df_cpl_smoke, ref_cpl_smoke = run_smoke_test(
#     "Isotropic",
#     bc_type=MECHANICAL_BC_TYPE,
# )

df_cpl_iso, ref_cpl_iso = run_case("Isotropic", bc_type=MECHANICAL_BC_TYPE)
df_cpl_ortho, ref_cpl_ortho = run_case("Orthotropic", bc_type=MECHANICAL_BC_TYPE)

df_cpl_all = pd.concat([df_cpl_iso, df_cpl_ortho], ignore_index=True)
df_cpl_valid = valid_converged_coupled_rows(df_cpl_all)

print_finest_converged_table(
    df_cpl_valid,
    title="FINEST-MESH CONVERGED COUPLED ROWS FOR MANUSCRIPT TABLE",
)

plot_paper2_coupled_convergence_iso_ortho(
    df_cpl_valid,
    ycol="L2_RMS_Error_w_m",
    ylabel=r"$L^2$ RMS error in $w_h$ [m]",
    x_mode="h",
    filename_base=f"paper2_isotropic_orthotropic_simply_supported_coupled_w_convergence_mesh_size_{STUDY_TAG}",
    rate_fit_slice=slice(-2, None),
)

histories_for_fig4 = [
    history_dataframe_from_ref(ref_cpl_iso, "Isotropic"),
    history_dataframe_from_ref(ref_cpl_ortho, "Orthotropic"),
]

plot_paper2_coupled_history_iso_ortho_combined(
    histories_for_fig4,
    filename_base=f"paper2_isotropic_orthotropic_simply_supported_coupled_reference_update_history_{STUDY_TAG}",
)

if RUN_REFERENCE_COMPARISON:
    ref_coarse_cpl, ref_reference_cpl, ref_compare = compare_two_reference_levels(
        "coupled",
        PLATE_CASES["Isotropic"]["mu_mech"],
        size_a=REFERENCE_COMPARISON_COARSE_SIZE,
        size_b=REF_SIZE,
        p=REF_P,
        label=f"Isotropic_coupled_{REFERENCE_COMPARISON_COARSE_SIZE}_vs_{REF_SIZE}_{MECHANICAL_BC_TYPE}",
        bc_type=MECHANICAL_BC_TYPE,
        fixed_heat=True,
        cache_label_a=f"Isotropic_coupled_refcheck_{REFERENCE_COMPARISON_COARSE_SIZE}_{MECHANICAL_BC_TYPE}",
        cache_label_b="Isotropic",
    )

# %% Cell 35 | id: b9f8b6b4-be40-466d-b413-eab3125972c1


# %% Cell 36 | id: 903c030a-698f-4b36-af63-1b2a50ee3e9f


# %% Cell 37 | id: 1586b8da-37f2-4822-8f53-fa30d5d65eba


# %% Cell 38 | id: 4b8d2ea1-2fc4-4678-ac0c-2065fa523cba


# %% Cell 39 | id: 79257a3f-95f2-4c76-93e1-5c2a5070bd9f


# %% Cell 40 | id: 989a2e5e-3048-4424-a623-1fd48a588138


# %% Cell 41 | id: c1459e74-3db5-4bd8-be08-13f637296bbe


# %% Cell 42 | id: e0913a43-84d7-40cc-a6d2-e4e85b46b6fb


# %% Cell 43 | id: 5d1a4278-e700-42cf-819a-b45333878197
# ============================================================
# Paper 2 thermomechanical FOM convergence with refined FOM reference
# ============================================================

PAPER2_OUTDIR = "paper2_fom_convergence"
os.makedirs(PAPER2_OUTDIR, exist_ok=True)

MECHANICAL_BC_TYPE = "simply_supported"
RESOLUTIONS_SMOKE = [8, 11, 16]
RESOLUTIONS_FINAL = [8, 11, 16, 23, 32, 45, 64]
RESOLUTIONS_COUPLED_FINAL = [8, 11, 16, 23, 32, 45, 64]
P_VALUES = [2, 3]

# ------------------------------------------------------------
# User-decision block
# ------------------------------------------------------------
RESUME_RUNS = True
FORCE_RERUN = False
RUN_REFERENCE_COMPARISON = True

# Current working reference setting.
REF_SIZE, REF_P = 100, 4

# Current working fixed heat discretization.
HEAT_NX, HEAT_NY, HEAT_NZ = 64, 32, 16
HEAT_DEGREE = 1
NZ_QUAD_T1 = 20

# General coarse/reference comparison size.
REFERENCE_COMPARISON_COARSE_SIZE = max(RESOLUTIONS_COUPLED_FINAL)

REFERENCE_CACHE_VERSION = 2
CONVERGENCE_TABLE_VERSION = 2

FIXED_HEAT_CONTROLS = dict(
    heat_nx=int(HEAT_NX),
    heat_ny=int(HEAT_NY),
    heat_nz=int(HEAT_NZ),
    heat_degree=int(HEAT_DEGREE),
    Nz_quad_T1=int(NZ_QUAD_T1),
)
REF_HEAT_CONTROLS = dict(FIXED_HEAT_CONTROLS)

# Settings-dependent label used in saved CSV/figure/reference-comparison names.
STUDY_TAG = (
    f"refN{int(REF_SIZE)}_p{int(REF_P)}"
    f"_heat{int(HEAT_NX)}x{int(HEAT_NY)}x{int(HEAT_NZ)}"
    f"_pT{int(HEAT_DEGREE)}"
    f"_qT1{int(NZ_QUAD_T1)}"
    f"_v{int(CONVERGENCE_TABLE_VERSION)}"
)

PLATE_CASES = {
    "Isotropic": dict(mu_mech=[1.0e4, 1.0e4, 0.3e4, 0.35e4, 1.0e3, -10000.0]),
    "Orthotropic": dict(mu_mech=[2.0e4, 1.0e4, 0.25e4, 0.30e4, 1.0e3, -10000.0]),
}

THERMAL_PARAMS = dict(
    T_amb=27.0 + 273.15,
    T_sub=24.0 + 273.15,
    h_con=10.0,
    eps_r=0.90,
    q_s=700.0,
    h_c_cont=200.0,
    h_c_gap=5.0,
    eta_c=50.0,
    w_contact=-0.1,
    kx=0.35,
    ky=0.35,
    kz=0.35,
    alpha1=1.3e-4,
    alpha2=1.3e-4,
    rho=1050.0,
)

COUPLING_PARAMS = dict(
    coupling_omega=0.7,
    coupling_tol_w=1e-6,
    coupling_tol_T1=1e-6,
    coupling_max_iters=40,
)

TABLE_COLS = """
mode case bc_type p size h inv_h plate_DoFs heat_DoFs theta_DoFs Total_DoFs
sqrt_plate_DoFs sqrt_Total_DoFs L2_RMS_Error_w_m Rel_L2_Error_w
L2_RMS_Error_theta Rel_L2_Error_theta coupled_iters coupled_converged
stopped_by_max_iters termination_reason err_w err_theta
""".split()

NEWTON_SOLVER_PARAMETERS["newton_solver"].update(
    linear_solver="petsc",
    absolute_tolerance=1e-4,
    relative_tolerance=1e-5,
    maximum_iterations=80,
    relaxation_parameter=1.0,
)


# ------------------------------------------------------------
# Mesh/thermal controls
# ------------------------------------------------------------

def heat_controls_from_size(size, *, ref=False, fixed=True):
    """
    Heat controls for convergence studies.

    fixed=True keeps the 3D heat discretization fixed so that the reported
    refinement behavior reflects the plate/FOM discretization rather than
    simultaneous heat-mesh refinement.
    """
    if ref:
        return dict(REF_HEAT_CONTROLS)

    if fixed:
        return dict(FIXED_HEAT_CONTROLS)

    return dict(
        heat_nx=int(size),
        heat_ny=max(2, int(round(size / 2))),
        heat_nz=max(8, int(round(size / 4))),
        heat_degree=int(HEAT_DEGREE),
        Nz_quad_T1=max(10, min(24, int(round(size / 4)) + 4)),
    )


# ------------------------------------------------------------
# Solver construction and single-case solve
# ------------------------------------------------------------

def make_solver(size, p, mu_mech, *, load_type="uniform", bc_type=MECHANICAL_BC_TYPE):
    solver = GeneralMultiphysicsSolver(study_case=1)
    solver.size = int(size)
    solver.degree = int(p)
    solver.load_type = load_type
    solver.bc_type = bc_type
    solver.define_domain(n_vert=0, n_horiz=0, plot_subdomains=False)
    solver.set_mu(mu_mech)
    solver.set_rom_thermal_parameters(**THERMAL_PARAMS)
    return solver


def copy_function(f):
    out = Function(f.function_space())
    out.assign(f)
    return out


def solve_paper2_case(
    *,
    mode,
    size,
    p,
    mu_mech,
    heat_nx,
    heat_ny,
    heat_nz,
    heat_degree,
    Nz_quad_T1,
    T1_cg_degree,
    bc_type=MECHANICAL_BC_TYPE,
):
    """
    Solve one Paper 2 FOM case.

    The one-way solver path is kept available because the framework still supports
    one-way thermomechanics. However, the revised refinement-study drivers below
    intentionally use mode='coupled' only.
    """
    solver = make_solver(size=size, p=p, mu_mech=mu_mech, bc_type=bc_type)

    heat_kwargs = dict(
        thermal_on=True,
        heat_nx=int(heat_nx),
        heat_ny=int(heat_ny),
        heat_nz=int(heat_nz),
        heat_degree=int(heat_degree),
        Nz_quad_T1=int(Nz_quad_T1),
        T1_cg_degree=int(T1_cg_degree),
        return_mode="global",
    )
    common = dict(solver=solver, mode=mode, bc_type=bc_type)

    if mode == "one_way":
        heat = solver.solve_rom_sample(mu_mech, coupled_on=False, **heat_kwargs)
        w = copy_function(solver.w_contact_global)
        theta = copy_function(solver.T1_from_heat)
        return dict(
            common,
            w=w,
            theta=theta,
            heat=heat,
            history=None,
            converged=True,
            stopped_by_max_iters=False,
            termination_reason="one_way",
            err_w=np.nan,
            err_T1=np.nan,
            iters=1,
            w_dofs=int(w.function_space().dim()),
            heat_dofs=int(heat["Vt"].dim()),
            theta_dofs=int(theta.function_space().dim()),
            total_dofs=int(w.function_space().dim() + heat["Vt"].dim() + theta.function_space().dim()),
            heat_nx=int(heat_nx),
            heat_ny=int(heat_ny),
            heat_nz=int(heat_nz),
            heat_degree=int(heat_degree),
            Nz_quad_T1=int(Nz_quad_T1),
            T1_cg_degree=int(T1_cg_degree),
        )

    if mode == "coupled":
        out = solver.solve_rom_sample(
            mu_mech,
            coupled_on=True,
            coupling_omega=COUPLING_PARAMS["coupling_omega"],
            coupling_tol_w=COUPLING_PARAMS["coupling_tol_w"],
            coupling_tol_T1=COUPLING_PARAMS["coupling_tol_T1"],
            coupling_max_iters=COUPLING_PARAMS["coupling_max_iters"],
            coupling_verbose=False,
            w0=0.0,
            return_coupled_dict=True,
            **heat_kwargs,
        )
        return dict(
            common,
            w=copy_function(out["w"]),
            theta=copy_function(out["T1"]),
            heat=out["heat"],
            history=out["history"],
            converged=bool(out["converged"]),
            stopped_by_max_iters=bool(out["stopped_by_max_iters"]),
            termination_reason=str(out["termination_reason"]),
            err_w=float(out["err_w"]),
            err_T1=float(out["err_T1"]),
            iters=int(out["iters"]),
            w_dofs=int(out["w_dofs"]),
            heat_dofs=int(out["heat_dofs"]),
            theta_dofs=int(out["T1_dofs"]),
            total_dofs=int(out["total_coupled_dofs"]),
            heat_nx=int(heat_nx),
            heat_ny=int(heat_ny),
            heat_nz=int(heat_nz),
            heat_degree=int(heat_degree),
            Nz_quad_T1=int(Nz_quad_T1),
            T1_cg_degree=int(T1_cg_degree),
        )

    raise ValueError("mode must be 'one_way' or 'coupled'.")


# ------------------------------------------------------------
# Error computation against refined reference
# ------------------------------------------------------------

def lift_field_to_reference_mesh(field, Vref, degree=5):
    """Interpolate a scalar plate field from its own mesh to the reference mesh."""
    try:
        field.set_allow_extrapolation(True)
    except Exception:
        pass

    class FieldOnReferenceMesh(UserExpression):
        def __init__(self, f, **kwargs):
            super().__init__(**kwargs)
            self.f = f

        def eval(self, values, x):
            values[0] = float(self.f(Point(float(x[0]), float(x[1]))))

        def value_shape(self):
            return ()

    return interpolate(FieldOnReferenceMesh(field, degree=degree), Vref)


def l2_rms_error_against_ref(field, ref_field, Vref, sqrt_area, degree=5):
    """
    RMS L2 error against the refined numerical FOM reference.

    This is not an exact-solution error.
    """
    field_refmesh = lift_field_to_reference_mesh(field, Vref, degree=degree)
    ref_projected = project(ref_field, Vref)
    return float(errornorm(ref_projected, field_refmesh, norm_type="L2", degree_rise=3) / sqrt_area)


def l2_relative_error_against_ref(field, ref_field, Vref, degree=5, eps=1e-14):
    """Relative L2 error against the refined numerical FOM reference."""
    dx_ref = Measure("dx", domain=Vref.mesh())
    field_refmesh = lift_field_to_reference_mesh(field, Vref, degree=degree)
    ref_projected = project(ref_field, Vref)

    abs_err = float(errornorm(ref_projected, field_refmesh, norm_type="L2", degree_rise=3))
    ref_norm = float(np.sqrt(max(float(assemble(ref_projected * ref_projected * dx_ref)), 0.0)))
    return abs_err / max(ref_norm, float(eps))


# ------------------------------------------------------------
# Checkpoint helpers
# ------------------------------------------------------------

def convergence_csv_path(mode, label, bc_type):
    return os.path.join(
        PAPER2_OUTDIR,
        f"paper2_{mode}_{label}_{bc_type}_convergence_{STUDY_TAG}.csv",
    )


def _as_bool(value, default=False):
    """Robust boolean conversion for bools, numpy bools, and CSV string values."""
    if value is None:
        return bool(default)

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, np.integer)):
        return bool(value)

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return bool(default)
        return bool(int(value))

    if isinstance(value, str):
        value_l = value.strip().lower()
        if value_l in ("true", "t", "1", "yes", "y"):
            return True
        if value_l in ("false", "f", "0", "no", "n", "nan", "none", ""):
            return False

    return bool(default)


def _bool_series(series, default=False):
    """Robust boolean mask for pandas Series that may contain strings from CSV."""
    return series.apply(lambda v: _as_bool(v, default=default)).astype(bool)


def _coupled_solution_is_converged(record):
    """True only if a coupled record/solution reached the coupled stopping criterion."""
    return (
        str(record.get("mode", "coupled")) == "coupled"
        and _as_bool(record.get("converged", record.get("coupled_converged", False)))
        and not _as_bool(record.get("stopped_by_max_iters", False))
    )


def _row_is_valid_converged_coupled(row):
    """True only for coupled rows that are valid for manuscript tables/plots."""
    try:
        if str(row.get("mode", "")) != "coupled":
            return False
        if not _as_bool(row.get("coupled_converged", False)):
            return False
        if _as_bool(row.get("stopped_by_max_iters", False)):
            return False
        if "reported" in row.index and not _as_bool(row.get("reported", True), default=True):
            return False
        if "stopping_norm" in row.index and str(row.get("stopping_norm", "")) != "L2_Omega":
            return False

        finite_cols = [
            "h",
            "L2_RMS_Error_w_m",
            "Rel_L2_Error_w",
            "L2_RMS_Error_theta",
            "Rel_L2_Error_theta",
            "err_w",
            "err_theta",
        ]
        return all(col not in row or np.isfinite(float(row[col])) for col in finite_cols)

    except Exception:
        return False


def load_existing_convergence_rows(out_csv):
    """
    Load old checkpoint rows.

    Only valid converged coupled rows are treated as completed. Failed or
    non-converged rows are kept in the CSV for traceability, but they are not
    allowed to block reruns.
    """
    if not (RESUME_RUNS and os.path.exists(out_csv) and not FORCE_RERUN):
        return None, set()

    df = pd.read_csv(out_csv)
    if df.empty:
        return None, set()

    required_key_cols = ["mode", "case", "bc_type", "p", "size"]
    valid_keys = {
        (str(row["mode"]), str(row["case"]), str(row["bc_type"]), int(row["p"]), int(row["size"]))
        for _, row in df.iterrows()
        if all(col in row.index for col in required_key_cols) and _row_is_valid_converged_coupled(row)
    }

    print(f"\nResuming from existing table: {out_csv}")
    print(f"Existing rows: {len(df)}")
    print(f"Valid converged coupled rows that will be skipped: {len(valid_keys)}")

    return df, valid_keys


def save_convergence_checkpoint(new_records, existing_df, out_csv):
    if not new_records:
        return existing_df if existing_df is not None else pd.DataFrame()

    new_df = pd.DataFrame.from_records(new_records)
    save_df = pd.concat([existing_df, new_df], ignore_index=True) if existing_df is not None and not existing_df.empty else new_df

    save_df = (
        save_df.drop_duplicates(["mode", "case", "bc_type", "p", "size"], keep="last")
        .sort_values(["mode", "case", "bc_type", "p", "h"], ascending=[True, True, True, True, False])
        .reset_index(drop=True)
    )
    save_df.to_csv(out_csv, index=False)
    print(f"Checkpoint saved: {out_csv}")

    return save_df


# ------------------------------------------------------------
# Reference-cache helpers
# ------------------------------------------------------------

def _json_safe(obj):
    """Convert numpy/scalar objects into JSON-safe Python objects."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _safe_filename_part(text):
    text = str(text)
    for ch in [" ", "/", "\\", ":", ";", ",", "(", ")", "[", "]", "{", "}"]:
        text = text.replace(ch, "_")
    return text


def reference_cache_key(*, mode, label, bc_type, ref_size, ref_p, mu_mech, ref_heat, fixed_heat):
    """Build a unique reference-cache key."""
    payload = dict(
        cache_version=int(REFERENCE_CACHE_VERSION),
        convergence_table_version=int(CONVERGENCE_TABLE_VERSION),
        study_tag=str(STUDY_TAG),
        mode=str(mode),
        case=str(label),
        bc_type=str(bc_type),
        ref_size=int(ref_size),
        ref_p=int(ref_p),
        mu_mech=[float(v) for v in mu_mech],
        thermal_params=_json_safe(THERMAL_PARAMS),
        ref_heat=_json_safe(ref_heat),
        fixed_heat=bool(fixed_heat),
        coupling_params=_json_safe(COUPLING_PARAMS) if mode == "coupled" else None,
        mechanical_bc_type=str(MECHANICAL_BC_TYPE),
        stopping_norm="L2_Omega",
        reported_refinement_branch="coupled_only",
    )
    payload_json = json.dumps(payload, sort_keys=True, indent=2)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:16], payload


def reference_cache_paths(*, mode, label, bc_type, ref_size, ref_p, cache_key):
    stem = (
        f"paper2_ref_{_safe_filename_part(mode)}_{_safe_filename_part(label)}"
        f"_{_safe_filename_part(bc_type)}_size{int(ref_size)}_p{int(ref_p)}_{cache_key}"
    )
    return dict(
        h5=os.path.join(PAPER2_OUTDIR, stem + ".h5"),
        json=os.path.join(PAPER2_OUTDIR, stem + ".json"),
    )


def _reference_meta(ref, w_save, theta_save, payload):
    return dict(
        payload=payload,
        history=ref.get("history", None),
        converged=bool(ref.get("converged", True)),
        stopped_by_max_iters=bool(ref.get("stopped_by_max_iters", False)),
        termination_reason=str(ref.get("termination_reason", "cached_reference")),
        err_w=float(ref.get("err_w", np.nan)),
        err_T1=float(ref.get("err_T1", np.nan)),
        iters=int(ref.get("iters", 1)),
        w_dofs=int(ref.get("w_dofs", w_save.function_space().dim())),
        heat_dofs=int(ref.get("heat_dofs", 0)),
        theta_dofs=int(ref.get("theta_dofs", theta_save.function_space().dim())),
        total_dofs=int(ref.get("total_dofs", 0)),
    )


def _assert_converged_coupled_record(record, *, context):
    """Stop immediately if a coupled reference/run did not converge."""
    if not bool(record.get("converged", False)):
        raise RuntimeError(
            f"{context} did not converge: "
            f"iters={record.get('iters', None)}, "
            f"err_w={record.get('err_w', np.nan)}, "
            f"err_T1={record.get('err_T1', np.nan)}, "
            f"termination={record.get('termination_reason', None)}"
        )

    if bool(record.get("stopped_by_max_iters", False)):
        raise RuntimeError(
            f"{context} reached max coupling iterations and must not be reported: "
            f"iters={record.get('iters', None)}, "
            f"err_w={record.get('err_w', np.nan)}, "
            f"err_T1={record.get('err_T1', np.nan)}, "
            f"termination={record.get('termination_reason', None)}"
        )


def save_reference_cache(ref, *, paths, payload, ref_p):
    """Save reference mesh, displacement, thermal driver, and metadata."""
    if payload.get("mode") == "coupled":
        _assert_converged_coupled_record(ref, context="Coupled reference before cache save")

    mesh = ref["solver"].mesh
    Vw_ref = FunctionSpace(mesh, "CG", int(ref_p))
    Vtheta_ref = FunctionSpace(mesh, "CG", int(ref_p))

    w_save = project(ref["w"], Vw_ref)
    theta_save = project(ref["theta"], Vtheta_ref)

    h5 = HDF5File(mesh.mpi_comm(), paths["h5"], "w")
    for name, obj in [("/mesh", mesh), ("/w", w_save), ("/theta", theta_save)]:
        h5.write(obj, name)
    h5.close()

    with open(paths["json"], "w") as f:
        json.dump(_json_safe(_reference_meta(ref, w_save, theta_save, payload)), f, indent=2, sort_keys=True)

    print("Saved reference cache:")
    print(f"  {paths['h5']}")
    print(f"  {paths['json']}")


def load_reference_cache(*, paths, ref_p):
    """Load cached reference mesh and fields."""
    if not (os.path.exists(paths["h5"]) and os.path.exists(paths["json"])):
        return None

    try:
        mesh = Mesh()
        h5 = HDF5File(MPI.comm_world, paths["h5"], "r")
        h5.read(mesh, "/mesh", False)

        Vw_ref = FunctionSpace(mesh, "CG", int(ref_p))
        Vtheta_ref = FunctionSpace(mesh, "CG", int(ref_p))
        w_ref = Function(Vw_ref)
        theta_ref = Function(Vtheta_ref)

        h5.read(w_ref, "/w")
        h5.read(theta_ref, "/theta")
        h5.close()

        with open(paths["json"], "r") as f:
            meta = json.load(f)

    except Exception as exc:
        print(f"Reference cache could not be loaded and will be rebuilt: {exc}")
        return None

    ref = dict(
        solver=None,
        mesh=mesh,
        w=w_ref,
        theta=theta_ref,
        heat=None,
        history=meta.get("history", None),
        converged=bool(meta.get("converged", True)),
        stopped_by_max_iters=bool(meta.get("stopped_by_max_iters", False)),
        termination_reason=str(meta.get("termination_reason", "loaded_reference_cache")),
        err_w=float(meta.get("err_w", np.nan)),
        err_T1=float(meta.get("err_T1", np.nan)),
        iters=int(meta.get("iters", 1)),
        w_dofs=int(meta.get("w_dofs", w_ref.function_space().dim())),
        heat_dofs=int(meta.get("heat_dofs", 0)),
        theta_dofs=int(meta.get("theta_dofs", theta_ref.function_space().dim())),
        total_dofs=int(meta.get("total_dofs", 0)),
    )

    if meta.get("payload", {}).get("mode") == "coupled":
        _assert_converged_coupled_record(ref, context="Loaded coupled reference cache")

    print("Loaded cached reference:")
    print(f"  {paths['h5']}")

    return ref


def get_or_build_reference(*, mode, label, bc_type, ref_size, ref_p, mu_mech, fixed_heat=True):
    """
    Build or load the refined FOM numerical reference.

    The revised reported refinement study uses coupled references only.
    """
    if mode != "coupled":
        raise ValueError(
            "The revised Paper 2 refinement study is coupled-only. "
            "Use mode='coupled' for reference construction."
        )

    ref_heat = heat_controls_from_size(ref_size, ref=True, fixed=fixed_heat)
    cache_key, payload = reference_cache_key(
        mode=mode,
        label=label,
        bc_type=bc_type,
        ref_size=ref_size,
        ref_p=ref_p,
        mu_mech=mu_mech,
        ref_heat=ref_heat,
        fixed_heat=fixed_heat,
    )
    paths = reference_cache_paths(
        mode=mode,
        label=label,
        bc_type=bc_type,
        ref_size=ref_size,
        ref_p=ref_p,
        cache_key=cache_key,
    )

    if RESUME_RUNS and not FORCE_RERUN:
        cached = load_reference_cache(paths=paths, ref_p=ref_p)
        if cached is not None:
            return cached

    print("\n" + "=" * 90)
    print(f"BUILDING COUPLED REFERENCE: {label}, bc={bc_type}, N_Omega={ref_size}, p={ref_p}")
    print("=" * 90)

    ref = solve_paper2_case(
        mode=mode,
        size=ref_size,
        p=ref_p,
        mu_mech=mu_mech,
        heat_nx=ref_heat["heat_nx"],
        heat_ny=ref_heat["heat_ny"],
        heat_nz=ref_heat["heat_nz"],
        heat_degree=ref_heat["heat_degree"],
        Nz_quad_T1=ref_heat["Nz_quad_T1"],
        T1_cg_degree=ref_p,
        bc_type=bc_type,
    )

    _assert_converged_coupled_record(ref, context="Newly built coupled reference")
    save_reference_cache(ref, paths=paths, payload=payload, ref_p=ref_p)
    return ref


def reference_mesh(ref):
    """Return the mesh from either a live reference solve or a cached reference."""
    return ref["solver"].mesh if ref.get("solver", None) is not None else ref["mesh"]


# ------------------------------------------------------------
# Convergence gathering
# ------------------------------------------------------------

def _success_record(mode, label, bc_type, p, res, sol, errors):
    ew_rms, etheta_rms, ew_rel, etheta_rel = errors
    return dict(
        mode=str(mode),
        case=str(label),
        bc_type=str(bc_type),
        p=int(p),
        size=int(res),
        h=float(sol["solver"].mesh.hmax()),
        inv_h=float(1.0 / sol["solver"].mesh.hmax()),
        plate_DoFs=int(sol["w_dofs"]),
        heat_DoFs=int(sol["heat_dofs"]),
        theta_DoFs=int(sol["theta_dofs"]),
        Total_DoFs=int(sol["total_dofs"]),
        sqrt_plate_DoFs=float(np.sqrt(sol["w_dofs"])),
        sqrt_Total_DoFs=float(np.sqrt(sol["total_dofs"])),
        heat_nx=int(sol["heat_nx"]),
        heat_ny=int(sol["heat_ny"]),
        heat_nz=int(sol["heat_nz"]),
        heat_degree=int(sol["heat_degree"]),
        Nz_quad_T1=int(sol["Nz_quad_T1"]),
        T1_cg_degree=int(sol["T1_cg_degree"]),
        L2_RMS_Error_w_m=float(ew_rms),
        L2_RMS_Error_theta=float(etheta_rms),
        Rel_L2_Error_w=float(ew_rel),
        Rel_L2_Error_theta=float(etheta_rel),
        coupled_iters=int(sol["iters"]),
        coupled_converged=bool(sol["converged"]),
        stopped_by_max_iters=bool(sol["stopped_by_max_iters"]),
        termination_reason=str(sol["termination_reason"]),
        err_w=float(sol["err_w"]),
        err_theta=float(sol["err_T1"]),
        stopping_norm="L2_Omega",
        reported=True,
    )


def _failed_record(mode, label, bc_type, p, res, hc, exc):
    return dict(
        mode=str(mode),
        case=str(label),
        bc_type=str(bc_type),
        p=int(p),
        size=int(res),
        h=np.nan,
        inv_h=np.nan,
        plate_DoFs=np.nan,
        heat_DoFs=np.nan,
        theta_DoFs=np.nan,
        Total_DoFs=np.nan,
        sqrt_plate_DoFs=np.nan,
        sqrt_Total_DoFs=np.nan,
        heat_nx=int(hc["heat_nx"]),
        heat_ny=int(hc["heat_ny"]),
        heat_nz=int(hc["heat_nz"]),
        heat_degree=int(hc["heat_degree"]),
        Nz_quad_T1=int(hc["Nz_quad_T1"]),
        T1_cg_degree=int(p),
        L2_RMS_Error_w_m=np.nan,
        L2_RMS_Error_theta=np.nan,
        Rel_L2_Error_w=np.nan,
        Rel_L2_Error_theta=np.nan,
        coupled_iters=np.nan,
        coupled_converged=False,
        stopped_by_max_iters=False,
        termination_reason=f"failed_or_not_reported: {exc}",
        err_w=np.nan,
        err_theta=np.nan,
        stopping_norm="L2_Omega",
        reported=False,
    )


def _excluded_record_from_solution(mode, label, bc_type, p, res, sol):
    """Save a non-converged coupled run for traceability, but mark it as not reportable."""
    return dict(
        mode=str(mode),
        case=str(label),
        bc_type=str(bc_type),
        p=int(p),
        size=int(res),
        h=float(sol["solver"].mesh.hmax()),
        inv_h=float(1.0 / sol["solver"].mesh.hmax()),
        plate_DoFs=int(sol["w_dofs"]),
        heat_DoFs=int(sol["heat_dofs"]),
        theta_DoFs=int(sol["theta_dofs"]),
        Total_DoFs=int(sol["total_dofs"]),
        sqrt_plate_DoFs=float(np.sqrt(sol["w_dofs"])),
        sqrt_Total_DoFs=float(np.sqrt(sol["total_dofs"])),
        heat_nx=int(sol["heat_nx"]),
        heat_ny=int(sol["heat_ny"]),
        heat_nz=int(sol["heat_nz"]),
        heat_degree=int(sol["heat_degree"]),
        Nz_quad_T1=int(sol["Nz_quad_T1"]),
        T1_cg_degree=int(sol["T1_cg_degree"]),
        L2_RMS_Error_w_m=np.nan,
        L2_RMS_Error_theta=np.nan,
        Rel_L2_Error_w=np.nan,
        Rel_L2_Error_theta=np.nan,
        coupled_iters=int(sol["iters"]),
        coupled_converged=bool(sol["converged"]),
        stopped_by_max_iters=bool(sol["stopped_by_max_iters"]),
        termination_reason=f"not_reported_nonconverged: {sol['termination_reason']}",
        err_w=float(sol["err_w"]),
        err_theta=float(sol["err_T1"]),
        stopping_norm="L2_Omega",
        reported=False,
    )


def valid_converged_coupled_rows(df):
    """
    Keep only rows admissible for manuscript tables and plots.

    Excludes:
        - one-way rows,
        - failed rows,
        - non-converged coupled rows,
        - rows stopped by max iterations,
        - rows with missing/non-finite error values.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out = out[out["mode"].astype(str) == "coupled"]

    if "coupled_converged" in out.columns:
        out = out[_bool_series(out["coupled_converged"], default=False)]
    if "stopped_by_max_iters" in out.columns:
        out = out[~_bool_series(out["stopped_by_max_iters"], default=False)]
    if "reported" in out.columns:
        out = out[_bool_series(out["reported"], default=True)]
    if "stopping_norm" in out.columns:
        out = out[out["stopping_norm"].astype(str) == "L2_Omega"]

    for col in [
        "h",
        "L2_RMS_Error_w_m",
        "Rel_L2_Error_w",
        "L2_RMS_Error_theta",
        "Rel_L2_Error_theta",
        "err_w",
        "err_theta",
    ]:
        if col in out.columns:
            out = out[np.isfinite(out[col].to_numpy(dtype=float))]

    return out.reset_index(drop=True)


def gather_paper2_convergence_data(
    *,
    mode,
    mu_mech,
    resolutions=RESOLUTIONS_COUPLED_FINAL,
    p_values=P_VALUES,
    ref_size=REF_SIZE,
    ref_p=REF_P,
    label="case",
    bc_type=MECHANICAL_BC_TYPE,
    fixed_heat=True,
):
    """
    Gather coupled-only convergence data using a cached refined FOM reference.

    Non-converged coupled runs are saved as failed/non-reported rows, but they
    are excluded from all manuscript tables and plots.
    """
    if mode != "coupled":
        raise ValueError(
            "The revised Paper 2 refinement study must be coupled-only. "
            "Do not call gather_paper2_convergence_data with mode='one_way'."
        )

    out_csv = convergence_csv_path(mode, label, bc_type)
    existing_df, existing_keys = load_existing_convergence_rows(out_csv)

    ref = get_or_build_reference(
        mode=mode,
        label=label,
        bc_type=bc_type,
        ref_size=ref_size,
        ref_p=ref_p,
        mu_mech=mu_mech,
        fixed_heat=fixed_heat,
    )
    _assert_converged_coupled_record(ref, context=f"Coupled reference for {label}, bc={bc_type}")

    ref_mesh_obj = reference_mesh(ref)
    Vw_ref = FunctionSpace(ref_mesh_obj, "CG", int(ref_p))
    Vtheta_ref = FunctionSpace(ref_mesh_obj, "CG", int(ref_p))
    dx_ref = Measure("dx", domain=ref_mesh_obj)
    sqrt_area = float(np.sqrt(assemble(Constant(1.0) * dx_ref)))

    w_ref = project(ref["w"], Vw_ref)
    theta_ref = project(ref["theta"], Vtheta_ref)

    for res in resolutions:
        for p in p_values:
            if int(res) == int(ref_size) and int(p) == int(ref_p):
                continue

            case_key = (mode, label, bc_type, int(p), int(res))
            if case_key in existing_keys:
                print(f"SKIPPING valid cached row: mode={mode}, case={label}, bc={bc_type}, size={res}, p={p}")
                continue

            hc = heat_controls_from_size(res, ref=False, fixed=fixed_heat)
            print("\n" + "-" * 90)
            print(
                f"COUPLED | {label} | bc={bc_type} | size={res}, p={p}, "
                f"heat=({hc['heat_nx']},{hc['heat_ny']},{hc['heat_nz']}), "
                f"pT={hc['heat_degree']}, Nz_quad_T1={hc['Nz_quad_T1']}"
            )
            print("-" * 90)

            try:
                sol = solve_paper2_case(
                    mode=mode,
                    size=res,
                    p=p,
                    mu_mech=mu_mech,
                    heat_nx=hc["heat_nx"],
                    heat_ny=hc["heat_ny"],
                    heat_nz=hc["heat_nz"],
                    heat_degree=hc["heat_degree"],
                    Nz_quad_T1=hc["Nz_quad_T1"],
                    T1_cg_degree=p,
                    bc_type=bc_type,
                )

                if not _coupled_solution_is_converged(sol):
                    print(
                        f"EXCLUDED NON-CONVERGED RUN: case={label}, size={res}, p={p}, "
                        f"iters={sol.get('iters', None)}, err_w={sol.get('err_w', np.nan)}, "
                        f"err_T1={sol.get('err_T1', np.nan)}, termination={sol.get('termination_reason', None)}"
                    )
                    record = _excluded_record_from_solution(mode, label, bc_type, p, res, sol)

                else:
                    deg = max(5, int(p) + 2)
                    errors = (
                        l2_rms_error_against_ref(sol["w"], w_ref, Vw_ref, sqrt_area, degree=deg),
                        l2_rms_error_against_ref(sol["theta"], theta_ref, Vtheta_ref, sqrt_area, degree=deg),
                        l2_relative_error_against_ref(sol["w"], w_ref, Vw_ref, degree=deg),
                        l2_relative_error_against_ref(sol["theta"], theta_ref, Vtheta_ref, degree=deg),
                    )
                    record = _success_record(mode, label, bc_type, p, res, sol, errors)

            except Exception as exc:
                print(f"FAILED OR EXCLUDED: mode={mode}, case={label}, bc={bc_type}, size={res}, p={p}: {exc}")
                record = _failed_record(mode, label, bc_type, p, res, hc, exc)

            existing_df = save_convergence_checkpoint([record], existing_df, out_csv)

            if _row_is_valid_converged_coupled(pd.Series(record)):
                existing_keys.add(case_key)

    df = existing_df.copy() if existing_df is not None and not existing_df.empty else pd.DataFrame()

    if not df.empty:
        df = df.sort_values(
            ["mode", "case", "bc_type", "p", "h"],
            ascending=[True, True, True, True, False],
        ).reset_index(drop=True)

    df.to_csv(out_csv, index=False)
    df_valid = valid_converged_coupled_rows(df)

    print(f"\nSaved convergence table: {out_csv}")
    print(f"Total rows in CSV: {len(df)}")
    print(f"Valid converged coupled rows for reporting: {len(df_valid)}")

    return df, ref


# ------------------------------------------------------------
# Table helper
# ------------------------------------------------------------

def finest_converged_rows_for_table(df):
    """Return the finest valid coupled row for each case and p."""
    df_valid = valid_converged_coupled_rows(df)
    if df_valid.empty:
        return df_valid

    rows = [grp.sort_values("size", ascending=False).iloc[0] for _, grp in df_valid.groupby(["case", "p"])]
    return pd.DataFrame(rows).sort_values(["case", "p"]).reset_index(drop=True)


def print_finest_converged_table(df, title="FINEST CONVERGED COUPLED ROWS"):
    table_df = finest_converged_rows_for_table(df)

    print("\n" + title)
    if table_df.empty:
        print("No valid converged coupled rows available.")
        return table_df

    print(table_df[TABLE_COLS].to_string(index=False, float_format="%.6e"))
    return table_df


# ------------------------------------------------------------
# Coupled-only convergence plotting: revised Fig. 3
# ------------------------------------------------------------

def plot_paper2_coupled_convergence_iso_ortho(
    df_coupled,
    *,
    ycol="L2_RMS_Error_w_m",
    ylabel=r"$L^2$ RMS error in $w_h$ [m]",
    x_mode="h",
    filename_base=None,
    rate_fit_slice=slice(None),
):
    """
    Revised Paper 2 Fig. 3:
    coupled-only displacement convergence for isotropic/orthotropic cases.

    Only valid converged coupled rows are plotted.

    Note:
    For x_mode='h', the x-axis is intentionally shown from coarse to fine,
    i.e. larger h on the left and smaller h on the right.

    Rate triangles:
    - rates are computed from a log-log least-squares fit using rate_fit_slice;
      the default slice(None) gives a global fitted rate over all refinement data.
    - all triangles are kept near the finest segment;
    - triangles are compact to avoid overlap;
    - for each polynomial degree p, the higher-error curve gets the upper triangle
      and the lower-error curve gets the lower triangle;
    - the label "1" is always placed above the horizontal triangle edge.
    """

    if filename_base is None:
        filename_base = f"paper2_isotropic_orthotropic_simply_supported_coupled_w_convergence_mesh_size_{STUDY_TAG}"

    df = valid_converged_coupled_rows(df_coupled)
    if df.empty:
        raise ValueError("No valid converged coupled rows available for Fig. 3.")

    plt.style.use(["science", "ieee", "notebook", "grid"])
    plt.rcParams["figure.autolayout"] = False

    x_options = {
        "h": ("h", r"Characteristic mesh length $\ell_{\mathcal{T}}$ [m]"),
        "inv_h": ("inv_h", r"Inverse mesh size $1/h$"),
        "sqrt_plate_dofs": ("sqrt_plate_DoFs", r"$\sqrt{\mathrm{plate\ DoFs}}$"),
        "plate_dofs": ("plate_DoFs", "Plate degrees of freedom"),
        "sqrt_total_dofs": ("sqrt_Total_DoFs", r"$\sqrt{\mathrm{total\ DoFs}}$"),
        "total_dofs": ("Total_DoFs", "Total degrees of freedom"),
    }
    if x_mode not in x_options:
        raise ValueError(f"x_mode must be one of {list(x_options.keys())}")

    xcol, xlabel = x_options[x_mode]

    style_specs = [
        ("Isotropic", 2, "black", "-", "o", r"Isotropic, $p=2$"),
        ("Isotropic", 3, "blue", "--", "s", r"Isotropic, $p=3$"),
        ("Orthotropic", 2, "darkorange", "-", "o", r"Orthotropic, $p=2$"),
        ("Orthotropic", 3, "purple", "--", "s", r"Orthotropic, $p=3$"),
    ]
    styles = {
        (case_name, p): dict(color=color, linestyle=linestyle, marker=marker, label=label)
        for case_name, p, color, linestyle, marker, label in style_specs
    }

    def positive_finite(vals):
        vals = np.asarray(vals, dtype=float)
        return vals[np.isfinite(vals) & (vals > 0.0)]

    def ordered_xy(x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
        x, y = x[mask], y[mask]
        order = np.argsort(x)[::-1] if x_mode == "h" else np.argsort(x)
        return x[order], y[order]

    def estimate_loglog_slope(x, y, fit_slice=slice(None)):
        x, y = ordered_xy(x, y)
        if len(x) < 2:
            return None

        xx, yy = x[fit_slice], y[fit_slice]
        if len(xx) < 2:
            return None

        slope, _ = np.polyfit(np.log10(xx), np.log10(yy), 1)
        return float(abs(slope))

    def fine_error_value(grp):
        x, y = ordered_xy(grp[xcol].to_numpy(float), grp[ycol].to_numpy(float))
        return np.nan if len(y) == 0 else float(y[-1])

    def build_triangle_specs(grouped):
        """
        Automatic compact triangle placement.

        For each polynomial degree:
            higher-error curve -> triangle above,
            lower-error curve  -> triangle below.

        All triangles stay near the finest segment. The p=3 triangles are made
        slightly smaller than p=2 because the p=3 curves are closer together.
        """
        specs = {}

        for p in sorted({key[1] for key in grouped}):
            keys_p = [key for key in grouped if key[1] == p]
            if not keys_p:
                continue

            keys_p = sorted(keys_p, key=lambda key: fine_error_value(grouped[key]))

            if len(keys_p) == 1:
                specs[keys_p[0]] = dict(
                    location="above",
                    start_idx=-2,
                    span_frac=0.34,
                    tri_gap=1.22,
                    rate_side_pad=0.020,
                    one_label_gap=1.055,
                    rate_label_shift=1.00,
                )
                continue

            lower_key = keys_p[0]
            upper_key = keys_p[-1]

            if int(p) == 2:
                specs[upper_key] = dict(
                    location="above",
                    start_idx=-2,
                    span_frac=0.36,
                    tri_gap=1.24,
                    rate_side_pad=0.020,
                    one_label_gap=1.055,
                    rate_label_shift=1.00,
                )
                specs[lower_key] = dict(
                    location="below",
                    start_idx=-2,
                    span_frac=0.36,
                    tri_gap=1.28,
                    rate_side_pad=0.020,
                    one_label_gap=1.055,
                    rate_label_shift=1.00,
                )
            else:
                specs[upper_key] = dict(
                    location="above",
                    start_idx=-2,
                    span_frac=0.28,
                    tri_gap=1.20,
                    rate_side_pad=0.018,
                    one_label_gap=1.060,
                    rate_label_shift=1.00,
                )
                specs[lower_key] = dict(
                    location="below",
                    start_idx=-2,
                    span_frac=0.28,
                    tri_gap=1.26,
                    rate_side_pad=0.018,
                    one_label_gap=1.060,
                    rate_label_shift=1.00,
                )

        return specs

    def add_rate_triangle(
        ax,
        x,
        y,
        rate,
        color,
        *,
        start_idx=-2,
        span_frac=0.34,
        tri_gap=1.22,
        rate_side_pad=0.020,
        one_label_gap=1.055,
        rate_label_shift=1.00,
        lw=1.00,
        fontsize=8,
        location="above",
    ):
        if rate is None or rate <= 0.0:
            return

        x, y = ordered_xy(x, y)
        if len(x) < 2:
            return

        start_idx = len(x) + start_idx if start_idx < 0 else start_idx
        if start_idx < 0 or start_idx + 1 >= len(x):
            return

        x_prev = float(x[start_idx])
        x_last = float(x[start_idx + 1])
        y_last = float(y[start_idx + 1])

        if x_prev <= 0.0 or x_last <= 0.0 or y_last <= 0.0:
            return

        if x_mode == "h":
            full_ratio = x_prev / x_last
            if full_ratio <= 1.0:
                return

            x_right = x_last
            x_left = x_last * (full_ratio ** span_frac)
            local_ratio = x_left / x_right
            x_rate = x_right / (local_ratio ** rate_side_pad)

        else:
            full_ratio = x_last / x_prev
            if full_ratio <= 1.0:
                return

            x_left = x_last / (full_ratio ** span_frac)
            x_right = x_last
            local_ratio = x_right / x_left
            x_rate = x_right * (local_ratio ** rate_side_pad)

        if location == "below":
            # Triangle is below the curve.
            # y_top is still the horizontal edge of the triangle.
            y_top = y_last / tri_gap
            y_bottom = y_top / (local_ratio ** rate)
        else:
            # Triangle is above the curve.
            # y_top is the horizontal edge of the triangle.
            y_bottom = y_last * tri_gap
            y_top = y_bottom * (local_ratio ** rate)

        if min(x_left, x_right, y_top, y_bottom) <= 0.0:
            return

        # Important fix:
        # Put the "1" label above the horizontal edge for both upper and lower triangles.
        one_label_y = y_top * one_label_gap
        one_label_va = "bottom"

        y_mid = np.sqrt(y_top * y_bottom) * float(rate_label_shift)

        ymin, ymax = ax.get_ylim()
        log_ymin, log_ymax = np.log10(ymin), np.log10(ymax)
        low = np.log10(min(y_top, y_bottom))
        high = np.log10(max(y_top, y_bottom, one_label_y))

        if low < log_ymin + 0.08:
            scale = 10 ** (log_ymin + 0.08) / min(y_top, y_bottom)
            y_top *= scale
            y_bottom *= scale
            one_label_y *= scale
            y_mid *= scale

        if high > log_ymax - 0.08:
            scale = 10 ** (log_ymax - 0.08) / max(y_top, y_bottom, one_label_y)
            y_top *= scale
            y_bottom *= scale
            one_label_y *= scale
            y_mid *= scale

        text_box = dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.12)

        ax.plot([x_left, x_right], [y_top, y_top], color=color, lw=lw, zorder=5)
        ax.plot([x_right, x_right], [y_top, y_bottom], color=color, lw=lw, zorder=5)
        ax.plot([x_left, x_right], [y_top, y_bottom], color=color, lw=lw, zorder=5)

        ax.text(
            np.sqrt(x_left * x_right),
            one_label_y,
            "1",
            ha="center",
            va=one_label_va,
            fontsize=fontsize,
            color=color,
            bbox=text_box,
            clip_on=True,
            zorder=6,
        )

        ax.text(
            x_rate,
            y_mid,
            f"{rate:.1f}",
            ha="left",
            va="center",
            fontsize=fontsize,
            color=color,
            bbox=text_box,
            clip_on=True,
            zorder=6,
        )

    all_x = positive_finite(df[xcol].to_numpy(float))
    all_y = positive_finite(df[ycol].to_numpy(float))
    if len(all_x) == 0 or len(all_y) == 0:
        raise ValueError("No positive finite data available for Fig. 3.")

    fig, ax = plt.subplots(figsize=(6.8, 5.2), facecolor="white")
    grouped = {}

    for key, style in styles.items():
        case_name, p = key
        grp = df[
            (df["case"].astype(str) == case_name)
            & (df["p"].astype(int) == int(p))
        ].dropna(subset=[xcol, ycol]).copy()

        if grp.empty:
            continue

        grp = grp.sort_values(xcol, ascending=(x_mode != "h"))
        grouped[key] = grp

        ax.loglog(
            grp[xcol].to_numpy(float),
            grp[ycol].to_numpy(float),
            linestyle=style["linestyle"],
            marker=style["marker"],
            color=style["color"],
            markeredgecolor=style["color"],
            markerfacecolor="none",
            linewidth=2.2,
            markersize=8.0,
            markeredgewidth=1.4,
            label=style["label"],
            clip_on=False,
        )

    ax.set_facecolor("white")
    ax.set_title(r"Partitioned coupled refinement study", fontsize=14, pad=8)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=13, length=4, pad=2)
    ax.tick_params(axis="both", which="minor", length=2)
    ax.grid(True, which="both", linestyle=":", linewidth=0.45, alpha=0.35)

    if x_mode == "h":
        ticks = []
        for size in [8, 16, 32, 64]:
            vals = positive_finite(df.loc[df["size"].astype(int) == size, "h"].to_numpy(float))
            if len(vals):
                ticks.append(float(np.median(vals)))

        if ticks:
            ticks = sorted(set(np.round(ticks, 12)), reverse=True)
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, pos: f"{x:.2f}" if x >= 0.1 else f"{x:.3f}")
            )
    else:
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=80))
    ax.xaxis.set_minor_formatter(NullFormatter())

    # Important:
    # For h-refinement, show coarse-to-fine from left to right:
    # left  = largest h, approximately 0.25
    # right = smallest h, approximately 0.031
    if x_mode == "h":
        ax.set_xlim(np.max(all_x) * 1.20, np.min(all_x) / 1.14)
    else:
        ax.set_xlim(np.min(all_x) / 1.14, np.max(all_x) * 1.20)

    ax.set_ylim(np.min(all_y) / 2.60, np.max(all_y) * 1.65)

    triangle_specs = build_triangle_specs(grouped)

    for key, grp in grouped.items():
        style = styles[key]
        spec = triangle_specs.get(
            key,
            dict(
                location="above",
                start_idx=-2,
                span_frac=0.34,
                tri_gap=1.22,
                rate_side_pad=0.020,
                one_label_gap=1.055,
                rate_label_shift=1.00,
            ),
        )

        rate = estimate_loglog_slope(
            grp[xcol].to_numpy(float),
            grp[ycol].to_numpy(float),
            fit_slice=rate_fit_slice,
        )

        add_rate_triangle(
            ax,
            grp[xcol].to_numpy(float),
            grp[ycol].to_numpy(float),
            rate,
            style["color"],
            location=spec["location"],
            start_idx=spec["start_idx"],
            span_frac=spec["span_frac"],
            tri_gap=spec["tri_gap"],
            rate_side_pad=spec["rate_side_pad"],
            one_label_gap=spec["one_label_gap"],
            rate_label_shift=spec["rate_label_shift"],
            lw=1.00,
            fontsize=8,
        )

    leg = ax.legend(
        loc="lower left",
        fontsize=12,
        frameon=True,
        borderpad=0.55,
        handlelength=2.2,
        handletextpad=0.55,
        labelspacing=0.35,
    )
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.8)

    os.makedirs(PAPER2_OUTDIR, exist_ok=True)
    for ext, dpi in (("pdf", 300), ("png", 600)):
        fig.savefig(
            os.path.join(PAPER2_OUTDIR, f"{filename_base}.{ext}"),
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
        )

    plt.show()

# ------------------------------------------------------------
# Coupled update-history plotting: revised Fig. 4
# ------------------------------------------------------------

def history_dataframe_from_ref(ref_or_solution, case_label):
    """Convert a coupled reference/solution history into a DataFrame for Fig. 4."""
    hist = pd.DataFrame(ref_or_solution.get("history", None)).copy()
    if hist.empty:
        return hist

    hist["case"] = str(case_label)
    return hist


def load_saved_coupled_reference_history(case_label, *, bc_type=MECHANICAL_BC_TYPE):
    """
    Load coupled-reference fixed-point history from saved reference-cache JSON.
    No FOM solve is executed.
    """

    pattern = os.path.join(
        PAPER2_OUTDIR,
        (
            f"paper2_ref_coupled_{_safe_filename_part(case_label)}"
            f"_{_safe_filename_part(bc_type)}"
            f"_size{int(REF_SIZE)}_p{int(REF_P)}_*.json"
        ),
    )
    candidates = sorted(glob.glob(pattern))

    if not candidates:
        raise FileNotFoundError(f"No coupled reference-cache JSON found for {case_label}: {pattern}")

    for path in candidates:
        with open(path, "r") as f:
            meta = json.load(f)

        payload = meta.get("payload", {})
        is_match = (
            payload.get("mode") == "coupled"
            and payload.get("case") == case_label
            and payload.get("bc_type") == bc_type
            and int(payload.get("ref_size", -1)) == int(REF_SIZE)
            and int(payload.get("ref_p", -1)) == int(REF_P)
            and payload.get("ref_heat", {}) == _json_safe(REF_HEAT_CONTROLS)
            and payload.get("stopping_norm") == "L2_Omega"
            and payload.get("reported_refinement_branch") == "coupled_only"
        )
        hist = meta.get("history", None)

        if is_match and hist:
            df = pd.DataFrame(hist).copy()
            df["case"] = case_label
            df["source_file"] = os.path.basename(path)
            return df

    raise FileNotFoundError(f"No valid coupled history found for {case_label} in {len(candidates)} JSON file(s).")


def plot_paper2_coupled_history_iso_ortho_combined(
    histories,
    *,
    filename_base=None,
):
    """
    Revised Paper 2 Fig. 4:
    fixed-point relative-update histories only.

    The old response-evolution panel is intentionally removed.
    """
    if filename_base is None:
        filename_base = f"paper2_isotropic_orthotropic_simply_supported_coupled_reference_update_history_{STUDY_TAG}"

    if not histories:
        print("No coupled histories available.")
        return

    hist = pd.concat(histories, ignore_index=True)
    if hist.empty:
        print("No coupled histories available.")
        return

    plt.style.use(["science", "ieee", "notebook", "grid"])
    plt.rcParams["figure.autolayout"] = False

    hist_styles = {
        ("Isotropic", "err_w"): dict(color="black", linestyle="-", marker="o", label=r"Isotropic, $\varepsilon_w$"),
        ("Isotropic", "err_T1"): dict(color="red", linestyle="--", marker="s", label=r"Isotropic, $\varepsilon_{\theta}$"),
        ("Orthotropic", "err_w"): dict(color="blue", linestyle="-", marker="o", label=r"Orthotropic, $\varepsilon_w$"),
        ("Orthotropic", "err_T1"): dict(color="purple", linestyle="--", marker="s", label=r"Orthotropic, $\varepsilon_{\theta}$"),
    }

    fig, ax = plt.subplots(figsize=(6.8, 5.2), facecolor="white")

    for case_name in ["Isotropic", "Orthotropic"]:
        grp = hist[hist["case"].astype(str) == case_name].copy()
        if grp.empty:
            continue

        grp = grp.sort_values("iter")
        it = grp["iter"].to_numpy(int)

        for col in ["err_w", "err_T1"]:
            vals = grp[col].to_numpy(float)
            if col == "err_T1" and not np.isfinite(vals[1:]).any():
                continue

            style = hist_styles[(case_name, col)]
            ax.semilogy(
                it,
                vals,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2.2,
                markersize=7.0,
                markerfacecolor=style["color"],
                label=style["label"],
            )

    ax.set_facecolor("white")
    ax.set_title("Coupled fixed-point update history", fontsize=14, pad=8)
    ax.set_xlabel("Coupling iteration", fontsize=14)
    ax.set_ylabel(r"Relative $L^2(\Omega)$ update", fontsize=14)
    ax.tick_params(axis="both", labelsize=13, length=3, pad=2)
    ax.grid(True, which="both", linestyle=":", linewidth=0.45, alpha=0.35)

    leg = ax.legend(
        loc="upper right",
        fontsize=11.5,
        frameon=True,
        borderpad=0.55,
        handlelength=2.1,
        handletextpad=0.55,
        labelspacing=0.35,
    )
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.8)

    os.makedirs(PAPER2_OUTDIR, exist_ok=True)
    for ext, dpi in (("pdf", 300), ("png", 600)):
        fig.savefig(
            os.path.join(PAPER2_OUTDIR, f"{filename_base}.{ext}"),
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
        )

    plt.show()
    
# ------------------------------------------------------------
# Optional reference-level comparison
# ------------------------------------------------------------

def compare_two_reference_levels(
    mode,
    mu_mech,
    *,
    size_a=REFERENCE_COMPARISON_COARSE_SIZE,
    size_b=REF_SIZE,
    p=REF_P,
    label="reference_check",
    bc_type=MECHANICAL_BC_TYPE,
    fixed_heat=True,
    cache_label_a=None,
    cache_label_b=None,
):
    """
    Optional cached comparison between two coupled refined reference levels.

    This is not the main manuscript convergence table. It is a diagnostic to
    check the separation between the finest tested mesh and the reference.
    """
    if mode != "coupled":
        raise ValueError("Reference-level comparison for the revised study must be coupled-only.")

    cache_label_a = cache_label_a or f"{label}_size{int(size_a)}"
    cache_label_b = cache_label_b or f"{label}_size{int(size_b)}"

    sol_a = get_or_build_reference(
        mode=mode,
        label=cache_label_a,
        bc_type=bc_type,
        ref_size=size_a,
        ref_p=p,
        mu_mech=mu_mech,
        fixed_heat=fixed_heat,
    )
    sol_b = get_or_build_reference(
        mode=mode,
        label=cache_label_b,
        bc_type=bc_type,
        ref_size=size_b,
        ref_p=p,
        mu_mech=mu_mech,
        fixed_heat=fixed_heat,
    )

    _assert_converged_coupled_record(sol_a, context=f"Reference level A size={size_a}")
    _assert_converged_coupled_record(sol_b, context=f"Reference level B size={size_b}")

    mesh_b = reference_mesh(sol_b)
    Vw_b = FunctionSpace(mesh_b, "CG", int(p))
    Vtheta_b = FunctionSpace(mesh_b, "CG", int(p))
    sqrt_area_b = float(np.sqrt(assemble(Constant(1.0) * Measure("dx", domain=mesh_b))))
    deg = max(5, int(p) + 2)

    comparison = dict(
        mode=mode,
        label=label,
        bc_type=bc_type,
        size_a=int(size_a),
        size_b=int(size_b),
        p=int(p),
        L2_RMS_Error_w_m=l2_rms_error_against_ref(sol_a["w"], sol_b["w"], Vw_b, sqrt_area_b, degree=deg),
        L2_RMS_Error_theta=l2_rms_error_against_ref(sol_a["theta"], sol_b["theta"], Vtheta_b, sqrt_area_b, degree=deg),
        Rel_L2_Error_w=l2_relative_error_against_ref(sol_a["w"], sol_b["w"], Vw_b, degree=deg),
        Rel_L2_Error_theta=l2_relative_error_against_ref(sol_a["theta"], sol_b["theta"], Vtheta_b, degree=deg),
    )

    out_json = os.path.join(
        PAPER2_OUTDIR,
        f"paper2_{mode}_{_safe_filename_part(label)}_{bc_type}_{STUDY_TAG}_reference_comparison.json",
    )

    with open(out_json, "w") as f:
        json.dump(_json_safe(comparison), f, indent=2, sort_keys=True)

    print("\nREFERENCE-LEVEL COMPARISON")
    for key, value in comparison.items():
        print(f"{key}: {value}")
    print(f"Saved reference comparison: {out_json}")

    return sol_a, sol_b, comparison


# ------------------------------------------------------------
# Driver utilities: coupled-only
# ------------------------------------------------------------

def run_smoke_test(case_label="Isotropic", *, bc_type=MECHANICAL_BC_TYPE):
    """Coupled-only smoke test. This avoids accidentally reintroducing one-way refinement rows."""
    mu_mech = PLATE_CASES[case_label]["mu_mech"]

    df_cpl, ref_cpl = gather_paper2_convergence_data(
        mode="coupled",
        mu_mech=mu_mech,
        resolutions=RESOLUTIONS_SMOKE,
        p_values=[2],
        ref_size=32,
        ref_p=3,
        label=f"{case_label}_smoke",
        bc_type=bc_type,
        fixed_heat=True,
    )

    print(f"\nCOUPLED SMOKE {case_label.upper()} ({bc_type})")
    print(valid_converged_coupled_rows(df_cpl)[TABLE_COLS].to_string(index=False, float_format="%.6e"))

    return df_cpl, ref_cpl


def run_case(case_label, *, bc_type=MECHANICAL_BC_TYPE):
    """Run the revised Paper 2 refinement study for one material case. Coupled-only."""
    mu_mech = PLATE_CASES[case_label]["mu_mech"]

    df_cpl, ref_cpl = gather_paper2_convergence_data(
        mode="coupled",
        mu_mech=mu_mech,
        resolutions=RESOLUTIONS_COUPLED_FINAL,
        p_values=P_VALUES,
        ref_size=REF_SIZE,
        ref_p=REF_P,
        label=case_label,
        bc_type=bc_type,
        fixed_heat=True,
    )

    print(f"\nCOUPLED {case_label.upper()} ({bc_type})")
    df_valid = valid_converged_coupled_rows(df_cpl)

    if df_valid.empty:
        print("No valid converged coupled rows available.")
    else:
        print(df_valid[TABLE_COLS].to_string(index=False, float_format="%.6e"))

    return df_cpl, ref_cpl


def replot_coupled_from_saved_csv(*, bc_type=MECHANICAL_BC_TYPE):
    """Replot revised Fig. 3 from saved coupled CSVs only."""
    frames = []
    for case_label in ["Isotropic", "Orthotropic"]:
        path = convergence_csv_path("coupled", case_label, bc_type)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing coupled convergence CSV: {path}")
        frames.append(pd.read_csv(path))

    df_all = pd.concat(frames, ignore_index=True)
    plot_paper2_coupled_convergence_iso_ortho(
        df_all,
        ycol="L2_RMS_Error_w_m",
        ylabel=r"$L^2$ RMS error in $w_h$ [m]",
        x_mode="h",
        filename_base=f"paper2_isotropic_orthotropic_simply_supported_coupled_w_convergence_mesh_size_{STUDY_TAG}",
    )

    return df_all


def replot_coupled_history_from_saved_json(*, bc_type=MECHANICAL_BC_TYPE):
    """Replot revised Fig. 4 from saved coupled reference-cache JSON histories only."""
    histories = [
        load_saved_coupled_reference_history("Isotropic", bc_type=bc_type),
        load_saved_coupled_reference_history("Orthotropic", bc_type=bc_type),
    ]
    plot_paper2_coupled_history_iso_ortho_combined(
        histories,
        filename_base=f"paper2_isotropic_orthotropic_simply_supported_coupled_reference_update_history_{STUDY_TAG}",
    )

    return histories


# ------------------------------------------------------------
# Execute revised coupled-only study
# ------------------------------------------------------------

# Optional quick test:
# df_cpl_smoke, ref_cpl_smoke = run_smoke_test(
#     "Isotropic",
#     bc_type=MECHANICAL_BC_TYPE,
# )

df_cpl_iso, ref_cpl_iso = run_case("Isotropic", bc_type=MECHANICAL_BC_TYPE)
df_cpl_ortho, ref_cpl_ortho = run_case("Orthotropic", bc_type=MECHANICAL_BC_TYPE)

df_cpl_all = pd.concat([df_cpl_iso, df_cpl_ortho], ignore_index=True)
df_cpl_valid = valid_converged_coupled_rows(df_cpl_all)

print_finest_converged_table(
    df_cpl_valid,
    title="FINEST-MESH CONVERGED COUPLED ROWS FOR MANUSCRIPT TABLE",
)

plot_paper2_coupled_convergence_iso_ortho(
    df_cpl_valid,
    ycol="L2_RMS_Error_w_m",
    ylabel=r"$L^2$ RMS error in $w_h$ [m]",
    x_mode="h",
    filename_base=f"paper2_isotropic_orthotropic_simply_supported_coupled_w_convergence_mesh_size_{STUDY_TAG}",
)

histories_for_fig4 = [
    history_dataframe_from_ref(ref_cpl_iso, "Isotropic"),
    history_dataframe_from_ref(ref_cpl_ortho, "Orthotropic"),
]

plot_paper2_coupled_history_iso_ortho_combined(
    histories_for_fig4,
    filename_base=f"paper2_isotropic_orthotropic_simply_supported_coupled_reference_update_history_{STUDY_TAG}",
)

if RUN_REFERENCE_COMPARISON:
    ref_coarse_cpl, ref_reference_cpl, ref_compare = compare_two_reference_levels(
        "coupled",
        PLATE_CASES["Isotropic"]["mu_mech"],
        size_a=REFERENCE_COMPARISON_COARSE_SIZE,
        size_b=REF_SIZE,
        p=REF_P,
        label=f"Isotropic_coupled_{REFERENCE_COMPARISON_COARSE_SIZE}_vs_{REF_SIZE}_{MECHANICAL_BC_TYPE}",
        bc_type=MECHANICAL_BC_TYPE,
        fixed_heat=True,
        cache_label_a=f"Isotropic_coupled_refcheck_{REFERENCE_COMPARISON_COARSE_SIZE}_{MECHANICAL_BC_TYPE}",
        cache_label_b="Isotropic",
    )

# %% Cell 44 | id: f553ac5f-5842-46ea-b4c0-831e6c65be56


# %% Cell 45 | id: 587ac670-06ba-46fa-8551-5a0a19aea587


# %% Cell 46 | id: f5f51ed7-59ad-49f6-8537-5d2e0b33e273
# ============================================================
# Paper 2 thermomechanical FOM convergence with refined FOM reference
# ============================================================

PAPER2_OUTDIR = "paper2_fom_convergence"
os.makedirs(PAPER2_OUTDIR, exist_ok=True)

MECHANICAL_BC_TYPE = "simply_supported"
RESOLUTIONS_SMOKE = [8, 11, 16]
RESOLUTIONS_FINAL = [8, 11, 16, 23, 32, 45, 64]
RESOLUTIONS_COUPLED_FINAL = [8, 11, 16, 23, 32, 45, 64]
P_VALUES = [2, 3]

# ------------------------------------------------------------
# User-decision block
# ------------------------------------------------------------
RESUME_RUNS = True
FORCE_RERUN = False
RUN_REFERENCE_COMPARISON = True

# Current working reference setting.
REF_SIZE, REF_P = 100, 4

# Current working fixed heat discretization.
HEAT_NX, HEAT_NY, HEAT_NZ = 64, 32, 16
HEAT_DEGREE = 2
NZ_QUAD_T1 = 20

# General coarse/reference comparison size.
REFERENCE_COMPARISON_COARSE_SIZE = max(RESOLUTIONS_COUPLED_FINAL)

REFERENCE_CACHE_VERSION = 3
CONVERGENCE_TABLE_VERSION = 3

FIXED_HEAT_CONTROLS = dict(
    heat_nx=int(HEAT_NX),
    heat_ny=int(HEAT_NY),
    heat_nz=int(HEAT_NZ),
    heat_degree=int(HEAT_DEGREE),
    Nz_quad_T1=int(NZ_QUAD_T1),
)
REF_HEAT_CONTROLS = dict(FIXED_HEAT_CONTROLS)

# Settings-dependent label used in saved CSV/figure/reference-comparison names.
STUDY_TAG = (
    f"refN{int(REF_SIZE)}_p{int(REF_P)}"
    f"_heat{int(HEAT_NX)}x{int(HEAT_NY)}x{int(HEAT_NZ)}"
    f"_pT{int(HEAT_DEGREE)}"
    f"_qT1{int(NZ_QUAD_T1)}"
    f"_v{int(CONVERGENCE_TABLE_VERSION)}"
)

PLATE_CASES = {
    "Isotropic": dict(mu_mech=[1.0e4, 1.0e4, 0.3e4, 0.35e4, 1.0e3, -10000.0]),
    "Orthotropic": dict(mu_mech=[2.0e4, 1.0e4, 0.25e4, 0.30e4, 1.0e3, -10000.0]),
}

THERMAL_PARAMS = dict(
    T_amb=27.0 + 273.15,
    T_sub=24.0 + 273.15,
    h_con=10.0,
    eps_r=0.90,
    q_s=700.0,
    h_c_cont=200.0,
    h_c_gap=5.0,
    eta_c=50.0,
    w_contact=-0.1,
    kx=0.35,
    ky=0.35,
    kz=0.35,
    alpha1=1.3e-4,
    alpha2=1.3e-4,
    rho=1050.0,
)

COUPLING_PARAMS = dict(
    coupling_omega=0.7,
    coupling_tol_w=1e-6,
    coupling_tol_T1=1e-6,
    coupling_max_iters=40,
)

TABLE_COLS = """
mode case bc_type p size h inv_h plate_DoFs heat_DoFs theta_DoFs Total_DoFs
sqrt_plate_DoFs sqrt_Total_DoFs L2_RMS_Error_w_m Rel_L2_Error_w
L2_RMS_Error_theta Rel_L2_Error_theta coupled_iters coupled_converged
stopped_by_max_iters termination_reason err_w err_theta
""".split()

NEWTON_SOLVER_PARAMETERS["newton_solver"].update(
    linear_solver="petsc",
    absolute_tolerance=1e-4,
    relative_tolerance=1e-5,
    maximum_iterations=80,
    relaxation_parameter=1.0,
)


# ------------------------------------------------------------
# Mesh/thermal controls
# ------------------------------------------------------------

def heat_controls_from_size(size, *, ref=False, fixed=True):
    """
    Heat controls for convergence studies.

    fixed=True keeps the 3D heat discretization fixed so that the reported
    refinement behavior reflects the plate/FOM discretization rather than
    simultaneous heat-mesh refinement.
    """
    if ref:
        return dict(REF_HEAT_CONTROLS)

    if fixed:
        return dict(FIXED_HEAT_CONTROLS)

    return dict(
        heat_nx=int(size),
        heat_ny=max(2, int(round(size / 2))),
        heat_nz=max(8, int(round(size / 4))),
        heat_degree=int(HEAT_DEGREE),
        Nz_quad_T1=max(10, min(24, int(round(size / 4)) + 4)),
    )


# ------------------------------------------------------------
# Solver construction and single-case solve
# ------------------------------------------------------------

def make_solver(size, p, mu_mech, *, load_type="uniform", bc_type=MECHANICAL_BC_TYPE):
    solver = GeneralMultiphysicsSolver(study_case=1)
    solver.size = int(size)
    solver.degree = int(p)
    solver.load_type = load_type
    solver.bc_type = bc_type
    solver.define_domain(n_vert=0, n_horiz=0, plot_subdomains=False)
    solver.set_mu(mu_mech)
    solver.set_rom_thermal_parameters(**THERMAL_PARAMS)
    return solver


def copy_function(f):
    out = Function(f.function_space())
    out.assign(f)
    return out


def solve_paper2_case(
    *,
    mode,
    size,
    p,
    mu_mech,
    heat_nx,
    heat_ny,
    heat_nz,
    heat_degree,
    Nz_quad_T1,
    T1_cg_degree,
    bc_type=MECHANICAL_BC_TYPE,
):
    """
    Solve one Paper 2 FOM case.

    The one-way solver path is kept available because the framework still supports
    one-way thermomechanics. However, the revised refinement-study drivers below
    intentionally use mode='coupled' only.
    """
    solver = make_solver(size=size, p=p, mu_mech=mu_mech, bc_type=bc_type)

    heat_kwargs = dict(
        thermal_on=True,
        heat_nx=int(heat_nx),
        heat_ny=int(heat_ny),
        heat_nz=int(heat_nz),
        heat_degree=int(heat_degree),
        Nz_quad_T1=int(Nz_quad_T1),
        T1_cg_degree=int(T1_cg_degree),
        return_mode="global",
    )
    common = dict(solver=solver, mode=mode, bc_type=bc_type)

    if mode == "one_way":
        heat = solver.solve_rom_sample(mu_mech, coupled_on=False, **heat_kwargs)
        w = copy_function(solver.w_contact_global)
        theta = copy_function(solver.T1_from_heat)
        return dict(
            common,
            w=w,
            theta=theta,
            heat=heat,
            history=None,
            converged=True,
            stopped_by_max_iters=False,
            termination_reason="one_way",
            err_w=np.nan,
            err_T1=np.nan,
            iters=1,
            w_dofs=int(w.function_space().dim()),
            heat_dofs=int(heat["Vt"].dim()),
            theta_dofs=int(theta.function_space().dim()),
            total_dofs=int(w.function_space().dim() + heat["Vt"].dim() + theta.function_space().dim()),
            heat_nx=int(heat_nx),
            heat_ny=int(heat_ny),
            heat_nz=int(heat_nz),
            heat_degree=int(heat_degree),
            Nz_quad_T1=int(Nz_quad_T1),
            T1_cg_degree=int(T1_cg_degree),
        )

    if mode == "coupled":
        out = solver.solve_rom_sample(
            mu_mech,
            coupled_on=True,
            coupling_omega=COUPLING_PARAMS["coupling_omega"],
            coupling_tol_w=COUPLING_PARAMS["coupling_tol_w"],
            coupling_tol_T1=COUPLING_PARAMS["coupling_tol_T1"],
            coupling_max_iters=COUPLING_PARAMS["coupling_max_iters"],
            coupling_verbose=False,
            w0=0.0,
            return_coupled_dict=True,
            **heat_kwargs,
        )
        return dict(
            common,
            w=copy_function(out["w"]),
            theta=copy_function(out["T1"]),
            heat=out["heat"],
            history=out["history"],
            converged=bool(out["converged"]),
            stopped_by_max_iters=bool(out["stopped_by_max_iters"]),
            termination_reason=str(out["termination_reason"]),
            err_w=float(out["err_w"]),
            err_T1=float(out["err_T1"]),
            iters=int(out["iters"]),
            w_dofs=int(out["w_dofs"]),
            heat_dofs=int(out["heat_dofs"]),
            theta_dofs=int(out["T1_dofs"]),
            total_dofs=int(out["total_coupled_dofs"]),
            heat_nx=int(heat_nx),
            heat_ny=int(heat_ny),
            heat_nz=int(heat_nz),
            heat_degree=int(heat_degree),
            Nz_quad_T1=int(Nz_quad_T1),
            T1_cg_degree=int(T1_cg_degree),
        )

    raise ValueError("mode must be 'one_way' or 'coupled'.")


# ------------------------------------------------------------
# Error computation against refined reference
# ------------------------------------------------------------

def lift_field_to_reference_mesh(field, Vref, degree=5):
    """Interpolate a scalar plate field from its own mesh to the reference mesh."""
    try:
        field.set_allow_extrapolation(True)
    except Exception:
        pass

    class FieldOnReferenceMesh(UserExpression):
        def __init__(self, f, **kwargs):
            super().__init__(**kwargs)
            self.f = f

        def eval(self, values, x):
            values[0] = float(self.f(Point(float(x[0]), float(x[1]))))

        def value_shape(self):
            return ()

    return interpolate(FieldOnReferenceMesh(field, degree=degree), Vref)


def l2_rms_error_against_ref(field, ref_field, Vref, sqrt_area, degree=5):
    """
    RMS L2 error against the refined numerical FOM reference.

    This is not an exact-solution error.
    """
    field_refmesh = lift_field_to_reference_mesh(field, Vref, degree=degree)
    ref_projected = project(ref_field, Vref)
    return float(errornorm(ref_projected, field_refmesh, norm_type="L2", degree_rise=3) / sqrt_area)


def l2_relative_error_against_ref(field, ref_field, Vref, degree=5, eps=1e-14):
    """Relative L2 error against the refined numerical FOM reference."""
    dx_ref = Measure("dx", domain=Vref.mesh())
    field_refmesh = lift_field_to_reference_mesh(field, Vref, degree=degree)
    ref_projected = project(ref_field, Vref)

    abs_err = float(errornorm(ref_projected, field_refmesh, norm_type="L2", degree_rise=3))
    ref_norm = float(np.sqrt(max(float(assemble(ref_projected * ref_projected * dx_ref)), 0.0)))
    return abs_err / max(ref_norm, float(eps))


# ------------------------------------------------------------
# Checkpoint helpers
# ------------------------------------------------------------

def convergence_csv_path(mode, label, bc_type):
    return os.path.join(
        PAPER2_OUTDIR,
        f"paper2_{mode}_{label}_{bc_type}_convergence_{STUDY_TAG}.csv",
    )


def _as_bool(value, default=False):
    """Robust boolean conversion for bools, numpy bools, and CSV string values."""
    if value is None:
        return bool(default)

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, np.integer)):
        return bool(value)

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return bool(default)
        return bool(int(value))

    if isinstance(value, str):
        value_l = value.strip().lower()
        if value_l in ("true", "t", "1", "yes", "y"):
            return True
        if value_l in ("false", "f", "0", "no", "n", "nan", "none", ""):
            return False

    return bool(default)


def _bool_series(series, default=False):
    """Robust boolean mask for pandas Series that may contain strings from CSV."""
    return series.apply(lambda v: _as_bool(v, default=default)).astype(bool)


def _coupled_solution_is_converged(record):
    """True only if a coupled record/solution reached the coupled stopping criterion."""
    return (
        str(record.get("mode", "coupled")) == "coupled"
        and _as_bool(record.get("converged", record.get("coupled_converged", False)))
        and not _as_bool(record.get("stopped_by_max_iters", False))
    )


def _row_is_valid_converged_coupled(row):
    """True only for coupled rows that are valid for manuscript tables/plots."""
    try:
        if str(row.get("mode", "")) != "coupled":
            return False
        if not _as_bool(row.get("coupled_converged", False)):
            return False
        if _as_bool(row.get("stopped_by_max_iters", False)):
            return False
        if "reported" in row.index and not _as_bool(row.get("reported", True), default=True):
            return False
        if "stopping_norm" in row.index and str(row.get("stopping_norm", "")) != "L2_Omega":
            return False

        finite_cols = [
            "h",
            "L2_RMS_Error_w_m",
            "Rel_L2_Error_w",
            "L2_RMS_Error_theta",
            "Rel_L2_Error_theta",
            "err_w",
            "err_theta",
        ]
        return all(col not in row or np.isfinite(float(row[col])) for col in finite_cols)

    except Exception:
        return False


def load_existing_convergence_rows(out_csv):
    """
    Load old checkpoint rows.

    Only valid converged coupled rows are treated as completed. Failed or
    non-converged rows are kept in the CSV for traceability, but they are not
    allowed to block reruns.
    """
    if not (RESUME_RUNS and os.path.exists(out_csv) and not FORCE_RERUN):
        return None, set()

    df = pd.read_csv(out_csv)
    if df.empty:
        return None, set()

    required_key_cols = ["mode", "case", "bc_type", "p", "size"]
    valid_keys = {
        (str(row["mode"]), str(row["case"]), str(row["bc_type"]), int(row["p"]), int(row["size"]))
        for _, row in df.iterrows()
        if all(col in row.index for col in required_key_cols) and _row_is_valid_converged_coupled(row)
    }

    print(f"\nResuming from existing table: {out_csv}")
    print(f"Existing rows: {len(df)}")
    print(f"Valid converged coupled rows that will be skipped: {len(valid_keys)}")

    return df, valid_keys


def save_convergence_checkpoint(new_records, existing_df, out_csv):
    if not new_records:
        return existing_df if existing_df is not None else pd.DataFrame()

    new_df = pd.DataFrame.from_records(new_records)
    save_df = pd.concat([existing_df, new_df], ignore_index=True) if existing_df is not None and not existing_df.empty else new_df

    save_df = (
        save_df.drop_duplicates(["mode", "case", "bc_type", "p", "size"], keep="last")
        .sort_values(["mode", "case", "bc_type", "p", "h"], ascending=[True, True, True, True, False])
        .reset_index(drop=True)
    )
    save_df.to_csv(out_csv, index=False)
    print(f"Checkpoint saved: {out_csv}")

    return save_df


# ------------------------------------------------------------
# Reference-cache helpers
# ------------------------------------------------------------

def _json_safe(obj):
    """Convert numpy/scalar objects into JSON-safe Python objects."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _safe_filename_part(text):
    text = str(text)
    for ch in [" ", "/", "\\", ":", ";", ",", "(", ")", "[", "]", "{", "}"]:
        text = text.replace(ch, "_")
    return text


def reference_cache_key(*, mode, label, bc_type, ref_size, ref_p, mu_mech, ref_heat, fixed_heat):
    """Build a unique reference-cache key."""
    payload = dict(
        cache_version=int(REFERENCE_CACHE_VERSION),
        convergence_table_version=int(CONVERGENCE_TABLE_VERSION),
        study_tag=str(STUDY_TAG),
        mode=str(mode),
        case=str(label),
        bc_type=str(bc_type),
        ref_size=int(ref_size),
        ref_p=int(ref_p),
        mu_mech=[float(v) for v in mu_mech],
        thermal_params=_json_safe(THERMAL_PARAMS),
        ref_heat=_json_safe(ref_heat),
        fixed_heat=bool(fixed_heat),
        coupling_params=_json_safe(COUPLING_PARAMS) if mode == "coupled" else None,
        mechanical_bc_type=str(MECHANICAL_BC_TYPE),
        stopping_norm="L2_Omega",
        reported_refinement_branch="coupled_only",
    )
    payload_json = json.dumps(payload, sort_keys=True, indent=2)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:16], payload


def reference_cache_paths(*, mode, label, bc_type, ref_size, ref_p, cache_key):
    stem = (
        f"paper2_ref_{_safe_filename_part(mode)}_{_safe_filename_part(label)}"
        f"_{_safe_filename_part(bc_type)}_size{int(ref_size)}_p{int(ref_p)}_{cache_key}"
    )
    return dict(
        h5=os.path.join(PAPER2_OUTDIR, stem + ".h5"),
        json=os.path.join(PAPER2_OUTDIR, stem + ".json"),
    )


def _reference_meta(ref, w_save, theta_save, payload):
    return dict(
        payload=payload,
        history=ref.get("history", None),
        converged=bool(ref.get("converged", True)),
        stopped_by_max_iters=bool(ref.get("stopped_by_max_iters", False)),
        termination_reason=str(ref.get("termination_reason", "cached_reference")),
        err_w=float(ref.get("err_w", np.nan)),
        err_T1=float(ref.get("err_T1", np.nan)),
        iters=int(ref.get("iters", 1)),
        w_dofs=int(ref.get("w_dofs", w_save.function_space().dim())),
        heat_dofs=int(ref.get("heat_dofs", 0)),
        theta_dofs=int(ref.get("theta_dofs", theta_save.function_space().dim())),
        total_dofs=int(ref.get("total_dofs", 0)),
    )


def _assert_converged_coupled_record(record, *, context):
    """Stop immediately if a coupled reference/run did not converge."""
    if not bool(record.get("converged", False)):
        raise RuntimeError(
            f"{context} did not converge: "
            f"iters={record.get('iters', None)}, "
            f"err_w={record.get('err_w', np.nan)}, "
            f"err_T1={record.get('err_T1', np.nan)}, "
            f"termination={record.get('termination_reason', None)}"
        )

    if bool(record.get("stopped_by_max_iters", False)):
        raise RuntimeError(
            f"{context} reached max coupling iterations and must not be reported: "
            f"iters={record.get('iters', None)}, "
            f"err_w={record.get('err_w', np.nan)}, "
            f"err_T1={record.get('err_T1', np.nan)}, "
            f"termination={record.get('termination_reason', None)}"
        )


def save_reference_cache(ref, *, paths, payload, ref_p):
    """Save reference mesh, displacement, thermal driver, and metadata."""
    if payload.get("mode") == "coupled":
        _assert_converged_coupled_record(ref, context="Coupled reference before cache save")

    mesh = ref["solver"].mesh
    Vw_ref = FunctionSpace(mesh, "CG", int(ref_p))
    Vtheta_ref = FunctionSpace(mesh, "CG", int(ref_p))

    w_save = project(ref["w"], Vw_ref)
    theta_save = project(ref["theta"], Vtheta_ref)

    h5 = HDF5File(mesh.mpi_comm(), paths["h5"], "w")
    for name, obj in [("/mesh", mesh), ("/w", w_save), ("/theta", theta_save)]:
        h5.write(obj, name)
    h5.close()

    with open(paths["json"], "w") as f:
        json.dump(_json_safe(_reference_meta(ref, w_save, theta_save, payload)), f, indent=2, sort_keys=True)

    print("Saved reference cache:")
    print(f"  {paths['h5']}")
    print(f"  {paths['json']}")


def load_reference_cache(*, paths, ref_p):
    """Load cached reference mesh and fields."""
    if not (os.path.exists(paths["h5"]) and os.path.exists(paths["json"])):
        return None

    try:
        mesh = Mesh()
        h5 = HDF5File(MPI.comm_world, paths["h5"], "r")
        h5.read(mesh, "/mesh", False)

        Vw_ref = FunctionSpace(mesh, "CG", int(ref_p))
        Vtheta_ref = FunctionSpace(mesh, "CG", int(ref_p))
        w_ref = Function(Vw_ref)
        theta_ref = Function(Vtheta_ref)

        h5.read(w_ref, "/w")
        h5.read(theta_ref, "/theta")
        h5.close()

        with open(paths["json"], "r") as f:
            meta = json.load(f)

    except Exception as exc:
        print(f"Reference cache could not be loaded and will be rebuilt: {exc}")
        return None

    ref = dict(
        solver=None,
        mesh=mesh,
        w=w_ref,
        theta=theta_ref,
        heat=None,
        history=meta.get("history", None),
        converged=bool(meta.get("converged", True)),
        stopped_by_max_iters=bool(meta.get("stopped_by_max_iters", False)),
        termination_reason=str(meta.get("termination_reason", "loaded_reference_cache")),
        err_w=float(meta.get("err_w", np.nan)),
        err_T1=float(meta.get("err_T1", np.nan)),
        iters=int(meta.get("iters", 1)),
        w_dofs=int(meta.get("w_dofs", w_ref.function_space().dim())),
        heat_dofs=int(meta.get("heat_dofs", 0)),
        theta_dofs=int(meta.get("theta_dofs", theta_ref.function_space().dim())),
        total_dofs=int(meta.get("total_dofs", 0)),
    )

    if meta.get("payload", {}).get("mode") == "coupled":
        _assert_converged_coupled_record(ref, context="Loaded coupled reference cache")

    print("Loaded cached reference:")
    print(f"  {paths['h5']}")

    return ref


def get_or_build_reference(*, mode, label, bc_type, ref_size, ref_p, mu_mech, fixed_heat=True):
    """
    Build or load the refined FOM numerical reference.

    The revised reported refinement study uses coupled references only.
    """
    if mode != "coupled":
        raise ValueError(
            "The revised Paper 2 refinement study is coupled-only. "
            "Use mode='coupled' for reference construction."
        )

    ref_heat = heat_controls_from_size(ref_size, ref=True, fixed=fixed_heat)
    cache_key, payload = reference_cache_key(
        mode=mode,
        label=label,
        bc_type=bc_type,
        ref_size=ref_size,
        ref_p=ref_p,
        mu_mech=mu_mech,
        ref_heat=ref_heat,
        fixed_heat=fixed_heat,
    )
    paths = reference_cache_paths(
        mode=mode,
        label=label,
        bc_type=bc_type,
        ref_size=ref_size,
        ref_p=ref_p,
        cache_key=cache_key,
    )

    if RESUME_RUNS and not FORCE_RERUN:
        cached = load_reference_cache(paths=paths, ref_p=ref_p)
        if cached is not None:
            return cached

    print("\n" + "=" * 90)
    print(f"BUILDING COUPLED REFERENCE: {label}, bc={bc_type}, N_Omega={ref_size}, p={ref_p}")
    print("=" * 90)

    ref = solve_paper2_case(
        mode=mode,
        size=ref_size,
        p=ref_p,
        mu_mech=mu_mech,
        heat_nx=ref_heat["heat_nx"],
        heat_ny=ref_heat["heat_ny"],
        heat_nz=ref_heat["heat_nz"],
        heat_degree=ref_heat["heat_degree"],
        Nz_quad_T1=ref_heat["Nz_quad_T1"],
        T1_cg_degree=ref_p,
        bc_type=bc_type,
    )

    _assert_converged_coupled_record(ref, context="Newly built coupled reference")
    save_reference_cache(ref, paths=paths, payload=payload, ref_p=ref_p)
    return ref


def reference_mesh(ref):
    """Return the mesh from either a live reference solve or a cached reference."""
    return ref["solver"].mesh if ref.get("solver", None) is not None else ref["mesh"]


# ------------------------------------------------------------
# Convergence gathering
# ------------------------------------------------------------

def _success_record(mode, label, bc_type, p, res, sol, errors):
    ew_rms, etheta_rms, ew_rel, etheta_rel = errors
    return dict(
        mode=str(mode),
        case=str(label),
        bc_type=str(bc_type),
        p=int(p),
        size=int(res),
        h=float(sol["solver"].mesh.hmax()),
        inv_h=float(1.0 / sol["solver"].mesh.hmax()),
        plate_DoFs=int(sol["w_dofs"]),
        heat_DoFs=int(sol["heat_dofs"]),
        theta_DoFs=int(sol["theta_dofs"]),
        Total_DoFs=int(sol["total_dofs"]),
        sqrt_plate_DoFs=float(np.sqrt(sol["w_dofs"])),
        sqrt_Total_DoFs=float(np.sqrt(sol["total_dofs"])),
        heat_nx=int(sol["heat_nx"]),
        heat_ny=int(sol["heat_ny"]),
        heat_nz=int(sol["heat_nz"]),
        heat_degree=int(sol["heat_degree"]),
        Nz_quad_T1=int(sol["Nz_quad_T1"]),
        T1_cg_degree=int(sol["T1_cg_degree"]),
        L2_RMS_Error_w_m=float(ew_rms),
        L2_RMS_Error_theta=float(etheta_rms),
        Rel_L2_Error_w=float(ew_rel),
        Rel_L2_Error_theta=float(etheta_rel),
        coupled_iters=int(sol["iters"]),
        coupled_converged=bool(sol["converged"]),
        stopped_by_max_iters=bool(sol["stopped_by_max_iters"]),
        termination_reason=str(sol["termination_reason"]),
        err_w=float(sol["err_w"]),
        err_theta=float(sol["err_T1"]),
        stopping_norm="L2_Omega",
        reported=True,
    )


def _failed_record(mode, label, bc_type, p, res, hc, exc):
    return dict(
        mode=str(mode),
        case=str(label),
        bc_type=str(bc_type),
        p=int(p),
        size=int(res),
        h=np.nan,
        inv_h=np.nan,
        plate_DoFs=np.nan,
        heat_DoFs=np.nan,
        theta_DoFs=np.nan,
        Total_DoFs=np.nan,
        sqrt_plate_DoFs=np.nan,
        sqrt_Total_DoFs=np.nan,
        heat_nx=int(hc["heat_nx"]),
        heat_ny=int(hc["heat_ny"]),
        heat_nz=int(hc["heat_nz"]),
        heat_degree=int(hc["heat_degree"]),
        Nz_quad_T1=int(hc["Nz_quad_T1"]),
        T1_cg_degree=int(p),
        L2_RMS_Error_w_m=np.nan,
        L2_RMS_Error_theta=np.nan,
        Rel_L2_Error_w=np.nan,
        Rel_L2_Error_theta=np.nan,
        coupled_iters=np.nan,
        coupled_converged=False,
        stopped_by_max_iters=False,
        termination_reason=f"failed_or_not_reported: {exc}",
        err_w=np.nan,
        err_theta=np.nan,
        stopping_norm="L2_Omega",
        reported=False,
    )


def _excluded_record_from_solution(mode, label, bc_type, p, res, sol):
    """Save a non-converged coupled run for traceability, but mark it as not reportable."""
    return dict(
        mode=str(mode),
        case=str(label),
        bc_type=str(bc_type),
        p=int(p),
        size=int(res),
        h=float(sol["solver"].mesh.hmax()),
        inv_h=float(1.0 / sol["solver"].mesh.hmax()),
        plate_DoFs=int(sol["w_dofs"]),
        heat_DoFs=int(sol["heat_dofs"]),
        theta_DoFs=int(sol["theta_dofs"]),
        Total_DoFs=int(sol["total_dofs"]),
        sqrt_plate_DoFs=float(np.sqrt(sol["w_dofs"])),
        sqrt_Total_DoFs=float(np.sqrt(sol["total_dofs"])),
        heat_nx=int(sol["heat_nx"]),
        heat_ny=int(sol["heat_ny"]),
        heat_nz=int(sol["heat_nz"]),
        heat_degree=int(sol["heat_degree"]),
        Nz_quad_T1=int(sol["Nz_quad_T1"]),
        T1_cg_degree=int(sol["T1_cg_degree"]),
        L2_RMS_Error_w_m=np.nan,
        L2_RMS_Error_theta=np.nan,
        Rel_L2_Error_w=np.nan,
        Rel_L2_Error_theta=np.nan,
        coupled_iters=int(sol["iters"]),
        coupled_converged=bool(sol["converged"]),
        stopped_by_max_iters=bool(sol["stopped_by_max_iters"]),
        termination_reason=f"not_reported_nonconverged: {sol['termination_reason']}",
        err_w=float(sol["err_w"]),
        err_theta=float(sol["err_T1"]),
        stopping_norm="L2_Omega",
        reported=False,
    )


def valid_converged_coupled_rows(df):
    """
    Keep only rows admissible for manuscript tables and plots.

    Excludes:
        - one-way rows,
        - failed rows,
        - non-converged coupled rows,
        - rows stopped by max iterations,
        - rows with missing/non-finite error values.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out = out[out["mode"].astype(str) == "coupled"]

    if "coupled_converged" in out.columns:
        out = out[_bool_series(out["coupled_converged"], default=False)]
    if "stopped_by_max_iters" in out.columns:
        out = out[~_bool_series(out["stopped_by_max_iters"], default=False)]
    if "reported" in out.columns:
        out = out[_bool_series(out["reported"], default=True)]
    if "stopping_norm" in out.columns:
        out = out[out["stopping_norm"].astype(str) == "L2_Omega"]

    for col in [
        "h",
        "L2_RMS_Error_w_m",
        "Rel_L2_Error_w",
        "L2_RMS_Error_theta",
        "Rel_L2_Error_theta",
        "err_w",
        "err_theta",
    ]:
        if col in out.columns:
            out = out[np.isfinite(out[col].to_numpy(dtype=float))]

    return out.reset_index(drop=True)


def gather_paper2_convergence_data(
    *,
    mode,
    mu_mech,
    resolutions=RESOLUTIONS_COUPLED_FINAL,
    p_values=P_VALUES,
    ref_size=REF_SIZE,
    ref_p=REF_P,
    label="case",
    bc_type=MECHANICAL_BC_TYPE,
    fixed_heat=True,
):
    """
    Gather coupled-only convergence data using a cached refined FOM reference.

    Non-converged coupled runs are saved as failed/non-reported rows, but they
    are excluded from all manuscript tables and plots.
    """
    if mode != "coupled":
        raise ValueError(
            "The revised Paper 2 refinement study must be coupled-only. "
            "Do not call gather_paper2_convergence_data with mode='one_way'."
        )

    out_csv = convergence_csv_path(mode, label, bc_type)
    existing_df, existing_keys = load_existing_convergence_rows(out_csv)

    ref = get_or_build_reference(
        mode=mode,
        label=label,
        bc_type=bc_type,
        ref_size=ref_size,
        ref_p=ref_p,
        mu_mech=mu_mech,
        fixed_heat=fixed_heat,
    )
    _assert_converged_coupled_record(ref, context=f"Coupled reference for {label}, bc={bc_type}")

    ref_mesh_obj = reference_mesh(ref)
    Vw_ref = FunctionSpace(ref_mesh_obj, "CG", int(ref_p))
    Vtheta_ref = FunctionSpace(ref_mesh_obj, "CG", int(ref_p))
    dx_ref = Measure("dx", domain=ref_mesh_obj)
    sqrt_area = float(np.sqrt(assemble(Constant(1.0) * dx_ref)))

    w_ref = project(ref["w"], Vw_ref)
    theta_ref = project(ref["theta"], Vtheta_ref)

    for res in resolutions:
        for p in p_values:
            if int(res) == int(ref_size) and int(p) == int(ref_p):
                continue

            case_key = (mode, label, bc_type, int(p), int(res))
            if case_key in existing_keys:
                print(f"SKIPPING valid cached row: mode={mode}, case={label}, bc={bc_type}, size={res}, p={p}")
                continue

            hc = heat_controls_from_size(res, ref=False, fixed=fixed_heat)
            print("\n" + "-" * 90)
            print(
                f"COUPLED | {label} | bc={bc_type} | size={res}, p={p}, "
                f"heat=({hc['heat_nx']},{hc['heat_ny']},{hc['heat_nz']}), "
                f"pT={hc['heat_degree']}, Nz_quad_T1={hc['Nz_quad_T1']}"
            )
            print("-" * 90)

            try:
                sol = solve_paper2_case(
                    mode=mode,
                    size=res,
                    p=p,
                    mu_mech=mu_mech,
                    heat_nx=hc["heat_nx"],
                    heat_ny=hc["heat_ny"],
                    heat_nz=hc["heat_nz"],
                    heat_degree=hc["heat_degree"],
                    Nz_quad_T1=hc["Nz_quad_T1"],
                    T1_cg_degree=p,
                    bc_type=bc_type,
                )

                if not _coupled_solution_is_converged(sol):
                    print(
                        f"EXCLUDED NON-CONVERGED RUN: case={label}, size={res}, p={p}, "
                        f"iters={sol.get('iters', None)}, err_w={sol.get('err_w', np.nan)}, "
                        f"err_T1={sol.get('err_T1', np.nan)}, termination={sol.get('termination_reason', None)}"
                    )
                    record = _excluded_record_from_solution(mode, label, bc_type, p, res, sol)

                else:
                    deg = max(5, int(p) + 2)
                    errors = (
                        l2_rms_error_against_ref(sol["w"], w_ref, Vw_ref, sqrt_area, degree=deg),
                        l2_rms_error_against_ref(sol["theta"], theta_ref, Vtheta_ref, sqrt_area, degree=deg),
                        l2_relative_error_against_ref(sol["w"], w_ref, Vw_ref, degree=deg),
                        l2_relative_error_against_ref(sol["theta"], theta_ref, Vtheta_ref, degree=deg),
                    )
                    record = _success_record(mode, label, bc_type, p, res, sol, errors)

            except Exception as exc:
                print(f"FAILED OR EXCLUDED: mode={mode}, case={label}, bc={bc_type}, size={res}, p={p}: {exc}")
                record = _failed_record(mode, label, bc_type, p, res, hc, exc)

            existing_df = save_convergence_checkpoint([record], existing_df, out_csv)

            if _row_is_valid_converged_coupled(pd.Series(record)):
                existing_keys.add(case_key)

    df = existing_df.copy() if existing_df is not None and not existing_df.empty else pd.DataFrame()

    if not df.empty:
        df = df.sort_values(
            ["mode", "case", "bc_type", "p", "h"],
            ascending=[True, True, True, True, False],
        ).reset_index(drop=True)

    df.to_csv(out_csv, index=False)
    df_valid = valid_converged_coupled_rows(df)

    print(f"\nSaved convergence table: {out_csv}")
    print(f"Total rows in CSV: {len(df)}")
    print(f"Valid converged coupled rows for reporting: {len(df_valid)}")

    return df, ref


# ------------------------------------------------------------
# Table helper
# ------------------------------------------------------------

def finest_converged_rows_for_table(df):
    """Return the finest valid coupled row for each case and p."""
    df_valid = valid_converged_coupled_rows(df)
    if df_valid.empty:
        return df_valid

    rows = [grp.sort_values("size", ascending=False).iloc[0] for _, grp in df_valid.groupby(["case", "p"])]
    return pd.DataFrame(rows).sort_values(["case", "p"]).reset_index(drop=True)


def print_finest_converged_table(df, title="FINEST CONVERGED COUPLED ROWS"):
    table_df = finest_converged_rows_for_table(df)

    print("\n" + title)
    if table_df.empty:
        print("No valid converged coupled rows available.")
        return table_df

    print(table_df[TABLE_COLS].to_string(index=False, float_format="%.6e"))
    return table_df


# ------------------------------------------------------------
# Coupled-only convergence plotting: revised Fig. 3
# ------------------------------------------------------------

def plot_paper2_coupled_convergence_iso_ortho(
    df_coupled,
    *,
    ycol="L2_RMS_Error_w_m",
    ylabel=r"$L^2$ RMS error in $w_h$ [m]",
    x_mode="h",
    filename_base=None,
    rate_fit_slice=slice(None),
):
    """
    Revised Paper 2 Fig. 3:
    coupled-only displacement convergence for isotropic/orthotropic cases.

    Only valid converged coupled rows are plotted.

    Note:
    For x_mode='h', the x-axis is intentionally shown from coarse to fine,
    i.e. larger h on the left and smaller h on the right.

    Rate triangles:
    - rates are computed from a log-log least-squares fit using rate_fit_slice;
      the default slice(None) gives a global fitted rate over all refinement data.
    - all triangles are kept near the finest segment;
    - triangles are compact to avoid overlap;
    - for each polynomial degree p, the higher-error curve gets the upper triangle
      and the lower-error curve gets the lower triangle;
    - the label "1" is always placed above the horizontal triangle edge.
    """

    if filename_base is None:
        filename_base = f"paper2_isotropic_orthotropic_simply_supported_coupled_w_convergence_mesh_size_{STUDY_TAG}"

    df = valid_converged_coupled_rows(df_coupled)
    if df.empty:
        raise ValueError("No valid converged coupled rows available for Fig. 3.")

    plt.style.use(["science", "ieee", "notebook", "grid"])
    plt.rcParams["figure.autolayout"] = False

    x_options = {
        "h": ("h", r"Characteristic mesh length $\ell_{\mathcal{T}}$ [m]"),
        "inv_h": ("inv_h", r"Inverse mesh size $1/h$"),
        "sqrt_plate_dofs": ("sqrt_plate_DoFs", r"$\sqrt{\mathrm{plate\ DoFs}}$"),
        "plate_dofs": ("plate_DoFs", "Plate degrees of freedom"),
        "sqrt_total_dofs": ("sqrt_Total_DoFs", r"$\sqrt{\mathrm{total\ DoFs}}$"),
        "total_dofs": ("Total_DoFs", "Total degrees of freedom"),
    }
    if x_mode not in x_options:
        raise ValueError(f"x_mode must be one of {list(x_options.keys())}")

    xcol, xlabel = x_options[x_mode]

    style_specs = [
        ("Isotropic", 2, "black", "-", "o", r"Isotropic, $p=2$"),
        ("Isotropic", 3, "blue", "--", "s", r"Isotropic, $p=3$"),
        ("Orthotropic", 2, "darkorange", "-", "o", r"Orthotropic, $p=2$"),
        ("Orthotropic", 3, "purple", "--", "s", r"Orthotropic, $p=3$"),
    ]
    styles = {
        (case_name, p): dict(color=color, linestyle=linestyle, marker=marker, label=label)
        for case_name, p, color, linestyle, marker, label in style_specs
    }

    def positive_finite(vals):
        vals = np.asarray(vals, dtype=float)
        return vals[np.isfinite(vals) & (vals > 0.0)]

    def ordered_xy(x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
        x, y = x[mask], y[mask]
        order = np.argsort(x)[::-1] if x_mode == "h" else np.argsort(x)
        return x[order], y[order]

    def estimate_loglog_slope(x, y, fit_slice=slice(None)):
        x, y = ordered_xy(x, y)
        if len(x) < 2:
            return None

        xx, yy = x[fit_slice], y[fit_slice]
        if len(xx) < 2:
            return None

        slope, _ = np.polyfit(np.log10(xx), np.log10(yy), 1)
        return float(abs(slope))

    def fine_error_value(grp):
        x, y = ordered_xy(grp[xcol].to_numpy(float), grp[ycol].to_numpy(float))
        return np.nan if len(y) == 0 else float(y[-1])

    def build_triangle_specs(grouped):
        """
        Automatic compact triangle placement.

        For each polynomial degree:
            higher-error curve -> triangle above,
            lower-error curve  -> triangle below.

        All triangles stay near the finest segment. The p=3 triangles are made
        slightly smaller than p=2 because the p=3 curves are closer together.
        """
        specs = {}

        for p in sorted({key[1] for key in grouped}):
            keys_p = [key for key in grouped if key[1] == p]
            if not keys_p:
                continue

            keys_p = sorted(keys_p, key=lambda key: fine_error_value(grouped[key]))

            if len(keys_p) == 1:
                specs[keys_p[0]] = dict(
                    location="above",
                    start_idx=-2,
                    span_frac=0.34,
                    tri_gap=1.22,
                    rate_side_pad=0.020,
                    one_label_gap=1.055,
                    rate_label_shift=1.00,
                )
                continue

            lower_key = keys_p[0]
            upper_key = keys_p[-1]

            if int(p) == 2:
                specs[upper_key] = dict(
                    location="above",
                    start_idx=-2,
                    span_frac=0.36,
                    tri_gap=1.24,
                    rate_side_pad=0.020,
                    one_label_gap=1.055,
                    rate_label_shift=1.00,
                )
                specs[lower_key] = dict(
                    location="below",
                    start_idx=-2,
                    span_frac=0.36,
                    tri_gap=1.28,
                    rate_side_pad=0.020,
                    one_label_gap=1.055,
                    rate_label_shift=1.00,
                )
            else:
                specs[upper_key] = dict(
                    location="above",
                    start_idx=-2,
                    span_frac=0.28,
                    tri_gap=1.20,
                    rate_side_pad=0.018,
                    one_label_gap=1.060,
                    rate_label_shift=1.00,
                )
                specs[lower_key] = dict(
                    location="below",
                    start_idx=-2,
                    span_frac=0.28,
                    tri_gap=1.26,
                    rate_side_pad=0.018,
                    one_label_gap=1.060,
                    rate_label_shift=1.00,
                )

        return specs

    def add_rate_triangle(
        ax,
        x,
        y,
        rate,
        color,
        *,
        start_idx=-2,
        span_frac=0.34,
        tri_gap=1.22,
        rate_side_pad=0.020,
        one_label_gap=1.055,
        rate_label_shift=1.00,
        lw=1.00,
        fontsize=8,
        location="above",
    ):
        if rate is None or rate <= 0.0:
            return

        x, y = ordered_xy(x, y)
        if len(x) < 2:
            return

        start_idx = len(x) + start_idx if start_idx < 0 else start_idx
        if start_idx < 0 or start_idx + 1 >= len(x):
            return

        x_prev = float(x[start_idx])
        x_last = float(x[start_idx + 1])
        y_last = float(y[start_idx + 1])

        if x_prev <= 0.0 or x_last <= 0.0 or y_last <= 0.0:
            return

        if x_mode == "h":
            full_ratio = x_prev / x_last
            if full_ratio <= 1.0:
                return

            x_right = x_last
            x_left = x_last * (full_ratio ** span_frac)
            local_ratio = x_left / x_right
            x_rate = x_right / (local_ratio ** rate_side_pad)

        else:
            full_ratio = x_last / x_prev
            if full_ratio <= 1.0:
                return

            x_left = x_last / (full_ratio ** span_frac)
            x_right = x_last
            local_ratio = x_right / x_left
            x_rate = x_right * (local_ratio ** rate_side_pad)

        if location == "below":
            # Triangle is below the curve.
            # y_top is still the horizontal edge of the triangle.
            y_top = y_last / tri_gap
            y_bottom = y_top / (local_ratio ** rate)
        else:
            # Triangle is above the curve.
            # y_top is the horizontal edge of the triangle.
            y_bottom = y_last * tri_gap
            y_top = y_bottom * (local_ratio ** rate)

        if min(x_left, x_right, y_top, y_bottom) <= 0.0:
            return

        # Important fix:
        # Put the "1" label above the horizontal edge for both upper and lower triangles.
        one_label_y = y_top * one_label_gap
        one_label_va = "bottom"

        y_mid = np.sqrt(y_top * y_bottom) * float(rate_label_shift)

        ymin, ymax = ax.get_ylim()
        log_ymin, log_ymax = np.log10(ymin), np.log10(ymax)
        low = np.log10(min(y_top, y_bottom))
        high = np.log10(max(y_top, y_bottom, one_label_y))

        if low < log_ymin + 0.08:
            scale = 10 ** (log_ymin + 0.08) / min(y_top, y_bottom)
            y_top *= scale
            y_bottom *= scale
            one_label_y *= scale
            y_mid *= scale

        if high > log_ymax - 0.08:
            scale = 10 ** (log_ymax - 0.08) / max(y_top, y_bottom, one_label_y)
            y_top *= scale
            y_bottom *= scale
            one_label_y *= scale
            y_mid *= scale

        text_box = dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.12)

        ax.plot([x_left, x_right], [y_top, y_top], color=color, lw=lw, zorder=5)
        ax.plot([x_right, x_right], [y_top, y_bottom], color=color, lw=lw, zorder=5)
        ax.plot([x_left, x_right], [y_top, y_bottom], color=color, lw=lw, zorder=5)

        ax.text(
            np.sqrt(x_left * x_right),
            one_label_y,
            "1",
            ha="center",
            va=one_label_va,
            fontsize=fontsize,
            color=color,
            bbox=text_box,
            clip_on=True,
            zorder=6,
        )

        ax.text(
            x_rate,
            y_mid,
            f"{rate:.1f}",
            ha="left",
            va="center",
            fontsize=fontsize,
            color=color,
            bbox=text_box,
            clip_on=True,
            zorder=6,
        )

    all_x = positive_finite(df[xcol].to_numpy(float))
    all_y = positive_finite(df[ycol].to_numpy(float))
    if len(all_x) == 0 or len(all_y) == 0:
        raise ValueError("No positive finite data available for Fig. 3.")

    fig, ax = plt.subplots(figsize=(6.8, 5.2), facecolor="white")
    grouped = {}

    for key, style in styles.items():
        case_name, p = key
        grp = df[
            (df["case"].astype(str) == case_name)
            & (df["p"].astype(int) == int(p))
        ].dropna(subset=[xcol, ycol]).copy()

        if grp.empty:
            continue

        grp = grp.sort_values(xcol, ascending=(x_mode != "h"))
        grouped[key] = grp

        ax.loglog(
            grp[xcol].to_numpy(float),
            grp[ycol].to_numpy(float),
            linestyle=style["linestyle"],
            marker=style["marker"],
            color=style["color"],
            markeredgecolor=style["color"],
            markerfacecolor="none",
            linewidth=2.2,
            markersize=8.0,
            markeredgewidth=1.4,
            label=style["label"],
            clip_on=False,
        )

    ax.set_facecolor("white")
    ax.set_title(r"Partitioned coupled refinement study", fontsize=14, pad=8)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=13, length=4, pad=2)
    ax.tick_params(axis="both", which="minor", length=2)
    ax.grid(True, which="both", linestyle=":", linewidth=0.45, alpha=0.35)

    if x_mode == "h":
        ticks = []
        for size in [8, 16, 32, 64]:
            vals = positive_finite(df.loc[df["size"].astype(int) == size, "h"].to_numpy(float))
            if len(vals):
                ticks.append(float(np.median(vals)))

        if ticks:
            ticks = sorted(set(np.round(ticks, 12)), reverse=True)
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, pos: f"{x:.2f}" if x >= 0.1 else f"{x:.3f}")
            )
    else:
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=80))
    ax.xaxis.set_minor_formatter(NullFormatter())

    # Important:
    # For h-refinement, show coarse-to-fine from left to right:
    # left  = largest h, approximately 0.25
    # right = smallest h, approximately 0.031
    if x_mode == "h":
        ax.set_xlim(np.max(all_x) * 1.20, np.min(all_x) / 1.14)
    else:
        ax.set_xlim(np.min(all_x) / 1.14, np.max(all_x) * 1.20)

    ax.set_ylim(np.min(all_y) / 2.60, np.max(all_y) * 1.65)

    triangle_specs = build_triangle_specs(grouped)

    for key, grp in grouped.items():
        style = styles[key]
        spec = triangle_specs.get(
            key,
            dict(
                location="above",
                start_idx=-2,
                span_frac=0.34,
                tri_gap=1.22,
                rate_side_pad=0.020,
                one_label_gap=1.055,
                rate_label_shift=1.00,
            ),
        )

        rate = estimate_loglog_slope(
            grp[xcol].to_numpy(float),
            grp[ycol].to_numpy(float),
            fit_slice=rate_fit_slice,
        )

        add_rate_triangle(
            ax,
            grp[xcol].to_numpy(float),
            grp[ycol].to_numpy(float),
            rate,
            style["color"],
            location=spec["location"],
            start_idx=spec["start_idx"],
            span_frac=spec["span_frac"],
            tri_gap=spec["tri_gap"],
            rate_side_pad=spec["rate_side_pad"],
            one_label_gap=spec["one_label_gap"],
            rate_label_shift=spec["rate_label_shift"],
            lw=1.00,
            fontsize=8,
        )

    leg = ax.legend(
        loc="lower left",
        fontsize=12,
        frameon=True,
        borderpad=0.55,
        handlelength=2.2,
        handletextpad=0.55,
        labelspacing=0.35,
    )
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.8)

    os.makedirs(PAPER2_OUTDIR, exist_ok=True)
    for ext, dpi in (("pdf", 300), ("png", 600)):
        fig.savefig(
            os.path.join(PAPER2_OUTDIR, f"{filename_base}.{ext}"),
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
        )

    plt.show()

# ------------------------------------------------------------
# Coupled update-history plotting: revised Fig. 4
# ------------------------------------------------------------

def history_dataframe_from_ref(ref_or_solution, case_label):
    """Convert a coupled reference/solution history into a DataFrame for Fig. 4."""
    hist = pd.DataFrame(ref_or_solution.get("history", None)).copy()
    if hist.empty:
        return hist

    hist["case"] = str(case_label)
    return hist


def load_saved_coupled_reference_history(case_label, *, bc_type=MECHANICAL_BC_TYPE):
    """
    Load coupled-reference fixed-point history from saved reference-cache JSON.
    No FOM solve is executed.
    """

    pattern = os.path.join(
        PAPER2_OUTDIR,
        (
            f"paper2_ref_coupled_{_safe_filename_part(case_label)}"
            f"_{_safe_filename_part(bc_type)}"
            f"_size{int(REF_SIZE)}_p{int(REF_P)}_*.json"
        ),
    )
    candidates = sorted(glob.glob(pattern))

    if not candidates:
        raise FileNotFoundError(f"No coupled reference-cache JSON found for {case_label}: {pattern}")

    for path in candidates:
        with open(path, "r") as f:
            meta = json.load(f)

        payload = meta.get("payload", {})
        is_match = (
            payload.get("mode") == "coupled"
            and payload.get("case") == case_label
            and payload.get("bc_type") == bc_type
            and int(payload.get("ref_size", -1)) == int(REF_SIZE)
            and int(payload.get("ref_p", -1)) == int(REF_P)
            and payload.get("ref_heat", {}) == _json_safe(REF_HEAT_CONTROLS)
            and payload.get("stopping_norm") == "L2_Omega"
            and payload.get("reported_refinement_branch") == "coupled_only"
        )
        hist = meta.get("history", None)

        if is_match and hist:
            df = pd.DataFrame(hist).copy()
            df["case"] = case_label
            df["source_file"] = os.path.basename(path)
            return df

    raise FileNotFoundError(f"No valid coupled history found for {case_label} in {len(candidates)} JSON file(s).")


def plot_paper2_coupled_history_iso_ortho_combined(
    histories,
    *,
    filename_base=None,
):
    """
    Revised Paper 2 Fig. 4:
    fixed-point relative-update histories only.

    The old response-evolution panel is intentionally removed.
    """
    if filename_base is None:
        filename_base = f"paper2_isotropic_orthotropic_simply_supported_coupled_reference_update_history_{STUDY_TAG}"

    if not histories:
        print("No coupled histories available.")
        return

    hist = pd.concat(histories, ignore_index=True)
    if hist.empty:
        print("No coupled histories available.")
        return

    plt.style.use(["science", "ieee", "notebook", "grid"])
    plt.rcParams["figure.autolayout"] = False

    hist_styles = {
        ("Isotropic", "err_w"): dict(color="black", linestyle="-", marker="o", label=r"Isotropic, $\varepsilon_w$"),
        ("Isotropic", "err_T1"): dict(color="red", linestyle="--", marker="s", label=r"Isotropic, $\varepsilon_{\theta}$"),
        ("Orthotropic", "err_w"): dict(color="blue", linestyle="-", marker="o", label=r"Orthotropic, $\varepsilon_w$"),
        ("Orthotropic", "err_T1"): dict(color="purple", linestyle="--", marker="s", label=r"Orthotropic, $\varepsilon_{\theta}$"),
    }

    fig, ax = plt.subplots(figsize=(6.8, 5.2), facecolor="white")

    for case_name in ["Isotropic", "Orthotropic"]:
        grp = hist[hist["case"].astype(str) == case_name].copy()
        if grp.empty:
            continue

        grp = grp.sort_values("iter")
        it = grp["iter"].to_numpy(int)

        for col in ["err_w", "err_T1"]:
            vals = grp[col].to_numpy(float)
            if col == "err_T1" and not np.isfinite(vals[1:]).any():
                continue

            style = hist_styles[(case_name, col)]
            ax.semilogy(
                it,
                vals,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2.2,
                markersize=7.0,
                markerfacecolor=style["color"],
                label=style["label"],
            )

    ax.set_facecolor("white")
    ax.set_title("Coupled fixed-point update history", fontsize=14, pad=8)
    ax.set_xlabel("Coupling iteration", fontsize=14)
    ax.set_ylabel(r"Relative $L^2(\Omega)$ update", fontsize=14)
    ax.tick_params(axis="both", labelsize=13, length=3, pad=2)
    ax.grid(True, which="both", linestyle=":", linewidth=0.45, alpha=0.35)

    leg = ax.legend(
        loc="upper right",
        fontsize=11.5,
        frameon=True,
        borderpad=0.55,
        handlelength=2.1,
        handletextpad=0.55,
        labelspacing=0.35,
    )
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.8)

    os.makedirs(PAPER2_OUTDIR, exist_ok=True)
    for ext, dpi in (("pdf", 300), ("png", 600)):
        fig.savefig(
            os.path.join(PAPER2_OUTDIR, f"{filename_base}.{ext}"),
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
        )

    plt.show()
    
# ------------------------------------------------------------
# Optional reference-level comparison
# ------------------------------------------------------------

def compare_two_reference_levels(
    mode,
    mu_mech,
    *,
    size_a=REFERENCE_COMPARISON_COARSE_SIZE,
    size_b=REF_SIZE,
    p=REF_P,
    label="reference_check",
    bc_type=MECHANICAL_BC_TYPE,
    fixed_heat=True,
    cache_label_a=None,
    cache_label_b=None,
):
    """
    Optional cached comparison between two coupled refined reference levels.

    This is not the main manuscript convergence table. It is a diagnostic to
    check the separation between the finest tested mesh and the reference.
    """
    if mode != "coupled":
        raise ValueError("Reference-level comparison for the revised study must be coupled-only.")

    cache_label_a = cache_label_a or f"{label}_size{int(size_a)}"
    cache_label_b = cache_label_b or f"{label}_size{int(size_b)}"

    sol_a = get_or_build_reference(
        mode=mode,
        label=cache_label_a,
        bc_type=bc_type,
        ref_size=size_a,
        ref_p=p,
        mu_mech=mu_mech,
        fixed_heat=fixed_heat,
    )
    sol_b = get_or_build_reference(
        mode=mode,
        label=cache_label_b,
        bc_type=bc_type,
        ref_size=size_b,
        ref_p=p,
        mu_mech=mu_mech,
        fixed_heat=fixed_heat,
    )

    _assert_converged_coupled_record(sol_a, context=f"Reference level A size={size_a}")
    _assert_converged_coupled_record(sol_b, context=f"Reference level B size={size_b}")

    mesh_b = reference_mesh(sol_b)
    Vw_b = FunctionSpace(mesh_b, "CG", int(p))
    Vtheta_b = FunctionSpace(mesh_b, "CG", int(p))
    sqrt_area_b = float(np.sqrt(assemble(Constant(1.0) * Measure("dx", domain=mesh_b))))
    deg = max(5, int(p) + 2)

    comparison = dict(
        mode=mode,
        label=label,
        bc_type=bc_type,
        size_a=int(size_a),
        size_b=int(size_b),
        p=int(p),
        L2_RMS_Error_w_m=l2_rms_error_against_ref(sol_a["w"], sol_b["w"], Vw_b, sqrt_area_b, degree=deg),
        L2_RMS_Error_theta=l2_rms_error_against_ref(sol_a["theta"], sol_b["theta"], Vtheta_b, sqrt_area_b, degree=deg),
        Rel_L2_Error_w=l2_relative_error_against_ref(sol_a["w"], sol_b["w"], Vw_b, degree=deg),
        Rel_L2_Error_theta=l2_relative_error_against_ref(sol_a["theta"], sol_b["theta"], Vtheta_b, degree=deg),
    )

    out_json = os.path.join(
        PAPER2_OUTDIR,
        f"paper2_{mode}_{_safe_filename_part(label)}_{bc_type}_{STUDY_TAG}_reference_comparison.json",
    )

    with open(out_json, "w") as f:
        json.dump(_json_safe(comparison), f, indent=2, sort_keys=True)

    print("\nREFERENCE-LEVEL COMPARISON")
    for key, value in comparison.items():
        print(f"{key}: {value}")
    print(f"Saved reference comparison: {out_json}")

    return sol_a, sol_b, comparison


# ------------------------------------------------------------
# Driver utilities: coupled-only
# ------------------------------------------------------------

def run_smoke_test(case_label="Isotropic", *, bc_type=MECHANICAL_BC_TYPE):
    """Coupled-only smoke test. This avoids accidentally reintroducing one-way refinement rows."""
    mu_mech = PLATE_CASES[case_label]["mu_mech"]

    df_cpl, ref_cpl = gather_paper2_convergence_data(
        mode="coupled",
        mu_mech=mu_mech,
        resolutions=RESOLUTIONS_SMOKE,
        p_values=[2],
        ref_size=32,
        ref_p=3,
        label=f"{case_label}_smoke",
        bc_type=bc_type,
        fixed_heat=True,
    )

    print(f"\nCOUPLED SMOKE {case_label.upper()} ({bc_type})")
    print(valid_converged_coupled_rows(df_cpl)[TABLE_COLS].to_string(index=False, float_format="%.6e"))

    return df_cpl, ref_cpl


def run_case(case_label, *, bc_type=MECHANICAL_BC_TYPE):
    """Run the revised Paper 2 refinement study for one material case. Coupled-only."""
    mu_mech = PLATE_CASES[case_label]["mu_mech"]

    df_cpl, ref_cpl = gather_paper2_convergence_data(
        mode="coupled",
        mu_mech=mu_mech,
        resolutions=RESOLUTIONS_COUPLED_FINAL,
        p_values=P_VALUES,
        ref_size=REF_SIZE,
        ref_p=REF_P,
        label=case_label,
        bc_type=bc_type,
        fixed_heat=True,
    )

    print(f"\nCOUPLED {case_label.upper()} ({bc_type})")
    df_valid = valid_converged_coupled_rows(df_cpl)

    if df_valid.empty:
        print("No valid converged coupled rows available.")
    else:
        print(df_valid[TABLE_COLS].to_string(index=False, float_format="%.6e"))

    return df_cpl, ref_cpl


def replot_coupled_from_saved_csv(*, bc_type=MECHANICAL_BC_TYPE):
    """Replot revised Fig. 3 from saved coupled CSVs only."""
    frames = []
    for case_label in ["Isotropic", "Orthotropic"]:
        path = convergence_csv_path("coupled", case_label, bc_type)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing coupled convergence CSV: {path}")
        frames.append(pd.read_csv(path))

    df_all = pd.concat(frames, ignore_index=True)
    plot_paper2_coupled_convergence_iso_ortho(
        df_all,
        ycol="L2_RMS_Error_w_m",
        ylabel=r"$L^2$ RMS error in $w_h$ [m]",
        x_mode="h",
        filename_base=f"paper2_isotropic_orthotropic_simply_supported_coupled_w_convergence_mesh_size_{STUDY_TAG}",
    )

    return df_all


def replot_coupled_history_from_saved_json(*, bc_type=MECHANICAL_BC_TYPE):
    """Replot revised Fig. 4 from saved coupled reference-cache JSON histories only."""
    histories = [
        load_saved_coupled_reference_history("Isotropic", bc_type=bc_type),
        load_saved_coupled_reference_history("Orthotropic", bc_type=bc_type),
    ]
    plot_paper2_coupled_history_iso_ortho_combined(
        histories,
        filename_base=f"paper2_isotropic_orthotropic_simply_supported_coupled_reference_update_history_{STUDY_TAG}",
    )

    return histories


# ------------------------------------------------------------
# Execute revised coupled-only study
# ------------------------------------------------------------

# Optional quick test:
# df_cpl_smoke, ref_cpl_smoke = run_smoke_test(
#     "Isotropic",
#     bc_type=MECHANICAL_BC_TYPE,
# )

df_cpl_iso, ref_cpl_iso = run_case("Isotropic", bc_type=MECHANICAL_BC_TYPE)
df_cpl_ortho, ref_cpl_ortho = run_case("Orthotropic", bc_type=MECHANICAL_BC_TYPE)

df_cpl_all = pd.concat([df_cpl_iso, df_cpl_ortho], ignore_index=True)
df_cpl_valid = valid_converged_coupled_rows(df_cpl_all)

print_finest_converged_table(
    df_cpl_valid,
    title="FINEST-MESH CONVERGED COUPLED ROWS FOR MANUSCRIPT TABLE",
)

plot_paper2_coupled_convergence_iso_ortho(
    df_cpl_valid,
    ycol="L2_RMS_Error_w_m",
    ylabel=r"$L^2$ RMS error in $w_h$ [m]",
    x_mode="h",
    filename_base=f"paper2_isotropic_orthotropic_simply_supported_coupled_w_convergence_mesh_size_{STUDY_TAG}",
)

histories_for_fig4 = [
    history_dataframe_from_ref(ref_cpl_iso, "Isotropic"),
    history_dataframe_from_ref(ref_cpl_ortho, "Orthotropic"),
]

plot_paper2_coupled_history_iso_ortho_combined(
    histories_for_fig4,
    filename_base=f"paper2_isotropic_orthotropic_simply_supported_coupled_reference_update_history_{STUDY_TAG}",
)

if RUN_REFERENCE_COMPARISON:
    ref_coarse_cpl, ref_reference_cpl, ref_compare = compare_two_reference_levels(
        "coupled",
        PLATE_CASES["Isotropic"]["mu_mech"],
        size_a=REFERENCE_COMPARISON_COARSE_SIZE,
        size_b=REF_SIZE,
        p=REF_P,
        label=f"Isotropic_coupled_{REFERENCE_COMPARISON_COARSE_SIZE}_vs_{REF_SIZE}_{MECHANICAL_BC_TYPE}",
        bc_type=MECHANICAL_BC_TYPE,
        fixed_heat=True,
        cache_label_a=f"Isotropic_coupled_refcheck_{REFERENCE_COMPARISON_COARSE_SIZE}_{MECHANICAL_BC_TYPE}",
        cache_label_b="Isotropic",
    )

# %% Cell 47 | id: 6a152f30-99a6-4253-b4ee-57c0b99c2aad


# %% Cell 48 | id: fb9c251c-d0cd-4a78-806f-bf9ef1c6764c


# %% Cell 49 | id: 3cfe4f7f-4e08-4f0f-895b-11bc4aa37f4c


# %% Cell 50 | id: 30f2ea05-b7f5-4a94-a726-a3e6b0834086


# %% Cell 51 | id: 644f0661-0b64-41c2-8e07-fed4932be133
# ============================================================
# Paper 2 one-way vs coupled FOM field comparison
# Cached matched solves + independent robust colorbars
# White HDPE isotropic outdoor flooring panel, Trieste summer noon
# ============================================================
# ------------------------------------------------------------
# Default manuscript benchmark data
# ------------------------------------------------------------

FIELD_OUTDIR = "Figures/FOM"
FIELD_CACHE_DIR = os.path.join("paper2_fom_convergence", "field_comparison_cache")
os.makedirs(FIELD_OUTDIR, exist_ok=True)
os.makedirs(FIELD_CACHE_DIR, exist_ok=True)

FIELD_RESUME_RUNS = True
FIELD_FORCE_RERUN = False

# Increased because the physical benchmark parameters are changed.
FIELD_CACHE_VERSION = 3

MECHANICAL_BC_TYPE = "free_edge"
MECHANICAL_LOAD_TYPE = "patch"

# Solar input convention used here:
# q_sol = incident solar irradiance.
# q_s   = absorbed solar heat flux used by the thermal solver.
Q_SOL_TRIESTE_SUMMER_NOON = 850.0
ALPHA_SOL_WHITE_HDPE = 0.30

FIELD_CASES = {
    # Original generic cases retained.
    "isotropic":   [1.0e4, 1.0e4, 0.30e4, 0.35e4, 1.0e6, -6000.0],
    "orthotropic": [2.0e4, 1.0e4, 0.25e4, 0.30e4, 1.0e6, -6000.0],

    # White HDPE, approximately homogeneous isotropic plate.
    # Effective bending values chosen for a practical outdoor flooring panel.
    "white_hdpe_isotropic": [
        6.0e3,     # Dx
        6.0e3,     # Dy
        2.5e3,     # Dxy
        1.8e3,     # Ds
        5.0e5,     # ks
        -3000.0,   # f
    ],

    # Retained as an optional effective model if the panel is ribbed/cellular.
    "white_hdpe_orthotropic_effective": [
        9.0e3,     # Dx
        6.0e3,     # Dy
        2.2e3,     # Dxy
        1.7e3,     # Ds
        5.0e5,     # ks
        -3000.0,   # f
    ],
}

FIELD_CASE_LABEL = "white_hdpe_isotropic"
MU_MECH_FIELD = FIELD_CASES[FIELD_CASE_LABEL]

MU_TH_FIELD = dict(
    # Trieste warm summer noon.
    T_amb=305.15,          # 32 °C air
    T_sub=306.15,          # 33 °C warm ground/substrate

    # Outdoor convection and radiation.
    h_con=10.0,
    eps_r=0.90,

    # Absorbed solar heat flux for white HDPE:
    # q_s = alpha_sol * q_sol = 0.30 * 850 = 255 W/m^2.
    q_s=ALPHA_SOL_WHITE_HDPE * Q_SOL_TRIESTE_SUMMER_NOON,

    # Contact/air-gap heat-transfer model.
    h_c_cont=150.0,
    h_c_gap=7.0,
    eta_c=20.0,
    w_contact=-0.002,

    # HDPE thermal properties.
    kx=0.40,
    ky=0.40,
    kz=0.40,
    alpha1=1.3e-4,
    alpha2=1.3e-4,
    rho=950.0,
)


# ------------------------------------------------------------
# Cache helpers
# ------------------------------------------------------------

def _field_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _field_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_field_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _field_safe_filename_part(text):
    text = str(text)
    for ch in [" ", "/", "\\", ":", ";", ",", "(", ")", "[", "]", "{", "}"]:
        text = text.replace(ch, "_")
    return text


def _copy_function(f):
    out = Function(f.function_space())
    out.assign(f)
    return out


def _field_cache_key(payload):
    payload_json = json.dumps(_field_json_safe(payload), sort_keys=True, indent=2)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:16]


def _field_cache_paths(payload):
    key = _field_cache_key(payload)
    stem = (
        f"paper2_field_{_field_safe_filename_part(payload['case_label'])}"
        f"_{_field_safe_filename_part(payload['bc_type'])}"
        f"_{_field_safe_filename_part(payload['load_type'])}"
        f"_p{int(payload['plate_degree'])}_{key}"
    )
    return dict(
        h5=os.path.join(FIELD_CACHE_DIR, stem + ".h5"),
        json=os.path.join(FIELD_CACHE_DIR, stem + ".json"),
    )


def _field_payload(
    *,
    case_label, mu_mech, mu_th, study_case, plate_degree, load_type, bc_type,
    n_vert, n_horiz, heat_nx, heat_ny, heat_nz, heat_degree, Nz_quad_T1,
    T1_cg_degree, coupling_omega, coupling_tol_w, coupling_tol_T1, coupling_max_iters,
):
    return dict(
        cache_version=int(FIELD_CACHE_VERSION),
        case_label=str(case_label),
        mu_mech=[float(v) for v in mu_mech],
        mu_th=_field_json_safe(mu_th),
        study_case=int(study_case),
        plate_degree=int(plate_degree),
        load_type=str(load_type),
        bc_type=str(bc_type),
        n_vert=int(n_vert),
        n_horiz=int(n_horiz),
        heat_nx=int(heat_nx),
        heat_ny=int(heat_ny),
        heat_nz=int(heat_nz),
        heat_degree=int(heat_degree),
        Nz_quad_T1=int(Nz_quad_T1),
        T1_cg_degree=int(T1_cg_degree),
        coupling_omega=float(coupling_omega),
        coupling_tol_w=float(coupling_tol_w),
        coupling_tol_T1=float(coupling_tol_T1),
        coupling_max_iters=int(coupling_max_iters),
    )


def save_field_comparison_cache(result, *, payload, paths):
    """Save matched one-way/coupled field solutions and metadata."""
    mesh_ow = result["w_one_way"].function_space().mesh()
    mesh_cpl = result["w_coupled"].function_space().mesh()

    h5 = HDF5File(mesh_ow.mpi_comm(), paths["h5"], "w")
    for name, obj in [
        ("/mesh_one_way", mesh_ow),
        ("/mesh_coupled", mesh_cpl),
        ("/w_one_way", result["w_one_way"]),
        ("/theta_one_way", result["theta_one_way"]),
        ("/w_coupled", result["w_coupled"]),
        ("/theta_coupled", result["theta_coupled"]),
    ]:
        h5.write(obj, name)
    h5.close()

    cpl = result.get("coupled_output", {})
    meta = dict(
        payload=payload,
        case_label=result.get("case_label", payload["case_label"]),
        bc_type=result.get("bc_type", payload["bc_type"]),
        load_type=result.get("load_type", payload["load_type"]),
        Lx=float(result.get("Lx", result["coupled_solver"].length)),
        Ly=float(result.get("Ly", result["coupled_solver"].width)),
        plate_degree=int(payload["plate_degree"]),
        T1_cg_degree=int(payload["T1_cg_degree"]),
        coupled_history=cpl.get("history", None) if isinstance(cpl, dict) else None,
        coupled_converged=bool(cpl.get("converged", True)) if isinstance(cpl, dict) else True,
        coupled_iters=int(cpl.get("iters", -1)) if isinstance(cpl, dict) else -1,
    )

    with open(paths["json"], "w") as f:
        json.dump(_field_json_safe(meta), f, indent=2, sort_keys=True)

    print("Saved field-comparison cache:")
    print(f"  {paths['h5']}")
    print(f"  {paths['json']}")


def load_field_comparison_cache(*, paths):
    """Load cached matched one-way/coupled fields. No FOM solve is executed."""
    if not (os.path.exists(paths["h5"]) and os.path.exists(paths["json"])):
        return None

    try:
        with open(paths["json"], "r") as f:
            meta = json.load(f)

        p, pT = int(meta["plate_degree"]), int(meta["T1_cg_degree"])
        mesh_ow, mesh_cpl = Mesh(), Mesh()

        h5 = HDF5File(MPI.comm_world, paths["h5"], "r")
        h5.read(mesh_ow, "/mesh_one_way", False)
        h5.read(mesh_cpl, "/mesh_coupled", False)

        Vw_ow, Vt_ow = FunctionSpace(mesh_ow, "CG", p), FunctionSpace(mesh_ow, "CG", pT)
        Vw_cpl, Vt_cpl = FunctionSpace(mesh_cpl, "CG", p), FunctionSpace(mesh_cpl, "CG", pT)

        w_ow, th_ow = Function(Vw_ow), Function(Vt_ow)
        w_cpl, th_cpl = Function(Vw_cpl), Function(Vt_cpl)

        h5.read(w_ow, "/w_one_way")
        h5.read(th_ow, "/theta_one_way")
        h5.read(w_cpl, "/w_coupled")
        h5.read(th_cpl, "/theta_coupled")
        h5.close()

    except Exception as exc:
        print(f"Field-comparison cache could not be loaded and will be rebuilt: {exc}")
        return None

    for fld in (w_ow, th_ow, w_cpl, th_cpl):
        try:
            fld.set_allow_extrapolation(True)
        except Exception:
            pass

    print("Loaded cached field-comparison result:")
    print(f"  {paths['h5']}")

    return dict(
        case_label=meta.get("case_label"),
        bc_type=meta.get("bc_type"),
        load_type=meta.get("load_type"),
        Lx=float(meta.get("Lx", 1.0)),
        Ly=float(meta.get("Ly", 1.0)),
        one_way_solver=None,
        coupled_solver=None,
        heat_one_way=None,
        coupled_output=dict(
            history=meta.get("coupled_history", None),
            converged=bool(meta.get("coupled_converged", True)),
            iters=int(meta.get("coupled_iters", -1)),
        ),
        w_one_way=w_ow,
        w_coupled=w_cpl,
        theta_one_way=th_ow,
        theta_coupled=th_cpl,
        source_cache=os.path.basename(paths["h5"]),
    )


# ------------------------------------------------------------
# Matched one-way and coupled FOM solves, with cache
# ------------------------------------------------------------

def run_fom_oneway_coupled_for_field_plot(
    *,
    mu_mech,
    mu_th,
    study_case=1,
    plate_degree=3,
    load_type=MECHANICAL_LOAD_TYPE,
    bc_type=MECHANICAL_BC_TYPE,
    n_vert=0,
    n_horiz=0,
    heat_nx=80,
    heat_ny=40,
    heat_nz=20,
    heat_degree=1,
    Nz_quad_T1=24,
    T1_cg_degree=1,
    coupling_omega=0.7,
    coupling_tol_w=1e-5,
    coupling_tol_T1=1e-5,
    coupling_max_iters=25,
    plot_subdomains=False,
    case_label=FIELD_CASE_LABEL,
):
    """
    Compute or load matched one-way and coupled FOM solutions for the field
    comparison. If the cache exists and FIELD_RESUME_RUNS=True, no FOM solve is run.
    """
    payload = _field_payload(
        case_label=case_label,
        mu_mech=mu_mech,
        mu_th=mu_th,
        study_case=study_case,
        plate_degree=plate_degree,
        load_type=load_type,
        bc_type=bc_type,
        n_vert=n_vert,
        n_horiz=n_horiz,
        heat_nx=heat_nx,
        heat_ny=heat_ny,
        heat_nz=heat_nz,
        heat_degree=heat_degree,
        Nz_quad_T1=Nz_quad_T1,
        T1_cg_degree=T1_cg_degree,
        coupling_omega=coupling_omega,
        coupling_tol_w=coupling_tol_w,
        coupling_tol_T1=coupling_tol_T1,
        coupling_max_iters=coupling_max_iters,
    )
    paths = _field_cache_paths(payload)

    if FIELD_RESUME_RUNS and not FIELD_FORCE_RERUN:
        cached = load_field_comparison_cache(paths=paths)
        if cached is not None:
            return cached

    def new_solver():
        solver = GeneralMultiphysicsSolver(study_case=study_case)
        solver.degree = int(plate_degree)
        solver.load_type = load_type
        solver.bc_type = bc_type
        solver.define_domain(n_vert=n_vert, n_horiz=n_horiz, plot_subdomains=plot_subdomains)
        solver.set_rom_thermal_parameters(**mu_th)
        return solver

    heat_kwargs = dict(
        heat_nx=int(heat_nx),
        heat_ny=int(heat_ny),
        heat_nz=int(heat_nz),
        heat_degree=int(heat_degree),
        Nz_quad_T1=int(Nz_quad_T1),
        T1_cg_degree=int(T1_cg_degree),
    )

    solver_ow = new_solver()
    heat_ow = solver_ow.solve_rom_sample(
        mu_mech,
        thermal_on=True,
        coupled_on=False,
        return_mode="global",
        **heat_kwargs,
    )
    w_ow = _copy_function(solver_ow.w_contact_global)
    theta_ow = _copy_function(solver_ow.T1_from_heat)

    solver_cpl = new_solver()
    out_cpl = solver_cpl.solve_coupled_thermo_mechanical(
        mu_mech=mu_mech,
        use_rom_thermal_parameters=True,
        alpha1=solver_cpl.alpha1_rom,
        alpha2=solver_cpl.alpha2_rom,
        w0=0.0,
        omega=float(coupling_omega),
        tol_w=float(coupling_tol_w),
        tol_T1=float(coupling_tol_T1),
        max_coupling_iters=int(coupling_max_iters),
        store_each_iter=True,
        verbose=False,
        convergence_plot=False,
        **heat_kwargs,
    )

    w_cpl = _copy_function(out_cpl["w"])
    theta_cpl = _copy_function(out_cpl["T1"])
    solver_cpl.w_contact_global = _copy_function(w_cpl)

    result = dict(
        case_label=case_label,
        bc_type=bc_type,
        load_type=load_type,
        Lx=float(solver_cpl.length),
        Ly=float(solver_cpl.width),
        one_way_solver=solver_ow,
        coupled_solver=solver_cpl,
        heat_one_way=heat_ow,
        coupled_output=out_cpl,
        w_one_way=w_ow,
        w_coupled=w_cpl,
        theta_one_way=theta_ow,
        theta_coupled=theta_cpl,
    )

    save_field_comparison_cache(result, payload=payload, paths=paths)
    return result


# ------------------------------------------------------------
# Field plotting utilities
# ------------------------------------------------------------

def _use_paper2_plot_style():
    try:
        plt.style.use(["science", "ieee", "notebook", "grid"])
    except Exception:
        pass
    plt.rcParams["figure.autolayout"] = False


def _save_dual(fig_obj, base_path, pad_inches=0.04):
    fig_obj.savefig(
        f"{base_path}.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=pad_inches,
        facecolor="white",
    )
    fig_obj.savefig(
        f"{base_path}.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=pad_inches,
        facecolor="white",
    )


def plot_oneway_coupled_fom_fields(
    result,
    *,
    output_dir=FIELD_OUTDIR,
    filename="paper2_oneway_coupled_field_comparison",
    nx_plot=320,
    ny_plot=180,
    project_degree=3,
    use_mm_for_w=True,
    field_cmap="viridis",
    diff_cmap="plasma",
    save_individual_rows=True,
):
    """
    Reordered 3 x 2 figure with independent robust colorbars.

    Column 1:
        row 1: one-way displacement
        row 2: coupled displacement
        row 3: displacement difference

    Column 2:
        row 1: one-way thermal driver
        row 2: coupled thermal driver
        row 3: thermal-driver difference
    """

    os.makedirs(output_dir, exist_ok=True)
    _use_paper2_plot_style()

    solver_ref = result.get("coupled_solver", None)
    Lx = float(result.get("Lx", getattr(solver_ref, "length", 1.0)))
    Ly = float(result.get("Ly", getattr(solver_ref, "width", 1.0)))

    w_ow, w_cpl = result["w_one_way"], result["w_coupled"]
    th_ow, th_cpl = result["theta_one_way"], result["theta_coupled"]

    for fld in (w_ow, w_cpl, th_ow, th_cpl):
        try:
            fld.set_allow_extrapolation(True)
        except Exception:
            pass

    def smooth(field, degree=project_degree):
        Vp = FunctionSpace(field.function_space().mesh(), "CG", int(degree))
        out = project(field, Vp)
        try:
            out.set_allow_extrapolation(True)
        except Exception:
            pass
        return out

    w_ow_p, w_cpl_p, th_ow_p, th_cpl_p = map(smooth, (w_ow, w_cpl, th_ow, th_cpl))

    xg = np.linspace(0.0, Lx, int(nx_plot))
    yg = np.linspace(0.0, Ly, int(ny_plot))
    Xg, Yg = np.meshgrid(xg, yg)

    def field_on_grid(field):
        Z = np.zeros_like(Xg, dtype=float)
        for j in range(Yg.shape[0]):
            for i in range(Xg.shape[1]):
                Z[j, i] = float(field(Point(float(Xg[j, i]), float(Yg[j, i]))))
        return Z

    W_ow, W_cpl = field_on_grid(w_ow_p), field_on_grid(w_cpl_p)
    Th_ow, Th_cpl = field_on_grid(th_ow_p), field_on_grid(th_cpl_p)

    if use_mm_for_w:
        W_ow, W_cpl = 1.0e3 * W_ow, 1.0e3 * W_cpl
        w_unit, w_unit_plain = r"$[\mathrm{mm}]$", "mm"
    else:
        w_unit, w_unit_plain = r"$[\mathrm{m}]$", "m"

    dW, dTh = W_cpl - W_ow, Th_cpl - Th_ow

    # --------------------------------------------------------
    # Independent, finite, ordered, nonzero-width color limits
    # --------------------------------------------------------

    def finite_values(Z):
        vals = np.asarray(Z, dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        return vals if len(vals) else np.array([0.0], dtype=float)

    def safe_limits(vmin, vmax, *, eps=1.0e-12):
        vmin, vmax = float(vmin), float(vmax)
        if not (np.isfinite(vmin) and np.isfinite(vmax)):
            return -1.0, 1.0
        lo, hi = (vmin, vmax) if vmin <= vmax else (vmax, vmin)
        if not hi > lo:
            pad = max(abs(lo), 1.0) * eps
            lo, hi = lo - pad, hi + pad
        return float(lo), float(hi)

    def limits_field(Z):
        vals = finite_values(Z)
        return safe_limits(np.min(vals), np.max(vals))

    def limits_signed(D):
        vals = finite_values(D)
        m = max(float(np.max(np.abs(vals))), 1.0e-14)
        return -m, m

    def contour_levels(vmin, vmax, n=181):
        vmin, vmax = safe_limits(vmin, vmax)
        levels = np.linspace(vmin, vmax, int(n))
        if len(levels) < 2 or np.min(np.diff(levels)) <= 0.0:
            pad = max(abs(vmin), 1.0) * 1.0e-12
            vmin, vmax = vmin - pad, vmin + pad
            levels = np.linspace(vmin, vmax, int(n))
        return vmin, vmax, levels

    W_ow_lim = limits_field(W_ow)
    W_cpl_lim = limits_field(W_cpl)
    Th_ow_lim = limits_field(Th_ow)
    Th_cpl_lim = limits_field(Th_cpl)
    dW_lim = limits_signed(dW)
    dTh_lim = limits_signed(dTh)

    rms_w = float(np.sqrt(np.mean(dW**2)))
    rel_rms_w = float(rms_w / max(np.sqrt(np.mean(W_cpl**2)), 1.0e-14))
    max_w = float(np.max(np.abs(dW)))
    rms_th = float(np.sqrt(np.mean(dTh**2)))
    rel_rms_th = float(rms_th / max(np.sqrt(np.mean(Th_cpl**2)), 1.0e-14))
    max_th = float(np.max(np.abs(dTh)))

    def make_ticks(vmin, vmax, positive_only=False, signed=False):
        vmin, vmax = safe_limits(vmin, vmax)
        if signed:
            m = max(abs(vmin), abs(vmax), 1.0e-14)
            return [-m, -0.5 * m, 0.0, 0.5 * m, m]
        if positive_only:
            return [0.0, vmax / 3.0, 2.0 * vmax / 3.0, vmax]
        return np.linspace(vmin, vmax, 4).tolist()

    def scaled_cbar_info(vmin, vmax, positive_only=False, signed=False, decimals=1):
        vmin, vmax = safe_limits(vmin, vmax)
        ticks = make_ticks(vmin, vmax, positive_only=positive_only, signed=signed)
        ref = max(abs(float(vmin)), abs(float(vmax)))
        exponent = 0 if ref < 1.0e-14 else int(np.floor(np.log10(ref)))
        scale = 1.0 if ref < 1.0e-14 else 10.0 ** exponent
        labels = [f"{t / scale:.{decimals}f}" for t in ticks]
        return ticks, labels, exponent

    def style_map_axis(ax, xlabel=True, ylabel=True):
        ax.set_facecolor("white")
        ax.set_aspect("equal")
        ax.set_xlim(0.0, Lx)
        ax.set_ylim(0.0, Ly)
        ax.set_xlabel("x [m]" if xlabel else "", fontsize=14)
        ax.set_ylabel("y [m]" if ylabel else "", fontsize=14)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: "" if np.isclose(x, 0.0) else f"{x:.1f}"))
        ax.tick_params(axis="x", labelsize=13, length=3, pad=1)
        ax.tick_params(axis="y", labelsize=13, length=3, pad=3)
        ax.grid(False)

    def draw_scalar(
        fig_obj, ax, Z, title, vmin, vmax, *,
        cmap, signed=False, positive_only=False, xlabel=True, ylabel=True,
    ):
        vmin, vmax, levels = contour_levels(vmin, vmax)

        mappable = ax.contourf(
            Xg,
            Yg,
            Z,
            levels=levels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        style_map_axis(ax, xlabel=xlabel, ylabel=ylabel)
        ax.set_title(title, fontsize=14, pad=8)

        ticks, ticklabels, exponent = scaled_cbar_info(
            vmin,
            vmax,
            positive_only=positive_only,
            signed=signed,
            decimals=1,
        )

        cax = make_axes_locatable(ax).append_axes("right", size="4.2%", pad=0.10)
        cbar = fig_obj.colorbar(mappable, cax=cax)
        cbar.set_ticks(ticks)
        cbar.ax.yaxis.set_major_locator(FixedLocator(ticks))
        cbar.ax.set_yticklabels(ticklabels)
        cbar.ax.minorticks_off()
        cbar.ax.tick_params(labelsize=13, length=2, pad=2)
        cbar.outline.set_linewidth(0.6)
        cbar.solids.set_edgecolor("face")
        cbar.ax.set_title(
            rf"$\times 10^{{{exponent}}}$" if exponent != 0 else "",
            fontsize=14,
            pad=6,
        )

        return mappable

    # --------------------------------------------------------
    # Main reordered 3 x 2 figure
    # --------------------------------------------------------

    fig = plt.figure(figsize=(12.0, 10.2), facecolor="white")
    gs = GridSpec(
        3,
        2,
        figure=fig,
        width_ratios=[1.0, 1.0],
        left=0.055,
        right=0.982,
        bottom=0.060,
        top=0.965,
        hspace=0.16,
        wspace=0.29,
    )
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(3)])

    panel_specs = [
        (axes[0, 0], W_ow,  rf"one-way displacement, $w_h^{{\rm ow}}$ {w_unit}",                  *W_ow_lim,  field_cmap, False),
        (axes[1, 0], W_cpl, rf"coupled displacement, $w_h^{{\rm cpl}}$ {w_unit}",                  *W_cpl_lim, field_cmap, False),
        (axes[2, 0], dW,    rf"$w_h^{{\rm cpl}}-w_h^{{\rm ow}}$ {w_unit}",                         *dW_lim,    diff_cmap,  True),
        (axes[0, 1], Th_ow, r"one-way thermal driver, $\theta_h^{\rm ow}$ $[\mathrm{K\,m^{-1}}]$",  *Th_ow_lim, field_cmap, False),
        (axes[1, 1], Th_cpl,r"coupled thermal driver, $\theta_h^{\rm cpl}$ $[\mathrm{K\,m^{-1}}]$", *Th_cpl_lim,field_cmap, False),
        (axes[2, 1], dTh,   r"$\theta_h^{\rm cpl}-\theta_h^{\rm ow}$ $[\mathrm{K\,m^{-1}}]$",      *dTh_lim,   diff_cmap,  True),
    ]

    for ax, Z, ttl, vmin, vmax, cmap, signed_flag in panel_specs:
        draw_scalar(fig, ax, Z, ttl, vmin, vmax, cmap=cmap, signed=signed_flag, xlabel=True, ylabel=True)

    base = os.path.join(output_dir, filename)
    _save_dual(fig, base)
    plt.show()

    # --------------------------------------------------------
    # Optional row-wise files with the same independent scales
    # --------------------------------------------------------

    row_paths = {}

    if save_individual_rows:
        row_specs = [
            ("displacement", [
                (W_ow,  rf"one-way displacement, $w_h^{{\rm ow}}$ {w_unit}", *W_ow_lim,  field_cmap, False),
                (W_cpl, rf"coupled displacement, $w_h^{{\rm cpl}}$ {w_unit}", *W_cpl_lim, field_cmap, False),
            ]),
            ("thermal_driver", [
                (Th_ow,  r"one-way thermal driver, $\theta_h^{\rm ow}$ $[\mathrm{K\,m^{-1}}]$",  *Th_ow_lim,  field_cmap, False),
                (Th_cpl, r"coupled thermal driver, $\theta_h^{\rm cpl}$ $[\mathrm{K\,m^{-1}}]$", *Th_cpl_lim, field_cmap, False),
            ]),
            ("signed_difference", [
                (dW,  rf"$w_h^{{\rm cpl}}-w_h^{{\rm ow}}$ {w_unit}",                    *dW_lim,  diff_cmap, True),
                (dTh, r"$\theta_h^{\rm cpl}-\theta_h^{\rm ow}$ $[\mathrm{K\,m^{-1}}]$", *dTh_lim, diff_cmap, True),
            ]),
        ]

        for tag, panels in row_specs:
            fig_r = plt.figure(figsize=(12.0, 3.55), facecolor="white")
            gs_r = GridSpec(
                1,
                2,
                figure=fig_r,
                width_ratios=[1.0, 1.0],
                left=0.055,
                right=0.982,
                bottom=0.150,
                top=0.900,
                wspace=0.29,
            )
            ax_r0, ax_r1 = fig_r.add_subplot(gs_r[0, 0]), fig_r.add_subplot(gs_r[0, 1])

            for ax, (Z, ttl, vmin, vmax, cmap, signed_flag) in zip((ax_r0, ax_r1), panels):
                draw_scalar(fig_r, ax, Z, ttl, vmin, vmax, cmap=cmap, signed=signed_flag, xlabel=True, ylabel=True)

            row_base = os.path.join(output_dir, f"{filename}_{tag}")
            _save_dual(fig_r, row_base, pad_inches=0.03)
            plt.close(fig_r)
            row_paths[tag] = {"pdf": f"{row_base}.pdf", "png": f"{row_base}.png"}

    diag = dict(
        case_label=result.get("case_label", FIELD_CASE_LABEL),
        bc_type=result.get("bc_type", MECHANICAL_BC_TYPE),
        load_type=result.get("load_type", MECHANICAL_LOAD_TYPE),
        colorbar_scaling="independent_per_panel",
        W_one_way_limits=list(W_ow_lim),
        W_coupled_limits=list(W_cpl_lim),
        theta_one_way_limits=list(Th_ow_lim),
        theta_coupled_limits=list(Th_cpl_lim),
        delta_w_limits=list(dW_lim),
        delta_theta_limits=list(dTh_lim),
        max_abs_delta_w=max_w,
        rms_delta_w=rms_w,
        relative_rms_delta_w=rel_rms_w,
        max_abs_delta_theta=max_th,
        rms_delta_theta=rms_th,
        relative_rms_delta_theta=rel_rms_th,
        pdf=f"{base}.pdf",
        png=f"{base}.png",
        row_files=row_paths,
    )

    with open(f"{base}_diagnostics.json", "w") as f:
        json.dump(_field_json_safe(diag), f, indent=2, sort_keys=True)

    print("\nOne-way vs coupled field-difference diagnostics")
    print("=" * 58)
    print(f"case              : {diag['case_label']}")
    print(f"boundary condition: {diag['bc_type']}")
    print(f"load type         : {diag['load_type']}")
    print(f"colorbar scaling  : independent per panel")
    print(f"max |Delta w|     : {max_w:.6e} {w_unit_plain}")
    print(f"RMS Delta w       : {rms_w:.6e} {w_unit_plain}")
    print(f"relative RMS w    : {rel_rms_w:.6e}")
    print(f"max |Delta theta| : {max_th:.6e} K/m")
    print(f"RMS Delta theta   : {rms_th:.6e} K/m")
    print(f"relative RMS theta: {rel_rms_th:.6e}")
    print(f"diagnostics JSON  : {base}_diagnostics.json")
    print("=" * 58)

    return diag


# ------------------------------------------------------------
# Manuscript run
# ------------------------------------------------------------

field_result = run_fom_oneway_coupled_for_field_plot(
    mu_mech=MU_MECH_FIELD,
    mu_th=MU_TH_FIELD,
    study_case=1,
    plate_degree=3,
    load_type=MECHANICAL_LOAD_TYPE,
    bc_type=MECHANICAL_BC_TYPE,
    n_vert=0,
    n_horiz=0,
    heat_nx=80,
    heat_ny=40,
    heat_nz=20,
    heat_degree=1,
    Nz_quad_T1=24,
    T1_cg_degree=1,
    coupling_omega=0.7,
    coupling_tol_w=1e-5,
    coupling_tol_T1=1e-5,
    coupling_max_iters=25,
    case_label=FIELD_CASE_LABEL,
)

diag = plot_oneway_coupled_fom_fields(
    field_result,
    output_dir=FIELD_OUTDIR,
    filename=(
        f"paper2_oneway_coupled_field_comparison_"
        f"{FIELD_CASE_LABEL}_{MECHANICAL_BC_TYPE}_{MECHANICAL_LOAD_TYPE}"
    ),
    nx_plot=320,
    ny_plot=180,
    project_degree=3,
    use_mm_for_w=True,
    save_individual_rows=True,
)

# %% Cell 52 | id: 77b3e1a2-baa9-4f35-92e6-c5138853757d


# %% Cell 53 | id: a0f8123d-e6d3-48e5-8bf5-67110fcbf739


# %% Cell 54 | id: b1f2d1a2


# %% Cell 55 | id: 7f27a36b


# %% [markdown] Cell 56 | id: d7276845
# # Snapshot Pre-processing for ROM training and testing

# %% Cell 57 | id: 60ba2fc0
# %matplotlib inline  

# %% [markdown] Cell 58 | id: 5c3274a4
# ### USER CONFIGURATION — snapshot choice

# %% Cell 59 | id: 76caa5c3
P2_SNAPGEN_ROOT = Path("~/Documents/PAPER_2/SNAPGEN").expanduser()

# Available in the copied MacBook dataset:
#   monolithic : cases 1,2,3,4,5,6,7,8,9,10
#   contact1  : cases 3,5
P2_TARGET_CASE  = 2              # 1..10 physical MP/TP case id
P2_TARGET_PANEL = "monolithic"   # "monolithic" or "contact1" for current copied dataset
P2_TARGET_BC    = "free_edge"    # current copied dataset: free_edge
P2_TARGET_LOAD  = "patch"        # current copied dataset: patch

# Default ROM/FEM discretization used in the SNAPGEN campaign.
P2_PLATE_RESOLUTION = 64
P2_PLATE_DEGREE = 2
P2_HEAT_NX = 64
P2_HEAT_NY = 32
P2_HEAT_NZ = 16
P2_HEAT_DEGREE = 1
P2_THETA_CG_DEGREE = 1
P2_THETA_QUADRATURE = 20
P2_COUPLING_OMEGA = 0.7
P2_COUPLING_TOL_W = 1.0e-5
P2_COUPLING_TOL_THETA = 1.0e-5
P2_COUPLING_MAX_ITERS = 25

# The default Paper-2 ROM evaluation uses the completed stored coupled FOM
# snapshots as reference data.  This avoids accidentally calling the old
# Paper-1 mechanical-only parameterization.
P2_TEST_SOURCE = "stored_snapshots"   # recommended. Optional advanced value: "fresh_coupled"
P2_DISABLE_MECHANICAL_ONLY_INTRUSIVE_ONLINE = True

_CASE_NAMES = {
    1: "CASE_01_SUMMER_0MP1TP",
    2: "CASE_02_SUMMER_1MP1TP",
    3: "CASE_03_SUMMER_1MP2TP",
    4: "CASE_04_SUMMER_2MP2TP",
    5: "CASE_05_SUMMER_2MP3TP",
    6: "CASE_06_WINTER_0MP1TP",
    7: "CASE_07_WINTER_1MP1TP",
    8: "CASE_08_WINTER_1MP2TP",
    9: "CASE_09_WINTER_2MP2TP",
    10: "CASE_10_WINTER_2MP3TP",
}

_PANEL = {
    "monolithic": dict(n_vert=0, n_horiz=0, label="MONOLITHIC_NV0_NH0"),
    "contact1":   dict(n_vert=1, n_horiz=0, label="CONTACT_NV1_NH0"),
    "contact2":   dict(n_vert=2, n_horiz=0, label="CONTACT_NV2_NH0"),
}

P2_CASE_SNAPSHOT_COUNTS = {
    1: 100,
    2: 150,
    3: 150,
    4: 200,
    5: 200,
    6: 100,
    7: 150,
    8: 150,
    9: 200,
    10: 200,
}

# Fallback metadata if THERMOMECHANICAL_SNAPGEN_CASES.py is not importable.
# The normal path is to import the real case configs from SNAPGEN.
_P2_FALLBACK_PARAMETER_DATA = {
    1: dict(names=["q_s"], units=["W/m^2"], roles=["TP"], ranges=[(100.0, 850.0)]),
    2: dict(names=["f", "q_s"], units=["N/m^2", "W/m^2"], roles=["MP", "TP"], ranges=[(-6500.0, -1500.0), (150.0, 850.0)]),
    3: dict(names=["ks", "T_amb", "T_sub"], units=["N/m^3", "K", "K"], roles=["MP", "TP", "TP"], ranges=[(1.0e5, 2.0e6), (298.15, 313.15), (295.15, 310.15)]),
    4: dict(names=["D_ratio", "f", "q_s", "h_con"], units=["-", "N/m^2", "W/m^2", "W/(m^2 K)"], roles=["MP", "MP", "TP", "TP"], ranges=[(0.5, 2.0), (-7000.0, -2000.0), (150.0, 850.0), (5.0, 20.0)]),
    5: dict(names=["ks", "f", "q_s", "h_c_cont", "h_c_gap"], units=["N/m^3", "N/m^2", "W/m^2", "W/(m^2 K)", "W/(m^2 K)"], roles=["MP", "MP", "TP", "TP", "TP"], ranges=[(1.0e5, 3.0e6), (-9000.0, -2000.0), (150.0, 850.0), (75.0, 300.0), (3.0, 15.0)]),
    6: dict(names=["T_amb"], units=["K"], roles=["TP"], ranges=[(268.15, 285.15)]),
    7: dict(names=["f", "T_sub"], units=["N/m^2", "K"], roles=["MP", "TP"], ranges=[(-7000.0, -1500.0), (273.15, 288.15)]),
    8: dict(names=["ks", "T_amb", "T_sub"], units=["N/m^3", "K", "K"], roles=["MP", "TP", "TP"], ranges=[(1.0e5, 2.0e6), (268.15, 285.15), (273.15, 288.15)]),
    9: dict(names=["D_ratio", "f", "T_amb", "h_con"], units=["-", "N/m^2", "K", "W/(m^2 K)"], roles=["MP", "MP", "TP", "TP"], ranges=[(0.5, 2.0), (-7500.0, -2000.0), (268.15, 285.15), (5.0, 25.0)]),
    10: dict(names=["ks", "D_eff", "T_amb", "T_sub", "q_s"], units=["N/m^3", "N m", "K", "K", "W/m^2"], roles=["MP", "MP", "TP", "TP", "TP"], ranges=[(1.0e5, 3.0e6), (4000.0, 12000.0), (268.15, 285.15), (273.15, 288.15), (0.0, 250.0)]),
}


def _safe_token(value):
    txt = str(value).strip().upper()
    out = []
    for ch in txt:
        if ch.isalnum():
            out.append(ch)
        elif ch in ("_", "-"):
            out.append("_")
        else:
            out.append("_")
    token = "".join(out)
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_") or "UNNAMED"


def p2_run_name(case_id=P2_TARGET_CASE, panel=P2_TARGET_PANEL, bc=P2_TARGET_BC, load=P2_TARGET_LOAD):
    if int(case_id) not in _CASE_NAMES:
        raise ValueError(f"Unknown Project-2 case id: {case_id}")
    if panel not in _PANEL:
        raise ValueError(f"Unknown panel '{panel}'. Use one of {list(_PANEL)}.")
    return "__".join([
        _safe_token(_CASE_NAMES[int(case_id)]),
        "PANEL_" + _PANEL[panel]["label"],
        "BC_" + _safe_token(bc),
        "LOAD_" + _safe_token(load),
    ])


def _p2_add_snapgen_to_path():
    root = Path(P2_SNAPGEN_ROOT).expanduser().resolve()
    if root.exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _p2_load_case_config(case_id):
    root = _p2_add_snapgen_to_path()
    try:
        from THERMOMECHANICAL_SNAPGEN_CASES import build_case_configs
        return build_case_configs()[int(case_id)]
    except Exception as exc:
        print(f"WARNING: using fallback Project-2 case metadata because build_case_configs() could not be imported: {exc}")
        dat = _P2_FALLBACK_PARAMETER_DATA[int(case_id)]
        prm = [SimpleNamespace(name=n, lower=lo, upper=hi, unit=u, role=r, scale="linear")
               for n, u, r, (lo, hi) in zip(dat["names"], dat["units"], dat["roles"], dat["ranges"])]
        return SimpleNamespace(
            case_id=int(case_id),
            case_name=_CASE_NAMES[int(case_id)],
            season=("summer" if int(case_id) <= 5 else "winter"),
            parameter_ranges=prm,
            nominal_mech=dict(Dx=6.0e3, Dy=6.0e3, Dxy=2.5e3, Ds=1.8e3, ks=5.0e5, f=-3.0e3),
            nominal_thermal=(
                dict(T_amb=305.15, T_sub=306.15, h_con=10.0, eps_r=0.90, q_s=255.0, h_c_cont=150.0, h_c_gap=7.0,
                     eta_c=20.0, kx=0.40, ky=0.40, kz=0.40, alpha1=1.3e-4, alpha2=1.3e-4, rho=950.0)
                if int(case_id) <= 5 else
                dict(T_amb=279.15, T_sub=283.15, h_con=12.0, eps_r=0.90, q_s=50.0, h_c_cont=150.0, h_c_gap=7.0,
                     eta_c=20.0, kx=0.40, ky=0.40, kz=0.40, alpha1=1.3e-4, alpha2=1.3e-4, rho=950.0)
            )
        )


def _p2_decode_active_sample(config, sample):
    """Decode a Project-2 active parameter vector into full mu_mech and mu_th."""
    try:
        from THERMOMECHANICAL_SNAPGEN_COMMON import decode_sample
        return decode_sample(config, sample)
    except Exception:
        mech = dict(config.nominal_mech)
        therm = dict(config.nominal_thermal)
        sampled = {pr.name: float(v) for pr, v in zip(config.parameter_ranges, sample)}
        for name, val in sampled.items():
            if name in ("Dx", "Dy", "Dxy", "Ds", "ks", "f"):
                mech[name] = val
            elif name == "rho":
                therm["rho"] = val
            elif name in therm:
                therm[name] = val
        if "D_eff" in sampled or "D_ratio" in sampled:
            D_eff_nom = math.sqrt(float(mech["Dx"]) * float(mech["Dy"]))
            ratio_nom = float(mech["Dx"]) / float(mech["Dy"])
            D_eff = float(sampled.get("D_eff", D_eff_nom))
            D_ratio = float(sampled.get("D_ratio", ratio_nom))
            mech["Dx"] = D_eff * math.sqrt(D_ratio)
            mech["Dy"] = D_eff / math.sqrt(D_ratio)
        mu_mech = [float(mech[k]) for k in ["Dx", "Dy", "Dxy", "Ds", "ks", "f"]]
        return mu_mech, therm, {
            "sampled_parameters": sampled,
            "decoded_mechanical": {k: float(mech[k]) for k in ["Dx", "Dy", "Dxy", "Ds", "ks", "f"]},
            "decoded_thermal": {k: float(therm[k]) for k in therm},
        }


def _p2_get_archive_paths():
    root = Path(P2_SNAPGEN_ROOT).expanduser().resolve()
    run = p2_run_name(P2_TARGET_CASE, P2_TARGET_PANEL, P2_TARGET_BC, P2_TARGET_LOAD)
    case_dir = root / "THERMO_SNAPSHOTS" / run
    archive = case_dir / f"SNAPSHOTS__{run}.npz"
    if not archive.exists():
        raise FileNotFoundError(
            f"Project-2 final snapshot archive not found:\n{archive}\n\n"
            "Check P2_SNAPGEN_ROOT, P2_TARGET_CASE, P2_TARGET_PANEL, P2_TARGET_BC, P2_TARGET_LOAD."
        )
    return root, run, case_dir, archive


def _p2_make_solver(n_vert, n_horiz, *, bc_type, load_type):
    solver = GeneralMultiphysicsSolver(study_case=1)  # Project-2 mechanical API uses full [Dx,Dy,Dxy,Ds,ks,f].
    solver.bc_type = str(bc_type)
    solver.load_type = str(load_type)
    solver.size = int(P2_PLATE_RESOLUTION)
    solver.degree = int(P2_PLATE_DEGREE)
    solver.define_domain(int(n_vert), int(n_horiz), plot_subdomains=False)
    solver.setup_kirchhoff_problem()
    return solver


def _p2_cg_to_solution_snapshots(cg_snapshots, solver):
    """
    Convert Project-2 stored global CG snapshots to the solver.W space used by
    intrusive/Solution-space POD blocks.

    Project-2 final archives store displacement snapshots in global V_CG.  For
    monolithic panels W and V_CG are effectively the same scalar layout.  For
    contact panels this expands the same physical global field to each mixed
    subdomain component, which is exactly what later Solution→CG checking code
    expects.
    """
    cg_snapshots = np.asarray(cg_snapshots, dtype=float)

    if solver.N_subdomains == 1:
        if solver.W.dim() == cg_snapshots.shape[1]:
            return cg_snapshots.copy()
        out = []
        u = Function(solver.V_CG)
        for row in tqdm(cg_snapshots, desc="CG→W monolithic projection"):
            u.vector()[:] = row
            w = project(u, solver.W)
            out.append(w.vector().get_local().copy())
        return np.asarray(out)

    out = []
    u_cg = Function(solver.V_CG)
    for row in tqdm(cg_snapshots, desc="CG→W contact expansion"):
        u_cg.vector()[:] = row
        w_mixed = Function(solver.W)
        w_mixed.vector().zero()
        for j, _sid in enumerate(solver.subdomain_ids):
            V_sub = solver.W.sub(j).collapse()
            comp = project(u_cg, V_sub)
            dofmap_mixed = solver.W.sub(j).dofmap()
            dofmap_sub = V_sub.dofmap()
            for dof_mixed, dof_sub in zip(dofmap_mixed.dofs(), dofmap_sub.dofs()):
                w_mixed.vector()[dof_mixed] = comp.vector()[dof_sub]
        try:
            w_mixed.vector().apply("insert")
        except Exception:
            pass
        out.append(w_mixed.vector().get_local().copy())
    return np.asarray(out)


def _p2_solution_to_cg_snapshots(solution_snapshots, solver):
    """Project solver.W snapshots to global solver.V_CG."""
    solution_snapshots = np.asarray(solution_snapshots, dtype=float)
    out = []
    w_solution = Function(solver.W)
    if solver.N_subdomains > 1:
        chi = {sid: Function(solver.V_DG, name=f"chi_{sid}") for sid in solver.subdomain_ids}
        for sid in solver.subdomain_ids:
            chi[sid].vector()[:] = (solver.subdomains.array() == sid).astype(float)
    for row in tqdm(solution_snapshots, desc="W→CG projection"):
        w_solution.vector()[:] = row
        if solver.N_subdomains > 1:
            w_global = project(sum(chi[sid] * w_solution.sub(j) for j, sid in enumerate(solver.subdomain_ids)), solver.V_CG)
        else:
            w_global = project(w_solution, solver.V_CG)
        out.append(w_global.vector().get_local().copy())
    return np.asarray(out)


def _p2_vectors_to_functions(vectors, space, prefix="f"):
    funcs = []
    for i, vec in enumerate(np.asarray(vectors, dtype=float)):
        f = Function(space, name=f"{prefix}_{i}")
        f.vector()[:] = vec
        try:
            f.vector().apply("insert")
        except Exception:
            pass
        funcs.append(f)
    return funcs


def _p2_load_wall_times(case_dir, n):
    summary = Path(case_dir) / "summary.csv"
    if summary.exists():
        try:
            df = pd.read_csv(summary)
            for col in ("wall_time_sec", "wall_time", "solve_time_sec", "time_sec"):
                if col in df.columns and len(df[col]) >= n:
                    vals = np.asarray(df[col].iloc[:n], dtype=float)
                    vals[~np.isfinite(vals)] = np.nanmedian(vals[np.isfinite(vals)]) if np.isfinite(vals).any() else 1.0
                    return vals.tolist()
        except Exception as exc:
            print(f"WARNING: could not read wall times from summary.csv: {exc}")
    return [1.0 for _ in range(n)]


def _p2_cap_basis(N_available, requested):
    try:
        return max(1, min(int(N_available), int(requested)))
    except Exception:
        return int(requested)
    
# ============================================================================
# Locate/load Project-2 final archive and configure solver
# ============================================================================
P2_CONFIG = _p2_load_case_config(P2_TARGET_CASE)
P2_SNAPGEN_ROOT, P2_RUN_NAME, P2_CASE_DIR, P2_ARCHIVE = _p2_get_archive_paths()
P2_PANEL_INFO = _PANEL[P2_TARGET_PANEL]
n_vert, n_horiz = P2_PANEL_INFO["n_vert"], P2_PANEL_INFO["n_horiz"]
bc_type, load_type = P2_TARGET_BC, P2_TARGET_LOAD

print("=" * 100)
print("PROJECT-2 NATIVE ROM DATA SETUP")
print("=" * 100)
print(f"SNAPGEN root          : {P2_SNAPGEN_ROOT}")
print(f"Selected archive      : {P2_ARCHIVE}")
print(f"Physical case         : {P2_TARGET_CASE} / {P2_CONFIG.case_name}")
print(f"Panel / BC / load     : {P2_TARGET_PANEL} / {P2_TARGET_BC} / {P2_TARGET_LOAD}")
print(f"n_vert, n_horiz       : {n_vert}, {n_horiz}")
print()

_p2_data = np.load(P2_ARCHIVE, allow_pickle=True)
_p2_archive_keys = list(_p2_data.files)
if "parameters" not in _p2_archive_keys or "snapshots" not in _p2_archive_keys:
    raise KeyError(f"Project-2 archive must contain 'parameters' and 'snapshots'. Keys found: {_p2_archive_keys}")

_p2_mu = np.asarray(_p2_data["parameters"], dtype=float)
_p2_cg_snapshots = np.asarray(_p2_data["snapshots"], dtype=float)
len_Snaps = int(_p2_mu.shape[0])

# Main selection variables used throughout the old notebook code.
CASE = int(P2_TARGET_CASE)
case = int(P2_TARGET_CASE)
VARIANT = str(P2_TARGET_PANEL)
variant = str(P2_TARGET_PANEL)

# Project-2 parameter metadata.
_p2_parameter_names = [p.name for p in P2_CONFIG.parameter_ranges]
_p2_parameter_units = [p.unit for p in P2_CONFIG.parameter_ranges]
_p2_parameter_roles = [getattr(p, "role", "") for p in P2_CONFIG.parameter_ranges]
_p2_parameter_ranges = [(float(p.lower), float(p.upper)) for p in P2_CONFIG.parameter_ranges]
parameter_ranges = {CASE: _p2_parameter_ranges}
parameter_labels = {CASE: {"names": _p2_parameter_names, "units": _p2_parameter_units, "roles": _p2_parameter_roles}}

solver = _p2_make_solver(n_vert, n_horiz, bc_type=bc_type, load_type=load_type)
# Make all ROM outputs case-specific and avoid mixing different Paper-2 cases.
P2_ROM_OUTPUT_DIR = P2_CASE_DIR / "ROM_OUTPUT"
P2_ROM_DATA_DIR = P2_CASE_DIR / "ROM_DATA"
P2_ROM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
P2_ROM_DATA_DIR.mkdir(parents=True, exist_ok=True)
solver.output_dir = str(P2_ROM_OUTPUT_DIR)

print(f"Archive keys          : {_p2_archive_keys}")
print(f"Active parameters     : {_p2_mu.shape} -> {_p2_parameter_names}")
print(f"Stored CG snapshots   : {_p2_cg_snapshots.shape}")
print(f"ROM output directory  : {solver.output_dir}")
print("=" * 100)

# %% [markdown] Cell 60 | id: f6c475dc
# ### Snapshot Projection tool

# %% Cell 61 | id: 576f3f67
# Project-2 archives already store the final displacement snapshot as global CG.
# This section creates native Project-2 ROM files, not Paper-1 compatibility files.
print("--- Project-2 Snapshot Projection Tool (native archive → Solution/CG ROM archives) ---")

snapshot_file_Solution = str(P2_ROM_DATA_DIR / f"{P2_RUN_NAME}__Solution_W.npz")
snapshot_file_proj = str(P2_ROM_DATA_DIR / f"{P2_RUN_NAME}__Projected_CG.npz")
snapshot_file_reprojected = str(P2_ROM_DATA_DIR / f"{P2_RUN_NAME}__Reprojected_Solution_W.npz")

# Solution/W archive needed by the Solution-space POD blocks.
fom_snapshots_Solution = _p2_cg_to_solution_snapshots(_p2_cg_snapshots, solver)

# Projected archive: Project-2 final snapshots are already global CG.
fom_snapshots_proj = _p2_cg_snapshots.copy()

# Reprojected Solution archive: project CG back to Solution/W representation.
fom_snapshots_reprojected = _p2_cg_to_solution_snapshots(fom_snapshots_proj, solver)

np.savez_compressed(
    snapshot_file_Solution,
    mus=_p2_mu,
    fom_snapshots=fom_snapshots_Solution,
    source_project2_archive=np.asarray(str(P2_ARCHIVE)),
    source_project2_keys=np.asarray(_p2_archive_keys, dtype=str),
    project2_run_name=np.asarray(P2_RUN_NAME),
    parameter_names=np.asarray(_p2_parameter_names, dtype=str),
    parameter_units=np.asarray(_p2_parameter_units, dtype=str),
    parameter_roles=np.asarray(_p2_parameter_roles, dtype=str),
)
np.savez_compressed(
    snapshot_file_proj,
    mus=_p2_mu,
    fom_snapshots=fom_snapshots_proj,
    source_project2_archive=np.asarray(str(P2_ARCHIVE)),
    note=np.asarray("Project-2 final displacement snapshots are already stored in global CG space."),
    parameter_names=np.asarray(_p2_parameter_names, dtype=str),
    parameter_units=np.asarray(_p2_parameter_units, dtype=str),
    parameter_roles=np.asarray(_p2_parameter_roles, dtype=str),
)
np.savez_compressed(
    snapshot_file_reprojected,
    mus=_p2_mu,
    fom_snapshots=fom_snapshots_reprojected,
    source_project2_archive=np.asarray(str(P2_ARCHIVE)),
    note=np.asarray("Reconstructed Solution/W representation from Project-2 global CG snapshots."),
    parameter_names=np.asarray(_p2_parameter_names, dtype=str),
    parameter_units=np.asarray(_p2_parameter_units, dtype=str),
    parameter_roles=np.asarray(_p2_parameter_roles, dtype=str),
)

print("Full Project-2 projection preparation complete.")
print(f"  Solution/W archive      : {snapshot_file_Solution}")
print(f"  Projected CG archive    : {snapshot_file_proj}")
print(f"  Reprojected W archive   : {snapshot_file_reprojected}")
print(f"  Solution/W shape        : {fom_snapshots_Solution.shape}")
print(f"  Projected CG shape      : {fom_snapshots_proj.shape}")
print(f"  Reprojected W shape     : {fom_snapshots_reprojected.shape}")

# %% Cell 62 | id: 8aeb4d67
# ============================================================================
# Snapshot File Check
# ============================================================================
index_to_show = min(20, len_Snaps - 1)
print("\n─ Located Project-2 ROM snapshot archives")
for tag, fn in [("Solution", snapshot_file_Solution), ("reproj", snapshot_file_reprojected), ("projCG", snapshot_file_proj)]:
    if not os.path.exists(fn):
        raise FileNotFoundError(fn)
    print(f"  {tag:<9s}: {fn}  ({os.path.getsize(fn)/1024**2:6.2f} MiB)")


def header(fn):
    ar = np.load(fn, allow_pickle=True)
    return {k: (ar[k].shape, ar[k].dtype) for k in ar.files}

print("\n─ Archive contents")
for lab, fn in [("Solution", snapshot_file_Solution), ("reproj", snapshot_file_reprojected), ("projCG", snapshot_file_proj)]:
    print(f"  {lab:<9}: {header(fn)}")

mix = np.load(snapshot_file_Solution, allow_pickle=True)
rep = np.load(snapshot_file_reprojected, allow_pickle=True)
cg = np.load(snapshot_file_proj, allow_pickle=True)
mu_all = mix["mus"]
snap_mix = mix["fom_snapshots"]
snap_rep = rep["fom_snapshots"]
snap_cg = cg["fom_snapshots"]
assert np.allclose(mu_all, rep["mus"]) and np.allclose(mu_all, cg["mus"])

print("\n─ Shapes (n_snapshots × n_dofs)")
for n, arr in [("Solution", snap_mix), ("reproj", snap_rep), ("projCG", snap_cg)]:
    print(f"  {n:<9}: {arr.shape}")

i = index_to_show
mu_i, vec_mix, vec_rep, vec_cg = mu_all[i], snap_mix[i], snap_rep[i], snap_cg[i]
print(f"\n─ Selected Project-2 snapshot {i} / {len_Snaps - 1}   active μ = {mu_i}")
print(f"  parameter names: {_p2_parameter_names}")

# Reconstruct Functions and compare CG consistency.
W = solver.W
w_raw, w_rep = Function(W), Function(W)
w_raw.vector()[:] = vec_mix
w_rep.vector()[:] = vec_rep
u_stored = Function(solver.V_CG)
u_stored.vector()[:] = vec_cg

chi = None
if solver.N_subdomains > 1:
    chi = {sid: Function(solver.V_DG) for sid in solver.subdomain_ids}
    for sid in solver.subdomain_ids:
        chi[sid].vector()[:] = (solver.subdomains.array() == sid).astype(float)


def mix_to_cg(fmix):
    if solver.N_subdomains == 1:
        return project(fmix, solver.V_CG)
    return project(sum(chi[sid] * fmix.sub(j) for j, sid in enumerate(solver.subdomain_ids)), solver.V_CG)


def physical_part(w, j, sid):
    if chi is None:
        return w.sub(j) if solver.N_subdomains > 1 else w
    return project(chi[sid] * w.sub(j), W.sub(j).collapse())

phys_raw = [physical_part(w_raw, j, sid) for j, sid in enumerate(solver.subdomain_ids)] if solver.N_subdomains > 1 else [w_raw]
phys_rep = [physical_part(w_rep, j, sid) for j, sid in enumerate(solver.subdomain_ids)] if solver.N_subdomains > 1 else [w_rep]
u_cg_raw, u_cg_rep = mix_to_cg(w_raw), mix_to_cg(w_rep)


def st(arr):
    if not isinstance(arr, np.ndarray):
        arr = arr.vector().get_local()
    return f"[{arr.min(): .3e}, {arr.max(): .3e}]  ‖·‖₂={np.linalg.norm(arr):.3e}"

print("\n─ Component-wise statistics (input Solution/W)")
if solver.N_subdomains > 1:
    for j, sid in enumerate(solver.subdomain_ids):
        print(f"  Ω{sid}: raw  {st(w_raw.sub(j))}")
        print(f"       phys {st(phys_raw[j])}")
else:
    print(f"  Ω: raw  {st(w_raw)}")
    print(f"     phys {st(phys_raw[0])}")

print("\n─ Component-wise statistics (reprojected Solution/W)")
if solver.N_subdomains > 1:
    for j, sid in enumerate(solver.subdomain_ids):
        print(f"  Ω{sid}: raw  {st(w_rep.sub(j))}")
        print(f"       phys {st(phys_rep[j])}")
else:
    print(f"  Ω: raw  {st(w_rep)}")
    print(f"     phys {st(phys_rep[0])}")

print("\n─ Global CG vectors")
print("  from raw   :", st(u_cg_raw))
print("  from repr  :", st(u_cg_rep))
print("  stored     :", st(u_stored))
print("  L2 diff raw↔stored  :", np.linalg.norm(u_cg_raw.vector() - u_stored.vector()))
print("  L2 diff rep↔stored  :", np.linalg.norm(u_cg_rep.vector() - u_stored.vector()))

# Lightweight visual sanity check for the selected snapshot.
plt.rcParams.update({"axes.titlesize": 12})

def draw(ax, f, title):
    vmax = abs(f.vector().max()) + 1e-30
    plt.sca(ax)
    p = plot(f, cmap="seismic", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(p, ax=ax, shrink=0.7)

N_subdomains = solver.N_subdomains
is_multi_subdomain = N_subdomains > 1
if is_multi_subdomain:
    rows = 5
    cols = N_subdomains + 1
    fig = plt.figure(figsize=(6 * cols, 3 * rows))
    gs = plt.GridSpec(rows, cols, figure=fig)
    for j, sid in enumerate(solver.subdomain_ids):
        draw(fig.add_subplot(gs[0, j]), w_raw.sub(j), f"RAW $w_{{{sid}}}$")
    draw(fig.add_subplot(gs[0, -1]), u_cg_raw, "RAW → CG")
    for j, sid in enumerate(solver.subdomain_ids):
        draw(fig.add_subplot(gs[1, j]), phys_raw[j], f"PHYS(raw) $w_{{{sid}}}$")
    draw(fig.add_subplot(gs[1, -1]), u_cg_raw, "PHYS(raw) → CG")
    for j, sid in enumerate(solver.subdomain_ids):
        draw(fig.add_subplot(gs[2, j]), w_rep.sub(j), f"Reproj-RAW $w_{{{sid}}}$")
    draw(fig.add_subplot(gs[2, -1]), u_cg_rep, "Reproj-RAW → CG")
    for j, sid in enumerate(solver.subdomain_ids):
        draw(fig.add_subplot(gs[3, j]), phys_rep[j], f"PHYS(rep) $w_{{{sid}}}$")
    draw(fig.add_subplot(gs[3, -1]), u_cg_rep, "PHYS(rep) → CG")
    draw(fig.add_subplot(gs[4, -1]), u_stored, "Stored CG")
else:
    rows, cols = 5, 3
    fig = plt.figure(figsize=(18, 15))
    gs = plt.GridSpec(rows, cols, figure=fig)
    draw(fig.add_subplot(gs[0, 1]), w_raw, "RAW $w$")
    draw(fig.add_subplot(gs[0, 2]), u_cg_raw, "RAW → CG")
    draw(fig.add_subplot(gs[1, 1]), phys_raw[0], "PHYS(raw) $w$")
    draw(fig.add_subplot(gs[1, 2]), u_cg_raw, "PHYS(raw) → CG")
    draw(fig.add_subplot(gs[2, 1]), w_rep, "Reproj-RAW $w$")
    draw(fig.add_subplot(gs[2, 2]), u_cg_rep, "Reproj-RAW → CG")
    draw(fig.add_subplot(gs[3, 1]), phys_rep[0], "PHYS(rep) $w$")
    draw(fig.add_subplot(gs[3, 2]), u_cg_rep, "PHYS(rep) → CG")
    draw(fig.add_subplot(gs[4, 1]), u_stored, "Stored CG")
fig.suptitle(f"Project-2 snapshot {i} – case {case} / panel '{variant}' ({N_subdomains} subdomains)\nactive μ = {mu_i}", fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# %% [markdown] Cell 63 | id: bd93df88
# # Reduced Order Model

# %% [markdown] Cell 64 | id: bfa4e0a3
#
# ## POD: Proper Orthogonal Decomposition

# %% Cell 65 | id: 157d769b
rom_dir = os.path.join(solver.output_dir, "POD")
os.makedirs(rom_dir, exist_ok=True)


def _pod_on_indices(solver, μ, snaps, idx, *, N_max=30, tol=1e-15, tag="", space_type="Solution"):
    """Run POD on a subset of snapshots."""
    if space_type == "Solution":
        pod = ProperOrthogonalDecomposition(solver.W, solver.inner_product)
        space, inner_prod = solver.W, solver.inner_product
    else:
        pod = ProperOrthogonalDecomposition(solver.V_CG, solver.inner_product_CG)
        space, inner_prod = solver.V_CG, solver.inner_product_CG

    for i in idx:
        s = Function(space)
        s.vector()[:] = snaps[i]
        pod.store_snapshot(s)

    eigvals, eigvecs, basis, N = pod.apply(min(N_max, len(idx)), tol)
    sweet_n = int(np.searchsorted(100 * np.cumsum(eigvals) / np.sum(eigvals), 99.9)) + 1

    Z = BasisFunctionsMatrix(space)
    Z.init("u")
    Z.enrich(basis)

    print(f"[POD] {tag:<9s}: {len(idx):4d} snaps → N={N}  (99.9% @ N*={sweet_n})")
    return eigvals, eigvecs, basis, N, Z, inner_prod, μ[idx], snaps[idx]


def get_rb(solver, μ, snaps, *, label=None, n_clusters=8, N_max=30, tol=1e-15, space_type="Solution"):
    if label is not None:
        km_attr = "_kmeans_cache"
        if (not hasattr(get_rb, km_attr) or getattr(get_rb, km_attr).n_clusters != n_clusters):
            setattr(get_rb, km_attr, KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(μ))
        kmeans = getattr(get_rb, km_attr)
        idx = np.where(kmeans.labels_ == label)[0]
    else:
        idx = np.arange(len(snaps))

    tag = "Global" if label is None else f"Cluster {label}"
    return _pod_on_indices(solver, μ, snaps, idx, N_max=N_max, tol=tol, tag=tag, space_type=space_type)


def _style_pod_axis(ax, *, log_y=False):
    ax.set_facecolor("white")
    if log_y:
        ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=13, length=3, pad=2)
    ax.grid(True, which="both", linestyle=":", linewidth=0.45, alpha=0.35)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    formatter_y = ScalarFormatter(useMathText=True)
    formatter_y.set_powerlimits((-2, 2))
    if not log_y:
        ax.yaxis.set_major_formatter(formatter_y)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax.yaxis.get_offset_text().set_size(12)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _style_pod_legend(ax):
    leg = ax.legend(loc="best", frameon=True, fontsize=12, borderpad=0.45,
                    handlelength=2.2, handletextpad=0.6, labelspacing=0.35)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_alpha(0.95)
    return leg


def _save_pod_figure(fig, save_name):
    base = os.path.join(rom_dir, os.path.splitext(save_name)[0])
    fig.savefig(f"{base}.pdf", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{base}.png", dpi=600, bbox_inches="tight", facecolor="white")


def plot_eigens(sets, solver, title_suffix="", n_display=10, save_name="pod_eigen_decay.png"):
    plt.rcParams["figure.autolayout"] = False
    fig, ax = plt.subplots(1, 3, figsize=(16.8, 4.8), dpi=150, facecolor="white")
    palette = (f"C{i}" for i in range(10))

    for (eig_vals, *_), label, colour in ((t, l, next(palette)) for t, l in sets):
        eig = np.asarray(eig_vals, dtype=float)
        n = np.arange(1, len(eig) + 1)
        sv = np.sqrt(eig)
        energy = 100.0 * np.cumsum(eig) / np.sum(eig)
        ax[0].plot(n, sv, "-o", c=colour, lw=2.2, ms=5.5, label=label)
        ax[1].plot(n, energy, "-o", c=colour, lw=2.2, ms=5.5, label=label)
        ax[2].plot(n, sv / sv[0], "-o", c=colour, lw=2.2, ms=5.5, label=label)
        print(f"\nPOD summary for '{label}':")
        print(f"\n{'N':>3} | {'σᵢ':>12} | {'σᵢ/σ₁':>10} | {'Energy %':>10}")
        print("-" * 50)
        for ii in range(min(n_display, len(eig))):
            print(f"{ii+1:3d} | {sv[ii]:12.4e} | {sv[ii]/sv[0]:10.4e} | {energy[ii]:10.2f}")
        if len(eig) > n_display:
            print(f"... | {'...':<12} | {'...':<10} | {'...':<10}")
            print(f"{len(eig):3d} | {sv[-1]:12.4e} | {sv[-1]/sv[0]:10.4e} | {100.00:10.2f}")

    titles = (r"Singular-value decay", r"Cumulative energy content", r"Normalized singular values")
    ylabels = (r"$\sigma_i$", r"Energy [\%]", r"$\sigma_i/\sigma_1$")
    for j, (axis, title, ylab) in enumerate(zip(ax, titles, ylabels)):
        _style_pod_axis(axis, log_y=(j in [0, 2]))
        axis.set_title(title + title_suffix, fontsize=14, pad=8)
        axis.set_xlabel(r"$N_{\mathrm{basis}}$", fontsize=14)
        axis.set_ylabel(ylab, fontsize=14)
        _style_pod_legend(axis)
    ax[1].axhline(99.9, ls="--", color="grey", lw=1.4)
    ax[1].set_ylim(0.0, 101.5)
    fig.tight_layout()
    _save_pod_figure(fig, save_name)
    plt.show()


def plot_normalized_singular_values_side_by_side(
    eig_solution,
    eig_projected,
    *,
    labels=(r"Solution space", r"Projected space"),
    save_name="pod_normalized_singular_values_side_by_side.png",
):
    plt.rcParams["figure.autolayout"] = False
    eig_solution = np.asarray(eig_solution, dtype=float)
    eig_projected = np.asarray(eig_projected, dtype=float)
    sv_solution = np.sqrt(eig_solution)
    sv_projected = np.sqrt(eig_projected)
    n_solution = np.arange(1, len(sv_solution) + 1)
    n_projected = np.arange(1, len(sv_projected) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), dpi=150, facecolor="white", sharey=False)
    plot_data = [(axes[0], n_solution, sv_solution / sv_solution[0], labels[0], "C0"),
                 (axes[1], n_projected, sv_projected / sv_projected[0], labels[1], "C0")]
    for ax, n, vals, title, color in plot_data:
        ax.plot(n, vals, "-o", color=color, lw=2.2, ms=5.5, label=title)
        _style_pod_axis(ax, log_y=True)
        ax.set_title(rf"Normalized POD singular values ({title})", fontsize=14, pad=8)
        ax.set_xlabel(r"$N_{\mathrm{basis}}$", fontsize=14)
        ax.set_ylabel(r"$\sigma_i/\sigma_1$", fontsize=14)
        _style_pod_legend(ax)
    fig.tight_layout()
    _save_pod_figure(fig, save_name)
    plt.show()

# ============================================================================
# SETUP SOLVER AND LOAD DATA — native Project-2 archives
# ============================================================================
print("\nLoading Solution/W snapshots from:", snapshot_file_Solution)
data_Solution = np.load(snapshot_file_Solution, allow_pickle=True)
mu_array_Solution = np.asarray(data_Solution["mus"], dtype=float)
fom_snapshots_Solution = np.asarray(data_Solution["fom_snapshots"], dtype=float)
print(f"Loaded {len(mu_array_Solution)} Solution/W snapshots")
print(f"  FOM snapshots shape: {fom_snapshots_Solution.shape}")

print("\nLoading projected CG snapshots from:", snapshot_file_proj)
data_proj = np.load(snapshot_file_proj, allow_pickle=True)
mu_array_proj = np.asarray(data_proj["mus"], dtype=float)
fom_snapshots_proj = np.asarray(data_proj["fom_snapshots"], dtype=float)
print(f"Loaded {len(mu_array_proj)} projected CG snapshots")
print(f"  FOM snapshots shape: {fom_snapshots_proj.shape}")

# BUILD POD BASIS FOR Solution SPACE
print("\n" + "=" * 60)
print("BUILDING POD BASIS - PROJECT-2 SOLUTION/W SPACE")
print("=" * 60)
candidates_Solution = [("rb_G_Solution", {}, 0, "Global")]
for var_name, kwargs, key, label in candidates_Solution:
    globals()[var_name] = get_rb(
        solver,
        mu_array_Solution,
        fom_snapshots_Solution,
        **kwargs,
        space_type="Solution",
    )

# BUILD POD BASIS FOR PROJECTED CG SPACE
print("\n" + "=" * 60)
print("BUILDING POD BASIS - PROJECT-2 PROJECTED CG SPACE")
print("=" * 60)
candidates_proj = [("rb_G_proj", {}, 0, "Global")]
for var_name, kwargs, key, label in candidates_proj:
    globals()[var_name] = get_rb(
        solver,
        mu_array_proj,
        fom_snapshots_proj,
        **kwargs,
        space_type="projected",
        tol=1e-12,
    )

rb_bank_Solution = {k: globals()[n] for n, _, k, _ in candidates_Solution if n in globals()}
plot_eigens(
    [(rb_bank_Solution[k], label) for _, _, k, label in candidates_Solution if k in rb_bank_Solution],
    solver,
    title_suffix=" (Project-2 Solution/W space)",
    save_name="pod_eigen_decay_Solution.png",
)

rb_bank_proj = {k: globals()[n] for n, _, k, _ in candidates_proj if n in globals()}
plot_eigens(
    [(rb_bank_proj[k], label) for _, _, k, label in candidates_proj if k in rb_bank_proj],
    solver,
    title_suffix=" (Project-2 projected CG space)",
    save_name="pod_eigen_decay_projected.png",
)

plot_normalized_singular_values_side_by_side(
    rb_bank_Solution[0][0],
    rb_bank_proj[0][0],
    labels=(r"Solution/W space", r"Projected CG space"),
    save_name="pod_normalized_singular_values_side_by_side.png",
)

# PICK REDUCED BASIS CHOICES - Solution SPACE
choice_podg, n_basis_podg = 0, None
choice_podlspg, n_basis_podlspg = 0, None
eigG, eigvG, basisG, N_podg, Z_podg, IP_G, mu_all_G, snaps_all_G = rb_bank_Solution[choice_podg]
N_podg = n_basis_podg or N_podg
eigL, eigvL, basisL, N_podlspg, Z_podlspg, IP_L, mu_all_L, snaps_all_L = rb_bank_Solution[choice_podlspg]
N_podlspg = n_basis_podlspg or N_podlspg

# PICK REDUCED BASIS CHOICES - PROJECTED SPACE
choice_podnn, choice_podI, n_basis_podnn, n_basis_podI = 0, 0, None, None
eigN, eigvN, basisN, N_podnn, Z_podnn, IP_N, mu_all_N, snaps_all_N = rb_bank_proj[choice_podnn]
N_podnn = n_basis_podnn or N_podnn
eigI, eigvI, basisI, N_podI, Z_podI, IP_I, mu_all_I, snaps_all_I = rb_bank_proj[choice_podI]
N_podI = n_basis_podI or N_podI

print(f"\nPOD basis selected (Solution/W Space):")
print(f"  POD-Galerkin/Projection basis: choice={choice_podg}, N_basis={N_podg}, n_snaps={len(snaps_all_G)}")
print(f"  POD-LSPG basis:                choice={choice_podlspg}, N_basis={N_podlspg}, n_snaps={len(snaps_all_L)}")
print(f"\nPOD basis selected (Projected CG Space):")
print(f"  PODNN: choice={choice_podnn}, N_basis={N_podnn}, n_snaps={len(snaps_all_N)}")
print(f"  PODI:  choice={choice_podI},  N_basis={N_podI},  n_snaps={len(snaps_all_I)}")

print(f"\nProject-2 Data Analysis:")
print(f"  - Physical case name             : {P2_CONFIG.case_name}")
print(f"  - Active parameter names         : {_p2_parameter_names}")
print(f"  - Parameter space dimension      : {mu_array_proj.shape[1]}")
print(f"  - Solution/W DOF dimension       : {fom_snapshots_Solution.shape[1]}")
print(f"  - Projected CG DOF dimension     : {fom_snapshots_proj.shape[1]}")
print(f"  - Number of snapshots            : {len_Snaps}")

# %% Cell 66 | id: 37b879a3
# ---------------------------------------------------------------------------
# User basis choice
# ---------------------------------------------------------------------------
choice_podg, choice_podlspg = 0, 0
n_basis_podg = _p2_cap_basis(N_podg, 8)
n_basis_podlspg = _p2_cap_basis(N_podlspg, 8)
eigG, eigvG, basisG, N_podg, Z_podg, IP_G, mu_all_G, snaps_all_G = rb_bank_Solution[choice_podg]
N_podg = n_basis_podg or N_podg
eigL, eigvL, basisL, N_podlspg, Z_podlspg, IP_L, mu_all_L, snaps_all_L = rb_bank_Solution[choice_podlspg]
N_podlspg = n_basis_podlspg or N_podlspg

choice_podnn, choice_podI = 0, 0
n_basis_podnn = _p2_cap_basis(N_podnn, 8)
n_basis_podI = _p2_cap_basis(N_podI, 8)
eigN, eigvN, basisN, N_podnn, Z_podnn, IP_N, mu_all_N, snaps_all_N = rb_bank_proj[choice_podnn]
N_podnn = n_basis_podnn or N_podnn
eigI, eigvI, basisI, N_podI, Z_podI, IP_I, mu_all_I, snaps_all_I = rb_bank_proj[choice_podI]
N_podI = n_basis_podI or N_podI

# %% [markdown] Cell 67 | id: a8f25584
# ## Test FOM - Solution and Global Spaces - Error Analysis

# %% Cell 68 | id: 81f6468c
# ============================================================================
# Test FOM - Solution and Global Spaces - Error Analysis
# ============================================================================
# Project-2 choices:
#   P2_TEST_SOURCE = "stored_snapshots" : use completed coupled SNAPGEN FOM snapshots.
#   P2_TEST_SOURCE = "fresh_coupled"    : generate/load fresh coupled thermo-mechanical FOM test solves.
#
# For Paper 2, active μ may contain mixed MP/TP entries, e.g.
# [ks, f, q_s, h_c_cont, h_c_gap]. Therefore fresh solves must decode:
# active μ -> full mechanical μ + thermal dictionary -> coupled thermo-mechanical solver.
# ============================================================================

print("\n" + "=" * 80)
print("PROJECT-2 FOM REFERENCE DATA FOR ROM ERROR ANALYSIS")
print("=" * 80)

# ---------------------------------------------------------------------------
# User controls
# ---------------------------------------------------------------------------
P2_TEST_SOURCE = globals().get("P2_TEST_SOURCE", "stored_snapshots")

# Used only for P2_TEST_SOURCE == "fresh_coupled".
# test_set: 0 = inner random 30%-70%, 1 = lower/upper corner grid, 2 = full-range LHS.
P2_FRESH_TEST_SET = globals().get("P2_FRESH_TEST_SET", 0)
P2_FRESH_N_SAMPLES = globals().get("P2_FRESH_N_SAMPLES", 20)
P2_FRESH_TEST_SEED = globals().get("P2_FRESH_TEST_SEED", 42)
P2_FRESH_INNER_FRACTION = globals().get("P2_FRESH_INNER_FRACTION", (0.30, 0.70))

_ALLOWED_P2_TEST_SOURCES = {"stored_snapshots", "fresh_coupled"}
if P2_TEST_SOURCE not in _ALLOWED_P2_TEST_SOURCES:
    raise ValueError(f"Unknown P2_TEST_SOURCE={P2_TEST_SOURCE!r}. Use one of {_ALLOWED_P2_TEST_SOURCES}.")

print(f"P2_TEST_SOURCE       = {P2_TEST_SOURCE}")
print(f"P2_TARGET_CASE       = {P2_TARGET_CASE} / {P2_CONFIG.case_name}")
print(f"P2_TARGET_PANEL      = {P2_TARGET_PANEL}")
print(f"P2_TARGET_BC         = {P2_TARGET_BC}")
print(f"P2_TARGET_LOAD       = {P2_TARGET_LOAD}")
print(f"Active parameter dim = {len(parameter_ranges[case])}")
print(f"Active parameters    = {_p2_parameter_names}")


# ---------------------------------------------------------------------------
# Compact helpers
# ---------------------------------------------------------------------------
def _p2_parameters_match(cached_params, target_params, rtol=1e-10, atol=1e-12):
    cached_params = np.asarray(cached_params, dtype=float)
    target_params = np.asarray(target_params, dtype=float)
    return cached_params.shape == target_params.shape and np.allclose(cached_params, target_params, rtol=rtol, atol=atol)


def _p2_generate_test_parameters(ranges, d, *, test_set, n_samples, seed):
    ranges_arr = np.asarray(ranges, dtype=float)
    if ranges_arr.shape != (int(d), 2):
        raise ValueError(f"Expected ranges shape ({d}, 2), got {ranges_arr.shape}.")

    lows, highs = ranges_arr[:, 0], ranges_arr[:, 1]

    if int(test_set) == 0:
        a, b = P2_FRESH_INNER_FRACTION
        lo = lows + float(a) * (highs - lows)
        hi = lows + float(b) * (highs - lows)
        return np.random.default_rng(int(seed)).uniform(lo, hi, size=(int(n_samples), int(d)))

    if int(test_set) == 1:
        return np.asarray(list(map(list, itertools.product(*[(lo, hi) for lo, hi in zip(lows, highs)]))), dtype=float)

    if int(test_set) == 2:
        return qmc.scale(qmc.LatinHypercube(d=int(d), seed=int(seed)).random(int(n_samples)), lows, highs)

    raise ValueError("P2_FRESH_TEST_SET must be 0, 1, or 2.")


def _p2_assign_vector_to_function(space, vector, name="f"):
    f = Function(space, name=name)
    f.vector().set_local(np.asarray(vector, dtype=float))
    try:
        f.vector().apply("insert")
    except Exception:
        pass
    return f


def _p2_copy_to_space(func, target_space, name="copy"):
    try:
        if func.function_space().mesh().id() == target_space.mesh().id() and func.function_space().dim() == target_space.dim():
            out = Function(target_space, name=name)
            out.assign(func)
            return out
    except Exception:
        pass

    try:
        out = project(func, target_space)
    except Exception:
        out = interpolate(func, target_space)

    out.rename(name, "")
    return out


def _p2_solve_fresh_coupled_global(active_mu, sample_index=None):
    active_mu = np.asarray(active_mu, dtype=float)
    mu_mech, mu_th, decoded_meta = _p2_decode_active_sample(P2_CONFIG, active_mu)
    solver.set_rom_thermal_parameters(**mu_th)

    label = "" if sample_index is None else f"sample {sample_index}: "
    print(f"  Coupled FOM {label}active μ = {np.round(active_mu, 6).tolist()}")
    print(f"      decoded mechanical μ = {np.round(mu_mech, 6).tolist()}")

    t0 = clock()
    out = solver.solve_rom_sample(
        mu_mech,
        thermal_on=True,
        coupled_on=True,
        heat_nx=int(P2_HEAT_NX),
        heat_ny=int(P2_HEAT_NY),
        heat_nz=int(P2_HEAT_NZ),
        heat_degree=int(P2_HEAT_DEGREE),
        Nz_quad_T1=int(P2_THETA_QUADRATURE),
        T1_cg_degree=int(P2_THETA_CG_DEGREE),
        coupling_omega=float(P2_COUPLING_OMEGA),
        coupling_tol_w=float(P2_COUPLING_TOL_W),
        coupling_tol_T1=float(P2_COUPLING_TOL_THETA),
        coupling_max_iters=int(P2_COUPLING_MAX_ITERS),
        coupling_verbose=False,
        return_mode="global",
        return_coupled_dict=True,
    )
    wall_time = clock() - t0

    u_global = out.get("w", getattr(solver, "coupled_w_global", None))
    if u_global is None:
        raise RuntimeError("Fresh coupled solve did not return out['w'] or solver.coupled_w_global.")

    u_global = _p2_copy_to_space(u_global, solver.V_CG, name="fresh_coupled_global")

    convergence_meta = dict(
        converged=bool(out.get("converged", False)),
        stopped_by_max_iters=bool(out.get("stopped_by_max_iters", False)),
        termination_reason=str(out.get("termination_reason", "")),
        iters=int(out.get("iters", -1)),
        err_w=float(out.get("err_w", np.nan)),
        err_T1=float(out.get("err_T1", np.nan)),
        wall_time_sec=float(wall_time),
        heat_nx=int(P2_HEAT_NX),
        heat_ny=int(P2_HEAT_NY),
        heat_nz=int(P2_HEAT_NZ),
        heat_degree=int(P2_HEAT_DEGREE),
        T1_cg_degree=int(P2_THETA_CG_DEGREE),
        Nz_quad_T1=int(P2_THETA_QUADRATURE),
    )

    print(
        "      done: "
        f"time={wall_time:.2f}s, "
        f"iters={convergence_meta['iters']}, "
        f"converged={convergence_meta['converged']}, "
        f"err_w={convergence_meta['err_w']:.3e}, "
        f"err_T1={convergence_meta['err_T1']:.3e}"
    )

    return u_global, wall_time, decoded_meta, convergence_meta


def _p2_solution_functions_from_global_vectors(global_vectors):
    solution_vectors = _p2_cg_to_solution_snapshots(np.asarray(global_vectors, dtype=float), solver)
    solution_functions = _p2_vectors_to_functions(solution_vectors, solver.W, prefix="fresh_fom_solution_W")
    return solution_vectors, solution_functions


# ---------------------------------------------------------------------------
# Mode A: stored Project-2 coupled snapshots
# ---------------------------------------------------------------------------
if P2_TEST_SOURCE == "stored_snapshots":
    print("\nUsing stored Project-2 coupled snapshot archive as FOM reference data.")

    mu_list_Solution = np.asarray(mu_array_Solution, dtype=float)
    mu_list = np.asarray(mu_array_proj, dtype=float)

    fom_solutions_Solution = _p2_vectors_to_functions(
        fom_snapshots_Solution, solver.W, prefix="stored_fom_solution_W"
    )
    fom_solutions = _p2_vectors_to_functions(
        fom_snapshots_proj, solver.V_CG, prefix="stored_fom_global_CG"
    )

    fom_times = _p2_load_wall_times(P2_CASE_DIR, len(mu_list))
    fom_times_Solution = list(fom_times)
    data_source = "stored Project-2 coupled snapshots"

    print(f"Stored Solution/W FOM references : {len(fom_solutions_Solution)}")
    print(f"Stored projected CG FOM refs     : {len(fom_solutions)}")
    print(f"Wall-time values available       : {len(fom_times)}")


# ---------------------------------------------------------------------------
# Mode B: fresh coupled Project-2 FOM test solves with automatic cache
# ---------------------------------------------------------------------------
elif P2_TEST_SOURCE == "fresh_coupled":
    print("\nUsing fresh Project-2 coupled thermomechanical FOM test solves.")
    print("Cache is automatic: existing consistent cache is loaded; otherwise it is recomputed.")

    test_set = int(P2_FRESH_TEST_SET)
    n_samples = int(P2_FRESH_N_SAMPLES)
    ranges = parameter_ranges[case]
    d = len(ranges)

    fom_solutions_dir = os.path.join(solver.output_dir, "FOM_Solutions")
    fresh_tag = (
        f"fresh_coupled__{P2_RUN_NAME}"
        f"__set{test_set}_n{n_samples}"
        f"__plateN{int(P2_PLATE_RESOLUTION)}_p{int(P2_PLATE_DEGREE)}"
        f"__heat{int(P2_HEAT_NX)}x{int(P2_HEAT_NY)}x{int(P2_HEAT_NZ)}_pT{int(P2_HEAT_DEGREE)}"
        f"__T1p{int(P2_THETA_CG_DEGREE)}_q{int(P2_THETA_QUADRATURE)}"
        f"__tolw{float(P2_COUPLING_TOL_W):.1e}_tolT{float(P2_COUPLING_TOL_THETA):.1e}"
    )

    fresh_cache_root = os.path.join(fom_solutions_dir, "Fresh_Coupled", fresh_tag)
    cache_dir_global = os.path.join(fresh_cache_root, "global")
    cache_dir_Solution = os.path.join(fresh_cache_root, "Solution")
    os.makedirs(cache_dir_global, exist_ok=True)
    os.makedirs(cache_dir_Solution, exist_ok=True)

    param_seed_file = os.path.join(fresh_cache_root, f"param_seed_case{case}_set{test_set}_n{n_samples}.npy")
    if os.path.exists(param_seed_file):
        stored_seed = int(np.load(param_seed_file).item())
    else:
        stored_seed = int(P2_FRESH_TEST_SEED)
        np.save(param_seed_file, stored_seed)

    print(f"Using random seed: {stored_seed}")

    mu_test_active = np.asarray(
        _p2_generate_test_parameters(ranges, d, test_set=test_set, n_samples=n_samples, seed=stored_seed),
        dtype=float,
    )

    print(
        f"Selected Project-2 fresh test_set={test_set}: "
        f"{len(mu_test_active)} active μ points "
        f"(d={d}, requested n_samples={n_samples})"
    )

    cache_file_global = os.path.join(cache_dir_global, "fom_data_global.npz")
    param_cache_global = os.path.join(cache_dir_global, "parameters_global.npy")
    cache_file_Solution = os.path.join(cache_dir_Solution, "fom_data_Solution.npz")
    param_cache_Solution = os.path.join(cache_dir_Solution, "parameters_Solution.npy")

    # 1) Load/compute global CG coupled FOM references.
    global_cache_valid = False

    if os.path.exists(cache_file_global) and os.path.exists(param_cache_global):
        try:
            data_global = np.load(cache_file_global, allow_pickle=True)
            cached_params_global = np.load(param_cache_global)

            if _p2_parameters_match(cached_params_global, mu_test_active):
                mu_list = np.asarray(cached_params_global, dtype=float)
                fom_times = np.asarray(data_global["times"], dtype=float).tolist()
                fom_vecs_global = np.asarray(data_global["snapshots"], dtype=float)
                fom_solutions = _p2_vectors_to_functions(
                    fom_vecs_global, solver.V_CG, prefix="fresh_cached_fom_global_CG"
                )
                global_cache_valid = True
                print(f"Loaded {len(fom_solutions)} fresh-coupled GLOBAL FOM solves from:\n  {cache_file_global}")
            else:
                print("⚠ Global cache parameter inconsistency detected. Will recompute global fresh coupled FOM solves.")

        except Exception as exc:
            print(f"⚠ Global cache loading failed: {exc}. Will recompute global fresh coupled FOM solves.")

    if not global_cache_valid:
        print("\nComputing fresh Project-2 coupled GLOBAL FOM solutions...")

        mu_list = np.asarray(mu_test_active, dtype=float)
        fom_solutions, fom_times, fom_vecs_global = [], [], []
        decoded_metadata, convergence_metadata, failed_samples = [], [], []

        for i_sample, active_mu in enumerate(mu_list):
            try:
                if (i_sample + 1) % 5 == 0 or i_sample < 5:
                    print(f"\n  Computing fresh coupled sample {i_sample + 1}/{len(mu_list)}")

                u_global, wall_time, decoded_meta, conv_meta = _p2_solve_fresh_coupled_global(
                    active_mu, sample_index=i_sample
                )

                fom_solutions.append(u_global)
                fom_times.append(float(wall_time))
                fom_vecs_global.append(u_global.vector().get_local().copy())
                decoded_metadata.append(decoded_meta)
                convergence_metadata.append(conv_meta)

            except Exception as exc:
                fail = dict(sample_index=int(i_sample), active_mu=np.asarray(active_mu, dtype=float).tolist(), error=repr(exc))
                failed_samples.append(fail)
                fail_file = os.path.join(fresh_cache_root, "failed_fresh_coupled_samples.json")
                with open(fail_file, "w") as f:
                    json.dump(failed_samples, f, indent=2)
                raise RuntimeError(f"Fresh coupled FOM solve failed. Failure details saved to: {fail_file}") from exc

        fom_vecs_global = np.asarray(fom_vecs_global, dtype=float)
        fom_times = np.asarray(fom_times, dtype=float).tolist()

        np.savez_compressed(
            cache_file_global,
            mus=np.asarray(mu_list, dtype=float),
            snapshots=fom_vecs_global,
            times=np.asarray(fom_times, dtype=float),
            parameter_names=np.asarray(_p2_parameter_names, dtype=str),
            parameter_units=np.asarray(_p2_parameter_units, dtype=str),
            parameter_roles=np.asarray(_p2_parameter_roles, dtype=str),
            decoded_metadata_json=np.asarray(json.dumps(decoded_metadata, default=str)),
            convergence_metadata_json=np.asarray(json.dumps(convergence_metadata, default=str)),
            source=np.asarray("fresh_coupled_project2_global"),
            project2_run_name=np.asarray(P2_RUN_NAME),
        )
        np.save(param_cache_global, np.asarray(mu_list, dtype=float))

        print(f"\nComputed and cached {len(fom_solutions)} fresh-coupled GLOBAL solves:\n  {cache_file_global}")

    # 2) Load/build Solution/W references from global CG fields.
    solution_cache_valid = False

    if os.path.exists(cache_file_Solution) and os.path.exists(param_cache_Solution):
        try:
            data_Solution_cache = np.load(cache_file_Solution, allow_pickle=True)
            cached_params_Solution = np.load(param_cache_Solution)

            if _p2_parameters_match(cached_params_Solution, mu_list):
                mu_list_Solution = np.asarray(cached_params_Solution, dtype=float)
                fom_times_Solution = np.asarray(data_Solution_cache["times"], dtype=float).tolist()
                fom_vecs_Solution = np.asarray(data_Solution_cache["snapshots"], dtype=float)
                fom_solutions_Solution = _p2_vectors_to_functions(
                    fom_vecs_Solution, solver.W, prefix="fresh_cached_fom_solution_W"
                )
                solution_cache_valid = True
                print(f"Loaded {len(fom_solutions_Solution)} fresh-coupled Solution/W references from:\n  {cache_file_Solution}")
            else:
                print("⚠ Solution/W cache parameter inconsistency detected. Will rebuild from global cache.")

        except Exception as exc:
            print(f"⚠ Solution/W cache loading failed: {exc}. Will rebuild from global cache.")

    if not solution_cache_valid:
        print("\nBuilding fresh-coupled Solution/W references from global CG fields...")

        fom_vecs_global = np.asarray([u.vector().get_local().copy() for u in fom_solutions], dtype=float)
        fom_vecs_Solution, fom_solutions_Solution = _p2_solution_functions_from_global_vectors(fom_vecs_global)

        mu_list_Solution = np.asarray(mu_list, dtype=float)
        fom_times_Solution = list(fom_times)

        np.savez_compressed(
            cache_file_Solution,
            mus=np.asarray(mu_list_Solution, dtype=float),
            snapshots=np.asarray(fom_vecs_Solution, dtype=float),
            times=np.asarray(fom_times_Solution, dtype=float),
            parameter_names=np.asarray(_p2_parameter_names, dtype=str),
            parameter_units=np.asarray(_p2_parameter_units, dtype=str),
            parameter_roles=np.asarray(_p2_parameter_roles, dtype=str),
            source=np.asarray("fresh_coupled_project2_solution_from_global"),
            project2_run_name=np.asarray(P2_RUN_NAME),
            linked_global_cache=np.asarray(cache_file_global),
        )
        np.save(param_cache_Solution, np.asarray(mu_list_Solution, dtype=float))

        print(f"Built and cached {len(fom_solutions_Solution)} fresh-coupled Solution/W references:\n  {cache_file_Solution}")

    data_source = "fresh coupled Project-2 FOM cache"

    print("\nFresh coupled FOM data ready:")
    print(f"  Global CG references      : {len(fom_solutions)}")
    print(f"  Solution/W references     : {len(fom_solutions_Solution)}")
    print(f"  Global cache              : {cache_file_global}")
    print(f"  Solution/W cache          : {cache_file_Solution}")


# ---------------------------------------------------------------------------
# Final FOM-reference report
# ---------------------------------------------------------------------------
print("\n" + "-" * 80)
print("PROJECT-2 FOM REFERENCE SUMMARY")
print("-" * 80)
print(f"Data source                    : {data_source}")
print(f"Number of active μ test points  : {len(mu_list)}")
print(f"Active parameter dimension      : {np.asarray(mu_list).shape[1]}")
print(f"Solution/W references           : {len(fom_solutions_Solution)}")
print(f"Global CG references            : {len(fom_solutions)}")
print(f"FOM times available             : {len(fom_times)}")
print(f"Parameter names                 : {_p2_parameter_names}")
print("-" * 80)


# ---------------------------------------------------------------------------
# Disable old Paper-1 mechanical-only intrusive online solvers if requested.
# This does not disable POD projection, PODI, POD-NN, POD-GPR, or POD-AE.
# ---------------------------------------------------------------------------
if P2_DISABLE_MECHANICAL_ONLY_INTRUSIVE_ONLINE:
    def _p2_disabled_online_PODG_solver(*args, **kwargs):
        raise RuntimeError(
            "Mechanical-only online_PODG_solver is disabled for Project-2 thermomechanical active parameters. "
            "Use POD projection / PODI / PODNN / PODGPR / POD-AE, or implement a dedicated coupled intrusive ROM residual."
        )

    def _p2_disabled_online_LSPG_solver(*args, **kwargs):
        raise RuntimeError(
            "Mechanical-only online_LSPG_solver is disabled for Project-2 thermomechanical active parameters. "
            "Use POD projection / PODI / PODNN / PODGPR / POD-AE, or implement a dedicated coupled LSPG residual."
        )

    solver.online_PODG_solver = _p2_disabled_online_PODG_solver
    solver.online_LSPG_solver = _p2_disabled_online_LSPG_solver


# ---------------------------------------------------------------------------
# Error computation: common global CG L2-type displacement error.
# Works for both Solution/W functions and global CG functions.
# ---------------------------------------------------------------------------
_mass_matrix_cache = {}

def compute_rom_errors(solver, u_fom, u_rom):
    V_CG = FunctionSpace(solver.mesh, "CG", solver.degree + 1)

    def to_global_if_Solution(solution_func):
        if solution_func.function_space().num_sub_spaces() > 0:
            V_DG = FunctionSpace(solver.mesh, "DG", 0)
            subs = np.unique(solver.subdomains.array())
            chi = {sid: Function(V_DG) for sid in subs}

            for sid in subs:
                chi[sid].vector()[:] = (solver.subdomains.array() == sid).astype(float)
                try:
                    chi[sid].vector().apply("insert")
                except Exception:
                    pass

            return project(sum(chi[sid] * solution_func.sub(i) for i, sid in enumerate(subs)), V_CG)

        return project(solution_func, V_CG)

    w_fom_global = to_global_if_Solution(u_fom)
    w_rom_global = to_global_if_Solution(u_rom)

    space_key = (V_CG.mesh().id(), V_CG.ufl_element().family(), V_CG.ufl_element().degree())
    if space_key not in _mass_matrix_cache:
        u, v = TrialFunction(V_CG), TestFunction(V_CG)
        _mass_matrix_cache[space_key] = assemble(u * v * dx)

    M = _mass_matrix_cache[space_key]
    fom_vec = w_fom_global.vector().get_local()
    rom_vec = w_rom_global.vector().get_local()
    err_vec = fom_vec - rom_vec

    abs_error = np.sqrt(max(float(np.dot(err_vec, M * err_vec)), 0.0))
    fom_norm = np.sqrt(max(float(np.dot(fom_vec, M * fom_vec)), 0.0))

    rel_error = 0.0 if abs_error < DOLFIN_EPS else float("inf") if fom_norm < DOLFIN_EPS else abs_error / fom_norm
    return abs_error, rel_error

print(f"Error computation function 'compute_rom_errors' loaded and ready for Project-2 FOM references from: {data_source}")

# %% [markdown] Cell 69 | id: bc6b1677
# ## Intrusive ROM 

# %% [markdown] Cell 70 | id: 99a8a65d
# ### ── Intrusive -- Comparison & Visualization ─────────

# %% Cell 71 | id: 057db187
P2_ENABLE_COUPLED_PODG = False

# %% Cell 72 | id: 7f86fe16
## Intrusive ROM with POD-Galerkin - Solution Space
### ── Intrusive -- Comparison & Visualization ─────────

print(f"POD-Galerkin: choice={choice_podg}, N_basis={N_podg}, n_snaps={len(snaps_all_G)}")
Intrusive_dir = os.path.join(solver.output_dir, "ROM_Intrusive")
os.makedirs(Intrusive_dir, exist_ok=True)

# If an earlier cell installed a disabled instance wrapper, remove it so the
# newly added class method GeneralMultiphysicsSolver.online_PODG_solver is used.
P2_DISABLE_MECHANICAL_ONLY_INTRUSIVE_ONLINE = False
if "online_PODG_solver" in getattr(solver, "__dict__", {}):
    del solver.__dict__["online_PODG_solver"]


# ─────────────────────────────────────────────────────────────────────────────
# ROM plotting helpers: matched to full-order post_processor / summary style
# ─────────────────────────────────────────────────────────────────────────────
def _rom_safe_name(text):
    return "".join(ch if ch.isalnum() else "_" for ch in str(text)).strip("_")


def _rom_save_dual(fig_obj, base_path, pad_inches=None):
    save_kw = dict(facecolor="white")
    if pad_inches is not None:
        save_kw["pad_inches"] = pad_inches
    fig_obj.savefig(f"{base_path}.pdf", dpi=300, bbox_inches="tight", **save_kw)
    fig_obj.savefig(f"{base_path}.png", dpi=600, bbox_inches="tight", **save_kw)


def _rom_project_to_global_factory(solver, V_plot):
    V_DG = FunctionSpace(solver.mesh, "DG", 0)

    if solver.N_subdomains > 1:
        subs = np.unique(solver.subdomains.array())
        chi = {sid: Function(V_DG, name=f"chi_{sid}") for sid in subs}
        for sid in subs:
            chi[sid].vector()[:] = (solver.subdomains.array() == sid).astype(float)
            try:
                chi[sid].vector().apply("insert")
            except Exception:
                pass

        def to_global(sol):
            return (
                project(sum(chi[sid] * sol.sub(i) for i, sid in enumerate(subs)), V_plot)
                if sol.function_space().num_sub_spaces() > 0
                else project(sol, V_plot)
            )

    else:
        def to_global(sol):
            return project(sol, V_plot)

    return to_global


def _rom_field_on_grid(field, Xg, Yg):
    Z = np.zeros_like(Xg)
    for j in range(Xg.shape[0]):
        for i in range(Xg.shape[1]):
            Z[j, i] = field(Point(float(Xg[j, i]), float(Yg[j, i])))
    return Z


def _rom_make_four_ticks(vmin, vmax, positive_only=False):
    if np.isclose(vmin, vmax, atol=1e-14):
        return [vmin, vmin, vmin, vmin]
    if positive_only:
        return (
            [0.0, 0.0, 0.0, 0.0]
            if np.isclose(vmax, 0.0, atol=1e-14)
            else [0.0, vmax / 3.0, 2.0 * vmax / 3.0, vmax]
        )
    return np.linspace(vmin, vmax, 4).tolist()


def _rom_scaled_cbar_info(vmin, vmax, positive_only=False, decimals=1):
    ticks = _rom_make_four_ticks(vmin, vmax, positive_only=positive_only)
    ref = max(abs(vmin), abs(vmax))
    exponent = 0 if ref < 1e-14 else int(np.floor(np.log10(ref)))
    scale = 1.0 if ref < 1e-14 else 10.0 ** exponent
    labels = [f"{t / scale:.{decimals}f}" for t in ticks]
    return ticks, labels, exponent


def _rom_style_map_axis(ax, solver, xlabel=True, ylabel=True):
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.set_xlim(0.0, solver.length)
    ax.set_ylim(0.0, solver.width)
    ax.set_xlabel("x [m]" if xlabel else "", fontsize=14)
    ax.set_ylabel("y [m]" if ylabel else "", fontsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: "" if np.isclose(x, 0.0) else f"{x:.1f}"))
    ax.tick_params(axis="x", labelsize=13, length=3, pad=1)
    ax.tick_params(axis="y", labelsize=13, length=3, pad=3)
    ax.grid(False)


def _rom_draw_scalar(
    fig_obj,
    ax,
    solver,
    Xg,
    Yg,
    Z,
    title,
    *,
    vmin=None,
    vmax=None,
    cmap="viridis",
    positive_only=False,
    xlabel=True,
    ylabel=True,
):
    vmin = float(np.nanmin(Z)) if vmin is None else float(vmin)
    vmax = float(np.nanmax(Z)) if vmax is None else float(vmax)

    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)):
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + max(1e-300, abs(vmin) * 1e-12)

    mappable = ax.contourf(
        Xg,
        Yg,
        Z,
        levels=np.linspace(vmin, vmax, 181),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    _rom_style_map_axis(ax, solver, xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, fontsize=14, pad=8)

    ticks, ticklabels, exponent = _rom_scaled_cbar_info(
        vmin,
        vmax,
        positive_only=positive_only,
        decimals=1,
    )

    cax = make_axes_locatable(ax).append_axes("right", size="4.2%", pad=0.10)
    cbar = fig_obj.colorbar(mappable, cax=cax)
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_locator(FixedLocator(ticks))
    cbar.ax.set_yticklabels(ticklabels)
    cbar.ax.minorticks_off()
    cbar.ax.tick_params(labelsize=13, length=2, pad=2)
    cbar.outline.set_linewidth(0.6)
    cbar.solids.set_edgecolor("face")
    cbar.ax.set_title(rf"$\times 10^{{{exponent}}}$" if exponent != 0 else "", fontsize=14, pad=6)

    return mappable


def _rom_style_legend(legend):
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(0.95)
    return legend


def _rom_style_line_axis(ax, *, show_legend=True, legend_loc="lower left"):
    ax.set_facecolor("white")
    ax.set_box_aspect(0.48)
    ax.tick_params(axis="both", labelsize=12, length=3, pad=2)
    ax.grid(True, linestyle=":", linewidth=0.45, alpha=0.35)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    formatter_y = ScalarFormatter(useMathText=True)
    formatter_y.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(formatter_y)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.yaxis.get_offset_text().set_size(12)

    if show_legend:
        _rom_style_legend(
            ax.legend(
                loc=legend_loc,
                frameon=True,
                fontsize=13,
                borderpad=0.45,
                handlelength=2.2,
                handletextpad=0.6,
                labelspacing=0.35,
            )
        )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _rom_add_shared_line_legend(fig_obj, ax_ref, *, y_anchor=0.900):
    handles, labels = ax_ref.get_legend_handles_labels()
    if not handles:
        return None

    leg = fig_obj.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, y_anchor),
        ncol=min(len(labels), 4),
        frameon=True,
        fontsize=13,
        borderpad=0.45,
        handlelength=2.2,
        handletextpad=0.6,
        columnspacing=1.2,
        labelspacing=0.35,
    )
    return _rom_style_legend(leg)


# ── 1D & 2D Comparison Plot ────────────────────────────────────────────
def plot_fom_rom_comparison(
    solver,
    fom_solution,
    rom_solutions,
    method_names,
    title_prefix="",
    save_dir=".",
):
    plt.rcParams["figure.autolayout"] = False

    V_plot = FunctionSpace(solver.mesh, "CG", 2)
    to_global = _rom_project_to_global_factory(solver, V_plot)

    fom_global = to_global(fom_solution)
    rom_globals = [to_global(rom_sol) for rom_sol in rom_solutions]

    x_vals = np.linspace(0.0, solver.length, 220)
    y_vals = np.linspace(0.0, solver.width, 220)
    diag_vals = np.sqrt(x_vals**2 + y_vals**2)
    y_fixed = solver.width / 2.0

    w_fom_x = np.array([fom_global(Point(float(x), float(y_fixed))) for x in x_vals])
    w_fom_d = np.array([fom_global(Point(float(x), float(y))) for x, y in zip(x_vals, y_vals)])

    w_rom_x = [
        np.array([rom_global(Point(float(x), float(y_fixed))) for x in x_vals])
        for rom_global in rom_globals
    ]
    w_rom_d = [
        np.array([rom_global(Point(float(x), float(y))) for x, y in zip(x_vals, y_vals)])
        for rom_global in rom_globals
    ]

    xg = np.linspace(0.0, solver.length, 260)
    yg = np.linspace(0.0, solver.width, 150)
    Xg, Yg = np.meshgrid(xg, yg)

    fom_grid = _rom_field_on_grid(fom_global, Xg, Yg)
    sol_vmin, sol_vmax = float(np.nanmin(fom_grid)), float(np.nanmax(fom_grid))

    fig = plt.figure(figsize=(12.0, 8.4), facecolor="white")
    gs = GridSpec(
        2,
        4,
        figure=fig,
        height_ratios=[1.0, 1.18],
        width_ratios=[1.0] * 4,
        left=0.060,
        right=0.982,
        bottom=0.070,
        top=0.805,
        hspace=0.36,
        wspace=0.36,
    )

    ax_x = fig.add_subplot(gs[0, 0:2])
    ax_d = fig.add_subplot(gs[0, 2:4])
    ax_f = fig.add_subplot(gs[1, 1:3])

    colors = ["C1", "C2", "C3", "C4", "C5"]
    linestyles = ["--", "-.", ":", (0, (5, 2, 1, 2)), "-"]

    ax_x.plot(x_vals, w_fom_x, color="C0", lw=3.2, label=r"$w_{\mathrm{fom}}$")
    for i, (rom_x, method_name) in enumerate(zip(w_rom_x, method_names)):
        ax_x.plot(
            x_vals,
            rom_x,
            color=colors[i % len(colors)],
            ls=linestyles[i % len(linestyles)],
            lw=2.7,
            label=method_name,
        )
    ax_x.set_title(r"Centerline comparison", fontsize=13, pad=8)
    ax_x.set_xlabel("x [m]", fontsize=12)
    ax_x.set_ylabel("w [m]", fontsize=12)
    _rom_style_line_axis(ax_x, show_legend=False)

    ax_d.plot(diag_vals, w_fom_d, color="C0", lw=3.2, label=r"$w_{\mathrm{fom}}$")
    for i, (rom_d, method_name) in enumerate(zip(w_rom_d, method_names)):
        ax_d.plot(
            diag_vals,
            rom_d,
            color=colors[i % len(colors)],
            ls=linestyles[i % len(linestyles)],
            lw=2.7,
            label=method_name,
        )
    ax_d.set_title(r"Main diagonal comparison", fontsize=13, pad=8)
    ax_d.set_xlabel("Diagonal coordinate [m]", fontsize=12)
    ax_d.set_ylabel("w [m]", fontsize=12)
    _rom_style_line_axis(ax_d, show_legend=False)

    _rom_add_shared_line_legend(fig, ax_x, y_anchor=0.890)

    _rom_draw_scalar(
        fig,
        ax_f,
        solver,
        Xg,
        Yg,
        fom_grid,
        r"FOM ($w_{\mathrm{fom}}$ $[m]$)",
        vmin=sol_vmin,
        vmax=sol_vmax,
        cmap="viridis",
        xlabel=True,
        ylabel=True,
    )

    if title_prefix:
        fig.suptitle(title_prefix, fontsize=14, y=0.975)

    _rom_save_dual(
        fig,
        os.path.join(save_dir, f"{_rom_safe_name(title_prefix)}_comparison"),
        pad_inches=0.03,
    )
    plt.show()


# ── ROM Solutions and Error Fields Plot ────────────────────────────────
def plot_fom_rom_error_fields(
    solver,
    u_fom,
    rom_solutions,
    method_names,
    mu=None,
    title_prefix="",
    save_dir=".",
):
    plt.rcParams["figure.autolayout"] = False

    V_plot = FunctionSpace(solver.mesh, "CG", 2)
    to_global = _rom_project_to_global_factory(solver, V_plot)

    w_fom_global = to_global(u_fom)
    w_rom_globals = [to_global(rom_sol) for rom_sol in rom_solutions]

    fom_vec = w_fom_global.vector().get_local()
    error_fields = []

    for w_rom_global in w_rom_globals:
        err = Function(V_plot)
        err.vector()[:] = np.abs(fom_vec - w_rom_global.vector().get_local())
        try:
            err.vector().apply("insert")
        except Exception:
            pass
        error_fields.append(err)

    xg = np.linspace(0.0, solver.length, 260)
    yg = np.linspace(0.0, solver.width, 150)
    Xg, Yg = np.meshgrid(xg, yg)

    rom_grids = [_rom_field_on_grid(fld, Xg, Yg) for fld in w_rom_globals]
    err_grids = [_rom_field_on_grid(fld, Xg, Yg) for fld in error_fields]

    n_methods = len(method_names)
    fig = plt.figure(figsize=(max(5.8 * n_methods, 7.2), 8.9), facecolor="white")
    gs = GridSpec(
        2,
        n_methods,
        figure=fig,
        left=0.055,
        right=0.985,
        bottom=0.065,
        top=0.895,
        hspace=0.20,
        wspace=0.30,
    )

    for j, (Z, method_name) in enumerate(zip(rom_grids, method_names)):
        _rom_draw_scalar(
            fig,
            fig.add_subplot(gs[0, j]),
            solver,
            Xg,
            Yg,
            Z,
            rf"{method_name} ($w$ $[m]$)",
            cmap="viridis",
            xlabel=True,
            ylabel=(j == 0),
        )

    for j, (Z, error_field, method_name) in enumerate(zip(err_grids, error_fields, method_names)):
        err_vmax = float(np.nanmax(error_field.vector().get_local()))
        if (not np.isfinite(err_vmax)) or err_vmax <= 0.0:
            err_vmax = 1e-300

        _rom_draw_scalar(
            fig,
            fig.add_subplot(gs[1, j]),
            solver,
            Xg,
            Yg,
            Z,
            rf"$|w_{{\mathrm{{fom}}}}-w_{{\mathrm{{rom}}}}|$ ({method_name}) $[m]$",
            vmin=0.0,
            vmax=err_vmax,
            cmap="plasma",
            positive_only=True,
            xlabel=True,
            ylabel=(j == 0),
        )

    mu_text = (
        rf"$\mathbf{{\mu}}={np.round(mu, 5).tolist()}$"
        if mu is not None
        else r"$\mathbf{\mu}=\mathrm{N/A}$"
    )
    fig.suptitle(f"{title_prefix}   {mu_text}", fontsize=14, y=0.975)

    _rom_save_dual(
        fig,
        os.path.join(save_dir, f"{_rom_safe_name(title_prefix)}_error"),
        pad_inches=0.03,
    )
    plt.show()


# ── μ Loop: Paper-2 coupled POD-Galerkin + POD projection ──────────────
# Requirements:
#   1) The Paper-2 online_PODG_solver must already be inside GeneralMultiphysicsSolver.
#   2) Do not pass active Paper-2 μ directly to PODG.
#   3) Decode active μ -> mu_mech + mu_th first.
#   4) POD-Galerkin is run only for monolithic cases.
#   5) POD projection is always run.

P2_ENABLE_COUPLED_PODG = globals().get("P2_ENABLE_COUPLED_PODG", True)

def _p2_int_or_none(x):
    return None if x is None else int(x)


is_monolithic_podg_case = (
    int(getattr(solver, "N_subdomains", 1)) == 1
    and str(globals().get("P2_TARGET_PANEL", "monolithic")).lower() == "monolithic"
)

default_test_index = min(50, len(mu_list_Solution) - 1)
test_indices = globals().get("P2_INTRUSIVE_TEST_INDICES", [default_test_index])
test_indices = [int(i) for i in test_indices if 0 <= int(i) < len(mu_list_Solution)]

if not test_indices:
    raise ValueError("No valid P2_INTRUSIVE_TEST_INDICES were found.")

print("\n" + "=" * 80)
print("PAPER-2 INTRUSIVE ROM / POD PROJECTION CHECK")
print("=" * 80)
print(f"POD-Galerkin enabled by user : {P2_ENABLE_COUPLED_PODG}")
print(f"Monolithic PODG-allowed case : {is_monolithic_podg_case}")
print(f"POD basis size               : {N_podg}")
print(f"Snapshot count               : {len(snaps_all_G)}")
print(f"Test indices                 : {test_indices}")
print("=" * 80)

for test_idx in test_indices:
    mu_active = np.asarray(mu_list_Solution[test_idx], dtype=float)
    u_fom = fom_solutions_Solution[test_idx]

    print("\n" + "-" * 80)
    print(f"Evaluating active Paper-2 μ[{test_idx}] = {np.round(mu_active, 6).tolist()}")
    print("-" * 80)

    rom_solutions = []
    method_names = []

    # ------------------------------------------------------------------
    # 1) POD projection: always valid
    # ------------------------------------------------------------------
    print("  ► Testing POD projection...")

    rbproj, u_rom_proj = solver.POD_projection(u_fom, N_podg, Z_podg)
    abs_err_proj, rel_err_proj = compute_rom_errors(solver, u_fom, u_rom_proj)

    print(
        f"    ➤ POD projection: "
        f"Abs Error = {abs_err_proj:.4e} | Rel Error = {rel_err_proj:.4%}"
    )

    rom_solutions.append(u_rom_proj)
    method_names.append("POD projection")

    # ------------------------------------------------------------------
    # 2) Coupled thermomechanical POD-Galerkin: monolithic only
    # ------------------------------------------------------------------
    podg_success = False
    rbpg = None
    u_rom_podg = None
    podg_info = None

    if not P2_ENABLE_COUPLED_PODG:
        print("  ► Skipping POD-Galerkin: P2_ENABLE_COUPLED_PODG=False.")

    elif not is_monolithic_podg_case:
        print("  ► Skipping POD-Galerkin: allowed only for monolithic Project-2 cases.")

    else:
        print("  ► Testing coupled thermomechanical POD-Galerkin...")

        try:
            mu_mech, mu_th, podg_meta = _p2_decode_active_sample(P2_CONFIG, mu_active)
            solver.set_rom_thermal_parameters(**mu_th)

            print(f"    decoded mechanical μ = {np.round(mu_mech, 6).tolist()}")
            print(f"    decoded thermal keys  = {sorted(mu_th.keys())}")

            rbpg, u_rom_podg, podg_info = solver.online_PODG_solver(
                mu_mech,
                N_podg,
                Z_podg,
                thermal_on=True,
                coupled_on=True,
                heat_nx=_p2_int_or_none(P2_HEAT_NX),
                heat_ny=_p2_int_or_none(P2_HEAT_NY),
                heat_nz=int(P2_HEAT_NZ),
                heat_degree=int(P2_HEAT_DEGREE),
                Nz_quad_T1=int(P2_THETA_QUADRATURE),
                T1_cg_degree=int(P2_THETA_CG_DEGREE),
                coupling_omega=float(P2_COUPLING_OMEGA),
                coupling_tol_w=float(P2_COUPLING_TOL_W),
                coupling_tol_T1=float(P2_COUPLING_TOL_THETA),
                coupling_max_iters=int(P2_COUPLING_MAX_ITERS),
                coupling_verbose=True,
                return_info=True,
            )

            abs_err_podg, rel_err_podg = compute_rom_errors(solver, u_fom, u_rom_podg)
            podg_success = True

            print(
                f"    ➤ Coupled POD-Galerkin: "
                f"Abs Error = {abs_err_podg:.4e} | Rel Error = {rel_err_podg:.4%}"
            )
            print(
                f"    ➤ PODG coupling: "
                f"converged={podg_info.get('converged', None)}, "
                f"iters={podg_info.get('iters', None)}, "
                f"err_w={podg_info.get('err_w', np.nan):.3e}, "
                f"err_T1={podg_info.get('err_T1', np.nan):.3e}"
            )

            rom_solutions.insert(0, u_rom_podg)
            method_names.insert(0, "Coupled POD-Galerkin")

        except Exception as e:
            print(f"    ➤ Coupled POD-Galerkin FAILED/SKIPPED: {e}")
            podg_success = False

    # ------------------------------------------------------------------
    # 3) Coefficient comparison, only if PODG succeeded
    # ------------------------------------------------------------------
    if podg_success:
        rbproj_vec = rbproj.vector().get_local()
        rbpg_vec = rbpg.vector().get_local() if hasattr(rbpg, "vector") else np.asarray(rbpg, dtype=float).ravel()
        coeff_diff = np.linalg.norm(rbproj_vec - rbpg_vec)

        print(f"    ➤ Coefficient difference PODG vs projection: {coeff_diff:.4e}")
        print("\n    Detailed coefficients:")
        print(f"      Projection ({N_podg} basis): {rbproj_vec}")
        print(f"      POD-Galerkin ({N_podg} basis): {rbpg_vec}")

    # ------------------------------------------------------------------
    # 4) Plot available methods
    # ------------------------------------------------------------------
    prefix = f"Methods N_PODG={N_podg}, μ[{test_idx}]"

    print(f"\n  ► Plotting comparison for {len(rom_solutions)} available methods...")

    plot_fom_rom_comparison(
        solver,
        u_fom,
        rom_solutions,
        method_names,
        title_prefix=prefix,
        save_dir=Intrusive_dir,
    )

    plot_fom_rom_error_fields(
        solver,
        u_fom,
        rom_solutions,
        method_names,
        mu=mu_active,
        title_prefix=prefix,
        save_dir=Intrusive_dir,
    )

print("\n" + "=" * 80)
print("INTRUSIVE ROM / POD PROJECTION ANALYSIS COMPLETE")
print(f"  Coupled POD-Galerkin enabled : {P2_ENABLE_COUPLED_PODG}")
print(f"  Monolithic PODG-allowed case : {is_monolithic_podg_case}")
print(f"  POD basis size               : {N_podg}")
print(f"  Snapshot count               : {len(snaps_all_G)}")
print(f"  Results saved to             : {Intrusive_dir}")
print("=" * 80)

# %% [markdown] Cell 73 | id: e6647523
# ### ── Intrusive -- Error Analysis ─────────

# %% Cell 74 | id: b08deaca
# PICK REDUCED BASIS CHOICES - Solution SPACE
choice_podg, choice_podlspg, n_basis_podg, n_basis_podlspg = 0, 0, _p2_cap_basis(N_podg, 15), _p2_cap_basis(N_podlspg, 15)
eigG, eigvG, basisG, N_podg, Z_podg, IP_G, mu_all_G, snaps_all_G = rb_bank_Solution[choice_podg]
N_podg = n_basis_podg or N_podg
eigL, eigvL, basisL, N_podlspg, Z_podlspg, IP_L, mu_all_L, snaps_all_L = rb_bank_Solution[choice_podlspg]
N_podlspg = n_basis_podlspg or N_podlspg

# PICK REDUCED BASIS CHOICES - PROJECTED SPACE
choice_podnn, choice_podI, n_basis_podnn, n_basis_podI = 0, 0, _p2_cap_basis(N_podnn, 15), _p2_cap_basis(N_podI, 15)
eigN, eigvN, basisN, N_podnn, Z_podnn, IP_N, mu_all_N, snaps_all_N = rb_bank_proj[choice_podnn]
N_podnn = n_basis_podnn or N_podnn
eigI, eigvI, basisI, N_podI, Z_podI, IP_I, mu_all_I, snaps_all_I = rb_bank_proj[choice_podI]
N_podI = n_basis_podI or N_podI

# %% Cell 75 | id: 67ff1a9c
# ── POD-Galerkin / Projection setup ──────────────────────────────────────────
print(f"POD-Galerkin/Projection: choice={choice_podg}, N_basis={N_podg}, n_snaps={len(snaps_all_G)}")

Intrusive_dir = os.path.join(solver.output_dir, "ROM_Intrusive")
os.makedirs(Intrusive_dir, exist_ok=True)

# User switch:
#   False -> run only POD projection.
#   True  -> additionally run coupled POD-Galerkin, but only for monolithic cases.
P2_ENABLE_COUPLED_PODG = globals().get("P2_ENABLE_COUPLED_PODG", False)

# If an earlier cell installed a disabled instance wrapper, remove it only when PODG is requested.
if P2_ENABLE_COUPLED_PODG and "online_PODG_solver" in getattr(solver, "__dict__", {}):
    del solver.__dict__["online_PODG_solver"]

is_monolithic_podg_case = (
    int(getattr(solver, "N_subdomains", 1)) == 1
    and str(globals().get("P2_TARGET_PANEL", "monolithic")).lower() == "monolithic"
)

P2_RUN_PODG_NOW = bool(P2_ENABLE_COUPLED_PODG and is_monolithic_podg_case)

test_range = globals().get("P2_PERF_TEST_RANGE", slice(20, 80))
test_start = 0 if test_range.start is None else int(test_range.start)
test_stop = min(len(mu_list_Solution), int(test_range.stop) if test_range.stop is not None else len(mu_list_Solution))
test_range = slice(test_start, test_stop)

N_basis_list = sorted(set([1] + np.linspace(1, N_podg, 5, dtype=int).tolist() + [N_podg]))
methods = (["podg"] if P2_RUN_PODG_NOW else []) + ["proj"]

print("\n" + "=" * 80)
print("PAPER-2 POD-GALERKIN / POD-PROJECTION PERFORMANCE STUDY")
print("=" * 80)
print(f"P2_ENABLE_COUPLED_PODG   : {P2_ENABLE_COUPLED_PODG}")
print(f"Monolithic PODG allowed  : {is_monolithic_podg_case}")
print(f"Methods active           : {methods}")
print(f"Testing parameter range  : {test_range.start} to {test_range.stop - 1} "
      f"(total: {test_range.stop - test_range.start})")
print(f"Testing basis sizes      : {N_basis_list}")
print("=" * 80)


# ── Plotting helpers: matched to full-order line/log-plot style ──────────────
def _rom_perf_save_dual(fig_obj, base_path, pad_inches=None):
    save_kw = {"facecolor": "white", **({} if pad_inches is None else {"pad_inches": pad_inches})}
    for ext, dpi in (("pdf", 300), ("png", 600)):
        fig_obj.savefig(f"{base_path}.{ext}", dpi=dpi, bbox_inches="tight", **save_kw)


def _rom_perf_style_legend(legend):
    legend.get_frame().set(edgecolor="black", linewidth=0.8, alpha=0.95)
    return legend


def _rom_perf_style_axis(ax, *, log_y=True):
    ax.set_facecolor("white")
    if log_y:
        ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=13, length=3, pad=2)
    ax.grid(True, which="both", linestyle=":", linewidth=0.45, alpha=0.35)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

    if not log_y:
        formatter_y = ScalarFormatter(useMathText=True)
        formatter_y.set_powerlimits((-2, 2))
        ax.yaxis.set_major_formatter(formatter_y)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax.yaxis.get_offset_text().set_size(12)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _rom_perf_log_floor(*arrays):
    vals = [np.asarray(a, dtype=float) for a in arrays]
    vals = [v[np.isfinite(v) & (v > 0.0)] for v in vals]
    vals = [v for v in vals if v.size]
    return 1e-16 if not vals else max(float(np.nanmin(np.concatenate(vals))) * 0.10, 1e-300)


def _rom_perf_plot_mean_band(ax, x_vals, mean_vals, min_vals, max_vals, *, color, marker, linestyle, label):
    x_vals, mean_vals, min_vals, max_vals = map(
        lambda a: np.asarray(a, dtype=float),
        (x_vals, mean_vals, min_vals, max_vals),
    )
    mask = (
        np.isfinite(x_vals)
        & np.isfinite(mean_vals)
        & np.isfinite(min_vals)
        & np.isfinite(max_vals)
        & (mean_vals > 0.0)
        & (max_vals > 0.0)
    )
    if not np.any(mask):
        return None

    x_plot, mean_plot, min_plot, max_plot = x_vals[mask], mean_vals[mask], min_vals[mask], max_vals[mask]
    min_plot = np.maximum(min_plot, _rom_perf_log_floor(mean_plot, min_plot, max_plot))
    max_plot = np.maximum(max_plot, min_plot)

    band = ax.fill_between(x_plot, min_plot, max_plot, color=color, alpha=0.16, linewidth=0.0, zorder=1)
    line, = ax.plot(
        x_plot,
        mean_plot,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=2.4,
        markersize=6.2,
        markeredgewidth=0.8,
        label=label,
        zorder=2,
    )
    return line, band


# ── Unified Error + Speedup Analysis Function ────────────────────────────────
def _p2_int_or_none(x):
    return None if x is None else int(x)


def _safe_speedup(t_fom, t_rom):
    return np.nan if (not np.isfinite(t_fom)) or (not np.isfinite(t_rom)) or t_rom <= 0.0 else float(t_fom / t_rom)


def run_all_methods_error_and_speedup(fom_sols, mus, Z_podg, N_podg, basis_list, fom_times):
    errors = defaultdict(dict)
    speedups = {m: defaultdict(list) for m in methods}
    timings = {m: defaultdict(list) for m in methods}

    for N in basis_list:
        print(f"\n\033[1mEvaluating ROM methods for N_basis = {N}\033[0m")

        n_use = min(int(N), int(N_podg))
        results = {
            m: {"abs": [], "rel": [], "times": [], "success": []}
            for m in methods
        }

        for idx, (mu_active, u_fom) in enumerate(zip(mus, fom_sols), 1):
            mu_active = np.asarray(mu_active, dtype=float)
            t_fom = float(fom_times[idx - 1])

            print(f"  ⟶ Test {idx}/{len(mus)} active μ = {np.round(mu_active, 5).tolist()}")

            # --------------------------------------------------------------
            # Coupled POD-Galerkin: optional, monolithic only, decoded μ.
            # --------------------------------------------------------------
            if "podg" in methods:
                t0 = clock()
                try:
                    mu_mech, mu_th, podg_meta = _p2_decode_active_sample(P2_CONFIG, mu_active)
                    solver.set_rom_thermal_parameters(**mu_th)

                    rbpg, u_rom_podg, podg_info = solver.online_PODG_solver(
                        mu_mech,
                        n_use,
                        Z_podg,
                        thermal_on=True,
                        coupled_on=True,
                        heat_nx=_p2_int_or_none(P2_HEAT_NX),
                        heat_ny=_p2_int_or_none(P2_HEAT_NY),
                        heat_nz=int(P2_HEAT_NZ),
                        heat_degree=int(P2_HEAT_DEGREE),
                        Nz_quad_T1=int(P2_THETA_QUADRATURE),
                        T1_cg_degree=int(P2_THETA_CG_DEGREE),
                        coupling_omega=float(P2_COUPLING_OMEGA),
                        coupling_tol_w=float(P2_COUPLING_TOL_W),
                        coupling_tol_T1=float(P2_COUPLING_TOL_THETA),
                        coupling_max_iters=int(P2_COUPLING_MAX_ITERS),
                        coupling_verbose=False,
                        return_info=True,
                    )

                    t_podg = clock() - t0
                    ae_podg, re_podg = compute_rom_errors(solver, u_fom, u_rom_podg)
                    sp_podg = _safe_speedup(t_fom, t_podg)

                    results["podg"]["abs"].append(ae_podg)
                    results["podg"]["rel"].append(re_podg)
                    results["podg"]["times"].append((t_fom, t_podg))
                    results["podg"]["success"].append(True)
                    speedups["podg"][N].append(sp_podg)

                    print(
                        f"      ✓ PODG: rel err={re_podg:.3e}, "
                        f"speedup={sp_podg:.1f}×, "
                        f"coupled={podg_info.get('converged', None)}, "
                        f"iters={podg_info.get('iters', None)}"
                    )

                except Exception as exc:
                    print(f"      ✗ PODG failed/skipped: {str(exc)[:100]}...")
                    results["podg"]["abs"].append(np.nan)
                    results["podg"]["rel"].append(np.nan)
                    results["podg"]["times"].append((t_fom, np.nan))
                    results["podg"]["success"].append(False)
                    speedups["podg"][N].append(np.nan)

            # --------------------------------------------------------------
            # POD projection: always active.
            # --------------------------------------------------------------
            t0 = clock()
            try:
                _, u_rom_proj = solver.POD_projection(u_fom, n_use, Z_podg)
                t_proj = clock() - t0
                ae_proj, re_proj = compute_rom_errors(solver, u_fom, u_rom_proj)
                sp_proj = _safe_speedup(t_fom, t_proj)

                results["proj"]["abs"].append(ae_proj)
                results["proj"]["rel"].append(re_proj)
                results["proj"]["times"].append((t_fom, t_proj))
                results["proj"]["success"].append(True)
                speedups["proj"][N].append(sp_proj)

                print(f"      ✓ Proj: rel err={re_proj:.3e}, speedup={sp_proj:.1f}×")

            except Exception as exc:
                print(f"      ✗ Projection failed: {str(exc)[:100]}...")
                results["proj"]["abs"].append(np.nan)
                results["proj"]["rel"].append(np.nan)
                results["proj"]["times"].append((t_fom, np.nan))
                results["proj"]["success"].append(False)
                speedups["proj"][N].append(np.nan)

        # Store per-basis results.
        for method in methods:
            errors[f"abs_{method}"][N] = results[method]["abs"]
            errors[f"rel_{method}"][N] = results[method]["rel"]
            timings[method][N] = results[method]["times"]

            success_rate = 100.0 * sum(results[method]["success"]) / max(len(results[method]["success"]), 1)
            valid_speedups = [s for s in speedups[method][N] if np.isfinite(s)]

            if success_rate > 0 and valid_speedups:
                print(
                    f"      ➤ {method.upper()}: {success_rate:.0f}% success, "
                    f"{np.mean(valid_speedups):.2f}× avg speedup"
                )
            else:
                print(f"      ➤ {method.upper()}: 0% success")

    return errors, speedups, timings


# ── Run unified analysis ─────────────────────────────────────────────────────
errors, speedups, timings = run_all_methods_error_and_speedup(
    fom_solutions_Solution[test_range],
    mu_list_Solution[test_range],
    Z_podg,
    N_podg,
    N_basis_list,
    fom_times_Solution[test_range],
)


# ── Collate statistics ───────────────────────────────────────────────────────
stats = defaultdict(dict)

def _mean_min_max_by_basis(data_by_N):
    out = {k: [] for k in ("mean", "min", "max")}
    for N in N_basis_list:
        vals = np.asarray([v for v in data_by_N[N] if np.isfinite(v)], dtype=float)
        for key, fn in (("mean", np.mean), ("min", np.min), ("max", np.max)):
            out[key].append(float(fn(vals)) if vals.size else np.nan)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


for method in methods:
    for err_type in ("abs", "rel"):
        stats[f"{err_type}_{method}"] = _mean_min_max_by_basis(errors[f"{err_type}_{method}"])
    stats[method] = _mean_min_max_by_basis(speedups[method])


# ── Individual visualizations ────────────────────────────────────────────────
plt.rcParams["figure.autolayout"] = False

colors = {"podg": "C1", "proj": "C2"}
method_names = {"podg": "Coupled POD-Galerkin", "proj": "POD-Projection"}
markers = {"podg": "s", "proj": "^"}
linestyles = {"podg": "--", "proj": "-."}
basis_array = np.asarray(N_basis_list, dtype=float)

def _plot_rom_perf(metric_key, title, ylabel, filename):
    fig = plt.figure(figsize=(6.8, 5.2), dpi=150, facecolor="white")
    ax = fig.add_subplot(
        GridSpec(1, 1, figure=fig, left=0.135, right=0.975, bottom=0.150, top=0.885)[0, 0]
    )

    for method in methods:
        key = metric_key(method)
        _rom_perf_plot_mean_band(
            ax,
            basis_array,
            stats[key]["mean"],
            stats[key]["min"],
            stats[key]["max"],
            color=colors[method],
            marker=markers[method],
            linestyle=linestyles[method],
            label=method_names[method],
        )

    ax.set_title(title, fontsize=14, pad=8)
    ax.set_xlabel(r"$N_{\mathrm{basis}}$", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    _rom_perf_style_axis(ax, log_y=True)

    _rom_perf_style_legend(
        ax.legend(
            loc="best",
            frameon=True,
            fontsize=12,
            borderpad=0.45,
            handlelength=2.2,
            handletextpad=0.6,
            labelspacing=0.35,
        )
    )

    _rom_perf_save_dual(fig, os.path.join(Intrusive_dir, filename), pad_inches=0.03)
    plt.show()


_plot_rom_perf(
    lambda m: f"rel_{m}",
    r"Relative error vs. basis size",
    r"Relative $L^2$ error",
    "all_methods_relative_error",
)

_plot_rom_perf(
    lambda m: m,
    r"Speed-up vs. basis size",
    r"Speed-up factor $(t_{\mathrm{FOM}}/t_{\mathrm{ROM}})$",
    "all_methods_speedup",
)


# ── Summary tables ───────────────────────────────────────────────────────────
for method in methods:
    method_name = method_names[method]
    rel_key = f"rel_{method}"
    valid_indices = [
        i for i, _ in enumerate(N_basis_list)
        if np.isfinite(stats[rel_key]["mean"][i])
    ]

    if valid_indices:
        df = pd.DataFrame(
            {
                "Mean Rel Err": [stats[rel_key]["mean"][i] for i in valid_indices],
                "Max Rel Err": [stats[rel_key]["max"][i] for i in valid_indices],
                "Mean Speed-up": [stats[method]["mean"][i] for i in valid_indices],
            },
            index=[N_basis_list[i] for i in valid_indices],
        )
        df.index.name = "N basis"
        print(f"\n{method_name} Solver Summary:")
        print(df.to_markdown(tablefmt="github", floatfmt=".3e"))
    else:
        print(f"\n{method_name} Solver: No successful runs to report")


print(f"\nAnalysis completed for {len(N_basis_list)} basis sizes and {test_range.stop - test_range.start} test cases")
print("\n" + "=" * 80)
print("INTRUSIVE ROM ANALYSIS COMPLETE")
print("Configuration Summary:")
print(f"  P2_ENABLE_COUPLED_PODG : {P2_ENABLE_COUPLED_PODG}")
print(f"  PODG actually active   : {P2_RUN_PODG_NOW}")
print(f"  Active methods         : {methods}")
print(f"  N_basis max            : {N_podg}")
print(f"  Snapshots              : {len(snaps_all_G)}")
print(f"  Results saved to       : {Intrusive_dir}")
print("=" * 80)

# %% [markdown] Cell 76 | id: 786b7f27
# ## Non-Intrusive ROM 

# %% Cell 77 | id: 376df81e
# PICK REDUCED BASIS CHOICES - Solution SPACE
choice_podg, choice_podlspg, n_basis_podg, n_basis_podlspg = 0, 0, _p2_cap_basis(N_podg, 4), _p2_cap_basis(N_podlspg, 4)
eigG, eigvG, basisG, N_podg, Z_podg, IP_G, mu_all_G, snaps_all_G = rb_bank_Solution[choice_podg]
N_podg = n_basis_podg or N_podg
eigL, eigvL, basisL, N_podlspg, Z_podlspg, IP_L, mu_all_L, snaps_all_L = rb_bank_Solution[choice_podlspg]
N_podlspg = n_basis_podlspg or N_podlspg

# PICK REDUCED BASIS CHOICES - PROJECTED SPACE
choice_podnn, choice_podI, n_basis_podnn, n_basis_podI = 0, 0, _p2_cap_basis(N_podnn, 4), _p2_cap_basis(N_podI, 4)
eigN, eigvN, basisN, N_podnn, Z_podnn, IP_N, mu_all_N, snaps_all_N = rb_bank_proj[choice_podnn]
N_podnn = n_basis_podnn or N_podnn
eigI, eigvI, basisI, N_podI, Z_podI, IP_I, mu_all_I, snaps_all_I = rb_bank_proj[choice_podI]
N_podI = n_basis_podI or N_podI

# %% Cell 78 | id: b0cccb44
### ── Snapshot file information ─────────
# ───────────────────── ROM Snapshot Data Inspection ────────────────────────────
label, choice, N_basis, snaps_all, mu_all = "PODNN", choice_podnn, N_podnn, snaps_all_N, mu_all_N  # or "PODG", "PODI", etc.
print(f"{label}: choice={choice}, N_basis={N_basis}, n_snaps={len(snaps_all)}")

# Convert snapshots into Functions
pod_snapshots = [(f := Function(solver.V_CG), f.vector().__setitem__(slice(None), arr), f)[2] for arr in snaps_all]

print(f"\n Inspecting {label} snapshot data for choice={choice}...\n")
snapshot_info = f"  Snapshot file: {snapshot_file_proj}\n   - Total snapshots      : {snaps_all.shape[0]}\n   - Size of each snapshot: {snaps_all.shape[1]} DOFs"
snapshot_info += f"\n   - Parameter vector dim  : {mu_all.shape[1]}\n   - Parameter samples     : {mu_all.shape[0]}"
snapshot_info += f"\n   - Snapshot dtype        : {snaps_all.dtype}\n   - Parameter dtype       : {mu_all.dtype}"
anomaly_info = f"\n Checking for anomalies...\n   - NaNs in snapshots?   {np.isnan(snaps_all).any()}\n   - NaNs in parameters?  {np.isnan(mu_all).any()}"
range_info = f"\n Snapshot value range:\n   - Min: {snaps_all.min():.4e}, Max: {snaps_all.max():.4e}\n Parameter ranges:"
param_ranges = "\n".join(f"   - Param {j+1}: Min = {col.min():.4e}, Max = {col.max():.4e}, Mean = {col.mean():.4e}, Std = {col.std():.4e}" for j, col in enumerate(mu_all.T))
print(snapshot_info + "\n" + anomaly_info + "\n" + range_info + "\n" + param_ranges)

# Plot a few random snapshots
print("\n  Plotting a few sample ROM snapshots (raw vectors)...")
n_plot, random_ids = 3, np.random.choice(len(snaps_all), 3, replace=False)
plt.figure(figsize=(15, 4))
for k, idx in enumerate(random_ids):
    ax = plt.subplot(1, n_plot, k+1); ax.plot(snaps_all[idx])
    ax.set(title=f"{label} Snapshot #{idx}", xlabel="DOF Index", ylabel="Deflection [m]"); ax.grid(True, linestyle=":")
plt.tight_layout(); plt.show()

# Show parameter table for those samples
param_names, sample_params_df = parameter_labels[CASE]["names"], pd.DataFrame(mu_all[random_ids], columns=parameter_labels[CASE]["names"])
print("\n Parameter values for plotted ROM snapshots:\n" + sample_params_df.to_string(index=False))


print("Testing projected projection solver...")
N_test = min(5, N_podI)
for i, mu_test in enumerate(mu_list[:3], 1):
    print(f"\nTest {i}: μ = {mu_test}")
    u_fom_cg = fom_solutions[i-1]
    try:
        reduced_coeffs, u_projected = solver.POD_projection_projected(u_fom_cg, N_test, Z_podI)
        abs_err, rel_err = compute_rom_errors(solver, u_fom_cg, u_projected)
        print(f"  ✓ Projected Projection: Coeffs shape: {reduced_coeffs.vector().get_local().shape}")
        print(f"    Abs Error: {abs_err:.6e} | Rel Error: {rel_err:.6e} ({rel_err*100:.4f}%)")
    except Exception as e:
        print(f"  ✗ Projected Projection FAILED: {e}")
print("\nProjected projection solver test complete!")

# %% [markdown] Cell 79 | id: 468b4cb6
# ### PODI : Proper Orthogonal Decomposition with Interpolation

# %% Cell 80 | id: dddec6d6
# =============================================================================
# PODI : Robust train/validation/test POD coefficient interpolation ROM
# Project-2 thermomechanical ROM
# -----------------------------------------------------------------------------
# Drop-in replacement for the current PODI block.
#
# Main strategy:
#   1) Compact Project-2 workflow preserved.
#   2) One fixed train/validation/test split, no scaler leakage.
#   3) Methods preserved: 'rbf' and 'linear'.
#   4) RBFInterpolator preferred, legacy Rbf fallback retained.
#   5) LinearNDInterpolator with NearestNDInterpolator fallback for NaNs/outside
#      convex hull, so prediction does not silently fail.
#   6) Full diagnostics on train/validation/test:
#        - coefficient error,
#        - POD-subspace field error,
#        - total field error against snapshots,
#        - POD projection floor,
#        - surrogate gap,
#        - fallback/extrapolation count for linear.
#   7) Worst-sample inspection with corresponding parameter values.
#   8) Save/load complete model, scalers, basis, split indices, diagnostics, plots.
# =============================================================================

# =============================================================================
# Folder convention
# =============================================================================
def _method_root(base_dir, method):
    m = str(method).lower()
    return os.path.join(base_dir, "PODI_RBF" if m == "rbf" else f"PODI_{m.upper()}")


choice_podi_run = globals().get("choice_podi_run", "live")  # "live" or "load"
podi_dir = os.path.join(solver.output_dir, "ROM_Non_Intrusive", "PODI")
os.makedirs(podi_dir, exist_ok=True)


# =============================================================================
# Configuration
# =============================================================================
PODI_CONFIG = dict(
    # Single deterministic split, matching final POD-NN/POD-GPR/POD-AE blocks.
    seed=100,
    split_seed=100,
    train_fraction=0.80,
    val_fraction=0.10,
    test_fraction=0.10,

    # Methods to train/load.
    method_list=("rbf", "linear"),

    # RBF settings.  Thin-plate spline is scale-robust and does not need epsilon.
    # smoothing=0.0 gives exact interpolation on the training coefficients.
    rbf_kwargs=dict(
        kernel="thin_plate_spline",
        smoothing=0.0,
    ),

    # Optional alternative for rougher data:
    # rbf_kwargs=dict(kernel="multiquadric", epsilon=1.0, smoothing=1e-10),

    # Linear settings.
    # LinearNDInterpolator returns NaN outside the convex hull; nearest_fallback=True
    # replaces those NaNs with nearest-neighbour coefficient values and reports counts.
    linear_kwargs=dict(),
    nearest_fallback=True,

    # Diagnostics / outputs.
    save_plots=True,
    save_csv=True,
    print_worst_samples=True,
    n_worst_samples=10,
    verbose=True,
)


# =============================================================================
# Utilities
# =============================================================================
def _podi_seed(seed=100):
    random.seed(int(seed))
    np.random.seed(int(seed))


def _podi_check_fractions(train_fraction, val_fraction, test_fraction):
    vals = np.array([train_fraction, val_fraction, test_fraction], dtype=float)
    if np.any(vals <= 0.0):
        raise ValueError("train_fraction, val_fraction and test_fraction must all be positive.")
    if not np.isclose(vals.sum(), 1.0):
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1.0.")


def _podi_split_indices(n_samples, train_fraction, val_fraction, test_fraction, seed):
    _podi_check_fractions(train_fraction, val_fraction, test_fraction)
    idx = np.arange(int(n_samples))
    train_idx, tmp_idx = train_test_split(
        idx, train_size=float(train_fraction), random_state=int(seed), shuffle=True
    )
    test_ratio_inside_tmp = float(test_fraction) / float(val_fraction + test_fraction)
    val_idx, test_idx = train_test_split(
        tmp_idx, test_size=test_ratio_inside_tmp, random_state=int(seed) + 1, shuffle=True
    )
    return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int), np.asarray(test_idx, dtype=int)


def _podi_basis_file_key(fname):
    try:
        return int(os.path.splitext(fname)[0].split("_")[-1])
    except Exception:
        return fname


def _podi_as_2d(A):
    A = np.asarray(A, dtype=np.float64)
    if A.ndim == 1:
        return A.reshape(1, -1)
    return A


# =============================================================================
# PODI ROM class
# =============================================================================
class PODInterpReducedOrderModel:
    """Non-intrusive POD + interpolation ROM. Methods: 'rbf', 'linear'."""

    def __init__(self, solver, reduced_basis, n_basis, inner_product, config=None):
        self.solver = solver
        self.reduced_basis = reduced_basis
        self.n_basis = int(n_basis)
        self.inner_product = inner_product
        self.config = dict(PODI_CONFIG if config is None else config)

        self.param_scaler = None
        self.coeff_scaler = None
        self.interpolator = None
        self.nearest_interpolator = None
        self.method = None
        self.interp_kwargs = {}

        self.split_indices = {}
        self.training_summary = {}
        self.metrics = {}
        self.error_df = None
        self._basis_matrix_cache = None
        self.last_nan_count = 0
        self.last_fallback_mask = None

    # ------------------------------------------------------------------
    # POD projection and reconstruction
    # ------------------------------------------------------------------
    def _basis_matrix(self):
        """Rows are POD basis vectors, shape = (n_basis, n_dofs)."""
        if self._basis_matrix_cache is None or self._basis_matrix_cache.shape[0] != self.n_basis:
            self._basis_matrix_cache = np.vstack([
                self.reduced_basis[i].vector().get_local()
                for i in range(self.n_basis)
            ]).astype(np.float64)
        return self._basis_matrix_cache

    def project_snapshots(self, snapshots):
        print(" Projecting snapshots onto reduced basis...")
        Z = self.reduced_basis[:self.n_basis]
        MZ = [self.inner_product * z.vector() for z in Z]
        A = np.array([[zi.vector().inner(mzj) for mzj in MZ] for zi in Z], dtype=np.float64)

        C = np.array([
            np.linalg.solve(A, np.array([s.vector().inner(mzj) for mzj in MZ], dtype=np.float64))
            for s in snapshots
        ], dtype=np.float64)

        print(f" Full-order space dim: {self.solver.V_CG.dim()}")
        print(f" Reduced basis vector dim: {Z[0].vector().size()}")
        print(f" Inner product shape: {self.inner_product.size(0)} × {self.inner_product.size(1)}")
        print(f" Z_N shape: {np.column_stack([z.vector().get_local() for z in Z]).shape}")
        print(f" Solution vector shape: {snapshots[0].vector().get_local().shape}")
        return C

    # ------------------------------------------------------------------
    # Fit interpolator
    # ------------------------------------------------------------------
    def fit_interpolator(self, mus, coeff_matrix, *, method="rbf", interp_kwargs=None,
                         snapshot_targets=None, split_indices=None):
        cfg = self.config
        self.method = method = str(method).lower()
        self.interp_kwargs = dict(interp_kwargs or {})

        _podi_seed(cfg.get("seed", 100))
        split_seed = int(cfg.get("split_seed", cfg.get("seed", 100)))

        X_raw = _podi_as_2d(mus)
        C_raw = _podi_as_2d(coeff_matrix)

        if X_raw.shape[0] != C_raw.shape[0]:
            raise ValueError(f"mus has {X_raw.shape[0]} rows, but coeff_matrix has {C_raw.shape[0]} rows.")
        if C_raw.shape[1] != self.n_basis:
            raise ValueError(f"coeff_matrix has {C_raw.shape[1]} columns, but this PODI uses n_basis={self.n_basis}.")

        S = None
        if snapshot_targets is not None:
            S = _podi_as_2d(snapshot_targets)
            if S.shape[0] != X_raw.shape[0]:
                raise ValueError(f"snapshot_targets has {S.shape[0]} rows, expected {X_raw.shape[0]}.")
            if S.shape[1] != self.solver.V_CG.dim():
                raise ValueError(f"snapshot_targets has {S.shape[1]} DOFs, expected {self.solver.V_CG.dim()}.")

        if split_indices is None:
            train_idx, val_idx, test_idx = _podi_split_indices(
                X_raw.shape[0],
                cfg.get("train_fraction", 0.80),
                cfg.get("val_fraction", 0.10),
                cfg.get("test_fraction", 0.10),
                split_seed,
            )
        else:
            train_idx = np.asarray(split_indices["train"], dtype=int)
            val_idx = np.asarray(split_indices["val"], dtype=int)
            test_idx = np.asarray(split_indices["test"], dtype=int)

        self.split_indices = dict(train=train_idx.tolist(), val=val_idx.tolist(), test=test_idx.tolist())

        print(" Fitting POD coefficient interpolator...")
        print(f" Method: {method.upper()} | seed={cfg.get('seed', 100)} | split_seed={split_seed}")
        print(f" Splitting data into train/validation/test sets ({len(train_idx)} / {len(val_idx)} / {len(test_idx)})...")

        # No scaler leakage: fit scalers only on training data.
        self.param_scaler = StandardScaler().fit(X_raw[train_idx])
        self.coeff_scaler = StandardScaler().fit(C_raw[train_idx])
        self.coeff_scaler.scale_[self.coeff_scaler.scale_ < 1e-12] = 1.0

        X = self.param_scaler.transform(X_raw)
        Y = self.coeff_scaler.transform(C_raw)
        Xtr, Ytr = X[train_idx], Y[train_idx]

        t0 = time.perf_counter()
        status = "ok"
        note = ""

        if method == "rbf":
            # Prefer the vector-valued modern RBFInterpolator.
            try:
                self.interpolator = RBFInterpolator(Xtr, Ytr, **self.interp_kwargs)
                self.nearest_interpolator = None
                backend = "RBFInterpolator"
            except Exception as e:
                # Retain legacy fallback, coefficient by coefficient.
                backend = "legacy_Rbf_fallback"
                note = str(e)[:180]
                print(f" RBFInterpolator failed ({note}); using legacy scipy.interpolate.Rbf fallback.")
                rbfs = [Rbf(*Xtr.T, Ytr[:, j], **self.interp_kwargs) for j in range(Ytr.shape[1])]
                self.interpolator = lambda Xi: np.vstack([rbf(*Xi.T) for rbf in rbfs]).T
                self.nearest_interpolator = None

        elif method == "linear":
            try:
                self.interpolator = LinearNDInterpolator(Xtr, Ytr, **self.interp_kwargs)
                self.nearest_interpolator = NearestNDInterpolator(Xtr, Ytr) if bool(cfg.get("nearest_fallback", True)) else None
                backend = "LinearNDInterpolator"
            except Exception as e:
                status = "fallback_nearest_only"
                note = str(e)[:180]
                print(f" LinearNDInterpolator failed ({note}); using nearest-neighbour only.")
                self.interpolator = None
                self.nearest_interpolator = NearestNDInterpolator(Xtr, Ytr)
                backend = "NearestNDInterpolator"
        else:
            raise ValueError(f"Unknown interpolation method '{method}'. Use 'rbf' or 'linear'.")

        elapsed = time.perf_counter() - t0
        self.training_summary = dict(
            method=method,
            backend=backend,
            status=status,
            note=note,
            elapsed_sec=float(elapsed),
            n_train=int(len(train_idx)),
            n_val=int(len(val_idx)),
            n_test=int(len(test_idx)),
            n_basis=int(self.n_basis),
            interp_kwargs=self.interp_kwargs,
            config=cfg,
            split_indices=self.split_indices,
        )

        print(f" PODI-{method.upper()} fit complete in {elapsed:.3f} s using {backend}.")

        # Diagnostics.
        metrics, df_errors = self.evaluate_splits(
            X_raw, C_raw, snapshot_targets=S,
            split_indices=self.split_indices,
            save_csv=os.path.join(_method_root(podi_dir, method), "podi_train_val_test_errors.csv")
            if bool(cfg.get("save_csv", True)) else None,
        )
        self.metrics = metrics
        self.error_df = df_errors

        if bool(cfg.get("save_csv", True)):
            root = _method_root(podi_dir, method)
            os.makedirs(root, exist_ok=True)
            pd.DataFrame([self.training_summary]).to_csv(
                os.path.join(root, "podi_training_summary.csv"), index=False
            )
            print(f"[PODI-{method.upper()}] Saved training summary CSV → {os.path.join(root, 'podi_training_summary.csv')}")

        if bool(cfg.get("print_worst_samples", True)):
            self.print_worst_samples(df_errors, mus=X_raw, n=int(cfg.get("n_worst_samples", 10)))

        return self, metrics, df_errors

    # ------------------------------------------------------------------
    # Prediction and reconstruction
    # ------------------------------------------------------------------
    def _predict_scaled(self, X_scaled):
        Xi = np.asarray(X_scaled, dtype=np.float64)
        if Xi.ndim == 1:
            Xi = Xi.reshape(1, -1)

        if self.method == "linear" and self.interpolator is None:
            Yi = np.asarray(self.nearest_interpolator(Xi), dtype=np.float64)
            self.last_fallback_mask = np.ones(Yi.shape[0], dtype=bool)
            self.last_nan_count = int(Yi.shape[0])
            return _podi_as_2d(Yi)

        Yi = np.asarray(self.interpolator(Xi), dtype=np.float64)
        Yi = _podi_as_2d(Yi)

        fallback_mask = np.zeros(Yi.shape[0], dtype=bool)
        if self.method == "linear" and self.nearest_interpolator is not None:
            # LinearNDInterpolator returns NaN outside the convex hull.  Fill those rows.
            row_nan = ~np.isfinite(Yi).all(axis=1)
            if np.any(row_nan):
                Yn = np.asarray(self.nearest_interpolator(Xi[row_nan]), dtype=np.float64)
                Yi[row_nan] = _podi_as_2d(Yn)
                fallback_mask[row_nan] = True

        if not np.isfinite(Yi).all():
            raise FloatingPointError(
                f"PODI-{self.method.upper()} produced non-finite coefficients after fallback."
            )

        self.last_fallback_mask = fallback_mask
        self.last_nan_count = int(np.sum(fallback_mask))
        return Yi

    def predict_coeffs_batch(self, mus, return_fallback_mask=False):
        if self.param_scaler is None or self.coeff_scaler is None:
            raise RuntimeError("PODI model is not fitted/loaded.")

        X = _podi_as_2d(mus)
        Xi = self.param_scaler.transform(X)
        Yi = self._predict_scaled(Xi)
        C = self.coeff_scaler.inverse_transform(Yi)

        if return_fallback_mask:
            return C, np.asarray(self.last_fallback_mask, dtype=bool)
        return C

    def predict_coeffs(self, mu):
        return self.predict_coeffs_batch(np.asarray(mu, dtype=np.float64).reshape(1, -1))[0]

    def reconstruct_vector(self, coeffs):
        coeffs = np.asarray(coeffs, dtype=np.float64).ravel()
        return coeffs @ self._basis_matrix()[:len(coeffs), :]

    def reconstruct_solution(self, coeffs):
        u = Function(self.solver.V_CG)
        u.vector().set_local(self.reconstruct_vector(coeffs))
        u.vector().apply("insert")
        return u

    def predict_vector(self, mu):
        return self.reconstruct_vector(self.predict_coeffs(mu))

    def predict_solution(self, mu):
        return self.reconstruct_solution(self.predict_coeffs(mu))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _evaluate_indices(self, mus, true_coeffs, snapshot_targets, ids, split_name):
        ids = np.asarray(ids, dtype=int)
        M = _podi_as_2d(mus)
        C_true = _podi_as_2d(true_coeffs)

        C_pred, fallback_mask = self.predict_coeffs_batch(M[ids], return_fallback_mask=True)
        Ct = C_true[ids]
        diff_C = C_pred - Ct

        coeff_mse = float(np.mean(diff_C ** 2))
        coeff_rmse = float(np.sqrt(coeff_mse))
        coeff_rel_global = float(np.linalg.norm(diff_C) / (np.linalg.norm(Ct) + 1e-30))
        coeff_rel_each = np.linalg.norm(diff_C, axis=1) / (np.linalg.norm(Ct, axis=1) + 1e-30)

        B = self._basis_matrix()
        U_pred = C_pred @ B
        U_proj = Ct @ B
        subspace_rel_each = np.linalg.norm(U_pred - U_proj, axis=1) / (np.linalg.norm(U_proj, axis=1) + 1e-30)

        out = dict(
            split=split_name,
            n_samples=int(len(ids)),
            coeff_mse=coeff_mse,
            coeff_rmse=coeff_rmse,
            coeff_rel_global=coeff_rel_global,
            subspace_field_rel_mean=float(np.mean(subspace_rel_each)),
            subspace_field_rel_median=float(np.median(subspace_rel_each)),
            subspace_field_rel_max=float(np.max(subspace_rel_each)),
            fallback_count=int(np.sum(fallback_mask)),
        )

        rows = []
        if snapshot_targets is not None:
            S = _podi_as_2d(snapshot_targets)[ids]
            total_rel_each = np.linalg.norm(U_pred - S, axis=1) / (np.linalg.norm(S, axis=1) + 1e-30)
            proj_floor_each = np.linalg.norm(U_proj - S, axis=1) / (np.linalg.norm(S, axis=1) + 1e-30)

            out.update(
                total_field_rel_mean=float(np.mean(total_rel_each)),
                total_field_rel_median=float(np.median(total_rel_each)),
                total_field_rel_max=float(np.max(total_rel_each)),
                projection_floor_mean=float(np.mean(proj_floor_each)),
                projection_floor_median=float(np.median(proj_floor_each)),
                projection_floor_max=float(np.max(proj_floor_each)),
                surrogate_gap_mean=float(np.mean(total_rel_each - proj_floor_each)),
                surrogate_gap_max=float(np.max(total_rel_each - proj_floor_each)),
            )
        else:
            total_rel_each = np.full(len(ids), np.nan)
            proj_floor_each = np.full(len(ids), np.nan)

        for local_k, global_i in enumerate(ids):
            rows.append(dict(
                split=split_name,
                sample_index=int(global_i),
                coeff_rel=float(coeff_rel_each[local_k]),
                subspace_field_rel=float(subspace_rel_each[local_k]),
                total_field_rel=float(total_rel_each[local_k]),
                projection_floor=float(proj_floor_each[local_k]),
                surrogate_gap=float(total_rel_each[local_k] - proj_floor_each[local_k]),
                used_fallback=bool(fallback_mask[local_k]),
            ))

        return out, rows

    def evaluate_splits(self, mus, true_coeffs, snapshot_targets=None, split_indices=None, save_csv=None):
        split_indices = split_indices or self.split_indices
        all_metrics, all_rows = {}, []

        print("\n" + "=" * 92)
        print(f"{'PODI-' + str(self.method).upper() + ' train/validation/test diagnostics':^92}")
        print("=" * 92)

        for split in ["train", "val", "test"]:
            if split not in split_indices:
                continue
            metrics, rows = self._evaluate_indices(
                mus, true_coeffs, snapshot_targets,
                split_indices[split],
                split,
            )
            all_metrics[split] = metrics
            all_rows.extend(rows)

            print(f"\n[{split.upper()}] n={metrics['n_samples']}")
            print(f"  coeff RMSE/global rel        : {metrics['coeff_rmse']:.4e} / {metrics['coeff_rel_global']:.4e}")
            print(f"  subspace field rel mean/max  : {metrics['subspace_field_rel_mean']:.4e} / {metrics['subspace_field_rel_max']:.4e}")
            print(f"  fallback count               : {metrics['fallback_count']} / {metrics['n_samples']}")
            if snapshot_targets is not None:
                print(f"  total field rel mean/median  : {metrics['total_field_rel_mean']:.4e} / {metrics['total_field_rel_median']:.4e}")
                print(f"  total field rel max          : {metrics['total_field_rel_max']:.4e}")
                print(f"  POD projection floor mean/max: {metrics['projection_floor_mean']:.4e} / {metrics['projection_floor_max']:.4e}")
                print(f"  surrogate gap mean/max       : {metrics['surrogate_gap_mean']:.4e} / {metrics['surrogate_gap_max']:.4e}")

        print("=" * 92 + "\n")

        df = pd.DataFrame(all_rows)

        if save_csv is not None:
            os.makedirs(os.path.dirname(save_csv), exist_ok=True)
            df.to_csv(save_csv, index=False)
            print(f"[PODI-{self.method.upper()}] Saved split diagnostics CSV → {save_csv}")

        if bool(self.config.get("save_plots", True)) and len(df) > 0:
            self.plot_split_errors(df, save_dir=_method_root(podi_dir, self.method))

        self.metrics = all_metrics
        self.error_df = df
        return all_metrics, df

    def print_worst_samples(self, df_errors, mus=None, n=10):
        if df_errors is None or len(df_errors) == 0:
            return

        sort_col = "total_field_rel" if "total_field_rel" in df_errors.columns else "subspace_field_rel"

        print("\n" + "=" * 92)
        print(f"{'Worst PODI-' + str(self.method).upper() + ' samples by relative field error':^92}")
        print("=" * 92)

        for split in ["train", "val", "test"]:
            d = df_errors[df_errors["split"] == split].copy()
            if len(d) == 0:
                continue
            d = d.sort_values(sort_col, ascending=False).head(int(n))
            print(f"\n[{split.upper()}] worst {min(int(n), len(d))} samples")
            cols = [
                "split", "sample_index", "total_field_rel", "projection_floor",
                "surrogate_gap", "subspace_field_rel", "coeff_rel", "used_fallback",
            ]
            cols = [c for c in cols if c in d.columns]
            print(d[cols].to_string(index=False))

            if mus is not None:
                M = np.asarray(mus, dtype=np.float64)
                print("  Parameter values:")
                for sample_idx in d["sample_index"].to_numpy(dtype=int):
                    print(f"    sample {sample_idx:4d}: mu = {np.array2string(M[sample_idx], precision=6, separator=', ')}")

        print("=" * 92 + "\n")

    def plot_split_errors(self, df_errors, save_dir=None):
        if save_dir is None:
            save_dir = _method_root(podi_dir, self.method)
        os.makedirs(save_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=140)

        for split in ["train", "val", "test"]:
            d = df_errors[df_errors["split"] == split]
            if len(d):
                y = d["total_field_rel"].to_numpy() if np.isfinite(d["total_field_rel"].to_numpy()).any() else d["subspace_field_rel"].to_numpy()
                ax.plot(np.arange(len(d)), y, marker="o", linewidth=1.3, label=split)

        ax.set_yscale("log")
        ax.set_xlabel("Sample index within split")
        ax.set_ylabel("Relative field error")
        ax.set_title(f"PODI-{self.method.upper()} split-wise field errors")
        ax.grid(True, which="both", linestyle=":")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"podi_{self.method}_split_errors.png"), dpi=300, bbox_inches="tight")
        plt.show()

    # ------------------------------------------------------------------
    # Save/load
    # ------------------------------------------------------------------
    def save(self, base_dir):
        root = _method_root(base_dir, self.method)
        os.makedirs(root, exist_ok=True)
        idx = max([
            int(d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d)) and d.isdigit()
        ] + [0]) + 1

        folder = os.path.join(root, str(idx))
        bdir = os.path.join(folder, "basis")
        os.makedirs(folder, exist_ok=True)
        os.makedirs(bdir, exist_ok=True)

        for i, phi in enumerate(self.reduced_basis[:self.n_basis]):
            np.save(os.path.join(bdir, f"basis_{i}.npy"), phi.vector().get_local())

        with open(os.path.join(folder, "model.pkl"), "wb") as f:
            pickle.dump(dict(
                method=self.method,
                config=self.config,
                interp_kwargs=self.interp_kwargs,
                param_scaler=self.param_scaler,
                coeff_scaler=self.coeff_scaler,
                interpolator=self.interpolator,
                nearest_interpolator=self.nearest_interpolator,
                n_basis=self.n_basis,
                split_indices=self.split_indices,
                training_summary=self.training_summary,
                metrics=self.metrics,
            ), f)

        if self.metrics:
            rows = []
            for split, m in self.metrics.items():
                row = dict(m)
                row["split"] = split
                rows.append(row)
            pd.DataFrame(rows).to_csv(os.path.join(folder, f"podi_{self.method}_metrics_summary.csv"), index=False)

        if self.error_df is not None:
            self.error_df.to_csv(os.path.join(folder, f"podi_{self.method}_train_val_test_errors.csv"), index=False)

        if self.training_summary:
            pd.DataFrame([self.training_summary]).to_csv(
                os.path.join(folder, f"podi_{self.method}_training_summary.csv"), index=False
            )

        print(f"[PODI-{self.method.upper()}] model saved → {folder}")
        return idx

    @classmethod
    def load(cls, solver, pod_index, method, base_dir):
        root = _method_root(base_dir, method)
        folder = os.path.join(root, str(pod_index))
        bdir = os.path.join(folder, "basis")

        basis = []
        for file in sorted(os.listdir(bdir), key=_podi_basis_file_key):
            vec = np.load(os.path.join(bdir, file))
            f = Function(solver.V_CG)
            f.vector().set_local(vec)
            f.vector().apply("insert")
            basis.append(f)

        with open(os.path.join(folder, "model.pkl"), "rb") as f:
            data = pickle.load(f)

        obj = cls(
            solver,
            reduced_basis=basis,
            n_basis=data["n_basis"],
            inner_product=solver.inner_product_CG,
            config=data.get("config", dict(PODI_CONFIG)),
        )
        obj.method = data.get("method", method)
        obj.interp_kwargs = data.get("interp_kwargs", {})
        obj.param_scaler = data["param_scaler"]
        obj.coeff_scaler = data["coeff_scaler"]
        obj.interpolator = data["interpolator"]
        obj.nearest_interpolator = data.get("nearest_interpolator", None)
        obj.split_indices = data.get("split_indices", {})
        obj.training_summary = data.get("training_summary", {})
        obj.metrics = data.get("metrics", {})

        print(f"[PODI-{method.upper()}] model loaded ← {folder}")
        return obj


# =============================================================================
# Live/load execution
# =============================================================================
method_list = list(PODI_CONFIG.get("method_list", ("rbf", "linear")))
model_indices = dict()
podi_models = dict()

# --- Prepare snapshot functions only for live ---
if choice_podi_run == "live":
    snapshot_functions = []
    C_I = None

    for vec in snaps_all_I:
        f = Function(solver.V_CG)
        f.vector().set_local(np.asarray(vec, dtype=np.float64).ravel())
        f.vector().apply("insert")
        snapshot_functions.append(f)

# --- Train or load all methods ---
for method in method_list:
    method = str(method).lower()
    interp_kwargs = dict(PODI_CONFIG.get("rbf_kwargs", {})) if method == "rbf" else dict(PODI_CONFIG.get("linear_kwargs", {}))

    if choice_podi_run == "live":
        print(f"\n[PODI-{method.upper()}] Training...")
        podi_rom = PODInterpReducedOrderModel(solver, Z_podI, N_podI, IP_I, config=PODI_CONFIG)

        if C_I is None:
            C_I = podi_rom.project_snapshots(snapshot_functions)
            print(f" Reduced coefficients computed. Shape: {C_I.shape}")

        podi_rom, podi_split_metrics, podi_error_df = podi_rom.fit_interpolator(
            mu_all_I,
            C_I,
            method=method,
            interp_kwargs=interp_kwargs,
            snapshot_targets=snaps_all_I,
        )

        idx = podi_rom.save(podi_dir)
        print(f"Saved PODI-{method.upper()} model at index {idx}.")
        podi_models[method] = podi_rom
        model_indices[method] = idx

        globals().update(
            podi_rom=podi_rom,
            podi_split_metrics=podi_split_metrics,
            podi_error_df=podi_error_df,
            C_I=C_I,
        )

    elif choice_podi_run == "load":
        idx = 1  # <-- set this to the correct index for each method if needed
        print(f"\n[PODI-{method.upper()}] Loading model index {idx}...")
        podi_rom = PODInterpReducedOrderModel.load(solver, pod_index=idx, method=method, base_dir=podi_dir)
        podi_models[method] = podi_rom
        model_indices[method] = idx
    else:
        raise ValueError(f"Unknown choice_podi_run={choice_podi_run!r}. Use 'live' or 'load'.")

# %% [markdown] Cell 81 | id: 97d90d3b
# ### POD-NN : Proper Orthogonal Decomposition with Artificial Neural Networks

# %% Cell 82 | id: da23ee7e
# Final single-seed pushed version, keeping the compact Project-2 workflow:
#   1) CPU/thread-safe PyTorch inside FEniCS/Jupyter.
#   2) One fixed train/validation/test split, no scaler leakage.
#   3) Improved default split: 80/10/10 for n_snapshots=150.
#   4) Stage 1: smoother coefficient-only training, μ -> normalized POD coefficients.
#   5) Stage 2: shorter, controlled metric-aware field-loss fine-tuning.
#   6) Single deterministic model seed = 100, split seed = 100. No multi-seed runs.
#   7) Full diagnostics on train/validation/test:
#        - coefficient error,
#        - POD-subspace field error,
#        - total field error against snapshots,
#        - POD projection floor,
#        - surrogate gap.
#   8) Worst-sample inspection, including the corresponding parameter values.
#   9) Save/load complete model, scalers, basis, split indices, diagnostics, plots.
# =============================================================================

choice_podnn_run, podnn_pod_folder = "live", 1   # "live" or "load"
print(f"PODNN: choice={choice_podnn},  N_basis={N_podnn},  n_snaps={len(snaps_all_N)}")

podnn_dir = os.path.join(solver.output_dir, "ROM_Non_Intrusive", "PODNN")
os.makedirs(podnn_dir, exist_ok=True)


# =============================================================================
# Configuration
# =============================================================================
PODNN_CONFIG = dict(
    # Single deterministic run.
    # Keep both fixed for reproducibility.
    seed=100,          # model initialization / PyTorch seed
    split_seed=100,    # train/val/test split seed

    # Recommended for 150 snapshots: more training support while preserving a real test set.
    # For this case: approximately 120 / 15 / 15.
    train_fraction=0.80,
    val_fraction=0.10,
    test_fraction=0.10,

    # Compact but slightly more expressive than (128, 64, 64).
    # Still small enough for CPU and n_snapshots=150.
    hidden_layers=(128, 96, 64),
    activation="elu",
    dropout=0.00,
    layer_norm=False,

    # Stage 1: smoother coefficient-only regression.
    coeff_epochs=2500,
    coeff_lr=5e-4,
    coeff_batch_size=256,          # full-batch for n_train < 256
    coeff_weight_decay=2e-6,
    coeff_scheduler_patience=120,
    coeff_early_stopping_patience=450,
    coeff_min_delta=5e-9,
    coeff_grad_clip=1.0,

    # Stage 2: controlled metric-aware fine-tuning.
    # Previous run showed field validation nearly flat after ~250-275 epochs.
    field_finetune=True,
    field_epochs=275,
    field_lr=3e-5,
    field_batch_size=256,
    field_weight_decay=5e-7,
    field_scheduler_patience=35,
    field_early_stopping_patience=80,
    field_min_delta=5e-10,
    field_grad_clip=1.0,
    field_loss_weight=1.0,
    coeff_anchor_weight=0.01,
    use_full_snapshot_target=True,  # True: compare C_pred@B against full snapshot S

    # Runtime / plots.
    device_preference="cpu",       # "cpu" recommended in FEniCS/Jupyter; can use "auto", "cuda", "mps"
    save_plots=True,
    save_csv=True,
    log_every=50,

    # Diagnostics.
    print_worst_samples=True,
    n_worst_samples=10,
)


# =============================================================================
# Utilities
# =============================================================================
def _podnn_seed(seed=100):
    if "set_seed" in globals():
        set_seed(seed)
    else:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))


def _podnn_device(preference="cpu"):
    pref = str(preference).lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "mps":
        return torch.device("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError("device_preference must be one of: 'cpu', 'cuda', 'mps', 'auto'.")


def _podnn_activation(name_or_cls):
    if isinstance(name_or_cls, str):
        acts = dict(tanh=nn.Tanh, relu=nn.ReLU, elu=nn.ELU, gelu=nn.GELU, silu=nn.SiLU)
        key = name_or_cls.lower()
        if key not in acts:
            raise ValueError(f"Unsupported PODNN activation '{name_or_cls}'. Use tanh/relu/elu/gelu/silu.")
        return acts[key]
    return name_or_cls


def _podnn_mlp(input_dim, output_dim, hidden_layers=(128, 96, 64), activation="elu",
               dropout=0.0, layer_norm=False):
    act = _podnn_activation(activation)
    dims = [int(input_dim), *map(int, hidden_layers), int(output_dim)]
    layers = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            if layer_norm:
                layers.append(nn.LayerNorm(b))
            layers.append(act())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
    return nn.Sequential(*layers)


def _podnn_clone_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _podnn_check_fractions(train_fraction, val_fraction, test_fraction):
    vals = np.array([train_fraction, val_fraction, test_fraction], dtype=float)
    if np.any(vals <= 0.0):
        raise ValueError("train_fraction, val_fraction and test_fraction must all be positive.")
    if not np.isclose(vals.sum(), 1.0):
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1.0.")


def _podnn_split_indices(n_samples, train_fraction, val_fraction, test_fraction, seed):
    _podnn_check_fractions(train_fraction, val_fraction, test_fraction)
    idx = np.arange(int(n_samples))
    train_idx, tmp_idx = train_test_split(
        idx, train_size=float(train_fraction), random_state=int(seed), shuffle=True
    )

    # Split remaining set into validation and test.
    test_ratio_inside_tmp = float(test_fraction) / float(val_fraction + test_fraction)
    val_idx, test_idx = train_test_split(
        tmp_idx, test_size=test_ratio_inside_tmp, random_state=int(seed) + 1, shuffle=True
    )
    return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int), np.asarray(test_idx, dtype=int)


def _podnn_rel_rows(A, B, eps=1e-30):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    return np.linalg.norm(A - B, axis=1) / (np.linalg.norm(B, axis=1) + eps)


# =============================================================================
# POD-NN ROM class
# =============================================================================
class PODNNReducedOrderModel:
    def __init__(self, solver, reduced_basis, n_basis, inner_product, config=None):
        self.solver = solver
        self.reduced_basis = reduced_basis
        self.n_basis = int(n_basis)
        self.inner_product = inner_product
        self.config = dict(PODNN_CONFIG if config is None else config)

        self.model = None
        self.device = _podnn_device(self.config.get("device_preference", "cpu"))

        self.X_scaler = None
        self.coeff_scaler = None
        self.coeff_mean = None
        self.coeff_std = None

        self.nn_hyper = {}
        self.split_indices = {}
        self.training_history = {}
        self.metrics = {}
        self.error_df = None
        self._basis_matrix_cache = None

    # ---------------------------------------------------------------------
    # POD projection and reconstruction
    # ---------------------------------------------------------------------
    def _basis_matrix(self):
        """Rows are POD basis vectors, shape = (n_basis, n_dofs)."""
        if self._basis_matrix_cache is None or self._basis_matrix_cache.shape[0] != self.n_basis:
            self._basis_matrix_cache = np.vstack([
                self.reduced_basis[i].vector().get_local()
                for i in range(self.n_basis)
            ]).astype(np.float64)
        return self._basis_matrix_cache

    def project_snapshots(self, snapshots):
        print(" Projecting snapshots onto reduced basis...")
        Z = self.reduced_basis[:self.n_basis]
        MZ = [self.inner_product * z.vector() for z in Z]
        A_N = np.array([[zi.vector().inner(mzj) for mzj in MZ] for zi in Z], dtype=np.float64)

        C = np.array([
            np.linalg.solve(A_N, np.array([s.vector().inner(mzj) for mzj in MZ], dtype=np.float64))
            for s in snapshots
        ], dtype=np.float64)

        print(f" Full-order space dim: {self.solver.V_CG.dim()}")
        print(f" Reduced basis vector dim: {Z[0].vector().size()}")
        print(f" Inner product shape: {self.inner_product.size(0)} × {self.inner_product.size(1)}")
        print(f" Z_N shape: {np.column_stack([z.vector().get_local() for z in Z]).shape}")
        print(f" Solution vector shape: {snapshots[0].vector().get_local().shape}")
        return C

    # ---------------------------------------------------------------------
    # Data preparation
    # ---------------------------------------------------------------------
    def _prepare_scalers(self, X, C, train_idx):
        self.X_scaler = StandardScaler().fit(X[train_idx])
        self.coeff_scaler = StandardScaler().fit(C[train_idx])
        self.coeff_scaler.scale_[self.coeff_scaler.scale_ < 1e-12] = 1.0

        Xs = self.X_scaler.transform(X).astype(np.float32)
        Cs = self.coeff_scaler.transform(C).astype(np.float32)

        self.coeff_mean = self.coeff_scaler.mean_.astype(np.float64)
        self.coeff_std = self.coeff_scaler.scale_.astype(np.float64)
        self.coeff_std[self.coeff_std < 1e-12] = 1.0
        return Xs, Cs

    def _make_loader(self, Xs, Cs, ids, batch_size, shuffle=False, S=None):
        ids = np.asarray(ids, dtype=int)
        if S is None:
            ds = TensorDataset(torch.tensor(Xs[ids]), torch.tensor(Cs[ids]))
        else:
            ds = TensorDataset(torch.tensor(Xs[ids]), torch.tensor(Cs[ids]), torch.tensor(S[ids]))
        return DataLoader(ds, batch_size=int(batch_size), shuffle=bool(shuffle))

    def _build_model(self, input_dim):
        self.nn_hyper = dict(
            input_dim=int(input_dim),
            hidden_layers=tuple(self.config.get("hidden_layers", (128, 96, 64))),
            activation=self.config.get("activation", "elu"),
            dropout=float(self.config.get("dropout", 0.0)),
            layer_norm=bool(self.config.get("layer_norm", False)),
        )
        self.model = _podnn_mlp(
            input_dim,
            self.n_basis,
            self.nn_hyper["hidden_layers"],
            self.nn_hyper["activation"],
            dropout=self.nn_hyper["dropout"],
            layer_norm=self.nn_hyper["layer_norm"],
        ).to(self.device)
        return self.model

    # ---------------------------------------------------------------------
    # Loss machinery
    # ---------------------------------------------------------------------
    def _make_loss_function(self, use_field_loss, snapshot_target_mode, field_loss_weight, coeff_anchor_weight):
        coeff_loss_fn = nn.MSELoss()
        use_field_loss = bool(use_field_loss)

        coeff_mean_t = coeff_std_t = B_t = None
        if use_field_loss:
            coeff_mean_t = torch.tensor(self.coeff_mean, dtype=torch.float32, device=self.device)
            coeff_std_t = torch.tensor(self.coeff_std, dtype=torch.float32, device=self.device)
            B_t = torch.tensor(self._basis_matrix(), dtype=torch.float32, device=self.device)

        def inv_coeff(y_scaled):
            return y_scaled * coeff_std_t + coeff_mean_t

        def batch_loss(batch):
            xb, yb = batch[0].to(self.device), batch[1].to(self.device)
            sb = batch[2].to(self.device) if (use_field_loss and len(batch) == 3) else None

            pred_s = self.model(xb)
            coeff_loss = coeff_loss_fn(pred_s, yb)

            if not use_field_loss:
                return coeff_loss, torch.tensor(0.0, device=self.device), coeff_loss

            pred_c = inv_coeff(pred_s)
            true_c = inv_coeff(yb)
            pred_u = pred_c @ B_t

            if snapshot_target_mode and sb is not None:
                true_u = sb
            else:
                true_u = true_c @ B_t

            field_loss = torch.mean(
                torch.sum((pred_u - true_u) ** 2, dim=1) /
                (torch.sum(true_u ** 2, dim=1) + 1e-24)
            )
            total = float(field_loss_weight) * field_loss + float(coeff_anchor_weight) * coeff_loss
            return total, field_loss, coeff_loss

        return batch_loss

    def _train_stage(self, train_loader, val_loader, *, stage_name, epochs, lr, weight_decay,
                     scheduler_patience, early_stopping_patience, min_delta, grad_clip,
                     use_field_loss, field_loss_weight, coeff_anchor_weight, log_every):
        optimizer = optim.AdamW(self.model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=int(scheduler_patience)
        )
        batch_loss = self._make_loss_function(
            use_field_loss=use_field_loss,
            snapshot_target_mode=bool(self.config.get("use_full_snapshot_target", True)),
            field_loss_weight=field_loss_weight,
            coeff_anchor_weight=coeff_anchor_weight,
        )

        hist = {k: [] for k in ["train", "val", "train_field", "val_field", "train_coeff", "val_coeff", "lr"]}
        best_val, best_epoch, wait, best_state = np.inf, 0, 0, None

        print(f" Training POD-NN stage: {stage_name}")
        t0 = time.perf_counter()

        for epoch in range(1, int(epochs) + 1):
            self.model.train()
            tr_sum = tr_field = tr_coeff = tr_count = 0.0

            for batch in train_loader:
                optimizer.zero_grad()
                loss, field_loss, coeff_loss = batch_loss(batch)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip))
                optimizer.step()

                bs = batch[0].shape[0]
                tr_sum += float(loss.item()) * bs
                tr_field += float(field_loss.item()) * bs
                tr_coeff += float(coeff_loss.item()) * bs
                tr_count += bs

            self.model.eval()
            va_sum = va_field = va_coeff = va_count = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    loss, field_loss, coeff_loss = batch_loss(batch)
                    bs = batch[0].shape[0]
                    va_sum += float(loss.item()) * bs
                    va_field += float(field_loss.item()) * bs
                    va_coeff += float(coeff_loss.item()) * bs
                    va_count += bs

            train_loss = tr_sum / max(tr_count, 1)
            val_loss = va_sum / max(va_count, 1)
            train_field = tr_field / max(tr_count, 1)
            val_field = va_field / max(va_count, 1)
            train_coeff = tr_coeff / max(tr_count, 1)
            val_coeff = va_coeff / max(va_count, 1)

            scheduler.step(val_loss)
            curr_lr = optimizer.param_groups[0]["lr"]

            hist["train"].append(train_loss)
            hist["val"].append(val_loss)
            hist["train_field"].append(train_field)
            hist["val_field"].append(val_field)
            hist["train_coeff"].append(train_coeff)
            hist["val_coeff"].append(val_coeff)
            hist["lr"].append(curr_lr)

            if epoch == 1 or epoch % int(log_every) == 0 or epoch == int(epochs):
                print(
                    f" [{stage_name}] Epoch {epoch:5d}/{int(epochs):5d} │ "
                    f"Train: {train_loss:.4e} │ Val: {val_loss:.4e} │ "
                    f"ValField: {val_field:.4e} │ ValCoeff: {val_coeff:.4e} │ "
                    f"ValCoeffRMSE: {np.sqrt(max(val_coeff, 0.0)):.4e} │ "
                    f"lr={curr_lr:.2e}"
                )

            if val_loss < best_val - float(min_delta):
                best_val = val_loss
                best_epoch = epoch
                wait = 0
                best_state = _podnn_clone_state(self.model)
            else:
                wait += 1

            if wait >= int(early_stopping_patience):
                print(f" [{stage_name}] Early stopping at epoch {epoch}; best epoch={best_epoch}, best val={best_val:.4e}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

        elapsed = time.perf_counter() - t0
        hist.update(best_val=float(best_val), best_epoch=int(best_epoch), elapsed_sec=float(elapsed))
        print(f" [{stage_name}] complete in {elapsed:.2f} s | best val={best_val:.4e} at epoch {best_epoch}")
        return hist

    # ---------------------------------------------------------------------
    # Full robust single-seed fit
    # ---------------------------------------------------------------------
    def fit(self, training_params, reduced_coeffs, snapshot_targets=None, split_indices=None):
        cfg = self.config
        model_seed = int(cfg.get("seed", 100))
        split_seed = int(cfg.get("split_seed", cfg.get("seed", 100)))

        _podnn_seed(model_seed)
        self.device = _podnn_device(cfg.get("device_preference", "cpu"))
        print(f" Using device: {self.device}")
        print(f" Model seed: {model_seed} | Split seed: {split_seed}")

        X = np.asarray(training_params, dtype=np.float64)
        C = np.asarray(reduced_coeffs, dtype=np.float64)
        if C.ndim == 1:
            C = C.reshape(-1, 1)

        if X.shape[0] != C.shape[0]:
            raise ValueError(f"training_params has {X.shape[0]} rows, but reduced_coeffs has {C.shape[0]} rows.")
        if C.shape[1] != self.n_basis:
            raise ValueError(f"reduced_coeffs has {C.shape[1]} columns, but this POD-NN uses n_basis={self.n_basis}.")

        S = None
        if snapshot_targets is not None:
            S = np.asarray(snapshot_targets, dtype=np.float64)
            if S.ndim == 1:
                S = S.reshape(1, -1)
            if S.shape[0] != X.shape[0]:
                raise ValueError(f"snapshot_targets has {S.shape[0]} rows, expected {X.shape[0]}.")
            if S.shape[1] != self.solver.V_CG.dim():
                raise ValueError(f"snapshot_targets has {S.shape[1]} DOFs, expected solver.V_CG.dim()={self.solver.V_CG.dim()}.")

        if split_indices is None:
            train_idx, val_idx, test_idx = _podnn_split_indices(
                X.shape[0],
                cfg.get("train_fraction", 0.80),
                cfg.get("val_fraction", 0.10),
                cfg.get("test_fraction", 0.10),
                split_seed,
            )
        else:
            train_idx = np.asarray(split_indices["train"], dtype=int)
            val_idx = np.asarray(split_indices["val"], dtype=int)
            test_idx = np.asarray(split_indices["test"], dtype=int)

        self.split_indices = dict(train=train_idx.tolist(), val=val_idx.tolist(), test=test_idx.tolist())

        print(
            f" Splitting data into train/validation/test sets "
            f"({len(train_idx)} / {len(val_idx)} / {len(test_idx)})..."
        )

        Xs, Cs = self._prepare_scalers(X, C, train_idx)
        self._build_model(input_dim=X.shape[1])

        # Stage 1: coefficient-only training.
        bs1 = int(cfg.get("coeff_batch_size", 256))
        shuffle1 = bs1 < len(train_idx)
        train_loader = self._make_loader(Xs, Cs, train_idx, bs1, shuffle=shuffle1, S=None)
        val_loader = self._make_loader(Xs, Cs, val_idx, bs1, shuffle=False, S=None)

        coeff_hist = self._train_stage(
            train_loader, val_loader,
            stage_name="coeff",
            epochs=cfg.get("coeff_epochs", 2500),
            lr=cfg.get("coeff_lr", 5e-4),
            weight_decay=cfg.get("coeff_weight_decay", 2e-6),
            scheduler_patience=cfg.get("coeff_scheduler_patience", 120),
            early_stopping_patience=cfg.get("coeff_early_stopping_patience", 450),
            min_delta=cfg.get("coeff_min_delta", 5e-9),
            grad_clip=cfg.get("coeff_grad_clip", 1.0),
            use_field_loss=False,
            field_loss_weight=0.0,
            coeff_anchor_weight=1.0,
            log_every=cfg.get("log_every", 50),
        )

        # Stage 2: field-aware fine-tuning.
        field_hist = None
        if bool(cfg.get("field_finetune", True)):
            if S is None:
                print(" Field fine-tuning requested, but snapshot_targets=None. Skipping field stage.")
            else:
                S32 = S.astype(np.float32, copy=False)
                bs2 = int(cfg.get("field_batch_size", 256))
                shuffle2 = bs2 < len(train_idx)
                train_loader_f = self._make_loader(Xs, Cs, train_idx, bs2, shuffle=shuffle2, S=S32)
                val_loader_f = self._make_loader(Xs, Cs, val_idx, bs2, shuffle=False, S=S32)

                field_hist = self._train_stage(
                    train_loader_f, val_loader_f,
                    stage_name="field",
                    epochs=cfg.get("field_epochs", 275),
                    lr=cfg.get("field_lr", 3e-5),
                    weight_decay=cfg.get("field_weight_decay", 5e-7),
                    scheduler_patience=cfg.get("field_scheduler_patience", 35),
                    early_stopping_patience=cfg.get("field_early_stopping_patience", 80),
                    min_delta=cfg.get("field_min_delta", 5e-10),
                    grad_clip=cfg.get("field_grad_clip", 1.0),
                    use_field_loss=True,
                    field_loss_weight=cfg.get("field_loss_weight", 1.0),
                    coeff_anchor_weight=cfg.get("coeff_anchor_weight", 0.01),
                    log_every=max(1, min(int(cfg.get("log_every", 50)), 25)),
                )

        self.training_history = dict(
            coeff=coeff_hist,
            field=field_hist,
            config=cfg,
            split_indices=self.split_indices,
        )

        if bool(cfg.get("save_plots", True)):
            self.plot_training_history(save_dir=podnn_dir)

        # Final diagnostics on train, val, test.
        metrics, df_errors = self.evaluate_splits(
            X, C, snapshot_targets=S,
            split_indices=self.split_indices,
            save_csv=os.path.join(podnn_dir, "podnn_train_val_test_errors.csv")
            if bool(cfg.get("save_csv", True)) else None,
        )
        self.metrics = metrics
        self.error_df = df_errors

        if bool(cfg.get("print_worst_samples", True)):
            self.print_worst_samples(df_errors, mus=X, n=int(cfg.get("n_worst_samples", 10)))

        return self, metrics, df_errors

    # ---------------------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------------------
    def predict_reduced_coefficients_batch(self, mus, batch_size=4096):
        if self.model is None or self.X_scaler is None:
            raise RuntimeError("POD-NN model is not trained/loaded.")

        X = np.asarray(mus, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        Xs = self.X_scaler.transform(X).astype(np.float32)
        device = next(self.model.parameters()).device
        preds = []

        self.model.eval()
        with torch.no_grad():
            for i in range(0, Xs.shape[0], int(batch_size)):
                xb = torch.tensor(Xs[i:i + int(batch_size)], dtype=torch.float32, device=device)
                yb = self.model(xb).cpu().numpy()
                preds.append(yb)

        Ys = np.vstack(preds)
        return Ys * self.coeff_std.reshape(1, -1) + self.coeff_mean.reshape(1, -1)

    def predict_reduced_coefficients(self, mu):
        return self.predict_reduced_coefficients_batch(np.asarray(mu, dtype=np.float64).reshape(1, -1))[0]

    def reconstruct_vector(self, reduced_coefficients):
        coeffs = np.asarray(reduced_coefficients, dtype=np.float64).ravel()
        return coeffs @ self._basis_matrix()[:len(coeffs), :]

    def reconstruct_solution(self, reduced_coefficients):
        full_solution_vector = self.reconstruct_vector(reduced_coefficients)
        rom_solution = Function(self.solver.V_CG)
        rom_solution.vector().set_local(full_solution_vector)
        rom_solution.vector().apply("insert")
        return rom_solution

    def predict_vector(self, mu):
        return self.reconstruct_vector(self.predict_reduced_coefficients(mu))

    def predict_solution(self, mu):
        return self.reconstruct_solution(self.predict_reduced_coefficients(mu))

    # ---------------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------------
    def _evaluate_indices(self, mus, true_coeffs, snapshot_targets, ids, split_name):
        ids = np.asarray(ids, dtype=int)
        M = np.asarray(mus, dtype=np.float64)
        C_true = np.asarray(true_coeffs, dtype=np.float64)
        C_pred = self.predict_reduced_coefficients_batch(M[ids])

        Ct = C_true[ids]
        diff_C = C_pred - Ct

        coeff_mse = float(np.mean(diff_C**2))
        coeff_rmse = float(np.sqrt(coeff_mse))
        coeff_rel_global = float(np.linalg.norm(diff_C) / (np.linalg.norm(Ct) + 1e-30))
        coeff_rel_each = np.linalg.norm(diff_C, axis=1) / (np.linalg.norm(Ct, axis=1) + 1e-30)

        B = self._basis_matrix()
        U_pred = C_pred @ B
        U_proj = Ct @ B

        subspace_rel_each = np.linalg.norm(U_pred - U_proj, axis=1) / (np.linalg.norm(U_proj, axis=1) + 1e-30)

        out = dict(
            split=split_name,
            n_samples=int(len(ids)),
            coeff_mse=coeff_mse,
            coeff_rmse=coeff_rmse,
            coeff_rel_global=coeff_rel_global,
            subspace_field_rel_mean=float(np.mean(subspace_rel_each)),
            subspace_field_rel_median=float(np.median(subspace_rel_each)),
            subspace_field_rel_max=float(np.max(subspace_rel_each)),
        )

        rows = []
        if snapshot_targets is not None:
            S = np.asarray(snapshot_targets, dtype=np.float64)[ids]
            total_rel_each = np.linalg.norm(U_pred - S, axis=1) / (np.linalg.norm(S, axis=1) + 1e-30)
            proj_floor_each = np.linalg.norm(U_proj - S, axis=1) / (np.linalg.norm(S, axis=1) + 1e-30)

            out.update(
                total_field_rel_mean=float(np.mean(total_rel_each)),
                total_field_rel_median=float(np.median(total_rel_each)),
                total_field_rel_max=float(np.max(total_rel_each)),
                projection_floor_mean=float(np.mean(proj_floor_each)),
                projection_floor_median=float(np.median(proj_floor_each)),
                projection_floor_max=float(np.max(proj_floor_each)),
                surrogate_gap_mean=float(np.mean(total_rel_each - proj_floor_each)),
                surrogate_gap_max=float(np.max(total_rel_each - proj_floor_each)),
            )
        else:
            total_rel_each = np.full(len(ids), np.nan)
            proj_floor_each = np.full(len(ids), np.nan)

        for local_k, global_i in enumerate(ids):
            rows.append(dict(
                split=split_name,
                sample_index=int(global_i),
                coeff_rel=float(coeff_rel_each[local_k]),
                subspace_field_rel=float(subspace_rel_each[local_k]),
                total_field_rel=float(total_rel_each[local_k]),
                projection_floor=float(proj_floor_each[local_k]),
                surrogate_gap=float(total_rel_each[local_k] - proj_floor_each[local_k]),
            ))

        return out, rows

    def evaluate_splits(self, mus, true_coeffs, snapshot_targets=None, split_indices=None, save_csv=None):
        split_indices = split_indices or self.split_indices
        all_metrics, all_rows = {}, []

        print("\n" + "=" * 92)
        print(f"{'POD-NN train/validation/test diagnostics':^92}")
        print("=" * 92)

        for split in ["train", "val", "test"]:
            if split not in split_indices:
                continue
            metrics, rows = self._evaluate_indices(
                mus, true_coeffs, snapshot_targets,
                split_indices[split],
                split,
            )
            all_metrics[split] = metrics
            all_rows.extend(rows)

            print(f"\n[{split.upper()}] n={metrics['n_samples']}")
            print(f"  coeff RMSE/global rel        : {metrics['coeff_rmse']:.4e} / {metrics['coeff_rel_global']:.4e}")
            print(f"  subspace field rel mean/max  : {metrics['subspace_field_rel_mean']:.4e} / {metrics['subspace_field_rel_max']:.4e}")
            if snapshot_targets is not None:
                print(f"  total field rel mean/median  : {metrics['total_field_rel_mean']:.4e} / {metrics['total_field_rel_median']:.4e}")
                print(f"  total field rel max          : {metrics['total_field_rel_max']:.4e}")
                print(f"  POD projection floor mean/max: {metrics['projection_floor_mean']:.4e} / {metrics['projection_floor_max']:.4e}")
                print(f"  surrogate gap mean/max       : {metrics['surrogate_gap_mean']:.4e} / {metrics['surrogate_gap_max']:.4e}")

        print("=" * 92 + "\n")

        df = pd.DataFrame(all_rows)

        if save_csv is not None:
            df.to_csv(save_csv, index=False)
            print(f"[POD-NN] Saved split diagnostics CSV → {save_csv}")

        if bool(self.config.get("save_plots", True)) and len(df) > 0:
            self.plot_split_errors(df, save_dir=podnn_dir)

        self.metrics = all_metrics
        return all_metrics, df

    def print_worst_samples(self, df_errors, mus=None, n=10):
        if df_errors is None or len(df_errors) == 0:
            return

        sort_col = "total_field_rel" if "total_field_rel" in df_errors.columns else "subspace_field_rel"

        print("\n" + "=" * 92)
        print(f"{'Worst POD-NN samples by relative field error':^92}")
        print("=" * 92)

        for split in ["train", "val", "test"]:
            d = df_errors[df_errors["split"] == split].copy()
            if len(d) == 0:
                continue
            d = d.sort_values(sort_col, ascending=False).head(int(n))
            print(f"\n[{split.upper()}] worst {min(int(n), len(d))} samples")
            cols = ["split", "sample_index", "total_field_rel", "projection_floor", "surrogate_gap", "subspace_field_rel", "coeff_rel"]
            cols = [c for c in cols if c in d.columns]
            print(d[cols].to_string(index=False))

            if mus is not None:
                M = np.asarray(mus, dtype=np.float64)
                print("  Parameter values:")
                for sample_idx in d["sample_index"].to_numpy(dtype=int):
                    print(f"    sample {sample_idx:4d}: mu = {np.array2string(M[sample_idx], precision=6, separator=', ')}")

        print("=" * 92 + "\n")

    # ---------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------
    def plot_training_history(self, save_dir=podnn_dir):
        os.makedirs(save_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=140)

        if self.training_history.get("coeff"):
            h = self.training_history["coeff"]
            ax.plot(h["train"], label="Coeff train", linestyle="-")
            ax.plot(h["val"], label="Coeff val", linestyle="--")

        if self.training_history.get("field"):
            h = self.training_history["field"]
            offset = len(self.training_history["coeff"]["train"]) if self.training_history.get("coeff") else 0
            x = np.arange(1, len(h["train"]) + 1) + offset
            ax.plot(x, h["train"], label="Field train", linestyle="-")
            ax.plot(x, h["val"], label="Field val", linestyle="--")

        ax.set_yscale("log")
        ax.set_xlabel("Epoch / stage-continuation epoch")
        ax.set_ylabel("Loss")
        ax.set_title("POD-NN training history")
        ax.grid(True, linestyle=":")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "podnn_training_history.png"), dpi=300, bbox_inches="tight")
        plt.show()

    def plot_split_errors(self, df_errors, save_dir=podnn_dir):
        os.makedirs(save_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=140)

        for split in ["train", "val", "test"]:
            d = df_errors[df_errors["split"] == split]
            if len(d):
                y = d["total_field_rel"].to_numpy() if np.isfinite(d["total_field_rel"].to_numpy()).any() else d["subspace_field_rel"].to_numpy()
                ax.plot(np.arange(len(d)), y, marker="o", linewidth=1.3, label=split)

        ax.set_yscale("log")
        ax.set_xlabel("Sample index within split")
        ax.set_ylabel("Relative field error")
        ax.set_title("POD-NN split-wise field errors")
        ax.grid(True, which="both", linestyle=":")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "podnn_split_errors.png"), dpi=300, bbox_inches="tight")
        plt.show()

    # ---------------------------------------------------------------------
    # Save/load
    # ---------------------------------------------------------------------
    def save(self, pod_root=podnn_dir):
        os.makedirs(pod_root, exist_ok=True)
        existing = [
            int(d) for d in os.listdir(pod_root)
            if os.path.isdir(os.path.join(pod_root, d)) and d.isdigit()
        ]
        pod_index = max(existing) + 1 if existing else 1
        folder = os.path.join(pod_root, str(pod_index))
        os.makedirs(folder, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(folder, "podnn_weights.pth"))

        with open(os.path.join(folder, "podnn_scalers_metadata.pkl"), "wb") as f:
            pickle.dump(dict(
                X_scaler=self.X_scaler,
                coeff_scaler=self.coeff_scaler,
                coeff_mean=self.coeff_mean,
                coeff_std=self.coeff_std,
                nn_hyper=self.nn_hyper,
                config=self.config,
                split_indices=self.split_indices,
                training_history=self.training_history,
                metrics=self.metrics,
            ), f)

        # Backward-friendly scaler file name.
        with open(os.path.join(folder, "podnn_scalers.pkl"), "wb") as f:
            pickle.dump(dict(
                X_scaler=self.X_scaler,
                coeff_scaler=self.coeff_scaler,
                coeff_mean=self.coeff_mean,
                coeff_std=self.coeff_std,
                nn_hyper=self.nn_hyper,
                training_history=self.training_history,
            ), f)

        basis_dir = os.path.join(folder, "basis")
        os.makedirs(basis_dir, exist_ok=True)
        for i, bf in enumerate(self.reduced_basis[:self.n_basis]):
            np.save(os.path.join(basis_dir, f"basis_{i}.npy"), bf.vector().get_local())

        # Metrics summary and per-sample diagnostics inside the saved model folder.
        if self.metrics:
            summary_rows = []
            for split, m in self.metrics.items():
                row = dict(m)
                row["split"] = split
                summary_rows.append(row)
            pd.DataFrame(summary_rows).to_csv(os.path.join(folder, "podnn_metrics_summary.csv"), index=False)

        if self.error_df is not None:
            self.error_df.to_csv(os.path.join(folder, "podnn_train_val_test_errors.csv"), index=False)

        print(f"[POD-NN] model saved → {folder}")
        return pod_index

    @classmethod
    def load(cls, solver, pod_index, pod_root=podnn_dir, architecture_kwargs=None):
        folder = os.path.join(pod_root, str(pod_index))
        basis_dir = os.path.join(folder, "basis")

        loaded_basis = []
        for fname in sorted(os.listdir(basis_dir)):
            vec = np.load(os.path.join(basis_dir, fname))
            f = Function(solver.V_CG)
            f.vector()[:] = vec
            loaded_basis.append(f)

        meta_path = os.path.join(folder, "podnn_scalers_metadata.pkl")
        if not os.path.exists(meta_path):
            meta_path = os.path.join(folder, "podnn_scalers.pkl")

        with open(meta_path, "rb") as f:
            data = pickle.load(f)

        cfg = data.get("config", dict(PODNN_CONFIG))
        model = cls(solver, reduced_basis=loaded_basis, n_basis=len(loaded_basis), inner_product=solver.inner_product_CG, config=cfg)

        model.X_scaler = data["X_scaler"]
        model.coeff_scaler = data.get("coeff_scaler", None)
        model.coeff_mean = data["coeff_mean"]
        model.coeff_std = data["coeff_std"]
        model.split_indices = data.get("split_indices", {})
        model.training_history = data.get("training_history", {})
        model.metrics = data.get("metrics", {})

        if architecture_kwargs is not None:
            input_dim = int(architecture_kwargs["input_dim"])
            h = architecture_kwargs["nn_hyper"]
            hidden_layers = h.get("hidden_layers", (64, 64, 64))
            activation = h.get("activation", "tanh")
            dropout = h.get("dropout", 0.0)
            layer_norm = h.get("layer_norm", False)
        else:
            h = data.get("nn_hyper", {})
            input_dim = int(h.get("input_dim", getattr(model.X_scaler, "n_features_in_", len(model.X_scaler.mean_))))
            hidden_layers = h.get("hidden_layers", (64, 64, 64))
            activation = h.get("activation", "tanh")
            dropout = h.get("dropout", 0.0)
            layer_norm = h.get("layer_norm", False)

        model.nn_hyper = dict(
            input_dim=int(input_dim),
            hidden_layers=tuple(hidden_layers),
            activation=activation,
            dropout=float(dropout),
            layer_norm=bool(layer_norm),
        )
        model.model = _podnn_mlp(
            input_dim,
            model.n_basis,
            hidden_layers,
            activation,
            dropout=dropout,
            layer_norm=layer_norm,
        )

        try:
            state_dict = torch.load(os.path.join(folder, "podnn_weights.pth"), map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(os.path.join(folder, "podnn_weights.pth"), map_location="cpu")

        model.model.load_state_dict(state_dict)
        model.model.to(model.device).eval()
        print(f"[POD-NN] model loaded ← {folder}")
        return model


# =============================================================================
# Live/load execution : single-seed only
# =============================================================================
if choice_podnn_run == "live":
    SEED = int(PODNN_CONFIG.get("seed", 100))
    SPLIT_SEED = int(PODNN_CONFIG.get("split_seed", SEED))
    print(f"\n[Step 1] Preparing training data... (Model seed = {SEED}, Split seed = {SPLIT_SEED})")
    _podnn_seed(SEED)

    snapshots = []
    for vec in snaps_all_N:
        f = Function(solver.V_CG)
        f.vector().set_local(np.asarray(vec, dtype=np.float64).ravel())
        f.vector().apply("insert")
        snapshots.append(f)

    print(f" Collected {len(snapshots)} snapshots with N_basis = {N_podnn}.")

    print("\n[Step 2] Initializing single-seed robust POD-NN model...")
    pod_nn_rom = PODNNReducedOrderModel(solver, Z_podnn, N_podnn, IP_N, config=PODNN_CONFIG)
    print(f" Initialized POD-NN with N_basis = {N_podnn}")

    print("\n[Step 3] Projecting snapshots onto reduced basis...")
    reduced_coeffs = pod_nn_rom.project_snapshots(snapshots)
    print(f" Reduced coefficients computed. Shape: {reduced_coeffs.shape}")

    print("\n[Step 4] Training single-seed robust POD-NN with fixed train/validation/test split...")
    pod_nn_rom, podnn_split_metrics, podnn_error_df = pod_nn_rom.fit(
        training_params=mu_all_N,
        reduced_coeffs=reduced_coeffs,
        snapshot_targets=snaps_all_N,
    )

    print("\n[Step 5] Saving single-seed robust POD-NN ROM...")
    podnn_index = pod_nn_rom.save()
    print(f" Saved POD-NN ROM into {os.path.join(podnn_dir, str(podnn_index))}/")

    # Useful globals for later comparison/error-analysis cells.
    globals().update(
        pod_nn_rom=pod_nn_rom,
        podnn_index=podnn_index,
        podnn_split_metrics=podnn_split_metrics,
        podnn_error_df=podnn_error_df,
        podnn_reduced_coeffs=reduced_coeffs,
    )

elif choice_podnn_run == "load":
    print(f"\n[Load] Loading POD-NN ROM from {os.path.join(podnn_dir, str(podnn_pod_folder))}/...")
    pod_nn_rom = PODNNReducedOrderModel.load(solver, pod_index=podnn_pod_folder)
    print(" Loaded POD-NN ROM successfully.")
else:
    raise ValueError(f"Unknown choice: {choice_podnn_run}. Use 'live' or 'load'.")

# %% [markdown] Cell 83 | id: d1e04f22
# ### POD-GPR : Proper Orthogonal Decomposition with Gaussian Process Regression

# %% Cell 84 | id: fc231376

# =============================================================================
# POD-GPR : Robust train/validation/test POD coefficient Gaussian Process ROM
# Project-2 thermomechanical ROM
# -----------------------------------------------------------------------------
# Drop-in replacement for the current POD-GPR block.
#
# Main strategy:
#   1) Compact Project-2 workflow preserved.
#   2) One fixed train/validation/test split, no scaler leakage.
#   3) Train one independent ARD Gaussian Process per POD coefficient.
#   4) Robust kernel, optimizer, convergence handling, and fallback model.
#   5) Full diagnostics on train/validation/test:
#        - coefficient error,
#        - POD-subspace field error,
#        - total field error against snapshots,
#        - POD projection floor,
#        - surrogate gap.
#   6) Worst-sample inspection with corresponding parameter values.
#   7) Prediction uncertainty for reduced coefficients.
#   8) Save/load complete model, scalers, basis, split indices, diagnostics, plots.
#   9) POD-GPR output folder is defined before the class, exactly like POD-NN,
#      so class method defaults never depend on a later variable definition.
# =============================================================================

# Keep the previous folder convention: all POD-GPR models live directly in podgpr_dir.
def _method_root_gpr(base_dir, method):
    return base_dir


# --- Settings / folder convention, defined before the class like POD-NN ---
choice_podgpr_run = "live"   # "live" or "load"
podgpr_dir = os.path.join(solver.output_dir, "ROM_Non_Intrusive", "PODGPR")
os.makedirs(podgpr_dir, exist_ok=True)


# =============================================================================
# Configuration
# =============================================================================
PODGPR_CONFIG = dict(
    # Single deterministic split.
    seed=100,
    split_seed=100,

    # Recommended for 150 snapshots: more training support while preserving a real test set.
    # For 150 snapshots this gives about 120 / 15 / 15.
    train_fraction=0.80,
    val_fraction=0.10,
    test_fraction=0.10,

    # Kernel choice:
    #   "rbf"    : smooth deterministic response, usually best first choice.
    #   "matern" : slightly less smooth and more robust if response has localized changes.
    kernel_type="rbf",
    use_white_kernel=True,

    # GPR numerical and optimizer settings.
    alpha=1e-9,                    # small diagonal jitter for deterministic data
    n_restarts_optimizer=10,        # strong but still fine for small/medium POD coefficient dimension
    random_state=100,
    normalize_y=False,              # coefficients are already standardized
    copy_X_train=True,

    # Kernel bounds in normalized parameter space.
    const_value=1.0,
    const_bounds=(1e-3, 1e3),
    length_scale_bounds=(1e-2, 1e2),
    white_noise_level=1e-7,
    white_noise_bounds=(1e-10, 1e-3),

    # Fallback if a coefficient GP fails.
    fallback_alpha=1e-7,
    fallback_n_restarts_optimizer=5,

    # Diagnostics / outputs.
    save_plots=True,
    save_csv=True,
    print_worst_samples=True,
    n_worst_samples=10,
    verbose=True,
)


# =============================================================================
# Utilities
# =============================================================================
def _podgpr_seed(seed=100):
    random.seed(int(seed))
    np.random.seed(int(seed))


def _podgpr_check_fractions(train_fraction, val_fraction, test_fraction):
    vals = np.array([train_fraction, val_fraction, test_fraction], dtype=float)
    if np.any(vals <= 0.0):
        raise ValueError("train_fraction, val_fraction and test_fraction must all be positive.")
    if not np.isclose(vals.sum(), 1.0):
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1.0.")


def _podgpr_split_indices(n_samples, train_fraction, val_fraction, test_fraction, seed):
    _podgpr_check_fractions(train_fraction, val_fraction, test_fraction)
    idx = np.arange(int(n_samples))
    train_idx, tmp_idx = train_test_split(
        idx, train_size=float(train_fraction), random_state=int(seed), shuffle=True
    )
    test_ratio_inside_tmp = float(test_fraction) / float(val_fraction + test_fraction)
    val_idx, test_idx = train_test_split(
        tmp_idx, test_size=test_ratio_inside_tmp, random_state=int(seed) + 1, shuffle=True
    )
    return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int), np.asarray(test_idx, dtype=int)


def _podgpr_basis_file_key(fname):
    # Sort basis_0.npy, basis_1.npy, ..., basis_10.npy correctly.
    try:
        return int(os.path.splitext(fname)[0].split("_")[-1])
    except Exception:
        return fname


def _podgpr_make_kernel(input_dim, cfg):
    d = int(input_dim)
    const = GP_Const(
        float(cfg.get("const_value", 1.0)),
        tuple(cfg.get("const_bounds", (1e-3, 1e3))),
    )

    kernel_type = str(cfg.get("kernel_type", "rbf")).lower()
    length_scale_bounds = tuple(cfg.get("length_scale_bounds", (1e-2, 1e2)))

    if kernel_type == "rbf":
        base = GP_RBF(length_scale=np.ones(d), length_scale_bounds=length_scale_bounds)
    elif kernel_type == "matern":
        base = GP_Matern(length_scale=np.ones(d), length_scale_bounds=length_scale_bounds, nu=2.5)
    else:
        raise ValueError("PODGPR_CONFIG['kernel_type'] must be 'rbf' or 'matern'.")

    kernel = const * base

    if bool(cfg.get("use_white_kernel", True)):
        kernel += GP_WhiteKernel(
            noise_level=float(cfg.get("white_noise_level", 1e-7)),
            noise_level_bounds=tuple(cfg.get("white_noise_bounds", (1e-10, 1e-3))),
        )
    return kernel


def _podgpr_length_scale_from_kernel(kernel):
    """Extract ARD length scales from common Constant*RBF(+White) or Constant*Matern(+White) kernels."""
    try:
        # Case: (Const * RBF/Matern) + White
        k = kernel.k1 if hasattr(kernel, "k1") and hasattr(kernel, "k2") else kernel
        if hasattr(k, "k2") and hasattr(k.k2, "length_scale"):
            return np.asarray(k.k2.length_scale, dtype=float)
        if hasattr(k, "length_scale"):
            return np.asarray(k.length_scale, dtype=float)
    except Exception:
        pass
    return None


def _podgpr_kernel_boundary_flags(gp, cfg, tol=1e-3):
    """Warn if optimized length scales are close to bounds."""
    ls = _podgpr_length_scale_from_kernel(gp.kernel_)
    if ls is None:
        return ""
    lo, hi = cfg.get("length_scale_bounds", (1e-2, 1e2))
    flags = []
    if np.any(ls <= float(lo) * (1.0 + tol)):
        flags.append("ℓ near lower bound")
    if np.any(ls >= float(hi) * (1.0 - tol)):
        flags.append("ℓ near upper bound")
    return "; ".join(flags)


# =============================================================================
# POD-GPR ROM class
# =============================================================================
class PODGPRReducedOrderModel:
    """Non-intrusive POD + Gaussian Process Regression ROM with train/val/test diagnostics."""

    def __init__(self, solver, reduced_basis, n_basis, inner_product, config=None):
        self.solver = solver
        self.reduced_basis = reduced_basis
        self.n_basis = int(n_basis)
        self.inner_product = inner_product
        self.config = dict(PODGPR_CONFIG if config is None else config)

        self.param_scaler = None
        self.coeff_scaler = None
        self.gp_models = []
        self.method = None
        self.gpr_kwargs = {}

        self.split_indices = {}
        self.training_summary = {}
        self.metrics = {}
        self.error_df = None
        self.last_prediction_std = None
        self._basis_matrix_cache = None

    # ---------------------------------------------------------------------
    # POD projection and reconstruction
    # ---------------------------------------------------------------------
    def _basis_matrix(self):
        """Rows are POD basis vectors, shape = (n_basis, n_dofs)."""
        if self._basis_matrix_cache is None or self._basis_matrix_cache.shape[0] != self.n_basis:
            self._basis_matrix_cache = np.vstack([
                self.reduced_basis[i].vector().get_local()
                for i in range(self.n_basis)
            ]).astype(np.float64)
        return self._basis_matrix_cache

    def project_snapshots(self, snapshots):
        print(" Projecting snapshots onto reduced basis...")
        Z = self.reduced_basis[:self.n_basis]
        MZ = [self.inner_product * z.vector() for z in Z]
        A_N = np.array([[zi.vector().inner(mzj) for mzj in MZ] for zi in Z], dtype=np.float64)

        C = np.array([
            np.linalg.solve(A_N, np.array([s.vector().inner(mzj) for mzj in MZ], dtype=np.float64))
            for s in snapshots
        ], dtype=np.float64)

        print(f" Full-order space dim: {self.solver.V_CG.dim()}")
        print(f" Reduced basis vector dim: {Z[0].vector().size()}")
        print(f" Inner product shape: {self.inner_product.size(0)} × {self.inner_product.size(1)}")
        print(f" Z_N shape: {np.column_stack([z.vector().get_local() for z in Z]).shape}")
        print(f" Solution vector shape: {snapshots[0].vector().get_local().shape}")
        return C

    # ---------------------------------------------------------------------
    # Main GPR fit
    # ---------------------------------------------------------------------
    def fit_gpr(self, mus, coeff_matrix, *, method="gpr", gpr_kwargs=None,
                snapshot_targets=None, split_indices=None):
        cfg = self.config
        self.method = method.lower()
        self.gpr_kwargs = dict(gpr_kwargs or {})

        _podgpr_seed(cfg.get("seed", 100))
        split_seed = int(cfg.get("split_seed", cfg.get("seed", 100)))

        X_raw = np.asarray(mus, dtype=np.float64)
        C_raw = np.asarray(coeff_matrix, dtype=np.float64)
        if C_raw.ndim == 1:
            C_raw = C_raw.reshape(-1, 1)

        if X_raw.shape[0] != C_raw.shape[0]:
            raise ValueError(f"mus has {X_raw.shape[0]} rows, but coeff_matrix has {C_raw.shape[0]} rows.")
        if C_raw.shape[1] != self.n_basis:
            raise ValueError(f"coeff_matrix has {C_raw.shape[1]} columns, but this POD-GPR uses n_basis={self.n_basis}.")

        S = None
        if snapshot_targets is not None:
            S = np.asarray(snapshot_targets, dtype=np.float64)
            if S.ndim == 1:
                S = S.reshape(1, -1)
            if S.shape[0] != X_raw.shape[0]:
                raise ValueError(f"snapshot_targets has {S.shape[0]} rows, expected {X_raw.shape[0]}.")
            if S.shape[1] != self.solver.V_CG.dim():
                raise ValueError(f"snapshot_targets has {S.shape[1]} DOFs, expected solver.V_CG.dim()={self.solver.V_CG.dim()}.")

        if split_indices is None:
            train_idx, val_idx, test_idx = _podgpr_split_indices(
                X_raw.shape[0],
                cfg.get("train_fraction", 0.80),
                cfg.get("val_fraction", 0.10),
                cfg.get("test_fraction", 0.10),
                split_seed,
            )
        else:
            train_idx = np.asarray(split_indices["train"], dtype=int)
            val_idx = np.asarray(split_indices["val"], dtype=int)
            test_idx = np.asarray(split_indices["test"], dtype=int)

        self.split_indices = dict(train=train_idx.tolist(), val=val_idx.tolist(), test=test_idx.tolist())

        print(" Fitting Gaussian Process Regression models...")
        print(f" Method: {self.method.upper()} | seed={cfg.get('seed', 100)} | split_seed={split_seed}")
        print(f" Splitting data into train/validation/test sets ({len(train_idx)} / {len(val_idx)} / {len(test_idx)})...")

        # No scaler leakage: fit scalers only on training data.
        self.param_scaler = StandardScaler().fit(X_raw[train_idx])
        self.coeff_scaler = StandardScaler().fit(C_raw[train_idx])
        self.coeff_scaler.scale_[self.coeff_scaler.scale_ < 1e-12] = 1.0

        X = self.param_scaler.transform(X_raw)
        Y = self.coeff_scaler.transform(C_raw)

        # Build robust default kernel unless user provided one.
        if "kernel" not in self.gpr_kwargs:
            self.gpr_kwargs["kernel"] = _podgpr_make_kernel(X.shape[1], cfg)

        defaults = dict(
            alpha=float(cfg.get("alpha", 1e-9)),
            random_state=int(cfg.get("random_state", cfg.get("seed", 100))),
            n_restarts_optimizer=int(cfg.get("n_restarts_optimizer", 10)),
            normalize_y=bool(cfg.get("normalize_y", False)),
            copy_X_train=bool(cfg.get("copy_X_train", True)),
        )
        for key, val in defaults.items():
            self.gpr_kwargs.setdefault(key, val)

        print(f" Kernel template: {self.gpr_kwargs['kernel']}")
        print(f" Training {Y.shape[1]} GP models on {len(train_idx)} samples (one per reduced coefficient)...")

        self.gp_models = []
        rows = []
        t0 = time.perf_counter()

        for j in range(Y.shape[1]):
            print(f"   GP {j+1:02d}/{Y.shape[1]:02d}...", end=" ", flush=True)
            try:
                gp = GaussianProcessRegressor(**self.gpr_kwargs)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.filterwarnings("always", category=ConvergenceWarning)
                    gp.fit(X[train_idx], Y[train_idx, j])

                ll = float(gp.log_marginal_likelihood(gp.kernel_.theta))
                boundary_msg = _podgpr_kernel_boundary_flags(gp, cfg)
                n_warn = sum(issubclass(w.category, ConvergenceWarning) for w in caught)

                self.gp_models.append(gp)

                ls = _podgpr_length_scale_from_kernel(gp.kernel_)
                ls_str = "n/a" if ls is None else np.array2string(ls, precision=3, separator=", ")
                status = "ok"
                note = boundary_msg

                print(f"✓ LL={ll:.3e}, ℓ={ls_str}" + (f" [{boundary_msg}]" if boundary_msg else ""))

            except Exception as e:
                print(f"✗ failed ({str(e)[:70]}). Using fallback...", end=" ", flush=True)
                fallback_kernel = GP_Const(1.0, (1e-3, 1e3)) * GP_RBF(
                    np.ones(X.shape[1]), tuple(cfg.get("length_scale_bounds", (1e-2, 1e2)))
                ) + GP_WhiteKernel(
                    noise_level=float(cfg.get("white_noise_level", 1e-7)),
                    noise_level_bounds=tuple(cfg.get("white_noise_bounds", (1e-10, 1e-3))),
                )
                gp = GaussianProcessRegressor(
                    kernel=fallback_kernel,
                    alpha=float(cfg.get("fallback_alpha", 1e-7)),
                    random_state=int(cfg.get("random_state", cfg.get("seed", 100))),
                    n_restarts_optimizer=int(cfg.get("fallback_n_restarts_optimizer", 5)),
                    normalize_y=False,
                )
                gp.fit(X[train_idx], Y[train_idx, j])
                self.gp_models.append(gp)
                ll = float(gp.log_marginal_likelihood(gp.kernel_.theta))
                ls = _podgpr_length_scale_from_kernel(gp.kernel_)
                ls_str = "n/a" if ls is None else np.array2string(ls, precision=3, separator=", ")
                status = "fallback"
                note = str(e)[:120]
                n_warn = np.nan
                print(f"✓ fallback LL={ll:.3e}, ℓ={ls_str}")

            rows.append(dict(
                coeff_index=int(j),
                status=status,
                log_marginal_likelihood=float(ll),
                kernel=str(gp.kernel_),
                length_scale=ls_str,
                n_convergence_warnings=n_warn,
                note=note,
            ))

        elapsed = time.perf_counter() - t0
        self.training_summary = dict(
            elapsed_sec=float(elapsed),
            n_models=int(Y.shape[1]),
            n_train=int(len(train_idx)),
            n_val=int(len(val_idx)),
            n_test=int(len(test_idx)),
            gp_rows=rows,
            config=cfg,
            gpr_kwargs={k: v for k, v in self.gpr_kwargs.items() if k != "kernel"},
            kernel_template=str(self.gpr_kwargs["kernel"]),
            split_indices=self.split_indices,
        )

        df_summary = pd.DataFrame(rows)
        n_ok = int(np.sum(df_summary["status"].to_numpy() == "ok"))
        print(f" GP training complete in {elapsed:.2f} s.")
        print(f" ✓ Success rate: {n_ok}/{Y.shape[1]} direct models")
        print(f" ✓ Average log-marginal likelihood: {df_summary['log_marginal_likelihood'].mean():.3e}")

        # Final diagnostics.
        metrics, df_errors = self.evaluate_splits(
            X_raw, C_raw, snapshot_targets=S,
            split_indices=self.split_indices,
            save_csv=os.path.join(podgpr_dir, "podgpr_train_val_test_errors.csv")
            if bool(cfg.get("save_csv", True)) else None,
        )
        self.metrics = metrics
        self.error_df = df_errors

        if bool(cfg.get("save_csv", True)):
            df_summary.to_csv(os.path.join(podgpr_dir, "podgpr_model_training_summary.csv"), index=False)
            print(f"[PODGPR] Saved GP training summary CSV → {os.path.join(podgpr_dir, 'podgpr_model_training_summary.csv')}")

        if bool(cfg.get("print_worst_samples", True)):
            self.print_worst_samples(df_errors, mus=X_raw, n=int(cfg.get("n_worst_samples", 10)))

        return self, metrics, df_errors

    # ---------------------------------------------------------------------
    # Prediction and reconstruction
    # ---------------------------------------------------------------------
    def predict_coeffs_batch(self, mus, return_std=False):
        if self.param_scaler is None or self.coeff_scaler is None or not self.gp_models:
            raise RuntimeError("POD-GPR model is not fitted/loaded.")

        X = np.asarray(mus, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        Xi = self.param_scaler.transform(X)
        means, stds = [], []
        for gp in self.gp_models:
            m, s = gp.predict(Xi, return_std=True)
            means.append(m)
            stds.append(s)

        Y_mean = np.vstack(means).T
        Y_std = np.vstack(stds).T

        C_mean = self.coeff_scaler.inverse_transform(Y_mean)
        C_std = Y_std * self.coeff_scaler.scale_.reshape(1, -1)

        if return_std:
            return C_mean, C_std
        return C_mean

    def predict_coeffs(self, mu):
        C_mean, C_std = self.predict_coeffs_batch(np.asarray(mu, dtype=np.float64).reshape(1, -1), return_std=True)
        self.last_prediction_std = C_std[0]
        return C_mean[0]

    def predict_with_uncertainty(self, mu):
        C_mean, C_std = self.predict_coeffs_batch(np.asarray(mu, dtype=np.float64).reshape(1, -1), return_std=True)
        self.last_prediction_std = C_std[0]
        return C_mean[0], C_std[0]

    def reconstruct_vector(self, coeffs):
        coeffs = np.asarray(coeffs, dtype=np.float64).ravel()
        return coeffs @ self._basis_matrix()[:len(coeffs), :]

    def reconstruct_solution(self, coeffs):
        u = Function(self.solver.V_CG)
        u.vector().set_local(self.reconstruct_vector(coeffs))
        u.vector().apply("insert")
        return u

    def predict_vector(self, mu):
        return self.reconstruct_vector(self.predict_coeffs(mu))

    def predict_solution(self, mu):
        return self.reconstruct_solution(self.predict_coeffs(mu))

    # ---------------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------------
    def _evaluate_indices(self, mus, true_coeffs, snapshot_targets, ids, split_name):
        ids = np.asarray(ids, dtype=int)
        M = np.asarray(mus, dtype=np.float64)
        C_true = np.asarray(true_coeffs, dtype=np.float64)
        C_pred, C_std = self.predict_coeffs_batch(M[ids], return_std=True)

        Ct = C_true[ids]
        diff_C = C_pred - Ct

        coeff_mse = float(np.mean(diff_C ** 2))
        coeff_rmse = float(np.sqrt(coeff_mse))
        coeff_rel_global = float(np.linalg.norm(diff_C) / (np.linalg.norm(Ct) + 1e-30))
        coeff_rel_each = np.linalg.norm(diff_C, axis=1) / (np.linalg.norm(Ct, axis=1) + 1e-30)

        B = self._basis_matrix()
        U_pred = C_pred @ B
        U_proj = Ct @ B

        subspace_rel_each = np.linalg.norm(U_pred - U_proj, axis=1) / (np.linalg.norm(U_proj, axis=1) + 1e-30)

        # Independent-GP approximate 1-sigma field uncertainty norm.
        # This ignores inter-coefficient covariance, but is still useful as a relative diagnostic.
        U_std_proxy = np.sqrt((C_std ** 2) @ (B ** 2))
        field_std_proxy_rel = np.linalg.norm(U_std_proxy, axis=1) / (np.linalg.norm(U_pred, axis=1) + 1e-30)

        out = dict(
            split=split_name,
            n_samples=int(len(ids)),
            coeff_mse=coeff_mse,
            coeff_rmse=coeff_rmse,
            coeff_rel_global=coeff_rel_global,
            coeff_std_mean=float(np.mean(C_std)),
            coeff_std_max=float(np.max(C_std)),
            subspace_field_rel_mean=float(np.mean(subspace_rel_each)),
            subspace_field_rel_median=float(np.median(subspace_rel_each)),
            subspace_field_rel_max=float(np.max(subspace_rel_each)),
            field_std_proxy_rel_mean=float(np.mean(field_std_proxy_rel)),
            field_std_proxy_rel_max=float(np.max(field_std_proxy_rel)),
        )

        rows = []
        if snapshot_targets is not None:
            S = np.asarray(snapshot_targets, dtype=np.float64)[ids]
            total_rel_each = np.linalg.norm(U_pred - S, axis=1) / (np.linalg.norm(S, axis=1) + 1e-30)
            proj_floor_each = np.linalg.norm(U_proj - S, axis=1) / (np.linalg.norm(S, axis=1) + 1e-30)

            out.update(
                total_field_rel_mean=float(np.mean(total_rel_each)),
                total_field_rel_median=float(np.median(total_rel_each)),
                total_field_rel_max=float(np.max(total_rel_each)),
                projection_floor_mean=float(np.mean(proj_floor_each)),
                projection_floor_median=float(np.median(proj_floor_each)),
                projection_floor_max=float(np.max(proj_floor_each)),
                surrogate_gap_mean=float(np.mean(total_rel_each - proj_floor_each)),
                surrogate_gap_max=float(np.max(total_rel_each - proj_floor_each)),
            )
        else:
            total_rel_each = np.full(len(ids), np.nan)
            proj_floor_each = np.full(len(ids), np.nan)

        for local_k, global_i in enumerate(ids):
            rows.append(dict(
                split=split_name,
                sample_index=int(global_i),
                coeff_rel=float(coeff_rel_each[local_k]),
                coeff_std_mean=float(np.mean(C_std[local_k])),
                subspace_field_rel=float(subspace_rel_each[local_k]),
                field_std_proxy_rel=float(field_std_proxy_rel[local_k]),
                total_field_rel=float(total_rel_each[local_k]),
                projection_floor=float(proj_floor_each[local_k]),
                surrogate_gap=float(total_rel_each[local_k] - proj_floor_each[local_k]),
            ))

        return out, rows

    def evaluate_splits(self, mus, true_coeffs, snapshot_targets=None, split_indices=None, save_csv=None):
        split_indices = split_indices or self.split_indices
        all_metrics, all_rows = {}, []

        print("\n" + "=" * 94)
        print(f"{'POD-GPR train/validation/test diagnostics':^94}")
        print("=" * 94)

        for split in ["train", "val", "test"]:
            if split not in split_indices:
                continue
            metrics, rows = self._evaluate_indices(
                mus, true_coeffs, snapshot_targets,
                split_indices[split],
                split,
            )
            all_metrics[split] = metrics
            all_rows.extend(rows)

            print(f"\n[{split.upper()}] n={metrics['n_samples']}")
            print(f"  coeff RMSE/global rel         : {metrics['coeff_rmse']:.4e} / {metrics['coeff_rel_global']:.4e}")
            print(f"  coeff std mean/max            : {metrics['coeff_std_mean']:.4e} / {metrics['coeff_std_max']:.4e}")
            print(f"  subspace field rel mean/max   : {metrics['subspace_field_rel_mean']:.4e} / {metrics['subspace_field_rel_max']:.4e}")
            print(f"  field std proxy rel mean/max  : {metrics['field_std_proxy_rel_mean']:.4e} / {metrics['field_std_proxy_rel_max']:.4e}")
            if snapshot_targets is not None:
                print(f"  total field rel mean/median   : {metrics['total_field_rel_mean']:.4e} / {metrics['total_field_rel_median']:.4e}")
                print(f"  total field rel max           : {metrics['total_field_rel_max']:.4e}")
                print(f"  POD projection floor mean/max : {metrics['projection_floor_mean']:.4e} / {metrics['projection_floor_max']:.4e}")
                print(f"  surrogate gap mean/max        : {metrics['surrogate_gap_mean']:.4e} / {metrics['surrogate_gap_max']:.4e}")

        print("=" * 94 + "\n")

        df = pd.DataFrame(all_rows)

        if save_csv is not None:
            df.to_csv(save_csv, index=False)
            print(f"[PODGPR] Saved split diagnostics CSV → {save_csv}")

        if bool(self.config.get("save_plots", True)) and len(df) > 0:
            self.plot_split_errors(df, save_dir=podgpr_dir)

        self.metrics = all_metrics
        self.error_df = df
        return all_metrics, df

    def print_worst_samples(self, df_errors, mus=None, n=10):
        if df_errors is None or len(df_errors) == 0:
            return

        sort_col = "total_field_rel" if "total_field_rel" in df_errors.columns else "subspace_field_rel"

        print("\n" + "=" * 94)
        print(f"{'Worst POD-GPR samples by relative field error':^94}")
        print("=" * 94)

        for split in ["train", "val", "test"]:
            d = df_errors[df_errors["split"] == split].copy()
            if len(d) == 0:
                continue
            d = d.sort_values(sort_col, ascending=False).head(int(n))
            print(f"\n[{split.upper()}] worst {min(int(n), len(d))} samples")
            cols = [
                "split", "sample_index", "total_field_rel", "projection_floor",
                "surrogate_gap", "subspace_field_rel", "field_std_proxy_rel",
                "coeff_rel", "coeff_std_mean",
            ]
            cols = [c for c in cols if c in d.columns]
            print(d[cols].to_string(index=False))

            if mus is not None:
                M = np.asarray(mus, dtype=np.float64)
                print("  Parameter values:")
                for sample_idx in d["sample_index"].to_numpy(dtype=int):
                    print(f"    sample {sample_idx:4d}: mu = {np.array2string(M[sample_idx], precision=6, separator=', ')}")

        print("=" * 94 + "\n")

    def plot_split_errors(self, df_errors, save_dir=None):
        if save_dir is None:
            save_dir = podgpr_dir
        os.makedirs(save_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=140)

        for split in ["train", "val", "test"]:
            d = df_errors[df_errors["split"] == split]
            if len(d):
                y = d["total_field_rel"].to_numpy() if np.isfinite(d["total_field_rel"].to_numpy()).any() else d["subspace_field_rel"].to_numpy()
                ax.plot(np.arange(len(d)), y, marker="o", linewidth=1.3, label=split)

        ax.set_yscale("log")
        ax.set_xlabel("Sample index within split")
        ax.set_ylabel("Relative field error")
        ax.set_title("POD-GPR split-wise field errors")
        ax.grid(True, which="both", linestyle=":")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "podgpr_split_errors.png"), dpi=300, bbox_inches="tight")
        plt.show()

    # ---------------------------------------------------------------------
    # Save/load
    # ---------------------------------------------------------------------
    def save(self, base_dir):
        root = _method_root_gpr(base_dir, self.method)
        os.makedirs(root, exist_ok=True)

        idx = max([int(d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d.isdigit()] + [0]) + 1
        folder = os.path.join(root, str(idx))
        bdir = os.path.join(folder, "basis")
        os.makedirs(folder, exist_ok=True)
        os.makedirs(bdir, exist_ok=True)

        for i, phi in enumerate(self.reduced_basis[:self.n_basis]):
            np.save(os.path.join(bdir, f"basis_{i}.npy"), phi.vector().get_local())

        with open(os.path.join(folder, "model.pkl"), "wb") as f:
            pickle.dump(dict(
                method=self.method,
                config=self.config,
                gpr_kwargs=self.gpr_kwargs,
                param_scaler=self.param_scaler,
                coeff_scaler=self.coeff_scaler,
                gp_models=self.gp_models,
                n_basis=self.n_basis,
                split_indices=self.split_indices,
                training_summary=self.training_summary,
                metrics=self.metrics,
            ), f)

        if self.metrics:
            rows = []
            for split, m in self.metrics.items():
                row = dict(m)
                row["split"] = split
                rows.append(row)
            pd.DataFrame(rows).to_csv(os.path.join(folder, "podgpr_metrics_summary.csv"), index=False)

        if self.error_df is not None:
            self.error_df.to_csv(os.path.join(folder, "podgpr_train_val_test_errors.csv"), index=False)

        if self.training_summary and self.training_summary.get("gp_rows"):
            pd.DataFrame(self.training_summary["gp_rows"]).to_csv(
                os.path.join(folder, "podgpr_model_training_summary.csv"), index=False
            )

        print(f"[PODGPR-{self.method.upper()}] model saved → {folder}")
        return idx

    @classmethod
    def load(cls, solver, pod_index, method, base_dir):
        root = _method_root_gpr(base_dir, method)
        folder = os.path.join(root, str(pod_index))
        bdir = os.path.join(folder, "basis")

        basis = []
        for file in sorted(os.listdir(bdir), key=_podgpr_basis_file_key):
            vec = np.load(os.path.join(bdir, file))
            f = Function(solver.V_CG)
            f.vector().set_local(vec)
            f.vector().apply("insert")
            basis.append(f)

        with open(os.path.join(folder, "model.pkl"), "rb") as f:
            data = pickle.load(f)

        obj = cls(
            solver,
            reduced_basis=basis,
            n_basis=data["n_basis"],
            inner_product=solver.inner_product_CG,
            config=data.get("config", dict(PODGPR_CONFIG)),
        )
        obj.method = data.get("method", method)
        obj.gpr_kwargs = data.get("gpr_kwargs", {})
        obj.param_scaler = data["param_scaler"]
        obj.coeff_scaler = data["coeff_scaler"]
        obj.gp_models = data["gp_models"]
        obj.split_indices = data.get("split_indices", {})
        obj.training_summary = data.get("training_summary", {})
        obj.metrics = data.get("metrics", {})
        print(f"[PODGPR-{method.upper()}] model loaded ← {folder}")
        return obj


# =============================================================================
# Live/load execution
# =============================================================================
method_list_gpr = ["gpr"]
model_indices_gpr, podgpr_models = {}, {}

# --- Prepare snapshot functions only for live ---
if choice_podgpr_run == "live":
    snapshot_functions_gpr = []
    C_I_gpr = None

    for vec in snaps_all_I:
        f = Function(solver.V_CG)
        f.vector().set_local(np.asarray(vec, dtype=np.float64).ravel())
        f.vector().apply("insert")
        snapshot_functions_gpr.append(f)

# --- Train or load all methods ---
for method in method_list_gpr:
    # You can override defaults here. Leaving kernel absent lets PODGPR_CONFIG build the robust default kernel.
    gpr_kwargs = dict(
        alpha=PODGPR_CONFIG["alpha"],
        random_state=PODGPR_CONFIG["random_state"],
        n_restarts_optimizer=PODGPR_CONFIG["n_restarts_optimizer"],
        normalize_y=PODGPR_CONFIG["normalize_y"],
    )

    if choice_podgpr_run == "live":
        print(f"\n[PODGPR-{method.upper()}] Training...")
        podgpr_rom = PODGPRReducedOrderModel(solver, Z_podI, N_podI, IP_I, config=PODGPR_CONFIG)

        if C_I_gpr is None:
            C_I_gpr = podgpr_rom.project_snapshots(snapshot_functions_gpr)
            print(f" Reduced coefficients computed. Shape: {C_I_gpr.shape}")

        podgpr_rom, podgpr_split_metrics, podgpr_error_df = podgpr_rom.fit_gpr(
            mu_all_I,
            C_I_gpr,
            method=method,
            gpr_kwargs=gpr_kwargs,
            snapshot_targets=snaps_all_I,
        )

        idx = podgpr_rom.save(podgpr_dir)
        print(f"Saved PODGPR-{method.upper()} model at index {idx}.")
        podgpr_models[method], model_indices_gpr[method] = podgpr_rom, idx

        globals().update(
            podgpr_rom=podgpr_rom,
            podgpr_split_metrics=podgpr_split_metrics,
            podgpr_error_df=podgpr_error_df,
            C_I_gpr=C_I_gpr,
        )

    elif choice_podgpr_run == "load":
        idx = 1  # <-- set this to the correct index for each method if needed
        print(f"\n[PODGPR-{method.upper()}] Loading model index {idx}...")
        podgpr_rom = PODGPRReducedOrderModel.load(solver, pod_index=idx, method=method, base_dir=podgpr_dir)
        podgpr_models[method], model_indices_gpr[method] = podgpr_rom, idx
    else:
        raise ValueError(f"Unknown choice: {choice_podgpr_run}")

# %% [markdown] Cell 85 | id: e97d4e59
# ### POD-enhanced Autoencoder ROM (POD-AE-ROM)

# %% Cell 86 | id: 3fc94683
# =============================================================================
# POD-AE-ROM : Robust train/validation/test POD-enhanced Autoencoder ROM
# Project-2 thermomechanical ROM
# -----------------------------------------------------------------------------
# Drop-in replacement for the current POD-AE block.
#
# Main strategy:
#   1) Compact Project-2 workflow preserved.
#   2) CPU/thread-safe PyTorch inside FEniCS/Jupyter.
#   3) One fixed train/validation/test split, no scaler leakage.
#   4) Uses active projected POD basis whenever available: Z_podnn/IP_N first,
#      then Z_podI/IP_I, otherwise vector-SVD fallback.
#   5) Stage 1: coefficient autoencoder C -> latent -> C.
#   6) Stage 2: parameter-to-latent map mu -> latent.
#   7) Stage 3: metric-aware online-path fine-tuning using field-relative loss.
#   8) Full diagnostics on train/validation/test:
#        - coefficient error,
#        - POD-subspace field error,
#        - total field error against snapshots,
#        - POD projection floor,
#        - surrogate gap,
#        - optional autoencoder reconstruction diagnostic.
#   9) Worst-sample inspection with corresponding parameter values.
#  10) Save/load complete model, scalers, basis, split indices, diagnostics, plots.
#  11) Keeps the compact comparison/plotting helpers used by the integrated ROM cells.
# =============================================================================


# =============================================================================
# Folder convention, defined before classes/functions that may use it
# =============================================================================
if "solver" in globals() and hasattr(solver, "output_dir"):
    Non_Intrusive_dir = os.path.join(solver.output_dir, "ROM_Non_Intrusive")
else:
    Non_Intrusive_dir = os.path.join(os.getcwd(), "ROM_Non_Intrusive")

podae_dir = os.path.join(Non_Intrusive_dir, "POD_AE")
for _d in [Non_Intrusive_dir, podae_dir]:
    os.makedirs(_d, exist_ok=True)

choice_podae_run = globals().get("choice_podae_run", "live")  # "live" or "load"
podae_run_folder = globals().get("podae_run_folder", None)    # None -> latest for load


# =============================================================================
# Configuration
# =============================================================================
POD_AE_CONFIG = dict(
    # Snapshot source. If None, the loader checks active in-memory snapshots first
    # and then known archive globals such as snapshot_file_proj.
    snapshot_archive=snapshot_file_proj if "snapshot_file_proj" in globals() else None,
    prefer_projected=True,
    snapshot_key=None,

    # Basis size. If None, uses active N_podnn, then N_podI, then SVD fallback.
    n_basis=None,

    # "basis" means latent_dim = active n_basis. This is the cleanest setting for
    # a POD-enhanced AE when the POD space is already very low-dimensional.
    # For stronger compression, use "compressed" or an integer < n_basis.
    latent_dim="basis",

    # Fallback vector POD options only used if FEniCS/RBniCS POD basis is unavailable.
    pod_energy_tol=0.999999,
    n_basis_max=40,

    # Single deterministic run.
    seed=100,
    split_seed=100,

    # Same robust split used in final POD-NN/POD-GPR blocks.
    train_fraction=0.80,
    val_fraction=0.10,
    test_fraction=0.10,

    # Device. CPU is recommended for FEniCS/Jupyter consistency.
    device_preference="cpu",  # "cpu", "cuda", "mps", or "auto"

    # Network sizes. Compact but stronger than the original minimal version.
    ae_hidden=(128, 96, 64),
    latent_hidden=(128, 96, 64),
    activation="elu",
    dropout=0.00,
    layer_norm=False,

    # Stage 1: coefficient autoencoder.
    ae_epochs=2500,
    lr_ae=5e-4,
    ae_batch_size=256,
    ae_weight_decay=2e-6,
    ae_scheduler_patience=120,
    ae_early_stopping_patience=450,
    ae_min_delta=5e-9,
    ae_grad_clip=1.0,

    # Stage 2: parameter-to-latent map.
    latent_epochs=2500,
    lr_latent=5e-4,
    latent_batch_size=256,
    latent_weight_decay=2e-6,
    latent_scheduler_patience=120,
    latent_early_stopping_patience=450,
    latent_min_delta=5e-9,
    latent_grad_clip=1.0,

    # Stage 3: metric-aware online-path fine-tuning.
    end_to_end_finetune=True,
    e2e_epochs=500,
    e2e_lr=3e-5,
    e2e_batch_size=256,
    e2e_train_decoder=True,
    e2e_train_encoder=False,        # normally keep encoder frozen
    e2e_weight_decay=5e-7,
    e2e_patience=120,
    e2e_scheduler_patience=50,
    e2e_min_delta=5e-10,
    e2e_loss="field_relative_plus_coeff",
    e2e_use_full_snapshot_target=True,
    e2e_field_weight=1.0,
    e2e_coeff_weight=0.01,
    e2e_grad_clip=1.0,

    # Runtime / outputs.
    save_model=True,
    save_plots=True,
    save_csv=True,
    model_name="pod_ae_rom_robust",
    log_every=50,

    # Diagnostics.
    print_worst_samples=True,
    n_worst_samples=10,
)


# =============================================================================
# Utilities
# =============================================================================
def _podae_seed(seed=100):
    if "set_seed" in globals():
        set_seed(int(seed))
    else:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))


def _podae_device(preference="cpu"):
    pref = str(preference).lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "mps":
        return torch.device("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError("device_preference must be one of: 'cpu', 'cuda', 'mps', 'auto'.")


def _podae_act(name_or_cls):
    if isinstance(name_or_cls, str):
        acts = dict(elu=nn.ELU, tanh=nn.Tanh, relu=nn.ReLU, gelu=nn.GELU, silu=nn.SiLU)
        key = name_or_cls.lower()
        if key not in acts:
            raise ValueError(f"Unsupported activation '{name_or_cls}'. Use elu/tanh/relu/gelu/silu.")
        return acts[key]
    return name_or_cls


def _podae_mlp(dims, activation="elu", dropout=0.0, layer_norm=False, final_activation=False):
    act = _podae_act(activation)
    dims = [int(x) for x in dims]
    layers = []
    for k, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = (k == len(dims) - 2)
        layers.append(nn.Linear(a, b))
        if (not is_last) or bool(final_activation):
            if layer_norm and (not is_last):
                layers.append(nn.LayerNorm(b))
            layers.append(act())
            if float(dropout) > 0.0 and (not is_last):
                layers.append(nn.Dropout(float(dropout)))
    return nn.Sequential(*layers)


def _mkdir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _clone_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _rel_rows(A, B, eps=1e-30):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    return np.linalg.norm(A - B, axis=1) / (np.linalg.norm(B, axis=1) + eps)


def _torch_inv_scaler(x_scaled, scaler):
    mean = torch.tensor(scaler.mean_, dtype=x_scaled.dtype, device=x_scaled.device)
    scale = torch.tensor(scaler.scale_, dtype=x_scaled.dtype, device=x_scaled.device)
    return x_scaled * scale + mean


def _check_fractions(train_fraction, val_fraction, test_fraction):
    vals = np.asarray([train_fraction, val_fraction, test_fraction], dtype=float)
    if np.any(vals <= 0.0):
        raise ValueError("train_fraction, val_fraction and test_fraction must all be positive.")
    if not np.isclose(vals.sum(), 1.0):
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1.0.")


def _split_indices(n_samples, train_fraction, val_fraction, test_fraction, seed):
    _check_fractions(train_fraction, val_fraction, test_fraction)
    idx = np.arange(int(n_samples))
    train_idx, tmp_idx = train_test_split(
        idx, train_size=float(train_fraction), random_state=int(seed), shuffle=True
    )
    test_ratio_inside_tmp = float(test_fraction) / float(val_fraction + test_fraction)
    val_idx, test_idx = train_test_split(
        tmp_idx, test_size=test_ratio_inside_tmp, random_state=int(seed) + 1, shuffle=True
    )
    return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int), np.asarray(test_idx, dtype=int)


def _basis_file_key(fname):
    try:
        return int(os.path.splitext(fname)[0].split("_")[-1])
    except Exception:
        return fname


def _resolve_latent_dim(latent_dim, n_basis, n_mu):
    if latent_dim is None:
        return int(n_basis)

    if isinstance(latent_dim, str):
        key = latent_dim.lower().strip()
        if key in ("basis", "same_as_basis", "n_basis", "full"):
            return int(n_basis)
        if key in ("compressed", "mu", "parameter"):
            return max(1, min(int(n_basis), int(n_mu)))
        raise ValueError("Unsupported latent_dim string. Use 'basis', 'compressed', or an integer.")

    latent_dim = int(latent_dim)
    if latent_dim <= 0:
        latent_dim = max(1, min(int(n_basis), int(n_mu)))
    if latent_dim > int(n_basis):
        print(f"[POD-AE] Requested latent_dim={latent_dim} exceeds n_basis={n_basis}; resetting to n_basis.")
        latent_dim = int(n_basis)
    return latent_dim


def _podae_next_run_folder(base_dir=podae_dir):
    _mkdir(base_dir)
    ids = [int(x) for x in os.listdir(base_dir)
           if x.isdigit() and os.path.isdir(os.path.join(base_dir, x))]
    return _mkdir(os.path.join(base_dir, str(max(ids + [0]) + 1)))


def _podae_latest_run(base_dir=podae_dir):
    ids = [int(x) for x in os.listdir(base_dir)
           if x.isdigit() and os.path.isdir(os.path.join(base_dir, x))]
    if not ids:
        raise FileNotFoundError(f"No POD-AE run folders found in: {base_dir}")
    return os.path.join(base_dir, str(max(ids)))


# =============================================================================
# Snapshot loading
# =============================================================================
def resolve_pod_ae_snapshot_archive(
    snapshot_archive=None,
    prefer_projected=True,
):
    if snapshot_archive is not None:
        path = os.path.expanduser(str(snapshot_archive))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Requested snapshot archive does not exist: {path}")
        return path

    g = globals()
    candidates = []

    for key in [
        "snapshot_file_proj",
        "snapshot_file_projected",
        "snapshot_file",
        "snapshot_file_Solution",
        "snapshot_file_solution",
    ]:
        if g.get(key):
            p = str(g[key])
            if prefer_projected:
                candidates.append(p)
            elif "proj" not in key.lower():
                candidates.append(p)

    if (not prefer_projected) and g.get("snapshot_file_proj"):
        candidates.append(str(g["snapshot_file_proj"]))

    if g.get("P2_ARCHIVE"):
        candidates.append(str(g["P2_ARCHIVE"]))

    if g.get("P2_ROM_DATA_DIR") and g.get("P2_RUN_NAME"):
        rom_data_dir = str(g["P2_ROM_DATA_DIR"])
        run_name = str(g["P2_RUN_NAME"])
        sol = os.path.join(rom_data_dir, f"{run_name}__Solution_W.npz")
        proj = os.path.join(rom_data_dir, f"{run_name}__Projected_CG.npz")
        rep = os.path.join(rom_data_dir, f"{run_name}__Reprojected_Solution_W.npz")
        candidates += [proj, sol, rep] if prefer_projected else [sol, proj, rep]

    if g.get("solver") is not None and hasattr(g["solver"], "output_dir"):
        candidates.append(os.path.join(g["solver"].output_dir, "snapshots.npz"))

    seen, clean = set(), []
    for p in candidates:
        p = os.path.expanduser(str(p))
        if p not in seen:
            seen.add(p)
            clean.append(p)

    for p in clean:
        if os.path.exists(p):
            print(f"[POD-AE] Using snapshot archive: {p}")
            return p

    return None


def _load_from_active_memory(prefer_projected=True):
    """Fallback/in-memory loader so POD-AE can use the same active snapshot bank as POD-NN/GPR."""
    g = globals()

    memory_candidates = []
    if prefer_projected:
        memory_candidates += [
            ("snaps_all_N", "mu_all_N"),
            ("snaps_all_I", "mu_all_I"),
            ("snaps_all_G", "mu_all_G"),
            ("snapshots", "mus"),
        ]
    else:
        memory_candidates += [
            ("snaps_all_I", "mu_all_I"),
            ("snaps_all_N", "mu_all_N"),
            ("snaps_all_G", "mu_all_G"),
            ("snapshots", "mus"),
        ]

    for s_key, m_key in memory_candidates:
        if s_key in g and m_key in g and g[s_key] is not None and g[m_key] is not None:
            S = np.asarray(g[s_key], dtype=np.float64)
            M = np.asarray(g[m_key], dtype=np.float64)
            if S.ndim == 1:
                S = S.reshape(1, -1)
            elif S.ndim > 2:
                S = S.reshape(S.shape[0], -1)
            if M.ndim == 1:
                M = M.reshape(-1, 1)
            if S.shape[0] == M.shape[0]:
                print(f"[POD-AE] Using active in-memory snapshots: {s_key}, parameters: {m_key}")
                return dict(
                    path=f"in_memory:{s_key}/{m_key}",
                    keys=[s_key, m_key],
                    snapshot_key=s_key,
                    snapshots=S.astype(np.float64, copy=False),
                    mus=M.astype(np.float64, copy=False),
                )

    return None


def load_pod_ae_snapshot_archive(snapshot_archive=None, prefer_projected=True, snapshot_key=None):
    # Prefer active in-memory data because the current POD-NN/POD-GPR cells already use it.
    mem = _load_from_active_memory(prefer_projected=prefer_projected)
    if mem is not None and snapshot_archive is None:
        S, M = mem["snapshots"], mem["mus"]
        print("\n[POD-AE] Snapshot data loaded")
        print(f"  source       : {mem['path']}")
        print(f"  n_snapshots  : {S.shape[0]}")
        print(f"  n_parameters : {M.shape[1]}")
        print(f"  FOM dimension: {S.shape[1]}")
        return mem

    path = resolve_pod_ae_snapshot_archive(snapshot_archive, prefer_projected)
    if path is None:
        raise FileNotFoundError(
            "Could not find an active snapshot archive or in-memory snapshot arrays. "
            "Run the snapshot/POD-loading cells first, or set POD_AE_CONFIG['snapshot_archive']."
        )

    data = np.load(path, allow_pickle=True)
    keys = list(data.files)

    if snapshot_key is None:
        if "fom_snapshots" in keys:
            snapshot_key = "fom_snapshots"
        elif "snapshots" in keys:
            snapshot_key = "snapshots"
        else:
            raise KeyError(f"No displacement snapshot key in {path}. Keys: {keys}")

    if "mus" not in keys:
        raise KeyError(f"Archive {path} does not contain key 'mus'. Keys: {keys}")

    S = np.asarray(data[snapshot_key])
    M = np.asarray(data["mus"])

    if S.ndim == 1:
        S = S.reshape(1, -1)
    elif S.ndim > 2:
        S = S.reshape(S.shape[0], -1)

    if M.ndim == 1:
        M = M.reshape(-1, 1)

    if S.shape[0] != M.shape[0]:
        raise ValueError(f"Snapshot/parameter mismatch: snapshots={S.shape}, mus={M.shape}")

    ok = np.isfinite(S).all(axis=1) & np.isfinite(M).all(axis=1)
    if (~ok).sum():
        print(f"[POD-AE] Removing {(~ok).sum()} non-finite rows.")
        S, M = S[ok], M[ok]

    S = S.astype(np.float64, copy=False)
    M = M.astype(np.float64, copy=False)

    print("\n[POD-AE] Snapshot archive loaded")
    print(f"  path         : {path}")
    print(f"  keys         : {keys}")
    print(f"  snapshot key : {snapshot_key}")
    print(f"  n_snapshots  : {S.shape[0]}")
    print(f"  n_parameters : {M.shape[1]}")
    print(f"  FOM dimension: {S.shape[1]}")
    print(f"  dtype        : snapshots={S.dtype}, mus={M.dtype}")

    return dict(path=path, keys=keys, snapshot_key=snapshot_key, snapshots=S, mus=M)


# =============================================================================
# POD coefficient preparation
# =============================================================================
def _basis_matrix_from_Z(Z, n_basis):
    return np.vstack([
        np.asarray(Z[i].vector().get_local(), dtype=np.float64).ravel()
        for i in range(int(n_basis))
    ])


def _project_existing_pod(S, solver_obj, Z, n_basis, IP):
    if solver_obj is None or not hasattr(solver_obj, "V_CG"):
        raise RuntimeError("solver.V_CG is required for FEniCS POD projection.")

    n_dofs = int(solver_obj.V_CG.dim())
    if S.shape[1] != n_dofs:
        raise ValueError(
            f"Snapshot dimension {S.shape[1]} does not match solver.V_CG.dim()={n_dofs}. "
            "Use the projected CG snapshot archive for non-intrusive ROMs."
        )

    n_basis = int(n_basis)
    ZN = Z[:n_basis]
    MZ = [IP * z.vector() for z in ZN]
    G = np.array([[zi.vector().inner(mzj) for mzj in MZ] for zi in ZN], dtype=np.float64)

    C = np.zeros((S.shape[0], n_basis), dtype=np.float64)
    f = Function(solver_obj.V_CG)

    print(f"[POD-AE] Projecting {S.shape[0]} snapshots onto active POD basis N={n_basis}...")
    for i, vec in enumerate(S):
        f.vector().set_local(np.asarray(vec, dtype=np.float64).ravel())
        f.vector().apply("insert")
        rhs = np.array([f.vector().inner(mzj) for mzj in MZ], dtype=np.float64)
        C[i] = np.linalg.solve(G, rhs)
        if (i + 1) % 100 == 0 or (i + 1) == S.shape[0]:
            print(f"[POD-AE] Projected {i + 1}/{S.shape[0]} snapshots")

    return C, _basis_matrix_from_Z(Z, n_basis), "existing_fenics_pod"


def _vector_pod_fallback(S, n_basis=None, energy_tol=0.999999, n_basis_max=40):
    print("[POD-AE] Existing FEniCS POD basis not found; using vector-SVD fallback.")
    _, s, Vt = np.linalg.svd(np.asarray(S, dtype=np.float64), full_matrices=False)
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)

    if n_basis is None:
        n_basis = int(np.searchsorted(energy, float(energy_tol)) + 1)
        n_basis = min(n_basis, int(n_basis_max), Vt.shape[0])
    else:
        n_basis = min(int(n_basis), Vt.shape[0])

    B = Vt[:n_basis].copy()
    C = S @ B.T
    print(f"[POD-AE] Vector POD basis size: N={n_basis}, captured energy={energy[n_basis - 1]:.8f}")
    return C, B, "vector_svd_pod"


def _active_projected_basis_size(cfg):
    g = globals()
    if cfg.get("n_basis") is not None:
        return int(cfg["n_basis"]), "config_n_basis"
    if g.get("N_podnn") is not None:
        return int(g["N_podnn"]), "active_N_podnn"
    if g.get("N_podI") is not None:
        return int(g["N_podI"]), "active_N_podI"
    if g.get("N_podg") is not None:
        return int(g["N_podg"]), "active_N_podg"
    return None, "fallback"


def prepare_pod_ae_data(config=None):
    cfg = dict(POD_AE_CONFIG)
    if config:
        cfg.update(config)

    loaded = load_pod_ae_snapshot_archive(
        cfg.get("snapshot_archive"),
        prefer_projected=cfg.get("prefer_projected", True),
        snapshot_key=cfg.get("snapshot_key", None),
    )

    S, M = loaded["snapshots"], loaded["mus"]
    g = globals()
    solver_obj = g.get("solver", None)
    requested_n_basis, basis_source = _active_projected_basis_size(cfg)

    if all(k in g for k in ["Z_podnn", "IP_N"]) and solver_obj is not None:
        n_basis = int(requested_n_basis if requested_n_basis is not None else g["N_podnn"])
        C, B, pod_source = _project_existing_pod(S, solver_obj, g["Z_podnn"], n_basis, g["IP_N"])
    elif all(k in g for k in ["Z_podI", "IP_I"]) and solver_obj is not None:
        n_basis = int(requested_n_basis if requested_n_basis is not None else g["N_podI"])
        C, B, pod_source = _project_existing_pod(S, solver_obj, g["Z_podI"], n_basis, g["IP_I"])
    else:
        C, B, pod_source = _vector_pod_fallback(
            S,
            n_basis=requested_n_basis,
            energy_tol=cfg.get("pod_energy_tol", 0.999999),
            n_basis_max=cfg.get("n_basis_max", 40),
        )
        n_basis = C.shape[1]

    train_idx, val_idx, test_idx = _split_indices(
        M.shape[0],
        cfg.get("train_fraction", 0.80),
        cfg.get("val_fraction", 0.10),
        cfg.get("test_fraction", 0.10),
        cfg.get("split_seed", cfg.get("seed", 100)),
    )

    print("\n[POD-AE] Prepared training data")
    print(f"  POD source       : {pod_source}")
    print(f"  basis-size source: {basis_source}")
    print(f"  active N_basis   : {n_basis}")
    print(f"  coefficient C    : {C.shape}")
    print(f"  basis matrix     : {B.shape}")
    print(f"  split            : train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    return dict(
        config=cfg,
        archive_path=loaded["path"],
        archive_keys=loaded["keys"],
        snapshot_key=loaded["snapshot_key"],
        mus=M.astype(np.float64, copy=False),
        snapshots=S.astype(np.float64, copy=False),
        coeffs=C.astype(np.float64, copy=False),
        basis_matrix=B.astype(np.float64, copy=False),
        n_basis=int(n_basis),
        pod_source=pod_source,
        basis_size_source=basis_source,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        split_indices=dict(train=train_idx.tolist(), val=val_idx.tolist(), test=test_idx.tolist()),
        solver=solver_obj,
    )


# =============================================================================
# Networks and early stopping
# =============================================================================
class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden=(128, 96, 64), activation="elu",
                 dropout=0.0, layer_norm=False):
        super().__init__()
        self.encoder = _podae_mlp(
            (input_dim, *hidden, latent_dim),
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
            final_activation=False,
        )
        self.decoder = _podae_mlp(
            (latent_dim, *reversed(hidden), input_dim),
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
            final_activation=False,
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class ParameterToLatentMLP(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden=(128, 96, 64), activation="elu",
                 dropout=0.0, layer_norm=False):
        super().__init__()
        self.net = _podae_mlp(
            (input_dim, *hidden, latent_dim),
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
            final_activation=False,
        )

    def forward(self, x):
        return self.net(x)


class _EarlyStopping:
    def __init__(self, patience=400, min_delta=1e-9):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = np.inf
        self.wait = 0
        self.best_state = None
        self.best_epoch = 0

    def step(self, value, model, epoch):
        value = float(value)
        if value < self.best - self.min_delta:
            self.best = value
            self.wait = 0
            self.best_state = _clone_state(model)
            self.best_epoch = int(epoch)
        else:
            self.wait += 1
        return self.wait >= self.patience


# =============================================================================
# POD-AE-ROM model
# =============================================================================
class PODAutoencoderROM:
    def __init__(self, basis_matrix, n_basis, latent_dim, solver=None, config=None):
        self.basis_matrix = np.asarray(basis_matrix, dtype=np.float64)
        self.n_basis = int(n_basis)
        self.latent_dim = int(latent_dim)
        self.solver = solver
        self.config = dict(POD_AE_CONFIG if config is None else config)
        self.device = _podae_device(self.config.get("device_preference", "cpu"))

        self.mu_scaler = StandardScaler()
        self.coeff_scaler = StandardScaler()
        self.latent_scaler = StandardScaler()

        self.autoencoder = None
        self.latent_map = None
        self.history = {}
        self.metadata = {}
        self.metrics = {}
        self.error_df = None
        self.split_indices = {}

    @staticmethod
    def _loader(X, Y, batch_size, shuffle):
        return DataLoader(
            TensorDataset(
                torch.tensor(np.asarray(X, dtype=np.float32)),
                torch.tensor(np.asarray(Y, dtype=np.float32)),
            ),
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
        )

    def _build_networks(self, n_mu):
        cfg = self.config
        self.autoencoder = DenseAutoencoder(
            self.n_basis,
            self.latent_dim,
            hidden=cfg.get("ae_hidden", (128, 96, 64)),
            activation=cfg.get("activation", "elu"),
            dropout=cfg.get("dropout", 0.0),
            layer_norm=cfg.get("layer_norm", False),
        ).to(self.device)

        self.latent_map = ParameterToLatentMLP(
            n_mu,
            self.latent_dim,
            hidden=cfg.get("latent_hidden", (128, 96, 64)),
            activation=cfg.get("activation", "elu"),
            dropout=cfg.get("dropout", 0.0),
            layer_norm=cfg.get("layer_norm", False),
        ).to(self.device)

    def _train_supervised(
        self, model, train_loader, val_loader, *,
        epochs, lr, weight_decay, patience, scheduler_patience, min_delta,
        grad_clip=None, label="stage", log_every=50
    ):
        model = model.to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=int(scheduler_patience)
        )
        loss_fn = nn.MSELoss()
        stopper = _EarlyStopping(patience, min_delta)
        hist = {k: [] for k in ["train", "val", "lr"]}
        t0 = time.perf_counter()

        for ep in range(1, int(epochs) + 1):
            model.train()
            tr_sum = tr_count = 0.0

            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = model(xb)
                pred = pred[0] if isinstance(pred, tuple) else pred
                loss = loss_fn(pred, yb)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                optimizer.step()
                tr_sum += float(loss.item()) * xb.shape[0]
                tr_count += xb.shape[0]

            tr_loss = tr_sum / max(tr_count, 1)

            model.eval()
            va_sum = va_count = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = model(xb)
                    pred = pred[0] if isinstance(pred, tuple) else pred
                    loss = loss_fn(pred, yb)
                    va_sum += float(loss.item()) * xb.shape[0]
                    va_count += xb.shape[0]

            va_loss = va_sum / max(va_count, 1)
            scheduler.step(va_loss)
            curr_lr = optimizer.param_groups[0]["lr"]

            hist["train"].append(tr_loss)
            hist["val"].append(va_loss)
            hist["lr"].append(curr_lr)

            if ep == 1 or ep % int(log_every) == 0 or ep == int(epochs):
                print(
                    f"[POD-AE:{label}] epoch {ep:5d}/{int(epochs):5d} | "
                    f"train={tr_loss:.4e} | val={va_loss:.4e} | lr={curr_lr:.2e}"
                )

            if stopper.step(va_loss, model, ep):
                print(
                    f"[POD-AE:{label}] early stopping at epoch {ep}; "
                    f"best epoch={stopper.best_epoch}, best val={stopper.best:.4e}"
                )
                break

        if stopper.best_state is not None:
            model.load_state_dict(stopper.best_state)
        model.eval()

        elapsed = time.perf_counter() - t0
        hist.update(best_val=float(stopper.best), best_epoch=int(stopper.best_epoch), elapsed_sec=float(elapsed))
        print(f"[POD-AE:{label}] complete in {elapsed:.2f}s | best val={stopper.best:.4e} at epoch {stopper.best_epoch}")
        return hist

    def fit(self, mus, coeffs, train_idx, val_idx, test_idx=None, snapshots=None):
        cfg = self.config
        _podae_seed(cfg.get("seed", 100))
        self.device = _podae_device(cfg.get("device_preference", "cpu"))
        print(f"[POD-AE] Training on device: {self.device}")
        print(f"[POD-AE] Model seed={cfg.get('seed', 100)} | Split seed={cfg.get('split_seed', cfg.get('seed', 100))}")

        M = np.asarray(mus, dtype=np.float64)
        C_raw = np.asarray(coeffs, dtype=np.float64)
        train_idx = np.asarray(train_idx, dtype=int)
        val_idx = np.asarray(val_idx, dtype=int)
        test_idx = np.asarray(test_idx if test_idx is not None else [], dtype=int)

        self.split_indices = dict(train=train_idx.tolist(), val=val_idx.tolist(), test=test_idx.tolist())

        self.mu_scaler.fit(M[train_idx])
        self.coeff_scaler.fit(C_raw[train_idx])
        self.coeff_scaler.scale_[self.coeff_scaler.scale_ < 1e-12] = 1.0

        Mu = self.mu_scaler.transform(M).astype(np.float32)
        C = self.coeff_scaler.transform(C_raw).astype(np.float32)

        n_mu = M.shape[1]
        if self.latent_dim <= 0:
            self.latent_dim = max(1, min(self.n_basis, n_mu))

        self._build_networks(n_mu)

        # Stage 1: coefficient autoencoder
        bs_ae = int(cfg.get("ae_batch_size", cfg.get("batch_size", 256)))
        ae_train = self._loader(C[train_idx], C[train_idx], bs_ae, shuffle=bs_ae < len(train_idx))
        ae_val = self._loader(C[val_idx], C[val_idx], bs_ae, shuffle=False)

        ae_hist = self._train_supervised(
            self.autoencoder, ae_train, ae_val,
            epochs=cfg.get("ae_epochs", 2500),
            lr=cfg.get("lr_ae", 5e-4),
            weight_decay=cfg.get("ae_weight_decay", cfg.get("weight_decay", 2e-6)),
            patience=cfg.get("ae_early_stopping_patience", cfg.get("early_stopping_patience", 450)),
            scheduler_patience=cfg.get("ae_scheduler_patience", cfg.get("scheduler_patience", 120)),
            min_delta=cfg.get("ae_min_delta", cfg.get("min_delta", 5e-9)),
            grad_clip=cfg.get("ae_grad_clip", 1.0),
            label="AE",
            log_every=cfg.get("log_every", 50),
        )

        # Latent codes from trained encoder
        self.autoencoder.eval()
        with torch.no_grad():
            latent_raw = self.autoencoder.encoder(
                torch.tensor(C, dtype=torch.float32, device=self.device)
            ).cpu().numpy()

        self.latent_scaler.fit(latent_raw[train_idx])
        self.latent_scaler.scale_[self.latent_scaler.scale_ < 1e-12] = 1.0
        Zs = self.latent_scaler.transform(latent_raw).astype(np.float32)

        # Stage 2: parameter-to-latent
        bs_lm = int(cfg.get("latent_batch_size", cfg.get("batch_size", 256)))
        lm_train = self._loader(Mu[train_idx], Zs[train_idx], bs_lm, shuffle=bs_lm < len(train_idx))
        lm_val = self._loader(Mu[val_idx], Zs[val_idx], bs_lm, shuffle=False)

        lm_hist = self._train_supervised(
            self.latent_map, lm_train, lm_val,
            epochs=cfg.get("latent_epochs", 2500),
            lr=cfg.get("lr_latent", 5e-4),
            weight_decay=cfg.get("latent_weight_decay", cfg.get("weight_decay", 2e-6)),
            patience=cfg.get("latent_early_stopping_patience", cfg.get("early_stopping_patience", 450)),
            scheduler_patience=cfg.get("latent_scheduler_patience", cfg.get("scheduler_patience", 120)),
            min_delta=cfg.get("latent_min_delta", cfg.get("min_delta", 5e-9)),
            grad_clip=cfg.get("latent_grad_clip", 1.0),
            label="mu-to-latent",
            log_every=cfg.get("log_every", 50),
        )

        self.history = dict(autoencoder=ae_hist, latent_map=lm_hist)
        self.metadata.update(
            n_mu=int(n_mu),
            n_basis=int(self.n_basis),
            latent_dim=int(self.latent_dim),
            ae_best_validation=float(ae_hist["best_val"]),
            latent_best_validation=float(lm_hist["best_val"]),
            ae_train_time=float(ae_hist["elapsed_sec"]),
            latent_train_time=float(lm_hist["elapsed_sec"]),
        )

        print("\n[POD-AE] Initial training complete")
        print(f"  N_basis             : {self.n_basis}")
        print(f"  latent_dim          : {self.latent_dim}")
        print(f"  AE best validation  : {ae_hist['best_val']:.4e}")
        print(f"  map best validation : {lm_hist['best_val']:.4e}")
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _coeff_scaled_from_mu_scaled_tensor(self, mu_scaled_tensor):
        z_scaled = self.latent_map(mu_scaled_tensor)
        z_raw = _torch_inv_scaler(z_scaled, self.latent_scaler)
        return self.autoencoder.decoder(z_raw)

    def predict_coeffs_batch(self, mus, batch_size=4096):
        if self.autoencoder is None or self.latent_map is None:
            raise RuntimeError("PODAutoencoderROM is not trained/loaded.")

        M = np.asarray(mus, dtype=np.float64)
        if M.ndim == 1:
            M = M.reshape(1, -1)

        Mu = self.mu_scaler.transform(M).astype(np.float32)
        preds = []

        self.autoencoder.eval()
        self.latent_map.eval()

        with torch.no_grad():
            for i in range(0, Mu.shape[0], int(batch_size)):
                xb = torch.tensor(Mu[i:i + int(batch_size)], dtype=torch.float32, device=self.device)
                c_scaled = self._coeff_scaled_from_mu_scaled_tensor(xb).cpu().numpy()
                preds.append(c_scaled)

        Cs = np.vstack(preds)
        return self.coeff_scaler.inverse_transform(Cs).astype(np.float64, copy=False)

    def predict_coeffs(self, mu_new):
        return self.predict_coeffs_batch(np.asarray(mu_new, dtype=np.float64).reshape(1, -1))[0]

    def reconstruct_vector(self, coeffs):
        coeffs = np.asarray(coeffs, dtype=np.float64).ravel()
        return coeffs @ self.basis_matrix[:len(coeffs), :]

    def predict_vector(self, mu_new):
        return self.reconstruct_vector(self.predict_coeffs(mu_new))

    def reconstruct_function(self, coeffs):
        if self.solver is None or not hasattr(self.solver, "V_CG"):
            raise RuntimeError("solver.V_CG is required to return a FEniCS Function.")
        vec = self.reconstruct_vector(coeffs)
        if vec.size != self.solver.V_CG.dim():
            raise ValueError(f"Vector size {vec.size} != solver.V_CG.dim()={self.solver.V_CG.dim()}.")
        u = Function(self.solver.V_CG)
        u.vector().set_local(vec)
        u.vector().apply("insert")
        return u

    def predict_function(self, mu_new):
        if self.solver is None or not hasattr(self.solver, "V_CG"):
            raise RuntimeError("solver.V_CG is required to return a FEniCS Function.")
        vec = self.predict_vector(mu_new)
        if vec.size != self.solver.V_CG.dim():
            raise ValueError(f"Predicted vector size {vec.size} != solver.V_CG.dim()={self.solver.V_CG.dim()}.")
        u = Function(self.solver.V_CG)
        u.vector().set_local(vec)
        u.vector().apply("insert")
        return u

    def predict_autoencoder_rom(self, mu_new, return_vector=False):
        return self.predict_vector(mu_new) if return_vector else self.predict_function(mu_new)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def autoencode_coeffs_batch(self, coeffs, batch_size=4096):
        C_raw = np.asarray(coeffs, dtype=np.float64)
        if C_raw.ndim == 1:
            C_raw = C_raw.reshape(1, -1)
        Cs = self.coeff_scaler.transform(C_raw).astype(np.float32)
        outs = []
        self.autoencoder.eval()
        with torch.no_grad():
            for i in range(0, Cs.shape[0], int(batch_size)):
                xb = torch.tensor(Cs[i:i + int(batch_size)], dtype=torch.float32, device=self.device)
                rec_scaled, _ = self.autoencoder(xb)
                outs.append(rec_scaled.cpu().numpy())
        return self.coeff_scaler.inverse_transform(np.vstack(outs)).astype(np.float64, copy=False)

    def _evaluate_indices(self, mus, true_coeffs, snapshot_targets, ids, split_name):
        ids = np.asarray(ids, dtype=int)
        M = np.asarray(mus, dtype=np.float64)
        C_true = np.asarray(true_coeffs, dtype=np.float64)

        t0 = time.perf_counter()
        C_pred = self.predict_coeffs_batch(M[ids])
        pred_time_each = (time.perf_counter() - t0) / max(len(ids), 1)

        Ct = C_true[ids]
        diff_C = C_pred - Ct

        coeff_mse = float(np.mean(diff_C ** 2))
        coeff_rmse = float(np.sqrt(coeff_mse))
        coeff_rel_global = float(np.linalg.norm(diff_C) / (np.linalg.norm(Ct) + 1e-30))
        coeff_rel_each = np.linalg.norm(diff_C, axis=1) / (np.linalg.norm(Ct, axis=1) + 1e-30)

        B = self.basis_matrix[:self.n_basis, :]
        U_pred = C_pred @ B
        U_proj = Ct @ B
        subspace_rel_each = np.linalg.norm(U_pred - U_proj, axis=1) / (np.linalg.norm(U_proj, axis=1) + 1e-30)

        # Offline autoencoder reconstruction diagnostic: C_true -> AE(C_true).
        C_ae_rec = self.autoencode_coeffs_batch(Ct)
        U_ae_rec = C_ae_rec @ B
        ae_coeff_rel_each = np.linalg.norm(C_ae_rec - Ct, axis=1) / (np.linalg.norm(Ct, axis=1) + 1e-30)
        ae_recon_field_rel_each = np.linalg.norm(U_ae_rec - U_proj, axis=1) / (np.linalg.norm(U_proj, axis=1) + 1e-30)

        out = dict(
            split=split_name,
            n_samples=int(len(ids)),
            coeff_mse=coeff_mse,
            coeff_rmse=coeff_rmse,
            coeff_rel_global=coeff_rel_global,
            subspace_field_rel_mean=float(np.mean(subspace_rel_each)),
            subspace_field_rel_median=float(np.median(subspace_rel_each)),
            subspace_field_rel_max=float(np.max(subspace_rel_each)),
            ae_coeff_recon_rel_mean=float(np.mean(ae_coeff_rel_each)),
            ae_coeff_recon_rel_max=float(np.max(ae_coeff_rel_each)),
            ae_field_recon_rel_mean=float(np.mean(ae_recon_field_rel_each)),
            ae_field_recon_rel_max=float(np.max(ae_recon_field_rel_each)),
            pred_time_mean=float(pred_time_each),
        )

        rows = []
        if snapshot_targets is not None:
            S = np.asarray(snapshot_targets, dtype=np.float64)[ids]
            total_rel_each = np.linalg.norm(U_pred - S, axis=1) / (np.linalg.norm(S, axis=1) + 1e-30)
            proj_floor_each = np.linalg.norm(U_proj - S, axis=1) / (np.linalg.norm(S, axis=1) + 1e-30)

            out.update(
                total_field_rel_mean=float(np.mean(total_rel_each)),
                total_field_rel_median=float(np.median(total_rel_each)),
                total_field_rel_max=float(np.max(total_rel_each)),
                projection_floor_mean=float(np.mean(proj_floor_each)),
                projection_floor_median=float(np.median(proj_floor_each)),
                projection_floor_max=float(np.max(proj_floor_each)),
                surrogate_gap_mean=float(np.mean(total_rel_each - proj_floor_each)),
                surrogate_gap_max=float(np.max(total_rel_each - proj_floor_each)),
            )
        else:
            total_rel_each = np.full(len(ids), np.nan)
            proj_floor_each = np.full(len(ids), np.nan)

        for local_k, global_i in enumerate(ids):
            rows.append(dict(
                split=split_name,
                sample_index=int(global_i),
                coeff_rel=float(coeff_rel_each[local_k]),
                subspace_field_rel=float(subspace_rel_each[local_k]),
                ae_coeff_recon_rel=float(ae_coeff_rel_each[local_k]),
                ae_field_recon_rel=float(ae_recon_field_rel_each[local_k]),
                total_field_rel=float(total_rel_each[local_k]),
                projection_floor=float(proj_floor_each[local_k]),
                surrogate_gap=float(total_rel_each[local_k] - proj_floor_each[local_k]),
                pred_time=float(pred_time_each),
            ))

        return out, rows

    def evaluate_splits(self, mus, true_coeffs, snapshot_targets=None, split_indices=None, save_csv=None):
        split_indices = split_indices or self.split_indices
        all_metrics, all_rows = {}, []

        print("\n" + "=" * 94)
        print(f"{'POD-AE train/validation/test diagnostics':^94}")
        print("=" * 94)

        for split in ["train", "val", "test"]:
            if split not in split_indices:
                continue

            metrics, rows = self._evaluate_indices(
                mus, true_coeffs, snapshot_targets,
                split_indices[split],
                split,
            )
            all_metrics[split] = metrics
            all_rows.extend(rows)

            print(f"\n[{split.upper()}] n={metrics['n_samples']}")
            print(f"  coeff RMSE/global rel        : {metrics['coeff_rmse']:.4e} / {metrics['coeff_rel_global']:.4e}")
            print(f"  subspace field rel mean/max  : {metrics['subspace_field_rel_mean']:.4e} / {metrics['subspace_field_rel_max']:.4e}")
            print(f"  AE offline recon mean/max    : {metrics['ae_field_recon_rel_mean']:.4e} / {metrics['ae_field_recon_rel_max']:.4e}")
            if snapshot_targets is not None:
                print(f"  total field rel mean/median  : {metrics['total_field_rel_mean']:.4e} / {metrics['total_field_rel_median']:.4e}")
                print(f"  total field rel max          : {metrics['total_field_rel_max']:.4e}")
                print(f"  POD projection floor mean/max: {metrics['projection_floor_mean']:.4e} / {metrics['projection_floor_max']:.4e}")
                print(f"  surrogate gap mean/max       : {metrics['surrogate_gap_mean']:.4e} / {metrics['surrogate_gap_max']:.4e}")
            print(f"  mean online prediction time  : {metrics['pred_time_mean']:.4e} s/sample")

        print("=" * 94 + "\n")

        df = pd.DataFrame(all_rows)

        if save_csv is not None:
            df.to_csv(save_csv, index=False)
            print(f"[POD-AE] Saved split diagnostics CSV → {save_csv}")

        if bool(self.config.get("save_plots", True)) and len(df) > 0:
            self.plot_split_errors(df, save_dir=podae_dir)

        self.metrics = all_metrics
        self.error_df = df
        return all_metrics, df

    def print_worst_samples(self, df_errors, mus=None, n=10):
        if df_errors is None or len(df_errors) == 0:
            return

        sort_col = "total_field_rel" if "total_field_rel" in df_errors.columns else "subspace_field_rel"

        print("\n" + "=" * 94)
        print(f"{'Worst POD-AE samples by relative field error':^94}")
        print("=" * 94)

        for split in ["train", "val", "test"]:
            d = df_errors[df_errors["split"] == split].copy()
            if len(d) == 0:
                continue
            d = d.sort_values(sort_col, ascending=False).head(int(n))
            print(f"\n[{split.upper()}] worst {min(int(n), len(d))} samples")
            cols = [
                "split", "sample_index", "total_field_rel", "projection_floor",
                "surrogate_gap", "subspace_field_rel", "ae_field_recon_rel", "coeff_rel",
            ]
            cols = [c for c in cols if c in d.columns]
            print(d[cols].to_string(index=False))

            if mus is not None:
                M = np.asarray(mus, dtype=np.float64)
                print("  Parameter values:")
                for sample_idx in d["sample_index"].to_numpy(dtype=int):
                    print(f"    sample {sample_idx:4d}: mu = {np.array2string(M[sample_idx], precision=6, separator=', ')}")

        print("=" * 94 + "\n")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    def plot_training_history(self, save_dir=podae_dir):
        _mkdir(save_dir)
        fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=140)

        offset = 0
        for key, label in [
            ("autoencoder", "AE"),
            ("latent_map", "mu-to-latent"),
            ("end_to_end", "online-path"),
        ]:
            h = self.history.get(key)
            if not h:
                continue
            x = np.arange(1, len(h["train"]) + 1) + offset
            ax.plot(x, h["train"], label=f"{label} train", linestyle="-")
            ax.plot(x, h["val"], label=f"{label} val", linestyle="--")
            offset += len(h["train"])

        ax.set_yscale("log")
        ax.set_xlabel("Epoch / stage-continuation epoch")
        ax.set_ylabel("Loss")
        ax.set_title("POD-AE-ROM training history")
        ax.grid(True, linestyle=":")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "pod_ae_training_history.png"), dpi=300, bbox_inches="tight")
        plt.show()

    def plot_split_errors(self, df_errors, save_dir=podae_dir):
        _mkdir(save_dir)
        fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=140)

        for split in ["train", "val", "test"]:
            d = df_errors[df_errors["split"] == split]
            if len(d):
                y = d["total_field_rel"].to_numpy() if np.isfinite(d["total_field_rel"].to_numpy()).any() else d["subspace_field_rel"].to_numpy()
                ax.plot(np.arange(len(d)), y, marker="o", linewidth=1.3, label=split)

        ax.set_yscale("log")
        ax.set_xlabel("Sample index within split")
        ax.set_ylabel("Relative field error")
        ax.set_title("POD-AE split-wise field errors")
        ax.grid(True, which="both", linestyle=":")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "pod_ae_split_errors.png"), dpi=300, bbox_inches="tight")
        plt.show()

    # ------------------------------------------------------------------
    # Save/load
    # ------------------------------------------------------------------
    def save(self, folder):
        _mkdir(folder)
        torch.save(
            dict(
                autoencoder_state=_clone_state(self.autoencoder),
                latent_map_state=_clone_state(self.latent_map),
                n_basis=self.n_basis,
                latent_dim=self.latent_dim,
                config=self.config,
                metadata=self.metadata,
                history=self.history,
                metrics=self.metrics,
                split_indices=self.split_indices,
            ),
            os.path.join(folder, "model_weights.pt"),
        )

        np.save(os.path.join(folder, "basis_matrix.npy"), self.basis_matrix)

        with open(os.path.join(folder, "scalers.pkl"), "wb") as f:
            pickle.dump(
                dict(
                    mu_scaler=self.mu_scaler,
                    coeff_scaler=self.coeff_scaler,
                    latent_scaler=self.latent_scaler,
                ),
                f,
            )

        with open(os.path.join(folder, "metadata.pkl"), "wb") as f:
            pickle.dump(
                dict(
                    config=self.config,
                    metadata=self.metadata,
                    history=self.history,
                    metrics=self.metrics,
                    split_indices=self.split_indices,
                ),
                f,
            )

        if self.metrics:
            rows = []
            for split, m in self.metrics.items():
                row = dict(m)
                row["split"] = split
                rows.append(row)
            pd.DataFrame(rows).to_csv(os.path.join(folder, "pod_ae_metrics_summary.csv"), index=False)

        if self.error_df is not None:
            self.error_df.to_csv(os.path.join(folder, "pod_ae_train_val_test_errors.csv"), index=False)

        print(f"[POD-AE] Saved model to: {folder}")
        return folder

    @classmethod
    def load(cls, folder, solver=None):
        try:
            weights = torch.load(os.path.join(folder, "model_weights.pt"), map_location="cpu", weights_only=True)
        except TypeError:
            weights = torch.load(os.path.join(folder, "model_weights.pt"), map_location="cpu")

        B = np.load(os.path.join(folder, "basis_matrix.npy"))
        cfg = weights["config"]

        obj = cls(B, weights["n_basis"], weights["latent_dim"], solver=solver, config=cfg)
        obj._build_networks(weights["metadata"]["n_mu"])

        obj.autoencoder.load_state_dict(weights["autoencoder_state"])
        obj.latent_map.load_state_dict(weights["latent_map_state"])

        with open(os.path.join(folder, "scalers.pkl"), "rb") as f:
            scalers = pickle.load(f)

        obj.mu_scaler = scalers["mu_scaler"]
        obj.coeff_scaler = scalers["coeff_scaler"]
        obj.latent_scaler = scalers["latent_scaler"]

        obj.metadata = weights.get("metadata", {})
        obj.history = weights.get("history", {})
        obj.metrics = weights.get("metrics", {})
        obj.split_indices = weights.get("split_indices", {})
        obj.autoencoder.to(obj.device).eval()
        obj.latent_map.to(obj.device).eval()

        print(f"[POD-AE] Loaded model from: {folder}")
        return obj


# =============================================================================
# End-to-end metric-aligned fine-tuning
# =============================================================================
def finetune_pod_ae_end_to_end(
    ae_rom, data,
    epochs=None, lr=None, batch_size=None, train_decoder=None, train_encoder=None,
    patience=None, scheduler_patience=None, weight_decay=None,
):
    cfg = dict(ae_rom.config)
    epochs = int(cfg.get("e2e_epochs", 500) if epochs is None else epochs)
    lr = float(cfg.get("e2e_lr", 3e-5) if lr is None else lr)
    batch_size = int(cfg.get("e2e_batch_size", 256) if batch_size is None else batch_size)
    train_decoder = bool(cfg.get("e2e_train_decoder", True) if train_decoder is None else train_decoder)
    train_encoder = bool(cfg.get("e2e_train_encoder", False) if train_encoder is None else train_encoder)
    patience = int(cfg.get("e2e_patience", 120) if patience is None else patience)
    scheduler_patience = int(cfg.get("e2e_scheduler_patience", 50) if scheduler_patience is None else scheduler_patience)
    weight_decay = float(cfg.get("e2e_weight_decay", 5e-7) if weight_decay is None else weight_decay)

    field_weight = float(cfg.get("e2e_field_weight", 1.0))
    coeff_weight = float(cfg.get("e2e_coeff_weight", 0.01))
    grad_clip = cfg.get("e2e_grad_clip", 1.0)
    grad_clip = None if grad_clip is None else float(grad_clip)
    use_full_target = bool(cfg.get("e2e_use_full_snapshot_target", True))
    min_delta = float(cfg.get("e2e_min_delta", cfg.get("min_delta", 5e-10)))

    M = np.asarray(data["mus"], dtype=np.float64)
    C = np.asarray(data["coeffs"], dtype=np.float64)
    S = np.asarray(data["snapshots"], dtype=np.float64)
    tr = np.asarray(data["train_idx"], dtype=int)
    va = np.asarray(data["val_idx"], dtype=int)

    Mu = ae_rom.mu_scaler.transform(M).astype(np.float32)
    Cs = ae_rom.coeff_scaler.transform(C).astype(np.float32)

    def make_dataset(ids):
        if use_full_target:
            return TensorDataset(
                torch.tensor(Mu[ids], dtype=torch.float32),
                torch.tensor(Cs[ids], dtype=torch.float32),
                torch.tensor(S[ids], dtype=torch.float32),
            )
        return TensorDataset(
            torch.tensor(Mu[ids], dtype=torch.float32),
            torch.tensor(Cs[ids], dtype=torch.float32),
        )

    shuffle_train = int(batch_size) < len(tr)
    train_loader = DataLoader(make_dataset(tr), batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(make_dataset(va), batch_size=batch_size, shuffle=False)

    ae_rom.autoencoder.to(ae_rom.device).eval()
    ae_rom.latent_map.to(ae_rom.device).train()

    for p in ae_rom.autoencoder.encoder.parameters():
        p.requires_grad = train_encoder
    for p in ae_rom.autoencoder.decoder.parameters():
        p.requires_grad = train_decoder
    for p in ae_rom.latent_map.parameters():
        p.requires_grad = True

    params = list(ae_rom.latent_map.parameters())
    if train_decoder:
        params += list(ae_rom.autoencoder.decoder.parameters())
    if train_encoder:
        params += list(ae_rom.autoencoder.encoder.parameters())

    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=scheduler_patience
    )

    coeff_loss_fn = nn.MSELoss()
    B = torch.tensor(ae_rom.basis_matrix[:ae_rom.n_basis, :], dtype=torch.float32, device=ae_rom.device)

    hist = {k: [] for k in ["train", "val", "train_field", "val_field", "train_coeff", "val_coeff", "lr"]}
    best_val, best_epoch, wait, best_state = np.inf, 0, 0, None

    def online_coeff_scaled(mu_batch):
        z_scaled = ae_rom.latent_map(mu_batch)
        z_raw = _torch_inv_scaler(z_scaled, ae_rom.latent_scaler)
        return ae_rom.autoencoder.decoder(z_raw)

    def inv_coeff(c_scaled):
        return _torch_inv_scaler(c_scaled, ae_rom.coeff_scaler)

    def losses(mu_batch, coeff_true_scaled, snapshot_true=None):
        coeff_pred_scaled = online_coeff_scaled(mu_batch)
        coeff_loss = coeff_loss_fn(coeff_pred_scaled, coeff_true_scaled)

        coeff_pred = inv_coeff(coeff_pred_scaled)
        u_pred = coeff_pred @ B

        if snapshot_true is None:
            u_true = inv_coeff(coeff_true_scaled) @ B
        else:
            u_true = snapshot_true

        field_loss = torch.mean(
            torch.sum((u_pred - u_true) ** 2, dim=1) /
            (torch.sum(u_true ** 2, dim=1) + 1e-24)
        )
        total = field_weight * field_loss + coeff_weight * coeff_loss
        return total, field_loss, coeff_loss

    print(
        "[POD-AE:e2e] Fine-tuning online path with field-relative loss "
        f"(train_decoder={train_decoder}, train_encoder={train_encoder}, "
        f"full_snapshot_target={use_full_target}, epochs={epochs}, lr={lr:.2e}, "
        f"field_weight={field_weight:g}, coeff_weight={coeff_weight:g})"
    )

    t0 = time.perf_counter()
    log_every = int(cfg.get("log_every", 50))

    for ep in range(1, epochs + 1):
        ae_rom.latent_map.train()
        if train_decoder:
            ae_rom.autoencoder.decoder.train()
        if train_encoder:
            ae_rom.autoencoder.encoder.train()

        sums = dict(total=0.0, field=0.0, coeff=0.0, count=0)

        for batch in train_loader:
            if use_full_target:
                xb, yb, sb = batch
                sb = sb.to(ae_rom.device)
            else:
                xb, yb = batch
                sb = None

            xb = xb.to(ae_rom.device)
            yb = yb.to(ae_rom.device)

            optimizer.zero_grad()
            loss, f_loss, c_loss = losses(xb, yb, sb)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)

            optimizer.step()

            bs = xb.shape[0]
            sums["total"] += float(loss.item()) * bs
            sums["field"] += float(f_loss.item()) * bs
            sums["coeff"] += float(c_loss.item()) * bs
            sums["count"] += bs

        tr_loss = sums["total"] / max(sums["count"], 1)
        tr_field = sums["field"] / max(sums["count"], 1)
        tr_coeff = sums["coeff"] / max(sums["count"], 1)

        ae_rom.latent_map.eval()
        ae_rom.autoencoder.decoder.eval()
        ae_rom.autoencoder.encoder.eval()
        sums = dict(total=0.0, field=0.0, coeff=0.0, count=0)

        with torch.no_grad():
            for batch in val_loader:
                if use_full_target:
                    xb, yb, sb = batch
                    sb = sb.to(ae_rom.device)
                else:
                    xb, yb = batch
                    sb = None
                xb = xb.to(ae_rom.device)
                yb = yb.to(ae_rom.device)

                loss, f_loss, c_loss = losses(xb, yb, sb)

                bs = xb.shape[0]
                sums["total"] += float(loss.item()) * bs
                sums["field"] += float(f_loss.item()) * bs
                sums["coeff"] += float(c_loss.item()) * bs
                sums["count"] += bs

        va_loss = sums["total"] / max(sums["count"], 1)
        va_field = sums["field"] / max(sums["count"], 1)
        va_coeff = sums["coeff"] / max(sums["count"], 1)
        scheduler.step(va_loss)
        curr_lr = optimizer.param_groups[0]["lr"]

        hist["train"].append(tr_loss)
        hist["val"].append(va_loss)
        hist["train_field"].append(tr_field)
        hist["val_field"].append(va_field)
        hist["train_coeff"].append(tr_coeff)
        hist["val_coeff"].append(va_coeff)
        hist["lr"].append(curr_lr)

        if ep == 1 or ep % log_every == 0 or ep == epochs:
            print(
                f"[POD-AE:e2e] epoch {ep:5d}/{epochs:5d} | "
                f"train={tr_loss:.4e} | val={va_loss:.4e} | "
                f"val_field={va_field:.4e} | val_coeff={va_coeff:.4e} | lr={curr_lr:.2e}"
            )

        if va_loss < best_val - min_delta:
            best_val = float(va_loss)
            best_epoch = int(ep)
            wait = 0
            best_state = dict(
                latent_map=_clone_state(ae_rom.latent_map),
                decoder=_clone_state(ae_rom.autoencoder.decoder),
                encoder=_clone_state(ae_rom.autoencoder.encoder),
            )
        else:
            wait += 1

        if wait >= patience:
            print(f"[POD-AE:e2e] early stopping at epoch {ep}; best epoch={best_epoch}, best val={best_val:.4e}")
            break

    if best_state is not None:
        ae_rom.latent_map.load_state_dict(best_state["latent_map"])
        ae_rom.autoencoder.decoder.load_state_dict(best_state["decoder"])
        ae_rom.autoencoder.encoder.load_state_dict(best_state["encoder"])

    ae_rom.latent_map.eval()
    ae_rom.autoencoder.eval()

    elapsed = time.perf_counter() - t0
    hist.update(best_val=float(best_val), best_epoch=int(best_epoch), elapsed_sec=float(elapsed))
    ae_rom.history["end_to_end"] = hist
    ae_rom.metadata.update(
        e2e_best_validation=float(best_val),
        e2e_best_epoch=int(best_epoch),
        e2e_elapsed_sec=float(elapsed),
        e2e_loss=cfg.get("e2e_loss", "field_relative_plus_coeff"),
        e2e_use_full_snapshot_target=use_full_target,
        e2e_field_weight=field_weight,
        e2e_coeff_weight=coeff_weight,
        e2e_train_decoder=train_decoder,
        e2e_train_encoder=train_encoder,
    )

    print(f"[POD-AE:e2e] complete in {elapsed:.2f}s | best val={best_val:.4e} at epoch {best_epoch}")
    return ae_rom


# =============================================================================
# Training / loading / prediction helpers
# =============================================================================
def train_pod_autoencoder_rom(config=None):
    cfg = dict(POD_AE_CONFIG)
    if config:
        cfg.update(config)

    _podae_seed(cfg.get("seed", 100))
    data = prepare_pod_ae_data(cfg)

    n_mu = data["mus"].shape[1]
    n_basis = int(data["n_basis"])
    latent_dim = _resolve_latent_dim(cfg.get("latent_dim", "basis"), n_basis, n_mu)

    cfg["n_basis"] = n_basis
    cfg["latent_dim"] = latent_dim

    ae = PODAutoencoderROM(
        basis_matrix=data["basis_matrix"],
        n_basis=n_basis,
        latent_dim=latent_dim,
        solver=data["solver"],
        config=cfg,
    )

    ae.metadata.update(
        archive_path=data["archive_path"],
        archive_keys=data["archive_keys"],
        snapshot_key=data.get("snapshot_key", None),
        pod_source=data["pod_source"],
        basis_size_source=data.get("basis_size_source", None),
    )

    ae.fit(
        data["mus"],
        data["coeffs"],
        data["train_idx"],
        data["val_idx"],
        data["test_idx"],
        snapshots=data["snapshots"],
    )

    if cfg.get("end_to_end_finetune", True):
        finetune_pod_ae_end_to_end(ae, data)

    run_folder = _podae_next_run_folder(podae_dir)

    metrics, error_df = ae.evaluate_splits(
        data["mus"],
        data["coeffs"],
        snapshot_targets=data["snapshots"],
        split_indices=data["split_indices"],
        save_csv=os.path.join(run_folder, "pod_ae_train_val_test_errors.csv")
        if bool(cfg.get("save_csv", True)) else None,
    )
    ae.metrics = metrics
    ae.error_df = error_df

    if cfg.get("print_worst_samples", True):
        ae.print_worst_samples(error_df, mus=data["mus"], n=int(cfg.get("n_worst_samples", 10)))

    if metrics.get("test"):
        ae.metadata["test_mean_rel_l2"] = float(metrics["test"].get("total_field_rel_mean", np.nan))
        ae.metadata["test_median_rel_l2"] = float(metrics["test"].get("total_field_rel_median", np.nan))
        ae.metadata["test_max_rel_l2"] = float(metrics["test"].get("total_field_rel_max", np.nan))
        ae.metadata["test_projection_floor_mean"] = float(metrics["test"].get("projection_floor_mean", np.nan))

    if cfg.get("save_model", True):
        ae.save(run_folder)

    if cfg.get("save_plots", True):
        ae.plot_training_history(save_dir=run_folder)
        ae.plot_split_errors(error_df, save_dir=run_folder)

    globals().update(
        pod_ae_data=data,
        pod_ae_error_df=error_df,
        pod_ae_test_df=error_df[error_df["split"] == "test"].copy(),
        pod_ae_split_metrics=metrics,
        pod_ae_folder=run_folder,
        ae_rom=ae,
    )

    print("\n[POD-AE] Ready for online prediction:")
    print("  u_ae = predict_autoencoder_rom(mu_new)")
    print("  v_ae = predict_autoencoder_rom(mu_new, return_vector=True)")
    print(f"  saved folder: {run_folder}")
    return ae, error_df


def load_pod_autoencoder_rom(run_idx=None, base_dir=podae_dir, solver_obj=None, prefer_global=False):
    if prefer_global and globals().get("ae_rom") is not None:
        print("[POD-AE] Using existing global ae_rom.")
        return globals()["ae_rom"]
    folder = os.path.join(base_dir, str(run_idx)) if run_idx is not None else _podae_latest_run(base_dir)
    model = PODAutoencoderROM.load(folder, solver=solver_obj if solver_obj is not None else globals().get("solver", None))
    globals()["ae_rom"] = model
    globals()["pod_ae_folder"] = folder
    return model


def predict_autoencoder_rom(mu_new, return_vector=False):
    if "ae_rom" not in globals():
        raise RuntimeError("ae_rom is not trained/loaded. Run train_pod_autoencoder_rom() first.")
    return globals()["ae_rom"].predict_autoencoder_rom(mu_new, return_vector=return_vector)


def diagnose_pod_ae_projection_gap(data=None, error_df=None, save_csv=True):
    if data is None:
        if "pod_ae_data" not in globals():
            raise RuntimeError("Run train_pod_autoencoder_rom() first.")
        data = globals()["pod_ae_data"]

    if error_df is None:
        if "pod_ae_error_df" not in globals():
            raise RuntimeError("Run train_pod_autoencoder_rom() first.")
        error_df = globals()["pod_ae_error_df"].copy()
    else:
        error_df = error_df.copy()

    if "projection_floor" not in error_df.columns or "total_field_rel" not in error_df.columns:
        raise ValueError("Expected columns projection_floor and total_field_rel in POD-AE error dataframe.")

    error_df["ae_surrogate_gap"] = error_df["total_field_rel"] - error_df["projection_floor"]

    print("\n[POD-AE diagnostic] POD projection baseline and AE surrogate gap")
    print(
        error_df[["split", "projection_floor", "total_field_rel", "ae_surrogate_gap"]]
        .groupby("split")
        .describe()
        .to_string()
    )

    print("\n[POD-AE diagnostic] Error-threshold counts")
    for split in ["train", "val", "test"]:
        d = error_df[error_df["split"] == split]
        if len(d) == 0:
            continue
        print(f"  [{split.upper()}]")
        for thr in [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05]:
            print(f"    AE error > {100 * thr:5.2f}% : {int((d['total_field_rel'] > thr).sum()):3d} / {len(d)}")

    print("\n[POD-AE diagnostic] Worst AE-ROM samples")
    print(error_df.sort_values("total_field_rel", ascending=False).head(10).to_string(index=False))

    if save_csv and "pod_ae_folder" in globals():
        out_csv = os.path.join(globals()["pod_ae_folder"], "pod_ae_projection_gap_diagnostic.csv")
        error_df.to_csv(out_csv, index=False)
        print(f"[POD-AE diagnostic] Saved: {out_csv}")

    globals()["pod_ae_diagnostic_df"] = error_df
    return error_df


# =============================================================================
# Compact POD-AE utilities used by integrated comparison/error-analysis sections
# =============================================================================
def _podae_safe_name(text):
    s = "".join(ch if ch.isalnum() else "_" for ch in str(text)).strip("_")
    return s or "podae_plot"


def _podae_save(fig, base, pad=0.03):
    fig.savefig(f"{base}.pdf", dpi=300, bbox_inches="tight", facecolor="white", pad_inches=pad)
    fig.savefig(f"{base}.png", dpi=600, bbox_inches="tight", facecolor="white", pad_inches=pad)


def _podae_load_model(solver, base_dir=podae_dir, run_idx=None, prefer_global=True):
    if prefer_global and globals().get("ae_rom") is not None:
        print("[POD-AE] Using existing global ae_rom.")
        return globals()["ae_rom"]
    return load_pod_autoencoder_rom(run_idx=run_idx, base_dir=base_dir, solver_obj=solver, prefer_global=False)


def _podae_vec_to_function(vec, solver=solver if "solver" in globals() else None):
    vec = np.asarray(vec, dtype=float).ravel()
    if solver is None or not hasattr(solver, "V_CG"):
        raise RuntimeError("solver.V_CG is required to convert POD-AE vector to FEniCS Function.")
    if vec.size != solver.V_CG.dim():
        raise ValueError(f"POD-AE vector size {vec.size} != solver.V_CG.dim()={solver.V_CG.dim()}.")
    u = Function(solver.V_CG)
    u.vector().set_local(vec)
    u.vector().apply("insert")
    return u


def _podae_project_fom_coeffs(u_fom, podae_rom):
    """
    Projection baseline in the same POD space used by POD-AE-ROM.
    Uses FEniCS/RBniCS projection if Z_podnn and IP_N exist; otherwise least squares.
    """
    n = int(podae_rom.n_basis)

    if all(k in globals() for k in ["Z_podnn", "IP_N"]) and len(Z_podnn) >= n:
        try:
            ZN = Z_podnn[:n]
            MZ = [IP_N * z.vector() for z in ZN]
            G = np.array([[zi.vector().inner(mzj) for mzj in MZ] for zi in ZN], dtype=float)
            rhs = np.array([u_fom.vector().inner(mzj) for mzj in MZ], dtype=float)
            return np.linalg.solve(G, rhs).ravel()
        except Exception as e:
            print(f"[POD-AE-Proj] FEniCS projection failed; using least squares. Reason: {e}")

    u_vec = u_fom.vector().get_local() if hasattr(u_fom, "vector") else np.asarray(u_fom, dtype=float).ravel()
    B = np.asarray(podae_rom.basis_matrix[:n, :], dtype=float)

    if B.shape[1] != u_vec.size:
        raise ValueError(f"POD-AE basis/FOM mismatch: basis shape={B.shape}, FOM vector size={u_vec.size}.")

    try:
        return np.linalg.solve(B @ B.T, B @ u_vec).ravel()
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(B.T, u_vec, rcond=None)[0].ravel()


def _podae_reconstruct_function(podae_rom, coeffs):
    return _podae_vec_to_function(podae_rom.reconstruct_vector(coeffs))


# ─────────────────────────────────────────────────────────────────────────────
# Compact plotting helpers retained from the integrated comparison section
# ─────────────────────────────────────────────────────────────────────────────
def _podae_to_global_factory(solver, V_plot):
    V_DG = FunctionSpace(solver.mesh, "DG", 0)

    if getattr(solver, "N_subdomains", 1) <= 1:
        return lambda sol: project(sol, V_plot)

    subs = np.unique(solver.subdomains.array())
    chi = {sid: Function(V_DG, name=f"chi_{sid}") for sid in subs}
    for sid in subs:
        chi[sid].vector()[:] = (solver.subdomains.array() == sid).astype(float)

    def to_global(sol):
        if sol.function_space().num_sub_spaces() > 0:
            return project(sum(chi[sid] * sol.sub(k) for k, sid in enumerate(subs)), V_plot)
        return project(sol, V_plot)

    return to_global


def _podae_grid(field, X, Y):
    Z = np.zeros_like(X)
    for j in range(X.shape[0]):
        for i in range(X.shape[1]):
            Z[j, i] = field(Point(float(X[j, i]), float(Y[j, i])))
    return Z


def _podae_cbar_ticks(vmin, vmax, positive=False, decimals=1):
    if vmax <= vmin:
        ticks = [vmin] * 4
    elif positive:
        ticks = [0.0, vmax / 3.0, 2.0 * vmax / 3.0, vmax] if vmax > 0 else [0.0] * 4
    else:
        ticks = np.linspace(vmin, vmax, 4).tolist()

    ref = max(abs(vmin), abs(vmax))
    exp = 0 if (not np.isfinite(ref) or ref <= 0.0) else int(np.floor(np.log10(ref)))
    scale = 1.0 if exp == 0 else 10.0 ** exp
    return ticks, [f"{t / scale:.{decimals}f}" for t in ticks], exp


def _podae_map_axis(ax, solver, xlabel=True, ylabel=True):
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.set_xlim(0.0, solver.length)
    ax.set_ylim(0.0, solver.width)
    ax.set_xlabel("x [m]" if xlabel else "", fontsize=14)
    ax.set_ylabel("y [m]" if ylabel else "", fontsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: "" if np.isclose(x, 0.0) else f"{x:.1f}"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis="x", labelsize=13, length=3, pad=1)
    ax.tick_params(axis="y", labelsize=13, length=3, pad=3)
    ax.grid(False)


def _podae_scalar(fig, ax, solver, X, Y, Z, title, *,
                  vmin=None, vmax=None, cmap="viridis", positive=False,
                  xlabel=True, ylabel=True):
    vmin = float(np.nanmin(Z)) if vmin is None else float(vmin)
    vmax = float(np.nanmax(Z)) if vmax is None else float(vmax)
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)):
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + max(1e-300, abs(vmin) * 1e-12)

    m = ax.contourf(X, Y, Z, levels=np.linspace(vmin, vmax, 181),
                    cmap=cmap, vmin=vmin, vmax=vmax)
    _podae_map_axis(ax, solver, xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, fontsize=14, pad=8)

    ticks, labels, exp = _podae_cbar_ticks(vmin, vmax, positive=positive)
    cax = make_axes_locatable(ax).append_axes("right", size="4.2%", pad=0.10)
    cb = fig.colorbar(m, cax=cax)
    cb.set_ticks(ticks)
    cb.ax.yaxis.set_major_locator(FixedLocator(ticks))
    cb.ax.set_yticklabels(labels)
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=13, length=2, pad=2)
    cb.outline.set_linewidth(0.6)
    cb.solids.set_edgecolor("face")
    cb.ax.set_title(rf"$\times 10^{{{exp}}}$" if exp != 0 else "", fontsize=14, pad=6)
    return m


def _podae_line_axis(ax):
    ax.set_facecolor("white")
    ax.set_box_aspect(0.48)
    ax.tick_params(axis="both", labelsize=12, length=3, pad=2)
    ax.grid(True, linestyle=":", linewidth=0.45, alpha=0.35)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(fmt)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.yaxis.get_offset_text().set_size(12)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)


def _podae_shared_legend(fig, ax, y=0.890):
    h, l = ax.get_legend_handles_labels()
    if not h:
        return None
    leg = fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, y),
                     ncol=min(len(l), 6), frameon=True, fontsize=13,
                     borderpad=0.45, handlelength=2.2, handletextpad=0.6,
                     columnspacing=1.2, labelspacing=0.35)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_alpha(0.95)
    return leg


def plot_podae_fom_rom_comparison(solver, fom_solution, rom_solutions, method_names,
                                  title_prefix="", save_dir="."):
    plt.rcParams["figure.autolayout"] = False

    V_plot = FunctionSpace(solver.mesh, "CG", 2)
    to_global = _podae_to_global_factory(solver, V_plot)

    fom = to_global(fom_solution)
    roms = [to_global(r) for r in rom_solutions]

    x = np.linspace(0.0, solver.length, 220)
    y = np.linspace(0.0, solver.width, 220)
    diag = np.sqrt(x**2 + y**2)
    y_mid = solver.width / 2.0

    fom_x = np.array([fom(Point(float(xx), float(y_mid))) for xx in x])
    fom_d = np.array([fom(Point(float(xx), float(yy))) for xx, yy in zip(x, y)])
    rom_x = [np.array([r(Point(float(xx), float(y_mid))) for xx in x]) for r in roms]
    rom_d = [np.array([r(Point(float(xx), float(yy))) for xx, yy in zip(x, y)]) for r in roms]

    X, Y = np.meshgrid(np.linspace(0.0, solver.length, 260),
                       np.linspace(0.0, solver.width, 150))
    fom_grid = _podae_grid(fom, X, Y)

    fig = plt.figure(figsize=(12.0, 8.4), facecolor="white")
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 1.18],
                  width_ratios=[1, 1, 1, 1], left=0.060, right=0.982,
                  bottom=0.070, top=0.805, hspace=0.36, wspace=0.36)
    ax_x, ax_d, ax_f = fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[1, 1:3])

    colors = ["C1", "C2", "C3", "C4", "C5", "C6"]
    styles = ["--", "-.", ":", (0, (5, 2, 1, 2)), (0, (3, 1, 1, 1)), "-"]

    for ax, abscissa, fom_vals, rom_vals, title, xlabel in [
        (ax_x, x, fom_x, rom_x, "Centerline comparison", "x [m]"),
        (ax_d, diag, fom_d, rom_d, "Main diagonal comparison", "Diagonal coordinate [m]"),
    ]:
        ax.plot(abscissa, fom_vals, color="C0", lw=3.2, label=r"$w_{\mathrm{fom}}$")
        for k, (vals, name) in enumerate(zip(rom_vals, method_names)):
            ax.plot(abscissa, vals, color=colors[k % len(colors)], ls=styles[k % len(styles)],
                    lw=2.7, label=name)
        ax.set_title(title, fontsize=13, pad=8)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("w [m]", fontsize=12)
        _podae_line_axis(ax)

    _podae_shared_legend(fig, ax_x, y=0.890)

    _podae_scalar(fig, ax_f, solver, X, Y, fom_grid,
                  r"FOM ($w_{\mathrm{fom}}$ [m])",
                  vmin=float(np.nanmin(fom_grid)),
                  vmax=float(np.nanmax(fom_grid)),
                  cmap="viridis")

    if title_prefix:
        fig.suptitle(title_prefix, fontsize=14, y=0.975)

    os.makedirs(save_dir, exist_ok=True)
    _podae_save(fig, os.path.join(save_dir, f"{_podae_safe_name(title_prefix)}_comparison"))
    plt.show()


def plot_podae_fom_rom_error_fields(solver, u_fom, rom_solutions, method_names,
                                    mu=None, title_prefix="", save_dir="."):
    if not rom_solutions:
        print("  No successful ROM solutions available for field/error plotting.")
        return

    plt.rcParams["figure.autolayout"] = False

    V_plot = FunctionSpace(solver.mesh, "CG", 2)
    to_global = _podae_to_global_factory(solver, V_plot)

    fom = to_global(u_fom)
    roms = [to_global(r) for r in rom_solutions]
    fom_vec = fom.vector().get_local()

    X, Y = np.meshgrid(np.linspace(0.0, solver.length, 260),
                       np.linspace(0.0, solver.width, 150))

    rom_grids, err_grids = [], []
    for r in roms:
        err = Function(V_plot)
        err.vector().set_local(np.abs(fom_vec - r.vector().get_local()))
        err.vector().apply("insert")
        rom_grids.append(_podae_grid(r, X, Y))
        err_grids.append(_podae_grid(err, X, Y))

    n = len(method_names)
    fig = plt.figure(figsize=(12.4, max(4.3 * n, 5.4)), facecolor="white")
    gs = GridSpec(n, 2, figure=fig, left=0.065, right=0.985,
                  bottom=0.055, top=0.900, hspace=0.34, wspace=0.34)

    for j, (rom_Z, err_Z, name) in enumerate(zip(rom_grids, err_grids, method_names)):
        ax_sol, ax_err = fig.add_subplot(gs[j, 0]), fig.add_subplot(gs[j, 1])
        _podae_scalar(fig, ax_sol, solver, X, Y, rom_Z, rf"{name} ($w$ [m])",
                      vmin=float(np.nanmin(rom_Z)), vmax=float(np.nanmax(rom_Z)), cmap="viridis")

        err_max = float(np.nanmax(err_Z))
        err_max = err_max if np.isfinite(err_max) and err_max > 0.0 else 1e-300
        _podae_scalar(fig, ax_err, solver, X, Y, err_Z,
                      rf"$|w_{{\mathrm{{fom}}}}-w_{{\mathrm{{rom}}}}|$ ({name}) [m]",
                      vmin=0.0, vmax=err_max, cmap="plasma", positive=True)

    mu_text = rf"$\mathbf{{\mu}}={np.round(mu, 5).tolist()}$" if mu is not None else r"$\mathbf{\mu}=\mathrm{N/A}$"
    fig.suptitle(f"{title_prefix}   {mu_text}", fontsize=14, y=0.975)

    os.makedirs(save_dir, exist_ok=True)
    _podae_save(fig, os.path.join(save_dir, f"{_podae_safe_name(title_prefix)}_error"))
    plt.show()


# =============================================================================
# Execute
# =============================================================================
RUN_POD_AE_ROM = globals().get("RUN_POD_AE_ROM", True)
RUN_POD_AE_DIAGNOSTIC = globals().get("RUN_POD_AE_DIAGNOSTIC", True)

if choice_podae_run == "live":
    if RUN_POD_AE_ROM:
        ae_rom, pod_ae_error_df = train_pod_autoencoder_rom(POD_AE_CONFIG)
        if RUN_POD_AE_DIAGNOSTIC:
            pod_ae_diagnostic_df = diagnose_pod_ae_projection_gap()
elif choice_podae_run == "load":
    ae_rom = load_pod_autoencoder_rom(run_idx=podae_run_folder, base_dir=podae_dir, solver_obj=globals().get("solver", None))
else:
    raise ValueError(f"Unknown choice_podae_run={choice_podae_run!r}. Use 'live' or 'load'.")

# %% [markdown] Cell 87 | id: 45e5ae6b
# ## Non-Intrusive ROM Comparison / Error Analysis

# %% Cell 88 | id: 5169225c
# =============================================================================
# Non-Intrusive ROM Comparison / Error Analysis / Combined Performance Study
# Project-2 thermomechanical ROM
# -----------------------------------------------------------------------------
# Complete replacement for:
#   1) ### ── Non-Intrusive -- Comparison & Visualization ─────────
#   2) ### ── Non-Intrusive -- Error Analysis ─────────
#   3) ## Combined ROM Performance Study
#
# Updated for the final robust ROM blocks:
#   - PODI-RBF / PODI-Linear robust train/val/test block
#   - POD-NN single-seed pushed robust block
#   - POD-GPR robust train/val/test block
#   - POD-AE robust train/val/test block
#
# Main design:
#   1) Compact structure preserved.
#   2) Robust model loading for latest saved run or specified index.
#   3) Unified prediction dispatch for all non-intrusive methods.
#   4) POD projection retained as a projection-floor reference.
#   5) Optional coupled Project-2 POD-Galerkin is integrated but disabled by default:
#        P2_ENABLE_COUPLED_PODG = False
#      If set True, it runs only when the case is monolithic and the required
#      Paper-2 coupled PODG functions/configuration exist.
#   6) Field/line plots keep the same paper-style formatting.
#   7) Error analysis stores pkl/csv/json and basis-sweep plots.
# =============================================================================


# =============================================================================
# Global directories and default controls
# =============================================================================
Non_Intrusive_dir = os.path.join(solver.output_dir, "ROM_Non_Intrusive")
podi_dir          = os.path.join(Non_Intrusive_dir, "PODI")
podnn_dir         = os.path.join(Non_Intrusive_dir, "PODNN")
podgpr_dir        = os.path.join(Non_Intrusive_dir, "PODGPR")
podae_dir         = os.path.join(Non_Intrusive_dir, "POD_AE")
podproj_dir       = os.path.join(Non_Intrusive_dir, "PODProj")
Intrusive_dir     = os.path.join(solver.output_dir, "ROM_Intrusive")

for _d in [Non_Intrusive_dir, podi_dir, podnn_dir, podgpr_dir, podae_dir, podproj_dir, Intrusive_dir]:
    os.makedirs(_d, exist_ok=True)

# Paper-2 default: coupled intrusive PODG is OFF, but the code is ready for True.
P2_ENABLE_COUPLED_PODG = globals().get("P2_ENABLE_COUPLED_PODG", False)

ROM_LOAD_INDEX = globals().get("ROM_LOAD_INDEX", 1)          # set None to load latest
ROM_TEST_INDICES = globals().get("ROM_TEST_INDICES", [50])   # selected field plots
ROM_RERUN_ANALYSIS = globals().get("ROM_RERUN_ANALYSIS", True)


# =============================================================================
# General utilities
# =============================================================================
def _rom_safe_name(text):
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(text)).strip("_")
    return safe if safe else "rom_plot"


def _rom_save_dual(fig_obj, base_path, pad_inches=0.03):
    fig_obj.savefig(f"{base_path}.pdf", dpi=300, bbox_inches="tight", facecolor="white", pad_inches=pad_inches)
    fig_obj.savefig(f"{base_path}.png", dpi=600, bbox_inches="tight", facecolor="white", pad_inches=pad_inches)


def _rom_latest_index(folder):
    if not os.path.isdir(folder):
        return None
    ids = [int(d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and str(d).isdigit()]
    return max(ids) if ids else None


def _rom_index_or_latest(folder, requested_idx=1):
    if requested_idx is None:
        idx = _rom_latest_index(folder)
        if idx is None:
            raise FileNotFoundError(f"No numeric model folders found in {folder}")
        return idx
    return int(requested_idx)


def _rom_method_root_podi(base_dir, method):
    # Use existing helper if already defined by the PODI block; otherwise use the same convention.
    if "_method_root" in globals():
        try:
            return _method_root(base_dir, method)
        except Exception:
            pass
    m = str(method).lower()
    return os.path.join(base_dir, "PODI_RBF" if m == "rbf" else f"PODI_{m.upper()}")


def _rom_method_root_gpr(base_dir, method="gpr"):
    if "_method_root_gpr" in globals():
        try:
            return _method_root_gpr(base_dir, method)
        except Exception:
            pass
    return base_dir


def _rom_as_numpy_vector(obj):
    if isinstance(obj, np.ndarray):
        return np.asarray(obj, dtype=float).ravel()
    if hasattr(obj, "vector"):
        return obj.vector().get_local()
    if hasattr(obj, "get_local"):
        return obj.get_local()
    return np.asarray(obj, dtype=float).ravel()


def _rom_set_function_vector(function_obj, vec):
    arr = np.asarray(vec, dtype=float).ravel()
    try:
        function_obj.vector().set_local(arr)
    except Exception:
        function_obj.vector()[:] = arr
    try:
        function_obj.vector().apply("insert")
    except Exception:
        pass
    return function_obj


def _rom_function_from_vector(solver, vec, name=None):
    f = Function(solver.V_CG) if name is None else Function(solver.V_CG, name=name)
    return _rom_set_function_vector(f, vec)


def _rom_first_existing_attr(obj, names):
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            try:
                return value() if callable(value) and name.startswith("get_") else value
            except Exception:
                return value
    return None


def _rom_matrix_from_basis(basis_obj, n_dofs):
    if isinstance(basis_obj, np.ndarray):
        B = np.asarray(basis_obj, dtype=float)
    elif isinstance(basis_obj, (list, tuple)):
        B = np.column_stack([_rom_as_numpy_vector(phi) for phi in basis_obj])
    else:
        B = np.asarray(basis_obj, dtype=float)

    B = np.squeeze(B)
    if B.ndim != 2:
        raise ValueError(f"POD basis must be a 2D array/list. Got shape {B.shape}.")
    if B.shape[0] != n_dofs and B.shape[1] == n_dofs:
        B = B.T
    if B.shape[0] != n_dofs:
        raise ValueError(f"POD basis dimension mismatch: basis shape={B.shape}, FOM dofs={n_dofs}.")
    return B


def _rom_dense_matrix(mat_obj):
    if mat_obj is None:
        return None
    if isinstance(mat_obj, np.ndarray):
        return np.asarray(mat_obj, dtype=float)
    if hasattr(mat_obj, "array"):
        return np.asarray(mat_obj.array(), dtype=float)
    if hasattr(mat_obj, "toarray"):
        return np.asarray(mat_obj.toarray(), dtype=float)
    return np.asarray(mat_obj, dtype=float)


def _rom_pod_project_coeffs(u_fom, pod_rom):
    """Direct POD projection of a FOM solution onto the POD space used by pod_rom."""
    u_vec = _rom_as_numpy_vector(u_fom)

    for method_name in [
        "project_coeffs", "project_solution_coeffs", "compute_projection_coeffs",
        "compute_projected_coeffs", "compute_coeffs_from_solution", "project_solution",
    ]:
        if hasattr(pod_rom, method_name):
            method = getattr(pod_rom, method_name)
            for arg in (u_fom, u_vec):
                try:
                    coeffs = method(arg)
                    return np.asarray(coeffs, dtype=float).ravel()
                except Exception:
                    pass

    basis_obj = _rom_first_existing_attr(
        pod_rom,
        ["pod_basis", "basis", "Phi", "V", "reduced_basis", "pod_modes", "modes", "basis_matrix", "_basis_matrix_cache"],
    )
    if basis_obj is None and hasattr(pod_rom, "_basis_matrix"):
        try:
            basis_obj = pod_rom._basis_matrix()
        except Exception:
            basis_obj = None
    if basis_obj is None:
        raise AttributeError("Could not find a POD basis inside pod_rom.")

    B = _rom_matrix_from_basis(basis_obj, len(u_vec))

    mean_obj = _rom_first_existing_attr(
        pod_rom,
        ["snapshot_mean", "mean_snapshot", "mean_vector", "pod_mean", "mean", "u_mean"],
    )
    centered = u_vec.copy()
    if mean_obj is not None:
        mean_vec = _rom_as_numpy_vector(mean_obj)
        if len(mean_vec) == len(centered):
            centered -= mean_vec

    M_obj = _rom_first_existing_attr(
        pod_rom,
        ["mass_matrix", "M", "inner_product_matrix", "projection_matrix", "gram_matrix", "inner_product"],
    )
    M = _rom_dense_matrix(M_obj)
    if M is not None and M.shape == (len(u_vec), len(u_vec)):
        G = B.T @ M @ B
        rhs = B.T @ M @ centered
        return np.linalg.solve(G, rhs).ravel()

    BtB = B.T @ B
    if np.allclose(BtB, np.eye(BtB.shape[0]), rtol=1e-8, atol=1e-10):
        return (B.T @ centered).ravel()
    return np.linalg.lstsq(B, centered, rcond=None)[0].ravel()


def _rom_project_to_global_factory(solver, V_plot):
    V_DG = FunctionSpace(solver.mesh, "DG", 0)
    if getattr(solver, "N_subdomains", 1) > 1:
        subs = np.unique(solver.subdomains.array())
        chi = {sid: Function(V_DG, name=f"chi_{sid}") for sid in subs}
        for sid in subs:
            _rom_set_function_vector(chi[sid], (solver.subdomains.array() == sid).astype(float))

        def to_global(sol):
            if hasattr(sol, "function_space") and sol.function_space().num_sub_spaces() > 0:
                return project(sum(chi[sid] * sol.sub(i) for i, sid in enumerate(subs)), V_plot)
            return project(sol, V_plot)
    else:
        def to_global(sol):
            return project(sol, V_plot)
    return to_global


def _rom_field_on_grid(field, Xg, Yg):
    Z = np.zeros_like(Xg)
    for j in range(Xg.shape[0]):
        for i in range(Xg.shape[1]):
            Z[j, i] = field(Point(float(Xg[j, i]), float(Yg[j, i])))
    return Z


def _rom_four_ticks(vmin, vmax, positive_only=False):
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmax <= vmin:
        return [vmin, vmin, vmin, vmin]
    if positive_only:
        return [0.0, 0.0, 0.0, 0.0] if vmax <= 0.0 else [0.0, vmax / 3.0, 2.0 * vmax / 3.0, vmax]
    return np.linspace(vmin, vmax, 4).tolist()


def _rom_scaled_cbar_info(vmin, vmax, positive_only=False, decimals=1):
    ticks = _rom_four_ticks(vmin, vmax, positive_only=positive_only)
    ref = max(abs(vmin), abs(vmax))
    exponent = 0 if (not np.isfinite(ref) or ref <= 1e-300) else int(np.floor(np.log10(ref)))
    scale = 1.0 if exponent == 0 else 10.0 ** exponent
    return ticks, [f"{t / scale:.{decimals}f}" for t in ticks], exponent


def _rom_style_map_axis(ax, solver, xlabel=True, ylabel=True):
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.set_xlim(0.0, solver.length)
    ax.set_ylim(0.0, solver.width)
    ax.set_xlabel("x [m]" if xlabel else "", fontsize=14)
    ax.set_ylabel("y [m]" if ylabel else "", fontsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: "" if np.isclose(x, 0.0) else f"{x:.1f}"))
    ax.tick_params(axis="x", labelsize=13, length=3, pad=1)
    ax.tick_params(axis="y", labelsize=13, length=3, pad=3)
    ax.grid(False)


def _rom_draw_scalar(fig_obj, ax, solver, Xg, Yg, Z, title, *, vmin=None, vmax=None,
                     cmap="viridis", positive_only=False, xlabel=True, ylabel=True):
    vmin = float(np.nanmin(Z)) if vmin is None else float(vmin)
    vmax = float(np.nanmax(Z)) if vmax is None else float(vmax)
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)):
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + max(1e-300, abs(vmin) * 1e-12)

    mappable = ax.contourf(Xg, Yg, Z, levels=np.linspace(vmin, vmax, 181), cmap=cmap, vmin=vmin, vmax=vmax)
    _rom_style_map_axis(ax, solver, xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, fontsize=14, pad=8)

    ticks, ticklabels, exponent = _rom_scaled_cbar_info(vmin, vmax, positive_only=positive_only, decimals=1)
    cax = make_axes_locatable(ax).append_axes("right", size="4.2%", pad=0.10)
    cbar = fig_obj.colorbar(mappable, cax=cax)
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_locator(FixedLocator(ticks))
    cbar.ax.set_yticklabels(ticklabels)
    cbar.ax.minorticks_off()
    cbar.ax.tick_params(labelsize=13, length=2, pad=2)
    cbar.outline.set_linewidth(0.6)
    cbar.solids.set_edgecolor("face")
    cbar.ax.set_title(rf"$\times 10^{{{exponent}}}$" if exponent != 0 else "", fontsize=14, pad=6)
    return mappable


def _rom_style_legend(legend):
    if legend is None:
        return None
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(0.95)
    return legend


def _rom_style_line_axis(ax, *, show_legend=True, legend_loc="lower left"):
    ax.set_facecolor("white")
    ax.set_box_aspect(0.48)
    ax.tick_params(axis="both", labelsize=12, length=3, pad=2)
    ax.grid(True, linestyle=":", linewidth=0.45, alpha=0.35)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    formatter_y = ScalarFormatter(useMathText=True)
    formatter_y.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(formatter_y)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.yaxis.get_offset_text().set_size(12)
    if show_legend:
        _rom_style_legend(ax.legend(loc=legend_loc, frameon=True, fontsize=13, borderpad=0.45,
                                    handlelength=2.2, handletextpad=0.6, labelspacing=0.35))
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _rom_style_log_axis(ax):
    ax.set_facecolor("white")
    ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=13, length=3, pad=2)
    ax.grid(True, which="both", linestyle=":", linewidth=0.45, alpha=0.35)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _rom_add_shared_line_legend(fig_obj, ax_ref, *, y_anchor=0.890, ncol_max=6):
    handles, labels = ax_ref.get_legend_handles_labels()
    if not handles:
        return None
    legend = fig_obj.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y_anchor),
                            ncol=min(len(labels), ncol_max), frameon=True, fontsize=13, borderpad=0.45,
                            handlelength=2.2, handletextpad=0.6, columnspacing=1.2, labelspacing=0.35)
    return _rom_style_legend(legend)


# =============================================================================
# Robust loaders and prediction dispatch
# =============================================================================
def _rom_load_podi(method, requested_idx=ROM_LOAD_INDEX):
    root = _rom_method_root_podi(podi_dir, method)
    idx = _rom_index_or_latest(root, requested_idx)
    return PODInterpReducedOrderModel.load(solver, pod_index=idx, method=method, base_dir=podi_dir), idx


def _rom_load_podgpr(requested_idx=ROM_LOAD_INDEX):
    root = _rom_method_root_gpr(podgpr_dir, "gpr")
    idx = _rom_index_or_latest(root, requested_idx)
    return PODGPRReducedOrderModel.load(solver, pod_index=idx, method="gpr", base_dir=podgpr_dir), idx


def _rom_load_podnn(requested_idx=ROM_LOAD_INDEX):
    idx = _rom_index_or_latest(podnn_dir, requested_idx)
    try:
        return PODNNReducedOrderModel.load(solver, pod_index=idx, pod_root=podnn_dir), idx
    except TypeError:
        # Backward-compatible fallback for the old POD-NN class.
        arch = {
            "input_dim": len(parameter_ranges[case]) if "parameter_ranges" in globals() and "case" in globals() else len(mu_all_N[0]),
            "nn_hyper": {"hidden_layers": [128, 96, 64], "activation": torch.nn.ELU if torch is not None else None},
        }
        return PODNNReducedOrderModel.load(solver, pod_index=idx, pod_root=podnn_dir, architecture_kwargs=arch), idx


def _rom_load_podae(requested_idx=ROM_LOAD_INDEX):
    if "PODAutoencoderROM" not in globals():
        return None, None
    try:
        if "load_pod_autoencoder_rom" in globals():
            rom = load_pod_autoencoder_rom(run_idx=requested_idx, base_dir=podae_dir, solver_obj=solver, prefer_global=True)
            return rom, requested_idx
        if "_podae_load_model" in globals():
            rom = _podae_load_model(solver, base_dir=podae_dir, run_idx=requested_idx, prefer_global=True)
            return rom, requested_idx
    except Exception as exc:
        print(f"[POD-AE] load failed: {exc}")
    return None, None


def _rom_predict_solution(method_key, rom, mu):
    mu = np.asarray(mu, dtype=float)
    if method_key in ["podi_rbf", "podi_linear"]:
        if hasattr(rom, "predict_solution"):
            return rom.predict_solution(mu)
        return rom.reconstruct_solution(rom.predict_coeffs(mu))
    if method_key == "podgpr":
        if hasattr(rom, "predict_solution"):
            return rom.predict_solution(mu)
        return rom.reconstruct_solution(rom.predict_coeffs(mu))
    if method_key == "podnn":
        if hasattr(rom, "predict_solution"):
            return rom.predict_solution(mu)
        return rom.reconstruct_solution(rom.predict_reduced_coefficients(mu))
    if method_key == "podae":
        return rom.predict_autoencoder_rom(mu, return_vector=False)
    raise KeyError(f"Unknown ROM method key: {method_key}")


def _rom_method_label(method_key):
    return {
        "podproj": "POD-Proj",
        "podi_rbf": "PODI-RBF",
        "podi_linear": "PODI-Linear",
        "podgpr": "POD-GPR",
        "podnn": "POD-NN",
        "podae": "POD-AE-ROM",
        "podg": "Coupled POD-Galerkin",
    }.get(method_key, str(method_key))


def _rom_fallback_value(obj, attr, default="N/A"):
    return getattr(obj, attr, default) if obj is not None else default

# %% [markdown] Cell 89 | id: 1d07ff42
# ### ── Non-Intrusive -- Comparison & Visualization ─────────

# %% Cell 90 | id: 2db031d8
# =============================================================================
# ### ── Non-Intrusive -- Comparison & Visualization ─────────
# =============================================================================
def print_nonintrusive_rom_status():
    print("\n" + "=" * 88)
    print("NON-INTRUSIVE ROM STATUS".center(88))
    print("=" * 88)
    if "choice_podI" in globals():
        print(f"POD-I       : choice={choice_podI}, N_basis={N_podI}, n_snaps={len(snaps_all_I)}")
    if "choice_podnn" in globals():
        print(f"POD-NN      : choice={choice_podnn}, N_basis={N_podnn}, n_snaps={len(snaps_all_N)}")
    if "N_podI" in globals():
        print(f"POD-GPR     : basis=POD-I, N_basis={N_podI}, n_snaps={len(snaps_all_I)}")
        print(f"POD-Proj    : basis=POD-I, N_basis={N_podI}, n_snaps={len(snaps_all_I)}")
    print(f"POD-AE      : active={'podae_rom' in globals() or 'ae_rom' in globals()}, "
          f"N_basis={getattr(globals().get('podae_rom', globals().get('ae_rom', None)), 'n_basis', 'N/A')}, "
          f"latent_dim={getattr(globals().get('podae_rom', globals().get('ae_rom', None)), 'latent_dim', 'N/A')}")
    print("=" * 88)


def _rom_unique_legend_entries(handles, labels):
    """Preserve legend order while removing accidental duplicate labels."""
    seen, h_out, l_out = set(), [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        h_out.append(h)
        l_out.append(l)
    return h_out, l_out


def _rom_add_balanced_two_row_line_legend(fig, ax, *, y_anchor=0.920, ncol=4):
    """
    Add a compact balanced two-row shared legend for the centerline/diagonal comparison plot.

    For the standard seven entries
        w_fom + POD-Proj + PODI-RBF + PODI-Linear + POD-NN + POD-GPR + POD-AE-ROM,
    ncol=4 gives a neat 4 + 3 split instead of one very long row or a 6 + 1 split.
    """
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = _rom_unique_legend_entries(handles, labels)
    if not labels:
        return None

    ncol_eff = min(max(1, int(ncol)), len(labels))
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, y_anchor),
        ncol=ncol_eff,
        frameon=True,
        fontsize=11.4,
        borderpad=0.42,
        handlelength=2.25,
        handletextpad=0.55,
        columnspacing=1.15,
        labelspacing=0.32,
    )
    _rom_style_legend(legend)
    return legend


def plot_fom_rom_comparison(solver, fom_solution, rom_solutions, method_names, title_prefix="", save_dir="."):
    if len(rom_solutions) == 0:
        print("  No successful ROM solutions available for line/scalar plotting.")
        return
    plt.rcParams["figure.autolayout"] = False

    V_plot = FunctionSpace(solver.mesh, "CG", 2)
    to_global = _rom_project_to_global_factory(solver, V_plot)
    fom_global = to_global(fom_solution)
    rom_globals = [to_global(rom_sol) for rom_sol in rom_solutions]

    x_vals = np.linspace(0.0, solver.length, 220)
    y_vals = np.linspace(0.0, solver.width, 220)
    diag_vals = np.sqrt(x_vals**2 + y_vals**2)
    y_fixed = solver.width / 2.0

    w_fom_x = np.array([fom_global(Point(float(x), float(y_fixed))) for x in x_vals])
    w_fom_d = np.array([fom_global(Point(float(x), float(y))) for x, y in zip(x_vals, y_vals)])
    w_rom_x = [np.array([rg(Point(float(x), float(y_fixed))) for x in x_vals]) for rg in rom_globals]
    w_rom_d = [np.array([rg(Point(float(x), float(y))) for x, y in zip(x_vals, y_vals)]) for rg in rom_globals]

    xg = np.linspace(0.0, solver.length, 260)
    yg = np.linspace(0.0, solver.width, 150)
    Xg, Yg = np.meshgrid(xg, yg)
    fom_grid = _rom_field_on_grid(fom_global, Xg, Yg)

    # Keep the original compact figure size.
    # Only reserve a slightly cleaner top band for a balanced two-row legend.
    fig = plt.figure(figsize=(12.0, 8.4), facecolor="white")
    gs = GridSpec(
        2, 4, figure=fig,
        height_ratios=[1.0, 1.18],
        width_ratios=[1, 1, 1, 1],
        left=0.060, right=0.982,
        bottom=0.070, top=0.790,
        hspace=0.37, wspace=0.36,
    )
    ax_x, ax_d, ax_f = fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[1, 1:3])

    colors = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    linestyles = ["--", "-.", ":", (0, (5, 2, 1, 2)), (0, (3, 1, 1, 1)), "-", (0, (1, 1))]

    ax_x.plot(x_vals, w_fom_x, color="C0", lw=3.2, label=r"$w_{\mathrm{fom}}$")
    for k, (rom_x, name) in enumerate(zip(w_rom_x, method_names)):
        ax_x.plot(x_vals, rom_x, color=colors[k % len(colors)], ls=linestyles[k % len(linestyles)], lw=2.7, label=name)
    ax_x.set_title(r"Centerline comparison", fontsize=13, pad=8)
    ax_x.set_xlabel("x [m]", fontsize=12)
    ax_x.set_ylabel("w [m]", fontsize=12)
    _rom_style_line_axis(ax_x, show_legend=False)

    ax_d.plot(diag_vals, w_fom_d, color="C0", lw=3.2, label=r"$w_{\mathrm{fom}}$")
    for k, (rom_d, name) in enumerate(zip(w_rom_d, method_names)):
        ax_d.plot(diag_vals, rom_d, color=colors[k % len(colors)], ls=linestyles[k % len(linestyles)], lw=2.7, label=name)
    ax_d.set_title(r"Main diagonal comparison", fontsize=13, pad=8)
    ax_d.set_xlabel("Diagonal coordinate [m]", fontsize=12)
    ax_d.set_ylabel("w [m]", fontsize=12)
    _rom_style_line_axis(ax_d, show_legend=False)

    # Balanced two-row legend: for seven entries this becomes 4 + 3, avoiding the old 6 + 1 layout.
    _rom_add_balanced_two_row_line_legend(fig, ax_x, y_anchor=0.920, ncol=4)

    _rom_draw_scalar(fig, ax_f, solver, Xg, Yg, fom_grid, r"FOM ($w_{\mathrm{fom}}$ $[m]$)",
                     vmin=float(np.nanmin(fom_grid)), vmax=float(np.nanmax(fom_grid)), cmap="viridis")

    if title_prefix:
        fig.suptitle(title_prefix, fontsize=14, y=0.975)

    os.makedirs(save_dir, exist_ok=True)
    _rom_save_dual(fig, os.path.join(save_dir, f"{_rom_safe_name(title_prefix)}_comparison"))
    plt.show()


def plot_fom_rom_error_fields(solver, u_fom, rom_solutions, method_names, mu=None, title_prefix="", save_dir="."):
    if len(rom_solutions) == 0:
        print("  No successful ROM solutions available for field/error plotting.")
        return
    plt.rcParams["figure.autolayout"] = False

    V_plot = FunctionSpace(solver.mesh, "CG", 2)
    to_global = _rom_project_to_global_factory(solver, V_plot)
    w_fom_global = to_global(u_fom)
    w_rom_globals = [to_global(rom_sol) for rom_sol in rom_solutions]

    fom_vec = w_fom_global.vector().get_local()
    error_fields = []
    for w_rom_global in w_rom_globals:
        err = Function(V_plot)
        _rom_set_function_vector(err, np.abs(fom_vec - w_rom_global.vector().get_local()))
        error_fields.append(err)

    xg = np.linspace(0.0, solver.length, 260)
    yg = np.linspace(0.0, solver.width, 150)
    Xg, Yg = np.meshgrid(xg, yg)

    rom_grids = [_rom_field_on_grid(field, Xg, Yg) for field in w_rom_globals]
    err_grids = [_rom_field_on_grid(field, Xg, Yg) for field in error_fields]

    n_methods = len(method_names)
    fig = plt.figure(figsize=(12.4, max(4.3 * n_methods, 5.4)), facecolor="white")
    gs = GridSpec(n_methods, 2, figure=fig, left=0.065, right=0.985, bottom=0.055,
                  top=0.900, hspace=0.34, wspace=0.34)

    for j, (rom_Z, err_Z, name) in enumerate(zip(rom_grids, err_grids, method_names)):
        ax_sol, ax_err = fig.add_subplot(gs[j, 0]), fig.add_subplot(gs[j, 1])
        _rom_draw_scalar(fig, ax_sol, solver, Xg, Yg, rom_Z, rf"{name} ($w$ $[m]$)",
                         vmin=float(np.nanmin(rom_Z)), vmax=float(np.nanmax(rom_Z)), cmap="viridis")
        err_max = float(np.nanmax(err_Z))
        if (not np.isfinite(err_max)) or err_max <= 0.0:
            err_max = 1e-300
        _rom_draw_scalar(fig, ax_err, solver, Xg, Yg, err_Z,
                         rf"$|w_{{\mathrm{{fom}}}}-w_{{\mathrm{{rom}}}}|$ ({name}) $[m]$",
                         vmin=0.0, vmax=err_max, cmap="plasma", positive_only=True)

    mu_text = rf"$\mathbf{{\mu}}={np.round(mu, 5).tolist()}$" if mu is not None else r"$\mathbf{\mu}=\mathrm{N/A}$"
    fig.suptitle(f"{title_prefix}   {mu_text}", fontsize=14, y=0.975)

    os.makedirs(save_dir, exist_ok=True)
    _rom_save_dual(fig, os.path.join(save_dir, f"{_rom_safe_name(title_prefix)}_error"))
    plt.show()


def plot_nonintrusive_error_fields_grid(solver, u_fom, rom_solutions, method_names, success_flags=None,
                                        mu=None, title_prefix="", save_dir="."):
    plt.rcParams["figure.autolayout"] = False
    if success_flags is None:
        success_flags = [True] * len(method_names)
    if len(method_names) == 0:
        print("No ROM methods available for error-only plotting.")
        return

    V_plot = FunctionSpace(solver.mesh, "CG", 2)
    to_global = _rom_project_to_global_factory(solver, V_plot)
    w_fom_global = to_global(u_fom)
    fom_vec = w_fom_global.vector().get_local()

    xg = np.linspace(0.0, solver.length, 260)
    yg = np.linspace(0.0, solver.width, 150)
    Xg, Yg = np.meshgrid(xg, yg)

    n_cols = 3 if len(method_names) > 4 else 2
    n_rows = int(np.ceil(len(method_names) / n_cols))
    fig = plt.figure(figsize=(6.2 * n_cols, 4.1 * n_rows), facecolor="white")
    gs = GridSpec(n_rows, n_cols, figure=fig, left=0.055, right=0.985, bottom=0.070,
                  top=0.905, hspace=0.16, wspace=0.30)

    for ax_id, (u_rom, name, ok) in enumerate(zip(rom_solutions, method_names, success_flags)):
        ax = fig.add_subplot(gs[ax_id // n_cols, ax_id % n_cols])
        if not ok:
            _rom_style_map_axis(ax, solver)
            ax.set_title(rf"$|w_{{\mathrm{{fom}}}}-w_{{\mathrm{{rom}}}}|$ ({name}) $[m]$", fontsize=14, pad=8)
            ax.text(0.5, 0.5, "FAILED", transform=ax.transAxes, ha="center", va="center", fontsize=16)
            continue

        w_rom_global = to_global(u_rom)
        err = Function(V_plot)
        _rom_set_function_vector(err, np.abs(fom_vec - w_rom_global.vector().get_local()))
        err_Z = _rom_field_on_grid(err, Xg, Yg)
        err_max = float(np.nanmax(err_Z))
        if (not np.isfinite(err_max)) or err_max <= 0.0:
            err_max = 1e-300
        _rom_draw_scalar(fig, ax, solver, Xg, Yg, err_Z,
                         rf"$|w_{{\mathrm{{fom}}}}-w_{{\mathrm{{rom}}}}|$ ({name}) $[m]$",
                         vmin=0.0, vmax=err_max, cmap="plasma", positive_only=True)

    mu_text = rf"$\mathbf{{\mu}}={np.round(mu, 5).tolist()}$" if mu is not None else r"$\mathbf{\mu}=\mathrm{N/A}$"
    fig.suptitle(f"{title_prefix}   {mu_text}", fontsize=14, y=0.975)
    os.makedirs(save_dir, exist_ok=True)
    _rom_save_dual(fig, os.path.join(save_dir, f"{_rom_safe_name(title_prefix)}_error_only_grid"))
    plt.show()


def load_final_nonintrusive_models(load_index=ROM_LOAD_INDEX, verbose=True):
    models, indices = {}, {}

    try:
        models["podi_rbf"], indices["podi_rbf"] = _rom_load_podi("rbf", load_index)
    except Exception as exc:
        print(f"[PODI-RBF] not loaded: {exc}")
    try:
        models["podi_linear"], indices["podi_linear"] = _rom_load_podi("linear", load_index)
    except Exception as exc:
        print(f"[PODI-Linear] not loaded: {exc}")
    try:
        models["podnn"], indices["podnn"] = _rom_load_podnn(load_index)
    except Exception as exc:
        print(f"[POD-NN] not loaded: {exc}")
    try:
        models["podgpr"], indices["podgpr"] = _rom_load_podgpr(load_index)
    except Exception as exc:
        print(f"[POD-GPR] not loaded: {exc}")
    try:
        models["podae"], indices["podae"] = _rom_load_podae(load_index)
        if models["podae"] is None:
            models.pop("podae", None)
            indices.pop("podae", None)
    except Exception as exc:
        print(f"[POD-AE] not loaded: {exc}")

    if verbose:
        print("\nLoaded non-intrusive ROM models:")
        for key, rom in models.items():
            extra = ""
            if key == "podae":
                extra = f", latent_dim={getattr(rom, 'latent_dim', 'N/A')}"
            print(f"  {_rom_method_label(key):14s}: index={indices.get(key, 'N/A')}, N_basis={getattr(rom, 'n_basis', 'N/A')}{extra}")

    # Keep old global names for downstream compatibility.
    globals().update(
        podi_rbf=models.get("podi_rbf"),
        podi_linear=models.get("podi_linear"),
        pod_nn_rom=models.get("podnn"),
        podgpr_rom=models.get("podgpr"),
        podae_rom=models.get("podae"),
        podi_models={k.replace("podi_", ""): v for k, v in models.items() if k.startswith("podi_")},
        podgpr_models={"gpr": models.get("podgpr")} if "podgpr" in models else {},
        podae_models={"ae": models.get("podae")} if "podae" in models else {},
    )
    return models, indices


def evaluate_selected_nonintrusive_samples(test_indices=None, load_index=ROM_LOAD_INDEX, include_pod_projection=True):
    print_nonintrusive_rom_status()
    models, indices = load_final_nonintrusive_models(load_index=load_index, verbose=True)

    if test_indices is None:
        test_indices = ROM_TEST_INDICES
    test_indices = [int(i) for i in test_indices if 0 <= int(i) < len(mu_list)]
    if not test_indices:
        print("No valid non-intrusive test indices. Skipping selected-sample comparison.")
        return {}, {}

    results_by_sample = {}
    for i in test_indices:
        mu_test, u_fom = np.asarray(mu_list[i], dtype=float), fom_solutions[i]
        print("\n" + "-" * 88)
        print(f"Evaluating non-intrusive μ[{i}] = {np.round(mu_test, 6).tolist()}")
        print("-" * 88)

        rom_solutions, method_names, success_flags = [], [], []
        sample_results = {}

        if include_pod_projection and "podi_rbf" in models:
            print("  ► Testing POD-Proj...")
            try:
                c_proj = _rom_pod_project_coeffs(u_fom, models["podi_rbf"])
                u_rom_proj = models["podi_rbf"].reconstruct_solution(c_proj)
                abs_err, rel_err = compute_rom_errors(solver, u_fom, u_rom_proj)
                print(f"    ➤ POD-Proj: Abs Error = {abs_err:.4e} | Rel Error = {rel_err:.4%}")
                rom_solutions.append(u_rom_proj)
                method_names.append("POD-Proj")
                success_flags.append(True)
                sample_results["podproj"] = dict(abs_err=abs_err, rel_err=rel_err)
            except Exception as exc:
                print(f"    ➤ POD-Proj FAILED: {exc}")

        for key in ["podi_rbf", "podi_linear", "podnn", "podgpr", "podae"]:
            if key not in models:
                continue
            print(f"  ► Testing {_rom_method_label(key)}...")
            try:
                tic = clock()
                u_rom = _rom_predict_solution(key, models[key], mu_test)
                t_rom = clock() - tic
                abs_err, rel_err = compute_rom_errors(solver, u_fom, u_rom)
                print(f"    ➤ {_rom_method_label(key)}: Abs Error = {abs_err:.4e} | Rel Error = {rel_err:.4%} | time={t_rom:.3e}s")
                rom_solutions.append(u_rom)
                method_names.append(_rom_method_label(key))
                success_flags.append(True)
                sample_results[key] = dict(abs_err=abs_err, rel_err=rel_err, time=t_rom)
            except Exception as exc:
                print(f"    ➤ {_rom_method_label(key)} FAILED: {exc}")

        prefix = f"NonIntrusive_All_Methods_N={getattr(models.get('podi_rbf'), 'n_basis', 'NA')}_mu_{i}"
        print(f"\n  ► Plotting comparison for {sum(success_flags)} successful methods...")
        if rom_solutions:
            plot_fom_rom_comparison(
                solver, u_fom, rom_solutions, method_names,
                title_prefix=prefix, save_dir=Non_Intrusive_dir
            )
            plot_fom_rom_error_fields(
                solver, u_fom, rom_solutions, method_names,
                mu=mu_test, title_prefix=prefix, save_dir=Non_Intrusive_dir
            )
            plot_nonintrusive_error_fields_grid(
                solver, u_fom, rom_solutions, method_names, success_flags,
                mu=mu_test, title_prefix=prefix, save_dir=Non_Intrusive_dir
            )
        results_by_sample[i] = sample_results

    return models, results_by_sample


# Run selected-sample comparison now. Set ROM_SKIP_SELECTED_COMPARISON=True to skip.
if not globals().get("ROM_SKIP_SELECTED_COMPARISON", False):
    final_nonintrusive_models, selected_nonintrusive_results = evaluate_selected_nonintrusive_samples(
        test_indices=ROM_TEST_INDICES,
        load_index=ROM_LOAD_INDEX,
        include_pod_projection=True,
    )

# %% [markdown] Cell 91 | id: bfbf6230
# ### ── Non-Intrusive -- Error Analysis ─────────

# %% Cell 92 | id: b7069212
# PICK REDUCED BASIS CHOICES - Solution SPACE
choice_podg, choice_podlspg, n_basis_podg, n_basis_podlspg = 0, 0, _p2_cap_basis(N_podg, 15), _p2_cap_basis(N_podlspg, 15)
eigG, eigvG, basisG, N_podg, Z_podg, IP_G, mu_all_G, snaps_all_G = rb_bank_Solution[choice_podg]
N_podg = n_basis_podg or N_podg
eigL, eigvL, basisL, N_podlspg, Z_podlspg, IP_L, mu_all_L, snaps_all_L = rb_bank_Solution[choice_podlspg]
N_podlspg = n_basis_podlspg or N_podlspg

# PICK REDUCED BASIS CHOICES - PROJECTED SPACE
choice_podnn, choice_podI, n_basis_podnn, n_basis_podI = 0, 0, _p2_cap_basis(N_podnn, 15), _p2_cap_basis(N_podI, 15)
eigN, eigvN, basisN, N_podnn, Z_podnn, IP_N, mu_all_N, snaps_all_N = rb_bank_proj[choice_podnn]
N_podnn = n_basis_podnn or N_podnn
eigI, eigvI, basisI, N_podI, Z_podI, IP_I, mu_all_I, snaps_all_I = rb_bank_proj[choice_podI]
N_podI = n_basis_podI or N_podI

# %% Cell 93 | id: 34f436d6
# =============================================================================
# ### ── Non-Intrusive -- Error Analysis ─────────
# =============================================================================
def _rom_basis_sweep_list(Nmax, n_points=5):
    """
    Build a compact monotone basis sweep list.

    Example:
      Nmax=4  -> [1, 2, 3, 4]
      Nmax=8  -> [1, 2, 4, 6, 8]
      Nmax=15 -> [1, 4, 8, 11, 15]

    This intentionally follows the same style as the intrusive performance
    study: sorted(set([1] + linspace(1, Nmax, n_points) + [Nmax])).
    """
    Nmax = int(Nmax)
    n_points = int(n_points)
    if Nmax <= 1:
        return [1]
    return sorted(set([1] + np.linspace(1, Nmax, n_points, dtype=int).tolist() + [Nmax]))


def _rom_build_snapshot_functions(snap_vectors, prefix="snap"):
    funcs = []
    for k, vec in enumerate(snap_vectors):
        funcs.append(_rom_function_from_vector(solver, vec, name=f"{prefix}_{k}"))
    return funcs


def _rom_train_podi_for_sweep(method, N, C_full, snapshot_targets=None):
    cfg = dict(globals().get("PODI_CONFIG", {}))
    if not cfg:
        cfg = dict(
            seed=100,
            split_seed=100,
            train_fraction=0.80,
            val_fraction=0.10,
            test_fraction=0.10,
            save_plots=False,
            save_csv=False,
            print_worst_samples=False,
        )
    cfg.update(
        save_plots=False,
        save_csv=False,
        print_worst_samples=False,
        n_basis=int(N),
    )

    interp_kwargs = (
        dict(cfg.get("rbf_kwargs", {"kernel": "thin_plate_spline", "smoothing": 0.0}))
        if method == "rbf"
        else dict(cfg.get("linear_kwargs", {}))
    )

    rom = PODInterpReducedOrderModel(solver, Z_podI, int(N), IP_I, config=cfg)
    rom.fit_interpolator(
        mu_all_I,
        C_full[:, :int(N)],
        method=method,
        interp_kwargs=interp_kwargs,
        snapshot_targets=snapshot_targets,
    )
    return rom


def _rom_train_podgpr_for_sweep(N, C_full, snapshot_targets=None):
    cfg = dict(globals().get("PODGPR_CONFIG", {})) if "PODGPR_CONFIG" in globals() else {}
    if cfg:
        cfg.update(
            save_plots=False,
            save_csv=False,
            print_worst_samples=False,
            n_basis=int(N),
        )
        rom = PODGPRReducedOrderModel(solver, Z_podI, int(N), IP_I, config=cfg)
    else:
        rom = PODGPRReducedOrderModel(solver, Z_podI, int(N), IP_I)

    gpr_kwargs = None
    if "PODGPR_CONFIG" in globals():
        gpr_kwargs = dict(PODGPR_CONFIG.get("gpr_kwargs", {}))

    if not gpr_kwargs:
        if GP_Const is not None:
            gpr_kwargs = dict(
                kernel=GP_Const(1.0, (1e-2, 1e2)) * GP_RBF(np.ones(mu_all_I.shape[1]), (1e-2, 1e2)),
                alpha=1e-9,
                random_state=100,
                n_restarts_optimizer=5,
                normalize_y=False,
            )
        else:
            gpr_kwargs = dict(
                alpha=1e-9,
                random_state=100,
                n_restarts_optimizer=5,
            )

    try:
        rom.fit_gpr(
            mu_all_I,
            C_full[:, :int(N)],
            method="gpr",
            gpr_kwargs=gpr_kwargs,
            snapshot_targets=snapshot_targets,
        )
    except TypeError:
        # Backward-compatible robust GPR API.
        rom.fit_gpr(mu_all_I, C_full[:, :int(N)], method="gpr", gpr_kwargs=gpr_kwargs)

    return rom


def _rom_train_podnn_for_sweep(N, C_full, mus, snapshot_targets=None):
    cfg = dict(globals().get("PODNN_CONFIG", {}))
    if cfg:
        cfg.update(
            save_model=False,
            save_plots=False,
            print_worst_samples=False,
            seed=100,
            split_seed=100,
            n_basis=int(N),
            N_basis=int(N),
        )

    rom = (
        PODNNReducedOrderModel(solver, Z_podnn, int(N), IP_N, config=cfg)
        if cfg
        else PODNNReducedOrderModel(solver, Z_podnn, int(N), IP_N)
    )

    if hasattr(rom, "fit"):
        rom.fit(mus, C_full[:, :int(N)], snapshot_targets=snapshot_targets)
    else:
        # Backward-compatible old POD-NN API.
        rom.train_nn(
            training_params=mus,
            reduced_coeffs=C_full[:, :int(N)],
            epochs=1500,
            lr=5e-4,
            batch_size=256,
            test_size=0.2,
        )

    return rom


def _rom_train_podae_for_sweep(N):
    cfg_ae_N = dict(globals().get("POD_AE_CONFIG", {}))
    cfg_ae_N.update(
        n_basis=int(N),
        N_basis=int(N),
        latent_dim="basis",
        save_model=False,
        save_plots=False,
        print_worst_samples=False,
    )

    data_ae_N = prepare_pod_ae_data(cfg_ae_N)
    latent_N = _resolve_latent_dim(
        cfg_ae_N.get("latent_dim", "basis"),
        int(data_ae_N["n_basis"]),
        data_ae_N["mus"].shape[1],
    )
    cfg_ae_N["latent_dim"] = latent_N

    rom_ae = PODAutoencoderROM(
        data_ae_N["basis_matrix"],
        data_ae_N["n_basis"],
        latent_N,
        solver=data_ae_N["solver"],
        config=cfg_ae_N,
    )

    snapshots_arg = data_ae_N.get("snapshots", data_ae_N.get("snapshot_targets", None))
    try:
        rom_ae.fit(
            data_ae_N["mus"],
            data_ae_N["coeffs"],
            data_ae_N["train_idx"],
            data_ae_N["val_idx"],
            data_ae_N["test_idx"],
            snapshots=snapshots_arg,
        )
    except TypeError:
        # Backward-compatible POD-AE API.
        rom_ae.fit(
            data_ae_N["mus"],
            data_ae_N["coeffs"],
            data_ae_N["train_idx"],
            data_ae_N["val_idx"],
            data_ae_N["test_idx"],
        )

    if cfg_ae_N.get("end_to_end_finetune", True):
        finetune_pod_ae_end_to_end(rom_ae, data_ae_N)

    return rom_ae


def _rom_save_sweep_model(rom, method_key, N_basis, model_dir):
    os.makedirs(model_dir, exist_ok=True)

    try:
        if hasattr(rom, "save") and method_key in ["podi_rbf", "podi_linear", "podgpr", "podnn"]:
            # Use compact custom pickle for sweep models to avoid nesting many numbered folders.
            model_path = os.path.join(model_dir, f"model_N{int(N_basis)}.pkl")
            data = dict(method_key=method_key, n_basis=int(N_basis))

            for attr in [
                "method",
                "config",
                "interp_kwargs",
                "gpr_kwargs",
                "param_scaler",
                "coeff_scaler",
                "interpolator",
                "nearest_interpolator",
                "gp_models",
                "X_scaler",
                "model",
                "model_state_dict",
                "nn_hyper",
                "coeff_mean",
                "coeff_std",
                "split_indices",
                "training_summary",
                "metrics",
            ]:
                if hasattr(rom, attr):
                    val = getattr(rom, attr)
                    data[attr] = val.state_dict() if attr == "model" and hasattr(val, "state_dict") else val

            data["basis_vectors"] = (
                [phi.vector().get_local() for phi in rom.reduced_basis[: rom.n_basis]]
                if hasattr(rom, "reduced_basis")
                else None
            )

            with open(model_path, "wb") as f:
                pickle.dump(data, f)

            return model_path

        if method_key == "podae" and hasattr(rom, "save"):
            folder = os.path.join(model_dir, f"model_N{int(N_basis)}")
            os.makedirs(folder, exist_ok=True)
            rom.save(folder)
            return folder

    except Exception as exc:
        print(f"    ✗ Save failed for {_rom_method_label(method_key)} N={N_basis}: {str(exc)[:100]}")

    return None


def _rom_available_basis_count(Z_raw, N_raw=None):
    """
    Robust available-basis counter for FEniCS basis lists or matrix-like basis arrays.
    """
    try:
        if isinstance(Z_raw, (list, tuple)):
            z_len = len(Z_raw)
        else:
            z_arr = np.asarray(Z_raw)
            if z_arr.ndim == 2:
                # Accept both (n_dofs, n_modes) and (n_modes, n_dofs).
                z_len = min(z_arr.shape) if min(z_arr.shape) < max(z_arr.shape) else z_arr.shape[1]
            else:
                z_len = len(Z_raw)
    except Exception:
        z_len = int(N_raw) if N_raw is not None else 0

    if N_raw is None:
        return int(z_len)

    return int(min(int(N_raw), int(z_len))) if z_len > 0 else int(N_raw)


def _rom_cap_basis_for_sweep(N_available, cap):
    """
    Apply the same Project-2 basis cap convention, but never exceed the available basis.
    """
    N_available = int(N_available)
    cap = int(cap)
    if "_p2_cap_basis" in globals():
        try:
            return int(_p2_cap_basis(N_available, cap))
        except Exception:
            pass
    return int(min(N_available, cap))


def _rom_prepare_projected_basis_for_sweep():
    """
    Prepare projected-space POD banks for non-intrusive basis sweep.

    Important fix:
      The sweep must use the raw available basis size from rb_bank_proj, not a
      previously overwritten active N_podI/N_podnn from single-model runs.

    Without this, a previous active setting such as N_podI=4 forces the sweep to
    [1,2,3,4], even if rb_bank_proj contains 8 or 15 available modes.
    """
    global choice_podnn, choice_podI
    global N_podnn, Z_podnn, IP_N, mu_all_N, snaps_all_N
    global N_podI, Z_podI, IP_I, mu_all_I, snaps_all_I
    global N_podnn_raw_for_sweep, N_podI_raw_for_sweep

    if "rb_bank_proj" not in globals():
        # Fallback: use already-active globals.
        N_podnn_raw_for_sweep = _rom_available_basis_count(Z_podnn, N_podnn)
        N_podI_raw_for_sweep = _rom_available_basis_count(Z_podI, N_podI)
        return

    choice_podnn = int(globals().get("choice_podnn", 0))
    choice_podI = int(globals().get("choice_podI", 0))

    eigN, eigvN, basisN, Nn_raw, Zn_raw, IPn_raw, mun_raw, snn_raw = rb_bank_proj[choice_podnn]
    eigI, eigvI, basisI, Ni_raw, Zi_raw, IPi_raw, mui_raw, sni_raw = rb_bank_proj[choice_podI]

    N_podnn_raw_for_sweep = _rom_available_basis_count(Zn_raw, Nn_raw)
    N_podI_raw_for_sweep = _rom_available_basis_count(Zi_raw, Ni_raw)

    # Install the raw bank objects globally so projection/training can access all available modes.
    # The capped N below is only the maximum basis size used for this sweep.
    Z_podnn, IP_N, mu_all_N, snaps_all_N = Zn_raw, IPn_raw, mun_raw, snn_raw
    Z_podI, IP_I, mu_all_I, snaps_all_I = Zi_raw, IPi_raw, mui_raw, sni_raw

    # These are now sweep maxima, not old single-run active values.
    NONINTRUSIVE_SWEEP_CAP = int(globals().get("NONINTRUSIVE_SWEEP_CAP", 15))
    N_podnn = _rom_cap_basis_for_sweep(N_podnn_raw_for_sweep, NONINTRUSIVE_SWEEP_CAP)
    N_podI = _rom_cap_basis_for_sweep(N_podI_raw_for_sweep, NONINTRUSIVE_SWEEP_CAP)


def run_nonintrusive_basis_sweep_error_analysis():
    print("\n" + "=" * 88)
    print("COMPREHENSIVE ERROR ANALYSIS: ALL NON-INTRUSIVE ROM METHODS".center(88))
    print("=" * 88)

    # -------------------------------------------------------------------------
    # FIXED BASIS SELECTION LOGIC
    # -------------------------------------------------------------------------
    # Use raw rb_bank_proj availability for the sweep, not the possibly capped
    # active N_podI/N_podnn left by a previous one-shot PODI/PODNN/PODAE run.
    # This makes the non-intrusive sweep behave like the intrusive study:
    #   Nmax=8  -> [1,2,4,6,8]
    #   Nmax=15 -> [1,4,8,11,15]
    # -------------------------------------------------------------------------
    _rom_prepare_projected_basis_for_sweep()

    NONINTRUSIVE_SWEEP_CAP = int(globals().get("NONINTRUSIVE_SWEEP_CAP", 15))
    NONINTRUSIVE_SWEEP_N_POINTS = int(globals().get("NONINTRUSIVE_SWEEP_N_POINTS", 5))

    nmax_podi = int(globals().get(
        "NONINTRUSIVE_SWEEP_NMAX_PODI",
        _rom_cap_basis_for_sweep(N_podI_raw_for_sweep, NONINTRUSIVE_SWEEP_CAP),
    ))
    nmax_podnn = int(globals().get(
        "NONINTRUSIVE_SWEEP_NMAX_PODNN",
        _rom_cap_basis_for_sweep(N_podnn_raw_for_sweep, NONINTRUSIVE_SWEEP_CAP),
    ))

    # Never exceed physically available modes in the projected-space banks.
    nmax_podi = int(min(nmax_podi, N_podI_raw_for_sweep))
    nmax_podnn = int(min(nmax_podnn, N_podnn_raw_for_sweep))

    # Keep global active maxima consistent for downstream sweep training.
    N_podI = nmax_podi
    N_podnn = nmax_podnn

    Error_Analysis_dir = os.path.join(Non_Intrusive_dir, "Error_Analysis")
    os.makedirs(Error_Analysis_dir, exist_ok=True)

    model_dirs = {
        "podi_rbf": os.path.join(Error_Analysis_dir, "PODI_RBF_Models"),
        "podi_linear": os.path.join(Error_Analysis_dir, "PODI_Linear_Models"),
        "podnn": os.path.join(Error_Analysis_dir, "PODNN_Models"),
        "podgpr": os.path.join(Error_Analysis_dir, "PODGPR_Models"),
        "podae": os.path.join(Error_Analysis_dir, "POD_AE_Models"),
    }
    for path in model_dirs.values():
        os.makedirs(path, exist_ok=True)

    N_basis_lists = {
        "podi": _rom_basis_sweep_list(nmax_podi, NONINTRUSIVE_SWEEP_N_POINTS),
        "podgpr": _rom_basis_sweep_list(nmax_podi, NONINTRUSIVE_SWEEP_N_POINTS),
        "podnn": _rom_basis_sweep_list(nmax_podnn, NONINTRUSIVE_SWEEP_N_POINTS),
        "podae": _rom_basis_sweep_list(nmax_podnn, NONINTRUSIVE_SWEEP_N_POINTS),
    }

    method_configs = {
        "podi_rbf": {
            "name": "PODI-RBF",
            "basis_list": N_basis_lists["podi"],
            "color": "C0",
            "marker": "s",
            "linestyle": "-",
        },
        "podi_linear": {
            "name": "PODI-Linear",
            "basis_list": N_basis_lists["podi"],
            "color": "C2",
            "marker": "D",
            "linestyle": "--",
        },
        "podgpr": {
            "name": "POD-GPR",
            "basis_list": N_basis_lists["podgpr"],
            "color": "C3",
            "marker": "o",
            "linestyle": "-.",
        },
        "podnn": {
            "name": "POD-NN",
            "basis_list": N_basis_lists["podnn"],
            "color": "C1",
            "marker": "^",
            "linestyle": ":",
        },
        "podae": {
            "name": "POD-AE-ROM",
            "basis_list": N_basis_lists["podae"],
            "color": "C4",
            "marker": "P",
            "linestyle": (0, (5, 2, 1, 2)),
        },
    }

    print("Projected-space basis source:")
    print(f"  choice_podI             : {choice_podI}")
    print(f"  raw available PODI modes : {N_podI_raw_for_sweep}")
    print(f"  sweep max PODI/PODGPR   : {nmax_podi}")
    print(f"  choice_podnn            : {choice_podnn}")
    print(f"  raw available PODNN modes: {N_podnn_raw_for_sweep}")
    print(f"  sweep max PODNN/PODAE   : {nmax_podnn}")
    print(f"  sweep cap               : {NONINTRUSIVE_SWEEP_CAP}")
    print(f"  sweep points            : {NONINTRUSIVE_SWEEP_N_POINTS}")

    print("\nBasis configurations:")
    for k, v in N_basis_lists.items():
        print(f"  {k:7s}: {v}")

    # Prepare coefficients only once per POD space.
    print("\nProjecting snapshots for basis sweep...")
    snap_funcs_I = _rom_build_snapshot_functions(snaps_all_I, prefix="snap_I")
    C_full_podi = PODInterpReducedOrderModel(
        solver,
        Z_podI,
        max(N_basis_lists["podi"]),
        IP_I,
    ).project_snapshots(snap_funcs_I)
    C_full_gpr = C_full_podi

    if (max(N_basis_lists["podnn"]) != max(N_basis_lists["podi"])) or (len(snaps_all_N) != len(snaps_all_I)):
        snap_funcs_N = _rom_build_snapshot_functions(snaps_all_N, prefix="snap_N")
        C_full_nn = PODNNReducedOrderModel(
            solver,
            Z_podnn,
            max(N_basis_lists["podnn"]),
            IP_N,
        ).project_snapshots(snap_funcs_N)
        mu_all_nn = mu_all_N
        snaps_for_nn = snaps_all_N
    else:
        C_full_nn = C_full_podi[:, : max(N_basis_lists["podnn"])]
        mu_all_nn = mu_all_I
        snaps_for_nn = snaps_all_I

    all_methods = list(method_configs.keys())
    all_basis_sizes = sorted(set(sum([cfg["basis_list"] for cfg in method_configs.values()], [])))

    method_funcs = {
        "podi_rbf": lambda rom, mu: _rom_predict_solution("podi_rbf", rom, mu),
        "podi_linear": lambda rom, mu: _rom_predict_solution("podi_linear", rom, mu),
        "podgpr": lambda rom, mu: _rom_predict_solution("podgpr", rom, mu),
        "podnn": lambda rom, mu: _rom_predict_solution("podnn", rom, mu),
        "podae": lambda rom, mu: _rom_predict_solution("podae", rom, mu),
    }

    errors = {method: defaultdict(list) for method in all_methods}
    speedup = {method: defaultdict(list) for method in all_methods}
    saved_models = {}

    print(f"\nStarting basis-sweep analysis on {len(mu_list)} parameter samples...")

    for N in all_basis_sizes:
        print("\n" + "=" * 70)
        print(f"Analyzing N_basis = {N}")
        print("=" * 70)

        roms = {}

        if N in N_basis_lists["podi"]:
            for key, method in [("podi_rbf", "rbf"), ("podi_linear", "linear")]:
                try:
                    rom = _rom_train_podi_for_sweep(method, N, C_full_podi, snapshot_targets=snaps_all_I)
                    roms[key] = rom
                    saved_models[f"{key}_N{N}"] = _rom_save_sweep_model(rom, key, N, model_dirs[key])
                    print(f"  ✓ Configured {method_configs[key]['name']} for N={N}")
                except Exception as exc:
                    print(f"  ✗ {method_configs[key]['name']} training failed for N={N}: {exc}")

        if N in N_basis_lists["podgpr"]:
            try:
                rom = _rom_train_podgpr_for_sweep(N, C_full_gpr, snapshot_targets=snaps_all_I)
                roms["podgpr"] = rom
                saved_models[f"podgpr_N{N}"] = _rom_save_sweep_model(rom, "podgpr", N, model_dirs["podgpr"])
                print(f"  ✓ Trained POD-GPR for N={N}")
            except Exception as exc:
                print(f"  ✗ POD-GPR training failed for N={N}: {exc}")

        if N in N_basis_lists["podnn"]:
            try:
                rom = _rom_train_podnn_for_sweep(N, C_full_nn, mu_all_nn, snapshot_targets=snaps_for_nn)
                roms["podnn"] = rom
                saved_models[f"podnn_N{N}"] = _rom_save_sweep_model(rom, "podnn", N, model_dirs["podnn"])
                print(f"  ✓ Trained POD-NN for N={N}")
            except Exception as exc:
                print(f"  ✗ POD-NN training failed for N={N}: {exc}")

        if N in N_basis_lists["podae"] and "PODAutoencoderROM" in globals():
            try:
                rom = _rom_train_podae_for_sweep(N)
                roms["podae"] = rom
                saved_models[f"podae_N{N}"] = _rom_save_sweep_model(rom, "podae", N, model_dirs["podae"])
                print(f"  ✓ Trained POD-AE-ROM for N={N}")
            except Exception as exc:
                print(f"  ✗ POD-AE-ROM training failed for N={N}: {exc}")

        print(f"\n  Testing {len(roms)} ROM methods...")
        for j, (mu, u_fom, t_fom) in enumerate(zip(mu_list, fom_solutions, fom_times), 1):
            if j % 10 == 0 or j <= 3:
                print(f"    • Sample {j:3d}/{len(mu_list)} - μ={np.round(mu, 4).tolist()}")

            for method_key, rom in roms.items():
                try:
                    tic = clock()
                    u_rom = method_funcs[method_key](rom, mu)
                    t_rom = clock() - tic
                    _, rel_err = compute_rom_errors(solver, u_fom, u_rom)
                    errors[method_key][N].append(rel_err)
                    speedup[method_key][N].append(t_fom / t_rom if t_rom > 0 else np.nan)
                except Exception as exc:
                    print(f"      ✗ {method_configs[method_key]['name']} failed: {str(exc)[:100]}")
                    errors[method_key][N].append(np.nan)
                    speedup[method_key][N].append(np.nan)

    print("\n" + "=" * 70)
    print("BUILDING STATISTICS".center(70))
    print("=" * 70)

    stats = {}
    for method in all_methods:
        basis_list = method_configs[method]["basis_list"]
        stats[method] = {
            "basis_sizes": basis_list,
            "err_mean": [],
            "err_min": [],
            "err_max": [],
            "spd_mean": [],
            "spd_min": [],
            "spd_max": [],
        }

        for N in basis_list:
            valid_errors = (
                np.asarray([e for e in errors[method][N] if np.isfinite(e)], dtype=float)
                if N in errors[method]
                else np.array([])
            )
            valid_speedups = (
                np.asarray([s for s in speedup[method][N] if np.isfinite(s) and s > 0.0], dtype=float)
                if N in speedup[method]
                else np.array([])
            )

            for prefix, data in [("err", valid_errors), ("spd", valid_speedups)]:
                stats[method][f"{prefix}_mean"].append(float(np.mean(data)) if data.size else np.nan)
                stats[method][f"{prefix}_min"].append(float(np.min(data)) if data.size else np.nan)
                stats[method][f"{prefix}_max"].append(float(np.max(data)) if data.size else np.nan)

        for key in ["err_mean", "err_min", "err_max", "spd_mean", "spd_min", "spd_max"]:
            stats[method][key] = np.asarray(stats[method][key], dtype=float)

    def _plot_mean_band(ax, x_vals, mean_vals, min_vals, max_vals, config):
        x_vals = np.asarray(x_vals, dtype=float)
        mean_vals = np.asarray(mean_vals, dtype=float)
        min_vals = np.asarray(min_vals, dtype=float)
        max_vals = np.asarray(max_vals, dtype=float)

        mask = (
            np.isfinite(x_vals)
            & np.isfinite(mean_vals)
            & np.isfinite(min_vals)
            & np.isfinite(max_vals)
            & (mean_vals > 0.0)
            & (max_vals > 0.0)
        )
        if not np.any(mask):
            return

        x, y, y0, y1 = x_vals[mask], mean_vals[mask], min_vals[mask], max_vals[mask]
        floor = max(float(np.nanmin(y[y > 0])) * 0.1, 1e-300)
        y0 = np.maximum(y0, floor)
        y1 = np.maximum(y1, y0)

        ax.fill_between(x, y0, y1, color=config["color"], alpha=0.16, linewidth=0.0)
        ax.plot(
            x,
            y,
            color=config["color"],
            marker=config["marker"],
            linestyle=config["linestyle"],
            linewidth=2.4,
            markersize=6.2,
            markeredgewidth=0.8,
            label=config["name"],
        )

    def _make_sweep_plot(metric_prefix, title, ylabel, filename, ylim=None):
        fig = plt.figure(figsize=(6.8, 5.2), dpi=150, facecolor="white")
        ax = fig.add_subplot(111)

        for method in all_methods:
            cfg = method_configs[method]
            _plot_mean_band(
                ax,
                cfg["basis_list"],
                stats[method][f"{metric_prefix}_mean"],
                stats[method][f"{metric_prefix}_min"],
                stats[method][f"{metric_prefix}_max"],
                cfg,
            )

        ax.set_title(title, fontsize=14, pad=8)
        ax.set_xlabel(r"$N_{\mathrm{basis}}$", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        if ylim is not None:
            ax.set_ylim(*ylim)

        _rom_style_log_axis(ax)
        _rom_style_legend(
            ax.legend(
                loc="best",
                frameon=True,
                fontsize=12,
                borderpad=0.45,
                handlelength=2.2,
                handletextpad=0.6,
                labelspacing=0.35,
            )
        )

        _rom_save_dual(fig, os.path.join(Error_Analysis_dir, filename))
        plt.show()

    _make_sweep_plot(
        "err",
        r"Relative error vs. basis size",
        r"Relative $L^2$ error",
        "comprehensive_rom_relative_error",
    )
    _make_sweep_plot(
        "spd",
        r"Speed-up vs. basis size",
        r"Speed-up factor",
        "comprehensive_rom_speedup",
        ylim=(1e0, 1e6),
    )

    with open(os.path.join(Error_Analysis_dir, "analysis_results.pkl"), "wb") as f:
        pickle.dump(
            dict(
                method_configs=method_configs,
                stats=stats,
                errors=dict(errors),
                speedup=dict(speedup),
                saved_models=saved_models,
                N_podI_raw_for_sweep=N_podI_raw_for_sweep,
                N_podnn_raw_for_sweep=N_podnn_raw_for_sweep,
                nmax_podi=nmax_podi,
                nmax_podnn=nmax_podnn,
            ),
            f,
        )

    rows = []
    print("\n" + "=" * 88)
    print("PERFORMANCE SUMMARY TABLES".center(88))
    print("=" * 88)

    for method in all_methods:
        cfg = method_configs[method]
        valid = ~np.isnan(stats[method]["err_mean"])
        if not np.any(valid):
            print(f"\n{cfg['name']}: No valid results")
            continue

        valid_basis = np.asarray(cfg["basis_list"])[valid]
        df = pd.DataFrame(
            {
                "Mean Rel Error": stats[method]["err_mean"][valid],
                "Max Rel Error": stats[method]["err_max"][valid],
                "Mean Speedup": stats[method]["spd_mean"][valid],
            },
            index=valid_basis,
        )
        df.index.name = "N_basis"

        print(f"\n{cfg['name']} Performance Summary:\n" + "=" * 50)
        print(df.to_markdown(tablefmt="github", floatfmt=".3e"))

        for N, row in df.iterrows():
            rows.append(dict(method=cfg["name"], N_basis=int(N), **row.to_dict()))

    pd.DataFrame(rows).to_csv(os.path.join(Error_Analysis_dir, "basis_sweep_summary.csv"), index=False)

    common_sizes = (
        sorted(set.intersection(*[set(cfg["basis_list"]) for cfg in method_configs.values()]))
        if method_configs
        else []
    )
    if common_sizes:
        comparison_rows = []
        for N in common_sizes:
            row = {"N_basis": N}
            for method in all_methods:
                cfg = method_configs[method]
                if N in cfg["basis_list"]:
                    j = cfg["basis_list"].index(N)
                    row[f"{cfg['name']}_err"] = stats[method]["err_mean"][j]
                    row[f"{cfg['name']}_spd"] = stats[method]["spd_mean"][j]
            comparison_rows.append(row)

        df_common = pd.DataFrame(comparison_rows).set_index("N_basis")
        df_common.to_csv(os.path.join(Error_Analysis_dir, "common_basis_comparison.csv"))

        print("\nComparison at common basis sizes:")
        print(df_common.to_markdown(tablefmt="github", floatfmt=".3e"))

    model_inventory = {
        method_configs[method_key]["name"]: {
            "basis_sizes": method_configs[method_key]["basis_list"],
            "model_dir": model_dirs[method_key],
            "saved_models": {
                f"N{N}": saved_models.get(f"{method_key}_N{N}")
                for N in method_configs[method_key]["basis_list"]
            },
        }
        for method_key in all_methods
    }

    with open(os.path.join(Error_Analysis_dir, "model_inventory.json"), "w") as f:
        json.dump(model_inventory, f, indent=2)

    print("\n" + "=" * 88)
    print("COMPREHENSIVE ERROR ANALYSIS COMPLETE".center(88))
    print("=" * 88)
    print(f"Results saved to: {Error_Analysis_dir}")
    print(f"Analysis results: {os.path.join(Error_Analysis_dir, 'analysis_results.pkl')}")
    print(f"Model inventory: {os.path.join(Error_Analysis_dir, 'model_inventory.json')}")
    print("=" * 88)

    globals().update(
        Error_Analysis_dir=Error_Analysis_dir,
        rom_error_analysis_stats=stats,
        rom_error_analysis_configs=method_configs,
        rom_error_analysis_saved_models=saved_models,
        rom_error_analysis_results_path=os.path.join(Error_Analysis_dir, "analysis_results.pkl"),
        N_basis_lists_nonintrusive=N_basis_lists,
        N_podI_raw_for_sweep=N_podI_raw_for_sweep,
        N_podnn_raw_for_sweep=N_podnn_raw_for_sweep,
        nmax_podi_sweep=nmax_podi,
        nmax_podnn_sweep=nmax_podnn,
    )

    return stats, method_configs, saved_models


if globals().get("RUN_NONINTRUSIVE_ERROR_ANALYSIS", True):
    rom_error_analysis_stats, rom_error_analysis_configs, rom_error_analysis_saved_models = run_nonintrusive_basis_sweep_error_analysis()
else:
    print("\n[Info] Skipping basis-sweep Error Analysis. Set RUN_NONINTRUSIVE_ERROR_ANALYSIS=True to run it.")

# %% [markdown] Cell 94 | id: 32ef635f
# ## Combined ROM Performance Study

# %% Cell 95 | id: 4a5f2510

# =============================================================================
# ## Combined ROM Performance Study
# =============================================================================
def _p2_int_or_none(x):
    return None if x is None else int(x)


def _p2_can_run_coupled_podg():
    is_monolithic = (
        int(getattr(solver, "N_subdomains", 1)) == 1
        and str(globals().get("P2_TARGET_PANEL", "monolithic")).lower() == "monolithic"
    )
    required = ["_p2_decode_active_sample", "P2_CONFIG", "N_podg", "Z_podg"]
    missing = [name for name in required if name not in globals()]
    return bool(P2_ENABLE_COUPLED_PODG) and is_monolithic and not missing, is_monolithic, missing


def _rom_prepare_solution_pod_basis_if_available():
    # Optional intrusive solution-space POD basis. This is only needed if coupled PODG/POD projection
    # in Solution space is requested.
    global choice_podg, choice_podlspg, N_podg, Z_podg, IP_G, mu_all_G, snaps_all_G
    global N_podlspg, Z_podlspg, IP_L, mu_all_L, snaps_all_L
    if "rb_bank_Solution" not in globals():
        return False
    try:
        choice_podg = globals().get("choice_podg", 0)
        choice_podlspg = globals().get("choice_podlspg", 0)
        eigG, eigvG, basisG, NG_raw, ZG_raw, IPG_raw, muG_raw, snapsG_raw = rb_bank_Solution[choice_podg]
        eigL, eigvL, basisL, NL_raw, ZL_raw, IPL_raw, muL_raw, snapsL_raw = rb_bank_Solution[choice_podlspg]
        N_podg = globals().get("N_podg", NG_raw)
        N_podlspg = globals().get("N_podlspg", NL_raw)
        Z_podg, IP_G, mu_all_G, snaps_all_G = ZG_raw, IPG_raw, muG_raw, snapsG_raw
        Z_podlspg, IP_L, mu_all_L, snaps_all_L = ZL_raw, IPL_raw, muL_raw, snapsL_raw
        return True
    except Exception as exc:
        print(f"[Combined] Could not prepare solution POD basis: {exc}")
        return False


def plot_comparison_grid(solver, mu, sample_num, output_dir, models_data):
    if not models_data:
        print(f"  Skipping plot for sample {sample_num}: No model data provided.")
        return
    plt.rcParams["figure.autolayout"] = False
    V_CG = FunctionSpace(solver.mesh, "CG", 2)
    to_global = _rom_project_to_global_factory(solver, V_CG)

    solutions, errors, names = [], [], []
    for name, data in models_data.items():
        u_rom_global = to_global(data["u_rom"])
        u_fom_global = to_global(data["u_fom"])
        err_func = Function(V_CG)
        _rom_set_function_vector(err_func, np.abs(u_fom_global.vector().get_local() - u_rom_global.vector().get_local()))
        solutions.append(u_rom_global)
        errors.append(err_func)
        names.append(name)

    nx_plot, ny_plot = 260, 150
    xg, yg = np.linspace(0.0, solver.length, nx_plot), np.linspace(0.0, solver.width, ny_plot)
    Xg, Yg = np.meshgrid(xg, yg)
    sol_grids = [_rom_field_on_grid(sol, Xg, Yg) for sol in solutions]
    err_grids = [_rom_field_on_grid(err, Xg, Yg) for err in errors]
    sol_min = min(float(np.nanmin(Z)) for Z in sol_grids)
    sol_max = max(float(np.nanmax(Z)) for Z in sol_grids)

    fig = plt.figure(figsize=(12.0, max(3.6 * len(solutions), 4.2)), dpi=150, facecolor="white")
    gs = GridSpec(len(solutions), 2, figure=fig, left=0.070, right=0.985, bottom=0.055,
                  top=0.930, wspace=0.28, hspace=0.46)
    for row, (Z_sol, Z_err, name) in enumerate(zip(sol_grids, err_grids, names)):
        err_max = float(np.nanmax(Z_err))
        err_max = 1e-300 if (not np.isfinite(err_max) or err_max <= 0.0) else err_max
        _rom_draw_scalar(fig, fig.add_subplot(gs[row, 0]), solver, Xg, Yg, Z_sol,
                         rf"{name} ROM solution ($w$ $[m]$)", vmin=sol_min, vmax=sol_max, cmap="viridis")
        _rom_draw_scalar(fig, fig.add_subplot(gs[row, 1]), solver, Xg, Yg, Z_err,
                         rf"{name} absolute error ($[m]$)", vmin=0.0, vmax=err_max,
                         cmap="plasma", positive_only=True, ylabel=False)
    os.makedirs(output_dir, exist_ok=True)
    _rom_save_dual(fig, os.path.join(output_dir, f"comparison_grid_sample_{sample_num}"), pad_inches=0.03)
    plt.show()


def plot_requested_comparison_grids(solver, plot_payloads, output_dir):
    if not isinstance(plot_payloads, dict):
        print("Skipping detailed comparison grids: plot_payloads is not a dictionary.")
        return
    for sample_num, payload in plot_payloads.items():
        plot_comparison_grid(solver, payload["mu"], sample_num, output_dir, payload["models_data"])


def _combined_available_models(load_index=ROM_LOAD_INDEX):
    models, _ = load_final_nonintrusive_models(load_index=load_index, verbose=False)
    model_map = {}
    if "podi_rbf" in models:    model_map["PODI-RBF"] = ("podi_rbf", models["podi_rbf"])
    if "podi_linear" in models: model_map["PODI-Linear"] = ("podi_linear", models["podi_linear"])
    if "podgpr" in models:      model_map["PODGPR"] = ("podgpr", models["podgpr"])
    if "podnn" in models:       model_map["PODNN"] = ("podnn", models["podnn"])
    if "podae" in models:       model_map["POD-AE-ROM"] = ("podae", models["podae"])
    return model_map


def run_combined_rom_analysis(solver, models_to_run, indices_to_plot=None, load_index=ROM_LOAD_INDEX):
    indices_to_plot = indices_to_plot or []
    _rom_prepare_solution_pod_basis_if_available()
    can_run_podg, is_monolithic, podg_missing = _p2_can_run_coupled_podg()

    if P2_ENABLE_COUPLED_PODG and not can_run_podg:
        print(f"[Combined] Coupled PODG requested but unavailable. monolithic={is_monolithic}, missing={podg_missing}")

    model_map = _combined_available_models(load_index=load_index)

    # Sort by active Project-2 parameter norm. This keeps the previous convention.
    sorted_indices = sorted(range(len(mu_list)), key=lambda i: np.linalg.norm(np.asarray(mu_list[i], dtype=float)))
    print("\n" + "=" * 88)
    print("COMBINED ROM PERFORMANCE STUDY".center(88))
    print("=" * 88)
    print(f"P2_ENABLE_COUPLED_PODG : {P2_ENABLE_COUPLED_PODG}")
    print(f"Coupled PODG available : {can_run_podg}")
    print(f"Monolithic case        : {is_monolithic}")
    print(f"Prepared test points   : {len(sorted_indices)}")
    print(f"Models requested       : {models_to_run}")
    print("=" * 88)

    results, plot_payloads = [], {}

    for rank_i, original_idx in enumerate(sorted_indices, 1):
        mu_cg = np.asarray(mu_list[original_idx], dtype=float)
        u_fom_cg = fom_solutions[original_idx]
        t_fom = float(fom_times[original_idx])
        record = {"sample_rank": rank_i, "sample_index": original_idx, "mu": mu_cg, "fom_time": t_fom}
        models_data_for_plot = {}

        print(f"➤ Processing Sample {rank_i}/{len(sorted_indices)}: idx={original_idx}, μ={np.round(mu_cg, 4).tolist()}")

        # Projection floor in projected/non-intrusive POD space.
        if "POD-Proj" in models_to_run and "PODI-RBF" in model_map:
            try:
                _, podproj_rom = model_map["PODI-RBF"]
                t0 = clock()
                c_proj = _rom_pod_project_coeffs(u_fom_cg, podproj_rom)
                u_proj = podproj_rom.reconstruct_solution(c_proj)
                t_proj = clock() - t0
                rel_err = compute_rom_errors(solver, u_fom_cg, u_proj)[1]
                record.update(podproj_rel_err=rel_err, podproj_time=t_proj, podproj_speedup=t_fom / t_proj if t_proj > 0 else np.nan)
                models_data_for_plot["POD-Proj"] = {"u_rom": u_proj, "u_fom": u_fom_cg}
            except Exception as exc:
                print(f"    POD-Proj skipped/failed for sample {original_idx}: {exc}")

        # Optional coupled Project-2 POD-Galerkin. Disabled by default.
        if "Coupled POD-Galerkin" in models_to_run or "PODG" in models_to_run:
            if not P2_ENABLE_COUPLED_PODG:
                if rank_i == 1:
                    print("    Coupled POD-Galerkin skipped because P2_ENABLE_COUPLED_PODG=False.")
            elif not can_run_podg:
                if rank_i == 1:
                    print(f"    Coupled POD-Galerkin unavailable. monolithic={is_monolithic}, missing={podg_missing}")
            else:
                try:
                    # Use active Project-2 mu and decode to mechanical/thermal pieces.
                    mu_mech, mu_th, podg_meta = _p2_decode_active_sample(P2_CONFIG, mu_cg)
                    solver.set_rom_thermal_parameters(**mu_th)
                    t0 = clock()
                    rbpg, u_podg, podg_info = solver.online_PODG_solver(
                        mu_mech,
                        N_podg,
                        Z_podg,
                        thermal_on=True,
                        coupled_on=True,
                        heat_nx=_p2_int_or_none(globals().get("P2_HEAT_NX", None)),
                        heat_ny=_p2_int_or_none(globals().get("P2_HEAT_NY", None)),
                        heat_nz=int(globals().get("P2_HEAT_NZ", 8)),
                        heat_degree=int(globals().get("P2_HEAT_DEGREE", 1)),
                        Nz_quad_T1=int(globals().get("P2_THETA_QUADRATURE", 20)),
                        T1_cg_degree=int(globals().get("P2_THETA_CG_DEGREE", 1)),
                        coupling_omega=float(globals().get("P2_COUPLING_OMEGA", 0.7)),
                        coupling_tol_w=float(globals().get("P2_COUPLING_TOL_W", 1e-5)),
                        coupling_tol_T1=float(globals().get("P2_COUPLING_TOL_THETA", 1e-5)),
                        coupling_max_iters=int(globals().get("P2_COUPLING_MAX_ITERS", 25)),
                        coupling_verbose=False,
                        return_info=True,
                    )
                    t_podg = clock() - t0
                    rel_err = compute_rom_errors(solver, u_fom_cg, u_podg)[1]
                    record.update(
                        podg_rel_err=rel_err,
                        podg_time=t_podg,
                        podg_speedup=t_fom / t_podg if t_podg > 0 else np.nan,
                        podg_converged=podg_info.get("converged", None) if isinstance(podg_info, dict) else None,
                        podg_iters=podg_info.get("iters", None) if isinstance(podg_info, dict) else None,
                    )
                    models_data_for_plot["Coupled POD-Galerkin"] = {"u_rom": u_podg, "u_fom": u_fom_cg}
                except Exception as exc:
                    print(f"    Coupled POD-Galerkin failed/skipped for sample {original_idx}: {exc}")

        for public_name, (method_key, rom) in model_map.items():
            if public_name not in models_to_run:
                continue
            try:
                t0 = clock()
                u_rom = _rom_predict_solution(method_key, rom, mu_cg)
                t_rom = clock() - t0
                rel_err = compute_rom_errors(solver, u_fom_cg, u_rom)[1]
                col = {
                    "PODI-RBF": "podi_rbf",
                    "PODI-Linear": "podi_linear",
                    "PODGPR": "podgpr",
                    "PODNN": "podnn",
                    "POD-AE-ROM": "podae",
                }[public_name]
                record.update({f"{col}_rel_err": rel_err, f"{col}_time": t_rom, f"{col}_speedup": t_fom / t_rom if t_rom > 0 else np.nan})
                models_data_for_plot[public_name] = {"u_rom": u_rom, "u_fom": u_fom_cg}
            except Exception as exc:
                print(f"    {public_name} skipped/failed for sample {original_idx}: {exc}")

        results.append(record)
        if rank_i in indices_to_plot or original_idx in indices_to_plot:
            plot_payloads[rank_i] = {"mu": mu_cg, "models_data": models_data_for_plot}

    df_results = pd.DataFrame(results)

    print("\n" + "=" * 88)
    print("PERFORMANCE SUMMARY".center(88))
    print("=" * 88)
    column_labels = {
        "podproj_rel_err": "POD-Proj Err", "podproj_speedup": "POD-Proj Speedup",
        "podg_rel_err": "PODG Err", "podg_speedup": "PODG Speedup",
        "podi_rbf_rel_err": "PODI-RBF Err", "podi_rbf_speedup": "PODI-RBF Speedup",
        "podi_linear_rel_err": "PODI-Linear Err", "podi_linear_speedup": "PODI-Linear Speedup",
        "podgpr_rel_err": "PODGPR Err", "podgpr_speedup": "PODGPR Speedup",
        "podnn_rel_err": "PODNN Err", "podnn_speedup": "PODNN Speedup",
        "podae_rel_err": "POD-AE Err", "podae_speedup": "POD-AE Speedup",
    }
    cols = [c for c in column_labels if c in df_results.columns]
    if cols:
        pretty = df_results[cols].rename(columns={k: column_labels[k] for k in cols})
        fmt = {c: ("{:.2e}".format if "Err" in c else "{:.2f}x".format) for c in pretty.columns}
        print(pretty.to_string(formatters=fmt))
    else:
        print("No model results available.")
    print("=" * 88)
    return df_results, plot_payloads


def plot_rom_performance(df_results, models_to_run, rom_dir, save_name="rom_performance_analysis"):
    if df_results is None or df_results.empty:
        print("No performance data available for plotting.")
        return

    plt.rcParams["figure.autolayout"] = False
    sample_indices = np.arange(1, len(df_results) + 1)

    plot_config = {
        "Coupled POD-Galerkin": {"marker": "o", "ls": "-",  "color": "C0", "err": "podg_rel_err",       "spd": "podg_speedup"},
        "PODG":                 {"marker": "o", "ls": "-",  "color": "C0", "err": "podg_rel_err",       "spd": "podg_speedup"},
        "POD-Proj":             {"marker": "h", "ls": "-",  "color": "C6", "err": "podproj_rel_err",    "spd": "podproj_speedup"},
        "PODI-RBF":             {"marker": "s", "ls": "--", "color": "C1", "err": "podi_rbf_rel_err",   "spd": "podi_rbf_speedup"},
        "PODI-Linear":          {"marker": "D", "ls": "-.", "color": "C3", "err": "podi_linear_rel_err","spd": "podi_linear_speedup"},
        "PODGPR":               {"marker": "*", "ls": "-",  "color": "C7", "err": "podgpr_rel_err",     "spd": "podgpr_speedup"},
        "PODNN":                {"marker": "^", "ls": ":",  "color": "C2", "err": "podnn_rel_err",      "spd": "podnn_speedup"},
        "POD-AE-ROM":           {"marker": "P", "ls": (0, (5, 2, 1, 2)), "color": "C4", "err": "podae_rel_err", "spd": "podae_speedup"},
    }

    active = {
        name: cfg for name, cfg in plot_config.items()
        if name in models_to_run and cfg["err"] in df_results.columns
    }
    if not active:
        print("No active methods available for performance plotting.")
        return

    # Wider canvas and reserved top space keep the legend in a single clean row.
    # This fixes the previous wrapping caused by ncol=min(len(labels), 5).
    fig = plt.figure(figsize=(15.8, 5.7), dpi=150, facecolor="white")
    gs = GridSpec(
        1, 2, figure=fig,
        left=0.070, right=0.985,
        bottom=0.155, top=0.780,
        wspace=0.20,
    )
    ax_err, ax_spd = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    for name, cfg in active.items():
        style = dict(
            label=name,
            color=cfg["color"],
            marker=cfg["marker"],
            linestyle=cfg["ls"],
            linewidth=2.35,
            markersize=6.0,
            markevery=max(1, len(df_results) // 25),
            markerfacecolor="none",
            markeredgewidth=0.9,
        )

        err = np.asarray(df_results[cfg["err"]], dtype=float)
        spd = (
            np.asarray(df_results[cfg["spd"]], dtype=float)
            if cfg["spd"] in df_results.columns
            else np.full_like(err, np.nan)
        )

        valid_err = np.isfinite(err) & (err > 0.0)
        valid_spd = np.isfinite(spd) & (spd > 0.0)

        if np.any(valid_err):
            ax_err.semilogy(sample_indices[valid_err], err[valid_err], **style)
        if np.any(valid_spd):
            ax_spd.semilogy(sample_indices[valid_spd], spd[valid_spd], **style)

    ax_err.set_title(r"ROM accuracy comparison", fontsize=14, pad=8)
    ax_err.set_xlabel(r"Sample index (sorted by $||\mu||_2$)", fontsize=14)
    ax_err.set_ylabel(r"Relative $L^2$ error", fontsize=14)
    ax_err.set_ylim(None, 1e0)

    ax_spd.set_title(r"ROM speed-up comparison", fontsize=14, pad=8)
    ax_spd.set_xlabel(r"Sample index (sorted by $||\mu||_2$)", fontsize=14)
    ax_spd.set_ylabel(r"Speed-up factor $(t_{\mathrm{FOM}}/t_{\mathrm{ROM}})$", fontsize=14)
    ax_spd.set_ylim(1e0, 1e6)

    for ax in (ax_err, ax_spd):
        _rom_style_log_axis(ax)

    handles, labels = ax_err.get_legend_handles_labels()
    legend_ncol = max(1, len(labels))
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=legend_ncol,
        frameon=True,
        fontsize=11.5,
        borderpad=0.42,
        handlelength=1.75,
        handletextpad=0.45,
        columnspacing=0.85,
        labelspacing=0.25,
    )
    _rom_style_legend(legend)

    os.makedirs(rom_dir, exist_ok=True)
    _rom_save_dual(fig, os.path.join(rom_dir, save_name), pad_inches=0.03)
    plt.show()


# Execution configuration for combined study.
# Default: non-intrusive methods + POD projection only.
# Coupled POD-Galerkin can be activated by setting P2_ENABLE_COUPLED_PODG=True and adding
# "Coupled POD-Galerkin" to COMBINED_MODELS_TO_RUN.
COMBINED_MODELS_TO_RUN = globals().get(
    "COMBINED_MODELS_TO_RUN",
    ["POD-Proj", "PODI-RBF", "PODI-Linear", "PODGPR", "PODNN", "POD-AE-ROM"],
)
if P2_ENABLE_COUPLED_PODG and "Coupled POD-Galerkin" not in COMBINED_MODELS_TO_RUN:
    COMBINED_MODELS_TO_RUN = ["Coupled POD-Galerkin"] + list(COMBINED_MODELS_TO_RUN)

# Folder uses the active loaded PODI-RBF basis size if available.
try:
    _tmp_models, _ = load_final_nonintrusive_models(load_index=ROM_LOAD_INDEX, verbose=False)
    _basis_for_folder = int(getattr(_tmp_models.get("podi_rbf"), "n_basis", globals().get("N_podI", 0)))
except Exception:
    _basis_for_folder = int(globals().get("N_podI", 0))

rom_dir = os.path.join(solver.output_dir, f"ROM_Performance_N_{_basis_for_folder}")
os.makedirs(rom_dir, exist_ok=True)
analysis_cache_pkl = os.path.join(rom_dir, "rom_performance_results.pkl")
analysis_cache_csv = os.path.join(rom_dir, "rom_performance_results.csv")

if ROM_RERUN_ANALYSIS or not os.path.exists(analysis_cache_pkl):
    df_results, plot_payloads = run_combined_rom_analysis(
        solver,
        COMBINED_MODELS_TO_RUN,
        indices_to_plot=globals().get("COMBINED_INDICES_TO_PLOT", ROM_TEST_INDICES),
        load_index=ROM_LOAD_INDEX,
    )
    df_results.to_pickle(analysis_cache_pkl)
    df_results.to_csv(analysis_cache_csv, index=False)
else:
    df_results = pd.read_pickle(analysis_cache_pkl)
    plot_payloads = {}
    print(f"Loaded cached ROM performance results from: {analysis_cache_pkl}")

plot_rom_performance(df_results, COMBINED_MODELS_TO_RUN, rom_dir, save_name="rom_performance_analysis")

if isinstance(plot_payloads, dict) and plot_payloads:
    plot_requested_comparison_grids(solver, plot_payloads, rom_dir)

print("\n" + "=" * 88)
print("COMBINED ROM PERFORMANCE STUDY COMPLETE".center(88))
print("=" * 88)
print(f"P2_ENABLE_COUPLED_PODG : {P2_ENABLE_COUPLED_PODG}")
print(f"Models run             : {COMBINED_MODELS_TO_RUN}")
print(f"Results folder         : {rom_dir}")
print(f"Results pickle         : {analysis_cache_pkl}")
print(f"Results csv            : {analysis_cache_csv}")
print("=" * 88)

# %% Cell 96 | id: 0611912e
