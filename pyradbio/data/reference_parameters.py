"""Cited reference parameter sets used by golden tests and as sane defaults.

Each entry MUST cite its source. Values transcribed from the primary papers.
"""

from __future__ import annotations

# Inaniwa et al. 2010 (doi:10.1088/0031-9155/55/22/008), aerobic HSG, mMKM.
HSG_mMKM = {
    "alpha0_Gy": 0.172,  # Sec. 2.2 (fit to Furusawa 2000 D10)
    "beta_Gy2": 0.0615,  # Sec. 2.2 (Kase 2008)
    "r_d_um": 0.32,  # domain radius
    "R_n_um": 3.9,  # nucleus radius
    "alpha0_current_Gy": 0.0708,  # Sec. 2.5, current HSG (alpha=0.282, z*=3.43)
}

# Elsasser & Scholz 2007 (doi:10.1667/RR0467.1) photon inputs.
LEM2_V79 = {"alpha_x_Gy": 0.093, "beta_x_Gy2": 0.026, "Dt_old_Gy": 30.0, "Dt_new_Gy": 7.5}
LEM2_CHO = {"alpha_x_Gy": 0.228, "beta_x_Gy2": 0.020, "Dt_old_Gy": 40.0, "Dt_new_Gy": 9.5}
# HSG (Fig. 11): alpha_x=0.313, beta_x=0.0615, Dt_old=30, Dt_new=6.0.
LEM2_HSG = {"alpha_x_Gy": 0.313, "beta_x_Gy2": 0.0615, "Dt_old_Gy": 30.0, "Dt_new_Gy": 6.0}

# Friedrich et al. 2012 (doi:10.3109/09553002.2011.611213) V79 carbon (Fig. 3).
LEM4_V79 = {"alpha_x_Gy": 0.184, "beta_x_Gy2": 0.02, "Dt_Gy": 17.0, "R_n_um": 4.7}
