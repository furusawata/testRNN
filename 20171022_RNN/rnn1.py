# -*- coding: utf-8 -*-
"""
Spyderエディタ

これは一時的なスクリプトファイルです
"""

from math import *
import numpy as np

def fu(x):
    return tanh(x)

def ru(x):
    return 1 - (tanh(x))**2

def fv(x):
    return 1/(1+exp(-x))

def rv(x):
    return x * (1 - x)

if __name__ == "__main__":
    l = 1000

    x = np.arange(l+1)
    x = np.sin(x / 100 * pi / 180)

    Win = np.zeros(l)
    W = np.zeros(l)
    Wout = np.zeros(l)
    b = np.zeros(l)
    c = np.zeros(l)
    u = np.zeros(l)
    z = np.zeros(l)
    v = np.zeros(l)
    y = np.zeros(l)
    E = np.zeros(l)

    Win_init = 0.1
    W_init = 0.2
    Wout_init = 0.3

    dlout = np.zeros(l)
    dl = np.zeros(l+1)
    dlWout = np.zeros(l)
    dlW = np.zeros(l)
    dlWin = np.zeros(l)
    
    eps = 0.01

#------------------------------------------------------------------------------
    for s in range(l-1):
        for t in range(s+1):
            if t != s:
                if t == 0:
                    u[t] = Win[t] * x[t] + b[t]
                else:
                    u[t] = Win[t] * x[t] + W[t] * z[t-1] + b[t]
                z[t] = fu(u[t])
            else:
                if t == 0:
                    Win[t] = Win_init
                    W[t] = W_init
                    Wout[t] = Wout_init
                    u[t] = Win[t] * x[t] + b[t]
                else:
#                    Win[t] = Win[t-1]
#                    W[t] = W[t-1]
#                    Wout[t] = Wout[t-1]
                    u[t] = Win[t] * x[t] + W[t] * z[t-1] + b[t]
    
                z[t] = fu(u[t])
                v[t] = Wout[t] * z[t] + c[t]
                y[t] = fv(v[t])
                E[t] = 0.5 * (y[t] - x[t+1])**2
            
    
        dlT = 0
        dloutT = 0
        dlWinT = 0
        dlWT = 0
        dlWoutT = 0
        for t in range(s, -1, -1):
            dlout[t] = (y[t] - x[t+1]) * rv(y[t])
            dloutT = dloutT + dlout[t]

            dlWoutT = dlWoutT + dlout[t] * z[t]

            dl[t] = (dlout[t] * Wout[t] + dl[t+1] * W[t]) * ru(u[t])
            dlT = dlT + dl[t]

            if t != 0:
                dlWT = dlWT + dl[t] * z[t-1]
            dlWinT = dlWinT + dl[t] * x[t]
            
        Win[s+1] = Win[s] - eps * dlWinT
        W[s+1] = W[s] - eps * dlWT
        Wout[s+1] = Wout[s] - eps * dlWoutT
        b[s+1] = b[s] - eps * dlT
        c[s+1] = c[s] - eps * dloutT
    