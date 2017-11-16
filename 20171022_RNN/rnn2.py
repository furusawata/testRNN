# -*- coding: utf-8 -*-
"""
Spyderエディタ

これは一時的なスクリプトファイルです
"""

import math
import numpy as np
import matplotlib.pyplot as plt

Nd = 10000
Nx = 7

Win = np.random.rand() * 2 - 1
W = np.random.rand() * 2 - 1
Wout = np.random.rand() * 2 - 1
b = 0
c = 0
E = 0

eps = 0.001

def fv(x):
    return math.tanh(x)

def rv(x):
    return 1 - x**2

def fu(x):
    return 1/(1+math.exp(-x))

def ru(x):
    return x * (1 - x)

def forward(x):
    global Win, W, Wout, b, c, E
    N = x.size
    z = np.zeros(N)
    y = np.zeros(N)
    for t in range(N-1):
        if t == 0:
            u = x[t] * Win + b
        else:
            u = x[t] * Win + b + z[t-1] * W
        z[t] = fu(u)
        v = z[t] * Wout + c
        y[t] = fv(v)
    return y

def prediction(x, N):
    global Win, W, Wout, b, c, E
    z = np.zeros(N)
    y = np.zeros(N)
    for t in range(N-1):
        if t == 0:
            u = x * Win + b
        else:
            u = y[t-1] * Win + b + z[t-1] * W
        z[t] = fu(u)
        v = z[t] * Wout + c
        y[t] = fv(v)
    return y
    
def train(x):
    global Win, W, Wout, b, c, E
    N = x.size
    u = np.zeros(N)
    z = np.zeros(N)
    y = np.zeros(N)
    for t in range(N-1):
        if t == 0:
            u[t] = x[t] * Win + b
        else:
            u[t] = x[t] * Win + b + z[t-1] * W
        z[t] = fu(u[t])
        v = z[t] * Wout + c
        y[t] = fv(v)

    dloutT = 0
    dlWoutT = 0
    dlT = 0
    dlWT = 0
    dlWinT = 0
    dl = 0
    E = 0
    for t in range(N-2, -1, -1):
        E = E + y[t] - x[t+1]
        dlout = (y[t] - x[t+1]) * rv(y[t])
        dloutT = dloutT + dlout

        dlWoutT = dlWoutT + dlout * z[t]

        dl = (dlout * Wout + dl * W) * ru(z[t])
        dlT = dlT + dl

        if t != 0:
            dlWT = dlWT + dl * z[t-1]
        dlWinT = dlWinT + dl * x[t]
        
    Win = Win - eps * dlWinT
    W = W - eps * dlWT
    Wout = Wout - eps * dlWoutT
    b = b - eps * dlT
    c = c - eps * dloutT
    

if __name__ == "__main__":

    for j in range(30):
        for i in range(Nd):
            s = np.random.rand() * 2 * np.pi
            t = np.linspace(0, s+2*np.pi, Nx)
            x = np.sin(t)
            train(x)
#            print(Win)
        print(j)

    t = np.linspace(0, 2*np.pi * 10, Nx * 10)
    x = np.sin(t)
    y = forward(x)
    p = prediction(0, Nx * 10)
    plt.plot(t, x)
    plt.plot(t, y)
    plt.plot(t, p)
    plt.show
    
    

#    Win = np.zeros(l)
#    W = np.zeros(l)
#    Wout = np.zeros(l)
#    b = np.zeros(l)
#    c = np.zeros(l)
#    u = np.zeros(l)
#    z = np.zeros(l)
#    v = np.zeros(l)
#    y = np.zeros(l)
#    E = np.zeros(l)
#
#    Win_init = 0.1
#    W_init = 0.2
#    Wout_init = 0.3
#
#    dlout = np.zeros(l)
#    dl = np.zeros(l+1)
#    dlWout = np.zeros(l)
#    dlW = np.zeros(l)
#    dlWin = np.zeros(l)
#    
#    eps = 0.01
#
#    fp = open('rnn.txt', 'w')
##------------------------------------------------------------------------------
#    for s in range(l-1):
#
#        t = s
#        if t == 0:
#            Win[t] = Win_init
#            W[t] = W_init
#            Wout[t] = Wout_init
#            u[t] = Win[t] * x[t] + b[t]
#        else:
#            u[t] = Win[t] * x[t] + W[t] * z[t-1] + b[t]
#
#        z[t] = fu(u[t])
#        v[t] = Wout[t] * z[t] + c[t]
#        y[t] = fv(v[t])
#        E[t] = 0.5 * (y[t] - x[t+1])**2
#    
#        dlT = 0
#        dloutT = 0
#        dlWinT = 0
#        dlWT = 0
#        dlWoutT = 0
#        for t in range(s, -1, -1):
#            dlout[t] = (y[t] - x[t+1]) * rv(y[t])
#            dloutT = dloutT + dlout[t]
#
#            dlWoutT = dlWoutT + dlout[t] * z[t]
#
#            dl[t] = (dlout[t] * Wout[t] + dl[t+1] * W[t]) * ru(u[t])
#            dlT = dlT + dl[t]
#
#            if t != 0:
#                dlWT = dlWT + dl[t] * z[t-1]
#            dlWinT = dlWinT + dl[t] * x[t]
#            
#        Win[s+1] = Win[s] - eps * dlWinT
#        W[s+1] = W[s] - eps * dlWT
#        Wout[s+1] = Wout[s] - eps * dlWoutT
#        b[s+1] = b[s] - eps * dlT
#        c[s+1] = c[s] - eps * dloutT
#
#        fp.write(str(Win[s]) + ',' + str(W[s]) + ',' + str(Wout[s]) + '\n')  
#    
#    fp.close()
#    