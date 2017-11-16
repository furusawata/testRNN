# -*- coding: utf-8 -*-
"""
Spyderエディタ

これは一時的なスクリプトファイルです
"""

import math
import numpy as np
import matplotlib.pyplot as plt

Nd = 10000
Nx = 17

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

    fp = open('rnn.txt', 'w')
    for j in range(100):
        for i in range(Nd):
            s = np.random.rand() * 2 * np.pi
            t = np.linspace(0, s+2*np.pi, Nx)
            x = np.sin(t)
            train(x)
#            print(Win)
        fp.write(str(E) + ', ' + str(Win) +  ', ' + str(W) +  ', ' + str(Wout) + '\n')  
#        print(j)

    fp.close()
    t = np.linspace(0, 2*np.pi * 10, Nx * 10)
    x = np.sin(t)
    y = forward(x)
    p = prediction(0, Nx * 10)
#    plt.plot(t, x)
#    plt.plot(t, y)
#    plt.plot(t, p)
#    plt.show
    
    print(str(E) + ', ' + str(Win) +  ', ' + str(W) +  ', ' + str(Wout) + '\n')  
    
