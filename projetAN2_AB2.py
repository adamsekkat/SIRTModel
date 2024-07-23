import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib.widgets import Button, Slider

# ----------------------------------- Realisé par Sekkat Adam , Sanchez Matthew , Roess Célia # ----------------------------------- 
def phiS(beta,delta,N,S,I,T):
    return (-beta/N ) * ( I + delta*T) * S

def phiI(alpha,beta,delta,gamma,N,S,I,T):
    return (beta/N ) * S * ( I + delta*T) - (alpha + gamma) * I

def phiT(alpha,eta,I,T):
    return alpha*I - eta*T

def phiR(gamma,eta,I,T):
    return gamma*I + eta*T

# Conditions initiales
init_beta = 1.3
init_delta = 0.2   
init_gamma = 0.1   
init_alpha = 0.1 
init_eta = 0.1  
init_S0 = 92
init_I0 = 2
init_T0 = 0
init_R0 = 0
N = init_S0 + init_I0 + init_T0 + init_R0
h = 0.2
eps = 1e-8
Nmax = 100
RR0 = (init_beta)/(init_alpha + init_gamma) + (init_alpha*init_delta*init_beta)/((init_alpha + init_gamma)*init_eta)

# Fonction qui resoud le systeme suivant Adam Bashforth 2
def Adam_Bashforth2(alpha,beta,delta,eta,gamma,N,S0,I0,T0,h,eps,Nmax):
    R0 = 0
    t = np.arange(0, (Nmax))
    #Calcul de S1,I1,T1,R1 grace à Euler Explicite 
    S1 = S0 +h*phiS(beta,delta,N,S0,I0,T0)
    I1 = I0 +h*phiI(alpha,beta,delta,gamma,N,S0,I0,T0)
    T1 = T0 +h*phiT(alpha,eta,I0,T0)
    R1 = R0 +h*phiR(gamma,eta,I0,T0)
    S = [S0,S1]
    I = [I0,I1]
    T = [T0,T1]
    R = [R0,R1]
    i = 1
    while i < Nmax-1 :
        S.append(S[i]+(h/2)*(3*phiS(beta,delta,N,S[i],I[i],T[i]) - phiS(beta,delta,N,S[i-1],I[i-1],T[i-1])))
        I.append(I[i]+(h/2)*(3*phiI(alpha,beta,delta,gamma,N,S[i],I[i],T[i])-phiI(alpha,beta,delta,gamma,N,S[i-1],I[i-1],T[i-1])))
        T.append(T[i]+(h/2)*(3*phiT(alpha,eta,I[i],T[i])-phiT(alpha,eta,I[i-1],T[i-1])))
        R.append(R[i]+(h/2)*(3*phiR(gamma,eta,I[i],T[i])-phiR(gamma,eta,I[i-1],T[i-1])))
        i = i+1
    return t,S,I,T,R



fig, ax = plt.subplots()
t,S,I,T,R = Adam_Bashforth2(init_alpha,init_beta,init_delta,init_eta,init_gamma,N,init_S0,init_I0,init_T0,h,eps,Nmax)
line_S, = ax.plot(t,S,label="S", lw=2)
line_I, = ax.plot(t,I,label="I", lw=2)
line_T, = ax.plot(t,T,label="T", lw=2)
line_R, = ax.plot(t,R,label="R", lw=2)


fig.subplots_adjust(left=0.15, bottom=0.55)

ax_S0 = fig.add_axes([0.1, 0.1, 0.35, 0.03])
S0_slider = Slider(
    ax=ax_S0,
    label='S0',
    valmin=0,
    valmax=1000,
    valinit=init_S0,
)

ax_I0 = fig.add_axes([0.1, 0.2, 0.35, 0.03])
I0_slider = Slider(
    ax=ax_I0,
    label='I0',
    valmin=0,
    valmax=1000,
    valinit=init_I0,
)

ax_T0 = fig.add_axes([0.1, 0.3, 0.35, 0.03])
T0_slider = Slider(
    ax=ax_T0,
    label='T0',
    valmin=0,
    valmax=1000,
    valinit=init_T0,
)


ax_alpha = fig.add_axes([0.55, 0.05, 0.0225, 0.4])
alpha_slider = Slider(
    ax=ax_alpha,
    label="alpha",
    valmin=0,
    valmax=1,
    valinit=init_alpha,
    orientation="vertical"
)

ax_beta = fig.add_axes([0.65, 0.05, 0.0225, 0.4])
beta_slider = Slider(
    ax=ax_beta,
    label="beta",
    valmin=0,
    valmax=10,
    valinit=init_beta,
    orientation="vertical"
)

ax_delta = fig.add_axes([0.75, 0.05, 0.0225, 0.4])
delta_slider = Slider(
    ax=ax_delta,
    label="delta",
    valmin=0,
    valmax=1,
    valinit=init_delta,
    orientation="vertical"
)

ax_eta = fig.add_axes([0.85, 0.05, 0.0225, 0.4])
eta_slider = Slider(
    ax=ax_eta,
    label="eta",
    valmin=0,
    valmax=1,
    valinit=init_eta,
    orientation="vertical"
)

ax_gamma = fig.add_axes([0.95, 0.05, 0.0225, 0.4])
gamma_slider = Slider(
    ax=ax_gamma,
    label="gamma",
    valmin=0,
    valmax=1,
    valinit=init_gamma,
    orientation="vertical"
)


def update(val):
    RR0 = (beta_slider.val)/(alpha_slider.val + gamma_slider.val) + (alpha_slider.val*delta_slider.val*beta_slider.val)/((alpha_slider.val + gamma_slider.val)*eta_slider.val)
    N_updated = S0_slider.val + I0_slider.val + T0_slider.val + init_R0
    t,S_val,I_val,T_val,R_val = Adam_Bashforth2(alpha_slider.val,beta_slider.val,delta_slider.val,gamma_slider.val,eta_slider.val,N_updated,S0_slider.val,I0_slider.val,T0_slider.val,h,eps,Nmax)
    line_S.set_ydata(S_val)
    line_I.set_ydata(I_val)
    line_T.set_ydata(T_val)
    line_R.set_ydata(R_val)
    ax.set_ylim(-10, N_updated+20)
    fig.canvas.draw_idle()
    text_RR0.set_text(f"R0 = {RR0:.2f}")


S0_slider.on_changed(update)
I0_slider.on_changed(update)
T0_slider.on_changed(update)
alpha_slider.on_changed(update)
beta_slider.on_changed(update)
delta_slider.on_changed(update)
eta_slider.on_changed(update)
gamma_slider.on_changed(update)

text_RR0 = plt.text(0.02, -0.40, f"R0 = {RR0:.2f}", transform=ax.transAxes)

resetax = fig.add_axes([0.8, 0.95, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    S0_slider.reset()
    I0_slider.reset()
    T0_slider.reset()
    alpha_slider.reset()
    beta_slider.reset()
    delta_slider.reset()
    eta_slider.reset()
    gamma_slider.reset()
button.on_clicked(reset)

ax.legend()
plt.show()
