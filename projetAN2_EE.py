import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib.widgets import Button, Slider
# ----------------------------------- Realisé par Sekkat Adam , Sanchez Matthew , Roess Célia # ----------------------------------- 

#Les paramètres alpha, beta, gamma, eta, delta peuvent être modifiés grâce à des Slider

#Paramètres du modèle 
alpha = 0.1
beta = 1.3
gamma = 0.1
eta = 0.1
delta = 0.2

#Conditions initiales
S0 = 92.0
I0 = 2.0
T0 = 0.0
R0 = 0
RR0 = (beta)/(alpha + gamma) + (alpha*delta*beta)/((alpha + gamma)*eta)

#Définition des autres constantes
N = S0 + I0 + T0 + R0
h = 0.2
eps = 1e-8
Nmax = 100

def phiS(S,I,T,beta,delta,N):
    return (-beta/N) * (I + delta*T) * S

def phiI(S,I,T,alpha,beta,delta,gamma,N):
    return (beta/N) * S * (I + delta*T) - (alpha + gamma) * I

def phiT(I,T,alpha,eta):
    return alpha*I - eta*T

def phiR(I,T,gamma,eta):
    return gamma*I + eta*T

def EulerExplicite(alpha,beta,gamma,eta,delta,S0,I0,T0,h,eps,Nmax):

    # np.arange() permet de créer un tableau allant de 0 à Nmax*h avec un pas de h
    t = np.arange(0, Nmax)
    
    i = 0
    S = [S0]
    I = [I0]
    T = [T0]
    R = [R0]

    while i < Nmax-1 :
        S.append(S[i]+h*phiS(S[i],I[i],T[i],beta,delta,N))
        I.append(I[i]+h*phiI(S[i],I[i],T[i],alpha,beta,delta,gamma,N))
        T.append(T[i]+h*phiT(I[i],T[i],alpha,eta))
        R.append(R[i]+h*phiR(I[i],T[i],gamma,eta))
        i = i+1
    
    return t,S,I,R,T



fig, ax = plt.subplots()
t,S,I,R,T = EulerExplicite(alpha,beta,gamma,eta,delta,S0,I0,T0,h,eps,Nmax)
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
    valinit=S0,
)

ax_I0 = fig.add_axes([0.1, 0.2, 0.35, 0.03])
I0_slider = Slider(
    ax=ax_I0,
    label='I0',
    valmin=0,
    valmax=1000,
    valinit=I0,
)

ax_T0 = fig.add_axes([0.1, 0.3, 0.35, 0.03])
T0_slider = Slider(
    ax=ax_T0,
    label='T0',
    valmin=0,
    valmax=1000,
    valinit=T0,
)


ax_alpha = fig.add_axes([0.55, 0.05, 0.0225, 0.4])
alpha_slider = Slider(
    ax=ax_alpha,
    label="alpha",
    valmin=0,
    valmax=1,
    valinit=alpha,
    orientation="vertical"
)

ax_beta = fig.add_axes([0.65, 0.05, 0.0225, 0.4])
beta_slider = Slider(
    ax=ax_beta,
    label="beta",
    valmin=0,
    valmax=5,
    valinit=beta,
    orientation="vertical"
)

ax_gamma = fig.add_axes([0.75, 0.05, 0.0225, 0.4])
gamma_slider = Slider(
    ax=ax_gamma,
    label="gamma",
    valmin=0,
    valmax=1,
    valinit=gamma,
    orientation="vertical"
)

ax_eta = fig.add_axes([0.85, 0.05, 0.0225, 0.4])
eta_slider = Slider(
    ax=ax_eta,
    label="eta",
    valmin=0,
    valmax=1,
    valinit=eta,
    orientation="vertical"
)

ax_delta = fig.add_axes([0.95, 0.05, 0.0225, 0.4])
delta_slider = Slider(
    ax=ax_delta,
    label="delta",
    valmin=0,
    valmax=1,
    valinit=delta,
    orientation="vertical"
)


def update(val):
    RR0 = (beta_slider.val)/(alpha_slider.val + gamma_slider.val) + (alpha_slider.val*delta_slider.val*beta_slider.val)/((alpha_slider.val + gamma_slider.val)*eta_slider.val)
    N_updated = S0_slider.val + I0_slider.val + T0_slider.val + R0
    t,S_val,I_val,R_val,T_val = EulerExplicite(alpha_slider.val,beta_slider.val,gamma_slider.val,eta_slider.val,delta_slider.val,S0_slider.val,I0_slider.val,T0_slider.val,h,eps,Nmax)
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
gamma_slider.on_changed(update)
eta_slider.on_changed(update)
delta_slider.on_changed(update)

text_RR0 = plt.text(0.02, -0.40, f"R0 = {RR0:.2f}", transform=ax.transAxes)

resetax = fig.add_axes([0.8, 0.95, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    S0_slider.reset()
    I0_slider.reset()
    T0_slider.reset()
    alpha_slider.reset()
    beta_slider.reset()
    gamma_slider.reset()
    eta_slider.reset()
    delta_slider.reset()
button.on_clicked(reset)

ax.legend()
plt.show()