import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib.widgets import Button, Slider
# ----------------------------------- Realisé par Sekkat Adam , Sanchez Matthew , Roess Célia # ----------------------------------- 

def phiS(beta,delta,N,S,I,T):
    return (-beta/N) * (I + delta*T) * S

def phiI(alpha,beta,delta,gamma,N,S,I,T):
    return (beta/N) * S * (I + delta*T) - (alpha + gamma) * I

def phiT(alpha,eta,I,T):
    return alpha*I - eta*T

def phiR(gamma,eta,I,T):
    return gamma*I + eta*T

#conditions initiales
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

#fonctions qui resoud le systeme suivant RK4
def RK4_SIRT(alpha,beta,delta,gamma,eta,N,S0,I0,T0,h,Nmax):
    rk4_s = [init_S0]
    rk4_i = [init_I0]
    rk4_t = [init_T0]
    rk4_r = [0]
    i = 0
    t = np.arange(0, (Nmax))
    while i < Nmax - 1 :
        # calcul des K1
        k1_s = phiS(beta, delta, N, rk4_s[i], rk4_i[i], rk4_t[i])
        k1_i = phiI(alpha, beta, delta, gamma, N, rk4_s[i], rk4_i[i], rk4_t[i])
        k1_t = phiT(alpha, eta, rk4_i[i], rk4_t[i])
        k1_r = phiR(gamma, eta, rk4_i[i], rk4_t[i])

        # calcul des K2
        k2_s = phiS(beta, delta, N, rk4_s[i] + h/2*k1_s, rk4_i[i] + h/2*k1_i, rk4_t[i] + h/2*k1_t)
        k2_i = phiI(alpha, beta, delta, gamma, N, rk4_s[i] + h/2*k1_s, rk4_i[i] + h/2*k1_i, rk4_t[i] + h/2*k1_t)
        k2_t = phiT(alpha, eta, rk4_i[i] + h/2*k1_i, rk4_t[i] + h/2*k1_t)
        k2_r = phiR(gamma, eta, rk4_i[i] + h/2*k1_i, rk4_t[i] + h/2*k1_t)

        # calcul des K3
        k3_s = phiS(beta, delta, N, rk4_s[i] + h/2*k2_s, rk4_i[i] + h/2*k2_i, rk4_t[i] + h/2*k2_t)
        k3_i = phiI(alpha, beta, delta, gamma, N, rk4_s[i] + h/2*k2_s, rk4_i[i] + h/2*k2_i, rk4_t[i] + h/2*k2_t)
        k3_t = phiT(alpha, eta, rk4_i[i] + h/2*k2_i, rk4_t[i] + h/2*k2_t)
        k3_r = phiR(gamma, eta, rk4_i[i] + h/2*k2_i, rk4_t[i] + h/2*k2_t)

        #calcul des K4
        k4_s = phiS(beta, delta, N, rk4_s[i] + h*k3_s, rk4_i[i] + h*k3_i, rk4_t[i] + h*k3_t)
        k4_i = phiI(alpha, beta, delta, gamma, N, rk4_s[i] + h*k3_s, rk4_i[i] + h*k3_i, rk4_t[i] + h*k3_t)
        k4_t = phiT(alpha, eta, rk4_i[i] + h*k3_i, rk4_t[i] + h*k3_t)
        k4_r = phiR(gamma, eta, rk4_i[i] + h*k3_i, rk4_t[i] + h*k3_t)

        rk4_s.append(rk4_s[i] + h/6 * (k1_s + 2*k2_s + 2*k3_s + k4_s))
        rk4_i.append(rk4_i[i] + h/6 * (k1_i + 2*k2_i + 2*k3_i + k4_i))
        rk4_t.append(rk4_t[i] + h/6 * (k1_t + 2*k2_t + 2*k3_t + k4_t))
        rk4_r.append(rk4_r[i] + h/6 * (k1_r + 2*k2_r + 2*k3_r + k4_r))
        i = i + 1
    return t,rk4_s,rk4_i,rk4_t,rk4_r



fig, ax = plt.subplots()
t,resu_S,resu_I,resu_T,resu_R = RK4_SIRT(init_alpha,init_beta,init_delta,init_gamma,init_eta,N,init_S0,init_I0,init_T0,h,Nmax)
line_S, = ax.plot(t,resu_S,label="S", lw=2)
line_I, = ax.plot(t,resu_I,label="I", lw=2)
line_T, = ax.plot(t,resu_T,label="T", lw=2)
line_R, = ax.plot(t,resu_R,label="R", lw=2)


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
    t,resu_S_val,resu_I_val,resu_T_val,resu_R_val = RK4_SIRT(alpha_slider.val,beta_slider.val,delta_slider.val,gamma_slider.val,eta_slider.val,N_updated,S0_slider.val,I0_slider.val,T0_slider.val,h,Nmax)
    line_S.set_ydata(resu_S_val)
    line_I.set_ydata(resu_I_val)
    line_T.set_ydata(resu_T_val)
    line_R.set_ydata(resu_R_val)
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