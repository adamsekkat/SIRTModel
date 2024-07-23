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

# Conditions Initiales
init_beta = 3.5
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

# la Fonction F(X) = 0 à resoudre
def F(S, I, T, R, S_n, I_n, T_n, R_n,alpha, beta, delta, eta, gamma, N,h):
    FS = S - S_n - h * phiS(beta,delta,N,S,I,T)
    FI = I - I_n - h * phiI(alpha,beta,delta,gamma,N,S,I,T)
    FT = T - T_n - h * phiT(alpha,eta,I,T)
    FR = R - R_n - h * phiR(gamma,eta,I,T)
    return np.array([FS, FI, FT, FR])

# calcul de la matrice Jacobienne
def JacobSIRT(S,I,R,T,alpha,beta,delta,eta,gamma,N,h) :
    J = np.zeros((4,4))
    J[0, 0] = 1 + h * beta/N * (I + delta * T)
    J[0, 1] = h * beta/N * S
    J[0, 2] = h * beta/N * delta * S
    J[1, 0] = -h * beta/N * I
    J[1, 1] = 1 + h *(alpha + gamma - beta/N * S)
    J[1, 2] = -h * beta/N * delta*S 
    J[2, 1] = -h * alpha
    J[2, 2] = 1 + h * eta
    J[3, 1] = -h * gamma
    J[3, 2] = -h * eta
    J[3, 3] = 1
    return J

# Iterations de Newton
def NewtonNd(f, df, x0,S_n,I_n,T_n,R_n,alpha,beta,delta,eta,gamma,N,eps, Nmax,h):
    x=x0
    k=0
    while(k<Nmax):
          x=x-np.dot(np.linalg.inv(JacobSIRT(x[0],x[1],x[2],x[3],alpha,beta,delta,eta,gamma,N,h)), F(x[0],x[1],x[2],x[3], S_n, I_n, T_n, R_n,alpha, beta, delta, eta, gamma, N,h))
          k=k+1
    return x

# Algorithme Euler Implicite
def EulerImplicite(S_0,I_0,T_0,R_0,alpha,beta,delta,eta,gamma,N,h,eps,Nmax):
    t = np.arange(0,Nmax+1)
    S_1 = S_0 +h*phiS(beta,delta,N,S_0,I_0,T_0)
    I_1 = I_0 +h*phiI(alpha,beta,delta,gamma,N,S_0,I_0,T_0)
    T_1 = T_0 +h*phiT(alpha,eta,I_0,T_0)
    R_1 = R_0 +h*phiR(gamma,eta,I_0,T_0)
    S = [S_0,S_1]
    I = [I_0,I_1]
    T = [T_0,T_1]
    R = [R_0,R_1]
    i = 1
    for k in range(1,Nmax):
        x0 = [S[k], I[k], T[k], R[k]] #vecteur contenant les valeur de Sn,In,Tn,Rn
        x00 = [S[k-1], I[k-1], T[k-1], R[k-1]]#vecteur contenant les valeur de Sn-1,In-1,Tn-1,Rn-1
        f = F(x0[0], x0[1], x0[2], x0[3], x00[0], x00[1], x00[2], x00[3], alpha, beta, delta, eta, gamma, N,h)
        df = JacobSIRT(x0[0], x0[1], x0[2], x0[3], alpha, beta, delta, eta, gamma, N,h)
        solution = NewtonNd(f, df, x0,x00[0],x00[1],x00[2],x00[3],alpha,beta,delta,eta,gamma,N,eps, Nmax,h)
        S.append(solution[0])
        I.append(solution[1])
        T.append(solution[2])
        R.append(solution[3])
    return t,S,I,T,R


fig, ax = plt.subplots()
t,S,I,T,R = EulerImplicite(init_S0,init_I0,init_T0,init_R0,init_alpha,init_beta,init_delta,init_eta,init_gamma,N,h,eps,Nmax)
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
    valmax=5,
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
    t,S_val,I_val,T_val,R_val = EulerImplicite(S0_slider.val,I0_slider.val,T0_slider.val,init_R0,alpha_slider.val,beta_slider.val,delta_slider.val,eta_slider.val,gamma_slider.val,N_updated,h,eps,Nmax)
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