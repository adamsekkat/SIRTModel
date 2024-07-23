import tkinter as tk
from tkinter import ttk

import subprocess

# ----------------------------------- Realisé par Sekkat Adam , Sanchez Matthew , Roess Célia # ----------------------------------- 

def run_script(script_name):
    subprocess.run(["python3", script_name])

def afficher_texte(event):
    bouton.grid_remove()
    texte = r"""Les paramètres du modèle
• alpha : fraction d'individus infectés sélectionnés pour être traitée par unité de temps;
• beta : nombre moyen d'individus rencontrés par un individu par unité de temps;
• gamma : le taux de guérison;
• eta : taux d'individus traités qui deviennent immunisés ou décédés;
• delta : facteur de réduction de l'infectivité grâce à un traitement."""
    label_texte.configure(text=texte, font=("TkDefaultFont",8), justify="left")

def cacher_texte(event):
    bouton.grid()
    bouton.lift()
    label_texte.configure(text="")


fenetre = tk.Tk()
fenetre.title("Modélisation de l'évolution d'une épidémie")
fenetre.state('zoomed')


bg_image = tk.PhotoImage(file="im2.png")
bg_label = tk.Label(fenetre, image=bg_image)
bg_label.place(relx=0.5, rely=0.5, anchor="center")


titre_label = tk.Label(fenetre, text="Evolution d'une épidémie", font=("Helvetica", 24))
titre_label.place(relx=0.5, rely=0.1, anchor="center")


bouton1_command = lambda: run_script("projetAN2_EE.py")
bouton2_command = lambda: run_script("projetAN2_EI.py")
bouton3_command = lambda: run_script("projetAN2_RK4.py")
bouton4_command = lambda: run_script("projetAN2_AB2.py")


button_frame = tk.Frame(fenetre)
button_frame.place(relx=0.5, rely=0.5, anchor="center")


bouton1 = tk.Button(button_frame, text="Euler Explicite", command=bouton1_command, font=("Helvetica", 14), width=20, height=2)
bouton1.grid(row=0, column=0)

bouton2 = tk.Button(button_frame, text="Euler Implicite", command=bouton2_command, font=("Helvetica", 14), width=20, height=2)
bouton2.grid(row=0, column=2)

bouton3 = tk.Button(button_frame, text="Runge Kutta 4", command=bouton3_command, font=("Helvetica", 14), width=20, height=2)
bouton3.grid(row=2, column=0)

bouton4 = tk.Button(button_frame, text="Adams Bashforth 2", command=bouton4_command, font=("Helvetica", 14), width=20, height=2)
bouton4.grid(row=2, column=2)


conteneur = tk.Frame(fenetre)
conteneur.grid(row=0, column=0, padx=10, pady=10, sticky="sw")


bouton = tk.Button(conteneur, text="Aide?", font=("TkDefaultFont", 10), width=15, height=2)
bouton.grid(row=0, column=0)


label_texte = tk.Label(conteneur, text="", font=("TkDefaultFont", 13))
label_texte.grid(row=0, column=0, padx=10)


conteneur.bind("<Enter>", afficher_texte)
conteneur.bind("<Leave>", cacher_texte)
bouton.lift()

fenetre.mainloop()
