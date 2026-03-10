import numpy as np
import matplotlib.pyplot as plt
from time import time as time
from reglinA import reglinA
from reglinB import reglinB
from newton import newton
from scipy.linalg import lu
from numpy.linalg import solve, norm
from reqfreqB(X,h) import

X = np.loadtxt("points.txt") #Converti le texte en array

# 2.1
# a)
plt.figure() # faire cet commande avant chaque nouvel figure
# Attention, erreur dans l'énoncé, on veut la colonne de gauche du array (= les x)
# et la colonne de droite (= les y), les indices sont donc tout de haut en bas et 0 et 1
plt.scatter(X[:,0],X[:,1],s=3)
plt.title("Figure 1: Tracer des 1000 points")
plt.xlabel("xi")
plt.ylabel("yi")
plt.show()

# c)
reglinA(X)
print(reglinA(X)) # affiche B calculé avec A (sous forme de vecteur/matrice, soit B1, B2)
reglinB(X)
print(reglinB(X)) # affiche B calculé avec B

# nuage de points et 2 droites sur même graphique
# intervalle de valeur (valeur de x sur graphique):
interx = np.linspace(1,6)
# Droite créer par A (valeur obtenus de Beta)
yA = 4.00933858 + interx * 1.42326685
# Droite créer par B
yB = 4.00933858 + interx * 1.42326685

plt.figure() #Les 3 plots vont se superposé
plt.scatter(X[:,0], X[:,1], s=3)
plt.plot(interx, yA)
plt.plot(interx, yB) # Remarque: Les 2 méthodes se supperpose parfaitement
plt.title("Figure 2: Tracer des points et des 2 droites de régression")
plt.xlabel("xi")
plt.ylabel("yi")
plt.show()

# 2.2
# f)
xi = X[:, 0:1]
yi = X[:, 1:2]

F = lambda beta : (beta[0] + beta[1] * np.sqrt(xi - beta[2])) - yi
J = lambda beta : np.array([np.ones_like(xi.flatten()), np.sqrt(xi.flatten()-beta[2]), -beta[1] / (2 * np.sqrt(xi.flatten() - beta[2]))]).T

beta_final = newton(np.array([[1.0], [1.0], [1.0]]), F, J, 1e-7, 20)

print(f"beta1 = {beta_final[0]}")
print(f"beta2 = {beta_final[1]}")
print(f"beta3 = {beta_final[2]}")

plt.figure()
plt.scatter(X[:,0],X[:,1],s=3)
x_plot = np.linspace(1, 6)
y_plot = beta_final[0] + beta_final[1] * np.sqrt(x_plot - beta_final[2])
plt.plot(x_plot, y_plot, color = "red")
plt.title("Figure 3: Régression non-linéaires avec la méthode Newton")
plt.xlabel("xi")
plt.ylabel("yi")
plt.show()

# 2.3
# j)

