import numpy as np
import matplotlib.pyplot as plt
from time import time as time
from reglinA import reglinA
from reglinB import reglinB
from newton import newton

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

 # e)
 # modification de relinA pour marcher avec la fonction Newton
def reglinA(X) :
    '''
    Args:
        X : le jeu de donnees n x 2

    Returns :
        beta : vecteur des coefficients de la droite de regression
    '''
    n = X.shape[0] # Nombre de points

    # Creation du membre de droite
    y = X[:,1].reshape(n,1)

    # Creation de la matrice A
    A = np.hstack((np.ones((n,1)), X[:,0].reshape(n,1)))

    # Resolution du systeme rectangulaire (approche A)
    AtA = A.T@A
    L, U = lu(AtA, permute_l=True)
    z = solve(L,A.T@y)
    beta = solve(U,z)

    print(f"Norme du residu ||F(beta)|| = ||A*beta - y|| = {norm(A@beta-y)}")

    return beta