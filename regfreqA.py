import numpy as np
from scipy.linalg import lu
from numpy.linalg import solve, norm

# on doit ajouter k en argument (pour le nombre d'inconnu beta)
def regfreqA(X,k) :
    '''
    Args:
        X : le jeu de donnees n x 2

    Returns :
        beta : vecteur des coefficients de la droite de regression
    '''
    n = X.shape[0] # Nombre de points
    # Creation du membre de droite
    y = X[:,1].reshape(n,1)

    # Creation des membres
    # On crée A
    A = np.ones((n, 2*k-1))
    # on crée x, vu qu'il va falloir le réutiliser c'est plus simple
    x = X[:,0]
    # première sommation (on doit précisé commence à 1 car i=1 dans la sommation et non 0)
    for i in range(1, k):
        # toute la colonne i (qui commence à 1)
        A[:,i] = np.cos(i*x)
    for i in range(1, k):
        A[:,k-1+i] = np.sin((i*x))
    
    # Resolution du systeme rectangulaire (approche A)
    AtA = A.T@A
    L, U = lu(AtA, permute_l=True)
    z = solve(L,A.T@y)
    beta = solve(U,z)

    print(f"Norme du residu ||F(beta)|| = ||A*beta - y|| = {norm(A@beta-y)}")

    return beta