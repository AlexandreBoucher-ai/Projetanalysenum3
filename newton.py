import numpy as np
from numpy.linalg import solve, norm
import scipy.linalg as la
from scipy.linalg import lu

def newton(beta_init, F, J, tol, nmax):
    '''
    Args:
        beta_init : point initial (vecteur colonne 3x1)
        F         : fonction vectorielle (vecteur colonne nx1) dont on cherche le zero. DEPEND DE BETA.
        J         : matrice jacobienne de F (matrice nxn). DEPEND DE BETA.
        # n n,est pas le nb de points ici, c'est mêlant
        tol       : tolerance pour determiner la convergence
        nmax      : nombre maximal d'iterations

    Returns:
        beta      : vecteur des coefficients de la courbe de regression
    '''
    # Préparation itération
    # on pose beta = beta0
    beta = beta_init;
    assert beta.shape == (3,1), "beta_init doit être un vecteur colonne!"
    n = 0;
    # on poe beta0 = la variable d'accumulation
    res = norm(F(beta.flatten()));
    # dbeta = delta beta = variation de beta
    # on initialise l'erreur sur beta à infini
    dbeta = np.inf;
    # on fait l'itération de newton
    # on arrête l'itération lorsque dbeta est plus grand que la tolérance pour la divergeance
    while res > tol and norm(dbeta) > tol and n < nmax :
        # cacul le jacobien et la fonction en fonction de beta
        Jb = J(beta.flatten());
        Fb = F(beta.flatten());
        assert Fb.shape == (1000,1), "F doit retourner un vecteur colonne!"
        # Resolution du systeme pour calculer la correction
        # Calcul de dbeta en utilisant approche (A) ou (B)
        # MODIFIER ICI !
        # rappel dbeta = -(Fb/Jb) ne fonctionne pas!




   
        #dbeta = ...


        # Appliquer la correction
        assert dbeta.shape == (3,1), "dbeta doit être un vecteur colonne!"
        beta += dbeta;

        # Calcul du residu au nouveau point
        res = norm(F(beta.flatten()));
        print(f"Iteration {n} : ||dbeta|| = {norm(dbeta)}, ||F(beta)|| = {res}")
        n += 1;

    return beta