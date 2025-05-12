import numpy as np
from numpy.linalg import svd, solve, qr
import pandas as pd
import matplotlib.pyplot as plt

import scipy
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from scipy.optimize import linear_sum_assignment
import networkx as nx

import cvxpy as cp
from cvxpy import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.problems.problem import Problem
from cvxpy import abs, log_det, sum, Variable
from cvxpy.problems.objective import Minimize
from cvxpy.problems.problem import Problem
from scipy.optimize import minimize
#from cvxpy.expressions.variables.semidef_var import Semidef



class pLSI(object):
    def __init__(
        self,
        eps=1e-05,
        use_mpi=False,
        return_anchor_docs=True,
        verbose=0,
        precondition=False,
        solver=False
    ):
        """
        Parameters
        -----------

        """
        self.return_anchor_docs = return_anchor_docs
        self.verbose = verbose
        self.use_mpi = use_mpi
        self.precondition = precondition
        self.solver = solver

    def fit(self, X, K):

        print("Running pLSI...")
        self.U, self.L, self.V = svds(X, k=K)
        self.L = np.diag(self.L)
        self.V = self.V.T
        self.U_init = None

        
        print("Running SPOC...")
        J, H_hat = self.preconditioned_spa(self.U, K, self.precondition)
        print("Estimating W")
        self.W_hat = self.get_W_hat(self.U, H_hat)
        print("Estimating A")
        self.A_hat = self.get_A_hat(self.W_hat, X, H_hat)
        if self.return_anchor_docs:
            self.anchor_indices = J

        return self

    @staticmethod
    def preprocess_U(U, K):
        for k in range(K):
            if U[0, k] < 0:
                U[:, k] = -1 * U[:, k]
        return U
    
    @staticmethod
    def precondition_M(M, K):
        Q = cp.Variable((K, K), symmetric=True)
        objective = cp.Maximize(cp.log_det(Q))
        constraints = [cp.norm(Q @ M, axis=0) <= 1]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=False)
        Q_value = Q.value
        return Q_value
    
    def preconditioned_spa(self, U, K, precondition=True):
        J = []
        M = self.preprocess_U(U, K).T
        if precondition:
            L = self.precondition_M(M, K)
            S = L @ M
        else:
            S = M
        
        for t in range(K):
                maxind = np.argmax(np.linalg.norm(S, axis=0))
                s = np.reshape(S[:, maxind], (K, 1))
                S1 = (np.eye(K) - np.dot(s, s.T) / np.linalg.norm(s) ** 2).dot(S)
                S = S1
                J.append(maxind)
        H_hat = U[J, :]
        return J, H_hat

    def get_W_hat(self, U, H):
        projector = H.T.dot(np.linalg.inv(H.dot(H.T)))
        theta = U.dot(projector)
        theta_simplex_proj = np.array([self._euclidean_proj_simplex(x) for x in theta])
        return theta_simplex_proj

    def get_A_hat(self, W_hat, M, H_hat):

        n_words = M.shape[1]
        n_topics = W_hat.shape[1]

        if self.solver == 'cvxpy':
            Theta = Variable((n_topics, n_words))
            constraints = [
                sum(Theta[i, :]) == 1 for i in range(n_topics)
            ]
            constraints += [
                Theta[i, j] >= 0 for i in range(n_topics)
                for j in range(n_words)
            ]
            obj = Minimize(cp.norm(M - W_hat @ Theta, 'fro'))
            prob = Problem(obj, constraints)
            prob.solve()
            return np.array(Theta.value)
        elif self.solver == 'projector':
            projector = (np.linalg.inv(W_hat.T.dot(W_hat))).dot(W_hat.T)
            theta = projector.dot(M)
            theta_simplex_proj = np.array([self._euclidean_proj_simplex(x) for x in theta])
            return theta_simplex_proj
        elif self.solver == 'scipy':
            return self._fast_theta_solver(M, W_hat)
        else:
            ### sparse SVD of M:
            A = H_hat @ self.L @ self. V.T
            theta_simplex_proj = np.array([self._euclidean_proj_simplex(x) for x in A])
            return(theta_simplex_proj)

    
    @staticmethod
    def _fast_theta_solver(M, W_hat):
        n_topics, n_words = M.shape[0], W_hat.shape[1]

        def objective(theta_flat):
            Theta = theta_flat.reshape((n_topics, n_words))
            return np.linalg.norm(M - W_hat @ Theta, 'fro')**2

        # Initial guess
        x0 = np.ones(n_topics * n_words) / n_words

        constraints = []
        # Row sums to 1
        for i in range(n_topics):
            def constr_factory(i):
                return {'type': 'eq', 'fun': lambda x, i=i: np.sum(x[i * n_words:(i + 1) * n_words]) - 1}
            constraints.append(constr_factory(i))
        
        # Non-negativity
        bounds = [(0, None) for _ in range(n_topics * n_words)]

        result = minimize(
            objective,
            x0,
            method='SLSQP',  # Or try 'trust-constr' for large-scale
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )

        return result.x.reshape((n_topics, n_words))
        
    @staticmethod
    def _euclidean_proj_simplex(v, s=1):
        (n,) = v.shape
        if v.sum() == s and np.all(v >= 0):
            return v
        
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
       
        theta = (cssv[rho] - s) / (rho + 1.0)
        w = (v - theta).clip(min=0)
        return w