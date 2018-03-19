#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Estimate Connections from inverse covariance
#Copyright (C) 2018  Jonathan Schiefer, Alexander Niederbühl, Volker Pernice
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Estimate connectivity with gradient descent algorithm."""
from numpy import *
from numpy.linalg import norm
import scipy.linalg as linalg
from scipy.linalg import eigvalsh
from scipy.linalg import expm


def cost(B, with_diag=True):
    """Calculate the L1-cost of matrix B.

    Parameters:
    B -- the matrix from which to calculate the costs
    with_diag -- if False the diagonal entries are ignored
    Returns: float
    """
    if with_diag:
        return sum(abs(B))
    else:
        return sum(abs(B - diag(diag(B))))  # set diagonal elements to zero by
        # subtracting a matrix containing the diagonal elements


def is_converged(y):
    """Determine if optimization is converged.

    Fitting a line to the last iterations of the cost functions y, the optimization is considered as converged if the slope of this line is given the costs of the last iterations y, the optimization is considered as converged if the slope of the best fitting line (in the least squares sense) is smaller bigger than -1e-5
    Parameters:
    y -- list (costs)
    Returns: bool
    """
    N = len(y)
    w = linalg.lstsq(array([range(N), ones(N)]).T, y)[0]
    return w[0] > -1e-5


def is_unitary(U):
    """Check wether matrix is unitary."""
    U = matrix(U)
    N = U.shape[0]
    assert abs((U * U.H - eye(N))).max() < 1e-10


def r_prod(X, Y):
    """Calculate product for conjugate gradient."""
    X = matrix(X)
    Y = matrix(Y)
    return 0.5 * real(trace(X * Y.H))


def get_gr_cost_m(M, with_diag=False):
    """Identify non-zero entries and return matrix multiplied by 0.5."""
    Magn = abs(M)

    if isnan(Magn).any():
        print('NAN in Magn')
    if isnan(M).any():
        print('NAN in M')
    if isinf(Magn).any():
        print('inf in Magn')
    if isinf(M).any():
        print('inf in M')

    gr_cost_m = where(Magn > 0, M/Magn, 0)
    if not(with_diag):
        gr_cost_m = gr_cost_m - diag(diag(gr_cost_m))
    gr_cost_m = 0.5 * matrix(gr_cost_m)

    return gr_cost_m


def gradient_cost_euclidian(B):
    """Calculate gradient of B, return matrix."""
    B = matrix(B)
    gr_cost_m = matrix(get_gr_cost_m(B))
    gr_cost_u = gr_cost_m * B.H

    return gr_cost_u


def gradient_cost_riemann(gr_euclid):
    """Calculate skew-hermitian part of euclidian gradient."""
    return 0.5 * (gr_euclid - gr_euclid.H)


def get_grad(B_est):
    """Calculate gradient of current estimate."""
    grad_euclidian = gradient_cost_euclidian(B_est)
    grad = gradient_cost_riemann(grad_euclidian)

    return grad


def armadijo(B, grad, stepsize):
    """Optimization step of armadijo algorithm, return matrix and float."""
    stepsize = float(stepsize)
    grad = matrix(grad)
    v = matrix(B)
    r_norm = 0.5 * real(trace((grad)*(grad).H))

    if isinf(stepsize) or stepsize < 1e-10:
        return matrix(eye(B.shape[0])), 0.

    P = matrix(expm(-stepsize * grad))
    Q = P*P

    while (cost(v) - cost(Q*v)) >= (stepsize * r_norm):
        P = Q
        Q = P*P
        stepsize *= 2
    while (cost(v) - cost(P*v)) < (stepsize/2 * r_norm):
        if isinf(stepsize):
            return matrix(eye(B.shape[0])), 0.
        P = matrix(expm(-stepsize * grad))
        stepsize /= 2.

    is_unitary(P)
    return P, stepsize


def final_armadijo_abort(B_est, stepsize):
    """Linesearch algorithm after convergence of gradient descent.

    Parameters:
    B_est -- matrix (current connectivity estimate)
    stepsize -- float (initial stepsize)
    Returns:
    B_est -- matrix (new connectivity estimate)
    costs -- list
    stepsizes -- list
    """
    B_est = matrix(B_est)
    costs = []
    stepsizes = []
    costs.append(cost(B_est))
    i = 0
    converged = False
    n_check_conv = 100

    while(not converged):
        i += 1
        if i % n_check_conv == 0:

            converged = is_converged(costs[-n_check_conv:])

        grad = get_grad(B_est)
        P, stepsize = armadijo(B_est, grad, stepsize)
        B_est = matrix(P) * B_est

        stepsizes.append(stepsize)
        costs.append(cost(B_est))

    return B_est, costs, stepsizes


def norm_grad(grad):
    """Normalize gradient, return matrix."""
    return grad/sqrt(r_prod(grad, grad))


def get_delta(grad, del_param=2000.):
    """Calculate stepsize.

    Parameters:
    grad -- matrix (the gradient for which to calculate the stepsize)
    del_param -- float (constant by which stepsize is divided)
    Returns:
    delta -- float (stepsize for current gradient)
    """
    del_param = float(del_param)
    wmax = abs(eigvalsh(1j*grad)).max()
    if wmax < 1e-10:
        return 0
    T = 2*pi/(wmax)
    delta = T/del_param
    return delta


def conjugate(B, xtol=.5, gtol=.5, ftol=1e-4, del_param=2000., check_unitary=True, max_iter=10000):
    """Conjugate gradient descent algorithm.

    Parameters:
    B -- initial condition for opimization (square root of inverse covariance matrix)
    xtol -- float (convergence parameter)
    gtol -- float (convergence parameter)
    ftol -- float (convergence parameter)
    del_param -- float (constant by which stepsize is divided)
    check_unitary -- bool (if True, in each step unitarity of new U is checked)
    max_iter -- int (max amount of iterations)
    Returns:
    B_est -- matrix (estimated connectivity matrix)
    """
    B = matrix(B)
    B_est = B.copy()
    costs = []
    stepsizes = []
    grads = []
    costs.append(cost(B_est))
    N = shape(B)[0]

    grad = get_grad(B_est)
    d = -grad
    grads.append(grad)
    delta = get_delta(grad, del_param=del_param)
    stepsize = delta

    U = matrix(expm(array(stepsize * d)))
    B_est = U * B_est
    costs.append(cost(B_est))
    stepsizes.append(stepsize)
    converged = False
    Best_solution = B_est.copy()
    j = 0
    while(not converged):
        j += 1

        d_old = d
        grad_old = grad.copy()
        grad = get_grad(B_est)
        beta = r_prod(grad, (grad - grad_old)) / r_prod(grad_old, grad_old)
        d = -grad + beta * d_old
        grads.append(d)
        stepsize = get_delta(d, del_param=del_param)

        U_old = U.copy()
        U = matrix(expm(stepsize * d))
        B_est = U * B_est
        if cost(B_est) < cost(Best_solution):
            Best_solution = B_est.copy()

        stepsizes.append(stepsize)
        costs.append(cost(B_est))

        dtX = d - U * (d.T * U)
        S = U - U_old - diag(diag(U - U_old))
        XDiff = linalg.norm(S, 'fro') / sqrt(N)
        FDiff = abs(costs[-2] - costs[-1]) / (abs(costs[-2])+1)
        nrmG = linalg.norm(dtX - diag(diag(dtX)), 'fro')

        if (XDiff < xtol and FDiff < ftol) or nrmG < gtol:
            if j <= 2:
                ftol = 0.1 * ftol
                xtol = 0.1 * xtol
                gtol = 0.1 * gtol
            else:
                converged = True
                print('converged')
                break

        if j > max_iter:
            converged = True
            print('reached max iteration')

    B_est = Best_solution.copy()
    B_est, dijo_costs, dijo_steps = final_armadijo_abort(B_est, stepsize)
    stepsizes.extend(dijo_steps)
    costs.extend(dijo_costs)

    if check_unitary:

        U = B_est * B.I
        U = matrix(U)
        N = U.shape[0]
        if abs((U*U.H-eye(N))).max() > 1e-10:
            print('not unitary')

        is_unitary(B_est * B.I)

    return -(B_est - diag(diag(B_est)))
