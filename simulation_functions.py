#!/usr/bin/env python
# -*- coding: utf-8 -*-


#Estimate Connections from inverse covariance
#Copyright (C) 2018  Jonathan Schiefer
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


import numpy as np
from scipy.linalg import expm
from scipy.linalg import solve_lyapunov
from scipy.signal import fftconvolve
from scipy.stats import gamma as gamma_func


def orn_uhl(G, simDur, rho, p, tau, dt, sigma):
    """Simulate an AR(1) process.

    Discrete sample from an Ornstein-Uhlenbeck process defined by
    d/dt v(t) = A_0 v(t)+ epsilon(t) where epsilon(t) ~ WN(0, C_0).

    Input:
    G -- array, shape (nodes, nodes), connectivity matrix
    simDur -- int, duration of simulation
    rho -- float, connection strength
    tau -- float, time constant
    dt -- float, temporal resolution of the sampled process
    sigma -- float, standard deviation of the white noise
    Returns:
    data - array, shape (N, simDur * 1/dt), simulated data
    """
    N = np.shape(G)[0]
    noDataPoints = int(simDur / dt)
    C_0 = sigma**2/tau**2 * np.eye(N)
    A_0 = ((rho / np.sqrt(N * p * (1 - p))) * G - np.eye(N))/tau
    A = expm(A_0 * dt)
    sig = solve_lyapunov(-A_0, C_0 - (np.dot(A, np.dot(C_0, A.T))))
    C = solve_lyapunov(-A_0, C_0)

    data = np.zeros((N, noDataPoints + 100))

    noise = np.transpose(np.random.multivariate_normal(
        np.zeros(N), sig, noDataPoints + 100))
    data[:, 0] = noise[:, 0]

    for time in range(1, noDataPoints + 100):
        data[:, time] = np.dot(A, data[:, time - 1]) + noise[:, time]

    return data[:, 100:]


def norm_data(data):
    """Norm data by mean and std.

    Input: arraylike structure, if 2D rows represent different
    trials
    Return: arraylike structure, normed data.
    """
    if len(np.shape(data)) == 2:
        data -= np.mean(data, 1)[:, np.newaxis]
        data /= np.std(data, 1)[:, np.newaxis]

    if len(np.shape(data)) == 1:
        data -= np.mean(data)
        data /= np.std(data)
    return data


def calc_inst_cov(data, mode='time'):
    """Calculate zero-lag covariance from data, return arraylike."""
    data = norm_data(data)
    N, T = np.shape(data)
    if mode == 'time':
        return (1. / (T - 1)) * np.dot(data, np.transpose(data))
    if mode == 'fourier':
        return (1. / (T - 1)) * np.dot(data, np.conjugate(np.transpose(data)))


def hrf(time_array):
    """Calculate canonical HRF for given timearray, return arraylike."""
    fil = gamma_func.pdf(time_array, 6) - \
        (1/6. * gamma_func.pdf(time_array, 16))
    dt = time_array[1]
    return fil / (np.linalg.norm(fil, ord=2)**2 * dt)


def filter_data(data, dt):
    """Filter data with canonical HRF for given temporal resolution, return arraylike."""
    N, T = np.shape(data)
    data = norm_data(data)
    xs = np.arange(0, 30, dt)
    fil = hrf(xs)
    data_filtered = np.zeros(np.shape(data))
    for i in range(N):
        data_filtered[i, :] = fftconvolve(data[i, :], fil, mode='same')

    return norm_data(data_filtered)
