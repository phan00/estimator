import numpy as np
import matplotlib.pyplot as plt
from arr2ltx import to_latex


import estimator as e
import math


#%% Инициализируем Фильтра Калмана
#%markdown
meas_var = 4.
dt = 0.2
process_var = 1.
Ae = np.array([[1., dt, 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., dt],
               [0., 0., 0., 1.]])
Af = np.array([[1., dt,  0., 0.],
               [0., 1.,  0., 0.],
               [0., 0.,  1., dt],
               [0., 0.,  0., 1.]])


H = np.array([[1., 0., 0., 0.],
              [0., 0., 1., 0.]])

Re = np.diag([meas_var, meas_var])
Rf = np.diag([meas_var, meas_var])

Qe = np.diag([process_var, process_var])
Qf = np.diag([process_var, process_var])

G = np.array([[dt**2/2, 0],
             [dt, 0],
             [0, dt**2/2],
             [0, dt]])

trueInitialState = np.array([10., 2., 30., 2.])
trueMeas = H@trueInitialState
trueMeas = trueMeas[:, np.newaxis]

to_latex(trueMeas)


def make_kalman_filter(measurement):
    max_speed = 4
    init_var  = Rf[0, 0]
    x0 = H.T @ measurement
    P0 = np.diag([init_var, (max_speed/3)**2, init_var, (max_speed/3)**2])

    kf = e.Kf(x0, P0, Af, Qf, G, H, Rf)
    return kf

def add_noise(data, meas_var):
    meas = H @ data
    meas = meas + np.sqrt(meas_var) @ np.random.normal(loc=0, scale=math.sqrt(1.0), size=(meas.shape[0], meas.shape[1]))
    return meas


def step(data, make_estimator):
    meas = add_noise(data, Re)
    estimator = make_estimator(meas[:, 0][:, np.newaxis])
    est = np.zeros((data.shape[0], data.shape[1]-1))

    for col in range(est.shape[1]):
        m = meas[:, col + 1]
        xp, _ = estimator.predict()
        m1 = np.array([m[0], m[1]])
        xc, _ = estimator.correct(m1.T)
        est[:, col] = np.squeeze(xc[:])

    return est,meas


def add_process_noise(data, data_var):
    data_noise = data + np.sqrt(data_var) @ np.random.normal(loc=0, scale=1.0, size=(data.shape[0], data.shape[1]))
    return data_noise

def calc_err(data, make_estimator):
    QQ = G@Qe@G.T
    data_noise = add_process_noise(data, QQ)
    est, _ = step(data_noise, make_estimator)
    err = est - data_noise[:, 1:]
    return err

def calc_std_err(data, make_estimator):
    num_iterations = 2000
    var_err = np.zeros((data.shape[0], data.shape[1]-1))

    for i in range(num_iterations):
        err = calc_err(data, make_estimator)
        var_err += err ** 2

    var_err /= num_iterations
    return np.sqrt(var_err)

def make_true_data(initial):
    data = np.zeros((initial.shape[0], 40))
    data[:, 0] = initial.T
    for i in range(data.shape[1]-1):
        data[:, i+1] = np.squeeze(Ae @ data[:, i][:, np.newaxis])
    return data


if __name__ == "__main__":
    

    data = make_true_data(trueInitialState)    
    data_noise = add_process_noise(data, G@Qe@G.T)
    est, meas = step(data_noise, make_kalman_filter)

    plt.figure()
    plt.plot(data_noise[0, :], data_noise[2, :], label='Truth')
    plt.plot(meas[0, :], meas[1, :], label='Measurements', linestyle='', marker='+')
    plt.plot(est[0, :], est[2, :], label='Estimates')
    plt.xlabel('x,met.')
    plt.ylabel('y,met.')
    plt.grid(True)
    plt.legend()
    plt.show()

    std_err = calc_std_err(data, make_kalman_filter)

    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot((np.arange(len(std_err[0, :]))+1)*dt, std_err[0, :].T)
    plt.xlabel('Time,s')
    plt.ylabel('std_x, met')
    plt.grid(True)
    plt.subplot(4, 1, 2)
    plt.plot((np.arange(len(std_err[1, :]))+1)*dt, std_err[1, :].T)
    plt.grid(True)
    plt.xlabel('Time,s')
    plt.ylabel('std_vx, m/s')
    plt.subplot(4, 1, 3)
    plt.plot((np.arange(len(std_err[2, :]))+1)*dt, std_err[2, :].T)
    plt.grid(True)
    plt.xlabel('Time,s')
    plt.ylabel('std_y, met')
    plt.subplot(4, 1, 4)
    plt.plot((np.arange(len(std_err[3, :]))+1)*dt, std_err[3, :].T)
    plt.grid(True)
    plt.xlabel('Time,s')
    plt.ylabel('std_vy, m/s')
    plt.subplots_adjust(wspace=8.0, hspace=0.7)
    plt.show()






