import numpy as np
import matplotlib.pyplot as plt

import estimator as e
import math

meas_var = 1.
A = np.array([1])
H = np.array([1])


def make_kalman_filter(measurement):
    process_var = 0.0
    x0 = np.array([measurement])
    P0 = np.array([meas_var])
    Q = np.array([process_var])
    G = np.array([1])
    R = np.array([meas_var])
    kf  = e.Kf(x0, P0, A, Q, G, H, R)
    return kf


def make_kalman_filter_processnoise(measurement):
    process_var = 0.1
    x0 = np.array([measurement])
    P0 = np.array([meas_var])
    Q = np.array([process_var])
    G = np.array([1])
    R = np.array([meas_var])
    kf  = e.Kf(x0, P0, A, Q, G, H, R)
    return kf


def add_noise(data):
    meas = H * data
    meas = meas + np.random.normal(loc=0, scale=math.sqrt(meas_var), size=(meas.shape[0], meas.shape[1]))
    return meas


def calc_err(data, make_estimator):
    meas = add_noise(data)
    estimator = make_estimator(meas[:, 0])
    est = np.zeros((meas.shape[0], meas.shape[1]-1))

    for col in range(est.shape[1]):
        m = meas[:, col + 1]
        xp, _ = estimator.predict()
        xc, _ = estimator.correct(m)
        est[:, col] = xc[:]

    err = est - data[:, 1:]

    return err


def calc_std_err(data, make_estimator):
    num_iterations = 1000
    var_err = np.zeros((data.shape[0], data.shape[1]-1))

    for i in range(num_iterations):
        err = calc_err(data, make_estimator)
        var_err += err ** 2

    var_err /= num_iterations
    return np.sqrt(var_err)


if __name__ == "__main__":
    data = np.full((1, 400), 5)
    data_noise = np.random.normal(loc=0, scale=2*math.sqrt(0.1), size=(1, data.shape[1] // 50))
    data_with_noise = data + np.repeat(data_noise, 50)
    data = data_with_noise

    meas = add_noise(data)
    plt.figure()
    plt.plot(np.arange(len(data.T)), data.T, label='signal')
    plt.plot(np.arange(len(meas.T)), meas.T, label='signal+noise', linestyle='', marker='+')
    plt.xlabel('Index')
    plt.ylabel('Data')
    plt.grid(True)
    plt.legend()
    plt.show()

    std_err = calc_std_err(data, make_kalman_filter)
    std_err_pn = calc_std_err(data, make_kalman_filter_processnoise)
    plt.figure()

    plt.plot(np.arange(len(std_err[0, :]))+1, std_err[0, :].T, label='kf')
    plt.plot(np.arange(len(std_err_pn[0, :]))+1, std_err_pn[0, :].T, label='kf_process_noise')

    plt.grid(True)
    plt.legend()
    plt.show()

    #plt.figure()





# def calculate_estimate(data, q, r):
#     n = data.shape[0]
#     noise = np.random.normal(loc=0, scale=math.sqrt(r), size=n)
#     meas = noise + data
#
#
#     for i, m in enumerate(meas[1:]):
#         xp, _ = kf.predict(A, H)
#         xc, _ = kf.correct(H, np.array([m]))
#         est[i + 1] = xc[0][0]
#
#     err_kf = est - data
#
#     return err_kf
#
# process_var = 0.1
# meas_var = 1.
# num_samples = 400  # Количество выборок
#
# data_const5 = np.full(num_samples//2, 5) # 5
# data_const6 = np.full(num_samples//4, 5) # 5.8
# data_const4 = np.full(num_samples//4, 5) # 4.2
#
#
# fig, axes = plt.subplots(1, 2)
#
# plt.figure()
# data = np.zeros(num_samples)
# data[0] = 1.
# indices = np.arange(1, num_samples)
# data[indices] = data[0] + 0.1 * indices
# dn = data + np.random.normal(loc=0, scale=math.sqrt(meas_var), size=data.shape[0])
# plt.plot(np.arange(len(data)), data, label='data')
# #plt.plot(np.arange(len(data)), data+dn, marker='+', linestyle='none')
# plt.show()
#
#
# repeat_num = 50
# num_iterations = 1000  # Количество итераций
# var_kf_step = np.zeros(num_samples)  # Вариация оценки Калмана по шагам
# var_kf_step_noise = np.zeros(num_samples)
#
# for i in range(num_iterations):
#     noise_process = np.random.normal(loc=0, scale=2*math.sqrt(process_var), size=num_samples//repeat_num)
#     data_with_noise = data + np.repeat(noise_process, repeat_num)
#     err_kf = calculate_estimate(data_with_noise, 0.0, meas_var)
#     err_kf_step_noise = calculate_estimate(data_with_noise, process_var, meas_var)
#     var_kf_step += err_kf ** 2
#     var_kf_step_noise += err_kf_step_noise ** 2
#
# var_kf_step /= num_iterations
# var_kf_step_noise /= num_iterations
# std_kf_step = np.sqrt(var_kf_step)
# std_kf_step_noise = np.sqrt(var_kf_step_noise)
#
# # Построение графика стандартного отклонения
# plt.plot(np.arange(len(std_kf_step)), std_kf_step, label='kf')
# plt.plot(np.arange(len(std_kf_step_noise)), std_kf_step_noise, label='kf_noise')
#
# plt.figure()
# plt.xlabel('Index')  # Метка оси X
# plt.ylabel('std')  # Метка оси Y
# plt.title('График std1')  # Заголовок графика
# plt.grid(True)  # Отображение сетки на графике
# plt.legend()  # Отображение легенды
# plt.show()  # Показать график