import numpy as np
import matplotlib.pyplot as plt

import estimator as e
import math

def calculate_estimate(data, q, r):
    n = data.shape[0]
    noise = np.random.normal(loc=0, scale=math.sqrt(r), size=n)
    meas = noise + data

    x0 = np.array([meas[0]])
    P0 = np.array([r])
    Q = np.array([q])
    G = np.array([1])
    R = np.array([r])

    kf = e.Kf(x0, P0, Q, G, R)

    A = np.array([1])
    H = np.array([1])

    est = np.zeros(n)
    est[0] = meas[0]

    for i, m in enumerate(meas[1:]):
        xp, _ = kf.predict(A, H)
        xc, _ = kf.correct(H, np.array([m]))
        est[i + 1] = xc[0][0]

    err_kf = est - data

    return err_kf

process_var = 0.1
num_samples = 400  # Количество выборок

data_const5 = np.full(num_samples//2, 5) # 5
data_const6 = np.full(num_samples//4, 5) # 5.8
data_const4 = np.full(num_samples//4, 5) # 4.2

data = np.concatenate((data_const5,data_const6,data_const4))
repeat_num = 50
num_iterations = 1000  # Количество итераций
var_kf_step = np.zeros(num_samples)  # Вариация оценки Калмана по шагам
var_kf_step_noise = np.zeros(num_samples)

for i in range(num_iterations):
    meas_var = 1.
    noise_process = np.random.normal(loc=0, scale=2*math.sqrt(process_var), size=num_samples//repeat_num)
    data_with_noise = data + np.repeat(noise_process, repeat_num)
    err_kf = calculate_estimate(data_with_noise, 0.0, meas_var)
    err_kf_step_noise = calculate_estimate(data_with_noise, process_var, meas_var)
    var_kf_step += err_kf ** 2
    var_kf_step_noise += err_kf_step_noise ** 2

var_kf_step /= num_iterations
var_kf_step_noise /= num_iterations
std_kf_step = np.sqrt(var_kf_step)
std_kf_step_noise = np.sqrt(var_kf_step_noise)

# Построение графика стандартного отклонения
plt.plot(np.arange(len(std_kf_step)), std_kf_step, label='kf')
plt.plot(np.arange(len(std_kf_step_noise)), std_kf_step_noise, label='kf_noise')

plt.xlabel('Index')  # Метка оси X
plt.ylabel('std')  # Метка оси Y
plt.title('График std1')  # Заголовок графика
plt.grid(True)  # Отображение сетки на графике
plt.legend()  # Отображение легенды
plt.show()  # Показать график