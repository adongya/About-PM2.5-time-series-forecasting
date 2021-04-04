import numpy as np
import matplotlib.pyplot as plt
from Part.part_evaluate import *


def projection(real_data, predict_data, name):
    predict_data = predict_data.tolist()
    predict_list = []
    for j in range(24):
        error = []
        for i in range(len(predict_data)):
            if i % 24 == j:
                error.append(predict_data[i])
        predict_list.append(error)
    # print(len(predict_list))#24
    real_data = real_data.tolist()
    real_list = []
    for j in range(24):
        error = []
        for i in range(len(real_data)):
            if i % 24 == j:
                error.append(real_data[i])
        real_list.append(error)

    error_list = []
    for i in range(len(predict_list)):##24
        mape = MAPE1(real_list[i], predict_list[i])
        error_list.append(mape)
    error_list.append(error_list[0])##len(error_list)=25

        # error_list.append([predict_list[i], real_list[i]])

    # theta = np.arange(0, 24, 1).tolist()
    titles = np.arange(0, 24, 1).tolist()
    theta = np.arange(0, 2 * np.pi, (2/24) * np.pi)
    theta = theta.tolist()
    theta.append(0)

    plt.figure(1, figsize=(6, 4))     # wide
    # plt.figure(1, figsize=(5, 5))     # full

    plt.rc('font', family='Times New Roman')
    ax1 = plt.subplot(projection='polar')
    # ax1.set_thetagrids(np.arange(0.0, 360.0, 15.0))
    ax1.set_thetagrids(np.arange(0.0, 360.0, 15.0), labels=titles, weight="bold", color="black", fontsize=16)
    ax1.set_rticks(np.arange(0, 100, 25))
    ax1.set_rlabel_position(0)
    ax1.set_rlim(0, 100)
    # ax1.set_theta_offset(0.5 * np.pi)
    ax1.set_theta_direction(-1)
    ax1.set_theta_zero_location('N')
    ax1.plot(theta, error_list, '--', linewidth=2.5, marker='o',color="black")
    plt.title(name, fontsize=24, y=1.1)

    plt.subplots_adjust(left=0.0, bottom=0.1, right=1.0, top=0.84)     # wide
    # plt.subplots_adjust(left=0.0, bottom=0.08, right=1.0, top=0.85)     # full

    # plt.savefig('result\\pics\\projection_' + str(name) + '_wide.png')
    # plt.savefig('result\\pics\\projection_' + str(name) + '.png')

    plt.show()
    # plt.close(1)

def abs_sub(data_A, data_B):
    data_A = data_A.tolist()
    data_B = data_B.tolist()

    result = []
    for i in range(len(data_A)):
        result.append(abs(data_A[i] - data_B[i]))
    # result = sorted(result)

    return np.array(result)


def linear_coefs(dataX, dataY):
    points = np.c_[dataX, dataY]
    M = len(points)
    x_bar = np.mean(points[:, 0])
    sum_yx = 0
    sum_x2 = 0
    sum_delta = 0
    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        sum_yx += y * (x - x_bar)
        sum_x2 += x ** 2
    # 根据公式计算w
    w = sum_yx / (sum_x2 - M * (x_bar ** 2))

    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        sum_delta += (y - w * x)
    b = sum_delta / M
    return w, b


def scatter_and_linear(real, predict, name):
    w, b = linear_coefs(real, predict)

    plt.figure(1, figsize=(8, 4))       # wide
    # plt.figure(1, figsize=(6, 6))       # full

    # plt.rc('font', family='Times New Roman')

    x = np.linspace(-25, 425, 450)
    y = x
    plt.plot(x, y, color='gray', linewidth=2, linestyle=":")

    plt.scatter(real, predict, s=18, color='white', edgecolors='red')

    x = np.linspace(min(real), max(real), len(real))
    y = x * w + b
    plt.plot(x, y, color='purple', linewidth=4, linestyle="--")

    plt.grid(True, linestyle=":", color="lightgray", linewidth=0.5, axis='both')
    # plt.subplots_adjust(left=0.04, bottom=0.1, right=0.99, top=0.95)
    plt.xlabel("Real Data", fontsize=14)
    plt.ylabel("Prediction by "+name, fontsize=14)
    plt.xlim(-25, 425)
    plt.ylim(-25, 425)

    plt.subplots_adjust(left=0.1, bottom=0.14, right=0.995, top=0.90)       # wide
    # plt.subplots_adjust(left=0.12, bottom=0.12, right=0.98, top=0.92)     # full

    # plt.legend(loc='lower right', fontsize=14)
    plt.title(name+": Y="+str(round(w, 2))+"X+"+str(round(b, 2)), fontsize=24)

    # plt.savefig('result\\pics\\scatter_' + str(name) + '_wide.png')
    # plt.savefig('result\\pics\\scatter_' + str(name) + '.png')

    plt.show()
    # plt.close(1)