# from matplotlib import pyplot as plt
# import os
# import openpyxl
# import pandas as pd
# import numpy as np
import csv
import warnings
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPRegressor  #### MLP 感知机 ####
from sklearn.tree import ExtraTreeRegressor  #### ExtraTree 极端随机树回归####
from sklearn import tree  #### 决策树回归 ####
from sklearn.ensemble import BaggingRegressor  #### Bagging回归 ####
from sklearn.ensemble import AdaBoostRegressor  #### Adaboost回归
from sklearn import linear_model  #### 线性回归####
from sklearn import svm  #### SVM回归####
from sklearn import ensemble  #### Adaboost回归####  ####3.7GBRT回归####  ####3.5随机森林回归####
from sklearn import neighbors  #### KNN回归####

from Model.model_major import *
# from model_lstm import lstmRegressor
# from model_tcn import TCN_LSTM
# from support_nested_lstm import NestedLSTM
# from model_slstm import SLSTM
from Model.model_NBeats import NBeatsNet

from pyhht.emd import EMD
from Support.support_wavelet import *
from Support.support_VMD import VMD

from Part.part_evaluate import *
from Part.part_data_preprocessing import *

warnings.filterwarnings("ignore")


def pre_model(model, trainX, trainY, testX):
    # time_callback = TimeHistory()
    model.fit(trainX, trainY)
    # print(time_callback.totaltime)

    predict = model.predict(testX)
    return predict


########################################################################


def load_data_ts(trainNum, testNum, startNum, data):
    print('General_data loading.')

    global ahead_num
    # all_data_checked = data

    targetData = data

    # 处理预测信息，划分训练集和测试集
    # PM = targetData[startNum : startNum + trainNum + testNum]
    # PM = np.array(PM).reshape(-1, 1)
    targetData = targetData[startNum + 1: startNum + trainNum + testNum + 1]
    targetData = np.array(targetData).reshape(-1, 1)

    # #归一化，每个特征分开归一化
    global scaler_target
    scaler_target = StandardScaler(copy=True, with_mean=True, with_std=True)
    targetData = scaler_target.fit_transform(targetData)
    # print("targetData:", targetData.shape)

    # scaler_PM = StandardScaler(copy=True, with_mean=True, with_std=True)
    # PM = scaler_PM.fit_transform(PM)
    # print("PM:", PM.shape)

    global x_mode

    time_series_y = create_time_series(targetData, ahead_num)
    allX = np.c_[time_series_y]
    allX = allX.T

    ###########======================================

    trainX = allX[:, : trainNum]
    trainY = targetData.T[:, ahead_num: trainNum + ahead_num]
    testX = allX[:, trainNum:]
    testY = targetData.T[:, trainNum + ahead_num: (trainNum + testNum)]

    # print("allX:", allX.shape)
    # print("trainX:", trainX.shape)
    # print("trainY:", trainY.shape)
    # print("testX:", testX.shape)
    # print("testY:", testY.shape)

    trainY = trainY.flatten()  # 降维
    testY = testY.flatten()  # 降维
    trainX = trainX.T
    testX = testX.T

    print('load_data complete.\n')

    return trainX, trainY, testX, testY


def load_data_emd(trainNum, testNum, startNum, data):
    print('EMD_data loading.')

    global ahead_num
    # all_data_checked = data

    targetData = data

    # 处理预测信息，划分训练集和测试集
    targetData = targetData[startNum + 1: startNum + trainNum + testNum + 1]
    targetData = np.array(targetData).reshape(-1, 1)

    # #归一化，每个特征分开归一化
    global scaler_target
    scaler_target = StandardScaler(copy=True, with_mean=True, with_std=True)
    targetData = scaler_target.fit_transform(targetData)

    decomposer = EMD(targetData)
    imfs = decomposer.decompose()
    # plot_imfs(targetData, imfs)
    data_decomposed = imfs.tolist()

    for h1 in range(len(data_decomposed)):
        data_decomposed[h1] = np.array(data_decomposed[h1]).reshape(-1, 1)
    for h2 in range(len(data_decomposed)):
        trainX, trainY, testX, testY = create_data(data_decomposed[h2], trainNum, ahead_num)
        dataset_imf = [trainX, trainY, testX, testY]
        data_decomposed[h2] = dataset_imf

    print('load_data complete.\n')

    return data_decomposed


def load_data_wvlt(trainNum, testNum, startNum, data):
    print('wavelet_data loading.')

    global ahead_num
    # all_data_checked = data
    targetData = data

    # 处理预测信息，划分训练集和测试集
    targetData = targetData[startNum + 1: startNum + trainNum + testNum + 1]
    targetData = np.array(targetData).reshape(-1, 1)

    # #归一化，每个特征分开归一化
    global scaler_target
    scaler_target = StandardScaler(copy=True, with_mean=True, with_std=True)
    targetData = scaler_target.fit_transform(targetData)

    testY = targetData[trainNum: (trainNum + testNum), :]
    wavefun = pywt.Wavelet('db1')
    global wvlt_lv

    coeffs = swt_decom(targetData, wavefun, wvlt_lv)

    # Dict_save['Wavelet_transform'] = [targetData,
    #                                   coeffs[0],
    #                                   coeffs[1],
    #                                   coeffs[2],
    #                                   coeffs[3]]
    #
    # plt.figure(figsize=(15, 12))
    # plt.title("Wavelet transform")
    # # plt.subplot(n_rows, n_cols, plot_num)
    # plt.subplot(5, 1, 1)
    # plt.plot(targetData[:1000, :], color='purple')
    #
    # plt.subplot(512)
    # plt.plot(coeffs[0][:1000, :], color='purple')
    #
    # plt.subplot(513)
    # plt.plot(coeffs[1][:1000, :], color='purple')
    #
    # plt.subplot(514)
    # plt.plot(coeffs[2][:1000, :], color='purple')
    #
    # plt.subplot(515)
    # plt.plot(coeffs[3][:1000, :], color='purple')
    #
    # plt.show()

    # imfs = targetData.T
    # for xx in range(len(coeffs)):
    #     imfs = np.r_[imfs, coeffs[xx].T]
    # plot_imfs(targetData, imfs)

    ### 测试滤波效果
    wvlt_level_list = []
    for wvlt_level in range(len(coeffs)):
        wvlt_trainX, wvlt_trainY, wvlt_testX, wvlt_testY = create_data(coeffs[wvlt_level], trainNum, ahead_num)
        wvlt_level_part = [wvlt_trainX, wvlt_trainY, wvlt_testX, wvlt_testY]
        wvlt_level_list.append(wvlt_level_part)

    print('load_data complete.\n')

    return wvlt_level_list, testY


#########################################################################


def Decide_Tree(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'decideTree'
    model_name_short = 'dTr'
    print(model_name + ' Start.')

    model_DecisionTreeRegressor = tree.DecisionTreeRegressor()  # 决策树
    predict_decideTree = pre_model(model_DecisionTreeRegressor, x_train, y_train, x_test)
    predict_decideTree = scaler_target.inverse_transform(predict_decideTree)
    flag_decideTree = deal_flag(predict_decideTree, minLen)
    accuracy_decideTree = deal_accuracy(y_rampflag, flag_decideTree)

    # mae_decideTree = MAE1(dataY, predict_decideTree)
    # rmse_decideTree = RMSE1(dataY, predict_decideTree)
    # mape_decideTree = MAPE1(dataY, predict_decideTree)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_decideTree: {}'.format(mae_decideTree)
    # eva_output += '\nrmse_decideTree: {}'.format(rmse_decideTree)
    # eva_output += '\nmape_decideTree: {}'.format(mape_decideTree)
    # eva_output += '\naccuracy_decideTree: {}'.format(accuracy_decideTree)
    #
    # result_all.append(['dTr', mae_decideTree, rmse_decideTree, mape_decideTree, accuracy_decideTree])

    global eva_output, result_all
    result_print, result_csv = Evaluate(model_name, model_name_short, dataY, predict_decideTree, accuracy_decideTree)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_decideTree


def Random_forest(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'randomForest'
    model_name_short = 'rdF'
    print(model_name + ' Start.')

    model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=50)  # 随机森林
    predict_randomForest = pre_model(model_RandomForestRegressor, x_train, y_train, x_test)
    predict_randomForest = scaler_target.inverse_transform(predict_randomForest)
    flag_randomForest = deal_flag(predict_randomForest, minLen)
    accuracy_randomForest = deal_accuracy(y_rampflag, flag_randomForest)

    # rmse_randomForest = RMSE1(dataY, predict_randomForest)
    # mape_randomForest = MAPE1(dataY, predict_randomForest)
    # mae_randomForest = MAE1(dataY, predict_randomForest)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_randomForest: {}'.format(mae_randomForest)
    # eva_output += '\nrmse_randomForest: {}'.format(rmse_randomForest)
    # eva_output += '\nmape_randomForest: {}'.format(mape_randomForest)
    # eva_output += '\naccuracy_randomForest: {}'.format(accuracy_randomForest)
    # result_all.append(['rdF', mae_randomForest, rmse_randomForest, mape_randomForest, accuracy_randomForest])

    global eva_output, result_all
    result_print, result_csv = Evaluate(model_name, model_name_short, dataY, predict_randomForest,
                                        accuracy_randomForest)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_randomForest


def SVR(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'svr'
    model_name_short = 'svr'
    print(model_name + ' Start.')

    model_SVR = svm.SVR()  # SVR回归
    predict_svr = pre_model(model_SVR, x_train, y_train, x_test)
    predict_svr = scaler_target.inverse_transform(predict_svr)
    flag_svr = deal_flag(predict_svr, minLen)
    accuracy_svr = deal_accuracy(y_rampflag, flag_svr)

    # rmse_svr = RMSE1(dataY, predict_svr)
    # mape_svr = MAPE1(dataY, predict_svr)
    # mae_svr = MAE1(dataY, predict_svr)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_svr: {}'.format(mae_svr)
    # eva_output += '\nrmse_svr: {}'.format(rmse_svr)
    # eva_output += '\nmape_svr: {}'.format(mape_svr)
    # eva_output += '\naccuracy_svr: {}'.format(accuracy_svr)
    # result_all.append(['svr', mae_svr, rmse_svr, mape_svr, accuracy_svr])

    global eva_output, result_all
    result_print, result_csv = Evaluate(model_name, model_name_short, dataY, predict_svr, accuracy_svr)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_svr


def MLP(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'mlp'
    model_name_short = 'mlp'
    print(model_name + ' Start.')

    model_MLP = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(20, 20, 20), random_state=2)  # MLP
    predict_mlp = pre_model(model_MLP, x_train, y_train, x_test)
    predict_mlp = scaler_target.inverse_transform(predict_mlp)
    flag_mlp = deal_flag(predict_mlp, minLen)
    accuracy_mlp = deal_accuracy(y_rampflag, flag_mlp)

    # rmse_mlp = RMSE1(dataY, predict_mlp)
    # mape_mlp = MAPE1(dataY, predict_mlp)
    # mae_mlp = MAE1(dataY, predict_mlp)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_mlp: {}'.format(mae_mlp)
    # eva_output += '\nrmse_mlp: {}'.format(rmse_mlp)
    # eva_output += '\nmape_mlp: {}'.format(mape_mlp)
    # eva_output += '\naccuracy_mlp: {}'.format(accuracy_mlp)
    # result_all.append(['mlp', mae_mlp, rmse_mlp, mape_mlp, accuracy_mlp])

    global eva_output, result_all
    result_print, result_csv = Evaluate(model_name,
                                        model_name_short,
                                        dataY,
                                        predict_mlp,
                                        accuracy_mlp)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_mlp


def gradient_Boosting(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'gradientBoosting'
    model_name_short = 'grB'
    print(model_name + ' Start.')

    model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=50)  # GDBT
    predict_gradientBoosting = pre_model(model_GradientBoostingRegressor, x_train, y_train, x_test)
    predict_gradientBoosting = scaler_target.inverse_transform(predict_gradientBoosting)
    flag_gradientBoosting = deal_flag(predict_gradientBoosting, minLen)
    accuracy_gradientBoosting = deal_accuracy(y_rampflag, flag_gradientBoosting)

    # rmse_gradientBoosting = RMSE1(dataY, predict_gradientBoosting)
    # mape_gradientBoosting = MAPE1(dataY, predict_gradientBoosting)
    # mae_gradientBoosting = MAE1(dataY, predict_gradientBoosting)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_gradientBoosting: {}'.format(mae_gradientBoosting)
    # eva_output += '\nrmse_gradientBoosting: {}'.format(rmse_gradientBoosting)
    # eva_output += '\nmape_gradientBoosting: {}'.format(mape_gradientBoosting)
    # eva_output += '\naccuracy_gradientBoosting: {}'.format(accuracy_gradientBoosting)
    # result_all.append(
    #     ['grB', mae_gradientBoosting, rmse_gradientBoosting, mape_gradientBoosting, accuracy_gradientBoosting])

    global eva_output, result_all
    result_print, result_csv = Evaluate(model_name,
                                        model_name_short,
                                        dataY,
                                        predict_gradientBoosting,
                                        accuracy_gradientBoosting)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_gradientBoosting


#########################################################################


def LSTM(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'LSTM'
    model_name_short = 'LSTM'
    print(model_name + ' Start.')

    feature_num = x_train.shape[1]
    x_train_lstm = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test_lstm = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Modelling and Predicting
    print('Build model...')

    model = build_LSTM(feature_num)
    # model = buildOLSTM(feature_num)

    global show_process
    time_callback = TimeHistory()
    history = model.fit(x_train_lstm, y_train, epochs=16, validation_split=0.05, verbose=show_process,
                        callbacks=[time_callback])
    train_time = time_callback.totaltime

    predict_lstm = model.predict(x_test_lstm)

    predict_lstm = predict_lstm.reshape(-1, )
    predict_lstm = scaler_target.inverse_transform(predict_lstm)
    flag_lstm = deal_flag(predict_lstm, minLen)
    accuracy_lstm = deal_accuracy(y_rampflag, flag_lstm)

    # rmse_nlstm = RMSE1(dataY, predict_nlstm)
    # mape_nlstm = MAPE1(dataY, predict_nlstm)
    # mae_nlstm = MAE1(dataY, predict_nlstm)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_nlstm: {}'.format(mae_nlstm)
    # eva_output += '\nrmse_nlstm: {}'.format(rmse_nlstm)
    # eva_output += '\nmape_nlstm: {}'.format(mape_nlstm)
    # eva_output += '\naccuracy_nlstm: {}'.format(accuracy_nlstm)
    # result_all.append(['nlstm', mae_nlstm, rmse_nlstm, mape_nlstm, accuracy_nlstm])

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_lstm,
                                           accuracy_lstm,
                                           train_time)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_lstm


def EMD_LSTM(emd_list, y_rampflag, dataY, emd_decay_rate):
    model_name = 'EMD'
    model_name_short = 'EMD'
    print(model_name + ' Start.')

    model = build_LSTM(ahead_num)

    global show_process

    predict_emd_list = []
    train_time_total = 0
    for ie in range(len(emd_list)):
        # emd_list[ie][0] = np.reshape(emd_list[ie][0], (emd_list[ie][0].shape[0], emd_list[ie][0].shape[1], 1))
        # emd_list[ie][2] = np.reshape(emd_list[ie][2], (emd_list[ie][2].shape[0], emd_list[ie][2].shape[1], 1))
        EMD_trX = np.reshape(emd_list[ie][0], (emd_list[ie][0].shape[0], emd_list[ie][0].shape[1], 1))
        EMD_teX = np.reshape(emd_list[ie][2], (emd_list[ie][2].shape[0], emd_list[ie][2].shape[1], 1))
        EMD_trY = emd_list[ie][1]

        time_callback = TimeHistory()
        model.fit(EMD_trX, EMD_trY, epochs=16, validation_split=0.05, verbose=show_process, callbacks=[time_callback])
        train_time = time_callback.totaltime
        train_time_total = train_time_total + train_time
        # model.fit(emd_list[ie][0], emd_list[ie][1], epochs=16, verbose=0, callbacks=[TQDMCallback()])
        predict_emd_part = model.predict(EMD_teX)
        predict_emd_part = predict_emd_part.reshape(-1, )
        predict_emd_list.append(predict_emd_part)
        print('EMD_imf ' + str(ie + 1) + ' Complete.')

    # wavefun = pywt.Wavelet('db1')
    # predict_EN = iswt_decom(predict_emd_list, wavefun)

    predict_EMD = predict_emd_list[-1]
    for i in range(len(predict_emd_list) - 1):
        predict_EMD = predict_EMD + predict_emd_list[i] * emd_decay_rate
    # predict_noise = predict_EN - predict_emd_list[-1]

    predict_EMD = scaler_target.inverse_transform(predict_EMD)
    flag_EMD = deal_flag(predict_EMD, minLen)
    accuracy_EMD = deal_accuracy(y_rampflag, flag_EMD)

    # rmse_EN = RMSE1(dataY, predict_EN)
    # mape_EN = MAPE1(dataY, predict_EN)
    # mae_EN = MAE1(dataY, predict_EN)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_EN: {}'.format(mae_EN)
    # eva_output += '\nrmse_EN: {}'.format(rmse_EN)
    # eva_output += '\nmape_EN: {}'.format(mape_EN)
    # eva_output += '\naccuracy_EN: {}'.format(accuracy_EN)
    # result_all.append(['EN', mae_EN, rmse_EN, mape_EN, accuracy_EN])

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_EMD,
                                           accuracy_EMD,
                                           train_time_total)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_EMD


def Wavelet_LSTM(wvlt_list, y_rampflag, dataY):
    model_name = 'Wvlt'
    model_name_short = 'Wvlt'
    print(model_name + ' Start.')

    model = build_LSTM(ahead_num)

    global show_process

    train_time_total = 0
    predict_Wvlt_list = []
    for i_wvlt in range(len(wvlt_list)):
        wvlt_trX = np.reshape(wvlt_list[i_wvlt][0],
                              (wvlt_list[i_wvlt][0].shape[0],
                               wvlt_list[i_wvlt][0].shape[1], 1))
        wvlt_teX = np.reshape(wvlt_list[i_wvlt][2],
                              (wvlt_list[i_wvlt][2].shape[0],
                               wvlt_list[i_wvlt][2].shape[1], 1))
        wvlt_trY = wvlt_list[i_wvlt][1]

        time_callback = TimeHistory()
        model.fit(wvlt_trX, wvlt_trY, epochs=16, validation_split=0.05, verbose=show_process, callbacks=[time_callback])
        # model.fit(wvlt_list[i_wvlt][0], wvlt_list[i_wvlt][1], epochs=16, verbose=0, callbacks=[TQDMCallback()])
        train_time = time_callback.totaltime
        train_time_total = train_time_total + train_time

        predict_Wvlt = model.predict(wvlt_teX)
        predict_Wvlt = predict_Wvlt.reshape(-1, )
        predict_Wvlt_list.append(predict_Wvlt)
        print('wvlt_level ' + str(i_wvlt + 1) + ' Complete.')

    wavefun = pywt.Wavelet('db1')

    predict_Wvlt = iswt_decom(predict_Wvlt_list, wavefun)
    predict_Wvlt = scaler_target.inverse_transform(predict_Wvlt)
    flag_Wvlt = deal_flag(predict_Wvlt, minLen)
    accuracy_Wvlt = deal_accuracy(y_rampflag, flag_Wvlt)

    # # dataY1 = scaler_target.inverse_transform(dataY1)
    # # dataY1 = dataY1.T.tolist()[0]
    # rmse_WN = RMSE1(dataY, predict_WN)
    # mape_WN = MAPE1(dataY, predict_WN)
    # mae_WN = MAE1(dataY, predict_WN)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_WN: {}'.format(mae_WN)
    # eva_output += '\nrmse_WN: {}'.format(rmse_WN)
    # eva_output += '\nmape_WN: {}'.format(mape_WN)
    # eva_output += '\naccuracy_WN: {}'.format(accuracy_WN)
    # result_all.append(['WN', mae_WN, rmse_WN, mape_WN, accuracy_WN])

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_Wvlt,
                                           accuracy_Wvlt,
                                           train_time_total)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_Wvlt


#########################################################################


def Stacked_LSTM(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'SLSTM'
    model_name_short = 'SLSTM'
    print(model_name + ' Start.')

    feature_num = x_train.shape[1]
    x_train_slstm = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test_slstm = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Modelling and Predicting
    print('Build model...')

    model = buildSLSTM(feature_num)
    # model = buildOLSTM(feature_num)

    global show_process

    time_callback = TimeHistory()
    history = model.fit(x_train_slstm, y_train, epochs=16, validation_split=0.05, verbose=show_process,
                        callbacks=[time_callback])
    train_time = time_callback.totaltime

    predict_slstm = model.predict(x_test_slstm)

    # history = model.fit(x_train, y_train, epochs=16, validation_split=0.05, verbose=2)
    #
    # predict_slstm = model.predict(x_test)
    predict_slstm = predict_slstm.reshape(-1, )
    predict_slstm = scaler_target.inverse_transform(predict_slstm)
    flag_slstm = deal_flag(predict_slstm, minLen)
    accuracy_slstm = deal_accuracy(y_rampflag, flag_slstm)

    # rmse_slstm = RMSE1(dataY, predict_slstm)
    # mape_slstm = MAPE1(dataY, predict_slstm)
    # mae_slstm = MAE1(dataY, predict_slstm)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_slstm: {}'.format(mae_slstm)
    # eva_output += '\nrmse_slstm: {}'.format(rmse_slstm)
    # eva_output += '\nmape_slstm: {}'.format(mape_slstm)
    # eva_output += '\naccuracy_slstm: {}'.format(accuracy_slstm)
    # result_all.append(['slstm', mae_slstm, rmse_slstm, mape_slstm, accuracy_slstm])

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_slstm,
                                           accuracy_slstm,
                                           train_time)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_slstm


def EMD_SLSTM(emd_list, y_rampflag, dataY, emd_decay_rate):
    model_name = 'ES'
    model_name_short = 'ES'
    print(model_name + ' Start.')

    model = buildSLSTM(ahead_num)

    global show_process

    predict_emd_list = []
    train_time_total = 0
    for ie in range(len(emd_list)):
        # emd_list[ie][0] = np.reshape(emd_list[ie][0], (emd_list[ie][0].shape[0], emd_list[ie][0].shape[1], 1))
        # emd_list[ie][2] = np.reshape(emd_list[ie][2], (emd_list[ie][2].shape[0], emd_list[ie][2].shape[1], 1))
        EMD_trX = np.reshape(emd_list[ie][0], (emd_list[ie][0].shape[0], emd_list[ie][0].shape[1], 1))
        EMD_teX = np.reshape(emd_list[ie][2], (emd_list[ie][2].shape[0], emd_list[ie][2].shape[1], 1))
        EMD_trY = emd_list[ie][1]
        time_callback = TimeHistory()
        model.fit(EMD_trX, EMD_trY, epochs=16, validation_split=0.05, verbose=show_process, callbacks=[time_callback])
        train_time = time_callback.totaltime
        train_time_total = train_time_total + train_time
        # model.fit(emd_list[ie][0], emd_list[ie][1], epochs=16, verbose=0, callbacks=[TQDMCallback()])
        predict_emd_part = model.predict(EMD_teX)
        predict_emd_part = predict_emd_part.reshape(-1, )
        predict_emd_list.append(predict_emd_part)
        print('EMD_imf ' + str(ie + 1) + ' Complete.')

    # wavefun = pywt.Wavelet('db1')
    # predict_ES = iswt_decom(predict_emd_list, wavefun)

    predict_ES = predict_emd_list[-1]
    for i in range(len(predict_emd_list) - 1):
        predict_ES = predict_ES + predict_emd_list[i] * emd_decay_rate
    # predict_noise = predict_ES - predict_emd_list[-1]

    predict_ES = scaler_target.inverse_transform(predict_ES)
    flag_ES = deal_flag(predict_ES, minLen)
    accuracy_ES = deal_accuracy(y_rampflag, flag_ES)

    # rmse_ES = RMSE1(dataY, predict_ES)
    # mape_ES = MAPE1(dataY, predict_ES)
    # mae_ES = MAE1(dataY, predict_ES)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_ES: {}'.format(mae_ES)
    # eva_output += '\nrmse_ES: {}'.format(rmse_ES)
    # eva_output += '\nmape_ES: {}'.format(mape_ES)
    # eva_output += '\naccuracy_ES: {}'.format(accuracy_ES)
    # result_all.append(['EN', mae_ES, rmse_ES, mape_ES, accuracy_ES])

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_ES,
                                           accuracy_ES,
                                           train_time_total)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_ES


def Wavelet_SLSTM(wvlt_list, y_rampflag, dataY):
    model_name = 'WS'
    model_name_short = 'WS'
    print(model_name + ' Start.')

    model = buildSLSTM(ahead_num)

    global show_process

    train_time_total = 0
    predict_WS_list = []
    for i_wvlt in range(len(wvlt_list)):
        wvlt_trX = np.reshape(wvlt_list[i_wvlt][0],
                              (wvlt_list[i_wvlt][0].shape[0],
                               wvlt_list[i_wvlt][0].shape[1], 1))
        wvlt_teX = np.reshape(wvlt_list[i_wvlt][2],
                              (wvlt_list[i_wvlt][2].shape[0],
                               wvlt_list[i_wvlt][2].shape[1], 1))
        wvlt_trY = wvlt_list[i_wvlt][1]

        time_callback = TimeHistory()
        model.fit(wvlt_trX, wvlt_trY, epochs=16, validation_split=0.05, verbose=show_process, callbacks=[time_callback])
        # model.fit(wvlt_list[i_wvlt][0], wvlt_list[i_wvlt][1], epochs=16, verbose=0, callbacks=[TQDMCallback()])
        train_time = time_callback.totaltime
        train_time_total = train_time_total + train_time

        predict_WS = model.predict(wvlt_teX)
        predict_WS = predict_WS.reshape(-1, )
        predict_WS_list.append(predict_WS)
        print('wvlt_level ' + str(i_wvlt) + ' Complete.')

    wavefun = pywt.Wavelet('db1')

    predict_WS = iswt_decom(predict_WS_list, wavefun)
    predict_WS = scaler_target.inverse_transform(predict_WS)
    flag_WS = deal_flag(predict_WS, minLen)
    accuracy_WS = deal_accuracy(y_rampflag, flag_WS)

    # # dataY1 = scaler_target.inverse_transform(dataY1)
    # # dataY1 = dataY1.T.tolist()[0]
    # rmse_WS = RMSE1(dataY, predict_WS)
    # mape_WS = MAPE1(dataY, predict_WS)
    # mae_WS = MAE1(dataY, predict_WS)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_WS: {}'.format(mae_WS)
    # eva_output += '\nrmse_WS: {}'.format(rmse_WS)
    # eva_output += '\nmape_WS: {}'.format(mape_WS)
    # eva_output += '\naccuracy_WS: {}'.format(accuracy_WS)
    # result_all.append(['WN', mae_WS, rmse_WS, mape_WS, accuracy_WS])

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_WS,
                                           accuracy_WS,
                                           train_time_total)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_WS


#########################################################################


def Nested_LSTM(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'NLSTM'
    model_name_short = 'NLSTM'
    print('NLSTM Begin.')

    feature_num = x_train.shape[1]
    x_train_nlstm = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test_nlstm = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Modelling and Predicting
    print('Build model...')

    model = buildNLSTM(feature_num)
    # model = buildOLSTM(feature_num)

    global show_process

    time_callback = TimeHistory()
    history = model.fit(x_train_nlstm, y_train, epochs=16, validation_split=0.05, verbose=show_process,
                        callbacks=[time_callback])
    train_time = time_callback.totaltime

    predict_nlstm = model.predict(x_test_nlstm)

    # history = model.fit(x_train, y_train, epochs=16, validation_split=0.05, verbose=2)
    #
    # predict_nlstm = model.predict(x_test)
    predict_nlstm = predict_nlstm.reshape(-1, )
    predict_nlstm = scaler_target.inverse_transform(predict_nlstm)
    flag_nlstm = deal_flag(predict_nlstm, minLen)
    accuracy_nlstm = deal_accuracy(y_rampflag, flag_nlstm)

    # rmse_nlstm = RMSE1(dataY, predict_nlstm)
    # mape_nlstm = MAPE1(dataY, predict_nlstm)
    # mae_nlstm = MAE1(dataY, predict_nlstm)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_nlstm: {}'.format(mae_nlstm)
    # eva_output += '\nrmse_nlstm: {}'.format(rmse_nlstm)
    # eva_output += '\nmape_nlstm: {}'.format(mape_nlstm)
    # eva_output += '\naccuracy_nlstm: {}'.format(accuracy_nlstm)
    # result_all.append(['nlstm', mae_nlstm, rmse_nlstm, mape_nlstm, accuracy_nlstm])

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_nlstm,
                                           accuracy_nlstm,
                                           train_time)
    eva_output += result_print
    result_all.append(result_csv)

    print('NLSTM Complete.')
    return predict_nlstm


def EMD_NLSTM(emd_list, y_rampflag, dataY, emd_decay_rate):
    model_name = 'EN'
    model_name_short = 'EN'
    print(model_name + ' Start.')

    model = buildNLSTM(ahead_num)

    global show_process

    predict_emd_list = []
    train_time_total = 0
    for ie in range(len(emd_list)):
        # emd_list[ie][0] = np.reshape(emd_list[ie][0], (emd_list[ie][0].shape[0], emd_list[ie][0].shape[1], 1))
        # emd_list[ie][2] = np.reshape(emd_list[ie][2], (emd_list[ie][2].shape[0], emd_list[ie][2].shape[1], 1))
        EMD_trX = np.reshape(emd_list[ie][0], (emd_list[ie][0].shape[0], emd_list[ie][0].shape[1], 1))
        EMD_teX = np.reshape(emd_list[ie][2], (emd_list[ie][2].shape[0], emd_list[ie][2].shape[1], 1))
        EMD_trY = emd_list[ie][1]
        time_callback = TimeHistory()
        model.fit(EMD_trX, EMD_trY, epochs=16, validation_split=0.05, verbose=show_process, callbacks=[time_callback])
        train_time = time_callback.totaltime
        train_time_total = train_time_total + train_time
        # model.fit(emd_list[ie][0], emd_list[ie][1], epochs=16, verbose=0, callbacks=[TQDMCallback()])
        predict_emd_part = model.predict(EMD_teX)
        predict_emd_part = predict_emd_part.reshape(-1, )
        predict_emd_list.append(predict_emd_part)
        print('EMD_imf ' + str(ie + 1) + ' Complete.')

    # wavefun = pywt.Wavelet('db1')
    # predict_EN = iswt_decom(predict_emd_list, wavefun)

    predict_EN = predict_emd_list[-1]
    for i in range(len(predict_emd_list) - 1):
        predict_EN = predict_EN + predict_emd_list[i] * emd_decay_rate
    # predict_noise = predict_EN - predict_emd_list[-1]

    predict_EN = scaler_target.inverse_transform(predict_EN)
    flag_EN = deal_flag(predict_EN, minLen)
    accuracy_EN = deal_accuracy(y_rampflag, flag_EN)

    # rmse_EN = RMSE1(dataY, predict_EN)
    # mape_EN = MAPE1(dataY, predict_EN)
    # mae_EN = MAE1(dataY, predict_EN)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_EN: {}'.format(mae_EN)
    # eva_output += '\nrmse_EN: {}'.format(rmse_EN)
    # eva_output += '\nmape_EN: {}'.format(mape_EN)
    # eva_output += '\naccuracy_EN: {}'.format(accuracy_EN)
    # result_all.append(['EN', mae_EN, rmse_EN, mape_EN, accuracy_EN])

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_EN,
                                           accuracy_EN,
                                           train_time_total)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_EN


def Wavelet_NLSTM(wvlt_list, y_rampflag, dataY):
    model_name = 'WN'
    model_name_short = 'WN'
    print(model_name + ' Start.')

    model = buildNLSTM(ahead_num)

    global show_process

    train_time_total = 0
    predict_WN_list = []
    test_WN_list = []

    loss_list = []
    val_loss_list = []

    for i_wvlt in range(len(wvlt_list)):
        wvlt_trX = np.reshape(wvlt_list[i_wvlt][0],
                              (wvlt_list[i_wvlt][0].shape[0],
                               wvlt_list[i_wvlt][0].shape[1], 1))
        wvlt_teX = np.reshape(wvlt_list[i_wvlt][2],
                              (wvlt_list[i_wvlt][2].shape[0],
                               wvlt_list[i_wvlt][2].shape[1], 1))
        wvlt_trY = wvlt_list[i_wvlt][1]

        time_callback = TimeHistory()
        history = model.fit(wvlt_trX, wvlt_trY, epochs=16, validation_split=0.05, verbose=show_process, callbacks=[time_callback])
        # model.fit(wvlt_list[i_wvlt][0], wvlt_list[i_wvlt][1], epochs=16, verbose=0, callbacks=[TQDMCallback()])
        train_time = time_callback.totaltime
        train_time_total = train_time_total + train_time

        predict_WN = model.predict(wvlt_teX)
        predict_WN = predict_WN.reshape(-1, )
        predict_WN_list.append(predict_WN)

        wvlt_teY = wvlt_list[i_wvlt][3].reshape(-1,)
        test_WN_list.append(wvlt_teY)

        print('wvlt_level ' + str(i_wvlt + 1) + ' Complete.')

        loss_list.append(history.history['loss'])
        val_loss_list.append(history.history['val_loss'])

    wavefun = pywt.Wavelet('db1')

    predict_WN = iswt_decom(predict_WN_list, wavefun)
    predict_WN = scaler_target.inverse_transform(predict_WN)
    flag_WN = deal_flag(predict_WN, minLen)
    accuracy_WN = deal_accuracy(y_rampflag, flag_WN)

    # # dataY1 = scaler_target.inverse_transform(dataY1)
    # # dataY1 = dataY1.T.tolist()[0]
    # rmse_WN = RMSE1(dataY, predict_WN)
    # mape_WN = MAPE1(dataY, predict_WN)
    # mae_WN = MAE1(dataY, predict_WN)
    #
    # global eva_output, result_all
    # eva_output += '\n\nmae_WN: {}'.format(mae_WN)
    # eva_output += '\nrmse_WN: {}'.format(rmse_WN)
    # eva_output += '\nmape_WN: {}'.format(mape_WN)
    # eva_output += '\naccuracy_WN: {}'.format(accuracy_WN)
    # result_all.append(['WN', mae_WN, rmse_WN, mape_WN, accuracy_WN])

    # plt.figure(figsize=(15, 10))
    #
    # plt.subplot(1, 4, 1)
    # plt.title('A3')
    # plt.plot(loss_list[0], label="loss", color='orange', marker='o')
    # # plt.plot(val_loss_list[0], label="val_loss", color='purple', marker='o')
    # plt.legend(loc='best')
    #
    # plt.subplot(1, 4, 2)
    # plt.title('D1')
    # plt.plot(loss_list[1], label="loss", color='orange', marker='o')
    # # plt.plot(val_loss_list[1], label="val_loss", color='purple', marker='o')
    # plt.legend(loc='best')
    #
    # plt.subplot(1, 4, 3)
    # plt.title('D2')
    # plt.plot(loss_list[2], label="loss", color='orange', marker='o')
    # # plt.plot(val_loss_list[2], label="val_loss", color='purple', marker='o')
    # plt.legend(loc='best')
    #
    # plt.subplot(1, 4, 4)
    # plt.title('D3')
    # plt.plot(loss_list[3], label="loss", color='orange', marker='o')
    # # plt.plot(val_loss_list[3], label="val_loss", color='purple', marker='o')
    # plt.legend(loc='best')
    #
    # plt.show()

    Dict_save['loss_list'] = loss_list
    Dict_save['val_loss_list'] = val_loss_list

    Dict_save['WN_sub_teY'] = test_WN_list
    Dict_save['WN_sub_predict'] = predict_WN_list

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_WN,
                                           accuracy_WN,
                                           train_time_total)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_WN


#########################################################################


def Bidirectional_LSTM(x_train, y_train, x_test, y_test, y_rampflag, dataY):
    model_name = 'BLSTM'
    model_name_short = 'BLSTM'
    print('BLSTM Begin.')

    feature_num = x_train.shape[1]
    x_train_blstm = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test_blstm = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Modelling and Predicting
    print('Build model...')

    model = buildBLSTM(feature_num)
    # model = buildOLSTM(feature_num)

    global show_process

    time_callback = TimeHistory()
    history = model.fit(x_train_blstm, y_train, epochs=16, validation_split=0.05, verbose=show_process,
                        callbacks=[time_callback])
    train_time = time_callback.totaltime

    predict_blstm = model.predict(x_test_blstm)

    # history = model.fit(x_train, y_train, epochs=16, validation_split=0.05, verbose=2)
    #
    # predict_blstm = model.predict(x_test)
    predict_blstm = predict_blstm.reshape(-1, )
    predict_blstm = scaler_target.inverse_transform(predict_blstm)
    flag_blstm = deal_flag(predict_blstm, minLen)
    accuracy_blstm = deal_accuracy(y_rampflag, flag_blstm)

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_blstm,
                                           accuracy_blstm,
                                           train_time)
    eva_output += result_print
    result_all.append(result_csv)

    print('BLSTM Complete.')
    return predict_blstm


def EMD_BLSTM(emd_list, y_rampflag, dataY, emd_decay_rate):
    model_name = 'EB'
    model_name_short = 'EB'
    print(model_name + ' Start.')

    model = buildBLSTM(ahead_num)

    global show_process

    predict_emd_list = []
    train_time_total = 0
    for ie in range(len(emd_list)):
        EMD_trX = np.reshape(emd_list[ie][0], (emd_list[ie][0].shape[0], emd_list[ie][0].shape[1], 1))
        EMD_teX = np.reshape(emd_list[ie][2], (emd_list[ie][2].shape[0], emd_list[ie][2].shape[1], 1))
        EMD_trY = emd_list[ie][1]
        time_callback = TimeHistory()
        model.fit(EMD_trX, EMD_trY, epochs=16, validation_split=0.05, verbose=show_process, callbacks=[time_callback])
        train_time = time_callback.totaltime
        train_time_total = train_time_total + train_time
        # model.fit(emd_list[ie][0], emd_list[ie][1], epochs=16, verbose=0, callbacks=[TQDMCallback()])
        predict_emd_part = model.predict(EMD_teX)
        predict_emd_part = predict_emd_part.reshape(-1, )
        predict_emd_list.append(predict_emd_part)
        print('EMD_imf ' + str(ie + 1) + ' Complete.')

    predict_EB = predict_emd_list[-1]
    for i in range(len(predict_emd_list) - 1):
        predict_EB = predict_EB + predict_emd_list[i] * emd_decay_rate

    predict_EB = scaler_target.inverse_transform(predict_EB)
    flag_EB = deal_flag(predict_EB, minLen)
    accuracy_EB = deal_accuracy(y_rampflag, flag_EB)

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_EB,
                                           accuracy_EB,
                                           train_time_total)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_EB


def Wavelet_BLSTM(wvlt_list, y_rampflag, dataY):
    model_name = 'WB'
    model_name_short = 'WB'
    print(model_name + ' Start.')

    model = buildBLSTM(ahead_num)

    global show_process

    train_time_total = 0
    predict_WB_list = []

    for i_wvlt in range(len(wvlt_list)):
        wvlt_trX = np.reshape(wvlt_list[i_wvlt][0],
                              (wvlt_list[i_wvlt][0].shape[0],
                               wvlt_list[i_wvlt][0].shape[1], 1))
        wvlt_teX = np.reshape(wvlt_list[i_wvlt][2],
                              (wvlt_list[i_wvlt][2].shape[0],
                               wvlt_list[i_wvlt][2].shape[1], 1))
        wvlt_trY = wvlt_list[i_wvlt][1]

        time_callback = TimeHistory()
        history = model.fit(wvlt_trX, wvlt_trY, epochs=16, validation_split=0.05, verbose=show_process, callbacks=[time_callback])
        # model.fit(wvlt_list[i_wvlt][0], wvlt_list[i_wvlt][1], epochs=16, verbose=0, callbacks=[TQDMCallback()])
        train_time = time_callback.totaltime
        train_time_total = train_time_total + train_time

        predict_WB = model.predict(wvlt_teX)
        predict_WB = predict_WB.reshape(-1, )
        predict_WB_list.append(predict_WB)

        print('wvlt_level ' + str(i_wvlt + 1) + ' Complete.')

    wavefun = pywt.Wavelet('db1')

    predict_WB = iswt_decom(predict_WB_list, wavefun)
    predict_WB = scaler_target.inverse_transform(predict_WB)
    flag_WB = deal_flag(predict_WB, minLen)
    accuracy_WB = deal_accuracy(y_rampflag, flag_WB)

    global eva_output, result_all
    result_print, result_csv = Evaluate_DL(model_name,
                                           model_name_short,
                                           dataY,
                                           predict_WB,
                                           accuracy_WB,
                                           train_time_total)
    eva_output += result_print
    result_all.append(result_csv)

    print(model_name + ' Complete.')
    return predict_WB


#########################################################################


def neo_prediction(start_num, interval_ori):

    # random_state = np.random.RandomState(7)
    np.random.RandomState(7)

    #########################################################################

    global Dict_save
    Dict_save = {}

    #########################################################################

    # lookback number
    global ahead_num
    ahead_num = 4

    global interval
    interval = interval_ori

    global minLen
    minLen = 0

    # 1 for features; 2 for timeseries; 3 for feature & timeseries.
    global x_mode
    x_mode = 2

    global wvlt_lv
    wvlt_lv = 3

    global show_process
    show_process = 1

    #########################################################################

    num = 12
    filename1 = "dataset\\PRSA_Data_"
    filename2 = ".csv"
    filename = [filename1, filename2]

    # training number
    startNum = start_num
    trainNum = (24 * 1000) // interval
    testNum = ((24 * 100) // interval) + ahead_num

    emd_decay_rate = 1.00

    #########################################################################

    dataset = read_csv_PM2(filename, trainNum, testNum, startNum, num, interval)

    #########################################################################

    x_train, y_train, x_test, y_test = load_data_ts(trainNum, testNum, startNum, dataset)
    wvlt_list, _ = load_data_wvlt(trainNum, testNum, startNum, dataset)
    emd_list = load_data_emd(trainNum, testNum, startNum, dataset)

    #########################################################################

    # #####culculate Accuracy by rampflag
    global scaler_target
    dataY = scaler_target.inverse_transform(y_test)
    # minLen = np.mean(dataY) * 0.25
    minLen = 0
    print('Accuracy Flag:', minLen)
    y_rampflag = deal_flag(dataY, minLen)

    Dict_save['true_value'] = dataY

    ######=========================Modelling and Predicting=========================#####
    print("======================================================")
    global eva_output, result_all
    eva_output = '\nEvaluation.'
    result_all = []

    # predict_decideTree = Decide_Tree(x_train, y_train, x_test, y_test, y_rampflag, dataY)
    # predict_randomForest = Random_forest(x_train, y_train, x_test, y_test, y_rampflag, dataY)
    # predict_SVR = SVR(x_train, y_train, x_test, y_test, y_rampflag, dataY)
    # predict_mlp = MLP(x_train, y_train, x_test, y_test, y_rampflag, dataY)
    # Dict_save['predict_dTr'] = predict_decideTree
    # Dict_save['predict_RdF'] = predict_randomForest
    # Dict_save['predict_SVR'] = predict_SVR
    # Dict_save['predict_MLP'] = predict_mlp
    #
    # predict_lstm = LSTM(x_train, y_train, x_test, y_test, y_rampflag, dataY)
    # predict_lstm_emd = EMD_LSTM(emd_list, y_rampflag, dataY, emd_decay_rate)
    # predict_wavelet = Wavelet_LSTM(wvlt_list, y_rampflag, dataY)
    # Dict_save['predict_lstm'] = predict_lstm
    # Dict_save['predict_EMD'] = predict_lstm_emd
    # Dict_save['predict_Wvlt'] = predict_wavelet
    #
    # predict_slstm = Stacked_LSTM(x_train, y_train, x_test, y_test, y_rampflag, dataY)
    # predict_ES = EMD_SLSTM(emd_list, y_rampflag, dataY, emd_decay_rate)
    # predict_WS = Wavelet_SLSTM(wvlt_list, y_rampflag, dataY)
    # Dict_save['predict_slstm'] = predict_slstm
    # Dict_save['predict_ES'] = predict_ES
    # Dict_save['predict_WS'] = predict_WS

    # predict_nlstm = Nested_LSTM(x_train, y_train, x_test, y_test, y_rampflag, dataY)
    # predict_EN = EMD_NLSTM(emd_list, y_rampflag, dataY, emd_decay_rate)
    predict_WN = Wavelet_NLSTM(wvlt_list, y_rampflag, dataY)
    # Dict_save['predict_nlstm'] = predict_nlstm
    # Dict_save['predict_EN'] = predict_EN
    Dict_save['predict_WN'] = predict_WN

    # predict_blstm = Bidirectional_LSTM(x_train, y_train, x_test, y_test, y_rampflag, dataY)
    # predict_EB = EMD_BLSTM(emd_list, y_rampflag, dataY, emd_decay_rate)
    # predict_WB = Wavelet_BLSTM(wvlt_list, y_rampflag, dataY)
    # Dict_save['predict_blstm'] = predict_blstm
    # Dict_save['predict_EB'] = predict_EB
    # Dict_save['predict_WB'] = predict_WB

    print(eva_output)

    # # print(result_all)
    save_file_name = "result\\WN_exhi_result.csv"

    csv_file = open(save_file_name, "w", newline="")  # 创建csv文件
    writer = csv.writer(csv_file)  # 创建写的对象
    writer.writerow(["", "MAE", "RMSE", "MAPE", "R2", "ACC", "TIME"])
    for wtr in range(len(result_all)):
        writer.writerow(result_all[wtr])

    # backup for Denied permission.
    save_file_name = "result\\WN_exhi_result_backup.csv"

    csv_file = open(save_file_name, "w", newline="")  # 创建csv文件
    writer = csv.writer(csv_file)  # 创建写的对象
    writer.writerow(["", "MAE", "RMSE", "MAPE", "R2", "ACC", "TIME"])
    for wtr in range(len(result_all)):
        writer.writerow(result_all[wtr])

    print('\nComplete.')

    choice_save = input('Do you want to save this result?(yes/no)\n')
    if choice_save == 'yes':
        print('The result shall be saved.')
        np.save('saved\\saved_data_ts4_allbase.npy', Dict_save)
        print('The result is saved successfully.')
    else:
        print('The result shall not be saved. If you want it saved, enter \'yes\' next time.')

    return result_all


def load_result_evaluation():
    Dict = np.load('saved\\saved_data_ts4_allbase.npy', allow_pickle=True).item()
    print('The result is loaded successfully.')

    #########################################################################

    # coeffs = [None, None, None, None]
    # [targetData, coeffs[0], coeffs[1], coeffs[2], coeffs[3]] = Dict['Wavelet_transform']

    # plt.figure(figsize=(15, 12))
    # # plt.subplot(n_rows, n_cols, plot_num)
    # plt.subplot(5, 1, 1)
    # plt.plot(targetData[:1000, :])
    #
    # plt.subplot(512)
    # plt.plot(coeffs[0][:1000, :])
    #
    # plt.subplot(513)
    # plt.plot(coeffs[1][:1000, :])
    #
    # plt.subplot(514)
    # plt.plot(coeffs[2][:1000, :])
    #
    # plt.subplot(515)
    # plt.plot(coeffs[3][:1000, :])
    #
    # plt.show()

    #########################################################################

    dataY = Dict['true_value']

    predict_decideTree = Dict['predict_dTr']
    predict_randomForest = Dict['predict_RdF']
    predict_SVR = Dict['predict_SVR']
    predict_mlp = Dict['predict_MLP']

    predict_lstm = Dict['predict_lstm']
    predict_lstm_emd = Dict['predict_EMD']
    predict_wavelet = Dict['predict_Wvlt']

    predict_slstm = Dict['predict_slstm']
    predict_ES = Dict['predict_ES']
    predict_WS = Dict['predict_WS']

    predict_nlstm = Dict['predict_nlstm']
    predict_EN = Dict['predict_EN']
    predict_WN = Dict['predict_WN']

    predict_blstm = Dict['predict_blstm']
    predict_EB = Dict['predict_EB']
    predict_WB = Dict['predict_WB']

    # loss_list = Dict['loss_list']
    # val_loss_list = Dict['val_loss_list']

    WN_testY_list = Dict['WN_sub_teY']
    WN_predict_list = Dict['WN_sub_predict']

    # 72: 150(0), 340(1), 430, 460, 880, 900, 1440(2), 2010(3)

    data_cut_start = 344
    data_cut_end = data_cut_start + 72

    dataY = dataY[data_cut_start:data_cut_end, ]

    predict_decideTree = predict_decideTree[data_cut_start:data_cut_end, ]
    predict_randomForest = predict_randomForest[data_cut_start:data_cut_end, ]
    predict_SVR = predict_SVR[data_cut_start:data_cut_end, ]
    predict_mlp = predict_mlp[data_cut_start:data_cut_end, ]

    predict_lstm = predict_lstm[data_cut_start:data_cut_end, ]
    predict_lstm_emd = predict_lstm_emd[data_cut_start:data_cut_end, ]
    predict_wavelet = predict_wavelet[data_cut_start:data_cut_end, ]

    predict_slstm = predict_slstm[data_cut_start:data_cut_end, ]
    predict_ES = predict_ES[data_cut_start:data_cut_end, ]
    predict_WS = predict_WS[data_cut_start:data_cut_end, ]

    predict_nlstm = predict_nlstm[data_cut_start:data_cut_end, ]
    predict_EN = predict_EN[data_cut_start:data_cut_end, ]
    predict_WN = predict_WN[data_cut_start:data_cut_end, ]

    predict_blstm = predict_blstm[data_cut_start:data_cut_end, ]
    predict_EB = predict_EB[data_cut_start:data_cut_end, ]
    predict_WB = predict_WB[data_cut_start:data_cut_end, ]

    for i in range(len(WN_testY_list)):
        WN_testY_list[i] = WN_testY_list[i][data_cut_start:data_cut_end, ]
        WN_predict_list[i] = WN_predict_list[i][data_cut_start:data_cut_end, ]

    #########################################################################

    # eva_output_save = '\nEvaluation.'
    # result_save = []

    #########################################################################

    # result_print, result_csv = Evaluate_short('DTr', dataY, predict_decideTree)
    # eva_output_save += result_print
    # result_save.append(result_csv)
    #
    # result_print, result_csv = Evaluate_short('RDf', dataY, predict_randomForest)
    # eva_output_save += result_print
    # result_save.append(result_csv)
    #
    # result_print, result_csv = Evaluate_short('SVR', dataY, predict_SVR)
    # eva_output_save += result_print
    # result_save.append(result_csv)
    #
    # result_print, result_csv = Evaluate_short('MLP', dataY, predict_mlp)
    # eva_output_save += result_print
    # result_save.append(result_csv)
    #
    # #########################################################################
    #
    # result_print, result_csv = Evaluate_short('LSTM', dataY, predict_lstm)
    # eva_output_save += result_print
    # result_save.append(result_csv)
    #
    # result_print, result_csv = Evaluate_short('EMD', dataY, predict_lstm_emd)
    # eva_output_save += result_print
    # result_save.append(result_csv)
    #
    # result_print, result_csv = Evaluate_short('Wavelet', dataY, predict_wavelet)
    # eva_output_save += result_print
    # result_save.append(result_csv)
    #
    # #########################################################################
    #
    # result_print, result_csv = Evaluate_short('SLSTM', dataY, predict_slstm)
    # eva_output_save += result_print
    # result_save.append(result_csv)
    #
    # result_print, result_csv = Evaluate_short('ES', dataY, predict_ES)
    # eva_output_save += result_print
    # result_save.append(result_csv)
    #
    # result_print, result_csv = Evaluate_short('WS', dataY, predict_WS)
    # eva_output_save += result_print
    # result_save.append(result_csv)
    #
    # #########################################################################
    #
    # result_print, result_csv = Evaluate_short('NLSTM', dataY, predict_nlstm)
    # eva_output_save += result_print
    # result_save.append(result_csv)
    #
    # result_print, result_csv = Evaluate_short('EN', dataY, predict_EN)
    # eva_output_save += result_print
    # result_save.append(result_csv)
    #
    # result_print, result_csv = Evaluate_short('WN', dataY, predict_WN)
    # eva_output_save += result_print
    # result_save.append(result_csv)
    #
    # #########################################################################
    #
    # result_print, result_csv = Evaluate_short('BLSTM', dataY, predict_blstm)
    # eva_output_save += result_print
    # result_save.append(result_csv)
    #
    # result_print, result_csv = Evaluate_short('EB', dataY, predict_EB)
    # eva_output_save += result_print
    # result_save.append(result_csv)
    #
    # result_print, result_csv = Evaluate_short('WB', dataY, predict_WB)
    # eva_output_save += result_print
    # result_save.append(result_csv)

    ########################################################################

    # result_print, result_csv = Evaluate_short('A3', WN_testY_list[0], WN_predict_list[0])
    # eva_output_save += result_print
    # result_save.append(result_csv)
    #
    # result_print, result_csv = Evaluate_short('D1', WN_testY_list[1], WN_predict_list[1])
    # eva_output_save += result_print
    # result_save.append(result_csv)
    #
    # result_print, result_csv = Evaluate_short('D2', WN_testY_list[2], WN_predict_list[2])
    # eva_output_save += result_print
    # result_save.append(result_csv)
    #
    # result_print, result_csv = Evaluate_short('D3', WN_testY_list[3], WN_predict_list[3])
    # eva_output_save += result_print
    # result_save.append(result_csv)

    ########################################################################

    # print(eva_output_save)
    #
    # save_file_name = "result\\WN_exhi_result_saved.csv"
    #
    # csv_file = open(save_file_name, "w", newline="")  # 创建csv文件
    # writer = csv.writer(csv_file)  # 创建写的对象
    # writer.writerow(["", "MAE", "RMSE", "MAPE"])
    # for wtr in range(len(result_save)):
    #     writer.writerow(result_save[wtr])

    #########################################################################

    # PE_list_mlp = perform_PE(predict_mlp, dataY)
    # PE_list_lstm = perform_PE(predict_lstm, dataY)
    # PE_list_emd = perform_PE(predict_lstm_emd, dataY)
    # PE_list_WN = perform_PE(predict_WN, dataY)
    #
    # SE_list_mlp = perform_SE(predict_mlp, dataY)
    # SE_list_lstm = perform_SE(predict_lstm, dataY)
    # SE_list_emd = perform_SE(predict_lstm_emd, dataY)
    # SE_list_WN = perform_SE(predict_WN, dataY)
    #
    # AE_list_mlp = perform_AE(predict_mlp, dataY)
    # AE_list_lstm = perform_AE(predict_lstm, dataY)
    # AE_list_emd = perform_AE(predict_lstm_emd, dataY)
    # AE_list_WN = perform_AE(predict_WN, dataY)

    #########################################################################

    # testset_length = 120
    # mape_list_long_mlp = perform_MAPE_long(predict_mlp, dataY, testset_length)
    # mape_list_long_lstm = perform_MAPE_long(predict_lstm, dataY, testset_length)
    # mape_list_long_emd = perform_MAPE_long(predict_lstm_emd, dataY, testset_length)
    # mape_list_long_WN = perform_MAPE_long(predict_WN, dataY, testset_length)

    #########################################################################

    # plt.figure(figsize=(19, 10))
    #
    # plt.subplot(1, 4, 1)
    # plt.title('A3')
    # plt.plot(WN_testY_list[0], label="Real", color='black', linestyle=':', linewidth=2)
    # plt.plot(WN_predict_list[0], label="Predict", color='red', linewidth=2)
    # plt.legend(loc='best')
    #
    # plt.subplot(1, 4, 2)
    # plt.title('D1')
    # plt.plot(WN_testY_list[1], label="Real", color='black', linestyle=':', linewidth=2)
    # plt.plot(WN_predict_list[1], label="Predict", color='red', linewidth=2)
    # plt.legend(loc='best')
    #
    # plt.subplot(1, 4, 3)
    # plt.title('D2')
    # plt.plot(WN_testY_list[2], label="Real", color='black', linestyle=':', linewidth=2)
    # plt.plot(WN_predict_list[2], label="Predict", color='red', linewidth=2)
    # plt.legend(loc='best')
    #
    # plt.subplot(1, 4, 4)
    # plt.title('D3')
    # plt.plot(WN_testY_list[3], label="Real", color='black', linestyle=':', linewidth=2)
    # plt.plot(WN_predict_list[3], label="Predict", color='red', linewidth=2)
    # plt.legend(loc='best')
    #
    # plt.show()

    #########################################################################

    # plt.figure(figsize=(15, 10))
    #
    # plt.subplot(1, 4, 1)
    # plt.title('A3')
    # plt.plot(loss_list[0], label="loss", color='orange', marker='o')
    # # plt.plot(val_loss_list[0], label="val_loss", color='purple', marker='o')
    # plt.legend(loc='best')
    #
    # plt.subplot(1, 4, 2)
    # plt.title('D1')
    # plt.plot(loss_list[1], label="loss", color='orange', marker='o')
    # # plt.plot(val_loss_list[1], label="val_loss", color='purple', marker='o')
    # plt.legend(loc='best')
    #
    # plt.subplot(1, 4, 3)
    # plt.title('D2')
    # plt.plot(loss_list[2], label="loss", color='orange', marker='o')
    # # plt.plot(val_loss_list[2], label="val_loss", color='purple', marker='o')
    # plt.legend(loc='best')
    #
    # plt.subplot(1, 4, 4)
    # plt.title('D3')
    # plt.plot(loss_list[3], label="loss", color='orange', marker='o')
    # # plt.plot(val_loss_list[3], label="val_loss", color='purple', marker='o')
    # plt.legend(loc='best')
    #
    # plt.show()

    #########################################################################

    main_linewidth = 2
    second_linewidth = 2
    third_linewidth = 1.25

    plt.figure(1, figsize=(19, 5))
    plt.plot(dataY, "black", label="Real data", linewidth=main_linewidth, linestyle='--', marker='.')

    # plt.plot(predict_mlp, "aqua", label="MLP", linewidth=third_linewidth)
    # plt.plot(predict_lstm_emd, "limegreen", label="EMD", linewidth=third_linewidth)
    # plt.plot(predict_lstm, "gold", label="LSTM", linewidth=third_linewidth)
    #
    # plt.plot(predict_WN, "red", label="Proposed", linewidth=main_linewidth)

    plt.plot(predict_decideTree, "aqua", label="Decision Tree", linewidth=third_linewidth)
    plt.plot(predict_randomForest, "grey", label="Random Foreest", linewidth=third_linewidth)
    plt.plot(predict_SVR, "orange", label="SVR", linewidth=third_linewidth)
    plt.plot(predict_mlp, "royalblue", label="MLP", linewidth=third_linewidth)

    plt.plot(predict_lstm, "red", label="LSTM", linewidth=third_linewidth)
    # plt.plot(predict_slstm, "orange", label="SLSTM", linewidth=third_linewidth)
    # plt.plot(predict_blstm, "darkkhaki", label="BLSTM", linewidth=third_linewidth)
    # plt.plot(predict_nlstm, "darkorange", label="NLSTM", linewidth=third_linewidth)

    # plt.plot(predict_lstm_emd, "limegreen", label="EMD-LSTM", linewidth=third_linewidth)
    # plt.plot(predict_ES, "lightgreen", label="EMD-SLSTM", linewidth=third_linewidth)
    # plt.plot(predict_EB, "seagreen", label="EMD-BLSTM", linewidth=third_linewidth)
    # plt.plot(predict_EN, "springgreen", label="EMD-NLSTM", linewidth=third_linewidth)

    # plt.plot(predict_wavelet, "lightcoral", label="Wavelet-LSTM", linewidth=third_linewidth)
    # plt.plot(predict_WS, "indianred", label="Wavelet-SLSTM", linewidth=third_linewidth)
    # plt.plot(predict_WB, "tomato", label="Wavelet-BLSTM", linewidth=third_linewidth)

    # plt.plot(predict_WN, "red", label="Proposed", linewidth=main_linewidth)

    plt.grid(True, linestyle=":", color="gray", linewidth="0.5", axis='both')
    plt.subplots_adjust(left=0.04, bottom=0.1, right=0.99, top=0.95)
    plt.xlabel("Time(Hours)")
    plt.ylabel("PM2.5 Concentration(ug/m^3)")
    plt.title('Forcasts and Actual Comparison')
    plt.legend(loc='best')
    plt.show()

    #########################################################################

    # plt.figure(2, figsize=(15, 5))
    # plt.plot(PE_list_lstm - PE_list_WN, "red", label="LSTM", linewidth=second_linewidth, marker='x')
    # plt.plot(PE_list_mlp - PE_list_WN, "orange", label="MLP", linewidth=second_linewidth, marker='x')
    # plt.plot(PE_list_emd - PE_list_WN, "blue", label="EMD", linewidth=second_linewidth, marker='x')
    # # plt.plot(MAPE_list_WN, "purple", label="WNLSTM", linewidth=main_linewidth, linestyle='--', marker='o')
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("P ERROR")
    # plt.legend(loc='best')
    # plt.show()
    #
    # plt.figure(3, figsize=(15, 5))
    # plt.plot(SE_list_lstm - SE_list_WN, "red", label="LSTM", linewidth=second_linewidth, marker='x')
    # plt.plot(SE_list_mlp - SE_list_WN, "orange", label="MLP", linewidth=second_linewidth, marker='x')
    # plt.plot(SE_list_emd - SE_list_WN, "blue", label="EMD", linewidth=second_linewidth, marker='x')
    # # plt.plot(MAPE_list_WN, "purple", label="WNLSTM", linewidth=main_linewidth, linestyle='--', marker='o')
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("S ERROR")
    # plt.legend(loc='best')
    # plt.show()

    # plt.figure(4, figsize=(15, 5))
    # plt.plot((AE_list_lstm - AE_list_WN), "red", label="LSTM", linewidth=second_linewidth, marker='x')
    # plt.plot((AE_list_mlp - AE_list_WN), "orange", label="MLP", linewidth=second_linewidth, marker='x')
    # plt.plot((AE_list_emd - AE_list_WN), "blue", label="EMD", linewidth=second_linewidth, marker='x')
    # plt.axhline(y=0, ls=":", c="black", linewidth=third_linewidth)
    # # plt.plot(MAPE_list_WN, "purple", label="WNLSTM", linewidth=main_linewidth, linestyle='--', marker='o')
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("A ERROR")
    # plt.legend(loc='best')
    # plt.show()

    #########################################################################

    print('\nComplete.')

    return None


if __name__ == "__main__":
    # start_num = 7000 * 24
    start_num = 24 * 1000 * 2
    interval_ori = 1

    # choice_load = input('Do you want to load the last result?(yes/no)\n')
    choice_load = 'no'
    if choice_load == 'yes':
        print('The result shall be loaded.')
        load_result_evaluation()
    else:
        print('The result shall not be loaded. If you want it saved, enter \'yes\' next time.')
        print('We shall conduct a new prediction process.\n')
        _ = neo_prediction(start_num, interval_ori)
