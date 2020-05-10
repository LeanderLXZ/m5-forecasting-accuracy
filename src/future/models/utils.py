import pickle
import pandas as pd
import numpy as np
import os
import time
import random
import csv
import copy
from os.path import isdir
from config import cfg


# Save Data
def save_data_to_pkl(data, data_path):
    print('Saving ' + data_path + '...')

    with open(data_path, 'wb') as f:
        pickle.dump(data, f)


# Save predictions to csv file
def save_pred_to_csv(file_path, index, pred):
    print('Saving Predictions To CSV File...')

    df = pd.DataFrame({'FORECASTDATE': index, 'FORECASTPRICE': pred})

    df.to_csv(file_path + 'result.csv', sep=',', index=False)


# Save pred of train set to csv file
def save_pred_train_to_csv(file_path, pred, label):
    print('Saving Probabilities of Train Set To CSV File...')

    df = pd.DataFrame({'pred_train': pred, 'label': label})

    df.to_csv(file_path + 'pred_train.csv', sep=',', index=True)


# Save Grid Search Logs
def save_grid_search_log(log_path, params, params_grid, best_score,
                         best_parameters, total_time):
    with open(log_path + 'grid_search_log.txt', 'a') as f:
        local_time = time.strftime('%Y/%m/%d-%H:%M:%S',
                                   time.localtime(time.time()))

        f.write('=====================================================\n')
        f.write('Time: {}\n'.format(local_time))
        f.write('------------------------------------------------------\n')
        f.write('Total Time: {:.3f}s\n'.format(total_time))
        f.write('Best Score: {:.8f}\n'.format(best_score))
        f.write('Parameters:\n')
        f.write('\t' + str(params) + '\n\n')
        f.write('Parameters Grid:\n')
        f.write('\t' + str(params_grid) + '\n\n')
        f.write('Best Parameters Set:\n')
        for param_name in sorted(params_grid.keys()):
            f.write('\t' + str(param_name) + ': {}\n'.format(
                str(best_parameters[param_name])))


# Save Final Losses
def save_loss_log(log_path, count, parameters, n_valid, n_cv, valid_era,
                  loss_train,
                  loss_valid, loss_train_w, loss_valid_w, train_seed=None,
                  cv_seed=None,
                  acc_train=None, acc_valid=None, acc_train_era=None,
                  acc_valid_era=None):
    with open(log_path + 'loss_log.txt', 'a') as f:
        print('------------------------------------------------------')
        print('Saving Losses...')

        local_time = time.strftime('%Y/%m/%d-%H:%M:%S',
                                   time.localtime(time.time()))

        f.write(
            '===================== CV: {}/{} =====================\n'.format(
                count, n_cv))
        f.write('Time: {}\n'.format(local_time))
        f.write('------------------------------------------------------\n')
        f.write('Train Seed: {}\n'.format(train_seed))
        f.write('CV Seed: {}\n'.format(cv_seed))
        f.write('Validation Era Number: {}\n'.format(n_valid))
        f.write('Validation Spilt Number: {}\n'.format(n_cv))
        f.write('Validation Set Index: ' + str(valid_era) + '\n')
        f.write('Parameters:\n')
        f.write('\t' + str(parameters) + '\n\n')
        f.write('Losses:\n')
        f.write('\tCV Train LogLoss: {:.8f}\n'.format(loss_train))
        f.write('\tCV Validation LogLoss: {:.8f}\n'.format(loss_valid))
        f.write('\tCV Train LogLoss with Weight: {:.8f}\n'.format(loss_train_w))
        f.write('\tCV Validation LogLoss with Weight: {:.8f}\n\n'.format(
            loss_valid_w))
        f.write('Accuracies:\n')
        f.write('\tCV Train Accuracy: {:.3f}%\n'.format(acc_train * 100))
        f.write('\tCV Valid Accuracy: {:.3f}%\n'.format(acc_valid * 100))
        f.write('\tTrain Eras Accuracy:\n')
        f.write('\t\t' + str(acc_train_era) + '\n')
        f.write('\tValid Eras Accuracy:\n')
        f.write('\t\t' + str(acc_valid_era) + '\n\n')


def save_final_loss_log(log_path, parameters, n_valid, n_cv, loss_train_mean,
                        loss_valid_mean, loss_train_w_mean, loss_valid_w_mean,
                        train_seed=None, cv_seed=None, acc_train=None,
                        acc_train_era=None):
    with open(log_path + 'loss_log.txt', 'a') as f:
        print('------------------------------------------------------')
        print('Saving Final Losses...')

        local_time = time.strftime('%Y/%m/%d-%H:%M:%S',
                                   time.localtime(time.time()))

        f.write('==================== Final Losses ===================\n')
        f.write('Time: {}\n'.format(local_time))
        f.write('------------------------------------------------------\n')
        f.write('Train Seed: {}\n'.format(train_seed))
        f.write('CV Seed: {}\n'.format(cv_seed))
        f.write('Validation Era Number: {}\n'.format(n_valid))
        f.write('Validation Spilt Number: {}\n'.format(n_cv))
        f.write('Parameters:\n')
        f.write('\t' + str(parameters) + '\n\n')
        f.write('Losses:\n')
        f.write('\tTotal Train LogLoss: {:.8f}\n'.format(loss_train_mean))
        f.write('\tTotal Validation LogLoss: {:.8f}\n'.format(loss_valid_mean))
        f.write('\tTotal Train LogLoss with Weight: {:.8f}\n'.format(
            loss_train_w_mean))
        f.write('\tTotal Validation LogLoss with Weight: {:.8f}\n\n'.format(
            loss_valid_w_mean))
        f.write('Accuracy:\n')
        f.write('\tTotal Train Accuracy: {:.3f}%\n\n'.format(acc_train * 100))
        f.write('\tTotal Train Eras Accuracies:\n')
        f.write('\t\t' + str(acc_train_era) + '\n\n')
        f.write('=====================================================\n')

    with open(log_path + 'final_loss_log.txt', 'a') as f:
        print('Saving Final Losses...')

        local_time = time.strftime('%Y/%m/%d-%H:%M:%S',
                                   time.localtime(time.time()))

        f.write('=====================================================\n')
        f.write('Time: {}\n'.format(local_time))
        f.write('------------------------------------------------------\n')
        f.write('Train Seed: {}\n'.format(train_seed))
        f.write('CV Seed: {}\n'.format(cv_seed))
        f.write('Validation Era Number: {}\n'.format(n_valid))
        f.write('Validation Spilt Number: {}\n'.format(n_cv))
        f.write('Parameters:\n')
        f.write('\t' + str(parameters) + '\n\n')
        f.write('Losses:\n')
        f.write('\tTotal Train LogLoss: {:.8f}\n'.format(loss_train_mean))
        f.write('\tTotal Validation LogLoss: {:.8f}\n'.format(loss_valid_mean))
        f.write('\tTotal Train LogLoss with Weight: {:.8f}\n'.format(
            loss_train_w_mean))
        f.write('\tTotal Validation LogLoss with Weight: {:.8f}\n\n'.format(
            loss_valid_w_mean))
        f.write('Accuracy:\n')
        f.write('\tTotal Train Accuracy: {:.3f}%\n\n'.format(acc_train * 100))
        f.write('\tTotal Train Eras Accuracies:\n')
        f.write('\t\t' + str(acc_train_era) + '\n\n')


# Save Loss Log to csv File
def save_final_loss_log_to_csv(idx, log_path, loss_train_w_mean,
                               loss_valid_w_mean, acc_train,
                               train_seed, cv_seed, n_valid, n_cv, parameters,
                               profit=None):
    if not os.path.isfile(log_path + 'csv_log.csv'):

        print('------------------------------------------------------')
        print('Creating csv File of Final Loss Log...')

        with open(log_path + 'csv_log.csv', 'w') as f:
            if profit is None:
                header = ['id', 'time', 'loss_train', 'loss_valid',
                          'train_accuracy',
                          'train_seed', 'cv_seed', 'n_valid', 'n_cv',
                          'parameters']
            else:
                header = ['id', 'time', 'profit', 'loss_train', 'loss_valid',
                          'train_accuracy',
                          'train_seed', 'cv_seed', 'n_valid', 'n_cv',
                          'parameters']
            writer = csv.writer(f)
            writer.writerow(header)

    with open(log_path + 'csv_log.csv', 'a') as f:

        print('------------------------------------------------------')
        print('Saving Final Losses to csv File...')

        local_time = time.strftime('%Y/%m/%d-%H:%M:%S',
                                   time.localtime(time.time()))
        if profit is None:
            log = [idx, local_time, loss_train_w_mean, loss_valid_w_mean,
                   acc_train,
                   train_seed, cv_seed, n_valid, n_cv, str(parameters)]
        else:
            log = [idx, local_time, profit, loss_train_w_mean,
                   loss_valid_w_mean, acc_train,
                   train_seed, cv_seed, n_valid, n_cv, str(parameters)]
        writer = csv.writer(f)
        writer.writerow(log)


# Save Loss Log with Global Validation to csv File
def save_log_with_glv_to_csv(idx, log_path, loss_train_w_mean,
                             loss_valid_w_mean, acc_train,
                             train_seed, loss_global_valid_w_mean,
                             acc_total_global_valid,
                             cv_seed, n_valid, n_cv, parameters, profit=None):
    if not os.path.isfile(log_path + 'csv_log.csv'):
        print('------------------------------------------------------')
        print('Creating csv File of Final Loss Log with Global Validation...')

        with open(log_path + 'csv_log.csv', 'w') as f:
            if profit is None:
                header = ['id', 'time', 'loss_train', 'valid_loss',
                          'global_valid_loss', 'train_accuracy',
                          'global_valid_accuracy', 'train_seed', 'cv_seed',
                          'n_valid', 'n_cv', 'parameters']
            else:
                header = ['id', 'time', 'profit', 'loss_train', 'valid_loss',
                          'global_valid_loss', 'train_accuracy',
                          'global_valid_accuracy', 'train_seed', 'cv_seed',
                          'n_valid', 'n_cv', 'parameters']
            writer = csv.writer(f)
            writer.writerow(header)

    with open(log_path + 'csv_log.csv', 'a') as f:
        print('------------------------------------------------------')
        print('Saving Final Losses with Global Validation to csv File...')

        local_time = time.strftime('%Y/%m/%d-%H:%M:%S',
                                   time.localtime(time.time()))
        if profit is None:
            log = [idx, local_time, loss_train_w_mean, loss_valid_w_mean,
                   loss_global_valid_w_mean, acc_train,
                   acc_total_global_valid, train_seed, cv_seed, n_valid, n_cv,
                   str(parameters)]
        else:
            log = [idx, local_time, profit, loss_train_w_mean,
                   loss_valid_w_mean, loss_global_valid_w_mean,
                   acc_train, acc_total_global_valid, train_seed, cv_seed,
                   n_valid, n_cv, str(parameters)]
        writer = csv.writer(f)
        writer.writerow(log)


# Save Grid Search Log to csv File
def save_grid_search_log_to_csv(idx, log_path, loss_train_w_mean,
                                loss_valid_w_mean, acc_train, train_seed,
                                cv_seed,
                                n_valid, n_cv, parameters, param_name_list,
                                param_value_list, profit=None):
    if not os.path.isfile(log_path + 'csv_log.csv'):
        print('------------------------------------------------------')
        print('Creating csv File of Final Loss Log...')

        with open(log_path + 'csv_log.csv', 'w') as f:
            if profit is None:
                header = ['id', 'time', 'loss_train', 'loss_valid',
                          'train_accuracy',
                          *param_name_list, 'train_seed', 'cv_seed', 'n_valid',
                          'n_cv', 'parameters']
            else:
                header = ['id', 'time', 'profit', 'loss_train', 'loss_valid',
                          'train_accuracy',
                          *param_name_list, 'train_seed', 'cv_seed', 'n_valid',
                          'n_cv', 'parameters']
            writer = csv.writer(f)
            writer.writerow(header)

    with open(log_path + 'csv_log.csv', 'a') as f:
        print('------------------------------------------------------')
        print('Saving Final Losses to csv File...')

        local_time = time.strftime('%Y/%m/%d-%H:%M:%S',
                                   time.localtime(time.time()))
        if profit is None:
            log = [idx, local_time, loss_train_w_mean, loss_valid_w_mean,
                   acc_train,
                   *param_value_list, train_seed, cv_seed, n_valid, n_cv,
                   str(parameters)]
        else:
            log = [idx, local_time, profit, loss_train_w_mean,
                   loss_valid_w_mean, acc_train,
                   *param_value_list, train_seed, cv_seed, n_valid, n_cv,
                   str(parameters)]
        writer = csv.writer(f)
        writer.writerow(log)


# Save Loss Log with Global Validation to csv File
def save_grid_search_log_with_glv_to_csv(idx, log_path, loss_train_w_mean,
                                         loss_valid_w_mean, acc_train,
                                         train_seed, loss_global_valid_w_mean,
                                         acc_total_global_valid, cv_seed,
                                         n_valid, n_cv, parameters,
                                         param_name_list, param_value_list,
                                         profit=None):
    if not os.path.isfile(log_path + 'csv_log.csv'):
        print('------------------------------------------------------')
        print('Creating csv File of Final Loss Log with Global Validation...')

        with open(log_path + 'csv_log.csv', 'w') as f:
            if profit is None:
                header = ['id', 'time', 'loss_train', 'valid_loss',
                          'global_valid_loss',
                          'train_accuracy', 'global_valid_accuracy',
                          *param_name_list, 'train_seed',
                          'cv_seed', 'n_valid', 'n_cv', 'parameters']
            else:
                header = ['id', 'time', 'profit', 'loss_train', 'valid_loss',
                          'global_valid_loss',
                          'train_accuracy', 'global_valid_accuracy',
                          *param_name_list, 'train_seed',
                          'cv_seed', 'n_valid', 'n_cv', 'parameters']
            writer = csv.writer(f)
            writer.writerow(header)

    with open(log_path + 'csv_log.csv', 'a') as f:
        print('------------------------------------------------------')
        print('Saving Final Losses with Global Validation to csv File...')

        local_time = time.strftime('%Y/%m/%d-%H:%M:%S',
                                   time.localtime(time.time()))
        if profit is None:
            log = [idx, local_time, loss_train_w_mean, loss_valid_w_mean,
                   loss_global_valid_w_mean, acc_train,
                   acc_total_global_valid, *param_value_list, train_seed,
                   cv_seed, n_valid, n_cv, str(parameters)]
        else:
            log = [idx, local_time, profit, loss_train_w_mean,
                   loss_valid_w_mean, loss_global_valid_w_mean, acc_train,
                   acc_total_global_valid, *param_value_list, train_seed,
                   cv_seed, n_valid, n_cv, str(parameters)]
        writer = csv.writer(f)
        writer.writerow(log)


# Save Boost Round Log to csv File
def save_boost_round_log_to_csv(model_name, boost_round_log_path,
                                boost_round_log_upper_path,
                                csv_idx, idx_round, valid_loss_round_mean,
                                train_loss_round_mean,
                                train_seed, cv_seed, parameters,
                                param_name_list,
                                param_value_list, param_name, profit=None):
    valid_loss_dict = {}
    train_loss_dict = {}
    lowest_loss_dict = {}

    for i, idx in enumerate(idx_round):
        valid_loss_dict[idx] = valid_loss_round_mean[i]
        train_loss_dict[idx] = train_loss_round_mean[i]

    lowest_valid_loss_idx = list(np.argsort(valid_loss_round_mean)[:5])
    lowest_valid_loss = valid_loss_round_mean[lowest_valid_loss_idx[0]]
    lowest_train_loss = train_loss_round_mean[lowest_valid_loss_idx[0]]
    lowest_round = np.array(idx_round)[lowest_valid_loss_idx[0]]
    lowest_idx = np.array(idx_round)[lowest_valid_loss_idx]
    lowest_idx = np.sort(lowest_idx)

    for idx in lowest_idx:
        lowest_loss_dict[idx] = (valid_loss_dict[idx], train_loss_dict[idx])

    def _save_log(log_path):

        if not os.path.isfile(log_path + 'boost_round_log.csv'):

            print('------------------------------------------------------')
            print('Creating csv File of Boost Round Log...')

            with open(log_path + 'boost_round_log.csv', 'w') as f:
                if profit is None:
                    header = ['idx', 'time', 'lowest_round',
                              'lowest_valid_loss',
                              'lowest_train_loss', 'round', 'valid_loss',
                              'train_loss',
                              *param_name_list, 'train_seed', 'cv_seed',
                              'parameters']
                else:
                    header = ['idx', 'time', 'profit', 'lowest_round',
                              'lowest_valid_loss',
                              'lowest_train_loss', 'round', 'valid_loss',
                              'train_loss',
                              *param_name_list, 'train_seed', 'cv_seed',
                              'parameters']
                writer = csv.writer(f)
                writer.writerow(header)

        with open(log_path + 'boost_round_log.csv', 'a') as f:

            print('------------------------------------------------------')
            print('Saving Boost Round Log to csv File...')

            for ii, (round_idx, (valid_loss, train_loss)) in enumerate(
                lowest_loss_dict.items()):
                if ii == 0:
                    local_time = time.strftime('%Y/%m/%d-%H:%M:%S',
                                               time.localtime(time.time()))
                    if profit is None:
                        log = [csv_idx, local_time, lowest_round,
                               lowest_valid_loss,
                               lowest_train_loss, round_idx, valid_loss,
                               train_loss,
                               *param_value_list, train_seed, cv_seed,
                               str(parameters)]
                    else:
                        log = [csv_idx, local_time, profit, lowest_round,
                               lowest_valid_loss,
                               lowest_train_loss, round_idx, valid_loss,
                               train_loss,
                               *param_value_list, train_seed, cv_seed,
                               str(parameters)]
                else:
                    placeholder = [''] * len(param_value_list)
                    if profit is None:
                        log = [csv_idx, '', '', '', '', round_idx, valid_loss,
                               train_loss, *placeholder, train_seed, cv_seed,
                               '']
                    else:
                        log = [csv_idx, '', '', '', '', '', round_idx,
                               valid_loss,
                               train_loss, *placeholder, train_seed, cv_seed,
                               '']
                writer = csv.writer(f)
                writer.writerow(log)

    _save_log(boost_round_log_path)
    boost_round_log_upper_path += model_name + param_name
    _save_log(boost_round_log_upper_path)


# Save Boost Round Log with Global Validation to csv File
def save_boost_round_log_gl_to_csv(model_name, boost_round_log_path,
                                   boost_round_log_upper_path,
                                   csv_idx, idx_round, valid_loss_round_mean,
                                   train_loss_round_mean,
                                   global_valid_loss_round_mean, train_seed,
                                   cv_seed, parameters,
                                   param_name_list, param_value_list,
                                   param_name, profit=None):
    gl_valid_loss_dict = {}
    valid_loss_dict = {}
    train_loss_dict = {}
    lowest_loss_dict = {}

    for i, idx in enumerate(idx_round):
        gl_valid_loss_dict[idx] = global_valid_loss_round_mean[i]
        valid_loss_dict[idx] = valid_loss_round_mean[i]
        train_loss_dict[idx] = train_loss_round_mean[i]

    lowest_gl_valid_loss_idx = list(
        np.argsort(global_valid_loss_round_mean)[:5])
    lowest_gl_valid_loss = valid_loss_round_mean[lowest_gl_valid_loss_idx[0]]
    lowest_valid_loss = valid_loss_round_mean[
        np.argsort(valid_loss_round_mean)[0]]
    lowest_round = np.array(idx_round)[lowest_gl_valid_loss_idx[0]]
    lowest_idx = np.array(idx_round)[lowest_gl_valid_loss_idx]
    lowest_idx = np.sort(lowest_idx)

    for idx in lowest_idx:
        lowest_loss_dict[idx] = (
        gl_valid_loss_dict[idx], valid_loss_dict[idx], train_loss_dict[idx])

    def _save_log(log_path):

        if not os.path.isfile(log_path + 'boost_round_log.csv'):
            print('------------------------------------------------------')
            print('Creating csv File of Boost Round Log...')

            with open(log_path + 'boost_round_log.csv', 'w') as f:
                if profit is None:
                    header = ['idx', 'time', 'lowest_round',
                              'lowest_global_valid_loss',
                              'lowest_cv_valid_loss', 'round',
                              'global_valid_loss',
                              'cv_valid_loss', 'cv_train_loss',
                              *param_name_list,
                              'train_seed', 'cv_seed', 'parameters']
                else:
                    header = ['idx', 'time', 'profit', 'lowest_round',
                              'lowest_global_valid_loss',
                              'lowest_cv_valid_loss', 'round',
                              'global_valid_loss', 'cv_valid_loss',
                              'cv_train_loss',
                              *param_name_list, 'train_seed', 'cv_seed',
                              'parameters']
                writer = csv.writer(f)
                writer.writerow(header)

        with open(log_path + 'boost_round_log.csv', 'a') as f:

            print('------------------------------------------------------')
            print(
                'Saving Boost Round Log with Global Validation to csv File...')

            for ii, (
            round_idx, (gl_valid_loss, valid_loss, train_loss)) in enumerate(
                lowest_loss_dict.items()):
                if ii == 0:
                    local_time = time.strftime('%Y/%m/%d-%H:%M:%S',
                                               time.localtime(time.time()))
                    if profit is None:
                        log = [csv_idx, local_time, lowest_round,
                               lowest_gl_valid_loss,
                               lowest_valid_loss, round_idx, gl_valid_loss,
                               valid_loss, train_loss,
                               *param_value_list, train_seed, cv_seed,
                               str(parameters)]
                    else:
                        log = [csv_idx, local_time, profit, lowest_round,
                               lowest_gl_valid_loss,
                               lowest_valid_loss, round_idx, gl_valid_loss,
                               valid_loss, train_loss,
                               *param_value_list, train_seed, cv_seed,
                               str(parameters)]
                else:
                    placeholder = [''] * len(param_value_list)
                    if profit is None:
                        log = [csv_idx, '', '', '', '', round_idx,
                               gl_valid_loss,
                               valid_loss, train_loss, *placeholder, train_seed,
                               cv_seed, '']
                    else:
                        log = [csv_idx, '', '', '', '', '', round_idx,
                               gl_valid_loss,
                               valid_loss, train_loss, *placeholder, train_seed,
                               cv_seed, '']
                writer = csv.writer(f)
                writer.writerow(log)

    _save_log(boost_round_log_path)
    boost_round_log_upper_path += model_name + param_name
    _save_log(boost_round_log_upper_path)


def save_final_boost_round_log(boost_round_log_path, idx_round,
                               train_loss_round_mean,
                               valid_loss_round_mean, profit=None):
    print('------------------------------------------------------')
    print('Saving Final Boost Round Log...')

    if profit is None:
        df = pd.DataFrame({'idx': idx_round,
                           'train_loss': train_loss_round_mean,
                           'valid_loss': valid_loss_round_mean})
        cols = ['idx', 'train_loss', 'valid_loss']
    else:
        df = pd.DataFrame({'idx': idx_round,
                           'profit': profit,
                           'train_loss': train_loss_round_mean,
                           'valid_loss': valid_loss_round_mean})
        cols = ['idx', 'profit', 'train_loss', 'valid_loss']
    df = df.loc[:, cols]
    df.to_csv(boost_round_log_path, sep=',', index=False)


def save_final_boost_round_gl_log(boost_round_log_path, idx_round,
                                  train_loss_round_mean,
                                  valid_loss_round_mean,
                                  global_valid_loss_round_mean, profit=None):
    print('------------------------------------------------------')
    print('Saving Final Boost Round Log with Global Validation...')

    if profit is None:
        df = pd.DataFrame({'idx': idx_round,
                           'cv_train_loss': train_loss_round_mean,
                           'cv_valid_loss': valid_loss_round_mean,
                           'global_valid_loss:': global_valid_loss_round_mean})
        cols = ['idx', 'cv_train_loss', 'cv_valid_loss', 'global_valid_loss:']
    else:
        df = pd.DataFrame({'idx': idx_round,
                           'profit': profit,
                           'cv_train_loss': train_loss_round_mean,
                           'cv_valid_loss': valid_loss_round_mean,
                           'global_valid_loss:': global_valid_loss_round_mean})
        cols = ['idx', 'profit', 'cv_train_loss', 'cv_valid_loss',
                'global_valid_loss:']
    df = df.loc[:, cols]
    print(df)
    df.to_csv(boost_round_log_path, sep=',', index=False)


# Save stacking outputs of layers
def save_stack_outputs(output_path, x_outputs, test_outputs, x_g_outputs,
                       test_g_outputs):
    print('Saving Stacking Outputs of Layer...')

    save_data_to_pkl(x_outputs, output_path + 'x_outputs.p')
    save_data_to_pkl(test_outputs, output_path + 'test_outputs.p')
    save_data_to_pkl(x_g_outputs, output_path + 'x_g_outputs.p')
    save_data_to_pkl(test_g_outputs, output_path + 'test_g_outputs.p')


# Load Data
def load_pkl_to_data(data_path):
    print('Loading ' + data_path + '...')

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    return data


# Load Stacked Layer
def load_stacked_data(output_path):
    print('Loading Stacked Data...')

    x_outputs = load_pkl_to_data(output_path + 'x_outputs.p')
    test_outputs = load_pkl_to_data(output_path + 'test_outputs.p')
    x_g_outputs = load_pkl_to_data(output_path + 'x_g_outputs.p')
    test_g_outputs = load_pkl_to_data(output_path + 'test_g_outputs.p')

    return x_outputs, test_outputs, x_g_outputs, test_g_outputs


# Load Preprocessed Data
def load_preprocessed_data(data_file_path):
    x_train = pd.read_pickle(data_file_path + 'x_train.p')
    y_train = pd.read_pickle(data_file_path + 'y_train.p')
    w_train = pd.read_pickle(data_file_path + 'w_train.p')
    e_train = pd.read_pickle(data_file_path + 'e_train.p')
    x_test = pd.read_pickle(data_file_path + 'x_test.p')
    pct_test = pd.read_pickle(data_file_path + 'pct_test.p')
    id_test = pd.read_pickle(data_file_path + 'id_test.p')

    return x_train, y_train, w_train, e_train, x_test, pct_test, id_test


# Load Preprocessed Test data
def load_preprocessed_test_data(data_file_path):
    y_test = pd.read_pickle(data_file_path + 'y_test.p')
    w_test = pd.read_pickle(data_file_path + 'w_test.p')
    e_test = pd.read_pickle(data_file_path + 'e_test.p')

    return y_test, w_test, e_test


# Load Preprocessed Data
def load_global_valid_data(data_file_path):
    x_global_valid = pd.read_pickle(data_file_path + 'x_global_valid.p')
    x_g_global_valid = pd.read_pickle(data_file_path + 'x_g_global_valid.p')
    y_global_valid = pd.read_pickle(data_file_path + 'y_global_valid.p')
    w_global_valid = pd.read_pickle(data_file_path + 'w_global_valid.p')
    e_global_valid = pd.read_pickle(data_file_path + 'e_global_valid.p')

    return x_global_valid, x_g_global_valid, y_global_valid, w_global_valid, e_global_valid


# Load Preprocessed Category Data
def load_preprocessed_data_g(data_file_path):
    x_g_train = pd.read_pickle(data_file_path + 'x_g_train.p')
    x_g_test = pd.read_pickle(data_file_path + 'x_g_test.p')

    return x_g_train, x_g_test


# Load Preprocessed Code ID
def load_preprocessed_code_id(data_file_path):
    code_id_train = pd.read_pickle(data_file_path + 'code_id_train.p')
    code_id_test = pd.read_pickle(data_file_path + 'code_id_test.p')

    return code_id_train, code_id_test


# Load Preprocessed Positive Data
def load_preprocessed_positive_data(data_file_path):
    x_train_p = pd.read_pickle(data_file_path + 'x_train_p.p')
    y_train_p = pd.read_pickle(data_file_path + 'y_train_p.p')
    w_train_p = pd.read_pickle(data_file_path + 'w_train_p.p')
    e_train_p = pd.read_pickle(data_file_path + 'e_train_p.p')
    x_g_train_p = pd.read_pickle(data_file_path + 'x_g_train_p.p')

    return x_train_p, y_train_p, w_train_p, e_train_p, x_g_train_p


# Load Preprocessed Negative Data
def load_preprocessed_negative_data(data_file_path):
    x_train_n = pd.read_pickle(data_file_path + 'x_train_n.p')
    y_train_n = pd.read_pickle(data_file_path + 'y_train_n.p')
    w_train_n = pd.read_pickle(data_file_path + 'w_train_n.p')
    e_train_n = pd.read_pickle(data_file_path + 'e_train_n.p')
    x_g_train_n = pd.read_pickle(data_file_path + 'x_g_train_n.p')

    return x_train_n, y_train_n, w_train_n, e_train_n, x_g_train_n


# Calculate LogLoss without weight
def log_loss(pred, y):
    loss = - np.sum(np.multiply(y, np.log(pred)) +
                    np.multiply((np.ones_like(y) - y),
                                np.log(np.ones_like(pred) - pred)))

    loss /= len(y)

    return loss


# Calculate LogLoss with weight
def log_loss_with_weight(pred, y, w):
    w = w / np.sum(w)

    loss = - np.sum(np.multiply(w, (np.multiply(y, np.log(pred)) +
                                    np.multiply((np.ones_like(y) - y), np.log(
                                        np.ones_like(pred) - pred)))))

    return loss


# Print Information of Grid Search
def print_grid_info(model_name, parameters, parameters_grid):
    print('\nModel: ' + model_name + '\n')
    print("Parameters:")
    print(parameters)
    print('\n')
    print("Parameters' Grid:")
    print(parameters_grid)
    print('\n')


# Print CV Losses
def print_loss(pred_train, y_train, w_train, pred_valid, y_valid, w_valid):
    loss_train = log_loss(pred_train, y_train)
    loss_valid = log_loss(pred_valid, y_valid)

    loss_train_w = log_loss_with_weight(pred_train, y_train, w_train)
    loss_valid_w = log_loss_with_weight(pred_valid, y_valid, w_valid)

    print('------------------------------------------------------')
    print('Train LogLoss: {:.8f}\n'.format(loss_train),
          'Validation LogLoss: {:.8f}\n'.format(loss_valid),
          'Train LogLoss with Weight: {:.8f}\n'.format(loss_train_w),
          'Validation LogLoss with Weight: {:.8f}'.format(loss_valid_w))

    return loss_train, loss_valid, loss_train_w, loss_valid_w


# Print Loss and Accuracy of Global Validation Set
def print_global_valid_loss_and_acc(pred_global_valid, y_global_valid,
                                    w_global_valid):
    loss_global_valid = log_loss(pred_global_valid, y_global_valid)
    loss_global_valid_w = log_loss_with_weight(pred_global_valid,
                                               y_global_valid, w_global_valid)
    acc_global_valid = get_accuracy(pred_global_valid, y_global_valid)
    print('------------------------------------------------------')
    print('Global Valid LogLoss: {:.8f}\n'.format(loss_global_valid),
          'Global Valid LogLoss with Weight: {:.8f}\n'.format(
              loss_global_valid_w),
          'Global Valid Accuracy: {:.8f}'.format(acc_global_valid))

    return loss_global_valid, loss_global_valid_w, acc_global_valid


# Print Total Losses
def print_total_loss(loss_train_mean, loss_valid_mean, loss_train_w_mean,
                     loss_valid_w_mean, profit=None):
    print('------------------------------------------------------')
    print('Total Train LogLoss: {:.8f}\n'.format(loss_train_mean),
          'Total Validation LogLoss: {:.8f}\n'.format(loss_valid_mean),
          'Total Train LogLoss with Weight: {:.8f}\n'.format(loss_train_w_mean),
          'Total Validation LogLoss with Weight: {:.8f}'.format(
              loss_valid_w_mean))
    if profit is not None:
        print('------------------------------------------------------')
        print('Test Profit: {:.8f}%\n'.format(profit * 100))


# Print Total Loss of Global Validation Set
def print_total_global_valid_loss_and_acc(pred_global_valid_mean,
                                          y_global_valid,
                                          loss_global_valid_mean,
                                          loss_global_valid_w_mean):
    acc_total_global_valid = get_accuracy(pred_global_valid_mean,
                                          y_global_valid)
    print('------------------------------------------------------')
    print('Total Global Valid LogLoss: {:.8f}\n'.format(loss_global_valid_mean),
          'Total Global Valid LogLoss with Weight: {:.8f}\n'.format(
              loss_global_valid_w_mean),
          'Total Global Valid Accuracy: {:.8f}'.format(acc_total_global_valid))

    return acc_total_global_valid


# Print Prediction of Positive Era Rate
def print_positive_rate_test(era_sign_test=None):
    if era_sign_test is None:
        era_sign_test = load_pkl_to_data(
            cfg.prejudged_data_path + 'binary_era_sign_test.p')

    positive_rate_test = np.sum(era_sign_test) / len(era_sign_test)

    print('------------------------------------------------------')
    print('Positive Rate Prediction of Test Set: {:.8f}%'.format(
        positive_rate_test * 100))


# Print Information of Cross Validation
def print_cv_info(cv_count, n_cv):
    print('======================================================')
    print('Training on the Cross Validation Set: {}/{}'.format(cv_count, n_cv))


# Check if directories exit or not
def check_dir(path_list):
    for dir_path in path_list:
        if not isdir(dir_path):
            os.makedirs(dir_path)


# Check if directories exit or not
def check_dir_model(pred_path, loss_log_path=None):
    if loss_log_path is not None:
        path_list = [pred_path,
                     pred_path + 'cv_results/',
                     pred_path + 'cv_pred_train/',
                     pred_path + 'final_results/',
                     pred_path + 'final_pred_train/',
                     loss_log_path]
    else:
        path_list = [pred_path,
                     pred_path + 'cv_results/',
                     pred_path + 'cv_pred_train/']

    check_dir(path_list)


# Get Accuracy
def get_accuracy(pred, label):
    prediction = [1 if pro > 0.5 else 0 for pro in pred]
    correct_pred = [1 if p == y else 0 for p, y in zip(prediction, label)]
    accuracy = np.mean(correct_pred)

    return accuracy


# Get Accuracies of Eras
def get_era_accuracy(pred, y, e, show_accuracy):
    pred_sorted = np.zeros_like(pred, dtype=np.float64)
    y_sorted = np.zeros_like(y, dtype=np.float64)
    e_sorted = np.zeros_like(e, dtype=int)

    for raw, idx in enumerate(np.argsort(e)):
        pred_sorted[raw] = pred[idx]
        y_sorted[raw] = y[idx]
        e_sorted[raw] = int(e[idx])

    era_index = []
    accuracy_eras = {}
    iter_era = e_sorted[0]

    for i, ele in enumerate(e_sorted):

        if i == len(e_sorted) - 1:
            pred_era = pred_sorted[era_index]
            y_era = y_sorted[era_index]
            acc_era = get_accuracy(pred_era, y_era)
            accuracy_eras[iter_era] = acc_era
            if show_accuracy:
                print('Accuracy of Era-{}: {:.3f}%'.format(iter_era,
                                                           acc_era * 100))
        elif ele == iter_era:
            era_index.append(i)
        else:
            pred_era = pred_sorted[era_index]
            y_era = y_sorted[era_index]
            acc_era = get_accuracy(pred_era, y_era)
            accuracy_eras[iter_era] = acc_era
            if show_accuracy:
                print('Accuracy of Era-{}: {:.3f}%'.format(iter_era,
                                                           acc_era * 100))
            iter_era = ele
            era_index = [i]

    return accuracy_eras


# Get Accuracies of Eras of  Train Probabilities
def get_train_era_accuracy(pred, y, e, show_accuracy):
    era_index = []
    accuracy_eras = {}
    iter_era = e[0]

    for i, ele in enumerate(e):

        if i == len(e) - 1:
            pred_era = pred[era_index]
            y_era = y[era_index]
            acc_era = get_accuracy(pred_era, y_era)
            accuracy_eras[iter_era] = acc_era
            if show_accuracy:
                print('Accuracy of Era-{}: {:.3f}%'.format(iter_era,
                                                           acc_era * 100))
        elif ele == iter_era:
            era_index.append(i)
        else:
            pred_era = pred[era_index]
            y_era = y[era_index]
            acc_era = get_accuracy(pred_era, y_era)
            accuracy_eras[iter_era] = acc_era
            if show_accuracy:
                print('Accuracy of Era-{}: {:.3f}%'.format(iter_era,
                                                           acc_era * 100))
            iter_era = ele
            era_index = [i]

    return accuracy_eras


# Print and Get Accuracy of Eras
def print_and_get_accuracy(pred_train, y_train, e_train, pred_valid, y_valid,
                           e_valid, show_accuracy):
    acc_train_cv = get_accuracy(pred_train, y_train)
    acc_valid_cv = get_accuracy(pred_valid, y_valid)

    if show_accuracy:
        print('------------------------------------------------------')
        print('Accuracies of CV:')
        print('Accuracy of Train CV: {:.3f}%'.format(acc_train_cv * 100))
        print('Accuracy of Validation CV: {:.3f}%'.format(acc_valid_cv * 100))
        print('------------------------------------------------------')
        print('Accuracies of Train Eras:')

    acc_train_era = get_era_accuracy(pred_train, y_train, e_train,
                                     show_accuracy)

    if show_accuracy:
        print('------------------------------------------------------')
        print('Accuracies of Validation Eras:')

    acc_valid_era = get_era_accuracy(pred_valid, y_valid, e_valid,
                                     show_accuracy)

    return acc_train_cv, acc_valid_cv, acc_train_era, acc_valid_era


# Print and Get Accuracies of Eras of  Train Probabilities
def print_and_get_train_accuracy(pred_train, y_train, e_train, show_accuracy):
    acc_train = get_accuracy(pred_train, y_train)

    if show_accuracy:
        print('------------------------------------------------------')
        print('Total Train Accuracy: {:.3f}%'.format(acc_train * 100))
        print('------------------------------------------------------')
        print('Accuracies of Total Train Eras:')

    acc_train_era = get_train_era_accuracy(pred_train, y_train, e_train,
                                           show_accuracy)

    return acc_train, acc_train_era


# Check If a CV is a Bad CV
def check_bad_cv(trained_cv, valid_era):
    cv_is_trained = any(set(valid_era) == i_cv for i_cv in trained_cv)
    if cv_is_trained:
        print('[W] This CV split has been chosen, choosing another one...')

    negative_era_counter = 0
    for era in valid_era:
        if era in cfg.preprocess.negative_era_list:
            negative_era_counter += 1
    bad_num_negative_era = negative_era_counter not in [0, 1]
    if bad_num_negative_era:
        print(
            '[W] Bad number of negative era in this CV split, choosing another one...')

    is_bad_cv = cv_is_trained or bad_num_negative_era

    return is_bad_cv


# Calculate Means
def calculate_means(pred_test_total, pred_train_total, loss_train_total,
                    loss_valid_total, loss_train_w_total, loss_valid_w_total,
                    weights=None):
    if weights is None:
        pred_test_mean = np.mean(np.array(pred_test_total), axis=0)
        pred_train_mean = np.mean(np.array(pred_train_total), axis=0)
        loss_train_mean = np.mean(loss_train_total)
        loss_valid_mean = np.mean(loss_valid_total)
        loss_train_w_mean = np.mean(loss_train_w_total)
        loss_valid_w_mean = np.mean(loss_valid_w_total)
    else:
        pred_test_mean = np.average(np.array(pred_test_total), axis=0,
                                    weights=weights)
        pred_train_mean = np.average(np.array(pred_train_total), axis=0,
                                     weights=weights)
        loss_train_mean = np.average(loss_train_total, weights=weights)
        loss_valid_mean = np.average(loss_valid_total, weights=weights)
        loss_train_w_mean = np.average(loss_train_w_total, weights=weights)
        loss_valid_w_mean = np.average(loss_valid_w_total, weights=weights)

    return pred_test_mean, pred_train_mean, loss_train_mean, \
           loss_valid_mean, loss_train_w_mean, loss_valid_w_mean


# Calculate Global Validation Means
def calculate_global_valid_means(pred_global_valid_total,
                                 loss_global_valid_total,
                                 loss_global_valid_w_total, weights=None):
    if weights is None:
        pred_global_valid_mean = np.mean(np.array(pred_global_valid_total),
                                         axis=0)
        loss_global_valid_mean = np.mean(loss_global_valid_total)
        loss_global_valid_w_mean = np.mean(loss_global_valid_w_total)
    else:
        pred_global_valid_mean = np.average(np.array(pred_global_valid_total),
                                            axis=0, weights=weights)
        loss_global_valid_mean = np.average(loss_global_valid_total,
                                            weights=weights)
        loss_global_valid_w_mean = np.average(loss_global_valid_w_total,
                                              weights=weights)

    return pred_global_valid_mean, loss_global_valid_mean, loss_global_valid_w_mean


# Calculate Boost Round Means
def calculate_boost_round_means(train_loss_round_total, valid_loss_round_total,
                                weights=None,
                                global_valid_loss_round_total=None):
    if weights is None:
        train_loss_round_mean = np.mean(np.array(train_loss_round_total),
                                        axis=0)
        valid_loss_round_mean = np.mean(np.array(valid_loss_round_total),
                                        axis=0)
    else:
        train_loss_round_mean = np.average(np.array(train_loss_round_total),
                                           axis=0, weights=weights)
        valid_loss_round_mean = np.average(np.array(valid_loss_round_total),
                                           axis=0, weights=weights)

    if global_valid_loss_round_total is not None:
        if weights is None:
            global_valid_loss_round_mean = np.mean(
                np.array(global_valid_loss_round_total), axis=0)
        else:
            global_valid_loss_round_mean = np.average(
                np.array(global_valid_loss_round_total), axis=0,
                weights=weights)
        return train_loss_round_mean, valid_loss_round_mean, global_valid_loss_round_mean
    else:
        return train_loss_round_mean, valid_loss_round_mean


# Generate random int list
def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (
    int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for _ in range(length):
        random_list.append(random.randint(start, stop))

    return random_list


# Get Path of Boost Round Train
def get_boost_round_log_path(boost_round_log_path, model_name, param_name_list,
                             param_value_list, append_info):
    param_info = ''
    param_name = ''
    for i in range(len(param_name_list)):
        if param_name_list[i] == 'cv_weights':
            param_info += '_' + get_simple_param_name(param_name_list[i])
        else:
            param_info += '_' + get_simple_param_name(
                param_name_list[i]) + '-' + str(param_value_list[i])
        param_name += '_' + get_simple_param_name(param_name_list[i])

    boost_round_log_path += model_name + '/'
    check_dir([boost_round_log_path])
    boost_round_log_path += model_name + '_' + append_info + '/'
    check_dir([boost_round_log_path])
    boost_round_log_path += model_name + param_name + '/'
    check_dir([boost_round_log_path])
    boost_round_log_path += model_name + param_info + '/'
    check_dir([boost_round_log_path])

    return boost_round_log_path, param_name


# Get Path of Boost Round Train
def get_boost_round_log_upper_path(boost_round_log_path, model_name,
                                   param_name_list, append_info):
    param_name = ''
    for i in range(len(param_name_list)):
        param_name += '_' + get_simple_param_name(param_name_list[i])

    boost_round_log_path += model_name + '/'
    check_dir([boost_round_log_path])
    boost_round_log_path += model_name + '_' + append_info + '/'
    check_dir([boost_round_log_path])
    boost_round_log_path += model_name + param_name + '/'
    check_dir([boost_round_log_path])

    return boost_round_log_path


# Get Path of Grid Search
def get_grid_search_log_path(csv_log_path, model_name, param_name_list,
                             param_value_list, append_info):
    param_info = ''
    param_name = ''
    for i in range(len(param_name_list)):
        if param_name_list[i] == 'cv_weights':
            param_info += '_' + get_simple_param_name(param_name_list[i])
        else:
            param_info += '_' + get_simple_param_name(
                param_name_list[i]) + '-' + str(param_value_list[i])
        param_name += '_' + get_simple_param_name(param_name_list[i])

    csv_log_path += model_name + '/'
    check_dir([csv_log_path])
    csv_log_path += model_name + '_' + append_info + '/'
    check_dir([csv_log_path])
    csv_log_path += model_name + param_name + '/'
    check_dir([csv_log_path])
    csv_log_path += model_name

    return csv_log_path, param_name, param_info


# Get Category Feature's Index
def get_idx_category(x_train, use_multi_group):
    if cfg.group_list is not None:
        if use_multi_group:
            print('------------------------------------------------------')
            print('[W] Using Multi Groups...')
            idx_category = [x_train.shape[1] - 2, x_train.shape[1] - 1]
        else:
            print('------------------------------------------------------')
            print('[W] Using Single Group...')
            idx_category = [x_train.shape[1] - 1]
        print('------------------------------------------------------')
        print('Index of categorical feature: {}'.format(idx_category))
        print('------------------------------------------------------')
    else:
        idx_category = None

    return idx_category


# Get Cross Validation Arguments
def get_cv_args(cv_args, append_info):
    cv_args_copy = copy.deepcopy(cv_args)
    if 'n_valid' in cv_args:
        n_valid = cv_args_copy['n_valid']
    elif 'valid_rate' in cv_args:
        n_valid = cv_args_copy['valid_rate']
    else:
        n_valid = ''
    n_cv = cv_args_copy['n_cv']
    n_era = cv_args_copy['n_era']
    cv_seed = cv_args_copy['cv_seed']

    # Append Information
    if append_info is None:
        append_info = 'v-' + str(n_valid) + '_c-' + str(n_cv) + '_e-' + str(
            n_era)
        if 'window_size' in cv_args_copy:
            append_info += '_w-' + str(cv_args_copy['window_size'])

    return cv_args_copy, n_valid, n_cv, n_era, cv_seed


# Get Simple Parameter's Name
def get_simple_param_name(param_name):
    param_name_convert_dict = {
        'num_boost_round': 'nbr',
        'learning_rate': 'lr',
        'gamma': 'gma',
        'max_depth': 'mdp',
        'min_child_weight': 'mcw',
        'subsample': 'sbs',
        'colsample_bytree': 'cst',
        'colsample_bylevel': 'csl',
        'lambda': 'lmd',
        'alpha': 'aph',
        'early_stopping_rounds': 'esr',
        'objective': 'obj',
        'seed': 's',
        'num_leaves': 'nl',
        'min_data_in_leaf': 'mdl',
        'min_sum_hessian_in_leaf': 'msh',
        'feature_fraction': 'ff',
        'feature_fraction_seed': 'ffs',
        'bagging_fraction': 'bf',
        'bagging_freq': 'bfq',
        'bagging_seed': 'bs',
        'lambda_l1': 'll1',
        'lambda_l2': 'll2',
        'min_gain_to_split': 'mgs',
        'max_bin': 'mb',
        'min_data_in_bin': 'mdb',
        'iterations': 'itr',
        'depth': 'dpt',
        'l2_leaf_reg': 'l2r',
        'bagging_temperature': 'btp',
        'border': 'bdr',
        'border_count': 'bct',
        'od_pval': 'odp',
        'od_wait': 'odw',
        'od_type': 'odt',
        'gradient_iterations': 'gitr',
        'random_seed': 's',
        'ctr_description': 'ctrd',
        'ctr_border_count': 'ctrb',
        'ctr_leaf_count_limit': 'ctrl',
        'ignored_features': 'igf',
        'epochs': 'epo',
        'unit_number': 'unn',
        'keep_probability': 'kpp',
        'batch_size': 'bs',
        'n_valid': 'va',
        'n_cv': 'cv',
        'n_era': 'er',
        'valid_rate': 'vr',
        'window_size': 'ws',
        'cv_weights': 'cw',
        'model_name': 'mdl',
        'start_year': 'st',
        'valid_range': 'va',
        'frequency': 'fq',
        'hw_seasonal': 'hws',
        'fill_mode': 'fm'
    }

    if param_name in param_name_convert_dict.keys():
        return param_name_convert_dict[param_name]
    else:
        return param_name


def thin_line():
    print('-' * 55)


def thick_line():
    print('=' * 55)


def get_suffix(sample_mode):

    if sample_mode == 'day':
        suffix = 'DAY'
    elif sample_mode == 'month':
        suffix = 'MONTH'
    elif sample_mode == 'year':
        suffix = 'YEAR'
    else:
        raise ValueError('Wrong Mode!')

    return suffix


# Save Log to csv File
def save_log_to_csv(log_path, grid_search_tuple_dict: dict,
                    cost, idx, append_info=''):
    check_dir([log_path])
    param_name_list = []
    param_value_list = []
    for name in grid_search_tuple_dict.keys():
        param_name_list.append(name)
        param_value_list.append(grid_search_tuple_dict[name])
    file_path = os.path.join(log_path, 'cost_log{}.csv'.format(append_info))

    if not os.path.isfile(file_path):
        with open(file_path, 'w') as f:
            header = ['idx', *param_name_list, 'cost']
            writer = csv.writer(f)
            writer.writerow(header)

    with open(file_path, 'a') as f:
        log = [idx, *param_value_list, cost]
        writer = csv.writer(f)
        writer.writerow(log)


def get_month_features():
    df = pd.read_csv(os.path.join(cfg.preprocessed_path, 'total_month.csv'))
    months = []
    prices = []
    for row_i in df.iterrows():
        row = row_i[1]
        months.append(
            str(int(row['YEAR'])) + '-{0:02d}'.format(int(row['MONTH'])))
        prices.append(row['CONTPRICE_MONTH'])
    dict_now = {month: price for month, price in zip(months, prices)}
    dict_pre = {}
    for i in range(1, len(months)):
        dict_pre[months[i]] = prices[i-1]
    return {'now': dict_now, 'pre': dict_pre}


def save_importance(mode, idx, features_list, importance_list):

    check_dir([cfg.log_path])
    file_path = os.path.join(cfg.log_path,
                             'feature_importance_{}.csv'.format(mode))

    f_row = [idx, *features_list]
    i_row = [idx, *importance_list]

    if not os.path.isfile(file_path):
        with open(file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['idx'])

    with open(file_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(f_row)
        writer.writerow(i_row)
