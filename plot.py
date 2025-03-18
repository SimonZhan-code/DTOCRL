import os
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from rich import print
from utils.tool import get_offline_dataset
from tqdm import tqdm
import pickle
from joblib import Parallel, delayed
import re
import torch
import d4rl
import gym

# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

colors = [
    '#1f77b4',  # blue
    '#ff7f0e',   # orange
    '#2ca02c',   # green
    '#d62728',   # red
    '#9467bd',   # purple
    '#8c564b',   # brown
    '#e377c2',   # pink
    '#7f7f7f',   # gray
    '#bcbd22',   # yellow-green
    '#17becf',   # cyan
]

def normalize_score(env, re):
    if 'halfcheetah' in env:
        re_max = 12135.0
        re_min = -280.178953
    elif 'walker2d' in env:
        re_max = 4592.3
        re_min = 1.629008
    elif 'hopper' in env:
        re_max = 3234.3
        re_min = -20.272305
    elif 'ant' in env:
        re_max = 3879.7
        re_min = -325.6

    return (re - re_min) / (re_max - re_min)

def split_list(lst, n=100):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def get_single_file_metric_dict(file_path, metric_name):
    metric_dict = {}
    ea = event_accumulator.EventAccumulator(file_path)
    ea.Reload()
    for tag in ea.Tags()['scalars']:
        if tag != metric_name:
            continue
        if tag not in metric_dict.keys():
            metric_dict.update({tag: {
                'step': [],
                'value': [],
            }})
        for event in ea.scalars.Items(tag):
            metric_dict[tag]['step'].append(event.step)
            metric_dict[tag]['value'].append(event.value)
    return metric_dict

def get_alg_metric_dict(alg_file_path, metric_name):
    metric_dict = {}

    for file_path in alg_file_path:
        metric = get_single_file_metric_dict(file_path, metric_name)
        print(file_path, metric.keys())
        for metric_key in metric.keys():
            if metric_key not in metric_dict.keys():
                metric_dict.update({metric_key: {
                                        'step': [],
                                        'value': [],}
                                    })
            if len(metric[metric_key]['value']) != 100:
                continue
            metric_dict[metric_key]['step'] += metric[metric_key]['step']
            metric_dict[metric_key]['value'] += metric[metric_key]['value']
    return metric_dict

def find_tensorboard_event_files(root_dir):
    event_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_file_path = os.path.join(root, file)
                event_files.append(event_file_path)
    return event_files

def plot_fig_from_event(env, delay):

    alg_file_path_dict = {
        
        # explicit belief baseline
        # 'DATS': { 
        #     'path': find_tensorboard_event_files(f'new_logs_online/aug_sac_w_br/mlp/{env}/{delay}/'),
        #     'metric': {},
        #     'color': colors[0],
        # },
        # 'DEER': { 
        #     'path': find_tensorboard_event_files(f'new_logs_online/aug_sac_w_br/gru/{env}/{delay}/'),
        #     'metric': {},
        #     'color': colors[1],
        # },
        # 'D-SAC': { 
        #     'path': find_tensorboard_event_files(f'new_logs_online/aug_sac_w_br/trans/{env}/{delay}/'),
        #     'metric': {},
        #     'color': colors[2],
        # },
        # 'DBT-SAC (ours)': { 
        #     'path': find_tensorboard_event_files(f'new_logs_online/bpql_belief_generate_8_step_w_br/trans/{env}/{delay}/'),
        #     'metric': {},
        #     'color': colors[6],
        # },

        # implicit belief baseline
        # 'A-SAC': { 
        #     'path': find_tensorboard_event_files(f'logs/aug_sac/{env}-random-v2/D_{delay}'),
        #     'metric': {},
        #     'color': colors[3],
        # },
        # 'ADRL': { 
        #     'path': find_tensorboard_event_files(f'logs/adrl/{env}-random-v2/D_{delay}'),
        #     'metric': {},
        #     'color': colors[4],
        # },
        # 'BPQL': { 
        #     'path': find_tensorboard_event_files(f'logs/bpql/{env}-random-v2/D_{delay}'),
        #     'metric': {},
        #     'color': colors[5],
        # },

        # 'DBT-SAC (ours)': { 
        #     'path': find_tensorboard_event_files(f'new_logs_online/bpql_belief_generate_8_step_w_br/trans/{env}/{delay}/'),
        #     'metric': {},
        #     'color': colors[6],
        # },


        # ablation
        # 'DBT-SAC 1-step': { 
        #     'path': find_tensorboard_event_files(f'new_logs_online/bpql_none_step_w_br/trans/{env}/{delay}/'),
        #     'metric': {},
        #     'color': colors[7],
        # },
        # 'DBT-SAC 2-step': { 
        #     'path': find_tensorboard_event_files(f'new_logs_online/bpql_belief_generate_2_step_w_br/trans/{env}/{delay}/'),
        #     'metric': {},
        #     'color': colors[8],
        # },
        # 'DBT-SAC 4-step': { 
        #     'path': find_tensorboard_event_files(f'new_logs_online/bpql_belief_generate_4_step_w_br/trans/{env}/{delay}/'),
        #     'metric': {},
        #     'color': colors[9],
        # },
        # 'DBT-SAC 8-step': { 
        #     'path': find_tensorboard_event_files(f'new_logs_online/bpql_belief_generate_8_step_w_br/trans/{env}/{delay}/'),
        #     'metric': {},
        #     'color': colors[6],
        # },




        # adrl
        # 'ADRL': { 
        #     'path': find_tensorboard_event_files(f'logs/adrl/{env}-random-v2/D_{delay}'),
        #     'metric': {},
        #     'color': colors[0],
        # },

        # 'DBT-SAC 1-step': { 
        #     'path': find_tensorboard_event_files(f'new_logs_online/dbt_adrl_1_step_w_br/trans/{env}/{delay}/'),
        #     'metric': {},
        #     'color': colors[1],
        # },
        # 'DBT-SAC 2-step': { 
        #     'path': find_tensorboard_event_files(f'new_logs_online/dbt_adrl_2_step_w_br/trans/{env}/{delay}/'),
        #     'metric': {},
        #     'color': colors[2],
        # },
        # 'DBT-SAC 4-step': { 
        #     'path': find_tensorboard_event_files(f'new_logs_online/dbt_adrl_4_step_w_br/trans/{env}/{delay}/'),
        #     'metric': {},
        #     'color': colors[3],
        # },
        # 'DBT-SAC 8-step': { 
        #     'path': find_tensorboard_event_files(f'new_logs_online/dbt_adrl_8_step_w_br/trans/{env}/{delay}/'),
        #     'metric': {},
        #     'color': colors[4],
        # },
        # 'DBT-SAC 16-step': { 
        #     'path': find_tensorboard_event_files(f'new_logs_online/dbt_adrl_16_step_w_br/trans/{env}/{delay}/'),
        #     'metric': {},
        #     'color': colors[4],
        # },
        # 'DBT-SAC 32-step': { 
        #     'path': find_tensorboard_event_files(f'new_logs_online/dbt_adrl_32_step_w_br/trans/{env}/{delay}/'),
        #     'metric': {},
        #     'color': colors[4],
        # },
        # 'DBT-SAC 64-step': { 
        #     'path': find_tensorboard_event_files(f'new_logs_online/dbt_adrl_64_step_w_br/trans/{env}/{delay}/'),
        #     'metric': {},
        #     'color': colors[4],
        # },
        # 'DBT-SAC 128-step': { 
        #     'path': find_tensorboard_event_files(f'new_logs_online/dbt_adrl_128_step_w_br/trans/{env}/{delay}/'),
        #     'metric': {},
        #     'color': colors[4],
        # },


        
        # 'DBT-SAC 1-step': { 
        #     'path': find_tensorboard_event_files(f'new_logs_online/dbt_adrl_1_step_w_br/trans/{env}/{delay}/'),
        #     'metric': {},
        #     'color': colors[1],
        # },
        # 'DBT-SAC 2-step': { 
        #     'path': find_tensorboard_event_files(f'new_logs_online/mixed_belief_generate_2_step_w_br/trans/{env}/{delay}/'),
        #     'metric': {},
        #     'color': colors[2],
        # },
        # 'DBT-SAC 4-step': { 
        #     'path': find_tensorboard_event_files(f'new_logs_online/mixed_belief_generate_4_step_w_br/trans/{env}/{delay}/'),
        #     'metric': {},
        #     'color': colors[3],
        # },
        # 'DBT-SAC 8-step': { 
        #     'path': find_tensorboard_event_files(f'new_logs_online/mixed_belief_generate_8_step_w_br/trans/{env}/{delay}/'),
        #     'metric': {},
        #     'color': colors[4],
        # },



        # stochastic version

        # 'DATS': { 
        #     'path': find_tensorboard_event_files(f'baseline_logs_stochastic/bpql_w_br/mlp/{env}/{delay}/'),
        #     'metric': {},
        #     'color': colors[0],
        # },
        # 'DEER': { 
        #     'path': find_tensorboard_event_files(f'baseline_logs_stochastic/bpql_w_br/gru/{env}/{delay}/'),
        #     'metric': {},
        #     'color': colors[1],
        # },
        # 'D-SAC': { 
        #     'path': find_tensorboard_event_files(f'baseline_logs_stochastic/bpql_w_br/trans/{env}/{delay}/'),
        #     'metric': {},
        #     'color': colors[2],
        # },
        # 'DBT-SAC (ours)': { 
        #     'path': find_tensorboard_event_files(f'new_logs_online/dbt_bpql_8_step_w_br/trans/{env}/stochastic_{delay}/'),
        #     'metric': {},
        #     'color': colors[6],
        # },



        # 'A-SAC': { 
        #     'path': find_tensorboard_event_files(f'logs/aug_sac_stochastic/{env}-random-v2/D_{delay}'),
        #     'metric': {},
        #     'color': colors[3],
        # },
        # 'BPQL': { 
        #     'path': find_tensorboard_event_files(f'logs/bpql_stochastic/{env}-random-v2/D_{delay}'),
        #     'metric': {},
        #     'color': colors[4],
        # },
        # 'ADRL': { 
        #     'path': find_tensorboard_event_files(f'logs/adrl_stochastic/{env}-random-v2/D_{delay}'),
        #     'metric': {},
        #     'color': colors[5],
        # },
        # 'DBT-SAC (ours)': { 
        #     'path': find_tensorboard_event_files(f'new_logs_online/dbt_bpql_8_step_w_br/trans/{env}/stochastic_{delay}/'),
        #     'metric': {},
        #     'color': colors[6],
        # },


        
        'DBT-SAC 1-step': { 
            'path': find_tensorboard_event_files(f'new_logs_online/dbt_bpql_1_step_w_br/trans/{env}/stochastic_{delay}/'),
            'metric': {},
            'color': colors[7],
        },
        'DBT-SAC 2-step': { 
            'path': find_tensorboard_event_files(f'new_logs_online/dbt_bpql_2_step_w_br/trans/{env}/stochastic_{delay}/'),
            'metric': {},
            'color': colors[8],
        },
        'DBT-SAC 4-step': { 
            'path': find_tensorboard_event_files(f'new_logs_online/dbt_bpql_4_step_w_br/trans/{env}/stochastic_{delay}/'),
            'metric': {},
            'color': colors[9],
        },
        'DBT-SAC 8-step': { 
            'path': find_tensorboard_event_files(f'new_logs_online/dbt_bpql_8_step_w_br/trans/{env}/stochastic_{delay}/'),
            'metric': {},
            'color': colors[6],
        },


    }
    metric_name = 'eval_r'
    plt.figure(figsize=(8, 6))
    for alg in alg_file_path_dict.keys():
        if 'diff' in alg:
            metric_name = 'eval/return'
        else:
            metric_name = 'eval_r'
        alg_file_path_dict[alg]['metric'] = get_alg_metric_dict(alg_file_path_dict[alg]['path'], metric_name)
        steps = alg_file_path_dict[alg]['metric'][metric_name]['step']
        values = alg_file_path_dict[alg]['metric'][metric_name]['value']
        norm_values = [normalize_score(env, v) for v in values]

        sns.lineplot(x=steps, y=norm_values, err_style='band', label=alg, linewidth=3, color=alg_file_path_dict[alg]['color'])
        # sns.lineplot(x=steps, y=values, err_style='band', label=alg, linewidth=3)
        # sns.lineplot(x=steps, y=norm_values, err_style='band', label=alg, linewidth=3, color=alg_file_path_dict[alg]['color'])
        # sns.lineplot(x=steps, y=values, err_style='band', label=alg, linewidth=2.5)
        

        performance = torch.tensor(split_list(norm_values, n=100))
        means = performance.mean(0)
        idx = means.argmax()
        mean = means[idx]
        stds = performance.std(0)
        std = stds[idx]
        info = f"{env}, {delay}, {alg}, ${mean:.2f}_{{\pm {std:.2f}}}$\n"
        print(info)
        with open('performance_eval.txt', 'a', encoding='utf-8') as file:
            file.write(info)

    plt.subplots_adjust(left=0.11,    # left side of the plot
                        right=0.99,    # right side
                        bottom=0.11,   # bottom margin
                        top=0.99)      # top margin
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Global Steps', fontsize=25)
    plt.ylabel('Return', fontsize=25)
    # plt.title(f'{delay} Delays')
    plt.legend(fontsize=18, loc=4)
    # plt.savefig(f'implicit_comparison_pdf/{env}_D{delay}.pdf')
    # plt.savefig(f'ablation_comparison_pdf/{env}_D{delay}.pdf')
    # plt.savefig(f'explicit_comparison_pdf/{env}_D{delay}.pdf')
    plt.savefig(f'{env}_D{delay}.pdf')
    plt.close()

def plot_fig_from_pickle(env, delay):
    plt.figure(figsize=(10, 6))
    # for alg in ['Aug_SAC', 'Latent']:
    for alg in ['BPQL/BPQL', 'BPQL/BPQL_Belief', 'BISIMULATION/Bisimulation']:
        with open(f'plot_figs/{alg}_{env}_D{delay}.pickle', 'rb') as f:
            data = pickle.load(f)
        steps = data['steps']
        values = data['values']
        sns.lineplot(x=steps, y=values, err_style='band', label=alg)


        from collections import defaultdict
        def calculate_mean(keys, values):
            data_dict = defaultdict(list)
            for key, value in zip(keys, values):
                data_dict[key].append(value)
            mean_dict = {}
            for key, value_list in data_dict.items():
                mean_value = sum(value_list) / len(value_list)
                mean_dict[key] = mean_value
            return mean_dict
        re = calculate_mean(steps, values)
        max_step = max(re, key=re.get)
        max_value = re[max_step]
        print(f'{alg} {env} {max_value}')

    plt.xlabel('Global Steps')
    plt.ylabel(f'R')
    plt.title(f'{delay} Delays')
    plt.savefig(f'plot_figs/{env}_D{delay}.png')
    plt.close()

envs = ['halfcheetah', 'hopper', 'walker2d']
delays = [8, 16, 32, 64, 128]
Parallel(n_jobs=1)(delayed(plot_fig_from_event)(env, delay) for env in envs for delay in delays)




def plot_belief_error():
    datasets = ['ant-random-v2', 'halfcheetah-random-v2', 'hopper-random-v2', 'walker2d-random-v2']
    dynamics = ['mlp', 'gru', 'trans']
    delays = [1, 2, 4, 8, 16, 32, 64, 128]
    error_dict = {}
    for dataset in datasets:
        if dataset not in error_dict.keys():
            error_dict.update({dataset: {}})
        for dynamic in dynamics:
            if dynamic not in error_dict[dataset].keys():
                error_dict[dataset].update({dynamic: {}})
            for delay in delays:
                if f'{delay}' not in error_dict[dataset][dynamic].keys():
                    error_dict[dataset][dynamic].update({f'{delay}': {}})
    with open('eval.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            print(line)
            pattern = r"(\w+)=([\w\.-]+)|(\w+\s\w+)=(\d+\.?\d*)\s?\\pm\s?(\d+\.?\d*)"

            matches = re.findall(pattern, line)
            print(matches)

            dataset = matches[0][1]
            dynamic = matches[1][1]
            delay = matches[2][1]
            mean = matches[3][-2]
            std = matches[3][-1]

            error_dict[dataset][dynamic][delay]['mean'] = float(mean)
            error_dict[dataset][dynamic][delay]['std'] = float(std)

    for dataset in datasets:
        plt.figure(figsize=(10,6))
        for dynamic in dynamics:
            means = []
            stds = []
            for delay in delays:
                means.append(error_dict[dataset][dynamic][f'{delay}']['mean'])
                stds.append(error_dict[dataset][dynamic][f'{delay}']['std'])
            print(delays)
            print(means)
            print(stds)

            plt.errorbar(np.array(delays[:6]), np.array(means[:6]), yerr=np.array(stds[:6]), label=dynamic)
        plt.legend()
        plt.savefig(f'{dataset}.png')
        # exit()


# plot_belief_error()

# Parallel(n_jobs=10)(delayed(plot_fig_from_pickle)(env, delay) for env in envs for delay in delays)

def plot_eval_performance():
    datasets = ['ant-random-v2', 'halfcheetah-random-v2', 'hopper-random-v2', 'walker2d-random-v2']
    dynamics = ['IMPLICIT', 'MLP', 'GRU', 'TRANS']
    delays = [1, 2, 4, 8, 16, 32, 64]
    error_dict = {}
    for dataset in datasets:
        if dataset not in error_dict.keys():
            error_dict.update({dataset: {}})
        for dynamic in dynamics:
            if dynamic not in error_dict[dataset].keys():
                error_dict[dataset].update({dynamic: {}})
            for delay in delays:
                if f'{delay}' not in error_dict[dataset][dynamic].keys():
                    error_dict[dataset][dynamic].update({f'{delay}': {}})
    with open('performance_eval.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            print(line)
            pattern = r'([a-zA-Z0-9\-]+),\s*(\d+),\s*([A-Z]+),\s*([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+)'

            # pattern = r'(\S+),\s*(\d+),\s*(\S+),\s*([\d\.]+),\s*([\d\.]+)'

            matches = re.findall(pattern, line)
            print(matches)

            dataset = matches[0][0]
            delay = matches[0][1]
            dynamic = matches[0][2]
            mean = matches[0][3]
            std = matches[0][4]

            error_dict[dataset][dynamic][delay]['mean'] = float(mean)
            error_dict[dataset][dynamic][delay]['std'] = float(std)

    for dataset in datasets:
        plt.figure(figsize=(10,6))
        for dynamic in dynamics:
            means = []
            stds = []
            for delay in delays:
                means.append(error_dict[dataset][dynamic][f'{delay}']['mean'])
                stds.append(error_dict[dataset][dynamic][f'{delay}']['std'])
            print(delays)
            print(means)
            print(stds)

            plt.errorbar(np.array(delays), np.array(means), yerr=np.array(stds), label=dynamic)
        plt.legend()
        plt.savefig(f'{dataset}.png')
# plot_eval_performance()