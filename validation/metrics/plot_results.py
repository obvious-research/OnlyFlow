import os
from os import abort

import pandas
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from ast import literal_eval

matplotlib.use('pgf')
plt.rcParams.update({
    "font.family": "serif",
    "pgf.rcfonts": False,
    "pgf.texsystem": "pdflatex",
    "text.usetex": False,
    "pgf.preamble": "\n".join([
         r"\usepackage[utf8x]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{cmbright}",
    ]),
    "text.latex.preamble": r"""
        \usepackage{amsmath}
        \usepackage{amssymb}
        \def\mathdefault#1{#1}
    """,
    "savefig.bbox": "tight",
    "mathtext.default": "regular",  # Ensure math text uses regular font
    "mathtext.fontset": "cm",       # Use Computer Modern fonts
})

folder_path_temporal = '/Users/mathis/Desktop/Obvious/24-11-FlowCtrl/output_2167578/artifacts/results/'
folder_path_temporal_spatial = '/Users/mathis/Desktop/Obvious/24-11-FlowCtrl/output_2168692/artifacts/results/'


metrics_dict = {'clip_score': 'CLIP Score',
                'diff_optical_flow': 'optical flow absolute mean difference',
                'fvd': 'FVD',
                'ssim': 'SSIM',
                'psnr': 'PSNR',
                'lpips': 'LPIPS'}

def load_results(folder_path):
    df_list = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            df = pandas.read_csv(folder_path + file, index_col='id')
            df_list.append(df)
    df = pandas.concat(df_list)
    for column in df.columns:
        # if type of column is object, convert using literal_eval
        if df[column].dtype == object:
            df[column] = df[column].apply(literal_eval)

    return df

def apply_operations(df, op):
    df['fvd'] = df['fvd'].apply(op)
    df['ssim'] = df['ssim'].apply(op)
    df['psnr'] = df['psnr'].apply(op)
    df['lpips'] = df['lpips'].apply(op)

    return df

def plot_results(df_1, df_2, metric, op, std=False):
    fig, ax1 = plt.subplots()
    metric_grouped_1 = df_1.groupby('flow_scale')[metric]
    metric_grouped_2 = df_2.groupby('flow_scale')[metric]


    mean_1 = metric_grouped_1.mean()
    mean_2 = metric_grouped_2.mean()

    if std == 1:
        std_1 = metric_grouped_1.std()
        std_2 = metric_grouped_2.std()
    else:
        first1 = df_1.groupby('id').first()
        first2 = df_2.groupby('id').first()

        mean_1 = (df_1 / first1).groupby('flow_scale')[metric].mean()
        mean_2 = (df_2 / first2).groupby('flow_scale')[metric].mean()

        std_1 = (df_1 / first1).groupby('flow_scale')[metric].std()
        std_2 = (df_2 / first2).groupby('flow_scale')[metric].std()


    if std == 0 or std == 1 or std == 3:
        l1 = ax1.plot(mean_1.index, mean_1, marker='x', color='tab:orange', label='temporal')
        l2 = ax1.plot(mean_2.index, mean_2, marker='o', color='tab:blue', label='temporal and spatial')
    elif std == 2:
        l1 = ax1.plot(mean_1.index, mean_1, marker='x', color='tab:orange', label='temporal')
        l2 = ax1.plot(mean_2.index, mean_2, marker='o', color='tab:orange', label='temporal and spatial')

    if std == 1 or std == 3:
        if metric == 'diff_optical_flow':
            l3 = ax1.fill_between(mean_1.index, (mean_1 - std_1).clip(0, None), mean_1 + std_1, alpha=0.2, color='tab:orange')
            l4 = ax1.fill_between(mean_2.index, (mean_2 - std_2).clip(0, None), mean_2 + std_2, alpha=0.2, color='tab:blue')
        else:
            l3 = ax1.fill_between(mean_1.index, mean_1 - std_1, mean_1 + std_1, alpha=0.2, color='tab:orange')
            l4 = ax1.fill_between(mean_2.index, mean_2 - std_2, mean_2 + std_2, alpha=0.2, color='tab:blue')
    elif std == 2:
        ax2 = ax1.twinx()
        l3 = ax2.plot(std_1.index, std_1, marker='x', color='tab:blue', label='temporal')
        l4 = ax2.plot(std_2.index, std_2, marker='o', color='tab:blue', label='temporal and spatial')

    if std == 2:
        ax1.set_ylabel(metric, color='tab:orange')
        ax2.set_ylabel('std', color='tab:blue')
    else:
        ax1.set_ylabel(metrics_dict[metric])

    leg = plt.legend()
    if std == 2:
        for marker in leg.legend_handles:
            marker.set_color('black')

    if metric in ['fvd', 'clip_score', 'diff_optical_flow']:
        pass
        #plt.title(f'{metric} evolution')
    else:
        pass
        #plt.title(f'Temporal {op.__name__} of {metric} accross frames')
    ax1.set_xlabel('optical flow strength in attention')

    plt.savefig(f'./graphics/{metric}_{op.__name__}{f"_std_{std}"}.png')
    plt.savefig(f'./graphics/{metric}_{op.__name__}{f"_std_{std}"}.pgf')
    plt.close()

def custom_index(df):
    df.reset_index(inplace=True)
    df.index = pandas.MultiIndex.from_frame(df.loc[:, ['id', 'flow_scale']])
    return df.drop(columns=['id', 'flow_scale']).sort_index()


def main():

    os.makedirs('../graphics', exist_ok=True)

    df_1_raw = custom_index(load_results(folder_path_temporal))
    df_2_raw = custom_index(load_results(folder_path_temporal_spatial))
    print(df_1_raw.columns)

    print("Num elements in df_1:", len(df_1_raw))
    print("Num elements in df_2:", len(df_2_raw))

    for op in [np.mean]:#, np.median, np.max, np.min]:
        df_1, df_2 = df_1_raw.copy(), df_2_raw.copy()
        df_1 = apply_operations(df_1, op)
        df_2 = apply_operations(df_2, op)
        for metric in ['fvd', 'clip_score', 'diff_optical_flow', 'ssim', 'psnr', 'lpips']:
            for std in [3]:
                plot_results(df_1, df_2, metric, op, std)



if __name__ == '__main__':
    main()
