import torch
import pandas as pd
import numpy as np
import os

import modlamp.descriptors
import modlamp.analysis
import modlamp.sequences

from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from scipy.stats import mannwhitneyu

from attribute_util.amp.utils import phys_chem_propterties as phys
from matplotlib import font_manager

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from matplotlib import rcParams

import modlamp.descriptors
import modlamp.analysis
import modlamp.sequences

# =======================
# 1. seaborn 初始化（必须放最前）
# =======================
sns.set_theme(style="whitegrid", font_scale=1.2)

# =======================
# 2. 手动加载 Times New Roman 字体
# =======================
font_paths = [
    # "/geniusland/home/chenyaping/anaconda3/envs/myenv/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/times.ttf",
    "/geniusland/home/chenyaping/anaconda3/envs/myenv/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/timesbd.ttf",
    # "/geniusland/home/chenyaping/anaconda3/envs/myenv/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/timesi.ttf",
    # "/geniusland/home/chenyaping/anaconda3/envs/myenv/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/timesbi.ttf",
]
for p in font_paths:
    font_manager.fontManager.addfont(p)

# =======================
# 3. 强制设置 Matplotlib 使用 Times New Roman
# =======================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.weight"] = "bold" 
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 18.0

# 调试输出
from matplotlib import rcParams
print(rcParams['font.serif'])
print("font.family:", rcParams["font.family"])
print("font.serif:", rcParams["font.serif"])
print("findfont:", font_manager.findfont("Times New Roman"))


base_dir = "checkpoints"

dfs = []
train_data = pd.read_csv("./data/train.csv")
lora_path = os.path.join(base_dir, "mul", f"fb_epoch_0", "result.csv")
lora_data = pd.read_csv(lora_path)

mic_high = train_data["MIC_predict"] > 0.90
mic_low  = train_data["MIC_predict"] <= 0.90

tox_low  = train_data["TOXIN_predict"] < 0.38
tox_high = train_data["TOXIN_predict"] >= 0.38

# 4 个子数据集
train_data_01 = train_data[mic_low & tox_low]     # MIC <= 0.90, TOXIN < 0.38
train_data_00 = train_data[mic_low & tox_high]    # MIC <= 0.90, TOXIN >= 0.38
train_data_11 = train_data[mic_high & tox_low]    # MIC > 0.90, TOXIN < 0.38
train_data_10 = train_data[mic_high & tox_high]   # MIC > 0.90, TOXIN >= 0.38 

uniport = pd.read_csv("./data/Uniprot_0.csv")
uniport_data = uniport['Sequence'].tolist()

dfs = {}
dfs["Uniport"] = uniport_data
dfs["train"] = train_data["SEQUENCE"].tolist()
dfs["train_mic_low & tox_low"] = train_data_01["SEQUENCE"].tolist()
dfs["train_mic_low & tox_high"] = train_data_00["SEQUENCE"].tolist()
dfs["train_mic_high & tox_low"] = train_data_11["SEQUENCE"].tolist()
dfs["train_mic_high & tox_high"] = train_data_10["SEQUENCE"].tolist()
dfs["lora"] = lora_data["SEQUENCE"].tolist()

dfsr = {}

for condition in ["mic", "mic_no_curri", "tox_2", "tox_no_curri", "mul", "mul_no_curri"]:
    
    path = os.path.join(base_dir, condition, f"fb_epoch_19", "result.csv")
    df = pd.read_csv(path)
    dfsr[condition] = df["SEQUENCE"].tolist()


def calculate_length(data:list):
    lengths = [len(x) for x in data]
    return lengths

def calculate_molarweight(x:list):
    h = modlamp.descriptors.GlobalDescriptor(x)
    h.calculate_MW()
    return list(h.descriptor.flatten())

def calculate_charge(data:list):
    h = modlamp.analysis.GlobalAnalysis(data)
    h.calc_charge()
    return h.charge

def calculate_isoelectricpoint(data:list):
    h = modlamp.analysis.GlobalDescriptor(data)
    h.isoelectric_point()
    return list(h.descriptor.flatten())

def calculate_aromaticity(data:list):
    h = modlamp.analysis.GlobalDescriptor(data)
    h.aromaticity()
    return list(h.descriptor.flatten())

def calculate_hydrophobicity(data:list):
    h = modlamp.analysis.GlobalAnalysis(data)
    h.calc_H(scale='eisenberg')
    return list(h.H)

def calculate_hydrophobicmoment(data:list):
    h = modlamp.descriptors.PeptideDescriptor(data, 'eisenberg')
    h.calculate_moment()
    return list(h.descriptor.flatten())

def calculate_alphahelixpropensity(data:list):
    h = modlamp.descriptors.PeptideDescriptor(data, 'levitt_alpha')
    h.calculate_global()
    return list(h.descriptor.flatten())

def calculate_instability_index(data:list):
    h = modlamp.analysis.GlobalDescriptor(data)
    h.instability_index()
    return list(h.descriptor.flatten())

def calculate_hscore(data:list):
    return [phys.helical_search(x) for x in data]

def calculate_hydrophobic_ratio(data:list):
    h = modlamp.analysis.GlobalDescriptor(data)
    h.hydrophobic_ratio()
    return list(h.descriptor.flatten())

def calculate_boman_index(data:list):
    h = modlamp.analysis.GlobalDescriptor(data)
    h.boman_index()
    return list(h.descriptor.flatten())

def calculate_physchem(peptides, datasets):
    print(len(peptides))
    print(datasets)
    physchem = {}
    physchem['dataset'] = []
    physchem['length'] = []
    physchem['charge'] = []
    physchem['pi'] = []
    physchem['aromacity'] = []
    physchem['hydrophobicity'] = []
    physchem['hm'] = []
    physchem['alpha'] = []
    physchem['boman'] = []
    physchem['h_score'] = []
    physchem['hydrophobic_ratio'] = []
    physchem['instability'] = []

    for dataset, name in zip(peptides, datasets):
        physchem['dataset'] += (len(dataset) * [name])
        physchem['length'] += calculate_length(dataset)
        physchem['charge'] += calculate_charge(dataset)[0].tolist()           # paint
        physchem['pi'] += calculate_isoelectricpoint(dataset)                 # paint
        physchem['aromacity'] += calculate_aromaticity(dataset)               
        physchem['hydrophobicity'] += calculate_hydrophobicity(dataset)[0].tolist()
        physchem['hm'] += calculate_hydrophobicmoment(dataset)
        physchem['alpha'] += calculate_alphahelixpropensity(dataset)          # paint
        physchem['boman'] += calculate_boman_index(dataset)
        physchem['hydrophobic_ratio'] += calculate_hydrophobic_ratio(dataset) # paint
        physchem['instability'] += calculate_instability_index(dataset)       # paint

    return pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in physchem.items() ]))


palette_0 = {
    'Uniport': 'grey', 
    'train_dataset': '#ff7900',
    'mic_low & tox_low':'#f1faee',
    'mic_low & tox_high':'#a8dadc',
    'mic_high & tox_low':'#457b9d',
    'mic_high & tox_high':'#1d3557',
    'lora': '#F7CF8B',
}
plaette_1 = {
    "mic":'#e0aaff', 
    "mic_no_curri":'#9d4edd', 
    "tox":'#B7E4A8', 
    "tox_no_curri":'#7CCF7B', 
    "mul":'#F2A7C2', 
    "mul_no_curri":'#E073A2'
}

datasets = [dfs, dfsr]
result_df = [calculate_physchem(dfs.values(), dfs.keys()), calculate_physchem(dfsr.values(), dfsr.keys())]
palettes = [palette_0, plaette_1]

properties = {
    'pi': 'Isoelectric point',
    'charge': 'Charge',
    'hydrophobic_ratio': 'Hydrophobic ratio',
    'alpha': 'Alpha Helix\nPropensity',
    'instability': 'Instability Index'
}



boxprops = dict(linewidth=0.0, color='k') # type: ignore
flierprops = dict(linewidth=0.5)
medianprops = dict(linewidth=0.5, color='k')
whiskerprops = dict(linewidth=0.5)
capprops = dict(linewidth=0.5)
yticks_dict = {
    'pi': [2, 4, 6, 8, 10, 12], 
    'charge': [-4, -1, 2, 5, 8, 11, 14],
    'hydrophobic_ratio': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    'alpha': None,
    'instability': None
}



for prop, prop_label in properties.items():

    # === 每个性质单独创建一张 1×2 的图 ===
    fig, axes = plt.subplots(
        ncols=2,
        nrows=1,
        figsize=(5, 2.5),
        dpi=300,
        gridspec_kw={'width_ratios': [10.5, 9]},
        sharey=True
    )

    for dataset, physchem, ax, palette in zip(
        datasets, result_df, axes, palettes
    ):

        data = [
            physchem[physchem['dataset'] == x][prop].tolist()
            for x in dataset.keys()
        ]

        parts = ax.boxplot(
            data,
            showfliers=False,
            patch_artist=True,
            boxprops=boxprops,
            flierprops=flierprops,
            medianprops=medianprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            widths=0.4,
        )

        for patch, color in zip(parts['boxes'], palette.values()):
            patch.set_facecolor(color)

        ax.set_xticklabels([])

    # === y 轴设置（只在左图） ===
    axes[0].set_ylabel(prop_label)
    if yticks_dict[prop]:
        axes[0].set_yticks(yticks_dict[prop])

    # axes[1].set_yticklabels([])

    # === 保存单张图 ===
    out_path = f"./figure/PhysiochcemicalProperties/Physiochemical_{prop}.svg"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
