import pandas as pd
import numpy as np
from scipy.stats import gamma, poisson
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib import rc
from matplotlib.lines import Line2D

# Path to the Neo Euler font file
font_path_math = '/Users/b450-admin/Library/Fonts/NeoEuler-VGO00.otf'
font_path_text = '/Users/b450-admin/Library/Fonts/SourceSansPro-Regular.ttf'

# Add the font to the font manager
fm.fontManager.addfont(font_path_math)
fm.fontManager.addfont(font_path_text)


# Load the font
euler_font_properties = fm.FontProperties(fname=font_path_math)
sans_font_properties = fm.FontProperties(fname=font_path_text)

# Update the custom font settings
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = euler_font_properties.get_name()
plt.rcParams['mathtext.rm'] = euler_font_properties.get_name()
plt.rcParams['font.family'] = sans_font_properties.get_name()
plt.rcParams['font.size'] = 20

output_pdf_path='fig/side-plot-gpu-trans.pdf'


chip_file_path = './code/data/chip_dataset.csv'
chip_data = pd.read_csv(chip_file_path)

# Filtering the data to include only GPUs
gpu_chip_data = chip_data[chip_data['Type'] == 'GPU']

# Extracting the year from the Release Date column
gpu_chip_data['Year'] = pd.to_datetime(gpu_chip_data['Release Date']).dt.year

# Filtering out rows with NaN values in the 'FP32 GFLOPS' column
gpu_chip_data_filtered = gpu_chip_data.dropna(subset=['FP32 GFLOPS'])
gpu_chip_data_filtered['Release Date'] = pd.to_datetime(gpu_chip_data_filtered['Release Date'])

# Grouping by the Year and selecting the row with the maximum FP32 GFLOPS for each year
top_gpus_chip_by_year_fp32 = gpu_chip_data_filtered.loc[gpu_chip_data_filtered.groupby('Year')['FP32 GFLOPS'].idxmax()]

# Creating two subplots: one for transistors and one for GFLOPS
fig, ax1 = plt.subplots(1, 1, figsize=[6, 6], sharex=True)

# Top Subplot: Plotting the transistors (in millions) by release date for all GPUs
ax1.scatter(gpu_chip_data_filtered['Release Date'], gpu_chip_data_filtered['Transistors (million)'], color='dodgerblue', alpha=0.5, s=20)
ax1.set_ylabel('Transistors (million)')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# top_gpus_chip_by_year_fp32['Release Date'] = pd.to_datetime(top_gpus_chip_by_year_fp32['Year'].astype(int).astype(str) + '-07-01')
# # Bottom Subplot: Plotting the top graphic card's FP32 GFLOPS by year as bars
# ax2.bar(top_gpus_chip_by_year_fp32['Release Date'], top_gpus_chip_by_year_fp32['FP32 GFLOPS'], color='darkorange', width=365*0.8)
# ax2.set_ylabel('TOP FP32 GFLOPS')
# ax2.tick_params(axis='y')
# ax2.set_xlabel('Release Date')
# ax2.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)


# print(top_gpus_chip_by_year_fp32.head())
# Adjusting x-axis ticks for better visibility
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(output_pdf_path, format='pdf')
plt.close()