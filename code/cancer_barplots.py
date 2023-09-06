import pandas as pd
import numpy as np
from scipy.stats import gamma, poisson
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import matplotlib.font_manager as fm
from matplotlib import rc



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
plt.rcParams['font.size'] = 25

output_pdf_path='fig/side-plot-cancer-cases-2020.pdf'

# Load the data from the CSV file
df = pd.read_csv('code/data/cancer_cases2020.csv', index_col=None)

cancer_types_abbr = {'Breast': 'Breast',
                    'Lung':'Lung',
                    'Colorectum': 'Colo.',
                    'Prostate': 'Pro.',
                    'Stomach': 'Stom.',
                    'Liver': 'Liver',
                    'Cervix uteri': 'Cerv.', 
                    'Thyroid': 'Thyr.', 
                    'Oesophagus': 'Oeso.', 
                    'Non-Hodgkin lymphoma': 'NHL'}



# Sort the DataFrame by 'Incidence' in descending order for better visualisation
df.sort_values(by='Incidence', ascending=True, inplace=True)

# Extract the types of cancer and their incidence rates
cancer_types = df['Cancer'].values
incidence_rates = df['Incidence'].values / 1e6
colors = ['gainsboro' if cancer != 'Breast' else 'orangered' for cancer in cancer_types]

# Plotting
plt.figure(figsize=(6, 8))

# Create horizontal bar chart with gradient fill
plt.barh(cancer_types, incidence_rates, color=colors)
# Set y-ticks and y-tick labels
plt.yticks(range(len(cancer_types)), labels=[cancer_types_abbr[x] for x in cancer_types])

# Label axes
plt.xlabel('Cases, millions')
# plt.ylabel('Cancer Type')

# Remove spines for aesthetics
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

# Save figure to a PDF
plt.tight_layout()
plt.savefig(output_pdf_path, format='pdf')
plt.close()
