import numpy as np
from scipy.stats import nbinom, poisson
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

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
plt.rcParams['font.size'] = 24

output_pdf_path='fig/side-plot-nb-dist.pdf'

# Parameters
mean = 10 # Mean of both Poisson and Negative Binomial distributions
overdispersion_values = [1/100, 1/5, 1] # Different overdispersion values for Negative Binomial

# X values for plotting
x = np.arange(0, 40)

# Plotting
plt.figure(figsize=(6, 10))

# Plotting Poisson PMF
plt.subplot(4, 1, 1)
poisson_pmf = poisson.pmf(x, mean)
plt.bar(x, poisson_pmf, color='dodgerblue')
plt.title('Poisson ($\\lambda$={})'.format(mean))
# plt.ylabel('Probability')
plt.grid(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().set_yticks([]) 
plt.gca().set_ylim(0,0.15)


colors = ['darkorange', 'orangered', 'mediumseagreen']
# Plotting Negative Binomial PMF for different overdispersion values
for i, (phi, color) in enumerate(zip(overdispersion_values, colors), 2):
    plt.subplot(4, 1, i)
    p = 1 / (1 + mean * phi)
    r = mean / (mean * phi)
    nb_pmf = nbinom.pmf(x, r, p)
    plt.bar(x, nb_pmf, color=color)
    plt.title('Negative Binomial ($\\mu$={}, $\\phi$={})'.format(mean, phi))
    # plt.ylabel('Probability')
    plt.grid(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().set_yticks([]) # Removing the y-axis
    plt.gca().set_ylim(0,0.15)


plt.xlabel('$x$')
plt.yticks([]) # Removing the y-axis

plt.tight_layout()
plt.savefig(output_pdf_path, format='pdf')
plt.close()
