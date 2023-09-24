import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2

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

output_pdf_path='code/raw_fig/side-plot-chi2-trick.pdf'


# Parameters for the normal distributions
mu1, sigma1 = 0, 1
mu2, sigma2 = 5, 4

# Generate x-values for PDF plotting
x_values = np.linspace(-20, 20, 1000)
transformed_x_values = x_values / sigma1  # Transform x-values for the first distribution
transformed_x_values_2 = x_values / sigma2  # Transform x-values for the second distribution

# Compute PDF using SciPy for original distributions
pdf_1 = norm.pdf(x_values, mu1, sigma1)
pdf_2 = norm.pdf(x_values, mu2, sigma2)

# Compute PDF for transformed distributions
pdf_1_transformed = norm.pdf(transformed_x_values, mu1/sigma1, 1)
pdf_2_transformed = norm.pdf(transformed_x_values_2, mu2/sigma2, 1)

# Generate 100 samples for each distribution
samples_1 = np.random.normal(mu1, sigma1, 100)
samples_2 = np.random.normal(mu2, sigma2,1050)

# Create plots
plt.figure(figsize=(6, 10))

# First Panel: Original Distributions and Samples
plt.subplot(3, 1, 1)
# plt.plot(x_values, pdf_1, label=f'Mean={mu1}, Sigma={sigma1}')
# plt.plot(x_values, pdf_2, label=f'Mean={mu2}, Sigma={sigma2}')
# plt.hist(samples_1, bins=40, density=True, alpha=0.5, color='blue', range=(-30,30))
plt.hist(samples_2, bins=30, density=True, alpha=0.5, color='darkorange', range=(-20,20))

# Second Panel: Transformed Distributions and Samples
plt.subplot(3, 1, 2)
plt.plot(transformed_x_values, pdf_1_transformed, color='dodgerblue')
# plt.plot(transformed_x_values_2, pdf_2_transformed, label=f'Transformed Mean={mu2/sigma2}, Sigma=1')
# plt.hist(samples_1 / sigma1, bins=40, density=True, alpha=0.5, color='blue', range=(-6,6))
plt.hist(samples_2 / sigma2, bins=30, density=True, alpha=0.5, color='darkorange', range=(-6,6))
plt.xlim(-6,6)

# plt.axvline(mu1/sigma1)
plt.axvline(mu2/sigma2,  c='k')


x_values = np.linspace(0, 5, 1000)
chi2_pdf = chi2.pdf(x_values, 1)

# Third Panel: Chi-Squared Distribution
plt.subplot(3, 1, 3)
plt.plot(x_values, chi2_pdf, color='dodgerblue')
plt.axvline((mu2/sigma2)**2, c='k')

# Shading area under the chi-squared curve after specific x-value
threshold_x = (mu2/sigma2)**2
mask = x_values > threshold_x  # Create a boolean mask for x-values greater than threshold
plt.fill_between(x_values[mask], chi2_pdf[mask], color='orangered', alpha=0.5)

plt.xlim(0, 3)
plt.ylim(0, 3)

plt.tight_layout()
plt.savefig(output_pdf_path, format='pdf')
plt.close()
