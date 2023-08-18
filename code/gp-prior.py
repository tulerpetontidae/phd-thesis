import numpy as np
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
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
plt.rcParams['font.size'] = 15

output_pdf_path='fig/gp-prior-samples.pdf'


def rbf_kernel(x, length_scale=0.1, std=1.0):
    # Radial Basis Function (RBF) kernel with a standard deviation parameter
    sqdist = np.subtract.outer(x, x) ** 2
    return std ** 2 * np.exp(-0.5 * sqdist / length_scale ** 2)

def softmax_transform(f_prior, tau):
    transformed_samples = np.zeros_like(f_prior)
    for i in range(f_prior.shape[1]):
        transformed_samples[:, i] = np.exp(f_prior[:, i] * tau) / np.sum(np.exp(f_prior * tau), axis=1)
    return transformed_samples

# Mean function (0 in this case)
x = np.linspace(0, 100, 10000)
num_samples = 3
kernel_func = rbf_kernel
mean = np.zeros(len(x))
np.random.seed(42)
# Covariance matrix using the kernel function
K = kernel_func(x)

# Drawing samples from the Gaussian process prior
L = cholesky(K + 1e-10 * np.eye(len(x)), lower=True)
f_prior = np.dot(L, np.random.normal(size=(len(x), num_samples)))

# Softmax transformations
low_tau_samples = softmax_transform(f_prior, tau=1)
high_tau_samples = softmax_transform(f_prior, tau=2)

samples_list = [f_prior, low_tau_samples, high_tau_samples]
titles = ['Gaussian Process prior samples', 'Softmax-transformed samples, $\\tau = 1$', 'Softmax-transformed samples $\\tau = 2$']
colors = ['dodgerblue', 'orangered', 'darkorange']

# Plotting
fig, axes = plt.subplots(3, 2, figsize=(12 , 6), gridspec_kw={'width_ratios': [5, 1]})
limit_n = int(len(x) / 15)

for i, samples in enumerate(samples_list):
    # GP plot
    axes[i, 0].set_title(titles[i])
    for j in range(num_samples):
        axes[i, 0].plot(x[:limit_n], samples[:limit_n, j], color=colors[j])
    axes[i, 0].spines['right'].set_visible(False)
    axes[i, 0].spines['top'].set_visible(False)

    # Histogram
    axes[i, 1].hist(samples, bins=30, color=colors, orientation='horizontal', stacked=True, density=True)
    axes[i, 1].spines['right'].set_visible(False)
    axes[i, 1].spines['top'].set_visible(False)
    axes[i, 1].spines['bottom'].set_visible(False)
    axes[i, 1].set_xticks([])

    if i != 0:
        axes[i, 0].set_ylim(0,1)

    if i == 0:
        axes[i, 1].set_title('Marginal distirubtion')

axes[-1, 0].set_xlabel('Spatial dimention, $x$')

plt.tight_layout()
plt.savefig(output_pdf_path, format='pdf')
plt.close()