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
plt.rcParams['font.size'] = 15

output_pdf_path='fig/simple-gamma-poisson.pdf'


# Custom colormap for gradient shading
def gradient_fill(x, y, fill_color=None, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    zorder = kwargs.pop('zorder', 0)
    alpha = kwargs.pop('alpha', 1)
    linewidth = kwargs.pop('linewidth', 1)

    line, = ax.plot(x, y, linewidth=linewidth, **kwargs)

    if fill_color is None:
        fill_color = line.get_color()

    z = np.empty((100, 1, 4), dtype=float)
    rgb = to_rgba(fill_color)[:3] # Here's the change
    z[:,:,:3] = rgb
    z[:,:,-1] = np.linspace(0, alpha, 100)[:,None]

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='lower', zorder=zorder)

    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    clip_path = plt.Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    return line, im

# Setting hyperparameters for the Gamma prior
alpha_prior = 3 / 3
beta_prior = 1 / 3

# Number of cells (observations)
n = 5

# Sampling lambda from the Gamma prior
np.random.seed(42)
lambda_true = gamma.rvs(alpha_prior, scale=1/beta_prior)

# Generating observations using the Poisson distribution with the selected lambda
observations = poisson.rvs(lambda_true, size=n)

print(lambda_true)

# Parameters for the posterior Gamma distribution
alpha_posterior = alpha_prior + np.sum(observations)
beta_posterior = beta_prior + n

# Creating a range of lambda values for plotting
lambda_range = np.linspace(0, 10, 1000)

# Prior Gamma distribution
prior_pdf = gamma.pdf(lambda_range, alpha_prior, scale=1/beta_prior)

# Likelihood Poisson distribution
likelihood_pdf = poisson.pmf(observations[0], lambda_range)

# Posterior Gamma distribution
posterior_pdf = gamma.pdf(lambda_range, alpha_posterior, scale=1/beta_posterior)

# Plotting
plt.figure(figsize=(12/1.5, 6/1.5))

gradient_fill(lambda_range, prior_pdf, fill_color='dodgerblue')
plt.plot(lambda_range, prior_pdf, c='dodgerblue')

gradient_fill(lambda_range, likelihood_pdf, fill_color='darkorange')
plt.plot(lambda_range, likelihood_pdf, c='darkorange')

gradient_fill(lambda_range, posterior_pdf, fill_color='orangered')
plt.plot(lambda_range, posterior_pdf, c='orangered')

plt.axvline(lambda_true, c='k', ls='--')
plt.xlabel('$\\lambda$')
plt.yticks([]) # Removing the y-axis
plt.grid(False) # Turning off the grid

# Creating custom legend handles
prior_patch = mpatches.Patch(color='dodgerblue', label='$p(\\lambda)$')
likelihood_patch = mpatches.Patch(color='darkorange', label='$p(D_0={}|\\lambda)$'.format(observations[0]))
posterior_patch = mpatches.Patch(color='orangered', label='$p(\\lambda|D_n)$')
true_alpha_line = Line2D([0], [0], color='k', ls='--', label='True $\\lambda$')

plt.legend(handles=[prior_patch, likelihood_patch, posterior_patch, true_alpha_line], frameon=False)

# Removing the top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.tight_layout()
plt.ylim(0,0.9)
plt.savefig(output_pdf_path, format='pdf')
plt.close()
