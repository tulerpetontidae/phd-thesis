import numpy as np
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
plt.rcParams['font.size'] = 25

output_pdf_path='fig/side-plot-elbo-bad.pdf'

with open('./code/data/unknown_loss.npy', 'rb') as f:
    elbo_good = np.load(f)

with open('./code/data/diverging_loss.npy', 'rb') as f:
    elbo_bad = np.load(f)


plt.figure(figsize=(6, 6))
plt.plot(np.arange(15000)[150::5], elbo_bad[150::5], color='orangered', lw=3, label='Diverging Loss')
plt.plot(np.arange(15000)[150::5], elbo_good[150::5], color='dodgerblue', lw=3, label='Converging Loss')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.legend(loc='upper right', frameon=False)
plt.ylabel('-ELBO')
plt.xlabel('Training iteration')
plt.tight_layout()
# plt.show()
plt.savefig(output_pdf_path, bbox_inches='tight')
plt.close()