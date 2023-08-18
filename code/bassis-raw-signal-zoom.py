import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mut_alleles = ['CKAP5mut','DENND1Amut', 'KIAA0652mut']
wt_alleles = ['CKAP5wt', 'DENND1Awt', 'KIAA0652wt']

colors = ['skyblue', 'orange', 'green']

tmp = pd.read_csv('./code/data/MutR1_PD9694d_Top_GMMdecoding.csv')


spatial_dims = [tmp.X.max(), tmp.Y.max()]

fig, axs = plt.subplots(1,3,figsize=(spatial_dims[0]/1000*2.7/4,
                                spatial_dims[1]/1000*2/4), facecolor='black')

print(spatial_dims)

df_mut = tmp[tmp.Probability>0.6]

#cut out df_mut between 40000 and 50000 on the x axis and 10000 and 15000 on the y axis
df_mut = df_mut[(df_mut.X > 43000) & (df_mut.X < 51000) & (df_mut.Y > 7000) & (df_mut.Y < 13000)]
spatial_dims = [df_mut.X.max(), df_mut.Y.max()]

pos_x = []
pos_y = []
p_color = []
for i, allele in enumerate(mut_alleles):
    pos_x.append(df_mut.X[df_mut.Name == allele])
    pos_y.append(df_mut.Y[df_mut.Name == allele])
    p_color.append([colors[i]]*(df_mut.Name == allele).sum())
pos_x = np.concatenate(pos_x)
pos_y = np.concatenate(pos_y)
p_color = np.concatenate(p_color)
p_order = np.random.permutation(p_color.shape[0])
axs[0].scatter(pos_x[p_order], pos_y[p_order], color=p_color[p_order], s=10/2)   

axs[0].plot([spatial_dims[0]*0.9,
          spatial_dims[0]*0.9 - 2.5e3 / 0.325 ],
         [spatial_dims[1]*(-0.1),
          spatial_dims[1]*(-0.1)], color='white', lw=10/2)

axs[0].set_aspect('equal')
axs[0].axis('off')
axs[0].set_facecolor('black')

pos_x = []
pos_y = []
p_color = []
for i, allele in enumerate(wt_alleles):
    pos_x.append(df_mut.X[df_mut.Name == allele])
    pos_y.append(df_mut.Y[df_mut.Name == allele])
    p_color.append([colors[i]]*(df_mut.Name == allele).sum())
pos_x = np.concatenate(pos_x)
pos_y = np.concatenate(pos_y)
p_color = np.concatenate(p_color)
p_order = np.random.permutation(p_color.shape[0])
axs[1].scatter(pos_x[p_order], pos_y[p_order], color=p_color[p_order], s=10/2)   

axs[1].plot([spatial_dims[0]*0.9,
          spatial_dims[0]*0.9 - 2.5e3 / 0.325 ],
         [spatial_dims[1]*(-0.1),
          spatial_dims[1]*(-0.1)], color='white', lw=10/2)

axs[1].set_aspect('equal')
axs[1].axis('off')
axs[1].set_facecolor('black')


axs[2].scatter(df_mut.X[df_mut.Name == 'FGFR1exp'], df_mut.Y[df_mut.Name == 'FGFR1exp'], color='orange', s=0.1)
axs[2].plot([spatial_dims[0]*0.9,
          spatial_dims[0]*0.9 - 2.5e3 / 0.325 ],
         [spatial_dims[1]*(-0.1),
          spatial_dims[1]*(-0.1)], color='white', lw=10/2)
axs[2].set_aspect('equal')
axs[2].axis('off')
axs[2].set_facecolor('black')
plt.savefig(f'./code/raw_fig/zoomed-basiss-signal.pdf')
plt.close()

