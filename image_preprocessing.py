import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # showing and rendering figures
# io related
from skimage.io import imread
import os
from glob import glob
# not needed in Kaggle, but required in Jupyter

from scipy.ndimage import gaussian_filter
from skimage import data
from skimage import img_as_float
from skimage import img_as_ubyte
from skimage.morphology import reconstruction


age_df = pd.read_csv('train.csv')
age_df['path'] = age_df['id'].map(lambda x: os.path.join('train','{}.png'.format(x)))
age_df['exists'] = age_df['path'].map(os.path.exists)
print(age_df['exists'].sum(), 'images found of', age_df.shape[0], 'total')
age_df['gender'] = age_df['male'].map(lambda x: 'male' if x else 'female')
age_df.dropna(inplace = True)
age_df.sample(3)

age_groups = 8
age_df['age_class'] = pd.qcut(age_df['boneage'], age_groups)
age_overview_df = age_df.groupby(['age_class', 
                                  'gender']).apply(lambda x: x.sample(1)
                                                             ).reset_index(drop = True
                                                                          )
fig, m_axs = plt.subplots( age_groups, 2, figsize = (12, 6*age_groups))
for c_ax, (_, c_row) in zip(m_axs.flatten(), 
                            age_overview_df.sort_values(['age_class', 'gender']).iterrows()):
    c_ax.imshow(imread(c_row['path']),
                cmap = 'viridis')
    c_ax.axis('off')
    c_ax.set_title('{boneage} months, {gender}'.format(**c_row))




# Convert to float: Important for subtraction later which won't work with uint8
for c_ax, (_, c_row) in zip(m_axs.flatten(), 
                            age_overview_df.sort_values(['age_class', 'gender']).iterrows()):
    image = img_as_ubyte(imread(c_row['path'])/255)

image= image.round()
image = gaussian_filter(image, 1)

seed = np.copy(image)
seed[1:-1, 1:-1] = image.min()
mask = image

dilated = reconstruction(seed, mask, method='dilation')


fig, (ax0, ax1, ax2) = plt.subplots(nrows=1,
                                    ncols=3,
                                    figsize=(18, 12.5),
                                    sharex=True,
                                    sharey=True)

ax0.imshow(image, cmap='gray')
ax0.set_title('original image')
ax0.axis('off')

ax1.imshow(dilated, vmin=image.min(), vmax=image.max(), cmap='gray')
ax1.set_title('dilated')
ax1.axis('off')

ax2.imshow(image - dilated, cmap='gray')
ax2.set_title('image - dilated')
ax2.axis('off')

plt.show()

fig.tight_layout()