import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # showing and rendering figures
# io related
from skimage.io import imread
import os
from glob import glob
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import keras.backend as K

bone_age_model = load_model('Bone_Age_Model.h5',custom_objects=None,compile=False)

age_df = pd.read_csv('test.csv')
age_df['path'] = age_df['id'].map(lambda x: os.path.join('test','{}.png'.format(x)))
age_df['exists'] = age_df['path'].map(os.path.exists)
print(age_df['exists'].sum(), 'images found of', age_df.shape[0], 'total')
age_df['gender'] = age_df['male'].map(lambda x: 'male' if x else 'female')


boneage_mean = 0
boneage_div = 1.0

age_df['boneage_zscore'] = age_df['boneage'].map(lambda x: (x-boneage_mean)/boneage_div)
age_df.dropna(inplace = True)
age_df.sample(3)

#----------------------------------------------------------------------------------------------------


age_df['boneage_category'] = pd.cut(age_df['boneage'], 10)

raw_train_df, valid_df = train_test_split(age_df, 
                                   test_size = 0.9, 
                                   random_state = 2018,
                                   stratify = age_df['boneage_category'])


print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0])


#----------------------------------------------------------------------------------------------------

IMG_SIZE = (384, 384) 

core_idg = ImageDataGenerator(samplewise_center=False, 
                              samplewise_std_normalization=False, 
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range = 0.15, 
                              width_shift_range = 0.15, 
                              rotation_range = 5, 
                              shear_range = 0.01,
                              fill_mode = 'nearest',
                              zoom_range=0.25,
                              preprocessing_function = preprocess_input)


#----------------------------------------------------------------------------------------------------



def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                    **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen


#----------------------------------------------------------------------------------------------------



test_X, test_Y = next(flow_from_dataframe(core_idg, 
                               valid_df, 
                             path_col = 'path',
                            y_col = 'boneage_zscore', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 10)) # one big batch

for attn_layer in bone_age_model.layers:
    c_shape = attn_layer.get_output_shape_at(0)
    if len(c_shape)==4:
        if c_shape[-1]==1:
            print(attn_layer)
            break



pred_Y = boneage_div*bone_age_model.predict(test_X, batch_size = 10, verbose = True)+boneage_mean
test_Y_months = boneage_div*test_Y+boneage_mean



ord_idx = np.argsort(test_Y)
ord_idx = ord_idx[np.linspace(0, len(ord_idx)-1, 8).astype(int)] # take 8 evenly spaced ones
fig, m_axs = plt.subplots(4, 2, figsize = (16, 32))
for (idx, c_ax) in zip(ord_idx, m_axs.flatten()):
    c_ax.imshow(test_X[idx, :,:,0], cmap = 'bone')
    accu=(100*pred_Y[idx])/test_Y_months[idx]
    if accu > 100:
        accu=accu-100
        accu=100-accu
    c_ax.set_title('Age: %2.1fY\nPredicted Age: %2.1fY\nAccuracy:%2.1f' % (test_Y_months[idx]/12.0, 
                                                           pred_Y[idx]/12.0,accu))
    c_ax.axis('off')
fig.savefig('trained_img_predictions.png', dpi = 300)

#----------------------------------------------------------------------------------------------------
