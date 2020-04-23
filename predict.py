import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # showing and rendering figures
# io related
from skimage.io import imread
import os
from glob import glob
from sklearn.model_selection import train_test_split
import seaborn as sns
from keras.metrics import mean_absolute_error
from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input


age_df = pd.read_csv('train.csv')
age_df['path'] = age_df['id'].map(lambda x: os.path.join('train','{}.png'.format(x)))
age_df['exists'] = age_df['path'].map(os.path.exists)
print(age_df['exists'].sum(), 'images found of', age_df.shape[0], 'total')
age_df['gender'] = age_df['male'].map(lambda x: 'male' if x else 'female')
boneage_mean = age_df['boneage'].mean()
boneage_div = 2*age_df['boneage'].std()
# we don't want normalization for now
boneage_mean = 0
boneage_div = 1.0
age_df['boneage_zscore'] = age_df['boneage'].map(lambda x: (x-boneage_mean)/boneage_div)
print("missing values:\n",(age_df==np.nan).sum())
age_df.dropna(inplace = True)
print(age_df.sample(3))
print(age_df['path'])

age_df['boneage_category'] = pd.cut(age_df['boneage'], 10)
#age_df['gender'].value_counts().plot(kind="bar")
sns.boxplot('gender', 'boneage',data = age_df)
plt.show()


train_df, valid_df = train_test_split(age_df,test_size = 0.25)
print('train', train_df.shape[0], 'validation', valid_df.shape[0])


IMG_SIZE = (224, 224) # slightly smaller than restnet50 normally expects
core_idg = ImageDataGenerator(samplewise_center=False, 
                              samplewise_std_normalization=False, 
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range = 0.15, 
                              width_shift_range = 0.15, 
                              rotation_range = 5, 
                              shear_range = 0.01,
                              fill_mode = 'nearest',
                              zoom_range=0.15)
                            # preprocessing_function = preprocess_input)
print(train_df['path'].values[0])
os.path.dirname(train_df['path'].values[0])


def from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print("basedir")
    print(base_dir)
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                    **dflow_args)
    df_gen.filenames = in_df[path_col].values
    print(df_gen.filenames)
    df_gen.classes = np.stack(in_df[y_col].values)
    print(df_gen.classes)
    df_gen.samples = in_df.shape[0]
    print(df_gen.samples)
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen

train_gen = from_dataframe(core_idg, train_df, 
                             path_col = 'path',
                            y_col = 'boneage_zscore', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 64)

valid_gen = from_dataframe(core_idg, valid_df, 
                             path_col = 'path',
                            y_col = 'boneage_zscore', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 64) # we can use much larger batches for evaluation

print(train_gen)
print(valid_gen)

test_X, test_Y = next(from_dataframe(core_idg, 
                               valid_df, 
                             path_col = 'path',
                            y_col = 'boneage_zscore', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 1024)) # one big batch
print(test_X)
print(test_Y)

t_x, t_y = next(train_gen)
fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone', vmin = -127, vmax = 127)
    c_ax.set_title('%2.0f months' % (c_y*boneage_div+boneage_mean))
    c_ax.axis('off')
plt.show()


def mae_months(in_gt, in_pred):
    return mean_absolute_error(boneage_div*in_gt, boneage_div*in_pred)


resnet_weights_path = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

bone_age = Sequential()
bone_age.add(ResNet50(input_shape =  t_x.shape[1:], include_top=False, pooling='max', weights= resnet_weights_path)) 
bone_age.add(Dense(1, activation = 'linear' ))
# bone_age.layers[0].trainable = False

bone_age.compile(optimizer = 'adam', loss = 'mse', metrics = [mae_months])
bone_age.summary()


weight_path="{}_weights.best.hdf5".format('bone_age')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,save_best_only=True, mode='min', save_weights_only = True)


reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss",mode="min",patience=5) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]

train_gen.batch_size = 16
bone_age.fit_generator(train_gen,epochs = 1,validation_data = (test_X, test_Y),steps_per_epoch= 3,validation_steps=1,callbacks = callbacks_list)
                       
                       

bone_age.load_weights(weight_path)

pred_Y = boneage_div*bone_age.predict(test_X, batch_size = 32, verbose = True)+boneage_mean
test_Y_months = boneage_div*test_Y+boneage_mean

ord_idx = np.argsort(test_Y)
ord_idx = ord_idx[np.linspace(0, len(ord_idx) - 1, 8).astype(int)]  # take 8 evenly spaced ones
fig, m_axs = plt.subplots(4, 2, figsize=(16, 32))
for (idx, c_ax) in zip(ord_idx, m_axs.flatten()):
    c_ax.imshow(test_X[idx, :, :, 0], cmap='bone')

    c_ax.set_title('Age: %2.1fY\nPredicted Age: %2.1fY' % (test_Y_months[idx] / 12.0,
                                                           pred_Y[idx] / 12.0))
    c_ax.axis('off')
fig.savefig('trained_img_predictions.png', dpi=300)