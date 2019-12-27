#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import json
from urllib.request import urlopen
import io
import cv2

from sklearn.feature_extraction.text import CountVectorizer

from PIL import Image, ImageFilter

from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D, Dropout, GlobalAveragePooling2D
from keras.models import Model

from keras.models import load_model

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


# In[ ]:


LOCATION_DATA = 'dataset/Face_Recognition.json'
LOCATION_AUGMENTED_IMAGE = 'images/augmented_images/'
LOCATION_TRAINED_MODEL = 'models/'

IMAGE_WIDTH = IMAGE_HEIGHT = 100
IMAGE_CHANNELS=1
TRAIN_TEST_SPLIT = 0.7


# In[ ]:


def get_label_separator():
    '''
    Returns the vectorizer that is fit on the corpus of labels: Not_FaceEmotion_Happy Emotion_Sad Emotion_Neutral Emotion_Angry Age_below20 Age_20_30 Age_30_40
    Age_40_50 Age_above_50 E_Asian E_White E_Black E_Hispanic E_Indian E_Arab G_Male G_ Female G_Other 
    '''
    corpus = ['''Not_Face Emotion_Happy Emotion_Sad Emotion_Neutral Emotion_Angry Age_below20 Age_20_30 Age_30_40 Age_40_50 Age_above_50 E_Asian E_White
    E_Black E_Hispanic E_Indian E_Arab G_Male G_ Female G_Other''']
    vectorizer = CountVectorizer()
    return vectorizer.fit(corpus)


# ## Segregates Faces from the Images Dataset

# In[ ]:


def get_dataset_dataframe():
    '''
    Returns a Pandas DataFrame object containing the dataset having 26 columns and Dictionary object that contains mapping to the output.
    '''
    X = get_label_separator()
    df = pd.DataFrame(columns=['img_path', "x1", "y1", "x2", "y2", "img_width", "img_height"])
    with open(LOCATION_DATA) as f:
        for line in f:
            j_content = json.loads(line)
            img_path = j_content['content']
            for sub_image in j_content['annotation']:
                row = {
                    'img_path':img_path,
                    "x1": sub_image['points'][0]['x'],
                    "y1": sub_image['points'][0]['y'],
                    "x2": sub_image['points'][1]['x'],
                    "y2": sub_image['points'][1]['y'],
                    "img_width": sub_image["imageWidth"],
                    "img_height": sub_image["imageHeight"]
                }
                labels = dict(zip(X.get_feature_names(), X.transform([str(sub_image['label'])]).toarray()[0]))
                row.update(labels)
                df = df.append(row, ignore_index=True)
    df.drop(columns=['g_'], inplace=True)
    df.rename(columns={'female':'g_female'}, inplace=True)
    pd.to_numeric(df.loc[:,"img_height"], errors='raise')
    datatype_conversion = dict( zip(set(X.get_feature_names()).intersection(df.columns.values), [int]*len(X.get_feature_names()) ))
    datatype_conversion.update({"img_width": float, "img_height": float, "g_female":int})
    df = df.astype(datatype_conversion) 
    def get_output_mapper():
        age_columns = df.columns.values[7:12]
        race_columns = df.columns.values[12:18]
        emotion_columns = df.columns.values[18:22]
        gender_columns = df.columns.values[22:25]
        return {
            'age': dict(zip(range(len(age_columns)), age_columns)),
            'race': dict(zip(range(len(race_columns)), race_columns)),
            'emotion': dict(zip(range(len(emotion_columns)), emotion_columns)),
            'gender': dict(zip(range(len(gender_columns)), gender_columns)),
        }
    return df, get_output_mapper()


# In[ ]:


df, output_mapper = get_dataset_dataframe()


# In[ ]:


import pickle
pickle.dump( output_mapper, open( "output_mapper.p", "wb" ) )


# ## Image Augmenter

# In[ ]:


def processed_data():
    '''
    Return processed (augmented) data to be fed to the CNN model.
    '''
    count = 0 
    df_new_images = pd.DataFrame(columns=df.columns.values[7:25])
    images, ages, races, emotions, genders = [], [], [], [], []
    for r in df.iterrows():
        row = r[1]
        image, gray=[], []
        fd = urlopen(row["img_path"])
        image_file = io.BytesIO(fd.read())
        image = Image.open(image_file)
        image = image.convert("RGB")
        image = np.asarray(image)
        if IMAGE_CHANNELS==1:
            image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2GRAY)
        roi_gray = image[int( (row["y1"]) * (row["img_height"]) ) : int( (row["y2"]) * (row["img_height"]) ),
                        int( (row["x1"]) * (row["img_width"])  ) : int( (row["x2"]) * (row["img_width"])  )]
        if roi_gray.size!=0:
            roi_gray = cv2.resize(roi_gray,(IMAGE_WIDTH, IMAGE_HEIGHT))
            df_new_images_dict ={}
            df_new_images_dict['img_path'] = str(count)+'n.png'
            df_new_images_dict.update(dict(zip(df.columns[7:25],row[7:25])))
            temp = Image.fromarray(roi_gray)
            temp.save(LOCATION_AUGMENTED_IMAGE+ str(count)+'n.png')
            df_new_images = df_new_images.append(df_new_images_dict, ignore_index=True)
            count+=1

            df_new_images_dict['img_path'] = str(count)+'fliplr.png'
            df_new_images = df_new_images.append(df_new_images_dict, ignore_index=True)
            temp = Image.fromarray(np.fliplr(roi_gray))
            temp.save(LOCATION_AUGMENTED_IMAGE+ str(count)+'fliplr.png')
            count+=1

            df_new_images_dict['img_path'] = str(count)+'flipud.png'
            df_new_images = df_new_images.append(df_new_images_dict, ignore_index=True)
            temp = Image.fromarray(np.flipud(roi_gray))
            temp.save(LOCATION_AUGMENTED_IMAGE+ str(count)+'flipud.png')
            count+=1

            df_new_images_dict['img_path'] = str(count)+'r30.png'
            df_new_images = df_new_images.append(df_new_images_dict, ignore_index=True)
            temp = Image.fromarray(roi_gray)
            temp = temp.rotate(30)
            temp.save(LOCATION_AUGMENTED_IMAGE+ str(count)+'r30.png')
            count+=1

            df_new_images_dict['img_path'] = str(count)+'r-30.png'
            df_new_images = df_new_images.append(df_new_images_dict, ignore_index=True)
            temp = Image.fromarray(roi_gray)
            temp = temp.rotate(-30)
            temp.save(LOCATION_AUGMENTED_IMAGE+ str(count)+'r-30.png')
            count+=1

            df_new_images_dict['img_path'] = str(count)+'sharpen.png'
            df_new_images = df_new_images.append(df_new_images_dict, ignore_index=True)
            temp = Image.fromarray(roi_gray)
            temp = temp.filter(ImageFilter.SHARPEN)
            temp.save(LOCATION_AUGMENTED_IMAGE+ str(count)+'sharpen.png')
            count+=1

            df_new_images_dict['img_path'] = str(count)+'edge_enhance.png'
            df_new_images = df_new_images.append(df_new_images_dict, ignore_index=True)
            temp = Image.fromarray(roi_gray)
            temp = temp.filter(ImageFilter.EDGE_ENHANCE)
            temp.save(LOCATION_AUGMENTED_IMAGE+ str(count)+'edge_enhance.png')
            count+=1

            df_new_images_dict['img_path'] = str(count)+'smooth.png'
            df_new_images = df_new_images.append(df_new_images_dict, ignore_index=True)
            temp = Image.fromarray(roi_gray)
            temp = temp.filter(ImageFilter.SMOOTH)
            temp.save(LOCATION_AUGMENTED_IMAGE+ str(count)+'smooth.png')
            count+=1

    return df_new_images


# In[ ]:


df_new_images = processed_data()


# ## Augmented Images Getter

# In[ ]:


def get_images(df_new_images, indices, for_training, batch_size):
    images, ages, races, emotions, genders = [], [], [], [], []
    while(True):    
        for r in df_new_images.iloc[indices].iterrows():
            row = r[1]
            image, gray=[], []
            image = Image.open(LOCATION_AUGMENTED_IMAGE+ row[18])
            image = image.convert("RGB")
            image = np.asarray(image)
            if IMAGE_CHANNELS==1:
                image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2GRAY)
            if image.size!=0:
                image = cv2.resize(image,(IMAGE_WIDTH, IMAGE_HEIGHT))
                image = image / 255.0
                images.append(image)
                ages.append(row[0:5].values)
                races.append(row[5:11])
                emotions.append(row[11:15])
                genders.append(row[15:18])
                if len(images)>=batch_size:
                    yield np.array(images).reshape(len(images), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), [np.array(ages), np.array(races), np.array(emotions), np.array(genders)]
            if not for_training:
                break


# ## Model Architecture without transfer learning

# In[ ]:


def model_without_transfer_learning():
    def CNN_conv(inp, filters=64, bn=True, pool=True):
        _ = Conv2D(filters=filters, kernel_size=3, activation='relu')(inp)
        if bn:
            _ = BatchNormalization()(_)
        if pool:
            _ = MaxPool2D()(_)
        return _

    input_layer = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    _ = CNN_conv(input_layer, filters=64, bn=False, pool=False)
    _ = CNN_conv(_, filters=64*2)
    _ = CNN_conv(_, filters=64*3)
    _ = CNN_conv(_, filters=64*4)
    CNN_shared_layer_end = GlobalMaxPool2D()(_)

    # for age prediction
    _ = Dense(units=320, activation='relu')(CNN_shared_layer_end)
    age_output = Dense(units=len(output_mapper['age']), activation='softmax', name='age_output')(_)

    # for race prediction
    _ = Dense(units=320, activation='relu')(CNN_shared_layer_end)
    _ = Dense(units=128, activation='relu')(_)
    race_output = Dense(units=len(output_mapper['race']), activation='softmax', name='race_output')(_)

    # for emotion prediction
    _ = Dense(units=320, activation='relu')(CNN_shared_layer_end)
    _ = Dense(units=128, activation='relu')(_)
    emotion_output = Dense(units=len(output_mapper['emotion']), activation='softmax', name='emotion_output')(_)

    # for gender prediction
    _ = Dense(units=320, activation='relu', )(CNN_shared_layer_end)
    gender_output = Dense(units=len(output_mapper['gender']), activation='softmax', name='gender_output')(_)

    model = Model(inputs=input_layer, outputs=[age_output, race_output, emotion_output, gender_output])
    model.compile(optimizer='Adam', 
                loss={'age_output': 'categorical_crossentropy', 'race_output': 'categorical_crossentropy', 'emotion_output': 'categorical_crossentropy', 'gender_output': 'categorical_crossentropy'},
                metrics={'age_output': 'accuracy', 'race_output': 'accuracy', 'emotion_output':'accuracy', 'gender_output': 'accuracy'})
    
#     model.summary()
    p = np.random.permutation(len(df_new_images))
    train_up_to = int(len(df_new_images) * TRAIN_TEST_SPLIT)
    train_idx = p[:train_up_to]
    valid_idx = p[train_up_to:]

    batch_size = 64
    valid_batch_size = 64

    train_gen = get_images(df_new_images, train_idx, for_training=True, batch_size=batch_size)
    valid_gen = get_images(df_new_images, valid_idx, for_training=True, batch_size=valid_batch_size)
    model.fit_generator(train_gen,
                    steps_per_epoch=len(train_idx)//batch_size,
                    epochs=20,
                    validation_data=valid_gen,
                    validation_steps=len(valid_idx)//valid_batch_size)
    
    return model


# ## Model Architecture with Transfer Learning InceptionV3

# In[ ]:


from keras.applications import InceptionV3

def model_with_inceptionv3():
    
    base_model = InceptionV3(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), weights='imagenet', include_top=False) 
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    CNN_shared_layer_end = Dropout(0.5)(x)

    # for age prediction
    _ = Dense(units=128, activation='relu')(CNN_shared_layer_end)
    age_output = Dense(units=len(output_mapper['age']), activation='softmax', name='age_output')(_)

    # # for race prediction
    _ = Dense(units=128, activation='relu')(CNN_shared_layer_end)
    race_output = Dense(units=len(output_mapper['race']), activation='softmax', name='race_output')(_)

    # for emotion prediction
    _ = Dense(units=128, activation='relu')(CNN_shared_layer_end)
    emotion_output = Dense(units=len(output_mapper['emotion']), activation='softmax', name='emotion_output')(_)

    # for gender prediction
    _ = Dense(units=320, activation='relu' )(CNN_shared_layer_end)
    gender_output = Dense(units=len(output_mapper['gender']), activation='softmax', name='gender_output')(_)

    for layer in base_model.layers:
        layer.trainable = False

    model = Model(inputs=base_model.input, outputs=[age_output, race_output, emotion_output, gender_output])
    model.compile(optimizer='rmsprop', 
                loss={'age_output': 'categorical_crossentropy', 
                        'race_output': 'categorical_crossentropy', 
                        'emotion_output': 'categorical_crossentropy', 
                        'gender_output': 'categorical_crossentropy'},
                loss_weights={'age_output': 1.5, 'emotion_output': 1.8, 'race_output': 2., 'gender_output': 1.},
                metrics={'age_output': 'accuracy', 'race_output': 'accuracy', 
                            'emotion_output':'accuracy', 'gender_output': 'accuracy'})
#     model.summary()
    
    p = np.random.permutation(len(df_new_images))
    train_up_to = int(len(df_new_images) * TRAIN_TEST_SPLIT)
    train_idx = p[:train_up_to]
    valid_idx = p[train_up_to:]

    batch_size = 64
    valid_batch_size = 64

    train_gen = get_images(df_new_images, train_idx, for_training=True, batch_size=batch_size)
    valid_gen = get_images(df_new_images, valid_idx, for_training=True, batch_size=valid_batch_size)

    model.fit_generator(train_gen,
                        steps_per_epoch=len(train_idx)//batch_size,
                        epochs=20,
                        validation_data=valid_gen,
                        validation_steps=len(valid_idx)//valid_batch_size)
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(train_gen,
                        steps_per_epoch=len(train_idx)//batch_size,
                        epochs=20,
                        validation_data=valid_gen,
                        validation_steps=len(valid_idx)//valid_batch_size)
    return model


# ## Model Architecture with Transfer Learning VGG19

# In[ ]:


from keras.applications import VGG19

def model_with_VGG19():
    base_model = VGG19(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), weights='imagenet', include_top=False)

    # # Top Model Block    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', name='fc1')(x)
    CNN_shared_layer_end = Dropout(0.5)(x)

    # for age prediction
    _ = Dense(units=16, activation='relu')(CNN_shared_layer_end)
    age_output = Dense(units=len(output_mapper['age']), activation='softmax', name='age_output')(_)

    # # for race prediction
    _ = Dense(units=16, activation='relu')(CNN_shared_layer_end)
    race_output = Dense(units=len(output_mapper['race']), activation='softmax', name='race_output')(_)

    # for emotion prediction
    _ = Dense(units=16, activation='relu')(CNN_shared_layer_end)
    emotion_output = Dense(units=len(output_mapper['emotion']), activation='softmax', name='emotion_output')(_)

    # for gender prediction
    _ = Dense(units=16, activation='relu', )(CNN_shared_layer_end)
    gender_output = Dense(units=len(output_mapper['gender']), activation='softmax', name='gender_output')(_)

    model = Model(inputs=base_model.input, outputs=[age_output, race_output, emotion_output, gender_output])
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='nadam', 
                loss={'age_output': 'categorical_crossentropy', 
                        'race_output': 'categorical_crossentropy', 
                        'emotion_output': 'categorical_crossentropy', 
                        'gender_output': 'categorical_crossentropy'},
                metrics={'age_output': 'accuracy', 'race_output': 'accuracy', 
                            'emotion_output':'accuracy', 'gender_output': 'accuracy'})
#     model.summary()
    
    p = np.random.permutation(len(df_new_images))
    train_up_to = int(len(df_new_images) * TRAIN_TEST_SPLIT)
    train_idx = p[:train_up_to]
    valid_idx = p[train_up_to:]

    batch_size = 64
    valid_batch_size = 64

    train_gen = get_images(df_new_images, train_idx, for_training=True, batch_size=batch_size)
    valid_gen = get_images(df_new_images, valid_idx, for_training=True, batch_size=valid_batch_size)

    model.fit_generator(train_gen,
                        steps_per_epoch=len(train_idx)//batch_size,
                        epochs=20,
                        validation_data=valid_gen,
                        validation_steps=len(valid_idx)//valid_batch_size)
    return model


# In[ ]:


model_wtl = model_without_transfer_learning()
model_wtl.save(LOCATION_TRAINED_MODEL + 'model_without_transfer_learning.h5')
model_i = model_with_inceptionv3()
model_i.save(LOCATION_TRAINED_MODEL + 'model_with_transfer_learning_InceptionV3.h5')
model_v = model_with_VGG19()
model_v.save(LOCATION_TRAINED_MODEL + 'model_with_transfer_learning_VGG19.h5') 