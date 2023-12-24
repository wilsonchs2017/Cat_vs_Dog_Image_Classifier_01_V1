#import library
#--------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

#Loading Data
#------------
import zipfile39 as zipfile
with zipfile.ZipFile('E:/ML Projects/Cat vs Dog Image Classifier/dogs-vs-cats/train.zip','r') as z:
    z.extractall('.')

with zipfile.ZipFile('E:/ML Projects/Cat vs Dog Image Classifier/dogs-vs-cats/test1.zip','r') as z:
    z.extractall('.')

path = 'E:/ML Projects/Cat vs Dog Image Classifier/train'
filenames = os.listdir(path)
filenames[:5]

label = []
for filename in filenames:
    if filename.split('.')[0] == 'cat':
        label.append('cat')
    else:
        label.append('dog')

df = pd.DataFrame({'name':filenames,'label':label})

df.head()

print(df['label'].value_counts())
sns.countplot(data=df, x=df['label'])

load_img(path+'cat.62.jpg')

load_img(path+'dog.62.jpg')

#Base Model
#----------
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape = (256,256,3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.summary()

model.compile(optimizer='Adam',loss='binary_crossentropy',metrics='acc')

#Data Preprocessing, Data Splitting Phase
#----------------------------------------
train, test_val = train_test_split(df, test_size = 0.2, stratify = df['label'], random_state = 17)

test, val = train_test_split(test_val, test_size = 0.5, stratify = test_val['label'], random_state = 17)

print('train size:', train.shape[0],
      '\nvalidation size:', val.shape[0],
      '\ntest size:', test.shape[0])

print('train labels:\n', train['label'].value_counts(),
      '\n\nvalidation labels:\n', val['label'].value_counts(),
      '\n\ntest labels:\n', test['label'].value_counts(),
      sep = '')

#Data Preprocessing, Data Normalization
#----------------------------------------
train_gen = ImageDataGenerator(rescale=1./255)
train_data = train_gen.flow_from_dataframe(train, directory = path, x_col = 'name',
                                           y_col = 'label', class_mode ='binary', seed = 17)
val_gen = ImageDataGenerator(rescale=1./255)
val_data = val_gen.flow_from_dataframe(val, directory = path, x_col = 'name', y_col = 'label',
                                       class_mode = 'binary', seed = 17)

#Base Model Training
#----------------------------------------
history = model.fit(train_data, validation_data = val_data, epochs = 10)

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure(figsize = (15,10))
plt.plot(loss, label = 'Train Loss')
plt.plot(val_loss, '--', label = 'Val Loss')
plt.title('Training and Validation Loss')
plt.xticks(np.arange(0,10))
plt.yticks(np.arange(0,0.7,0.05))
plt.grid()
plt.legend()

#Data Preprocessing: Augmention
#----------------------------------------
aug_gen = ImageDataGenerator(rescale = 1./255,
                             shear_range = 0.2,
                             zoom_range = 0.2,
                             rotation_range = 40,
                             width_shift_range = 0.2,
                             height_shift_range = 0.2,
                             horizontal_flip = True,
                             fill_mode = 'nearest')

#creating img_ variables
img_cat = load_img(path+'cat.62.jpg')
img_dog = load_img(path+'dog.62.jpg')
#
img_cat_arr = image.img_to_array(img_cat)
img_cat_arr = img_cat_arr.reshape((1,)+img_cat_arr.shape)
#
img_dog_arr = image.img_to_array(img_dog)
img_dog_arr = img_dog_arr.reshape((1,)+ img_dog_arr.shape)


aug_images_cat = aug_gen.flow(img_cat_arr, batch_size = 1)
aug_images_dog = aug_gen.flow(img_dog_arr, batch_size = 1)

plt.figure(figsize = (15,10))
plt.subplot(141)
plt.imshow(img_cat)
plt.title("original")
i = 2
for batch in aug_images_cat:
    plt.subplot(14*10+i)
    plt.imshow(image.array_to_img(batch[0]))
    plt.title("Augmented")
    i += 1
    if i % 5 == 0:
        break

plt.figure(figsize = (15,10))
plt.subplot(141)
plt.imshow(img_dog)
plt.title("original")
i = 2
for batch in aug_images_dog:
    plt.subplot(14*10+i)
    plt.imshow(image.array_to_img(batch[0]))
    plt.title("Augmented")
    i += 1
    if i % 5 == 0:
        break

train_data = aug_gen.flow_from_dataframe(train,
                                         directory = path,
                                         x_col = 'name',
                                         y_col = 'label',
                                         class_mode = 'binary',
                                         target_size = (224,224),
                                         seed = 17)
val_data = val_gen.flow_from_dataframe(val,
                                       directory = path,
                                       x_col='name',
                                       y_col='label',
                                       class_mode='binary',
                                       target_size=(224,224),
                                       seed=17)

#Model Tuning
#----------------------------------------
model.summary()

best_model = models.Sequential()
best_model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape = (224,224,3)))
best_model.add(layers.MaxPooling2D((2,2)))
best_model.add(layers.Conv2D(64,(3,3),activation='relu'))
best_model.add(layers.MaxPooling2D((2,2)))
best_model.add(layers.Conv2D(64,(3,3),activation='relu'))
best_model.add(layers.MaxPooling2D((2,2)))
best_model.add(layers.Conv2D(128,(3,3),activation='relu'))
best_model.add(layers.MaxPooling2D((2,2)))
best_model.add(layers.Conv2D(128,(3,3),activation='relu'))
best_model.add(layers.MaxPooling2D((2,2)))
best_model.add(layers.Flatten())
best_model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
best_model.add(layers.Dropout(0.2))
best_model.add(layers.Dense(1,activation='sigmoid'))

best_model.summary()

best_model.compile(optimizer = optimizers.Adam(learning_rate=5e-4), loss='binary_crossentropy', metrics='acc')

#Tuned Model
#----------------------------------------
history2 = best_model.fit(train_data, validation_data = val_data, epochs=60,
                          callbacks=[EarlyStopping(monitor='val_acc', min_delta=0.001,
                                                   patience=5, verbose=1)])

loss = history2.history['acc']
val_loss = history2.history['val_acc']
plt.figure(figsize = (15,10))
plt.plot(loss, label = 'Train Acc')
plt.plot(val_loss, '--', label = 'Val Acc')
plt.title('Training and Validation Accuracy')
plt.xticks(np.arange(0,26))
plt.yticks(np.arange(0.5,1,0.05))
plt.grid()
plt.legend()

best_model.save('best_model_cat_vs_dog.h5')

test_data = val_gen.flow_from_dataframe(test,
                                        directory=path,
                                        x_col='name',
                                        y_col='label',
                                        class_mode='binary',
                                        target_size=(224,224),
                                        shuffle=False,
                                        seed=17
                                        )

test_pred = best_model.predict(test_data)

pred_label = test_pred > 0.5
true_label = test_data.classes

ConfusionMatrixDisplay(confusion_matrix(true_label,pred_label), display_labels=test_data.class_indices).plot();

best_model.evaluate(test_data)
