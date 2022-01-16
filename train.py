from extract import get_data_training
from utils import visualize, group_images, masks_Unet
from model import unet
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
import datetime,dateutil
import os
import matplotlib.pyplot as plt

now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y%m%d_%H%M%S')
name_experiment = f'experiment/{timestamp}/'
epochs = 10
batch_size = 32

if not os.path.exists(name_experiment):
    os.makedirs(name_experiment)

#============ Load the data and divided in patches
patches_imgs_train, patches_masks_train = get_data_training(
    patch_height = 48,
    patch_width = 48,
    N_subimgs = 190000,
    inside_FOV = False #select the patches only inside the FOV  (default == True)
)

#========= Save a sample of what you're feeding to the neural network ==========
N_sample = min(patches_imgs_train.shape[0],40)
visualize(group_images(patches_imgs_train[0:N_sample,:,:,:],5),f'{name_experiment}/sample_input_imgs')#.show()
visualize(group_images(patches_masks_train[0:N_sample,:,:,:],5),f'{name_experiment}/sample_input_masks')#.show()


#=========== Construct and save the model arcitecture =====
n_ch = patches_imgs_train.shape[1]
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[3]
print(patches_imgs_train.shape) #(190000, 1, 48, 48)

model = unet(n_ch, patch_height, patch_width)  #the U-net model

plot_model(model, to_file=f'{name_experiment}/model.png')   #check how the model looks like
json_string = model.to_json()
with open(f'{name_experiment}/architecture.json', 'w') as f:
    f.write(json_string)

#============  Training ==================================
checkpointer = ModelCheckpoint(filepath=f'{name_experiment}/best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased

patches_masks_train = masks_Unet(patches_masks_train)  #reduce memory consumption
history = model.fit(patches_imgs_train, patches_masks_train, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True, validation_split=0.1, callbacks=[checkpointer])

#========== Save and test the last model ===================
model.save_weights(f'{name_experiment}/last_weights.h5', overwrite=True)

#绘制loss
print(history.history)
epochs=range(len(history.history['accuracy']))
plt.figure()
plt.plot(epochs,history.history['accuracy'],'b',label='Training acc')
plt.plot(epochs,history.history['val_accuracy'],'r',label='Validation acc')
plt.title('Traing and Validation accuracy')
plt.legend()
plt.savefig(f'{name_experiment}/acc.jpg')

plt.figure()
plt.plot(epochs,history.history['loss'],'b',label='Training loss')
plt.plot(epochs,history.history['val_loss'],'r',label='Validation val_loss')
plt.title('Traing and Validation loss')
plt.legend()
plt.savefig(f'{name_experiment}/loss.jpg')