import numpy as np
from matplotlib import pyplot as plt
#Keras
from tensorflow.keras.models import model_from_json
#scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from utils import *
from extract import recompone,recompone_overlap,kill_border, pred_only_FOV, get_data_testing, get_data_testing_overlap

path_data = './datasets'

name_experiment = f'experiment/20220108_083956/' # drop=0.2

#original test images (for FOV selection)
DRIVE_test_imgs_original = './hdf5/DRIVE_dataset_originalImgs_test.hdf5'
DRIVE_test_border_masks = './hdf5/DRIVE_dataset_borderMasks_test.hdf5'
DRIVE_test_ground_truth = './hdf5/DRIVE_dataset_groundTruth_test.hdf5'

test_imgs_orig = load_hdf5(DRIVE_test_imgs_original)
full_img_height = test_imgs_orig.shape[2]
full_img_width = test_imgs_orig.shape[3]
#the border masks provided by the DRIVE
test_border_masks = load_hdf5(DRIVE_test_border_masks)
# dimension of the patches
patch_height = 48
patch_width = 48
#the stride in case output with average
stride_height = 5
stride_width = 5
assert (stride_height < patch_height and stride_width < patch_width)

#Grouping of the predicted images
N_visual = 1 #N_group_visual'
#====== average mode ===========
average_mode = True

#============ Load the data and divide in patches
patches_imgs_test = None
new_height = None
new_width = None
masks_test  = None
patches_masks_test = None
if average_mode == True:
    patches_imgs_test, new_height, new_width, masks_test = get_data_testing_overlap(
        DRIVE_test_imgs_original = DRIVE_test_imgs_original,  #original
        DRIVE_test_groudTruth = DRIVE_test_ground_truth,
        Imgs_to_test = 20, # full
        patch_height = patch_height,
        patch_width = patch_width,
        stride_height = stride_height,
        stride_width = stride_width
    )
else:
    patches_imgs_test, patches_masks_test = get_data_testing(
        DRIVE_test_imgs_original = DRIVE_test_imgs_original,  #original
        DRIVE_test_groudTruth = DRIVE_test_ground_truth, 
        Imgs_to_test = 20,
        patch_height = patch_height,
        patch_width = patch_width,
    )

#================ Run the prediction of the patches ==================================
best_last = 'best'

model = model_from_json(open(f'{name_experiment}/architecture.json').read())
model.load_weights(f'{name_experiment}/best_weights.h5')

predictions = model.predict(patches_imgs_test, batch_size=10, verbose=2)

#===== Convert the prediction arrays in corresponding images
pred_patches = pred_to_imgs(predictions, patch_height, patch_width, "threshold")

#========== Elaborate and visualize the predicted images ====================
pred_imgs = None
orig_imgs = None
gtruth_masks = None
if average_mode == True:
    pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)# predictions
    orig_imgs = test_imgs_orig[0:pred_imgs.shape[0],:,:,:]
    gtruth_masks = masks_test  #ground truth masks
else:
    pred_imgs = recompone(pred_patches,13,12)       # predictions
    orig_imgs = recompone(patches_imgs_test,13,12)  # originals
    gtruth_masks = recompone(patches_masks_test,13,12)  #masks

kill_border(pred_imgs, test_border_masks)
## back to original dimensions
orig_imgs = orig_imgs[:,:,0:full_img_height,0:full_img_width]
pred_imgs = pred_imgs[:,:,0:full_img_height,0:full_img_width]
gtruth_masks = gtruth_masks[:,:,0:full_img_height,0:full_img_width]

visualize(group_images(orig_imgs,N_visual),f'{name_experiment}/all_originals')#.show()
visualize(group_images(pred_imgs,N_visual),f'{name_experiment}/all_predictions')#.show()
visualize(group_images(gtruth_masks,N_visual),f'{name_experiment}/all_groundTruths')#.show()
#visualize results comparing mask and prediction:
assert (orig_imgs.shape[0]==pred_imgs.shape[0] and orig_imgs.shape[0]==gtruth_masks.shape[0])
N_predicted = orig_imgs.shape[0]
group = N_visual
assert (N_predicted%group==0)
for i in range(int(N_predicted/group)):
    orig_stripe = group_images(orig_imgs[i*group:(i*group)+group,:,:,:],group)
    masks_stripe = group_images(gtruth_masks[i*group:(i*group)+group,:,:,:],group)
    pred_stripe = group_images(pred_imgs[i*group:(i*group)+group,:,:,:],group)
    total_img = np.concatenate((orig_stripe,masks_stripe,pred_stripe),axis=1)
    visualize(total_img,f'{name_experiment}/Original_GroundTruth_Prediction_{str(i)}')#.show()

#====== Evaluate the results
print('\n\n========  Evaluate the results =======================')
#predictions only inside the FOV
y_scores, y_true = pred_only_FOV(pred_imgs,gtruth_masks, test_border_masks)  #returns data only inside the FOV
print('Calculating results only inside the FOV:')
print(f'y scores pixels:{y_scores.shape[0]} (radius 270: 270*270*3.14==228906), including background around retina:{pred_imgs.shape[0]*pred_imgs.shape[2]*pred_imgs.shape[3]} (584*565==329960)')
print(f'y true pixels: {y_true.shape[0]} (radius 270: 270*270*3.14==228906), including background around retina: {gtruth_masks.shape[2]*gtruth_masks.shape[3]*gtruth_masks.shape[0]} (584*565==329960)')

#Area under the ROC curve
fpr, tpr, thresholds = roc_curve((y_true), y_scores)
AUC_ROC = roc_auc_score(y_true, y_scores)
# test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
print(f'Area under the ROC curve: {AUC_ROC}')
roc_curve =plt.figure()
plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
plt.title('ROC curve')
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.legend(loc="lower right")
plt.savefig(f'{name_experiment}/ROC.png')

#Precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
precision = np.fliplr([precision])[0]  #so the array is increasing (you won't get negative AUC)
recall = np.fliplr([recall])[0]  #so the array is increasing (you won't get negative AUC)
AUC_prec_rec = np.trapz(precision,recall)
print(f'Area under Precision-Recall curve: {AUC_prec_rec}')
prec_rec_curve = plt.figure()
plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
plt.title('Precision - Recall curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower right")
plt.savefig(f'{name_experiment}/Precision_recall.png')

#Confusion matrix
threshold_confusion = 0.5
print(f'Confusion matrix:Custom threshold (for positive) of {threshold_confusion}')
y_pred = np.empty((y_scores.shape[0]))
for i in range(y_scores.shape[0]):
    if y_scores[i]>=threshold_confusion:
        y_pred[i]=1
    else:
        y_pred[i]=0
confusion = confusion_matrix(y_true, y_pred)
print(confusion)
accuracy = 0
if float(np.sum(confusion))!=0:
    accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
print(f'Global Accuracy: {accuracy}')
specificity = 0
if float(confusion[0,0]+confusion[0,1])!=0:
    specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
print(f'Specificity:{specificity}')
sensitivity = 0
if float(confusion[1,1]+confusion[1,0])!=0:
    sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
print(f'Sensitivity:{sensitivity}')
precision = 0
if float(confusion[1,1]+confusion[0,1])!=0:
    precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
print(f'Precision:{precision}')

#Jaccard similarity inde
jaccard_index = jaccard_score(y_true, y_pred)
print(f'Jaccard similarity score:{jaccard_index}')

#F1 score
F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
print(f'F1 score (F-measure):{F1_score}')

#Save the results
with open(f'{name_experiment}/performances.txt', 'w') as f:
    f.write("Area under the ROC curve: "+str(AUC_ROC)
                    + "\nArea under Precision-Recall curve: " +str(AUC_prec_rec)
                    + "\nJaccard similarity score: " +str(jaccard_index)
                    + "\nF1 score (F-measure): " +str(F1_score)
                    +"\n\nConfusion matrix:"
                    +str(confusion)
                    +"\nACCURACY: " +str(accuracy)
                    +"\nSENSITIVITY: " +str(sensitivity)
                    +"\nSPECIFICITY: " +str(specificity)
                    +"\nPRECISION: " +str(precision)
                    )