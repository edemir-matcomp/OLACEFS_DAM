from __future__ import division
import os
import models as M
import utils as U
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from tensorflow import keras
import skimage
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.color import gray2rgb

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataset_add = "/home/users/DATASETS/MapBiomas_SAR/"

model = M.BCDU_net_D3(input_size = (32,32,10))
model.summary()
model.load_weights('weights')

dataset_mean_ind = (-15.34541179, -8.87553847)
dataset_std_ind = (1.53520544, 1.45585154)

file = open("data_split/test.txt", "r")
te_list = [line.rstrip('\n') for line in file]
file.close()

for idx in range(len(te_list)):

    print(idx)
    path = te_list[idx]
   
    mask = skimage.img_as_float64(imread(dataset_add + "/GEE_mapbiomas_masks/"+ path + ".tif"))

    if np.max(mask.shape) <= 400:  

        mask = mask/255.
        mask  = np.expand_dims(mask, axis=2)

        img1  = U.normalization(skimage.img_as_float64(imread(dataset_add + "GEE_mapbiomas/"+ path + "_2019-01-01.tif")), mean=dataset_mean_ind, std=dataset_std_ind)
        img2  = U.normalization(skimage.img_as_float64(imread(dataset_add + "GEE_mapbiomas/"+ path + "_2019-04-01.tif")), mean=dataset_mean_ind, std=dataset_std_ind)
        img3  = U.normalization(skimage.img_as_float64(imread(dataset_add + "GEE_mapbiomas/"+ path + "_2019-07-01.tif")), mean=dataset_mean_ind, std=dataset_std_ind)
        img4  = U.normalization(skimage.img_as_float64(imread(dataset_add + "GEE_mapbiomas/"+ path + "_2019-10-01.tif")), mean=dataset_mean_ind, std=dataset_std_ind)
        img5  = U.normalization(skimage.img_as_float64(imread(dataset_add + "GEE_mapbiomas/"+ path + "_2020-01-01.tif")), mean=dataset_mean_ind, std=dataset_std_ind)

        imgcon = np.concatenate((img1,img2,img3,img4,img5),axis=2)

        data   = U.forward_crop(imgcon, window=(32,32), channels=10, stride=4)
        labels = U.forward_crop(mask, (32,32), channels=1, stride=4)
        
        pred = model.predict(data, batch_size=8, verbose=100)
        pred = U.reconstruct(pred, mask.shape, window=(32,32), channels=1, stride=4)


        y_scores_i = pred.reshape(pred.shape[0]*pred.shape[1]*pred.shape[2], 1)

        y_true_i = mask.reshape(mask.shape[0]*mask.shape[1]*mask.shape[2], 1)


        y_scores_i = np.where(y_scores_i>0.5, 1, 0)
        y_true_i   = np.where(y_true_i>0.5, 1, 0)

        if idx == 0:
            y_scores = y_scores_i
            y_true   = y_true_i
        else:
            overlap = y_scores_i*y_true_i # Logical AND
            union = y_scores_i + y_true_i # Logical OR
            IOU = overlap.sum()/float(union.sum()) #

            # only evaluate imgs with iou bigger than 0.2
            if IOU > 0.2:
                y_scores = np.concatenate((y_scores, y_scores_i), axis=0)
                y_true  = np.concatenate((y_true, y_true_i), axis=0)
            else:
                print("ops_iou: ", IOU)

output_folder = 'output/'

#Area under the ROC curve
fpr, tpr, thresholds = roc_curve((y_true), y_scores)
AUC_ROC = roc_auc_score(y_true, y_scores)
print ("\nArea under the ROC curve: " +str(AUC_ROC))
roc_curve =plt.figure()
plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
plt.title('ROC curve')
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.legend(loc="lower right")
plt.savefig(output_folder+"ROC.png")

#Precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
precision = np.fliplr([precision])[0] 
recall = np.fliplr([recall])[0]
AUC_prec_rec = np.trapz(precision,recall)
print ("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))
prec_rec_curve = plt.figure()
plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
plt.title('Precision - Recall curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower right")
plt.savefig(output_folder+"Precision_recall.png")

#Confusion matrix
threshold_confusion = 0.5
print ("\nConfusion matrix:  Custom threshold (for positive) of " +str(threshold_confusion))
y_pred = np.empty((y_scores.shape[0]))
for i in range(y_scores.shape[0]):
    if y_scores[i]>=threshold_confusion:
        y_pred[i]=1
    else:
        y_pred[i]=0
confusion = confusion_matrix(y_true, y_pred)
print (confusion)
accuracy = 0
if float(np.sum(confusion))!=0:
    accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
print ("\nGlobal Accuracy: " +str(accuracy))
if float(np.sum(confusion))!=0:
    baccuracy = (float(confusion[0,0])/float(confusion[0,0]+confusion[0,1]) + 
                float(confusion[1,1])/float(confusion[1,0]+confusion[1,1]))/2
print ("Balanced Accuracy: " +str(baccuracy))
specificity = 0
if float(confusion[0,0]+confusion[0,1])!=0:
    specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
print ("Specificity: " +str(specificity))
sensitivity = 0
if float(confusion[1,1]+confusion[1,0])!=0:
    sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
print ("Sensitivity: " +str(sensitivity))
precision = 0
if float(confusion[1,1]+confusion[0,1])!=0:
    precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
print ("Precision: " +str(precision))

#Jaccard similarity index
jaccard_index = jaccard_score(y_true, y_pred)
print ("\nJaccard score: " +str(jaccard_index))

#F1 score
F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
print ("\nF1 score (F-measure): " +str(F1_score))

#Save the results
file_perf = open(output_folder+'test.txt', 'w')
file_perf.write("Area under the ROC curve: "+str(AUC_ROC)
                + "\nArea under Precision-Recall curve: " +str(AUC_prec_rec)
                + "\nJaccard similarity score: " +str(jaccard_index)
                + "\nF1 score (F-measure): " +str(F1_score)
                +"\n\nConfusion matrix:"
                +str(confusion)
                +"\n\nACCURACY: " +str(accuracy)
                +"\nBALANCED ACCURACY: " +str(baccuracy)
                +"\nSENSITIVITY: " +str(sensitivity)
                +"\nSPECIFICITY: " +str(specificity)
                +"\nPRECISION: " +str(precision)
                )
file_perf.close()