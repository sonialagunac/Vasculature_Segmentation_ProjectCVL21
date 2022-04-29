
#Unsupervised classical methods

import numpy as np
import tifffile as T
from skimage.filters import threshold_otsu
from skimage.filters import threshold_li
from skimage.filters import threshold_isodata
from skimage import io
import time
from skimage.segmentation import morphological_chan_vese

#Functions for metrics computation
def batch_intersection_union(predict, target, nclass=2, thr=0.5):
    predict = predict > thr
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = predict + 1
    target = target + 1

    predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target)
    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    IoU = (np.float64(1.0) * area_inter / (np.spacing(1, dtype=np.float64) + area_union)).mean()
    return IoU

def batch_jaccard_index_and_dice_coefficient(predict, target, thr=0.5):
    predict = predict > thr
    predict = predict + 1
    target = target + 1

    tp = np.sum(((predict == 2) * (target == 2)) * (target > 0))
    fp = np.sum(((predict == 2) * (target == 1)) * (target > 0))
    fn = np.sum(((predict == 1) * (target == 2)) * (target > 0))
    tn = np.sum(((predict == 1) * (target == 1)) * (target > 0))

    ji = float(np.nan_to_num(tp / (tp + fp + fn)))
    dice = float(np.nan_to_num(2 * tp / (2 * tp + fp + fn)))
    return ji, dice

def batch_precision_recall(predict, target, thr=0.5):
    predict = predict > thr
    predict = predict + 1
    target = target + 1

    tp = np.sum(((predict == 2) * (target == 2)) * (target > 0))
    fp = np.sum(((predict == 2) * (target == 1)) * (target > 0))
    fn = np.sum(((predict == 1) * (target == 2)) * (target > 0))

    precision = float(np.nan_to_num(tp / (tp + fp)))
    recall = float(np.nan_to_num(tp / (tp + fn)))
    return precision, recall

def batch_sens_spec(predict, target, thr=0.5):
    predict = predict > thr
    predict = predict + 1
    target = target + 1

    tp = np.sum(((predict == 2) * (target == 2)) * (target > 0))
    fp = np.sum(((predict == 2) * (target == 1)) * (target > 0))
    tn = np.sum(((predict == 1) * (target == 1)) * (target > 0))
    fn = np.sum(((predict == 1) * (target == 2)) * (target > 0))

    sensitivity = float(np.nan_to_num(tp / (tp + fn)))
    specificity = float(np.nan_to_num(tn / (tn + fp)))
    return sensitivity, specificity

def batch_pix_accuracy(predict, target, thr=0.5):
    predict = predict > thr
    predict = predict + 1
    target = target + 1
    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target) * (target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    acc = np.float64(1.0) * pixel_correct / (np.spacing(1, dtype=np.float64) + pixel_labeled)
    return acc

#DataLoader

path_img_train=r'C:\Users\sonil\PycharmProjects\Semester_project\DeepVess\input_data\deepvess_slice\image_test\HaftJavaherian_DeepVess2018_GroundTruthImage_25.tif'
img_train=np.divide(T.imread(path_img_train),np.amax(T.imread(path_img_train)))
#Include label path if metrics computation are desired
path_label_train=r'C:\Users\sonil\PycharmProjects\Semester_project\DeepVess\input_data\deepvess_slice\label_test\HaftJavaherian_DeepVess2018_GroundTruthLabel_25.tif'
lbl_train=np.divide(T.imread(path_label_train),np.amax(T.imread(path_label_train)))

#Dataloader for inference

#Test crop of deepvess data or inhouse data if cross-inference is desired
path_img_test=r'C:\Users\sonil\PycharmProjects\Semester_project\DeepVess\input_data\deepvess_slice\image_test\HaftJavaherian_DeepVess2018_GroundTruthImage_25.tif'
img_test=np.divide(T.imread(path_img_test),np.amax(T.imread(path_img_test)))
#Include label path if metrics computation are desired
path_label_test=r'C:\Users\sonil\PycharmProjects\Semester_project\DeepVess\input_data\deepvess_slice\label_test\HaftJavaherian_DeepVess2018_GroundTruthLabel_25.tif'
lbl_test=np.divide(T.imread(path_label_test),np.amax(T.imread(path_label_test)))


#Models threshold computation

#Computation of the time required can be added as seen below for every model
#Otsu
tic=time.time()
thres_otsu=threshold_otsu(img_train)
toc=time.time()
print("Time ellapsed in Otsu threshold computation:", toc-tic, "seconds")
#Li
thres_li=threshold_li(img_train)
#ISODATA
thres_iso=threshold_isodata(img_train)

#Prediction computation
#Otsu
out_test_otsu=img_test>=thres_otsu
#Li
out_test_li=img_test>=thres_li
#ISODATA
out_test_iso=img_test>=thres_iso
#ACWE
out_actcont=morphological_chan_vese(img_test, 10)

io.imsave(r'C:\Users\sonil\PycharmProjects\Semester project\Data_output\classical_unsup\out_otsu_inhouse.tif', out_test_otsu, plugin='tifffile')
io.imsave(r'C:\Users\sonil\PycharmProjects\Semester project\Data_output\classical_unsup\out_li_inhouse.tif', out_test_li, plugin='tifffile')
io.imsave(r'C:\Users\sonil\PycharmProjects\Semester project\Data_output\classical_unsup\out_iso_inhouse.tif', out_test_iso, plugin='tifffile')
io.imsave(r'C:\Users\sonil\PycharmProjects\Semester project\Data_output\classical_unsup\out_ACWE_inhouse.tif', out_actcont, plugin='tifffile')

#Metrics computation

ji_actcont, dice_actcont = batch_jaccard_index_and_dice_coefficient(out_actcont, lbl_test)
iou_actcont=batch_intersection_union(out_actcont, lbl_test)
prec_actcont, rec = batch_precision_recall(out_actcont, lbl_test)
sens_actcont,spec_actcont = batch_sens_spec(out_actcont, lbl_test)
acc_actcont = batch_pix_accuracy(out_actcont, lbl_test)
print('ACWE metrics:',iou_actcont,'iou',ji_actcont,'JI', dice_actcont, 'Dice',prec_actcont, 'prec', sens_actcont,'sens', spec_actcont, 'spec', acc_actcont, 'acc')

ji_otsu, dice_otsu = batch_jaccard_index_and_dice_coefficient(out_test_otsu, lbl_test)
iou_otsu=batch_intersection_union(out_test_otsu, lbl_test)
prec_otsu, rec = batch_precision_recall(out_test_otsu, lbl_test)
sens_otsu,spec_otsu=batch_sens_spec(out_test_otsu, lbl_test)
acc_otsu = batch_pix_accuracy(out_test_otsu, lbl_test)
print('Otsu metrics:',iou_otsu,'iou',ji_otsu,'JI', dice_otsu, 'Dice',prec_otsu, 'prec', sens_otsu,'sens', spec_otsu, 'spec', acc_otsu, 'acc')

ji_li, dice_li = batch_jaccard_index_and_dice_coefficient(out_test_li, lbl_test)
iou_li=batch_intersection_union(out_test_li, lbl_test)
prec_li, rec = batch_precision_recall(out_test_li, lbl_test)
sens_li,spec_li = batch_sens_spec(out_test_li, lbl_test)
acc_li = batch_pix_accuracy(out_test_li, lbl_test)
print('Li metrics:',iou_li,'iou',ji_li,'JI', dice_li, 'Dice',prec_li, 'prec', sens_li,'sens', spec_li, 'spec', acc_li, 'acc')

ji_iso, dice_iso = batch_jaccard_index_and_dice_coefficient(out_test_iso, lbl_test)
iou_iso = batch_intersection_union(out_test_iso, lbl_test)
prec_iso, rec =batch_precision_recall(out_test_iso, lbl_test)
sens_iso,spec_iso = batch_sens_spec(out_test_iso, lbl_test)
acc_iso=batch_pix_accuracy(out_test_iso, lbl_test)
print('ISODATA metrics:',iou_iso,'iou',ji_iso,'JI', dice_iso, 'Dice',prec_iso, 'prec', sens_iso,'sens', spec_iso, 'spec', acc_iso, 'acc')
