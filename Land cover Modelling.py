import time
import gdal
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import cohen_kappa_score
from scipy import interp
from itertools import cycle
from sklearn.utils.multiclass import type_of_target
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd

# compute the time of execution
start_time = time.time()
# fixing the random seed for reproducibility
seed=np.random.seed(7)
n_trails = 10 # set the number of folds to run during random forest training

# Output of the raster
output_fname1 ='Simulated landcover_2028 BAU_R3_lcmodel.tif'
output_fname2='Simulated land cover 2028 REDD Scenario_R3_lcmodel.tif'
# Importing the rasters
# land cover maps
lc_path =os.path.join( r"E:\Erasmus Mundus Master Thesis\Data\Processed Rasters")

# get the names of the raster dataset
lc_ras = [os.path.join(lc_path,i) for i in os.listdir(lc_path) if i.endswith('.tif')]
lc_ras.sort()
#get the names of the raster dataset
lc_rasname = [i.split('_')[-1] for i in os.listdir(lc_path) if i.endswith('.tif')]
lc_rasname.sort()
print('Raster to  Process:', lc_rasname)
bands_data = []
for i in lc_ras:
    ras = gdal.Open(i,gdal.GA_ReadOnly)
    #geotransformation
    gt = ras.GetGeoTransform()
    proj = ras.GetProjectionRef()
    #Importing bands as a set of arrays
    n_bads = ras.RasterCount
    band =  ras.GetRasterBand(1)
    band= band.ReadAsArray()
    band_vector=band.reshape(1, band.shape[0]*band.shape[1])
    band_vector=np.transpose(band_vector)
    bands_data.append(band_vector)
    print('Band: ', i, ' is imported')
# extract the raster vector 
# land cover maps raster
lc2004 =bands_data[0]
d2009=bands_data[1]
d2016=bands_data[2]
# soil raster
soil=bands_data[3] 

# biophysical variables raster
slope=bands_data[13]
cities=bands_data[10]
towns=bands_data[11]
roads=bands_data[12]
# land cover neighbourhood distance maps for 2004 and 2016
ba4=bands_data[4]
blt4=bands_data[5]
c4=bands_data[6]
f4=bands_data[7]
nonf4=bands_data[8]
w4=bands_data[9]
ba16=bands_data[14]
blt16=bands_data[15]
c16=bands_data[16]
f16=bands_data[17]
nonf16=bands_data[18]
w16=bands_data[19]

# subtracting one to enable the one hot encoding to give the respective classes
lc2004=lc2004-1
d2016=d2016-1
soil=soil-1
# one hot encoding for the independent and dependent variables
indLabels=to_categorical(lc2004)
dLabels=to_categorical(d2016)
dsoil=to_categorical(soil)
type_of_target(dLabels)

# combine independent and dependent variables for sampling
data= np.hstack((indLabels, dsoil, slope, cities, towns, roads, ba4, blt4, c4, f4, nonf4, w4, d2016))

# Sampling the data
n_samples=3000 
# function to sample points for each land cover class
def sampledata(data, column):
    landcover=data[np.where(data[:, column] ==1)]
    Sample =landcover[np.random.choice(landcover.shape[0], size=n_samples, replace=False), :]
    return Sample
cSample=sampledata(data, 0)
fSample=sampledata(data, 1)
nfSample=sampledata(data, 2)
buSample=sampledata(data, 3)
bSample=sampledata(data, 4)
wSample=sampledata(data, 5)

sample_data=np.vstack((cSample, fSample, nfSample, buSample, bSample, wSample)) # combine all samples
ind= sample_data[:,0:21] # extract the independent variables
d= sample_data[:,21:22] # extract the dependent variables
ind_all= data [:,0:21]# all the independent variable matrix

# CREATE AND TRAIN MODEL
''' Using Random Forest with 200 estimators and entropy criteria with percentage split 0f 67% and 
33% for training and testing dataset, a 10 folds validation was carried out as follows
'''
oa = []
for t in range(n_trails):
    #data split
    ind_train, ind_test, d_train, d_test = train_test_split(ind, d, test_size=0.33)
    # train the model
    model=RandomForestClassifier(n_estimators = 200, criterion='entropy')
    # evaluate the model
    model1=model.fit(ind_train, d_train)   
    pred=model1.predict(ind_test)
    oa_t = model1.score(ind_test, d_test)
    oa.append(oa_t)
    print("Test set score for % i: %f" % (t, oa_t))

oa_mean = np.mean(oa)
print('The mean accuracy is: ' , oa_mean)
oa_std = np.std(oa)
print('The standard deviation is: ', oa_std)

# Analyze the contribution of each variable to the classifier
''' The digit 1-6 represents the land cover classes; Cl, L.s, vcl and w_o represent the soil classess that is clayey, loamy,
sandy, and very clayey; sL is for slope, d1,d2,d3, d4, d5 and d6 are for the normalised neighbourhood distances; dC, dT and
dRd represnt the distance rasters for the cities, major towns and roads
'''
labels=['1', '2','3', '4', '5', '6', 'cl','L', 's', 'vcl', 'w_o', 'sL', 'dC', 'dT', 'dRd', 'd5','d4','d1','d2','d3', 'd6']

# convert the matrix to a data frame
df = pd.DataFrame(ind, index=range(1, ind.shape[0] + 1), columns=range(1, ind.shape[1] + 1))
df.columns = labels
# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(12, 12), dpi=100)
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
#plt.title('Variable Correlation Heatmap')
plt.savefig("Correlation Heatmap_lcmodel .pdf")
plt.show()

ind1=ind[:, 11:21]
df1 = pd.DataFrame(ind1, index=range(1, ind1.shape[0] + 1), columns=range(1, ind1.shape[1] + 1))
df1.columns = labels[11:21]
f, ax1 = plt.subplots(figsize=(12, 8), dpi=100)
ax1 = sns.boxplot(data=df1)
plt.savefig('Biophysical Variables Boxplots_lcmodel.pdf')
plt.show()

importance = model1.feature_importances_
y_pos = np.arange(len(importance))
significance=plt.figure(1)
plt.figure(figsize = (10,8), dpi=80)
plt.bar(y_pos,importance,align ='center',alpha=0.5)
plt.xticks(y_pos,labels)
plt.ylabel("Importance")
plt.title('Variable Importance in Random Forest')
significance.savefig("Level of Variable Importance_lcmodel .pdf", bbox_inches='tight')
plt.show()

# Predict the landcover classes for 2016
pred=model1.predict(ind_test)
# calculate the confusion matrix
conf=confusion_matrix(pred, d_test)
print("The confusion matrix: ", conf)   

lclabels={0:'cropland', 1:'Forest',2:'Non Forest', 3:'Built-up areas', 4:'Bare areas', 5:'Water'}
df_conf = pd.DataFrame(conf, range(6), range(6))
df_conf.columns = lclabels.values()
df_conf.rename(index=lclabels, inplace=True)
np.savetxt('Confusion Matrix_lcmodel.csv',df_conf, delimiter=",") # save the confusion matrix

# computation of the kappa statistics
kappa= cohen_kappa_score(d_test, d_test)
print("Kappa value is: ",kappa)

# compute the ROC curve and ROC area for each class
n_classes=6 # define the number of classes
# plot line width
lw=2
cScores=to_categorical(pred) # convert the predict classes to binary (one hot encoding)
dlab=to_categorical(d_test)
# create the dictionary for false positive rate, true positive rate and area under courve
fpr={}
tpr={}
roc_auc={}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(dlab[: , i], cScores[ :, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(dlab.ravel(), cScores.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# compute aggregate for all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# interpolate all the ROC curves
mean_tpr=np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# compute AUC
mean_tpr/=n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# plot all ROC curves
roc=plt.figure(2)
colors=cycle(['darkorange', 'lawnGreen', 'Green', 'brown','silver', 'aqua'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Land Cover Classes ROC Curves')
L=plt.legend(loc="lower right")
L.get_texts()[0].set_text('cropland (area = {1:0.2f})' ''.format(0, roc_auc[0]))
L.get_texts()[1].set_text('forest (area = {1:0.2f})' ''.format(1, roc_auc[1]))
L.get_texts()[2].set_text('non forest (area = {1:0.2f})' ''.format(2, roc_auc[2]))
L.get_texts()[3].set_text('built-up (area = {1:0.2f})' ''.format(3, roc_auc[3]))
L.get_texts()[4].set_text('bare-areas (area = {1:0.2f})' ''.format(4, roc_auc[4]))
L.get_texts()[5].set_text('water (area = {1:0.2f})' ''.format(5, roc_auc[5]))
plt.show()
roc.savefig("Land cover relative operating characteristicsR3.pdf", bbox_inches='tight')

'''SIMULATION FOR 2028 LAND COVER UNDER BUSINESS AS USUAL SCENARIO'''
# Generate the independent variable for simulating 2028 land cover
ind28= np.hstack((dLabels, dsoil, slope, cities, towns, roads, ba16, blt16, c16, f16, nonf16, w16))
# predict land cover for 2028
pred28=model.predict(ind28)
simlc28=(pred28+1).astype(int)

# reshape the vector raster to its original state to form the raster
simlc28=np.reshape(simlc28, (band.shape[0], band.shape[1]))

# Output the simulated landcover raster
def write_geotiff(fname, data, geo_transform, projection):
    #"""Create a GeoTIFF file with the given data."""
    driver= gdal.GetDriverByName('GTiff')
    rows, cols= data.shape
    dataset= driver.Create(fname, cols, rows, 1, gdal.GDT_Float64)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    band= dataset.GetRasterBand(1)
    band.WriteArray(data)
    dataset=None
# write and close the file for 2028 Business as usual
write_geotiff(output_fname1, simlc28, gt, proj)

'''SIMULATION FOR 2028 LAND COVER UNDER REDD SCENARIO
First start by retraining the model, therefore constrain the forest, cropland and built-up area from not changing and allow other 
land cover class to compete. This is done by only resampling pixels that did not change over the two epochs (2004 and 2016)
for forest, cropland and built-up areas, and the other remaining classes resampling done as before
'''
n_samples1=3000
def sampledata1(data, column1, column2, lcClass):
    landcover=data[np.where(data[:, column1] ==1)]
    landcover=landcover[np.where(landcover[:,column2]==lcClass)]
    Sample =landcover[np.random.choice(landcover.shape[0], size=n_samples1, replace=False), :]
    return Sample

cSample1=sampledata1(data,column1=0, column2=21, lcClass=0)
fSample1=sampledata1(data,column1=1, column2=21, lcClass=1)
buSample1=sampledata1(data,column1=3, column2=21, lcClass=3)

def sampledata2(data, column):
    landcover1=data[np.where(data[:, column] ==1)]
    landcover1=landcover1[np.where(landcover1[:,21]!=0)]
    Sample =landcover1[np.random.choice(landcover1.shape[0], size=n_samples1, replace=False), :]
    return Sample
nfSample1=sampledata2(data,2)
bSample1=sampledata2(data, 4)
wSample1=sampledata2(data, 5)

sample_data1=np.vstack((cSample1, fSample1, nfSample1, buSample1, bSample1, wSample1))

ind_redd= sample_data1[:,0:21] # extract the independent variables
d_redd= sample_data1[:,21:22] # extract the dependent variables

# CREATE AND RETRAIN MODEL
oa1 = []
for t1 in range(n_trails):
    #data split
    ind_train1, ind_test1, d_train1, d_test1 = train_test_split(ind_redd, d_redd, test_size=0.33)
    model2=model.fit(ind_train1, d_train1)   
    pred=model2.predict(ind_test1)
    oa_t1 = model2.score(ind_test1, d_test1)
    oa1.append(oa_t1)
    print("Test set score for %i: %f" % (t1, oa_t1))

oa_mean1 = np.mean(oa1)
print('The mean accuracy is: ' , oa_mean1)
oa_std1 = np.std(oa1)
print('The standard deviation is: ', oa_std1)
# predict 
pred28redd=model2.predict(ind28)
simlc28redd=(pred28redd+1).astype(int)
simlc28redd=np.reshape(simlc28redd, (band.shape[0], band.shape[1]))

# write and close the file for 2028 Business as usual
write_geotiff(output_fname2, simlc28redd, gt, proj)

# display time taken to execute the script
print("--- %s seconds ---" % (time.time() - start_time))