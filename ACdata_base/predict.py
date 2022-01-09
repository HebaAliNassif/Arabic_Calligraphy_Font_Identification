from __future__ import division
from sklearn import svm, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from scipy.signal import convolve2d
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold,train_test_split, cross_val_score
from statistics import mean, stdev
from scipy import signal
from skimage.morphology import skeletonize
from sklearn.neighbors import KNeighborsClassifier
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys
import time
import re

def read_mu_sigma():
    mu = np.loadtxt("mu.txt", dtype=np.float32)
    sigma = np.loadtxt("sigma.txt", dtype=np.float32)
    return mu,sigma

def write_mu_sigma(mu,sigma):
    with open("mu.txt", "w") as mu_file:
        np.savetxt(mu_file,mu, fmt="%s")
        mu_file.close()

    with open("sigma.txt", "w") as sigma_file:
        np.savetxt(sigma_file,sigma, fmt="%s")
        sigma_file.close()
   
def standardize_train(X):
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    normalized_X = (X-mu)/sigma
    
    return normalized_X, mu, sigma

def standardize_test(X,mu,sigma):
    normalized_X = (X-mu)/sigma
    return normalized_X

def hyperparameter_tuning(trainFeatures, y_train, standardized_train, standardized_test):
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
  C_values = [2**(-5),2**(-3),2**(-1),2**1,2**3,2**5,2**7,2**9,2**10,2**11]
  gamma_values = [2**(-15), 2**(-13), 2**(-11), 2**(-9), 2**(-7), 2**(-5), 2**(-3), 2**(-1), 2**(1),2**(3)]
  kernels = ['rbf', 'poly', 'sigmoid']
  maximum_acc = float('-inf')
  maximum = []
  stdv = 0
  best_C = None
  best_gamma = None
  best_kernel = None
  for C_test in C_values:
    for gamma_test in gamma_values:
      for kernel_test in kernels:
        clf_accu_stratified = []
        clf_svm = svm.SVC(C=C_test, gamma=gamma_test, kernel=kernel_test)
        for train_index, test_index in skf.split(trainFeatures,y_train):
          x_train_fold, x_test_fold = standardized_train[train_index], standardized_test[test_index]
          y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
          clf_svm.fit(x_train_fold, y_train_fold)
          clf_accu_stratified.append(clf_svm.score(x_test_fold, y_test_fold))
        acc = mean(clf_accu_stratified)*100
        print('\nOverall Accuracy:', acc, '%')
        print('\nStandard Deviation is:', stdev(clf_accu_stratified))
        if acc >= maximum_acc:
          maximum_acc = acc
          best_C = C_test
          best_gamma = gamma_test
          best_kernel = kernel_test
          maximum.append((acc, best_C, best_gamma, best_kernel))
          stdv = stdev(clf_accu_stratified)

  maximum.sort(reverse=True)
  # print(maximum[0])
  # print(maximum[1])
  # print(maximum[2])
  # print(maximum[3])
  # print(maximum[4])
  return maximum
## Reference: https://stackoverflow.com/questions/23548863/converting-a-specific-matlab-script-to-python/23575137
#Perform local Phase Quantization
def lpq(img, winSize = 3, freqestim = 1, mode = 'nh'):
    
    rho = 0.90
    STFTalpha = 1/winSize    # alpha in STFT approaches (for Gaussian derivative alpha=1)
    sigmaS = (winSize-1)/4   # Sigma for STFT Gaussian window (applied if freqestim==2)
    sigmaA = 8/(winSize-1)   # Sigma for Gaussian derivative quadrature filters (applied if freqestim==3)

    convmode = 'valid'   # Compute descriptor responses only on part that have full neigborhood. Use 'same' if all pixels are included (extrapolates np.image with zeros).

    img = np.float64(img)                # Convert np.image to double
    r = (winSize-1)/2                    # Get radius from window size
    x = np.arange(-r, r+1)[np.newaxis]   # Form spatial coordinates in window

    if freqestim == 1:  #  STFT uniform window
        # Basic STFT filters
        w0 = np.ones_like(x)
        w1 = np.exp(-2*np.pi*x*STFTalpha*1j)
        w2 = np.conj(w1)

    ## Run filters to compute the frequency response in the four points. Store np.real and np.imaginary parts separately
    # Run first filter
    filterResp1 = convolve2d(convolve2d(img,w0.T,convmode),w1,convmode)
    filterResp2 = convolve2d(convolve2d(img,w1.T,convmode),w0,convmode)
    filterResp3 = convolve2d(convolve2d(img,w1.T,convmode),w1,convmode)
    filterResp4 = convolve2d(convolve2d(img,w1.T,convmode),w2,convmode)

    # Initilize frequency domain matrix for four frequency coordinates (np.real and np.imaginary parts for each frequency).
    freqResp = np.dstack([filterResp1.real, filterResp1.imag,
                         filterResp2.real, filterResp2.imag,
                         filterResp3.real, filterResp3.imag,
                         filterResp4.real, filterResp4.imag])

    ## Perform quantization and compute LPQ codewords
    inds = np.arange(freqResp.shape[2])[np.newaxis,np.newaxis,:]
    LPQdesc = ((freqResp>0)*(2**inds)).sum(2)

    ## Switch format to uint8 if LPQ code np.image is required as output
    if mode =='im':
        LPQdesc = np.uint8(LPQdesc)

    ## Histogram if needed
    if mode == 'nh' or mode == 'h':
        LPQdesc = np.histogram(LPQdesc.flatten(),range(256))[0]

    ## Normalize histogram if needed
    if mode == 'nh':
        LPQdesc = LPQdesc/LPQdesc.sum()

    #print(LPQdesc)
    return LPQdesc

    # 1. Perform Preprocessing
def preprocessing(image):
    ret2,th2 = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    result = np.array(th2, dtype = 'float')
    return result

# 2. Feature Extraction using Local Phase Quantization
def extractFeatures(image):
    return lpq(image, winSize = 5, mode ='nh')
def train_model_sum():
  print('Training and Fitting Model..')
  data = []
  data_labels = []
  # Read font names from file
  fontFile = open("names.txt",'r')
  fonts = np.loadtxt(fontFile, dtype='str')
  for font in fonts:
      fontDir, fontName = font.split("__")
      for file in os.listdir(fontDir):
          image = cv2.imread(fontDir+"/"+file,0)
          image_processed = preprocessing(image)
          data.append(image_processed)
          data_labels.append(fontDir)

  # Convert data to numpy array
  data = np.asarray(data, dtype=np.ndarray)
  data_labels = np.asarray(data_labels)

  N = data.shape[0]
  trainFeatures = np.zeros((N, 255))

  # Extract features from training data
  for i in range(trainFeatures.shape[0]):
      trainFeatures[i] = extractFeatures(data[i])

  # Standardize training data
  standardized_train, mu, sigma = standardize_train(trainFeatures)
  y_train = data_labels
  
  # Output mu and sigma to text file
  write_mu_sigma(mu,sigma)

  # Classifiers Combination
  clf_knn = KNeighborsClassifier(n_neighbors=5)
  clf_RF = RandomForestClassifier(max_depth=13, random_state=0)
  clf_svm1 = svm.SVC(C=32, gamma=0.001953125, kernel='rbf',probability=True)
  clf_svm2 = svm.SVC(C=8, gamma=0.001953125, kernel='rbf',probability=True)
  clf_svm3 = svm.SVC(C=2, gamma=8, kernel='poly',probability=True)
  clf_svm4 = svm.SVC(C=2, gamma=2, kernel='poly',probability=True)

  # Soft Majority vote
  clf_sum = VotingClassifier(estimators=[('knn', clf_knn),('svm1', clf_svm1),('rf', clf_RF),('svm2', clf_svm2),('svm3', clf_svm3),('svm4', clf_svm4)], voting='hard')
  clf_sum.fit(standardized_train, y_train)
  filename = 'finalized_model_sum.sav'
  pickle.dump(clf_sum, open(filename, 'wb'))

def train_model():
  print('Training and Fitting Model..')
  data = []
  data_labels = []
  # Read font names from file
  fontFile = open("names.txt",'r')
  fonts = np.loadtxt(fontFile, dtype='str')
  for font in fonts:
      fontDir, fontName = font.split("___")
      for file in os.listdir(fontDir):
          image = cv2.imread(fontDir+"/"+file,0)
          image_processed = preprocessing(image)
          data.append(image_processed)
          data_labels.append(fontDir)

  # Convert data to numpy array
  data = np.asarray(data, dtype=np.ndarray)
  data_labels = np.asarray(data_labels)

  N = data.shape[0]
  trainFeatures = np.zeros((N, 255))

  # Extract features from training data
  for i in range(trainFeatures.shape[0]):
      trainFeatures[i] = extractFeatures(data[i])

  # Standardize training data
  standardized_train, mu, sigma = standardize_train(trainFeatures)
  y_train = data_labels
  
  # Output mu and sigma to text file
  write_mu_sigma(mu,sigma)
  
  # Hard Majority Vote
  # ____________________________________________________________________________
  # clf_knn = KNeighborsClassifier(n_neighbors=5)
  # clf_svm1 = svm.SVC(C=32, gamma=0.001953125, kernel='rbf')
  # clf_svm2 = svm.SVC(C=8, gamma=0.001953125, kernel='rbf')
  # clf_svm3 = svm.SVC(C=2, gamma=8, kernel='poly')
  # clf_svm4 = svm.SVC(C=2, gamma=2, kernel='poly')

  # clf_max = VotingClassifier(estimators=[('knn', clf_knn),('svm1', clf_svm1),('svm2', clf_svm2),('svm3', clf_svm3),('svm4', clf_svm4)], voting='hard')

  # clf_max.fit(standardized_train, y_train)
  # filename = 'finalized_model_max.sav'
  # pickle.dump(clf_max, open(filename, 'wb'))

  # Classifiers Combination
  clf_knn = KNeighborsClassifier(n_neighbors=5)
  clf_svm1 = svm.SVC(C=32, gamma=0.001953125, kernel='rbf',probability=True)
  clf_svm2 = svm.SVC(C=8, gamma=0.001953125, kernel='rbf',probability=True)
  clf_svm3 = svm.SVC(C=2, gamma=8, kernel='poly',probability=True)
  clf_svm4 = svm.SVC(C=2, gamma=2, kernel='poly',probability=True)

  # Soft Majority vote
  clf_sum = VotingClassifier(estimators=[('knn', clf_knn),('svm1', clf_svm1),('svm2', clf_svm2),('svm3', clf_svm3),('svm4', clf_svm4)], voting='soft')
  clf_sum.fit(standardized_train, y_train)
  filename = 'finalized_model_sum.sav'
  pickle.dump(clf_sum, open(filename, 'wb'))

# Stratified Cross Validation function for determining accuracy
def stratified_cross_validation(trainFeatures, y_train, standardized_train, standardized_test, k, clf_sum):
  skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
  for train_index, test_index in skf.split(trainFeatures,y_train):
      x_train_fold, x_test_fold = standardized_train[train_index], standardized_test[test_index]
      y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
      clf_sum.fit(x_train_fold, y_train_fold)
      y_pred_max = clf_sum.predict(x_test_fold)
      print("Accuracy-Majority Vote:",metrics.accuracy_score(y_test_fold, y_pred_max)*100)
      print("F1 Score Micro-Majority Vote:",metrics.f1_score(y_test_fold, y_pred_max, average='micro')*100)
      print('--------------------------------------------------')

# Stratified Cross Validation function for determining accuracy
def normal_cross_validation(standardized_train, y_train, clf_sum, k):
  scores = cross_val_score(clf_sum, standardized_train, y_train, cv=k, scoring='f1_macro')
  print(scores) 
  print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


if len(sys.argv) > 1:
  test_dir = sys.argv[1]
  time_dir = os.path.join(sys.argv[2], "times.txt")
  results_dir = os.path.join(sys.argv[2], "results.txt")

# When using different scikit learn versions we encoutered this error.
# UserWarning: Trying to unpickle estimator LabelEncoder from version 0.24.1 when using version 1.0.2. 
# This might lead to breaking code or invalid results. Use at your own risk.
# To resolve it we re-trained the model

  #train_model_sum()
  #train_model()

  loaded_model = pickle.load(open('finalized_model_sum.sav', 'rb'))
  mu,sigma = read_mu_sigma()
  images = []
  i = 0
  with open(time_dir, "w") as time_file, open(results_dir, "w") as result_file:
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    img_filenames = sorted(os.listdir(test_dir),key= alphanum_key)      
    for file in img_filenames:
      try:
        image = cv2.imread(test_dir+"/"+file,0)
        start_time = time.time()
        image_processed = preprocessing(image)
        test_features = extractFeatures(image_processed)
        standardized_test = standardize_test(test_features,mu,sigma)
        y_pred_sum = loaded_model.predict([standardized_test])
        end_time = time.time()
        duration=end_time - start_time
        i += 1
        if str(round(duration, 2))==0:
          time_file.write(str(0.001))
        else:
          time_file.write(str(round(duration, 2))) 

        result_file.write(str(y_pred_sum[0]))

        if i < len(os.listdir(test_dir)):
          time_file.write('\n')
          result_file.write('\n')
      except:
        time_file.write(str(round(0.76, 2)))
        result_file.write(str(-1))  
      
    result_file.close()
    time_file.close()