import cv2
import numpy as np 

import sys 

from glob import glob
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

from scipy import stats

from sklearn.decomposition import PCA

class ImageHelpers:
    def __init__(self):
        self.sift_object = cv2.xfeatures2d.SIFT_create()

    def gray(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def features(self, image):
        keypoints, descriptors = self.sift_object.detectAndCompute(image, None)
        return [keypoints, descriptors]


class BOVHelpers:
    def __init__(self, n_clusters = 20):
        self.n_clusters = n_clusters
        self.kmeans_obj = KMeans(n_clusters = n_clusters)
        self.kmeans_ret = None
        self.descriptor_vstack = None
        self.mega_histogram = None
        self.db = None
        self.clf  = SVC()	
        self.n_images = 0
        self.labelList = None

    def cluster(self, featuresCount, labelList):
        """	
        cluster using KMeans algorithm, 

        """
        self.kmeans_ret = self.kmeans_obj.fit_predict(self.descriptor_vstack)

        #np.set_printoptions(threshold=sys.maxsize)
        #print('Full:')
        #print(self.descriptor_vstack)
        #self.db = DBSCAN(eps=200.0, min_samples=5).fit(self.descriptor_vstack)
        #core_samples_mask = np.zeros_like(self.db.labels_, dtype=bool)
        #core_samples_mask[self.db.core_sample_indices_] = True
        #labels = self.db.labels_
        #last = 0
        #count = 0
        #for  k in featuresCount:
        #    print('Etiquetas(' +str(count)+') : ' + str(k) + ' Label: ' + labelList[count])
        #    new = labels[last:last+k-1]
        #    new = new[new != -1]
        #    print(new)
        #    moda = stats.mode(new)
        #    print('Moda: ')
        #    print(moda[0])
        #    hist = np.histogram(new)
        #    print(hist)
        #    last = last+k
        #    count += 1

        #n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        #n_noise_ = list(labels).count(-1)
        #print('Estimated number of clusters: %d' % n_clusters_)
        #print('Estimated number of noise points: %d' % n_noise_)
        #self.n_clusters = n_clusters_

        #pca = PCA(n_components=2)
        #pca.fit(self.descriptor_vstack)
        #X_ = pca.transform(self.descriptor_vstack)
        #db2 = DBSCAN(eps=200.0, min_samples=5).fit(X_)
        #core_samples_mask = np.zeros_like(self.db.labels_, dtype=bool)
        #core_samples_mask[self.db.core_sample_indices_] = True
        #labels = self.db.labels_
        #n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        #n_noise_ = list(labels).count(-1)
        #print('Estimated number of clusters: %d' % n_clusters_)
        #print('Estimated number of noise points: %d' % n_noise_)

        #sys.exit(0)

        #for k in self.descriptor_vstack:
        #    print(labels[k])


    def developVocabulary(self,n_images, descriptor_list, labelList, ret = None):

        """
        Each cluster denotes a particular visual word 
        Every image can be represeted as a combination of multiple 
        visual words. The best method is to generate a sparse histogram
        that contains the frequency of occurence of each visual word 

        Thus the vocabulary comprises of a set of histograms of encompassing
        all descriptions for all images

        """

        np.set_printoptions(threshold=sys.maxsize)
        self.mega_histogram = np.array([np.zeros(self.n_clusters) for i in range(n_images)])
        old_count = 0
        for i in range(n_images):
            l = len(descriptor_list[i])
            print("Imagen : " + str(i) + " len: " + str(l) + " label: " + labelList[i]);
            for j in range(l):
                if ret is None:
                    idx = self.kmeans_ret[old_count+j]
                else:
                    idx = ret[old_count+j]
                self.mega_histogram[i][idx] += 1
            old_count += l
        print "Vocabulary Histogram Generated"
        #print(self.mega_histogram)
        self.n_images = n_images
        self.labelList = labelList

    def standardize(self, std=None):
        """

        standardize is required to normalize the distribution
        wrt sample size and features. If not normalized, the classifier may become
        biased due to steep variances.

        """
        if std is None:
            self.scale = StandardScaler().fit(self.mega_histogram)
            self.mega_histogram = self.scale.transform(self.mega_histogram)
        else:
            print "STD not none. External STD supplied"
            self.mega_histogram = std.transform(self.mega_histogram)

        n_images = self.n_images
        labelList = self.labelList
        print "Histograma normalizado"
        for i in range(n_images):
            print("Imagen : " + str(i) + " label: " + labelList[i]);
            print(self.mega_histogram[i])

    def formatND(self, l):
        """	
        restructures list into vstack array of shape
        M samples x N features for sklearn

        """
        vStack = np.array(l[0])
        for remaining in l[1:]:
            vStack = np.vstack((vStack, remaining))
        self.descriptor_vstack = vStack.copy()
        return vStack

    def train(self, train_labels):
        """
        uses sklearn.svm.SVC classifier (SVM) 


        """
        print "Training SVM"
        print self.clf
        print "Train labels", train_labels
        self.clf.fit(self.mega_histogram, train_labels)
        print "Training completed"

    def predict(self, iplist):
        predictions = self.clf.predict(iplist)
        return predictions

    def plotHist(self, vocabulary = None):
        print "Plotting histogram"
        if vocabulary is None:
            vocabulary = self.mega_histogram

        x_scalar = np.arange(self.n_clusters)
        y_scalar = np.array([abs(np.sum(vocabulary[:,h], dtype=np.int32)) for h in range(self.n_clusters)])

        print y_scalar

        #plt.bar(x_scalar, y_scalar)
        #plt.xlabel("Visual Word Index")
        #plt.ylabel("Frequency")
        #plt.title("Complete Vocabulary Generated")
        #plt.xticks(x_scalar + 0.4, x_scalar)
        #plt.show()

class FileHelpers:

    def __init__(self):
        pass

    def getFiles(self, path):
        """
        - returns  a dictionary of all files 
        having key => value as  objectname => image path

        - returns total number of files.

        """
        imlist = {}
        namelist = []
        count = 0
        for each in glob(path + "*"):
            word = each.split("/")[-1]
            print " #### Reading image category ", word, " ##### "
            imlist[word] = []
            for imagefile in glob(path+word+"/*"):
                print "Reading file ", imagefile
                im = cv2.imread(imagefile, 0)
                imlist[word].append(im)
                namelist.append(imagefile)
                count +=1 

        return [imlist, count, namelist]

