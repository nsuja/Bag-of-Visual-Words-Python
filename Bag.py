import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

import sys

import cv2
import numpy as np 
from glob import glob 
import argparse
from helpers import *



class BOV:
    def __init__(self, no_clusters):
        self.no_clusters = no_clusters
        self.train_path = None
        self.test_path = None
        self.im_helper = ImageHelpers()
        self.bov_helper = BOVHelpers(no_clusters)
        self.file_helper = FileHelpers()
        self.images = None
        self.trainImageCount = 0
        self.train_labels = np.array([])
        self.name_dict = {}
        self.descriptor_list = []
        self.name = []
        self.labelList = []

    def trainModel(self):
        """
        This method contains the entire module 
        required for training the bag of visual words model

        Use of helper functions will be extensive.

        """

        # read file. prepare file lists.
        self.images, self.trainImageCount, self.names = self.file_helper.getFiles(self.train_path)
        # extract SIFT Features from each image
        label_count = 0 
        counter = 0
        featuresCount = []
        labelList = []
        for word, imlist in self.images.iteritems():
            self.name_dict[str(label_count)] = word
            print "Computing Features for ", word
            for im in imlist:
                # cv2.imshow("im", im)
                # cv2.waitKey()
                self.train_labels = np.append(self.train_labels, label_count)
                kp, des = self.im_helper.features(im)
                print(self.names[counter] + " count " + str(np.shape(des)))
                num, features = np.shape(des)
                featuresCount.append(num)
                labelList.append(word)
                self.descriptor_list.append(des)
                counter += 1

            label_count += 1

        self.labelList = labelList

        # perform clustering
        bov_descriptor_stack = self.bov_helper.formatND(self.descriptor_list)
        self.bov_helper.cluster(featuresCount, labelList)
        self.bov_helper.developVocabulary(n_images = self.trainImageCount, descriptor_list=self.descriptor_list, labelList = labelList)

        # show vocabulary trained
        self.bov_helper.standardize()
        #sys.exit(0)

        self.bov_helper.train(self.train_labels)
        self.bov_helper.plotHist()


    def recognize(self,test_img, test_image_path=None):

        """ 
        This method recognizes a single image 
        It can be utilized individually as well.


        """

        print("Reconociendo " + test_image_path)
        kp, des = self.im_helper.features(test_img)
        # print kp
        print des.shape

        # generate vocab for test image
        vocab = np.array( [[ 0 for i in range(self.no_clusters)]])
        # locate nearest clusters for each of 
        # the visual word (feature) present in the image
        
        # test_ret =<> return of kmeans nearest clusters for N features
        test_ret = self.bov_helper.kmeans_obj.predict(des)
        print "Prediccion de clases para cada descriptor"
        print test_ret

        # print vocab
        for each in test_ret:
            vocab[0][each] += 1

        # Scale the features
        vocab = self.bov_helper.scale.transform(vocab)
        print "Vocabulario normalizado"
        print vocab

        # predict the class of the image
        lb = self.bov_helper.clf.predict(vocab)
        print "Image belongs to class : ", self.name_dict[str(int(lb[0]))]

        neighbor = NearestNeighbors(n_neighbors = 10)
        neighbor.fit(self.bov_helper.mega_histogram)
        dist, result = neighbor.kneighbors(vocab)
        print "kNN:"
       # print(dist)
       # print(result[0])
        for i in result[0]:
            print("label: "+self.labelList[i])

        return lb



    def testModel(self):
        """ 
        This method is to test the trained classifier

        read all images from testing path 
        use BOVHelpers.predict() function to obtain classes of each image

        """

        self.testImages, self.testImageCount, nameList = self.file_helper.getFiles(self.test_path)

        predictions = []

        counter = 0
        for word, imlist in self.testImages.iteritems():
            print "processing " ,word
            for im in imlist:
                # print imlist[0].shape, imlist[1].shape
                print im.shape
                cl = self.recognize(im, nameList[counter])
                print cl
                predictions.append({
                    'image':im,
                    'class':cl,
                    'object_name':self.name_dict[str(int(cl[0]))]
                    })
            counter += 1

        num = 0
        #print predictions
        for each in predictions:
            # cv2.imshow(each['object_name'], each['image'])
            # cv2.waitKey()
            # cv2.destroyWindow(each['object_name'])
            # 
            plt.imshow(cv2.cvtColor(each['image'], cv2.COLOR_GRAY2RGB))
            #plt.title(each['object_name'])
            #plt.show()
            plt.title(each['object_name'])
            name = 'result_' + str(num) + '.png'
            plt.savefig(name)
            num = num + 1;

    def cluster(self):
        print("Clustering con DBSCAN")
        #DIFICIL
        mega_histogram = self.bov_helper.mega_histogram
        #print(mega_histogram)

        #db = DBSCAN(eps=5, min_samples=3).fit(mega_histogram)
        #core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        #core_samples_mask[db.core_sample_indices_] = True
        #labels = db.labels_
        #last = 0
        #count = 0
        #print labels
        ##for  k in featuresCount:
        ##    print('Etiquetas(' +str(count)+') : ' + str(k) + ' Label: ' + labelList[count])
        ##    new = labels[last:last+k-1]
        ##    new = new[new != -1]
        ##    print(new)
        ##    moda = stats.mode(new)
        ##    print('Moda: ')
        ##    print(moda[0])
        ##    hist = np.histogram(new)
        ##    print(hist)
        ##    last = last+k
        ##    count += 1

        #n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        #n_noise_ = list(labels).count(-1)
        #print('Estimated number of clusters: %d' % n_clusters_)
        #print('Estimated number of noise points: %d' % n_noise_)

        print('PCA::')
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(mega_histogram)
        principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
        finalDf = pd.concat([principalDf, pd.Series(self.labelList)], axis = 1)
        print(finalDf)
        db = DBSCAN(eps=1, min_samples=2).fit(principalComponents)
        #db = DBSCAN(eps=1.2, min_samples=2).fit(principalComponents) #6 clusters mas feo
        #db = DBSCAN(eps=1, min_samples=2).fit(principalComponents) #6 clusters
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        last = 0
        print labels
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
# Black removed and is used for noise instead.
        unique_labels = set(labels)
        contador = 0
        for i in labels:
            if(i == -1):
                print("label: None")
            else:
                print("i: " +str(contador) +"label: "+str(i))
            contador += 1
        colors = [plt.cm.Spectral(each)
                for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = [0, 0, 0, 1]
            class_member_mask = (labels == k)

            xy = principalComponents[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)

            xy = principalComponents[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

            plt.title('Estimated number of clusters: %d' % n_clusters_)
            plt.savefig('pepe.png')


    def print_vars(self):
        pass


if __name__ == '__main__':

    # parse cmd args
    parser = argparse.ArgumentParser(
            description=" Bag of visual words example"
        )
    parser.add_argument('--train_path', action="store", dest="train_path", required=True)
    parser.add_argument('--test_path', action="store", dest="test_path", required=True)

    args =  vars(parser.parse_args())
    print args

    bov = BOV(no_clusters=20)

    # set training paths
    bov.train_path = args['train_path'] 
    # set testing paths
    bov.test_path = args['test_path'] 
    # train the model
    bov.trainModel()
    # test model
    #bov.testModel()
    bov.cluster()
