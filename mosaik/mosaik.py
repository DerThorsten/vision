import numpy
import vigra
from numpy import random,argsort,sqrt
from pylab import plot,show
import glob
import os


def knn_search(x, D, K):
	""" find K nearest neighbours of data among D """
	ndata = D.shape[0]
	
	X=numpy.array([x,]*ndata)
	
	# euclidean distances from the other points
	sqd = numpy.sqrt(((D - X)**2).sum(axis=1))
	idx = numpy.argsort(sqd) # sorting
	# return the indexes of K nearest neighbours
	return (idx[:K],sqd[idx[:K]])
 
 
#import opengm
inputFolder='/home/tbeier/Desktop/img'
imagePath='/home/tbeier/Desktop/img/302003.jpg'
outputImage='/home/tbeier/Desktop/mosaik4.png'
labels=5
labelChangeRuns=10
uniqueRadius=2
patchSize=[30,30]


numPatchesX=80

mixing=0.3


images=[]
os.chdir(inputFolder)
for imgPath in glob.glob("*.jpg"):
    img=vigra.impex.readImage(imgPath)
    img=vigra.sampling.resizeImageSplineInterpolation(img,patchSize)
    images.append(img)
    print imgPath

image=vigra.impex.readImage(imagePath)
numPatches=[numPatchesX,int(numPatchesX*(float(image.shape[1])/float(image.shape[0]))+0.5)]
image=vigra.sampling.resizeImageSplineInterpolation(image,numPatches)

labelMap=numpy.ones([numPatches[0],numPatches[1],2],dtype=numpy.uint32)
finalImageRaw=vigra.sampling.resizeImageSplineInterpolation(image,[numPatches[0]*patchSize[0],numPatches[1]*patchSize[1]])
finalImage=finalImageRaw.copy()

print "numPatches: ",numPatches
print "imageSize: ",image.shape

means=numpy.ones([len(images),3])
candidatesAndDist=[]

dimX=image.shape[0]
dimY=image.shape[1]
numPixels=dimX*dimY


localCandidateLabels=[]

print "compute means of image database"
for i in range(len(images)):
   for c in range(3):
      means[i,c]=numpy.mean(images[i][:,:,c])
		  
print "find KNN for each pixel"
for y in range(image.shape[1]):
	if y%50==0 :print y
	for x in range(image.shape[0]):
	   pixelValue=image[x,y,:]
	   localCandidateLabels.append(knn_search(pixelValue,means,labels))
	   labelMap[x,y,0]=localCandidateLabels[x+y*image.shape[0]][0][0]
	   newLabel=numpy.random.randint(0,labels)
	   newLabel=localCandidateLabels[x+y*image.shape[0]][0][newLabel]	
	   labelMap[x,y,0]=newLabel
	   labelMap[x,y,1]=0
	   finalImage[  x*patchSize[0]:(x+1)*patchSize[0],y*patchSize[1]:(y+1)*patchSize[1],: ]=images[localCandidateLabels[-1][0][0]][:,:,:]

print "remove bad labeling"
numChanges=0
for y in range(image.shape[1]):
	if y%50==0 :print y
	for x in range(image.shape[0]):
		label=labelMap[x,y,0]
		change=False
		if(x+1<image.shape[0]):
			if(labelMap[x+1,y,0]==label):
				change=True
		if(y+1<image.shape[1]):
			if(labelMap[x,y+1,0]==label):
				change=True	
		if(x+1<image.shape[0] and y+1<image.shape[1]):
			if(labelMap[x+1,y+1,0]==label):
				change=True
	if(change==True):
		numChanges+=1
		newLabel=label
		while(	newLabel ==label):
			newLabel=numpy.random.randint(0,labels)
			newLabel=localCandidateLabels[x+y*image.shape[0]][0][newLabel]	
	labelMap[x,y,0]=newLabel
print "numberOfChanges", numChanges	

print "write final image"
for y in range(image.shape[1]):
	if y%50==0 :print y
	for x in range(image.shape[0]):
	   finalImage[  x*patchSize[0]:(x+1)*patchSize[0],y*patchSize[1]:(y+1)*patchSize[1],: ]=images[labelMap[x,y,0]][:,:,:]	
	   
finalImage[:,:,:]=finalImage[:,:,:]*(1.0-mixing) + mixing*finalImageRaw[:,:,:]							
vigra.impex.writeImage(finalImage,outputImage)
