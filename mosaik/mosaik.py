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
imagePath='/home/tbeier/Desktop/img/258089.jpg'
outputImage='/home/tbeier/Desktop/a3.png'
labels=50
patchSize=[35,35]
numPatchesX=160
mixing=0.001

#noise trick
sigma=10


images=[]
os.chdir(inputFolder)
i=0
for imgPath in glob.glob("*.jpg"):
    img=vigra.impex.readImage(imgPath)
    img=vigra.sampling.resizeImageSplineInterpolation(img,patchSize)
    images.append(img)
    i+=1
    if(i%10==0):
		print imgPath
    

image=vigra.impex.readImage(imagePath)
numPatches=[numPatchesX,int(numPatchesX*(float(image.shape[1])/float(image.shape[0]))+0.5)]
image=vigra.sampling.resizeImageSplineInterpolation(image,numPatches)
dimX=image.shape[0]
dimY=image.shape[1]
numPixels=dimX*dimY

noiseImage=image.copy()
noise=sigma * numpy.random.randn(numPixels*3)
noise=noise.reshape([dimX,dimY,3])
noiseImage+=noise
for c in range(3):
	noiseImage[numpy.where(noiseImage>255)]=255
	noiseImage[numpy.where(noiseImage<0)]=0	


labelMap=numpy.ones([numPatches[0],numPatches[1],2],dtype=numpy.uint32)
finalImageRaw=vigra.sampling.resizeImageSplineInterpolation(image,[numPatches[0]*patchSize[0],numPatches[1]*patchSize[1]])
finalImage=finalImageRaw.copy()

print "numPatches: ",numPatches
print "imageSize: ",image.shape

means=numpy.ones([len(images),3])
candidatesAndDist=[]




localCandidateLabels=[]

print "compute means of image database"
for i in range(len(images)):
   for c in range(3):
      means[i,c]=numpy.mean(images[i][:,:,c])
		  
print "find KNN for each pixel"
for y in range(image.shape[1]):
	if y%50==0 :
        print y
    for x in range(image.shape[0]):
       pixelValue=noiseImage[x,y,:]
       localCandidateLabels.append(knn_search(pixelValue,means,labels))
       labelMap[x,y,0]=localCandidateLabels[x+y*image.shape[0]][0][0]
       newLabel=numpy.random.randint(0,labels)
       newLabel=localCandidateLabels[x+y*image.shape[0]][0][newLabel]	
       labelMap[x,y,0]=newLabel
       labelMap[x,y,1]=0

print "write final image"
for y in range(image.shape[1]):
	if y%50==0 :print y
	for x in range(image.shape[0]):
	   finalImage[  x*patchSize[0]:(x+1)*patchSize[0],y*patchSize[1]:(y+1)*patchSize[1],: ]=images[labelMap[x,y,0]][:,:,:]	
	   
finalImage[:,:,:]=finalImage[:,:,:]*(1.0-mixing) + mixing*finalImageRaw[:,:,:]							
vigra.impex.writeImage(finalImage,outputImage)
