import numpy
import glob
import os
import sys


def loadSegGtFile(segfile):
    f = open(segfile, 'r')
    lines= f.readlines()
    i=0
    start=0
    for line in lines:
        if line.startswith("width"):
            width=[int(s) for s in line.split() if s.isdigit()][0]
            #print "width ", width
        if line.startswith("height"):
            height=[int(s) for s in line.split() if s.isdigit()][0]
            #print "height ", height    

        
        if line.startswith("data"):
            start=i+1
            break
        i+=1
    seg=numpy.ones([width,height],dtype=numpy.uint32)
    #seg[:,:]=-1
    for line in lines[start:len(lines)]:
        #print line
        [label,row,cstart,cend]=[int(s) for s in line.split() if s.isdigit()]
        assert (cend +1 <= width)
        assert (row <= height)
        seg[cstart:cend+1,row]=label
    return seg 



# input folders
humanFolder      = "/home/tbeier/Desktop/BSDS300/human/color/"
imageFolderTest  = "/home/tbeier/Desktop/BSDS300/images/test/"
imageFolderTrain = "/home/tbeier/Desktop/BSDS300/images/train/"

# combine images of test and training
isTrainingImage={}
os.chdir(imageFolderTrain)    
for image in glob.glob("*.jpg"):
    #print image
    isTrainingImage.update({image:True})
os.chdir(imageFolderTest)
for image in glob.glob("*.jpg"):
    isTrainingImage.update({image:False})
#print "#images: ",len(isTrainingImage)
# dict where image path is key and list of gt files is the value
imagesGt={}
for root, dirs, files in os.walk(humanFolder):
    for userDir in dirs:
        fullUserDir=humanFolder+userDir
        os.chdir(fullUserDir)
        for gtSeg in glob.glob("*.seg"):
            fullGtPath=fullUserDir+"/"+gtSeg
            name=gtSeg.rsplit( ".", 1 )[ 0 ]+".jpg"
            if(isTrainingImage[name]): fullImgPath=imageFolderTrain+name
            else :                      fullImgPath=imageFolderTest+name
            if(imagesGt.has_key(fullImgPath)):
                imagesGt[fullImgPath].append(fullGtPath)
            else:
                imagesGt.update({fullImgPath:[fullGtPath]})

del isTrainingImage

for key in imagesGt.keys():
    print key ,"  gt segmentations: " ,len(imagesGt[key])








