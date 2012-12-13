import numpy
import vigra
#dispaly image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Slic():
    def __init__(self,img,k,weight,gradmag,rMinSearch=2):
        self.rMinSearch=rMinSearch
        self.gradmag=numpy.squeeze(gradmag)
        self.weight=weight
        self.labImg=vigra.colors.transform_RGB2Lab(img)
        self.k=k
        self.shape=img.shape
        self.centers=[]
        self.centerDists=None
        self.centers=None
        self.centerMeans=None
        self.numSuperPixel=None
        #create array wich has x,y,5 shape with coordinates additional "colorchannel"
        self.data=numpy.ones(self.shape[0:2]+(5,),dtype=numpy.float32)
        self.seedMap=numpy.zeros(self.shape[0:2],dtype=numpy.int)
        for c in range(3):
            self.data[:,:,c]=self.labImg[:,:,c]
        self.data[:,:,3]=numpy.transpose(numpy.array([numpy.arange(0,self.shape[0],dtype=numpy.float32),]*self.shape[1],dtype=numpy.float32))
        self.data[:,:,4]=numpy.array([numpy.arange(0,self.shape[1],dtype=numpy.float32),]*self.shape[0],dtype=numpy.float32)

        self.findInitialCenters()
        self.labels=-1.0*numpy.ones(self.shape[0:2],dtype=numpy.int64)
        self.distance=numpy.inf*numpy.ones(self.shape[0:2],dtype=numpy.float32)


    def findInitialCenters(self):
        shape=self.shape
        # a*shape[0]*a*shape[1]=k
        a=numpy.sqrt(float(self.k)/float(shape[0]*shape[1]))
        #nCenters=(int(a*shape[0]),int(a*shape[1]))
        #print "nCenters ",nCenters
        centerDist=int(numpy.sqrt(   (float(shape[0])*float(shape[1]))/float(self.k))+0.5)
        self.centerDists=(
            centerDist,
            centerDist
        )
        #print "self.centerDists ",self.centerDists
        self.centers=[]
        k=0
        r=self.rMinSearch
        for x in range(0,shape[0],self.centerDists[0]):
            for y in range(0,shape[1],self.centerDists[1]):
                assert self.data[x,y,3]==x and self.data[x,y,4]==y

                startX=max(0,x-r)
                startY=max(0,y-r)
                endX=min(self.shape[0],x+r+1)
                endY=min(self.shape[1],y+r+1)
                minVal=numpy.inf
                found=False
                bx=None
                by=None
                for xx in range(startX,endX):
                    for yy in range(startY,endY):
                        if self.gradmag[xx,yy]<minVal and self.seedMap[xx,yy]==0:
                            bx=xx
                            by=yy
                            Found=True
                assert Found 
                self.seedMap[bx,by]=1    
                self.centers.append((bx,by))    
                #move to minimum gradient
                k+=1
        self.k=k
        self.centerMeans=numpy.ones([self.k,5],dtype=numpy.float32)
        centerIndex=0
        for center in self.centers:
            assert len(center)==2
            cX=center[0]
            cY=center[1]
            self.centerMeans[centerIndex,3]=cX
            self.centerMeans[centerIndex,4]=cY
            for c in range(3):
                self.centerMeans[centerIndex,c]=self.data[cX,cY,c]
            centerIndex+=1    

    def iterate(self,iterations=10):
        for i in range(iterations):
            #print "updateAssignments"
            numUpdates=self.updateAssignments()
            print "num updates ",numUpdates
            if numUpdates==0:
                break
            self.updateCenters() 
    

    def computeDistance(self,center,pixels):
        distance=numpy.ones(pixels.shape,dtype=numpy.float32)
        for c in xrange(5):
            distance[:,:,c]=center[c]
        ##################    
        #compute distance!
        ##################
        distance-=pixels
        distance[:,:,3:5]*=self.weight
        distance=numpy.sqrt(numpy.sum(distance**2,2));
        return distance
    def updateAssignments(self):
        centerIndex=0
        numChanges=0
        cDistX=self.centerDists[0]
        cDistY=self.centerDists[1]
        for center in self.centers:
            #print "centerIndex=",centerIndex
            assert len(center)==2
            cX=center[0]
            cY=center[1]
            startX=max(0,cX-cDistX)
            startY=max(0,cY-cDistY)
            endX=min(self.shape[0],cX+cDistX)
            endY=min(self.shape[1],cY+cDistY)
            #get subdata
            subData=self.data[startX:endX,startY:endY,:]
            subDistance=self.distance[startX:endX,startY:endY]
            subLabels=self.labels[startX:endX,startY:endY]
            distance=self.computeDistance(self.centerMeans[centerIndex,:],subData)
            #distance=numpy.sqrt(numpy.sum(distance,2))
            #distance=numpy.sqrt(distance)
            updateLabelSubCoordinates=numpy.where(distance<subDistance)
            numChanges+=len(updateLabelSubCoordinates[0])
            #update labels
            subLabels[updateLabelSubCoordinates]=centerIndex
            subDistance[updateLabelSubCoordinates]=distance[updateLabelSubCoordinates]
            #increment center index
            centerIndex+=1
        return numChanges    

    def updateCenters(self):
        cDistX=self.centerDists[0]
        cDistY=self.centerDists[1]
        centerIndex=0
        for center in self.centers:
            #print "centerIndex=",centerIndex
            assert len(center)==2
            cX=center[0]
            cY=center[1]
            startX=max(0,cX-cDistX)
            startY=max(0,cY-cDistY)
            endX=min(self.shape[0],cX+cDistX)
            endY=min(self.shape[1],cY+cDistY)
            subData=self.data[startX:endX,startY:endY,:]
            subDistance=self.distance[startX:endX,startY:endY]
            subLabels=self.labels[startX:endX,startY:endY]
            #find own members
            ownMembersCoordinates=numpy.where(subLabels==centerIndex)
            if(len(ownMembersCoordinates[0])>0):
                self.centerMeans[centerIndex,:]=numpy.mean(subData[ownMembersCoordinates],0)
            centerIndex+=1
    def relabel(self):
        
        self.labels=self.labels.astype(numpy.uint32)
        print "shape",self.labels.shape
        print "dtype",self.labels.dtype
        self.labelImage=vigra.analysis.labelImage(self.labels.astype(numpy.float32))
        self.numSuperPixel=numpy.max(self.labelImage)

                    
img=vigra.impex.readImage('/home/tbeier/Desktop/img/12003.jpg')
gradmag=vigra.filters.gaussianGradientMagnitude(img,1.5)

k=1000
slic=Slic(img,k,2,gradmag)
slic.iterate(15)
slic.relabel()
print "number of superpixels ",slic.numSuperPixel
labels=slic.labelImage
# plot labels
cmap = matplotlib.colors.ListedColormap ( numpy.random.rand ( int(k*1.2),3))
plt.imshow(labels,cmap)
plt.show()    