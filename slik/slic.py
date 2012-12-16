import numpy
import vigra
#timing
import time
#dispaly image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# make a region graph in pure python ,the code fore the region graph is from
# http://peekaboo-vision.blogspot.de/2011/08/region-connectivity-graphs-in-python.html
def make_graph(grid):
    # get unique labels
    vertices = numpy.unique(grid)
 
    # map unique labels to [1,...,num_labels]
    reverse_dict = dict(zip(vertices,numpy.arange(len(vertices))))
    grid = numpy.array([reverse_dict[x] for x in grid.flat]).reshape(grid.shape)
   
    # create edges
    down = numpy.c_[grid[:-1, :].ravel(), grid[1:, :].ravel()]
    right = numpy.c_[grid[:, :-1].ravel(), grid[:, 1:].ravel()]
    all_edges = numpy.vstack([right, down])
    all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1], :]
    all_edges = numpy.sort(all_edges,axis=1)
    num_vertices = len(vertices)
    edge_hash = all_edges[:,0] + num_vertices * all_edges[:, 1]
    # find unique connections
    edges = numpy.unique(edge_hash)
    # undo hashing
    edges = [[vertices[x%num_vertices],
              vertices[x/num_vertices]] for x in edges]
 
    return vertices, edges , reverse_dict

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
        breaked=False
        for i in range(iterations):
            #print "updateAssignments"
            numUpdates=self.updateAssignments()
            print "num updates ",numUpdates
            if numUpdates==0:
                breaked=True
                break
            self.updateCenters()
        if(breaked==False):
            self.updateAssignments()     
            

    def computeDistance(self,center,pixels):
        distance=numpy.ones(pixels.shape,dtype=numpy.float32)
        for c in xrange(5):
            distance[:,:,c]=center[c]
        ##################    
        #compute distance!
        ##################
        distance-=pixels
        distC=numpy.sum(distance[:,:,0:3]**2,2);
        distS=numpy.sum(distance[:,:,3:5]**2,2);
        distance=numpy.sqrt( distC  + distS/(float(self.centerDists[0])**2)*(self.weight**2) )
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
    def relabel(self,threshold):
        numRelabeling=1
        while(numRelabeling!=0):
            numRelabeling=self.relabelIt(self.labels,threshold)
            print "num relabelings=",numRelabeling
            self.labels[:,:]=self.finalLabels

    def relabelIt(self,labels,threshold):
        numRelabeling=0
        self.labelImage=vigra.analysis.labelImage(labels.astype(numpy.float32))
        self.labelImage=self.labelImage.astype(numpy.uint32)
        self.labelImage-=1
        self.finalLabels=self.labelImage.copy()
        #find member
        [vertices,edges,revdict]=make_graph(self.labelImage)
        #inverse adj.
        regionsEdges=dict((k, []) for k in vertices)
        edgeIndex=0
        for edge in edges:
            regionsEdges[edge[0]].append(edgeIndex)
            regionsEdges[edge[1]].append(edgeIndex)
            edgeIndex+=1
        centerSizes = dict()
        coords={}
        coords=dict((k, [[],[]]) for k in vertices)
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):    
                coords[self.labelImage[x,y]][0].append(x)
                coords[self.labelImage[x,y]][1].append(y)
        mergedRegions=set()        
        for centerIndex in vertices:
            size=len(coords[centerIndex][0])
            #print size
            if(size<threshold and (centerIndex not in mergedRegions) ):
                numRelabeling+=1
                redges=regionsEdges[centerIndex]
                maxSize=0;
                maxIndex=-1;
                for edge in redges:
                    assert len(edges[edge])==2
                    r=edges[edge][0]
                    r2=edges[edge][1]
                    if r==centerIndex:
                        r=r2
                    sizeReg=len(coords[r][0])
                    if sizeReg>maxSize:
                        maxSize=sizeReg
                        maxIndex=r
                mergedRegions.add(r)
                mergedRegions.add(centerIndex)
                self.finalLabels[coords[centerIndex]]=r             

        self.numSuperPixel=len(numpy.unique(self.finalLabels))
        return numRelabeling

                    
img=vigra.impex.readImage('/home/tbeier/Desktop/BSDS300/images/train/8143.jpg')
gradmag=vigra.filters.gaussianGradientMagnitude(img,1.5)

k=1000
slic=Slic(img,k,40,gradmag)
slic.iterate(4)
slic.relabel(6)
print "number of superpixels ",slic.numSuperPixel
labels=slic.finalLabels
# plot labels
cmap = matplotlib.colors.ListedColormap ( numpy.random.rand ( int(k*4),3))
#plt.imshow(slic.labelImage,cmap)
plt.imshow(slic.finalLabels,cmap)
plt.show()    