import random
import pylab as plt
import numpy as np

from matplotlib.widgets import Button
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# %matplotlib notebook

def gauss_2d(mu, sigma, size=10):
  x, y = np.meshgrid(np.linspace(-1,1,size), np.linspace(-1,1,size))
  d    = np.sqrt(x*x+y*y)
  g    = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
  return g

def getContourCase(top,left,thres,cells):
    # YOUR CODE HERE
    # raise NotImplementedError
    h, w = cells.shape
    if (top >= h-1) or (left >= w-1):
        return 0 
    cell = cells[top:top+2, left:left+2] >= thres
    cell = cell.astype(int)
    val = 2**3*cell[0, 0] + 2**2*cell[0, 1] + 2*cell[1, 1] + 1*cell[1, 0]
    return val

def disambiguateSaddle(top,left,thres,cells):
    # YOUR CODE HERE
    # raise NotImplementedError
    h, w = cells.shape
    if (top >= h-1) or (left >= w-1):
        return False  
    cell = cells[top:top+2, left:left+2]
    avg = np.mean(cell)
    return avg >= thres 

def interpolate(v1,v2,t):
    # YOUR CODE HERE
    # raise NotImplementedError
    if v1 == v2:
        return v1 
    return (t-v1) / (v2-v1)

def getCellSegments(top,left,thres,cells):
    # YOUR CODE HERE
    h, w = cells.shape
    if (top >= h-1) or (left>=w-1):
        return []
    case = getContourCase(top,left,thres,cells)
    cell = cells[top:top+2, left:left+2]
    inter_12 = interpolate(cell[0, 0], cell[0, 1], thres)
    inter_23 = interpolate(cell[0, 1], cell[1, 1], thres)
    inter_34 = interpolate(cell[1, 0], cell[1, 1], thres)
    inter_41 = interpolate(cell[0, 0], cell[1, 0], thres)
    if case in [0, 15]:
        return []
    if case == 14:
        p1_x = left
        p2_y = top+1
        p1_y = top + inter_41
        p2_x = left + inter_34
    if case == 13:
        p1_x = left + inter_34 
        p1_y = top + 1 
        p2_x = left + 1 
        p2_y = top + inter_23
    if case == 11: 
        p1_x = left+inter_12 
        p1_y = top
        p2_x = left+1 
        p2_y = top+inter_23
    if case == 7: 
        p1_x = left
        p1_y = top + inter_41
        p2_x = left+inter_12
        p2_y = top
    if case == 1:
        p1_x = left
        p1_y = top + inter_41 
        p2_x = left+inter_34
        p2_y = top +1 
    if case == 2:
        p1_x = left+inter_34
        p1_y = top+1
        p2_x = left+1
        p2_y = top+inter_23
    if case == 4:
        p1_x = left+inter_12
        p1_y = top
        p2_x = left+1 
        p2_y = top+inter_23 
    if case == 8:
        p1_x = left 
        p1_y = top + inter_41
        p2_x = left+inter_12
        p2_y = top 
    if case == 12:
        p1_x = left 
        p2_x = left+1 
        p1_y = top+inter_41 
        p2_y = top+inter_23
    if case == 9:
        p1_x = left+inter_12
        p1_y = top 
        p2_x = left+inter_34
        p2_y = top+1 
    if case == 3:
        p1_x = left 
        p1_y = top + inter_41 
        p2_x = left+1 
        p2_y = top+inter_23 
    if case == 6:
        p1_x = left+inter_12
        p1_y = top 
        p2_x = left+inter_34 
        p2_y = top +1 
    if case not in [10, 5]: 
        return [[(p1_x, p1_y), (p2_x, p2_y)]]
    else:
        p1_x = left
        p1_y = left+inter_41
        p2_x = left+inter_12
        p2_y = top
        p3_x = left+inter_34
        p3_y = top + 1 
        p4_x = left+1 
        p4_y = top+inter_23
        saddle = disambiguateSaddle(top, left, thres, cells) < thres
        if saddle:
            if case == 10:
                return [[(p1_x, p1_y), (p2_x, p2_y)], [(p3_x, p3_y), (p4_x, p4_y)]]
            if case == 5:
                return [[(p2_x, p2_y), (p4_x, p4_y)], [(p1_x, p1_y), (p3_x, p3_y)]]
        else:
            if case == 10:
                return [[(p1_x, p1_y), (p3_x, p3_y)], [(p2_x, p2_y), (p4_x, p4_y)]]
            if case == 5:
                return [[(p1_x, p1_y), (p2_x, p2_y)], [(p3_x, p3_y), (p4_x, p4_y)]]

def getContourSegments(thres,cells):
    # YOUR CODE HERE
    # raise NotImplementedError
    h, w = cells.shape
    bounds = []
    tops, lefts = np.meshgrid(np.arange(h-1), np.arange(w-1))
    for top, left in zip(np.ravel(tops), np.ravel(lefts)):
        res = getCellSegments(top, left, thres, cells)
        bounds += res
    return bounds

def colorMapPlasma():
    idx = np.array([0, 0.142857143, 0.285714286, 0.428571429, 0.571428571, 0.714285714, 0.857142857, 1])
    r = np.array([47, 98, 146, 186, 216, 238, 246, 228])/255
    g = np.array([0, 0, 0, 47, 91, 137, 189, 250]) / 255 
    b = np.array([135, 164, 166, 138, 105, 73, 39, 21]) / 255
    cdict = {"red": np.stack([idx, r, r]).T,
             "green": np.stack([idx, g, g]).T,
             "blue": np.stack([idx, b, b]).T}
    return cdict



class March(object):
    def __init__(self,res=32,thres=0.5,size=320):

        #Initialize variables
        self.res      = res                      #Number of grid cells per axis
        self.thres    = thres                    #Threshold for binarization
        self.size     = size                     #Size of image (in pixels)
        self.contours = 0                        #Whether we're showing contours (0 = off,  1 = normal, 2 = interpolated)
        self.cmap     = self.colorMapGrayscale() #Default grayscale color map
        self.cmapi    = 0                        #Index of color map (0 = gray, 1 = plasma, 2 = custom)

        #Hardcode some cells to start with to test all cases
        self.cells    = gauss_2d(0.5,0.4,self.res)

        #Compute other useful variables from grid size
        self.step     = self.size // self.res #Spacing between grid lines (in pixels)

        #Set up axes
        self.fig, self.axes = plt.subplots()
        self.axes.set_aspect('equal')
        plt.subplots_adjust(bottom=0.2)

        #Set up buttons
        self.btog = Button(plt.axes([0.61, 0.05, 0.2, 0.075]), 'No Contours')
        self.btog.on_clicked(self.toggle_contours)
        self.bmap = Button(plt.axes([0.41, 0.05, 0.2, 0.075]), 'Grayscale')
        self.bmap.on_clicked(self.toggle_colormap)

        #Perform initial drawing
        self.redraw()

    def show(self):
        plt.show()

    def update(self):
        self.fig.canvas.draw()

    def toggle_contours(self,event):
        #Toggle whether we draw contours or not
        self.contours = (self.contours + 1) % 3
        self.redraw()

    def toggle_colormap(self,event):
        self.cmapi = (self.cmapi+1)%2
        if self.cmapi == 0:
          self.cmap = self.colorMapGrayscale()
          self.bmap.label.set_text("Grayscale")
        elif self.cmapi == 1:
          self.cmap = colorMapPlasma()
          self.bmap.label.set_text("Plasma")
        self.redraw()

    def redraw(self):
        # Regenerate a blank white canvas withou axis lines or tick marks
        self.axes.clear()
        self.axes.set_yticks([])
        self.axes.set_xticks([])
        self.axes.set_yticklabels([])
        self.axes.set_xticklabels([])

        #Invert y axis to match up with array ordering
        self.axes.invert_yaxis()

        #Draw the image from our img matrix
        self.drawImage()
        if self.contours == 0:
          for i in range(1,self.res): #Draw image grid
            self.axes.plot([0,self.size-1], [self.step*i,self.step*i], color='black', linestyle='-', linewidth=1)
            self.axes.plot([self.step*i,self.step*i], [0,self.size-1], color='black', linestyle='-', linewidth=1)
          self.btog.label.set_text('No Contours')
        else:  # Draw contours and contour grid
          for i in range(self.res): #Draw contour grid
            self.axes.plot([0,self.size-1], [self.step*(i+0.5),self.step*(i+0.5)], color='gray', linestyle='-', linewidth=1)
            self.axes.plot([self.step*(i+0.5),self.step*(i+0.5)], [0,self.size-1], color='gray', linestyle='-', linewidth=1)
          if self.contours == 1:
            self.btog.label.set_text('Rough Contours')
            self.drawTableLookupContours()
          else:
            self.btog.label.set_text('Interp. Contours')
            self.drawInterpolatedContours()

        #Update the underlying plot
        self.update()

    def colorMapGrayscale(self):
        cdict = {'red':   [[0, 0, 0],
                           [1, 1, 1]],
                 'green': [[0, 0, 0],
                           [1, 1, 1]],
                 'blue':  [[0, 0, 0],
                           [1, 1, 1]]}
        return cdict

    def drawImage(self):
        newcmp = LinearSegmentedColormap('testCmap', segmentdata=self.cmap, N=256)
        self.axes.imshow(gauss_2d(0.5,0.4,self.size),cmap=newcmp)

    def drawTableLookupContours(self):
        for y,row in enumerate(self.cells):
          for x,cell in enumerate(row):
            case = getContourCase(y,x,self.thres,self.cells)
            self.drawCellContourByCase(y,x,case)

    def drawInterpolatedContours(self):
        segments = getContourSegments(self.thres,self.cells)
        for s in segments:
          x1 = self.step*(0.5+s[0][0])
          x2 = self.step*(0.5+s[1][0])
          y1 = self.step*(0.5+s[0][1])
          y2 = self.step*(0.5+s[1][1])
          self.axes.plot([x1,x2], [y1,y2], color='green', linestyle='-', linewidth=1)

    def drawCellContourByCase(self,yrow,xcol,case):
        if case in [0,15]:
          return #Nothing to draw for empty cells, completely surrounded cells, or border cells

        #Handle saddle points
        if case in [5]:
          if disambiguateSaddle(yrow,xcol,self.thres,self.cells):
            self.drawCellContourByCase(yrow,xcol,2)
            self.drawCellContourByCase(yrow,xcol,7)
          else:
            self.drawCellContourByCase(yrow,xcol,11)
            self.drawCellContourByCase(yrow,xcol,14)
          return
        if case in [10]:
          if disambiguateSaddle(yrow,xcol,self.thres,self.cells):
            self.drawCellContourByCase(yrow,xcol,11)
            self.drawCellContourByCase(yrow,xcol,14)
          else:
            self.drawCellContourByCase(yrow,xcol,2)
            self.drawCellContourByCase(yrow,xcol,7)
          return

        #Compute coordinates based on case lookup table
        s    = self.step
        ymin = s*yrow + (0         if case in [4,6,7,8,9,11]   else s//2)
        ymax = s*yrow + (self.step if case in [1,2,6,9,13,14]  else s//2)
        xmin = s*xcol + (0         if case in [1,3,7,8,12,14]  else s//2)
        xmax = s*xcol + (self.step if case in [2,3,4,11,12,13] else s//2)
        if case in [2,7,8,13]: #Reverse direction for lines drawn up and right (i.e., x increases while y decreases)
          xmin,xmax = xmax,xmin

        #Contour lines should be drawn halfway between grid cells, so set an offset
        off = s//2
        #Smooth contours should have different color
        color = 'red' if self.contours == 1 else 'green'
        #Actually draw the contour lines
        self.axes.plot([xmin+off, xmax+off], [ymin+off, ymax+off], color=color, linestyle='-', linewidth=1)
        return