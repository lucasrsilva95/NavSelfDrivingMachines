import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

class Line:
    def __init__(self, coords, x_center, avgParameters, avgRoadSize, img):
      coords = np.reshape(coords,4)
      # y2 > y1
      if coords[1] > coords[3]:
          self.coords = (coords[2],coords[3],coords[0],coords[1])
      else:
          self.coords = coords
      self.x_center = x_center
      self.avgParameters = avgParameters
      self.avgRoadSize = avgRoadSize
      self.img = img

      self.parameters, self.size = self.getInitAttributes()
      self.distCenter = self.calcDistRoadCenter(x_center)
      self.projCoords = self.projectToBase()
      self.lineType = self.classification()
      self.percDistBase = self.coords[3]/self.img.shape[0]
      self.score = self.calcScore(avgParameters, avgRoadSize)
    
    def getInitAttributes(self):
        x1,y1,x2,y2 = np.reshape(self.coords,4)
        if x1 != x2:
            a = (y2-y1)/(x2-x1)
            b = y1 - a*x1
            angRad = np.arctan(a)
            pars = (angRad,b)
        else:
            pars = (1.5708, -10000) 
        size = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        return pars, size

    def calcDistRoadCenter(self, x_center):
        height = self.img.shape[0]
        x1, y1, x2, y2 = np.reshape(self.coords,4)
        x = x2 - ((height - y2)/(y1 - y2))*(x2 - x1)
        

        return abs(x_center - x)
    
    def classification(self):
        if self.projCoords[2] < self.x_center:
            return "Left"
        else:
            return "Right"

    def pointInLine(self, x = None, y = None):
        ang, b = self.parameters
        a = np.tan(ang)
        if x is not None:
            y = int(a * x + b)
            return y
        elif y is not None:
            x = int((y - b)/a)
            return x

    def projectToBase(self, newLine=False):
      height = self.img.shape[0]
      x1, y1, x2, y2 = np.reshape(self.coords,4)
      new_x2 = int(x2 - ((height - y2)/(y1 - y2))*(x2 - x1))
      ang = self.parameters[0]
      if x1 == x2:
        new_x1 = new_x2
      else:
        new_x1 = int(new_x2 - (abs(height - height/2)/np.tan(ang)))
      coords = (new_x1, int(height/2), new_x2, height)
      if not newLine:
        return coords
      else:
        projLine = Line(coords, self.x_center, self.avgParameters, self.avgRoadSize, self.img)
        return projLine

    def draw(self, img = None, show=False, color=(0,255,255), thickness=8):
        if img is None:
            img = self.img
        imgWithLine = img.copy()
        cv.line(imgWithLine, 
            (self.coords[0],self.coords[1]), (self.coords[2], self.coords[3]), 
            color = color,
            thickness = thickness)

        # imgWithLine = cv.addWeighted(img, 0.6, imgWithLine, 1, 1)
        
        if show:
            plt.imshow(cv.cvtColor(imgWithLine, cv.COLOR_BGR2RGB), cmap='gray'), plt.show()
        return imgWithLine
    
    def drawProjLine(self, img, show=False, color=(0,255,255), thickness=8):
        imgWithLine = self.projectToBase(newLine=True).draw(img = img, show=show, color=color, thickness=thickness)
        return imgWithLine

    def calcScore(self, avgParameters, avgRoadSize):
        y_error = 1000
        if self.lineType == "Left":
            ang_error = abs(avgParameters[0][0] - self.parameters[0])
            if self.parameters[1] != None and avgParameters[0][1] != None:
                y_error = abs(avgParameters[0][1] - self.parameters[1])
        else:
            ang_error = abs(avgParameters[1][0] - self.parameters[0])
            if self.parameters[1] != None:
                y_error = abs(avgParameters[1][1] - self.parameters[1])
        dist_error = abs(avgRoadSize/2 - self.distCenter)
        self.score = self.size + self.percDistBase*100 - y_error/10 - ang_error*10 - dist_error/5
        return self.score

    def status(self):
        print(f"Coords: {self.coords}")
        print(f"Parameters: {self.parameters}")
        print(f"Size: {self.size}")
        print(f"Dist Center: {self.distCenter:.2f}")
        print(f"Dist Base: {self.percDistBase*100:.1f}%")
        print(f"Score: {self.score:.1f}")
