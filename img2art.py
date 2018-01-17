#! /usr/bin/env python

"""
Image2Path
@author: Vincent Politzer <https://github.com/vjapolitzer>

Dependencies: numpy, scipy, pillow, pypotrace, skimage
"""

import sys, os, getopt, math
import time
from os.path import basename, splitext
from copy import deepcopy
import random
from operator import itemgetter
import potrace
import numpy as np
import scipy
from scipy import spatial
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import skimage.io as skio
from skimage import img_as_float
from skimage.restoration import denoise_bilateral
from skimage import feature

def compPt(p0, p1):
    return p0[0] == p1[0] and p0[1] == p1[1]

def distance(p0, p1):
    """
    Distance between p0 and p1
    """
    return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)

def mapToRange(val, src, dst):
    """
    Map the given value from the range of src to the range of dst.
    """
    return ((val - src[0]) / (src[1] - src[0])) * (dst[1] - dst[0]) + dst[0]

def floatRange(start, stop, step):
    """
    Generate an iterator from start to stop, allowing for floats
    """
    i = start
    while (i < stop) if stop > start else (i > stop):
        yield i
        i += step

def distance(p0, p1):
    """
    Distance between p0 and p1
    """
    return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)

def diagonals(w, h):
    return [[[p - q, q]
             for q in range(max(p-w+1, 0), min(p+1, h))]
            for p in range(w + h - 1) ]

class ImgPathGenerator:
    def __init__(self, plotBounds):
        self.xMaxPlot = plotBounds[0]
        self.yMaxPlot = plotBounds[1]
        self.align = [0.0, 0.0]
        self.noNegative = False
        self.hasHome = False
        self.design = None
        self.drawSpeed = 20.0
        self.travelSpeed = 40.0
        self.toolWidth = None
        self.lowerTool = None
        self.raiseTool = None
        self.imageWidth = None
        self.imageHeight = None
        self.xRange = None
        self.yRange = None
        self.imagePath = None
        self.gcodePath = None
        self.patternPath = None
        self.image = None
        self.durationTSP = None

    def configure(self, drawSpeed, travelSpeed, toolWidth, durationTSP, hasHome):
        self.drawSpeed = drawSpeed
        self.travelSpeed = travelSpeed
        self.toolWidth = toolWidth
        self.durationTSP = durationTSP
        self.hasHome = hasHome

    def lowerCommand(self, command):
        self.lowerTool = command

    def raiseCommand(self, command):
        self.raiseTool = command

    def loadImage(self, path):
        self.imagePath = path
        self.image = Image.open(self.imagePath)
        self.image = ImageOps.flip(self.image)

    def setGCodePath(self, path):
        self.gcodePath = path

    def setImagePath(self, path):
        self.imagePath = path

    def setPatternPath(self, path):
        self.patternPath = path

    def noNegativeCoords(self):
        self.noNegative = True
        self.hasHome = True

    def __setDimensions(self, desiredWidth = None, desiredHeight = None, polar = False):
        aspectRatio = float(self.image.size[0]) / float(self.image.size[1])
        if ((desiredWidth is not None and desiredHeight is None)
            or (desiredWidth is not None and desiredHeight is not None
                and desiredWidth/desiredHeight < aspectRatio)):
            self.imageWidth = desiredWidth
            self.imageHeight = desiredWidth / aspectRatio
        else:
            self.imageWidth = desiredHeight * aspectRatio
            self.imageHeight = desiredHeight
        if polar:
            self.xRange = (-self.imageWidth/2.0, self.imageWidth/2.0)
            self.yRange = (-self.imageHeight/2.0, self.imageHeight/2.0)
        else:
            self.xRange = (0.0, self.imageWidth)
            self.yRange = (0.0, self.imageHeight)
        return (self.imageWidth <= self.xMaxPlot and self.imageHeight <= self.yMaxPlot)

    def __fit(self, width = None, height = None, polar = False):
        """
        Fit image into area bounded by width(mm) and height(mm),
        preserving aspect ratio. If only one dimension is
        provided, only said dimension is considered.
        Returns true if the calculated image dimensions fit
        within the plotting boundaries, else returns false
        """
        return self.__setDimensions(width, height, polar)

    def __fitSquare(self, width = None, height = None):
        dim = min(self.image.size[0], self.image.size[1])
        self.image = ImageOps.fit(self.image,(dim, dim))
        return self.__fit(width, height, polar = True)

    def __noFitError(self):
        print '\n' + '\033[91m' + '\nDesired dimensions do not fit within the plotting boundaries!' + '\033[0m'
        print 'Requested width: ' + str(self.imageWidth/10.0) + 'cm'
        print 'Requested height: ' + str(self.imageHeight/10.0) + 'cm'
        print 'Max width: ' + str((self.xMaxPlot)/10.0) + 'cm'
        print 'Max height: ' + str((self.yMaxPlot)/10.0) + 'cm'

    def __iTransOrigin(self, point):
        """
        Translate point to be centered about the image center
        """
        t = np.array([((self.imageWidth) / 2.0), ((self.imageHeight) / 2.0)])
        return point + t

    def __sierpinskiIndex(self, p):
        x, y = p[0], p[1]
        maxQualifier = (2**8) * max(self.image.size[0], self.image.size[1])
        loopIndex = maxQualifier
        spIndex = 0

        if x > y:
            spIndex += 1
            x = maxQualifier - x
            y = maxQualifier - y

        while loopIndex > 0:
            spIndex = spIndex + spIndex

            if x + y > maxQualifier:
                spIndex += 1
                temp = x
                x = maxQualifier - y
                y = temp

            x = x + x
            y = y + y

            spIndex = spIndex + spIndex

            if y > maxQualifier:
                spIndex += 1
                temp = x
                x = y - maxQualifier
                y = maxQualifier - temp

            loopIndex /= 2

        return spIndex

    def __initGcode(self):
        with open(self.gcodePath, 'w') as f:
            f.write('; Aether Art GCode\n')
            if self.imagePath is not None:
                f.write('; Original image: ' + basename(self.imagePath) + '\n')
            f.write('; Design: ' + self.design + '\n')
            f.write('; Width: ' + str(self.imageWidth/10.0) + 'cm\n')
            f.write('; Height: ' + str(self.imageHeight/10.0) + 'cm\n')
            f.write('; Tool width: ' + str(self.toolWidth) + 'mm\n')
            f.write('; Draw speed: ' + str(self.drawSpeed) + 'mm/s\n')
            f.write('; Travel speed: ' + str(self.travelSpeed) + 'mm/s\n')
            f.write(';\n; Begin GCode\n')
            f.write('M106 S255\nM400\nM340 P0 S600\nG4 P250\n')
            f.write('G21\nG90\n') # preparatory gcode
            initialPosition = 'G92 X' + str(self.align[0]) + ' Y' + str(self.align[1])
            if 'Z' in self.lowerTool:
                initialPosition = initialPosition + ' Z0.0\n'
            else:
                initialPosition = initialPosition + '\n'
            f.write(initialPosition)
            # if self.hasHome:
            #     f.write('G28')
            if self.design not in ('spiral'):
                f.write(self.raiseTool + '\n')

    def __endGcode(self):
        with open(self.gcodePath, 'a') as f:
            f.write(self.raiseTool + '\n')
            if self.hasHome:
                f.write('G28 Y\n')
                f.write('G28 X\n')
            else:
                f.write('G0 X' + format(self.xRange[0], '.8f') + ' Y' + format(self.yRange[1], '.8f') + ' F' + str(self.travelSpeed*60.0) +'\n')

    def __drawSingleLine(self, path):
        self.__initGcode()
        with open(self.gcodePath, 'a') as f:
            for i, point in enumerate(path):
                if i == 0:
                    if self.design not in ('spiral'):
                        f.write('G0 X' + format(point[0], '.8f') + ' Y' + format(point[1], '.8f') + ' F' + str(self.travelSpeed*60.0) +'\n')
                    f.write(self.lowerTool + '\n')
                else:
                    f.write('G1 X' + format(point[0], '.8f') + ' Y' + format(point[1], '.8f') + ' F' + str(self.drawSpeed*60.0) +'\n')
            f.write(self.raiseTool + '\n')
        self.__endGcode()

    def __drawSegments(self, path):
        self.__initGcode()
        with open(self.gcodePath, 'a') as f:
            prevPoint = [float('inf'), float('inf')]
            for segment in path:
                for i, point in enumerate(segment):
                    if i == 0:
                        if point[0] != prevPoint[0] or point[1] != prevPoint[1]:
                            f.write(self.raiseTool + '\n')
                            f.write('G0 X' + format(point[0], '.8f') + ' Y' + format(point[1], '.8f') + ' F' + str(self.travelSpeed*60.0) +'\n')
                            f.write(self.lowerTool + '\n')
                    else:
                        f.write('G1 X' + format(point[0], '.8f') + ' Y' + format(point[1], '.8f') + ' F' + str(self.drawSpeed*60.0) +'\n')
                        prevPoint = point
        self.__endGcode()

    def __drawShapes(self, shapes, includeDots = True):
        self.__initGcode()
        with open(self.gcodePath, 'a') as f:
            for s in shapes:
                if not includeDots and len(s) == 1:
                    continue
                prevPoint = [float('inf'), float('inf')]
                for i, point in enumerate(s):
                    if i == 0:
                        f.write(self.raiseTool + '\n')
                        f.write('G0 X' + format(point[0], '.8f') + ' Y' + format(point[1], '.8f') + ' F' + str(self.travelSpeed*60.0) +'\n')
                        f.write(self.lowerTool + '\n')
                    elif point[0] != prevPoint[0] or point[1] != prevPoint[1]:
                        f.write('G1 X' + format(point[0], '.8f') + ' Y' + format(point[1], '.8f') + ' F' + str(self.drawSpeed*60.0) +'\n')
                        prevPoint = point
            f.write(self.raiseTool + '\n')
        self.__endGcode()

    def __drawDots(self, dots):
        self.__initGcode()
        with open(self.gcodePath, 'a') as f:
            for point in dots:
                f.write('G0 X' + format(point[0], '.8f') + ' Y' + format(point[1], '.8f') + ' F' + str(self.travelSpeed*60.0) +'\n')
                f.write(self.lowerTool + '\n')
                f.write(self.raiseTool + '\n')
        self.__endGcode()

    def __generateSpiral(self, spacingFactor = 5.0):
        """
        Generate points along an Archimedes spiral, with amplitude
        modulated according to the corresponding pixel's value
        """
        self.image = self.image.convert('L')
        self.image = ImageOps.autocontrast(self.image)
        self.image = ImageEnhance.Sharpness(self.image).enhance(4.0)

        rMax = self.xRange[1]
        spacing = spacingFactor * self.toolWidth
        maxAmp = spacing / 2.0
        coils = rMax / spacing
        thetaMax = 2.0 * np.pi * coils
        rStep = rMax / thetaMax
        chord = self.toolWidth

        r = spacing
        theta = (2.0 * np.pi) + (chord / r)
        ziggy = 1.0

        if self.noNegative:
            path = [[self.imageWidth / 2.0, self.imageHeight / 2.0]]
            self.align = [(self.imageWidth/2.0)+maxAmp, (self.imageHeight/2.0)+maxAmp]
        else:
            path = [[0.0, 0.0]]

        while theta < thetaMax:
            x, y = r * np.cos(theta), r * np.sin(theta)
            xPx = int(mapToRange(x, self.xRange, (0, self.image.size[0]-1)))
            yPx = int(mapToRange(y, self.yRange, (0, self.image.size[1]-1)))
            amp = (maxAmp) * (float(255 - self.image.getpixel((xPx,yPx))) / 255.0)
            r += ziggy * amp
            # if r > spacing - maxAmp:
            if self.noNegative:
                path.append(self.__iTransOrigin([r * np.cos(theta)+maxAmp, r * np.sin(theta)+maxAmp]))
            else:
                path.append([r * np.cos(theta), r * np.sin(theta)])
            ziggy = -ziggy
            theta = theta + (chord / r)
            r = rStep * theta

        self.__drawSingleLine(path)

    def __generateZigZag(self, shift = False, spacingFactor = 4):
        """
        Generate points along horizontal lines, with amplitude
        modulated according to the corresponding pixel's value
        """
        self.image = self.image.convert('L')
        self.image = ImageOps.autocontrast(self.image)

        path = []
        spacing = float(spacingFactor) * self.toolWidth
        segment = self.toolWidth

        ziggy = 1.0
        reverse = False
        xRez = int(self.imageWidth / segment)
        yRez = int(self.imageHeight / segment)
        self.image = self.image.resize((xRez, yRez), Image.ANTIALIAS)

        for i in range(yRez-1, 0, -int(spacingFactor)):
            for j in range(0, xRez):
                xCoord = segment * (j if not reverse else (xRez-1) - j)
                if shift and not reverse:
                    xCoord += self.toolWidth / 2.0
                yCoord = segment * i
                amp = (spacing / 2.5) * ((255 - self.image.getpixel(((j if not reverse else (xRez-1)-j),i))) / 255.0)
                yCoord += ziggy * amp
                path.append([xCoord, yCoord])
                ziggy = -ziggy
            if xRez - 1 % 2 != 0:
                ziggy = -ziggy
            reverse = not reverse

        self.__drawSingleLine(path)

    def __generateRadiate(self, origin = 'TL', spacingFactor = 4.0):
        """
        Generate points along concentric curves radiating from
        a point, with amplitude modulated according to the
        corresponding pixel's value
        """
        self.image = self.image.convert('L')
        self.image = ImageOps.autocontrast(self.image)

        spacing = spacingFactor * self.toolWidth
        chord = self.toolWidth
        r = chord
        ziggy = 1.0
        reverse = False

        if origin == 'BL':
            xShift = 0.0
            yShift = 0.0
            thetaStart = 0.0
            thetaEnd = np.pi / 2.0
            rMax = max(self.imageWidth, self.imageHeight)
        elif origin == 'L':
            xShift = 0.0
            yShift = self.imageHeight / 2.0
            thetaStart = - np.pi / 2.0
            thetaEnd = np.pi / 2.0
            rMax = self.imageWidth
        elif origin == 'TL':
            xShift = 0.0
            yShift = self.imageHeight
            thetaStart = - np.pi / 2.0
            thetaEnd = 0.0
            rMax = max(self.imageWidth, self.imageHeight)
        elif origin == 'T':
            xShift = self.imageWidth / 2.0
            yShift = self.imageHeight
            thetaStart = - np.pi
            thetaEnd = 0.0
            rMax = self.imageHeight
        elif origin == 'TR':
            xShift = self.imageWidth
            yShift = self.imageHeight
            thetaStart = - np.pi
            thetaEnd = -np.pi / 2.0
            rMax = max(self.imageWidth, self.imageHeight)
        elif origin == 'R':
            xShift = self.imageWidth
            yShift = self.imageHeight / 2.0
            thetaStart = np.pi / 2.0
            thetaEnd = 3 * np.pi / 2.0
            rMax = self.imageWidth
        elif origin == 'BR':
            xShift = self.imageWidth
            yShift = 0.0
            thetaStart = np.pi / 2.0
            thetaEnd = np.pi
            rMax = max(self.imageWidth, self.imageHeight)
        elif origin == 'B':
            xShift = self.imageWidth / 2.0
            yShift = 0.0
            thetaStart = 0.0
            thetaEnd = np.pi
            rMax = self.imageHeight

        path = [[xShift, yShift]]

        while r < rMax:
            if reverse:
                thetaStep = - chord / r
            else:
                thetaStep = chord / r

            for theta in floatRange(thetaStart, thetaEnd, thetaStep):
                x = r * np.cos(theta) + xShift
                y = r * np.sin(theta) + yShift
                # rawPoint = self.__iTransOrigin([x, y])
                xPx = int(mapToRange(x, self.xRange, (0, self.image.size[0]-1)))
                yPx = int(mapToRange(y, self.yRange, (0, self.image.size[1]-1)))
                if xPx < self.image.size[0] and yPx < self.image.size[1] and xPx >= 0 and yPx >= 0:
                    # print self.image.size[0], self.image.size[1]
                    amp = (spacing / 2.5) * ((255 - self.image.getpixel((xPx,yPx))) / 255.0)
                    rAmp = r + (ziggy * amp)
                    path.append([rAmp * np.cos(theta) + xShift, rAmp * np.sin(theta) + yShift])
                # theta += chord / r
                ziggy = -ziggy
            r += spacing
            reverse = not reverse
            thetaTemp = thetaStart
            thetaStart = thetaEnd
            thetaEnd = thetaTemp

        self.__drawSingleLine(path)

    def __optimizeDots(self, dots, weightedGreedy = False):
        """
        Optimize the order of the dots with nearest-neighbor approach.
        weightedGreedy finds the TWO nearest points and favors the one
        closest to the initial point . This yields a more "human-like" path
        """
        if type(dots) is not np.ndarray:
            dots = np.array(dots)
        firstPoint = dots[0]
        maxLookAhead = int(math.sqrt(self.image.size[0]**2 + self.image.size[1]**2)*16.0)
        lookAhead = min(len(dots), maxLookAhead) if weightedGreedy else len(dots)
        for i in range(1, len(dots)):
            lastPoint = np.array([dots[i-1]])
            dist = scipy.spatial.distance.cdist(lastPoint, dots[i:i+lookAhead], 'euclidean')

            d1 = np.argmin(dist)
            if weightedGreedy:
                dist[0][d1] = float('inf')
                d2 = np.argmin(dist)
                d1, d2 = d1 + i, d2 + i

                if distance(firstPoint, dots[d1]) < distance(firstPoint, dots[d2]):
                    bestDot = d1
                else:
                    bestDot = d2
            else:
                bestDot = d1 + i

            tempDot = np.copy(dots[i])
            dots[i] = dots[bestDot]
            dots[bestDot] = tempDot
            if weightedGreedy:
                lookAhead = min(len(dots) - i- 1, maxLookAhead)
            else:
                lookAhead = len(dots) - i - 1
        return dots

    def __optimizeDotsTSP(self, dots):
        """
        Find a pairs of segments that would be shorter if part
        of the path is reversed (2-opt implementation)
        """
        if type(dots) is not np.ndarray:
            dots = np.array(dots)
        startTime = time.time()
        while time.time() - startTime < self.durationTSP:
            indexA = random.randint(0, len(dots)-2)
            indexB = random.randint(0, len(dots)-2)

            if abs(indexA - indexB) < 2: # cant be same or next
                continue                 # to each other

            if (indexB < indexA): # force indexA to be < indexB
                temp = indexB
                indexB = indexA
                indexA = temp

            a0 = dots[indexA]
            a1 = dots[indexA + 1]
            b0 = dots[indexB]
            b1 = dots[indexB + 1]

            dist1 = distance(a0, a1) + distance(b0, b1)
            dist2 = distance(a0, b0) + distance(a1, b1)

            if dist2 < dist1: # better if part of path is reversed
                dots[indexA+1:indexB+1] = np.flipud(dots[indexA+1:indexB+1])

        return dots

    def __optimizeSegments(self, segments):
        """
        Find optimized path for the lines in segments (GREEDY)
        """
        endpoints0 = np.array([segments[i][0] for i in range(len(segments))])
        endpoints1 = np.array([segments[j][-1] for j in range(len(segments))])
        # construct arrays of endpoints in this manner. array[:,0] and array[:,-1]
        # will not work in cases where the number of points within each line varies,
        # since a 2D array cannot be constructed

        optimizedPath = np.copy(segments)

        lastPoint = np.array([[0.0, 0.0]])
        for i in range(len(optimizedPath)):
            distances0 = scipy.spatial.distance.cdist(lastPoint, endpoints0[i:], 'euclidean')
            distances1 = scipy.spatial.distance.cdist(lastPoint, endpoints1[i:], 'euclidean')

            bestIndex0 = np.argmin(distances0)
            bestIndex1 = np.argmin(distances1)

            dist0 = distances0[0][bestIndex0]
            dist1 = distances1[0][bestIndex1]
            # print dist0,dist1
            currLine = np.copy(optimizedPath[i])
            currEndpoint0 = np.copy(endpoints0[i])
            currEndpoint1 = np.copy(endpoints1[i])

            if dist0 < dist1:
                bestLine = np.copy(optimizedPath[i + bestIndex0])
                optimizedPath[i + bestIndex0] = currLine
                optimizedPath[i] = bestLine

                bestEndpoint0 = np.copy(endpoints0[i + bestIndex0])
                endpoints0[i + bestIndex0] = currEndpoint0
                endpoints0[i] = bestEndpoint0

                bestEndpoint1 = np.copy(endpoints1[i + bestIndex0])
                endpoints1[i + bestIndex0] = currEndpoint1
                endpoints1[i] = bestEndpoint1
            else:
                bestLine = np.copy(optimizedPath[i + bestIndex1])
                bestLine = np.flipud(bestLine)
                optimizedPath[i + bestIndex1] = currLine
                optimizedPath[i] = bestLine

                bestEndpoint0 = np.copy(endpoints0[i + bestIndex1])
                endpoints0[i + bestIndex1] = currEndpoint0
                endpoints0[i] = bestEndpoint0

                bestEndpoint1 = np.copy(endpoints1[i + bestIndex1])
                endpoints1[i + bestIndex1] = currEndpoint1
                endpoints1[i] = bestEndpoint1

            lastPoint = np.array([optimizedPath[i][-1]])

        return optimizedPath

    # def __filterLongEdges(self, edges):
    #     # endpoints0 = np.array([edges[i][0] for i in range(len(edges))])
    #     # endpoints1 = np.array([edges[j][-1] for j in range(len(edges))])
    #     edgesFiltered = np.array(edges)
    #     # edgesFiltered =  edgesFiltered.reshape(edgesFiltered.shape[0] * edgesFiltered.shape[1], edgesFiltered.shape[2])
    #     # print edgesFiltered
    #
    #     distances = np.array([ distance(x, y) for x, y in edgesFiltered])
    #     threshold = np.mean(distances) * 2.0
    #     filter = np.argwhere(distances > threshold)
    #     # print filter.reshape(len(filter))
    #     edges = [v for i,v in enumerate(edges) if i not in filter]
    #     return edges

    def __optimizeDotsDelaunay(self, dots):
        dots = np.array(dots)
        tri = scipy.spatial.Delaunay(dots)
        edges = []
        for i in xrange(tri.nsimplex):
            if i > tri.neighbors[i,2]:
                edges.append([dots[tri.vertices[i,0]], dots[tri.vertices[i,1]]])
            if i > tri.neighbors[i,0]:
                edges.append([dots[tri.vertices[i,1]], dots[tri.vertices[i,2]]])
            if i > tri.neighbors[i,1]:
                edges.append([dots[tri.vertices[i,2]], dots[tri.vertices[i,0]]])
        # print np.array(edges)
        # edges = self.__filterLongEdges(edges)
        return self.__optimizeSegments(edges)

    def __generateDots(self, mode, humanError):
        """
        Generate field of dots with density corresponding
        to the values of the pixels
        """
        self.image = self.image.convert('L')
        self.image = ImageOps.autocontrast(self.image)

        # self.image.show()

        # First, resize the image so that the pixels correspond to the toolWidth
        rez = self.toolWidth if mode != 'Delaunay' else 2.0 * self.toolWidth

        pxWidth = int(self.imageWidth / rez)
        pxHeight = int(self.imageHeight / rez)
        self.image = self.image.resize((pxWidth, pxHeight), Image.ANTIALIAS)

        # Then, apply mode-specific enhancements
        if mode == 'Delaunay':
            self.image = ImageEnhance.Sharpness(self.image).enhance(27.0)
            self.image = ImageEnhance.Contrast(self.image).enhance(0.77)
            self.image = ImageEnhance.Brightness(self.image).enhance(1.27)
            self.image = ImageEnhance.Contrast(self.image).enhance(1.33)
        elif mode == 'Stipple' or mode == 'TSP':
            self.image = ImageEnhance.Sharpness(self.image).enhance(27.0)
        # self.image.show()
        self.image = self.image.convert('1')
        # self.image.save('dithered.png')

        # self.image.show()
        # sys.exit()

        # Now, create dot locations corresponding to black numpixels
        # in the processed image
        dots = []
        dotSize = rez
        reverse = False
        for d in diagonals(self.image.size[0], self.image.size[1]):
            if reverse:
                d.reverse()
            for i, j in d:
            # for i in range(self.image.size[1]):
            #     for j in range(self.image.size[0]):
                # if self.image.getpixel((j if not reverse else self.image.size[0]-1-j, i)) < 128:
                if self.image.getpixel((i, j)) < 128:
                    # x, y = (j if not reverse else self.image.size[0]-1-j)*dotSize, i*dotSize
                    x, y = i*dotSize, j*dotSize
                    if humanError: # silly humans...
                        r = random.uniform(0.0, dotSize/7.0)
                        theta = random.uniform(0.0, 2.0*np.pi)
                        x = x + r * np.cos(theta)
                        y = y + r * np.sin(theta)
                    if self.noNegative:
                        x += rez / 2.0
                        y += rez / 2.0
                    if j % 2 == 0:
                        x += rez / 2.0
                    dots.append([x, y])
            reverse = not reverse
        if mode == 'Stipple':
            dots = self.__optimizeDots(dots, weightedGreedy = True)
            self.__drawDots(dots)
        elif mode == 'TSP':
            dots = self.__optimizeDots(dots, weightedGreedy = False)
            dots = self.__optimizeDotsTSP(dots)
            self.__drawSingleLine(dots)
        elif mode == 'Delaunay':
            dots = self.__optimizeDotsDelaunay(dots)
            self.__drawSegments(dots)

    def __generateStipple(self):
        self.__generateDots(mode = 'Stipple', humanError = True)

    def __generateTSP(self):
        self.__generateDots(mode = 'TSP', humanError = False)

    def __generateSierpinskiTSP(self):
        rez = self.toolWidth
        pointsToVisit = []
        dotSize = rez
        pxWidth = int(self.imageWidth / rez)
        pxHeight = int(self.imageHeight / rez)
        self.image = self.image.resize((pxWidth, pxHeight), Image.ANTIALIAS)
        self.image = self.image.convert('1')

        for i in range(self.image.size[0]):
            for j in range(self.image.size[1]):
                if self.image.getpixel((i, j)) < 128:
                    pointsToVisit.append([i, j])
        traverse = [[p, self.__sierpinskiIndex(p)] for p in pointsToVisit]
        traverse = sorted(traverse, key=itemgetter(1))
        traverse = [q[0] for q in traverse]

        path = []
        for p in traverse:
            x, y = p[0] * dotSize, p[1] * dotSize
            path.append([x,y])
        self.__drawSingleLine(path)

    def __generateDelaunay(self, lowPoly = False):
        self.__generateDots(mode = 'Delaunay', humanError = True)

    def __generateSquares(self):
        """
        Generate field of squares with size corresponding
        to the values of the pixels
        """
        self.image = self.image.convert('L')
        self.image = ImageOps.autocontrast(self.image)

        rez = self.toolWidth * 6.0

        pxWidth = int(self.imageWidth / rez)
        pxHeight = int(self.imageHeight / rez)
        self.image = self.image.resize((pxWidth, pxHeight), Image.ANTIALIAS)
        self.image = self.image.convert('L')

        squares = []
        dotSize = rez

        reverse = False
        for i in range(self.image.size[1]):
            for j in range(self.image.size[0]):
                pxBrightness = self.image.getpixel((j if not reverse else self.image.size[0]-1-j, i))

                x, y = (j if not reverse else self.image.size[0]-1-j)*dotSize, i*dotSize
                x += rez / 2.0
                y += rez / 2.0

                deltaCorner = (rez * ((255.0 - float(pxBrightness)) / 255.0) ) / 2.0

                cornerBL = [x - deltaCorner, y - deltaCorner]
                cornerBR = [x + deltaCorner, y - deltaCorner]
                cornerTR = [x + deltaCorner, y + deltaCorner]
                cornerTL = [x - deltaCorner, y + deltaCorner]
                squares.append([cornerBL, cornerBR, cornerTR, cornerTL, cornerBL])
            reverse = not reverse

        self.__drawShapes(squares)

    def __generateCircles(self):
        """
        Generate field of circles with size corresponding
        to the values of the pixels
        """
        self.image = self.image.convert('L')
        self.image = ImageOps.autocontrast(self.image)

        rez = self.toolWidth * 6.0
        numSegs = 16
        thetaStep = (2.0 * np.pi) / float(numSegs)

        pxWidth = int(self.imageWidth / rez)
        pxHeight = int(self.imageHeight / rez)
        self.image = self.image.resize((pxWidth, pxHeight), Image.ANTIALIAS)
        self.image = self.image.convert('L')

        circles = []
        dotSize = rez

        reverse = False
        for i in range(self.image.size[1]):
            for j in range(self.image.size[0]):
                pxBrightness = self.image.getpixel((j if not reverse else self.image.size[0]-1-j, i))

                x, y = (j if not reverse else self.image.size[0]-1-j)*dotSize, i*dotSize
                x += rez / 1.8
                y += rez / 1.8

                radius = (rez * ((255.0 - float(pxBrightness)) / 255.0) ) / 1.8

                circle = []

                for k in range(numSegs + 1):
                    theta = thetaStep * float(k)
                    xCirc = x + (radius * np.cos(theta))
                    yCirc = y + (radius * np.sin(theta))
                    circle.append([xCirc, yCirc])

                circles.append(circle)

            reverse = not reverse

        self.__drawShapes(circles)

    def __generateShapesFromImage(self):
        """
        Generate field of custom shapes with size corresponding
        to the values of the pixels
        """
        self.image = self.image.convert('L')
        self.image = ImageOps.autocontrast(self.image)

        rez = self.toolWidth * 9.0

        pxWidth = int(self.imageWidth / rez)
        pxHeight = int(self.imageHeight / rez)
        self.image = self.image.resize((pxWidth, pxHeight), Image.ANTIALIAS)
        self.image = self.image.convert('L')

        pattern = Image.open(self.patternPath)
        # Filter on small images for better traced outlines
        if max(pattern.size) < 800: # TODO: Make it possible to force extra filtering
            pattern = pattern.convert('P', palette = Image.ADAPTIVE, colors = int(3))
            pattern = pattern.convert('RGB')
            pattern = np.array(pattern)

            for i in range(10):
                pattern = denoise_bilateral(pattern, sigma_color=0.088, sigma_spatial=0.5, multichannel = True)
            for i in range(13):
                pattern = denoise_bilateral(pattern, sigma_color=0.030, sigma_spatial=1.0, multichannel = True)

            # skio.imshow(pattern)
            # skio.show()
            # return

            pattern = Image.fromarray((pattern * 255).astype(np.uint8))

        # pattern = pattern.convert('P', palette = Image.ADAPTIVE, colors = int(2))
        # name = imagePath.split('.')
        # newname = name[0] + "WITHFILTER.png"
        # pattern.convert('RGB').save(newname)
        #
        # return
        pattern = ImageOps.flip(pattern)

        # make pattern image into 2 color image for tracing
        pattern = pattern.convert('P', palette = Image.ADAPTIVE, colors = int(2))

        data = np.array(pattern.convert('RGBA'))
        red, green, blue, alpha = data.T
        pattern = Image.fromarray(data)

        patternColors = pattern.convert('RGB').getcolors()
        patternColors.sort()
        patternColors.reverse()

        # name = imagePath.split('.')
        # newname = name[0] + "_adapt.jpg"
        # pattern.convert('RGB').save(newname)

        background_color = pattern.getpixel((0,0))[:3] # RGB value tuple for background color

        count = 0
        for numpixels, color in patternColors:
            if color != background_color:

                color_area = (red == color[0]) & (green == color[1]) & (blue == color[2]) # T/F Matrix of color location

                indices = np.where(color_area == True)
                coordinates = zip(indices[0], indices[1]) # stored col,row

                width, height = pattern.size

                mask = np.zeros((width, height))

                for c in coordinates:
                    mask[c[0]][c[1]] = 1

        bmp = potrace.Bitmap(mask)

        poopoo = int(math.pi * (((self.toolWidth * (pattern.size[0]/self.imageWidth)) / 2.0) ** 2))

        # trace the image
        path = bmp.trace(turdsize = poopoo, opttolerance = 0.2)
        # patternCurves = path.curves_tree[0].tesselate()

        # extract all outermost outlines
        patternCurves = []
        for c in path.curves_tree:
            c = c.tesselate()
            c = np.array(c)
            c = np.flip(c, 1)
            patternCurves.append(c)

        # Now that the shape is traced, center it about (0, 0) and
        # scale it to measure 1.0 across the larger axis
        patternMinX = float('inf')
        patternMaxX = - float('inf')
        patternMinY = float('inf')
        patternMaxY = - float('inf')

        # find the bounding box for the outlines
        for c in patternCurves:
            for p in c:
                if p[0] < patternMinX:
                    patternMinX = deepcopy(p[0])
                elif p[0] > patternMaxX:
                    patternMaxX = deepcopy(p[0])
                if p[1] < patternMinY:
                    patternMinY = deepcopy(p[1])
                elif p[1] > patternMaxY:
                    patternMaxY = deepcopy(p[1])

        patternTransX = (patternMinX + patternMaxX) / 2
        patternTransY = (patternMinY + patternMaxY) / 2
        patternTrans = np.array([patternTransX, patternTransY])
        patternScaleNormalize = max(patternMaxX - patternMinX, patternMaxY - patternMinY)

        # center the outlines about (0,0)
        for c in patternCurves:
            for p in c:
                p -= patternTrans
            # p /= patternScaleNormalize

        # scale the outlines to fit within a 1 by 1 bounding box
        for c in patternCurves:
            c /= patternScaleNormalize

        # sys.exit()

        patterns = []
        dotSize = rez

        reverse = False
        for i in range(self.image.size[1]):
            for j in range(self.image.size[0]):
                pxBrightness = self.image.getpixel((j if not reverse else self.image.size[0]-1-j, i))

                x, y = (j if not reverse else self.image.size[0]-1-j)*dotSize, i*dotSize
                x += rez / 2.0
                y += rez / 2.0
                patternTrans = np.array([x, y])

                patternSize = (rez * ((255.0 - float(pxBrightness)) / 255.0) )
                for c in patternCurves:
                    # scale the outline to the correct size
                    scaledPattern = (deepcopy(c) * patternSize).tolist()
                    patternReduced = [scaledPattern.pop(0)]

                    # reduce the number of segments. Many segments are likely too
                    # small to be resolved and are unnecessary
                    lastPoint = patternReduced[-1]
                    for p in scaledPattern:
                        if distance(p, lastPoint) >= self.toolWidth / 4.0:
                            lastPoint = p
                            patternReduced.append(p)

                    if not compPt(patternReduced[0], patternReduced[-1]):
                        patternReduced.append(patternReduced[0])

                    patternReduced = np.array(patternReduced)

                    # translate the outline to its proper position
                    for p in patternReduced:
                        p += patternTrans

                    patterns.append(patternReduced)
            reverse = not reverse

        self.__drawShapes(patterns)

    def __generateRGB(self):
        rez = self.toolWidth * 3.0

        pxWidth = int(self.imageWidth / rez)
        pxHeight = int(self.imageHeight / rez)
        self.image = self.image.resize((pxWidth, pxHeight), Image.ANTIALIAS)
        self.image = self.image.convert('RGB')

        imageData = np.array(self.image)
        red, green, blue = imageData.T

        colorData = (red, green, blue)

        originalGcodePath = self.gcodePath

        dotSize = rez
        for c, color in enumerate(colorData):
            bars = []
            reverse = False
            for i in range(self.image.size[0]):
                for j in range(self.image.size[1]):
                    pxBrightness = color[i, j if not reverse else self.image.size[1]-1-j]

                    x, y = i*dotSize, (j if not reverse else self.image.size[1]-1-j)*dotSize
                    y += rez / 2.0

                    deltaBar = ((rez - self.toolWidth) * ((255.0 - float(pxBrightness)) / 255.0) ) / 2.0

                    top = [x + c * (rez / 3.0), y + deltaBar]
                    bottom = [x + c * (rez / 3.0), y - deltaBar]

                    if reverse:
                        bars.append([top, bottom])
                    else:
                        bars.append([bottom, top])
                reverse = not reverse

            if c == 0:
                colorName = 'RED'
            elif c == 1:
                colorName = 'GREEN'
            else:
                colorName = 'BLUE'
            self.gcodePath = splitext(basename(originalGcodePath))[0] + colorName + '.gcode'
            self.__drawShapes(bars)

    def __generateCMY(self):

        rez = self.toolWidth
        dotDist = (rez * 4.0) / 3.0

        pxWidth = int(self.imageWidth / dotDist)
        pxHeight = int(self.imageHeight / dotDist)
        self.image = self.image.resize((pxWidth, pxHeight), Image.ANTIALIAS)
        self.image = self.image.convert('CMYK')

        imageData = np.asarray(self.image)
        cyan, magenta, yellow, keyline = imageData.T
        rawColorData = (cyan, magenta, yellow) #, keyline)

        colorData = []
        for d in rawColorData:
            imChannel = Image.fromarray(d.transpose(), mode="L")
            imChannel = imChannel.convert('1')
            imChannel = np.array(imChannel)
            colorData.append(imChannel)

        originalGcodePath = self.gcodePath

        magentaOffsetX = (rez * 1.0) / 3.0
        yellowOffsetX = ((rez * 1.0) / 3.0) * np.cos(np.radians(60.0))
        yellowOffsetY = ((rez * 1.0) / 3.0) * np.sin(np.radians(60.0))
        #   y
        # c   m

        for c, color in enumerate(colorData):
            dots = []
            reverse = False
            for i in range(self.image.size[0]):
                for j in range(self.image.size[1]):
                    if color[j if not reverse else self.image.size[1]-1-j, i]:
                        y, x = (j if not reverse else self.image.size[1]-1-j)*dotDist, i*dotDist

                        if c == 1:
                            x += magentaOffsetX
                        elif c == 2:
                            x += yellowOffsetX
                            y += yellowOffsetY

                        dots.append([x, y])

                reverse = not reverse

            if c == 0:
                colorName = 'CYAN'
            elif c == 1:
                colorName = 'MAGENTA'
            elif c == 2:
                colorName = 'YELLOW'

            self.gcodePath = splitext(basename(originalGcodePath))[0] + colorName + '.gcode'
            self.__drawDots(dots)

    def __generateCrosshatch(self):
        self.image = self.image.convert('L')
        self.image = ImageOps.autocontrast(self.image)

        # First, resize the image so that the pixels correspond to the toolWidth
        rez = 4.0 * self.toolWidth

        pxWidth = int(self.imageWidth / rez)
        pxHeight = int(self.imageHeight / rez)
        self.image = self.image.resize((pxWidth, pxHeight), Image.ANTIALIAS)
        self.image = self.image.convert('P', palette = Image.ADAPTIVE, colors = 5)


        data = np.array(self.image.convert('RGB'))
        red, green, blue = data.T
        self.image = Image.fromarray(data)

        shades = self.image.convert('RGB').getcolors()
        shades = [s[1][0] for s in shades]
        shades.sort()
        shades.reverse()

        # self.image.show()
        # print shades
        # sys.exit()

        lines = []
        dotSize = rez
        for d in diagonals(self.image.size[0], self.image.size[1]):
            lineSegment = []
            for i, j in d:
                i = self.image.size[0] - 1 - i
                pxBrightness = self.image.getpixel((i, j))[0]
                if pxBrightness < shades[0] and len(lineSegment) == 0:
                    x, y = i*dotSize, j*dotSize
                    lineSegment.append([x, y])
                elif ((pxBrightness >= shades[0] or (i == self.image.size[0]-1
                      or j == self.image.size[1]-1) or (i == 0 or j == 0))
                      and len(lineSegment) == 1):
                    x, y = i*dotSize, j*dotSize
                    lineSegment.append([x, y])
                    lines.append(deepcopy(lineSegment))
                    lineSegment = []

        for d in diagonals(self.image.size[0], self.image.size[1]):
            lineSegment = []
            for i, j in d:
                pxBrightness = self.image.getpixel((i, j))[0]
                if pxBrightness < shades[1] and len(lineSegment) == 0:
                    x, y = i*dotSize, j*dotSize
                    lineSegment.append([x, y])
                elif ((pxBrightness >= shades[1] or (i == self.image.size[0]-1
                      or j == self.image.size[1]-1) or (i == 0 or j == 0))
                      and len(lineSegment) == 1):
                    x, y = i*dotSize, j*dotSize
                    lineSegment.append([x, y])
                    lines.append(deepcopy(lineSegment))
                    lineSegment = []

        for j in range(self.image.size[1]):
            lineSegment = []
            for i in range(self.image.size[0]):
                pxBrightness = self.image.getpixel((i, j))[0]
                if pxBrightness < shades[2] and len(lineSegment) == 0:
                    x, y = i*dotSize, j*dotSize
                    lineSegment.append([x, y])
                elif ((pxBrightness >= shades[2] or (i == self.image.size[0]-1)
                      or (i == 0)) and len(lineSegment) == 1):
                    x, y = i*dotSize, j*dotSize
                    lineSegment.append([x, y])
                    lines.append(deepcopy(lineSegment))
                    lineSegment = []

        for i in range(self.image.size[0]):
            lineSegment = []
            for j in range(self.image.size[1]):
                pxBrightness = self.image.getpixel((i, j))[0]
                if pxBrightness < shades[3] and len(lineSegment) == 0:
                    x, y = i*dotSize, j*dotSize
                    lineSegment.append([x, y])
                elif ((pxBrightness >= shades[3] or (j == self.image.size[1]-1)
                      or (j == 0)) and len(lineSegment) == 1):
                    x, y = i*dotSize, j*dotSize
                    lineSegment.append([x, y])
                    lines.append(deepcopy(lineSegment))
                    lineSegment = []

        lines = self.__optimizeSegments(lines)
        self.__drawSegments(lines)

    def generate(self, design, width = None, height = None):
        self.design = design
        if self.design == 'spiral':
            if not self.__fitSquare(width, height):
                self.__noFitError()
                return False
            self.__generateSpiral()
            return True
        elif self.design == 'zigzag':
            if not self.__fit(width, height):
                self.__noFitError()
                return False
            self.__generateZigZag()
            return True
        elif self.design == 'radiateBL':
            if not self.__fit(width, height):
                self.__noFitError()
                return False
            self.__generateRadiate(origin = 'BL')
            return True
        elif self.design == 'radiateL':
            if not self.__fit(width, height):
                self.__noFitError()
                return False
            self.__generateRadiate(origin = 'L')
            return True
        elif self.design == 'radiateTL':
            if not self.__fit(width, height):
                self.__noFitError()
                return False
            self.__generateRadiate(origin = 'TL')
            return True
        elif self.design == 'radiateT':
            if not self.__fit(width, height):
                self.__noFitError()
                return False
            self.__generateRadiate(origin = 'T')
            return True
        elif self.design == 'radiateTR':
            if not self.__fit(width, height):
                self.__noFitError()
                return False
            self.__generateRadiate(origin = 'TR')
            return True
        elif self.design == 'radiateR':
            if not self.__fit(width, height):
                self.__noFitError()
                return False
            self.__generateRadiate(origin = 'R')
            return True
        elif self.design == 'radiateBR':
            if not self.__fit(width, height):
                self.__noFitError()
                return False
            self.__generateRadiate(origin = 'BR')
            return True
        elif self.design == 'radiateB':
            if not self.__fit(width, height):
                self.__noFitError()
                return False
            self.__generateRadiate(origin = 'B')
            return True
        elif self.design == 'stipple':
            if not self.__fit(width, height):
                self.__noFitError()
                return False
            self.__generateStipple()
            return True
        elif self.design == 'TSP':
            if self.durationTSP is None or self.durationTSP < 0.0:
                print'\n' + '\033[91m' + str(self.durationTSP) + ' is not a duration!' + '\033[0m'
                return False
            if not self.__fit(width, height):
                self.__noFitError()
                return False
            self.__generateTSP()
            return True
        elif self.design == 'sierpinskiTSP':
            if not self.__fit(width, height):
                self.__noFitError()
                return False
            self.__generateSierpinskiTSP()
            return True
        elif self.design == 'delaunay':
            if not self.__fit(width, height):
                self.__noFitError()
                return False
            self.__generateDelaunay()
            return True
        elif self.design == 'squares':
            if not self.__fit(width, height):
                self.__noFitError()
                return False
            self.__generateSquares()
            return True
        elif self.design == 'circles':
            if not self.__fit(width, height):
                self.__noFitError()
                return False
            self.__generateCircles()
            return True
        elif self.design == 'shapes':
            if not self.__fit(width, height):
                self.__noFitError()
                return False
            self.__generateShapesFromImage()
            return True
        elif self.design == 'RGB':
            if not self.__fit(width, height):
                self.__noFitError()
                return False
            self.__generateRGB()
            return True
        elif self.design == 'CMY':
            if not self.__fit(width, height):
                self.__noFitError()
                return False
            self.__generateCMY()
            return True
        elif self.design == 'crosshatch':
            if not self.__fit(width, height):
                self.__noFitError()
                return False
            self.__generateCrosshatch()
            return True
        else:
            print'\n' + '\033[91m' + self.design + ' is not a valid design!' + '\033[0m'
            return False

def usage():
    print('To convert to gcode:')
    print('-i <path/to/image>')
    print('-W <desired width(mm)> and/or -H <desired  height(mm)>')
    print('-t <tool(i.e. B1)>')
    print('-w <tool line width(mm)>')
    print('-s <print speed(mm/s>), defaults to 25')
    print('-z <zHop(mm) for travel moves>, defaults to 1.5')
    print('-d <design>')
    print('\nPossible designs:')
    print ('spiral, zigzag, radiateBL, radiateL, radiateTL')
    print('\nexample:')
    print('./img2art.py -i octopus.jpg -W 8 -H 8 -w 0.5 -s 25 -d spiral')

def main(argv):
    opts, args = getopt.getopt(argv, 'hi:w:s:z:W:H:d:p:T:a', ['help', 'input=', 'toolWidth=', 'speed=',
                                                              'width=', 'height=', 'design=', 'pattern=',
                                                              'time=', 'absolute'])
    imagePath = None
    gcodePath = None
    patternPath = None
    toolWidth = None
    speed = 25.0
    newWidth = None
    newHeight = None
    design = None
    noNegative = False
    durationTSP = 5.0

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit()
        elif opt in ('-i', '--input='):
            imagePath = arg
            print 'Input Image: ' + imagePath
        elif opt in ('-w', '--toolWidth='):
            toolWidth = float(arg)
            if toolWidth < 0.0:
                print '\n' + '\033[91m' + arg + 'mm is not a valid tool width!' + '\033[0m'
                sys.exit()
        elif opt in ('-s', '--speed='):
            speed = float(arg)
            if speed < 5.0  or speed > 100.0:
                print '\n' + '\033[91m' + arg + 'mm/s is not a valid speed!' + '\033[0m'
                sys.exit()
            print 'Speed: ' + arg + 'mm/s'
        elif opt in ('-W', '--width='):
            newWidth = float(arg)*10.0 # convert to mm
        elif opt in ('-H', '--height='):
            newHeight = float(arg)*10.0 # convert to mm
        elif opt in ('-d', '--design='):
            design = arg
        elif opt in ('-p', '--pattern='):
            patternPath = arg
            print 'Input Pattern: ' + patternPath
        elif opt in ('-T', '--time='):
            durationTSP = float(arg)
            print 'TSP duration: ' + str(durationTSP) + 's'
        elif opt in ('-a', '--absolute'):
            noNegative = True

    if imagePath is None:
        print'\n' + '\033[91m' + 'Please select an input file with -i <imagePath>' + '\033[0m'
        sys.exit()
    if design is None:
        print '\n' + '\033[91m' + 'Please select a design!' + '\033[0m'
        sys.exit()
    elif design == 'shapes' and patternPath is None:
        print'\n' + '\033[91m' + 'Please select a pattern file with -p <patternPath>' + '\033[0m'
        sys.exit()

    plotBounds = (300.0, 300.0)
    patherator = ImgPathGenerator(plotBounds)

    patherator.loadImage(imagePath)
    patherator.setImagePath(imagePath)
    gcodePath = splitext(basename(imagePath))[0] + '.gcode'
    patherator.setGCodePath(gcodePath)

    if patternPath != None:
        patherator.setPatternPath(patternPath)

    if speed == None:
        print '\n' + '\033[91m' + 'Please provide a speed in mm/s!' + '\033[0m'
        sys.exit()
    if toolWidth == None:
        print '\n' + '\033[91m' + 'Please set the tool width!' + '\033[0m'
        sys.exit()

    patherator.configure(speed, 100.0, toolWidth, durationTSP, False)
    patherator.lowerCommand('M400\nM340 P0 S1500\nG4 P250')
    patherator.raiseCommand('M400\nM340 P0 S600\nG4 P250')
    if noNegative:
        patherator.noNegativeCoords()

    if newWidth is None and newHeight is None:
        print '\n' + '\033[91m' + 'Please provide a dimension!' + '\033[0m'
        sys.exit()
    else:
        if not patherator.generate(design, newWidth, newHeight):
            print '\n' + '\033[91m' + 'Error generating plot!' + '\033[0m'
            sys.exit()
    print 'Tool width:' + str(patherator.toolWidth) + 'mm'
    print 'Width: ' + str(patherator.imageWidth/10.0) + 'cm'
    print 'Height: ' + str(patherator.imageHeight/10.0) + 'cm'

    if design == 'spiral':
        print '\n' + '\033[93m' + 'Align tool to center of plot before running!' + '\033[0m'
    else:
        print '\n' + '\033[93m' + 'Align tool to bottom left of plot before running!' + '\033[0m'

if __name__ == "__main__":
    startTime = time.time()
    print ''
    main(sys.argv[1:])
    endTime = time.time()
    print '\033[94m' + '\nTool path generated successfully in:'
    print format((endTime - startTime), '.2f') + ' seconds' + '\033[0m'
