"""
Patherator Trace N Fill Module
@author: Vincent Politzer <https://github.com/vjapolitzer>

Example:
  mediaSize = (mediaWidth, mediaHeight) # dimension of plotting media
  pathy = TraceNFill()
  pathy.configure(mediaSize = mediaSize, alignToMedium = False)
  pathy.lowerCommand('Z') # Z for configured Z-Hop moves, o/w any GCode
  pathy.raiseCommand('Z') # Z for configured Z-Hop moves, o/w any GCode
  pathy.setGCodeSavePath(gcodeSavePath) # Save path
  pathy.fit(width, height)  # Desired bounding dimensions
  pathy.addImageData(imageData) # Example below for generating data
  newTool = ToolConfig()
  newTool.lineWidth = lineWidth
  newTool.lowerTool = lowerTool
  newTool.raiseTool = raiseTool
  newTool.toolSelect = toolSelect
  newTool.perimeters = perimeters
  newTool.drawSpeed = drawSpeed
  newTool.infillDensity = infillDensity
  newTool.infillPattern = infillPattern
  newTool.infillAngle = infillAngle
  newTool.infillOverlap = infillOverlap
  pathy.addTool(newTool)
  # Repeat for each imageData/toolConfig pair
  pathy.generate() # Generate GCode and save to file

  Example of imageData generation off of loaded Image:
    im = Image.open(srcImagePath)
    im = im.convert('L')
    im = ImageOps.flip(im)
    im_px = im.load()
    imageData = np.zeros((im.size[0], im.size[1]), np.uint8)
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            if im_px[i,j] < 128:
                imageData[i,j] = 1
  """

def mapToRange(val, srcMax, dst):
    """
    Map the given value from the range of src to the range of dst.
    """
    return (float(val) / float(srcMax)) * (float(dst[1]) - float(dst[0])) + float(dst[0])

def floatRange(start, stop, step):
    """
    Generate an iterator from start to stop, allowing for floats
    """
    i = start
    while i < stop:
        yield i
        i += step

def compPt(p0, p1):
    """
    Returns True if 2-dim points p0 and p1
    are equal. Otherwise, returns False.
    """
    return p0[0] == p1[0] and p0[1] == p1[1]

def distance(p0, p1):
    """
    Distance between p0 and p1
    """
    return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)

def brightness(color):
    """
    Returns the luminance of a color in RGB form
    """
    return 0.2126*color[0] + 0.7152*color[1] + 0.0722*color[2]

def getBackgroundColor(im):
    """
    Detect the background color of an image.
    The color is assumed to be the the most common
    pixel color along the border of the image.
    """
    borderPixels = []
    borderColors = []
    for x in range(im.size[0]):
        # Get pixel along top
        px = im.getpixel((x, 0))[:3]
        borderPixels.append(px)
        if px not in borderColors:
            borderColors.append(px)

        # Get pixel along bottom
        px = im.getpixel((x, im.size[1] - 1))[:3]
        borderPixels.append(px)
        if px not in borderColors:
            borderColors.append(px)

    for y in range(1, im.size[1] - 1):
        # Get pixel along left
        px = im.getpixel((1, y))[:3]
        borderPixels.append(px)
        if px not in borderColors:
            borderColors.append(px)

        # Get pixel along right
        px = im.getpixel((im.size[0] - 1 - 1, y))[:3]
        borderPixels.append(px)
        if px not in borderColors:
            borderColors.append(px)

    mostOccurrences = 0
    mostCommonColor = None
    for color in borderColors:
        numOccurrences = borderPixels.count(color)
        if numOccurrences > mostOccurrences:
            mostOccurrences = numOccurrences
            mostCommonColor = color

    return mostCommonColor # RGB value tuple for background color

def filterLoRez(im, desiredNumColors):
    """
    Filter low resolution images to reduce compression
    and anti-alias artifacts
    """
    # TODO: Tweak to play nicer with certain colors
    im = im.convert('P', palette = Image.ADAPTIVE, colors = int(desiredNumColors+1))
    im = im.convert('RGB')
    im = np.array(im)

    for i in range(10):
        im = denoise_bilateral(im, sigma_color=0.088, sigma_spatial=0.5, multichannel = True)
    for i in range(13):
        im = denoise_bilateral(im, sigma_color=0.030, sigma_spatial=1.0, multichannel = True)

    # skio.imshow(im)
    # skio.show()
    # return

    im = Image.fromarray((im * 255).astype(np.uint8))

    return im

def extractColors(im, numColors):
    """
    Reduced im to numColors, and returns a list
    of mask, color pairs
    """
    im = im.convert('P', palette = Image.ADAPTIVE, colors = int(numColors))

    data = np.array(im.convert('RGBA'))
    colorData = data.T
    red, green, blue, alpha = data.T
    im = Image.fromarray(data)

    im_colors = im.convert('RGB').getcolors()
    im_colors = sorted(im_colors, key=lambda x: brightness(x[1]))
    im_colors.reverse()

    # name = srcImagePath.split('.')
    # newname = name[0] + "_adapt.jpg"
    # im.convert('RGB').save(newname)

    width, height = im.size

    backgroundColor = getBackgroundColor(im)

    colorSegments = []
    for numpixels, color in im_colors:
        if color != backgroundColor:

            color_area = (red == color[0]) & (green == color[1]) & (blue == color[2]) # T/F Matrix of color location
            indices = np.where(color_area == True)
            coordinates = zip(indices[0], indices[1]) # stored col,row

            mask = np.zeros((width, height))

            for c in coordinates:
                mask[c[0]][c[1]] = 1

            colorSegments.append([mask, color])

    return colorSegments

class TraceNFill:
    def __init__(self):
        self.mediaWidth = 0.0
        self.mediaHeight = 0.0
        self.startPoint = np.array([0.0, 0.0])
        self.alignToMedium = False
        self.travelSpeed = None
        self.xPx = None
        self.yPx = None
        self.plotWidth = None
        self.plotHeight = None
        self.xRange = None
        self.yRange = None
        self.align = np.array([0.0, 0.0])
        self.srcImagePath = None
        self.gcodeSavePath = None
        self.imData = []
        self.lineColorData = []
        self.toolData = []
        self.preamble = None
        self.postamble = None
        self.previewData = None

    def __transOrigin(self, point):
        """
        Translate point to be centered about the origin
        """
        t = np.array([(self.plotWidth / 2.0), (self.plotHeight / 2.0)])
        return point - t

    def __iTransOrigin(self, point):
        """
        Translate point to be centered about the bed center
        """
        t = np.array([(self.plotWidth / 2.0), (self.plotHeight / 2.0)])
        return point + t

    def __rotateAboutOrigin(self, point, theta):
        """
        Rotate point about the origin by theta radians.
        """
        c, s = np.cos(theta), np.sin(theta)
        R = np.matrix([[c, s], [-s, c]])
        return np.array(point * R)[0]

    def __rotatePoint(self, point, theta):
        """
        Rotate point about the bed center by theta radians.
        """
        point = np.array(point)
        point = self.__transOrigin(point)
        point = self.__rotateAboutOrigin(point, theta)
        point = self.__iTransOrigin(point)
        return point[0], point[1]

    def __translateAllPoints(self, islands, tx, ty):
        """
        Translate all points by [tx, ty]
        """
        for i in islands:
            for line in i:
                for point in line:
                    point[0] += tx
                    point[1] += ty

    def __fixLoop(self, loop):
        """
        Ensure loop, a list of points, is closed properly
        """
        if compPt(loop[0], loop[-1]):
            loop.pop()

    def __tesselateAndScale(self, curve):
        """
        Translate vector curves into points in the plotting space
        """
        outline = []
        prevPoint = [-1.0, -1.0]
        for point in curve.tesselate():
            scaledPoint = [mapToRange(point[1], self.xPx, self.xRange), mapToRange(point[0], self.yPx, self.yRange)]
            if prevPoint[0] != scaledPoint[0] or prevPoint[1] != scaledPoint[1]: # filter duplicates
                outline.append(scaledPoint)
            prevPoint = scaledPoint
        self.__fixLoop(outline)
        return outline

    def __splitIntoIslands(self, parentCurve, islands):
        """
        Group outlines together into islands (independent outer outlines + enclosed child shapes),
        resize the image, and center it on the bed
        """
        island = []
        island.append(self.__tesselateAndScale(parentCurve))
        for child in parentCurve.children:
            island.append(self.__tesselateAndScale(child))
            for subChild in child.children:
                self.__splitIntoIslands(subChild, islands)
        islands.append(island)

    def __generateConcentric(self, islands, spacing, toolConfig):
        """
        Generate concentric infill from the outlines, with spacing mm
        between the concentric lines
        """
        path = []
        while True:
            islands = self.__offsetIslands(islands, -spacing)
            islands = self.__cleanIslands(islands, toolConfig)
            if len(islands) == 0:
                break
            for island in islands:
                path.append(island)
        return path

    def __generateSpiral(self, spacing, toolConfig):
        """
        Generate points along an Archimedes spiral, with spacing mm
        between the spiral curves.
        """
        rMax = math.sqrt((self.plotWidth)**2 + (self.plotHeight)**2) / 2.0
        maxAmp = spacing / 2.0
        coils = rMax / spacing
        thetaMax = 2.0 * np.pi * coils
        rStep = rMax / thetaMax
        chord = toolConfig.lineWidth

        r = spacing
        theta = (2.0 * np.pi) + (chord / r)

        path = [[self.plotWidth / 2.0, self.plotHeight / 2.0]]

        while theta < thetaMax:
            x, y = r * np.cos(theta), r * np.sin(theta)
            path.append(self.__iTransOrigin([x, y]))
            theta = theta + (chord / r)
            r = rStep * theta
        return path

    def __generateGoldenSpiral(self, counterClockwise = True):
        """
        Generate points along a golden spiral
        """
        path = []
        rMax = math.sqrt((self.plotWidth)**2 + (self.plotHeight)**2) / 2.0
        golden = (1 + 5 ** 0.5) / 2
        omega = 0.27
        t = 0.0
        r = 0.0
        while r < rMax:
            r = golden ** ((omega * t * 2) / np.pi)
            if counterClockwise:
                x = r * np.cos(omega * t)
                y = r * np.sin(omega * t)
            else:
                x = r * np.cos(omega * -t)
                y = r * np.sin(omega * -t)
            path.append(self.__iTransOrigin([x, y]))
            t += 0.3

        return path

    def __generateLsysCurve(self, pStart, forwardCmds, productionRules, axiom, turnAngle, stepLength, n):
        """
        Generate a curve using an L-System. pStart is the starting point, forwardCmds is the set
        of chars that constitute a forward move, productionRules are the L-System production rules,
        axiom is the L-System axiom, turnAngle is the angle in degrees to turn at + or - commands,
        stepLength is the length in mm for each forward move, n is the number of iterations.
        """
        theta = 0.0
        x, y = pStart[0], pStart[1]
        curve = [[x, y]]
        minX = maxX = x
        minY = maxY = y
        rules = axiom
        for iteration in range(n):
            newRules = str()
            for char in rules:
                if char in productionRules:
                    newRules += productionRules[char]
                else:
                    newRules += char
            rules = newRules

        for step in rules:
            if step in forwardCmds:
                x, y = x + stepLength * np.cos(theta), y + stepLength * np.sin(theta)
                curve.append([x,y])
                if x > maxX:
                    maxX = x
                elif x < minX:
                    minX = x
                if y > maxY:
                    maxY = y
                elif y < minY:
                    minY = y
            elif step == '+':
                theta = (theta - np.radians(turnAngle)) % (2 * np.pi)
            elif step == '-':
                theta = (theta + np.radians(turnAngle)) % (2 * np.pi)

        tX = (((self.plotWidth) / 2.0) - (minX + maxX) / 2.0)
        tY = (((self.plotHeight) / 2.0) - (minY + maxY) / 2.0)

        for point in curve: # center the curve on the plotting area
            point[0], point[1] = point[0] + tX, point[1] + tY

        return curve

    def __generateHilbert(self, x, y, length, n):
        """
        Generate points on the Hilbert curve
        """
        forwardCmds = ('F')
        productionRules = {
            'L': '+RF-LFL-FR+',
            'R': '-LF+RFR+FL-'
        }
        axiom = 'L'
        turnAngle = 90.0

        return self.__generateLsysCurve((x,y), forwardCmds, productionRules, axiom, turnAngle, length, n)

    def __generateGosper(self, x, y, length, n):
        """
        Generate points on the Peano-Gosper curve
        """
        forwardCmds = ('A', 'B')
        productionRules = {
            'A': 'A-B--B+A++AA+B-',
            'B': '+A-BB--B-A++A+B'
        }
        axiom = 'A'
        turnAngle = 60.0

        return self.__generateLsysCurve((x,y), forwardCmds, productionRules, axiom, turnAngle, length, n)

    def __generatePeano(self, x, y, length, n):
        """
        Generate points on the Peano curve
        """
        forwardCmds = ('F')
        productionRules = {
            'L': 'LFRFL+F+RFLFR-F-LFRFL',
            'R': 'RFLFR-F-LFRFL+F+RFLFR'
        }
        axiom = 'L'
        turnAngle = 90.0

        return self.__generateLsysCurve((x,y), forwardCmds, productionRules, axiom, turnAngle, length, n)

    def __generateSierpinski(self, x, y, length, n):
        """
        Generate points on the Sierpinski curve
        """
        forwardCmds = ('F')
        productionRules = {
            'X': 'XF-F+F-XF+F+XF-F+F-X'
        }
        axiom = 'X'
        turnAngle = 90.0

        return self.__generateLsysCurve((x,y), forwardCmds, productionRules, axiom, turnAngle, length, n)

    def __generateHex(self, fillRange, xFillMin, yFillMin, sideLength):
        """
        Generate hexagon fill pattern by iterating over 'center' vertices
        and appending the corresponding segments to the list of fill lines
          |       |       |
          p00     p01     p02   ...
        /   \   /   \   /   \
              |       |       |
              p10     p11     p12  ...
            /   \   /   \   /   \
          |       |       |
          p20     p21     p22 ...
        /   \   /   \   /   \

             (px, py+sideLength)
                    |
                 (px, py)
                  /   \
        (px-w, py-h) (px+w, py-h)
        """
        hexFill = []
        h = sideLength * np.cos(np.radians(60))
        w = sideLength * np.sin(np.radians(60))
        N = 1
        M = 1
        while N * (sideLength + h) < fillRange:
            N += 1
        while M * (2 * w) < fillRange:
            M += 1

        for i, y in enumerate(floatRange(yFillMin, yFillMin + (N * (sideLength + h)), sideLength + h)):
            if i % 2 == 0:
                xStart = xFillMin
            else:
                xStart = xFillMin + w
            for x in floatRange(xStart, xStart + (M * (2 * w)), 2 * w):
                    hexFill.append([[x-w, y-h], [x, y]])
                    hexFill.append([[x, y], [x, y+sideLength]])
                    hexFill.append([[x, y], [x+w, y-h]])

        return hexFill

    def __generateOctagramSpiral(self, spacing):
        """
        Generate points along an Octagram spiral.
        Thanks to: https://github.com/alexrj/Slic3r/
        """
        rMax = math.sqrt((self.plotWidth)**2 + (self.plotHeight)**2) / 2.0
        r = 0.0
        ogsFill = [[self.plotWidth / 2.0, self.plotHeight / 2.0]]
        while r < rMax:
            r += spacing
            rx = r / math.sqrt(2.0)
            r2 = r + rx
            ogsFill.append(self.__iTransOrigin([r,  0.0]))
            ogsFill.append(self.__iTransOrigin([r2, rx]))
            ogsFill.append(self.__iTransOrigin([rx, rx]))
            ogsFill.append(self.__iTransOrigin([rx, r2]))
            ogsFill.append(self.__iTransOrigin([0.0,  r]))
            ogsFill.append(self.__iTransOrigin([-rx, r2]))
            ogsFill.append(self.__iTransOrigin([-rx, rx]))
            ogsFill.append(self.__iTransOrigin([-r2, rx]))
            ogsFill.append(self.__iTransOrigin([-r,  0.0]))
            ogsFill.append(self.__iTransOrigin([-r2, -rx]))
            ogsFill.append(self.__iTransOrigin([-rx, -rx]))
            ogsFill.append(self.__iTransOrigin([-rx, -r2]))
            ogsFill.append(self.__iTransOrigin([0.0, -r]))
            ogsFill.append(self.__iTransOrigin([rx, -r2]))
            ogsFill.append(self.__iTransOrigin([rx, -rx]))
            ogsFill.append(self.__iTransOrigin([r2+spacing, -rx]))

        return ogsFill

    def __generateShapeFill(self, bounds, toolConfig):

        xFillMin = bounds[0]
        xFillMax = bounds[1]
        yFillMin = bounds[2]
        yFillMax = bounds[3]

        pattern = Image.open(toolConfig.infillPattern)
        # Filter on small images for better traced outlines
        if max(pattern.size) < 800: # TODO: Make it possible to force extra filtering
            pattern = filterLoRez(pattern, 3)

        # pattern = pattern.convert('P', palette = Image.ADAPTIVE, colors = int(2))
        # name = srcImagePath.split('.')
        # newname = name[0] + "WITHFILTER.png"
        # pattern.convert('RGB').save(newname)
        #
        # return
        pattern = ImageOps.flip(pattern)

        # make pattern image into 2 color image for tracing
        mask, color = extractColors(pattern, 2)[0]

        bmp = potrace.Bitmap(mask)

        poopoo = int(math.pi * (((toolConfig.lineWidth * (pattern.size[0]/self.plotWidth)) / 2.0) ** 2))

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

        # thumbnailCurve = np.array(thumbnailCurve)
        # thumbnailCurve = np.flip(thumbnailCurve, 1)
        # # print thumbnailCurve.shape

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

        patternDimX = patternMaxX - patternMinX
        patternDimY = patternMaxY - patternMinY

        fillShapeMax = mapToRange(toolConfig.infillDensity, 100.0, (self.plotWidth, 3.0*toolConfig.lineWidth))
        if patternDimX > patternDimY:
            fillShapeWidth = fillShapeMax
            fillShapeHeight = (fillShapeMax * patternDimY) / patternDimX
        else:
            fillShapeHeight = fillShapeMax
            fillShapeWidth = (fillShapeMax * patternDimX) / patternDimY

        # scale the outlines
        for c in patternCurves:
            c /= patternScaleNormalize
            c *= (4 * fillShapeMax) / 5

        patternCurvesReduced = []
        for c in patternCurves:
            pattern = c.tolist()
            patternReduced = [pattern.pop(0)]

            # reduce the number of segments. Many segments are likely too
            # small to be resolved and are unnecessary
            lastPoint = patternReduced[-1]
            for p in pattern:
                if distance(p, lastPoint) >= toolConfig.lineWidth / 4.0:
                    lastPoint = p
                    patternReduced.append(p)

            if not compPt(patternReduced[0], patternReduced[-1]):
                patternReduced.append(patternReduced[0])

            patternCurvesReduced.append(np.array(patternReduced))

        shapeFill = []
        everyOtherShift = False
        for i in floatRange(yFillMin, yFillMax, fillShapeHeight):
            for j in floatRange(xFillMin, xFillMax, fillShapeWidth):

                if everyOtherShift:
                    j += fillShapeWidth / 2.0

                patternTrans = np.array([j, i])

                for pattern in deepcopy(patternCurvesReduced):
                    # translate the outline to its proper position
                    for p in pattern:
                        p += patternTrans
                    shapeFill.append(pattern)
            everyOtherShift = not everyOtherShift
        # print shapeFill
        return shapeFill

    def __offsetIslands(self, islands, offset):
        """
        Offset islands by offset
        islands => list of islands, which are lists of outlines, each a list of points
        """
        newOutlines = []
        pco = pyclipper.PyclipperOffset()
        scaleFactor = 2 ** 32
        for i in islands:
            pco.AddPaths(pyclipper.scale_to_clipper(i, scaleFactor), pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON) # JT_ROUND is EVIL
            offsetIsle = pyclipper.scale_from_clipper(pco.Execute(offset * scaleFactor), scaleFactor)
            for outline in offsetIsle:
                self.__fixLoop(outline)
            newOutlines.append(offsetIsle)
            pco.Clear()        # sys.exit()

        return newOutlines

    def __cleanIslands(self, islands, toolConfig):
        """
        Remove islands too small for tool to resolve
        """
        cleaned = []
        for island in islands:
            tooSmall = []
            for i, outline in enumerate(island):
                totalDistance = 0.0
                for j, point in enumerate(outline):
                    if j != 0:
                        totalDistance += distance(outline[j-1], outline[j])
                if totalDistance < toolConfig.lineWidth:
                    tooSmall.append(i)
            if len(tooSmall) != len(island) and len(island) > 0:
                cleaned.append([v for i,v in enumerate(island) if i not in tooSmall])
        return np.array(cleaned)

    def __closestIsland(self, refPoint, unsortedIslands):
        """
        Find the island closest to refPoint, return its index
        within unsortedIslands
        """
        distances = np.zeros(len(unsortedIslands))
        refPoint = np.array([refPoint])
        for index, island in enumerate(unsortedIslands):
            allPoints = [[np.array(island[line][point]) for point in range(len(island[line]))] for line in range(len(island))]
            allPoints = np.array([item for sublist in allPoints for item in sublist])
            dist = scipy.spatial.distance.cdist(refPoint, allPoints, 'euclidean')
            dist = dist[0][np.argmin(dist)]
            distances[index] = dist
        return np.argmin(distances)

    def __closestLine(self, refPoint, unsortedLines):
        """
        Find the line closest to refPoint, return its index
        within unsortedLines
        """
        distances = np.zeros(len(unsortedLines))
        bestIndices = range(len(unsortedLines))
        refPoint = np.array([refPoint])
        for index, line in enumerate(unsortedLines):
            allPoints = np.array([np.array(line[point]) for point in range(len(line))])
            dist = scipy.spatial.distance.cdist(refPoint, allPoints, 'euclidean')
            bestDex = np.argmin(dist)
            dist = dist[0][bestDex]
            distances[index] = dist
            bestIndices[index] = bestDex
        bestLine = np.argmin(distances)
        bestDex = bestIndices[bestLine]
        return bestLine, bestDex

    def __optimizeLineOrder(self, islands, toolConfig):
        """
        Optimize the order of lines to be drawn
        """
        islands = self.__cleanIslands(islands, toolConfig)
        optimizedOutlines = np.array([[np.array(i[j]) for j in range(len(i))] for i in islands])
        optimizedOutlines = np.array([item for sublist in optimizedOutlines for item in sublist])

        lastPoint = self.startPoint
        for i in range(len(optimizedOutlines)):
            bestLine, bestDex = self.__closestLine(lastPoint, optimizedOutlines[i:])

            currLine = np.copy(optimizedOutlines[i])
            newLine = np.copy(optimizedOutlines[i + bestLine])
            newLine = np.roll(newLine, (len(newLine)-bestDex)%len(newLine), axis=0)

            optimizedOutlines[i + bestLine] = currLine
            optimizedOutlines[i] = newLine

            lastPoint = np.array(optimizedOutlines[i][0])

        return optimizedOutlines

    def __optimizeInfill(self, islandFill, lastPoint):
        """
        Find optimized path for the lines in islandFill (GREEDY)
        """
        endpoints0 = np.array([islandFill[i][0] for i in range(len(islandFill))])
        endpoints1 = np.array([islandFill[j][-1] for j in range(len(islandFill))])
        # construct arrays of endpoints in this manner. array[:,0] and array[:,-1]
        # will not work in cases where the number of points within each line varies,
        # since a 2D array cannot be constructed

        optimizedFill = np.array([np.array(k) for k in islandFill])

        lastPoint = np.array([lastPoint])
        for i in range(len(optimizedFill)):
            distances0 = scipy.spatial.distance.cdist(lastPoint, endpoints0[i:], 'euclidean')
            distances1 = scipy.spatial.distance.cdist(lastPoint, endpoints1[i:], 'euclidean')

            bestIndex0 = np.argmin(distances0)
            bestIndex1 = np.argmin(distances1)

            dist0 = distances0[0][bestIndex0]
            dist1 = distances1[0][bestIndex1]
            # print dist0,dist1
            currLine = np.copy(optimizedFill[i])
            currEndpoint0 = np.copy(endpoints0[i])
            currEndpoint1 = np.copy(endpoints1[i])

            if dist0 < dist1:
                bestLine = np.copy(optimizedFill[i + bestIndex0])
                optimizedFill[i + bestIndex0] = currLine
                optimizedFill[i] = bestLine

                bestEndpoint0 = np.copy(endpoints0[i + bestIndex0])
                endpoints0[i + bestIndex0] = currEndpoint0
                endpoints0[i] = bestEndpoint0

                bestEndpoint1 = np.copy(endpoints1[i + bestIndex0])
                endpoints1[i + bestIndex0] = currEndpoint1
                endpoints1[i] = bestEndpoint1
            else:
                bestLine = np.copy(optimizedFill[i + bestIndex1])
                bestLine = np.flipud(bestLine)
                optimizedFill[i + bestIndex1] = currLine
                optimizedFill[i] = bestLine

                bestEndpoint0 = np.copy(endpoints0[i + bestIndex1])
                endpoints0[i + bestIndex1] = currEndpoint0
                endpoints0[i] = bestEndpoint0

                bestEndpoint1 = np.copy(endpoints1[i + bestIndex1])
                endpoints1[i + bestIndex1] = currEndpoint1
                endpoints1[i] = bestEndpoint1

            lastPoint = np.array([optimizedFill[i][-1]])

        return optimizedFill

    def __calculateInfill(self, islands, lastPoint, toolConfig):
        """
        Calculate infill paths by generating the requested pattern, then clipping it
        using the outlines of the islands. lastPoint is the last visited point on
        the outlines, used for infill path optimization
        """
        print '  Calculating infill...'
        scaleFactor = 2 ** 32

        if toolConfig.infillPattern not in ('concentric'):
            if toolConfig.perimeters > 0 and toolConfig.infillOverlap < 100.0:
                overlapOffset = -toolConfig.lineWidth * (1.0 - (toolConfig.infillOverlap/100.0))
                islands = self.__offsetIslands(islands, overlapOffset)
        islands = self.__cleanIslands(islands, toolConfig)

        infillUnclipped = []
        infillSpacing = toolConfig.lineWidth/(toolConfig.infillDensity/100.0)

        fillRange = math.sqrt(self.plotWidth**2 + self.plotHeight**2)
        xFillMax = (self.plotWidth + fillRange) / 2.0
        xFillMin = (self.plotWidth - fillRange) / 2.0
        yFillMax = (self.plotHeight + fillRange) / 2.0
        yFillMin = (self.plotHeight - fillRange) / 2.0

        if toolConfig.infillPattern in ('linear', 'zigzag', 'grid'):
            for i in floatRange(yFillMin, yFillMax, infillSpacing):
                infillUnclipped.append([[xFillMin, i], [xFillMax, i]])
            if toolConfig.infillPattern == 'grid':
                for i in floatRange(xFillMin, xFillMax, infillSpacing):
                    infillUnclipped.append([[i, yFillMin], [i, yFillMax]])
            if toolConfig.infillPattern == 'zigzag':
                for j in range(len(infillUnclipped)-2):
                    infillUnclipped.append([infillUnclipped[j][-1], infillUnclipped[j+1][0]])
        elif toolConfig.infillPattern in ('triangle', 'david'):
            numLines = int(math.floor(fillRange/infillSpacing))
            # shift endpoints to ensure a line intersects the rotation
            # point so that triangles are formed correctly
            if numLines % 2 == 0:
                yShift = (fillRange - ((numLines) * infillSpacing)) / 2.0
            else:
                yShift = (fillRange - ((numLines - 1) * infillSpacing)) / 2.0
            if toolConfig.infillPattern == 'david':
                yShift += infillSpacing / 2.0

            for i in range(3):
                for j in floatRange(yFillMin + yShift, yFillMax + yShift, infillSpacing):
                    infillUnclipped.append([[xFillMin, j], [xFillMax, j]])
                if i < 2:
                    for line in infillUnclipped:
                        for point in line:
                            point[0], point[1] = self.__rotatePoint(point, np.radians(60.0))
        elif toolConfig.infillPattern == 'spiral':
            infillLine = self.__generateSpiral(infillSpacing, toolConfig)
            infillUnclipped.append(infillLine)
        elif toolConfig.infillPattern in ('golden', 'sunflower'):
            numLines = max(int(36 * (toolConfig.infillDensity/100.0)), 1)
            for i in range(numLines):
                infillLine = self.__generateGoldenSpiral(counterClockwise = True)
                infillUnclipped.append(infillLine)
                if toolConfig.infillPattern == 'sunflower':
                    infillLine = self.__generateGoldenSpiral(counterClockwise = False)
                    infillUnclipped.append(infillLine)
                if i < numLines - 1:
                    for line in infillUnclipped:
                        for point in line:
                            point[0], point[1] = self.__rotatePoint(point, np.radians(360.0/float(numLines)))
        elif toolConfig.infillPattern == 'hilbert':
            numSegs = int(fillRange / (infillSpacing)) + 1
            n = 0
            totalSegs = 0
            while totalSegs < numSegs:
                n += 1
                totalSegs = (2 ** n) - 1
            infillLine = self.__generateHilbert(0.0, 0.0, infillSpacing, n)
            infillUnclipped.append(infillLine)
        elif toolConfig.infillPattern == 'gosper':
            n = 0
            while math.sqrt(7.0) ** n < fillRange:
                n += 1
            infillLine = self.__generateGosper(0.0, 0.0, (infillSpacing * 2.0) / math.sqrt(3.0), n+1)
            infillUnclipped.append(infillLine)
        elif toolConfig.infillPattern == 'peano':
            numSegs = int(fillRange / (infillSpacing)) + 1
            n = 0
            totalSegs = 0
            while totalSegs < numSegs:
                n += 1
                totalSegs = (3 ** n) - 1
            infillLine = self.__generatePeano(0.0, 0.0, infillSpacing, n)
            infillUnclipped.append(infillLine)
        elif toolConfig.infillPattern == 'sierpinski':
            numSegs = int((2.5 * fillRange) / (infillSpacing)) + 1
            n = 0
            totalSegs = 0
            while totalSegs < numSegs:
                n += 1
                totalSegs = 4.0 * ((2 ** n) - 1)
            infillLine = self.__generateSierpinski(0.0, 0.0, infillSpacing, n)
            infillUnclipped.append(infillLine)
        elif toolConfig.infillPattern == 'concentric':
            infill = self.__generateConcentric(islands, infillSpacing, toolConfig)
            print '  Optimizing infill paths...'
            infill = [self.__optimizeLineOrder(infill, toolConfig)]
            return infill
        elif toolConfig.infillPattern == 'hexagon':
            infillUnclipped = self.__generateHex(fillRange, xFillMin, yFillMin, infillSpacing / (2 * np.sin(np.radians(60))))
        elif toolConfig.infillPattern == 'octagramspiral':
            infillLine = self.__generateOctagramSpiral(infillSpacing)
            infillUnclipped.append(infillLine)
        else: # infillPattern must be shapeFill
            infillUnclipped = self.__generateShapeFill((xFillMin, xFillMax, yFillMin, yFillMax), toolConfig)

        if toolConfig.infillAngle != 0.0:
            for island in islands:
                for outline in island:
                    for point in outline:
                        point[0], point[1] = self.__rotatePoint(point, np.radians(toolConfig.infillAngle))
        rawInfill = []
        for island in islands:
            islandFill = []
            pc = pyclipper.Pyclipper()
            pc.AddPaths(pyclipper.scale_to_clipper(island, scale = scaleFactor), pyclipper.PT_CLIP, True)
            pc.AddPaths(pyclipper.scale_to_clipper(infillUnclipped, scale = scaleFactor), pyclipper.PT_SUBJECT, False)

            polyFill = (pc.Execute2(pyclipper.CT_INTERSECTION, pyclipper.PFT_POSITIVE, pyclipper.PFT_EVENODD))
            if len(polyFill.Childs) != 0:
                for line in polyFill.Childs:
                    linePath = []
                    for i, point in enumerate(line.Contour):
                        linePath.append([float(point[0]) / scaleFactor, float(point[1]) / scaleFactor])
                    if len(linePath) > 0: # reject empty lines
                        if toolConfig.infillAngle != 0.0:
                            for point in linePath:
                                point[0], point[1] = self.__rotatePoint(point, np.radians(-toolConfig.infillAngle))
                        islandFill.append(linePath)
            if len(islandFill) > 0:
                rawInfill.append(islandFill)
        if len(islands) == 0:
            infill = rawInfill
            return infill

        infill = []
        print '  Optimizing infill paths...'
        while len(rawInfill) > 0:
            closestIslandIndex = self.__closestIsland(lastPoint, rawInfill)
            infill.append(self.__optimizeInfill(rawInfill.pop(closestIslandIndex), lastPoint))
            lastPoint = infill[-1][-1][-1]
        return infill

    def __setDimensions(self, desiredWidth = None, desiredHeight = None):
        aspectRatio = float(self.xPx) / float(self.yPx)
        if ((desiredWidth is not None and desiredHeight is None)
            or (desiredWidth is not None and desiredHeight is not None
                and desiredWidth/desiredHeight < aspectRatio)):
            self.plotWidth = desiredWidth
            self.plotHeight = desiredWidth / aspectRatio
        else:
            self.plotWidth = desiredHeight * aspectRatio
            self.plotHeight = desiredHeight

        xMin = 0.0
        xMax = self.plotWidth
        yMin = 0.0
        yMax = self.plotHeight
        self.xRange = (0.0, xMax)
        self.yRange = (0.0, yMax)

        if self.alignToMedium: # Center the paths in the plotting area
            self.align[0] = - (self.mediaWidth - self.plotWidth) / 2.0
            self.align[1] = - (self.mediaHeight - self.plotHeight) / 2.0

        if self.plotWidth > self.plotHeight:
            previewW = 1024
            previewH = int((1024*self.plotHeight)/self.plotWidth)
        else:
            previewW = int((1024*self.plotWidth)/self.plotHeight)
            previewH = 1024

        self.previewData = np.ones((previewH, previewW, 3), dtype=np.uint8) * 255

        return (xMax <= self.mediaWidth and yMax <= self.mediaHeight)

    def fit(self, width = None, height = None):
        """
        Fit image into area bounded by width(mm) and height(mm),
        preserving aspect ratio. If only one dimension is
        provided, only said dimension is considered.
        Returns true if the calculated image dimensions fit
        within the plotting boundaries, else returns false
        """
        return self.__setDimensions(width, height)

    def __outlinePreview(self, allOutlines, lineColor, previewW, previewH, dilateRadius):
        """
        Generate anti-aliased image data for preview of outlines
        """
        perimPreview = np.zeros((previewH, previewW))
        for outline in allOutlines:
            for i in range(len(outline)):
                p1, p2 = deepcopy(outline[i-1]), deepcopy(outline[i])
                p1[0] = mapToRange(p1[0], self.plotWidth, (0, previewW))
                p1[1] = mapToRange(p1[1], self.plotHeight, (0, previewH))
                p2[0] = mapToRange(p2[0], self.plotWidth, (0, previewW))
                p2[1] = mapToRange(p2[1], self.plotHeight, (0, previewH))
                rr, cc, val = drawLine(int(p1[1]), int(p1[0]), int(p2[1]), int(p2[0]))
                perimPreview[rr, cc] = val
        dilated = skmorph.dilation(perimPreview, skmorph.disk(int(dilateRadius)))
        dilated = dilated.reshape(dilated.shape[0], dilated.shape[1], 1)

        lineColorInv = (np.ones(3) * 255.0) - np.array(lineColor)

        perimPreviewColored = (np.ones((previewH, previewW, 3)) * lineColorInv)  * dilated
        return perimPreviewColored.astype(np.uint8)

    def __infillPreview(self, infill, lineColor, previewW, previewH, dilateRadius, pattern):
        """
        Generate anti-aliased image data for preview of infill
        """
        fillPreview = np.zeros((previewH, previewW))
        for islandFill in infill:
            for line in islandFill:
                if pattern == 'concentric':
                    startDex = 0
                else:
                    startDex = 1
                for i in range(startDex, len(line)):
                    p1, p2 = deepcopy(line[i-1]), deepcopy(line[i])
                    p1[0] = mapToRange(p1[0], self.plotWidth, (0, previewW))
                    p1[1] = mapToRange(p1[1], self.plotHeight, (0, previewH))
                    p2[0] = mapToRange(p2[0], self.plotWidth, (0, previewW))
                    p2[1] = mapToRange(p2[1], self.plotHeight, (0, previewH))
                    rr, cc, val = drawLine(int(p1[1]), int(p1[0]), int(p2[1]), int(p2[0]))
                    fillPreview[rr, cc] = val
        dilated = skmorph.dilation(fillPreview, skmorph.disk(int(dilateRadius)))
        dilated = dilated.reshape(dilated.shape[0], dilated.shape[1], 1)

        lineColorInv = (np.ones(3) * 255.0) - np.array(lineColor)

        fillPreviewColored = (np.ones((previewH, previewW, 3)) * lineColorInv)  * dilated
        return fillPreviewColored.astype(np.uint8)

    def __initGcode(self):
        """
        Create GCode file and insert basic comments
        srcImagePath => path/to/input/image
        gcodeSavePath => path/to/output.gcode
        width and height from calling calculateDim
        """
        with open(self.gcodeSavePath, 'w') as f:
            f.write('; Patherator Trace N Fill GCode\n;\n')
            f.write('; Original image: ' + basename(self.srcImagePath) + '\n;\n')
            f.write('; Width: ' + str(self.plotWidth/10.0) + 'cm\n')
            f.write('; Height: ' + str(self.plotHeight/10.0) + 'cm\n;\n')

    def __startGcode(self):
        """
        Append GCode file with preparatory gcode
        """
        with open(self.gcodeSavePath, 'a') as f:
            f.write('; Begin GCode\n')
            if self.preamble is not None:
                f.write(self.preamble + '\n')
            f.write('G21\nG90\n') # preparatory gcode
            f.write('G92 X' + format(self.align[0], '.8f') + ' Y'
                    + format(self.align[1], '.8f') + '\n')

    def __endGcode(self, lastFile = True):
        """
        Append GCode file with ending gcode
        """
        with open(self.gcodeSavePath, 'a') as f:
            if self.postamble is not None:
                f.write(self.postamble + '\n')
            if lastFile:
                f.write('G0 X' + format(self.xRange[0]+self.align[0], '.8f') + ' Y' + format(self.yRange[1]-self.align[1], '.8f')
                        + ' F' + str(self.travelSpeed*60.0) +'\n')
            else:
                f.write('G0 X' + format(self.align[0], '.8f') + ' Y' + format(self.align[1], '.8f')
                        + ' F' + str(self.travelSpeed*60.0) +'\n')

    def __insertToolComment(self, toolConfig):
        """
        Append GCode file with tool parameter comments
        """
        with open(self.gcodeSavePath, 'a') as f:
            if toolConfig.toolSelect is not None:
                f.write('; Tool select: ' + toolConfig.toolSelect.replace('\n', '|') + '\n')
            f.write('; Tool width: ' + str(toolConfig.lineWidth) + 'mm\n')
            f.write('; Number of perimeters: ' + str(toolConfig.perimeters) + '\n')
            if toolConfig.infillDensity > 0.0:
                f.write('; Infill type: ' + toolConfig.infillPattern + '\n')
                f.write('; Infill density: ' + str(toolConfig.infillDensity) + '%\n')
                f.write('; Infill angle: ' + str(toolConfig.infillAngle) + 'degrees\n')
                f.write('; Infill overlap: ' + str(toolConfig.infillOverlap) + '%\n')
            f.write('; Draw speed: ' + str(toolConfig.drawSpeed) + 'mm/s\n')
            f.write('; Travel speed: ' + str(self.travelSpeed) + 'mm/s\n;\n')

    def __insertAllToolComments(self):
        """
        Append GCode file with all tool parameter comments
        """
        for toolConfig in self.toolData:
            self.__insertToolComment(toolConfig)

    def __generatePath(self, bmp, toolConfig, generatePreview, lineColor):
        """
        Generate gcode from binary image data and additional parameters
        and append to GCode file
        """
        print '  Calculating outlines...'
        # ignore outlines that encapsulate an area of less than poopoo
        poopoo = 2
        if toolConfig.lineWidth > 0.0:
            poopoo = int(math.pi * (((toolConfig.lineWidth * (self.xPx/self.plotWidth)) / 2.0) ** 2))
        # Trace the bitmap to a path
        path = bmp.trace(turdsize = poopoo, opttolerance = 0.2)

        islands = [] # list of of list of outlines
        for curve in path.curves_tree:
            self.__splitIntoIslands(curve, islands)

        if toolConfig.lineWidth > 0.0:
            islands = self.__offsetIslands(islands, -toolConfig.lineWidth / 2.0)
            infillPerimeter = deepcopy(islands)
            if toolConfig.perimeters > 0:
                if toolConfig.perimeters > 1:
                    for i in range(toolConfig.perimeters - 1):
                        infillPerimeter = self.__offsetIslands(infillPerimeter, -toolConfig.lineWidth)
                        if len(infillPerimeter) == 0:
                            break
                        islands = islands + deepcopy(infillPerimeter)
        else:
            infillPerimeter = deepcopy(islands)

        print'  Optimizing outline paths...'
        allOutlines = self.__optimizeLineOrder(islands, toolConfig)

        if len(allOutlines) == 0:
            print '\033[91m' + '  No resolvable details found!' + '\033[0m'
            return

        self.startPoint = allOutlines[-1][-1]

        if toolConfig.infillPattern != 'none':
            infill = self.__calculateInfill(infillPerimeter, np.copy(allOutlines[-1][0]), toolConfig)
            if len(infill) > 0:
                self.startPoint = infill[-1][-1][-1]

        if generatePreview:
            print '  Generating preview...'
            previewW, previewH = self.previewData.shape[1], self.previewData.shape[0]
            dilateRadius = ((toolConfig.lineWidth * float(previewW)) / self.plotWidth) / 3.0
            if toolConfig.perimeters > 0:
                self.previewData -= self.__outlinePreview(allOutlines, lineColor, previewW, previewH, dilateRadius)

            if toolConfig.infillDensity > 0.0:
                self.previewData -= self.__infillPreview(infill, lineColor, previewW, previewH,
                                                       dilateRadius, toolConfig.infillPattern)

        # Now we are ready to save the paths as GCode
        with open(self.gcodeSavePath, 'a') as f:
            if toolConfig.toolSelect is not None:
                f.write(toolConfig.toolSelect + '\n') # tool select code

            if toolConfig.perimeters > 0:
                for i, outline in enumerate(allOutlines):
                    f.write('; Begin Loop ' + str(i) + '\n')
                    firstPoint = outline[0]
                    for j, point in enumerate(outline):
                        if j == 0:
                            f.write(toolConfig.raiseTool + '\n')
                            f.write('G0 X' + format(point[0], '.8f') + ' Y' + format(point[1], '.8f')
                                    + ' F' + str(self.travelSpeed*60.0) +'\n')
                            f.write(toolConfig.lowerTool + '\n')
                        else:
                            f.write('G1 X' + format(point[0], '.8f') + ' Y' + format(point[1], '.8f')
                                    + ' F' + str(toolConfig.drawSpeed*60.0) +'\n')
                    f.write('G1 X' + format(firstPoint[0], '.8f') + ' Y' + format(firstPoint[1], '.8f')
                            + ' F' + str(toolConfig.drawSpeed*60.0) +'\n')

            if toolConfig.infillDensity > 0.0:
                for i, islandFill in enumerate(infill):
                    f.write('; Begin Infill ' + str(i) + '\n')
                    for j, line in enumerate(islandFill):
                        firstPoint = line[0]
                        for k, point in enumerate(line):
                            if k == 0:
                                if j > 0 and compPt(point, islandFill[j-1][-1]):
                                    continue # previous was same position, already in position
                                f.write(toolConfig.raiseTool + '\n')
                                f.write('G0 X' + format(point[0], '.8f') + ' Y' + format(point[1], '.8f')
                                        + ' F' + str(self.travelSpeed*60.0) +'\n')
                                f.write(toolConfig.lowerTool + '\n')
                            else:
                                f.write('G1 X' + format(point[0], '.8f') + ' Y' + format(point[1], '.8f')
                                        + ' F' + str(toolConfig.drawSpeed*60.0) +'\n')
                        if toolConfig.infillPattern == 'concentric':
                            f.write('G1 X' + format(firstPoint[0], '.8f') + ' Y' + format(firstPoint[1], '.8f')
                                    + ' F' + str(toolConfig.drawSpeed*60.0) +'\n')

            f.write(toolConfig.raiseTool + '\n')

    def __generateAllPaths(self, generatePreview):
        """
        Generate gcode from binary image data and additional parameters
        and append to GCode file for all parts of the image
        """
        segmentIndex = 1
        for bmp, toolConfig, lineColor in zip(self.imData, self.toolData, self.lineColorData):
            print 'Part ' + str(segmentIndex) + '/' + str(len(self.imData))
            self.__generatePath(bmp, toolConfig, generatePreview, lineColor)
            segmentIndex += 1

    def configure(self, mediaSize, travelSpeed, preamble = None, postamble = None, alignToMedium = False):
        self.mediaWidth = mediaSize[0]
        self.mediaHeight = mediaSize[1]
        self.travelSpeed = travelSpeed
        self.preamble = preamble
        self.postamble = postamble
        self.alignToMedium = alignToMedium

    def setGCodeSavePath(self, path):
        self.gcodeSavePath = path

    def setSrcImagePath(self, path):
        self.srcImagePath = path

    def addImageData(self, data, lineColor):
        """
        Convert data into potrace Bitmap and store internally
        """
        if self.xPx is not None and self.yPx is not None:
            if data.shape[0] != self.xPx or data.shape[1] != self.yPx:
                print 'New image dimensions do not match previous image dimensions!'
                return # TODO: Perhaps raise an exception of some sort here.
        else:
            self.xPx = data.shape[0]
            self.yPx = data.shape[1]
        self.imData.append(potrace.Bitmap(data))
        self.lineColorData.append(lineColor)

    def numImages(self):
        return len(self.imData)

    def addTool(self, tool):
        """
        Store tool configuration object with given parameters
        """
        self.toolData.append(tool)

    def numTools(self):
        return len(self.toolData)

    def generate(self, singleFile = False, preview = True, savePreview = False):
        print('\nGenerating toolpaths...')
        basePath = self.gcodeSavePath.split('.')[0]
        if singleFile:
            self.__initGcode()
            self.__insertAllToolComments()
            self.__startGcode()
            self.__generateAllPaths(preview or savePreview)
            self.__endGcode()
        else:
            fileNum = 1
            for bmp, toolConfig, lineColor in zip(self.imData, self.toolData, self.lineColorData):
                print 'Part ' + str(fileNum) + '/' + str(len(self.imData))
                self.setGCodeSavePath(basePath + str(fileNum) + '.gcode')
                self.__initGcode()
                self.__insertToolComment(toolConfig)
                self.__startGcode()
                self.__generatePath(bmp, toolConfig, preview or savePreview, lineColor)
                if fileNum < len(self.imData):
                    self.__endGcode(False)
                else:
                    self.__endGcode()
                fileNum += 1
        if preview or savePreview:
            previewIm = Image.fromarray(self.previewData)
            previewIm = ImageOps.flip(previewIm)
            if preview:
                previewIm.show()
            if savePreview:
                newname = basePath + "_preview.png"
                previewIm.save(newname)
