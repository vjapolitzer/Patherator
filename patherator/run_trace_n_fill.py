"""
Patherator Run Trace N Fill Module
@author: Vincent Politzer <https://github.com/vjapolitzer>

This module is used to parse the JSON configuration file, setup and
finally run Trace N Fill module according to the given parameters.
"""

import os, sys
from PIL import Image, ImageOps
from trace_n_fill import TraceNFill
from imagetools import extractColors, filterLoRez, getNumColors
from toolconfig import ToolConfig
import json
from os.path import basename, splitext

def checkDefined(requiredItems, configItems):
    """
    Check if all required items are defined
    """
    allReqsFound = True
    for i in requiredItems:
        if i not in configItems:
            print '\033[91m' + i + ' not defined!' + '\033[0m'
            allReqsFound = False
    return allReqsFound

def runTraceNFill(configFile, preview, savePreview):
    """
    Configure Patherator Trace N Fill according to parameters
    in configFile(JSON) and run the Trace N Fill module
    """
    print 'Reading ' + basename(configFile)

    with open(configFile) as jsonFile:
        jsonConfig = json.load(jsonFile)

    requiredItems = ['plotter', 'plot', 'tools']
    if not checkDefined(requiredItems, jsonConfig):
        raise LookupError('Configuration file missing required items!')

    plotterConfig = jsonConfig['plotter']
    plotConfig = jsonConfig['plot']
    toolConfigList = jsonConfig['tools']

    requiredItems = ['mediaWidth', 'mediaHeight', 'travelSpeed',
                     'alignToMedium', 'singleFile']
    if not checkDefined(requiredItems, plotterConfig):
        raise LookupError('plotter attribute missing required items!')

    mediaWidth = plotterConfig['mediaWidth']
    mediaHeight = plotterConfig['mediaHeight']
    mediaSize = (mediaWidth, mediaHeight)

    for dimension in mediaSize:
        if not dimension > 0.0:
            raise ValueError("dimensions must be greater than 0.0mm!")

    travelSpeed = plotterConfig['travelSpeed']

    if not travelSpeed > 0.0:
        raise ValueError("travelSpeed must be greater than 0.0mm/s!")

    alignToMedium = plotterConfig['alignToMedium']
    singleFile = plotterConfig['singleFile']

    if not isinstance(alignToMedium, bool):
        raise ValueError("alignToMedium must be a bool!")

    if not isinstance(singleFile, bool):
        raise ValueError("alignToMedium must be a bool!")

    requiredItems = ['srcImagePath', 'plotWidth', 'plotHeight']
    if not checkDefined(requiredItems, plotConfig):
        raise LookupError('plot attribute missing required items!')

    srcImagePath = plotConfig['srcImagePath']
    if not os.path.isfile(srcImagePath):
        raise IOError("source image not found in specified directory!")

    if 'gcodeSavePath' in plotConfig:
        gcodeSavePath = plotConfig['gcodeSavePath']
    else:
        gcodeSavePath = splitext(srcImagePath)[0] + '.gcode'

    plotWidth = plotConfig['plotWidth']
    plotHeight = plotConfig['plotHeight']

    print gcodeSavePath
    sys.exit()

    print '\nOriginal image: ' + basename(srcImagePath)

    pathy = TraceNFill()
    pathy.configure(mediaSize = mediaSize, travelSpeed = travelSpeed,
                         preamble = preamble, postamble = postamble,
                         alignToMedium = alignToMedium,)

    pathy.setSrcImagePath(srcImagePath)
    pathy.setGCodeSavePath(gcodeSavePath)

    segmentNumber = 1
    for plotConfig in configuration[2:]:
        if plotConfig[0][0] == ';':
            continue
        for config in plotConfig:
            configTok = config.split('=')
            configTok[0] = configTok[0].lower()
            if configTok[0] == 'toolwidth':
                lineWidth = float(configTok[1])
                if lineWidth < 0.0:
                    print '\033[91m' + configTok[1] + 'mm is not a valid tool width!' + '\033[0m'
                    sys.exit()
            elif configTok[0] == 'lowercommand':
                lowerTool = configTok[1].replace('|', '\n')
            elif configTok[0] == 'raisecommand':
                raiseTool = configTok[1].replace('|', '\n')
            elif configTok[0] == 'toolselect':
                toolSelect = configTok[1].replace('|', '\n')
            elif configTok[0] == 'perimeters':
                perimeters = int(configTok[1])
                if perimeters < 0:
                    print '\033[91m' + configTok[1] + ' is not a valid number of perimeters!' + '\033[0m'
                    sys.exit()
            elif configTok[0] == 'drawspeed':
                drawSpeed = float(configTok[1])
                if drawSpeed < 0.0:
                    print '\033[91m' + configTok[1] + 'mm/s is not a valid drawSpeed!' + '\033[0m'
                    sys.exit()
            elif configTok[0] == 'infilldensity':
                infillDensity = float(configTok[1])
                if infillDensity < 0.0 or infillDensity > 100.0:
                    print '\033[91m' + configTok[1] + '% is not a valid density!' + '\033[0m'
                    sys.exit()
            elif configTok[0] == 'infillpattern':
                infillPattern = configTok[1].lower()
                if not infillPattern in ('linear', 'zigzag', 'grid', 'triangle', 'spiral', 'golden',
                                         'sunflower', 'hilbert', 'gosper', 'peano', 'sierpinski',
                                         'concentric',  'hexagon', 'octagramspiral', 'david', 'shapefill'):
                    print '\033[91m' + infillPattern + ' is not a valid infill type!' + '\033[0m'
                    sys.exit()
            elif configTok[0] == 'infillangle':
                infillAngle = float(configTok[1])
            elif configTok[0] == 'infilloverlap':
                infillOverlap = float(configTok[1])
                if infillOverlap < 0.0 or infillOverlap > 100.0:
                    print '\033[91m' + infillOverlap + '% is not a valid overlap!' + '\033[0m'
                    sys.exit()
            elif configTok[0] == 'patternpath':
                patternPath = configTok[1]
            else:
                print '\033[91m' + 'Unknown parameter: ' + config + '\033[0m'
                sys.exit()
        if lowerTool is None or raiseTool is None:
            print '\033[91m' + 'Undefined raise or lower commands!' + '\033[0m'
            sys.exit()
        if infillPattern == 'shapefill' and patternPath is None:
            print '\033[91m' + 'Path to pattern image not provided!' + '\033[0m'
            sys.exit()
        if infillPattern != 'shapefill' and patternPath is not None:
            print '\n' + '\033[93m' + 'Selected fill pattern is ' + infillPattern + ', supplied pattern image will not be used!' + '\033[0m'

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
        newTool.patternPath = patternPath
        pathy.addTool(newTool)

        print '\nPart ' + str(segmentNumber) + ' Configuration:'
        if toolSelect is not None:
            print 'Tool select: ' + toolSelect.replace('\n', '|')
        print 'Tool width:' + str(lineWidth) + 'mm'
        print 'Perimeters: ' + str(perimeters)
        print 'Speed: ' + str(drawSpeed) + 'mm/s'
        print 'Infill density: ' + str(infillDensity) + '%'
        print 'Infill type: ' + infillPattern
        print 'Infill angle: ' + str(infillAngle) + 'deg'
        if perimeters > 0:
            print 'Infill overlap: ' + str(infillOverlap) + '%'
        segmentNumber += 1

    numColors = pathy.numTools() + 1

    im = Image.open(srcImagePath)
    if max(im.size) < 800: # TODO: Make it possible to force extra filtering
        print '\nPerforming filtering to improve image clarity...'
        im = filterLoRez(im, numColors)

    # im = im.convert('P', palette = Image.ADAPTIVE, colors = int(numColors))
    # name = srcImagePath.split('.')
    # newname = name[0] + "WITHFILTER.png"
    # im.convert('RGB').save(newname)
    #
    # return

    im = ImageOps.flip(im)

    print '\nSeparating colors...'

    colorSegments = extractColors(im, numColors)

    for mask, color in colorSegments:
        pathy.addImageData(mask, color)

    if not pathy.fit(plotWidth, plotHeight):
        print 'Requested width: ' + str(pathy.plotWidth/10.0) + 'cm'
        print 'Requested height: ' + str(pathy.plotHeight/10.0) + 'cm'
        print 'Max width: ' + str(pathy.mediaWidth/10.0) + 'cm'
        print 'Max height: ' + str(pathy.mediaHeight/10.0) + 'cm'
        raise ValueError("Desired dimensions do not fit with the plotting boundaries!")

    print '\nCalculated dimensions:'
    print 'Width: ' + format(pathy.plotWidth/10.0, '.2f') + 'cm'
    print 'Height: ' + format(pathy.plotHeight/10.0, '.2f') + 'cm'

    pathy.generate(singleFile, preview, savePreview)

    if alignToMedium:
        print '\n' + '\033[93m' + 'Align tool to bottom left of medium before running!' + '\033[0m'
    else:
        print '\n' + '\033[93m' + 'Align tool to bottom left of plot before running!' + '\033[0m'

    print '\033[94m' + '\nTool path generated successfully in:' + '\033[0m'
