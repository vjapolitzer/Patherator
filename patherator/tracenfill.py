"""
Patherator Run Trace N Fill Module
@author: Vincent Politzer <https://github.com/vjapolitzer>

This module is used to parse the JSON configuration file, setup and
finally run Trace N Fill module according to the given parameters.
"""

import os, sys
from PIL import Image, ImageOps
from tracenfillcore import TraceNFill
from imagetools import extractColors, filterLoRez, getNumColors
from toolconfig import ToolConfig
import json
from os.path import basename, splitext

def _checkDefined(requiredItems, configItems):
    """
    Check if all required items are defined
    """
    allReqsFound = True
    for i in requiredItems:
        if i not in configItems:
            print '\033[91m' + i + ' not defined!' + '\033[0m'
            allReqsFound = False
    return allReqsFound

def run(configFile, preview, savePreview):
    """
    Configure Patherator Trace N Fill according to parameters
    in configFile(JSON) and run the Trace N Fill module
    """
    print 'Reading ' + basename(configFile)

    with open(configFile) as jsonFile:
        jsonConfig = json.load(jsonFile)

    requiredItems = ['plotter', 'plot', 'tools']
    if not _checkDefined(requiredItems, jsonConfig):
        raise LookupError('Configuration file missing required items!')

    plotterConfig = jsonConfig['plotter']
    plotConfig = jsonConfig['plot']
    toolConfigList = jsonConfig['tools']

    requiredItems = ['mediaWidth', 'mediaHeight', 'travelSpeed',
                     'alignToMedium', 'singleFile']
    if not _checkDefined(requiredItems, plotterConfig):
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

    if 'preamble' in plotterConfig:
        preamble = plotterConfig['preamble']
    else:
        preamble = None

    if 'postamble' in plotterConfig:
        postamble = plotterConfig['postamble']
    else:
        postamble = None

    requiredItems = ['srcImagePath', 'plotWidth', 'plotHeight']
    if not _checkDefined(requiredItems, plotConfig):
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

    print '\nOriginal image: ' + basename(srcImagePath)

    pathy = TraceNFill()
    pathy.configure(mediaSize = mediaSize, travelSpeed = travelSpeed,
                         preamble = preamble, postamble = postamble,
                         alignToMedium = alignToMedium,)

    pathy.setSrcImagePath(srcImagePath)
    pathy.setGCodeSavePath(gcodeSavePath)

    segmentNumber = 1
    requiredItems = ['lineWidth', 'perimeters', 'drawSpeed', 'infillDensity',
                     'infillPattern', 'infillAngle', 'infillOverlap',
                     'lowerCommand', 'raiseCommand']
    for tool in toolConfigList:
        if not _checkDefined(requiredItems, tool):
            raise LookupError('tool attribute missing required items!')

        newTool = ToolConfig()
        newTool.lineWidth = tool['lineWidth']
        newTool.perimeters = tool['perimeters']
        newTool.drawSpeed = tool['drawSpeed']
        newTool.infillDensity = tool['infillDensity']
        newTool.infillPattern = tool['infillPattern']
        newTool.infillAngle = tool['infillAngle']
        newTool.infillOverlap = tool['infillOverlap']
        newTool.lowerCommand = tool['lowerCommand']
        newTool.raiseCommand = tool['raiseCommand']

        if 'toolSelect' in tool:
            newTool.toolSelect = tool['toolSelect']

        pathy.addTool(newTool)

        print '\nPart ' + str(segmentNumber) + ' Configuration:'
        if newTool.toolSelect is not None:
            print 'Tool select: ' + newTool.toolSelect.replace('\n', '|')
        print 'Tool width:' + str(newTool.lineWidth) + 'mm'
        print 'Perimeters: ' + str(newTool.perimeters)
        print 'Speed: ' + str(newTool.drawSpeed) + 'mm/s'
        print 'Infill density: ' + str(newTool.infillDensity) + '%'
        print 'Infill pattern: ' + basename(newTool.infillPattern)
        print 'Infill angle: ' + str(newTool.infillAngle) + 'deg'
        if newTool.perimeters > 0:
            print 'Infill overlap: ' + str(newTool.infillOverlap) + '%'
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
