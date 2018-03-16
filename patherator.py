#! /usr/bin/env python

"""
Patherator
@author: Vincent Politzer <https://github.com/vjapolitzer>

Dependencies: numpy, scipy, pillow, pypotrace, pyclipper, skimage, sklearn

TraceNFill Configuration Parameters:
  mediaWidth: Width of media in mm, i.e. 200.0
  mediaHeight: Height of media in mm, i.e. 200.0
  alignToMedium: whether to align point is bottom left of medium,
    or bottom left of drawing
  travelSpeed: Speed for non-draw moves in mm/s, i.e. 100.0

ToolConfig Configuration Parameters:
  lineWidth: Drawn line width in mm, i.e. 0.5
  perimeters: Number of perimeters/outlines to be drawn, i.e. 1
  drawSpeed: Speed for draw moves in mm/s, i.e. 25.0
  infillDensity: Density of fill lines, in %, i.e. 100.0
    50.0 means infill lines separated by lineWidth
    100.0 means infill lines touching
  infillPattern: String containing desired pattern
    Currently implemented patterns:
      linear, zigzag, grid, triangle, spiral, golden, sunflower,
      hilbert, gosper, peano, concentric, hexagon, david,
      octagramspiral, sierpinski, pathToShapeFill
  infillAngle: Angle of infill pattern in degrees, i.e. 45.0
  infillOverlap: Amount of overlap of infill lines on perimeters,
                 expressed in percent of the lineWidth
  lowerTool: GCode string for machine specific lower command
  raiseTool: GCode string for machine specific raise command
  toolSelect: Optional GCode string for machine specific tool selection
"""

import sys, os, argparse, time
from patherator import tracenfill, artsy

def main(argv):
    # opts, args = getopt.getopt(argv, 'hspi:c:n:', ['help', 'savePreview', 'preview', 'input=',
    #                                                'colorDetect', 'numColors'])

    parser = argparse.ArgumentParser(description="Plotter Path Generator")
    parser.add_argument("-i", "--input", dest="configFile", required=True,
                        help="input JSON file with configuration parameters")
    parser.add_argument("--tracenfill", action="store_true",
                        help="run Trace N Fill module using CONFIGFILE")
    parser.add_argument("--artsy", action="store_true",
                        help="run Artsy module using CONFIGFILE")
    parser.add_argument("-p", "--preview", action="store_true",
                        help="generate preview in Trace N Fill")
    parser.add_argument("-s", "--savepreview", action="store_true",
                        help="save preview in Trace N Fill as image file")
    # parser.add_argument("-c", "--colorDetect", dest="colorDetect",
    #                     help="input image file to detect number of colors")
    args = parser.parse_args()

    if args.tracenfill and args.artsy:
        print "img2path.py: error: cannot select both Trace N Fill and Artsy modules"
        return

    if not args.tracenfill and not args.artsy:
        print "img2path.py: error: select either Trace N Fill and Artsy module"
        return

    if args.tracenfill:
        tracenfill.run(args.configFile, args.preview, args.savepreview)

    if args.artsy:
        raise NotImplementedError("JSON configuration for artsy.py is not "
                                  + "implemented, run directly with args")


if __name__ == "__main__":
    startTime = time.time()
    main(sys.argv[1:])
    endTime = time.time()
    print '\033[94m' + format((endTime - startTime), '.2f') + ' seconds' + '\033[0m'
