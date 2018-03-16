"""
Patherator Run Artsy Module
@author: Vincent Politzer <https://github.com/vjapolitzer>

This module is used to parse the JSON configuration file, setup and
finally run Artsy module according to the given parameters.
"""

def run(configFile):
    opts, args = getopt.getopt(argv, 'hi:w:s:z:W:H:d:p:T:a', ['help', 'input=', 'lineWidth=', 'speed=',
                                                              'width=', 'height=', 'design=', 'pattern=',
                                                              'time=', 'absolute'])
    imagePath = None
    gcodePath = None
    patternPath = None
    lineWidth = None
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
        elif opt in ('-w', '--lineWidth='):
            lineWidth = float(arg)
            if lineWidth < 0.0:
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
    pathy = Artsy(plotBounds)

    pathy.loadImage(imagePath)
    pathy.setImagePath(imagePath)
    gcodePath = splitext(imagePath)[0] + '.gcode'
    pathy.setGCodePath(gcodePath)

    if patternPath != None:
        pathy.setPatternPath(patternPath)

    if speed == None:
        print '\n' + '\033[91m' + 'Please provide a speed in mm/s!' + '\033[0m'
        sys.exit()
    if lineWidth == None:
        print '\n' + '\033[91m' + 'Please set the tool width!' + '\033[0m'
        sys.exit()

    pathy.configure(speed, 100.0, lineWidth, durationTSP, False)
    pathy.setLowerCommand('M400\nM340 P0 S1500\nG4 P250')
    pathy.setRaiseCommand('M400\nM340 P0 S1000\nG4 P250')
    # pathy.setLowerCommand('G1 Z0 F300')
    # pathy.setRaiseCommand('G1 Z2 F300')
    if noNegative:
        pathy.noNegativeCoords()

    if newWidth is None and newHeight is None:
        print '\n' + '\033[91m' + 'Please provide a dimension!' + '\033[0m'
        sys.exit()
    else:
        if not pathy.generate(design, newWidth, newHeight):
            print '\n' + '\033[91m' + 'Error generating plot!' + '\033[0m'
            sys.exit()
    print 'Tool width:' + str(pathy.lineWidth) + 'mm'
    print 'Width: ' + str(pathy.imageWidth/10.0) + 'cm'
    print 'Height: ' + str(pathy.imageHeight/10.0) + 'cm'

    if design == 'spiral':
        print '\n' + '\033[93m' + 'Align tool to center of plot before running!' + '\033[0m'
    else:
        print '\n' + '\033[93m' + 'Align tool to bottom left of plot before running!' + '\033[0m'
