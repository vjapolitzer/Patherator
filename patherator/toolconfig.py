"""
Patherator ToolConfig Module
@author: Vincent Politzer <https://github.com/vjapolitzer>

ToolConfig object for storing tool configuration parameters for TraceNFill path
generation. Ensures valid configuration parameters upon setting.
"""

import os

validInfillPatterns = ('none', 'linear', 'zigzag', 'grid', 'triangle', 'spiral', 'golden',
                       'sunflower', 'hilbert', 'gosper', 'peano', 'sierpinski',
                       'concentric',  'hexagon', 'octagramspiral', 'david')
# TODO: Move valid infill patterns into infill file once created, and import value.

class ToolConfig(object):
    """
    Class ToolConfig stores tool parameters for path generation
    """
    def __init__(self):
        self._lineWidth = None
        self._perimeters = None
        self._drawSpeed = None
        self._infillDensity = None
        self._infillPattern = None
        self._infillAngle = None
        self._infillOverlap = None
        self._lowerCommand = None
        self._raiseCommand= None
        self._toolSelect = None

    @property
    def lineWidth(self):
        """Width of drawn line in mm"""
        return self._lineWidth
    @lineWidth.setter
    def lineWidth(self, val):
        if not isinstance(val, float):
            val = float(val)
        if val > 0.0:
            self._lineWidth = val
        else:
            raise ValueError("lineWidth must be greater than 0.0mm!")

    @property
    def perimeters(self):
        """Number of perimeters to be drawn"""
        return self._perimeters
    @perimeters.setter
    def perimeters(self, val):
        if not isinstance(val, int):
            val = int(val)
        if val >= 0:
            self._perimeters = val
        else:
            raise ValueError("perimeters must be 0 or greater!")

    @property
    def drawSpeed(self):
        """Speed of drawing moves in mm/s"""
        return self._drawSpeed
    @drawSpeed.setter
    def drawSpeed(self, val):
        if not isinstance(val, float):
            val = float(val)
        if val > 0.0:
            self._drawSpeed = val
        else:
            raise ValueError("drawSpeed must be greater than 0.0mms!")

    @property
    def infillDensity(self):
        """Density of fill lines in %"""
        return self._infillDensity
    @infillDensity.setter
    def infillDensity(self, val):
        if not isinstance(val, float):
            val = float(val)
        if val >= 0.0 and val <= 100.0:
            self._infillDensity = val
        else:
            raise ValueError("infillDensity must be in range 0.0-100.0 (inclusive)!")

    @property
    def infillPattern(self):
        """Name of desired pattern. Can also """
        return self._infillPattern
    @infillPattern.setter
    def infillPattern(self, val):
        if val.lower() in validInfillPatterns:
            self._infillPattern = val.lower()
        elif os.path.isfile(val):
            self._infillPattern = val
        else:
            raise ValueError("infillPattern not valid or shapeFill image not found!")

    @property
    def infillAngle(self):
        """Angle of infill pattern in degrees"""
        return self._infillAngle
    @infillAngle.setter
    def infillAngle(self, val):
        if not isinstance(val, float):
            val = float(val)
        if val >= -360.0 and val <= 360.0:
            self._infillAngle = val
        else:
            raise ValueError("infillAngle must be in range -360.0-360.0 (inclusive)!")

    @property
    def infillOverlap(self):
        """Overlap of fill lines on innermost perimeter in %"""
        return self._infillOverlap
    @infillOverlap.setter
    def infillOverlap(self, val):
        if not isinstance(val, float):
            val = float(val)
        if val >= 0.0 and val <= 100.0:
            self._infillOverlap = val
        else:
            raise ValueError("infillOverlap must be in range 0.0-100.0 (inclusive)!")

    @property
    def lowerCommand(self):
        """GCode string for machine specific lower command"""
        return self._lowerCommand
    @lowerCommand.setter
    def lowerCommand(self, val):
        if isinstance(val, (str, unicode)):
            self._lowerCommand = val
        else:
            raise ValueError("lowerCommand must be a string!")

    @property
    def raiseCommand(self):
        """GCode string for machine specific raise command"""
        return self._raiseCommand
    @raiseCommand.setter
    def raiseCommand(self, val):
        if isinstance(val, (str, unicode)):
            self._raiseCommand = val
        else:
            raise ValueError("raiseCommand must be a string!")

    @property
    def toolSelect(self):
        """Optional GCode string for machine specific tool selection"""
        return self._toolSelect
    @toolSelect.setter
    def toolSelect(self, val):
        if isinstance(val, (str, unicode)):
            self._toolSelect = val
        else:
            raise ValueError("toolSelect must be a string!")
