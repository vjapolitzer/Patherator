"""
Patherator Image Tools Module
@author: Vincent Politzer <https://github.com/vjapolitzer>

This module contains various image processing utility functions.
"""
import numpy as np
from PIL import Image
from skimage.restoration import denoise_bilateral
from sklearn.cluster import KMeans
from sklearn import metrics
# import skimage.io as skio

def brightness(color):
    """
    Returns the luminance of a color in RGB form
    """
    return 0.2126*color[0] + 0.7152*color[1] + 0.0722*color[2]

def getNumColors(path):
    """
    Detect the number of colors in the supplied image
    (Only up to 8 colors not including the background)
    NOTE: This feature is experimental and may not work
    well for ALL images
    """
    im = Image.open(path)

    # Resize to reduce processing time
    w, h = im.size
    wSmall = int(66 * w / max(w, h))
    hSmall = int(66 * h / max(w, h))
    im = im.resize((wSmall, hSmall))

    # Convert into numpy data structure
    im = im.convert('RGB')
    im = np.array(im)

    # Filter to remove non-unique colors
    # This sequence of filters was experimentally determined
    # And may not work well for ALL images
    for i in range(10):
        im = denoise_bilateral(im, sigma_color=0.025, sigma_spatial=4, multichannel = True)
    for i in range(5):
        im = denoise_bilateral(im, sigma_color=0.035, sigma_spatial=4, multichannel = True)
    for i in range(3):
        im = denoise_bilateral(im, sigma_color=0.05, sigma_spatial=4, multichannel = True)
    for i in range(2):
        im = denoise_bilateral(im, sigma_color=0.06, sigma_spatial=4, multichannel = True)

    # skio.imshow(im)
    # skio.show()
    # return

    # Reshape into a list of pixels
    imArray = im.reshape((im.shape[0] * im.shape[1], 3))

    bestSilhouette = -1
    bestNumClusters = 0
    for numClusters in range(2,9):
        # Cluster the colors
        clt = KMeans(n_clusters = numClusters)
        clt.fit(imArray)

        # Calculate Silhouette Coefficient
        silhouette = metrics.silhouette_score(imArray, clt.labels_, metric = 'euclidean')

        # Find the best one
        if silhouette > bestSilhouette:
            bestSilhouette = silhouette
            bestNumClusters = numClusters

    return bestNumClusters - 1 # Subtract one: background color is not included

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
