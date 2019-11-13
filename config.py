precision = 11 #µm/px
radiusGlobal = 1008 #px
radius = 875 #px
radiusCenter = 261 #px
dimDefect = 150 #µm
dimDefectCenter = 75 #µm
sizeDefect = (dimDefect // precision) + 1 #px
sizeDefectCenter = (dimDefectCenter // precision) + 1 #px
sizeMin = (radius * 2 // sizeDefect) + 1 #px
sizeMinCenter = (radius * 2 // sizeDefectCenter) + 1 #px
sizeMinCropingCenter = (radiusCenter * 2 // sizeDefectCenter) + 1 #px
datasetSize = 1430
datasetTestSize = int(datasetSize * 0.2)
datasetTrainSize = datasetSize - datasetTestSize
path = "img"

if __name__ == '__main__':
    print(f"radius: {radius}px")
    print(f"sizeDefect: {sizeDefect}px")
    print(f"sizeDefectCenter: {sizeDefectCenter}px")
    print(f"sizeMin: {sizeMin}px")
    print(f"sizeMinCenter: {sizeMinCenter}px")
    print(f"sizeMinCropingCenter: {sizeMinCropingCenter}px")
    print(f"datasetNb: {datasetSize}")



