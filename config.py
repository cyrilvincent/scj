precision = 11 #µm/px
rayonGlobal = 1008 #px
rayon = 876 #px
rayonCenter = 261 #px
dimDefect = 150 #µm
dimDefectCenter = 75 #µm
sizeDefect = (dimDefect // precision) + 1 #px
sizeDefectCenter = (dimDefectCenter // precision) + 1 #px
sizeMin = (rayon * 2 // sizeDefect) + 1 #px
sizeMinCenter = (rayon * 2 // sizeDefectCenter) + 1 #px
sizeMinCropingCenter = (rayonCenter * 2 // sizeDefectCenter) + 1 #px
datasetNb = 5 * 11 * 26
path = "img"

if __name__ == '__main__':
    print(f"rayon: {rayon}px")
    print(f"sizeDefect: {sizeDefect}px")
    print(f"sizeDefectCenter: {sizeDefectCenter}px")
    print(f"sizeMin: {sizeMin}px")
    print(f"sizeMinCenter: {sizeMinCenter}px")
    print(f"sizeMinCropingCenter: {sizeMinCropingCenter}px")
    print(f"datasetNb: {datasetNb}")



