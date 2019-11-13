import config

print("DNN sizing")
print("==========")
field = 22
print(f"Field={field}*{field}mm")
res = 1008 * 2
print(f"Global width={res}px")
print(f"Resolution 1px={config.precision}µm")
print(f"NbPixel={int(res * res /1e6)}Mpx")
print(f"Defect size={config.dimDefect}µ")
print(f"DefectCenter size={config.dimDefectCenter}µ")
print(f"Defect={config.sizeDefect}px")
config.sizeDefectcenter = int(config.dimDefectCenter / config.precision)+1
print(f"DefectCenter={config.sizeDefectcenter}px")
res = config.radius * 2
print(f"Resolution after croping={res}px")
resdefectmin = config.sizeMin
print(f"Resolution min for detect defect={resdefectmin}px")
print(f"Resolution min for detect defectCenter={config.sizeMinCenter}px")
print(f"Resolution center zone={config.sizeMinCenter}px")
print(f"Resolution min for detect defectCenter after croping center={config.sizeMinCropingCenter}px")
print(f"Defect resolution with VGG16={224 / resdefectmin:.1f}px")
print(f"Dataset size={config.datasetSize}")
datasetAugmentedNb = config.datasetSize * 5
print(f"Dataset augmented size={datasetAugmentedNb}")
print(f"Test size 20%={config.datasetSize // 5}")
print(f"Training size={datasetAugmentedNb - config.datasetSize // 5}")

print()
print("Main CNN sequential")
print("""VGG:224*224*3|224*224*64|112*112*128|56*56*256|28*28*512|14*14*512|7*7*512|25088|4096|4096|1000
CIFAR10:28*28*1|28*28*32|14*14*64|7*7*64|3136|512|10
CholletSmall:150*150*3|150*150*32|75*75*32|38*38*64|19*19*64|23104|256|1""")

def calcNbW(depths = [32], hiddens = [512], out = 10, input = 224, drop = 2, dim = 3, window = 9):
    res = 0
    prev = 1
    for d in depths:
        res += prev*dim*window*d
        prev = d
    if len(hiddens) > 0:
        res += (input/(drop**len(depths)))**2*prev*hiddens[0]
        prev = hiddens[0]
    for h in hiddens[1:]:
        res += h * prev
        prev = h
    if prev == 1:
        prev = input
    res += out * prev
    return int(res)

def calcNbIter(depths = [32], hiddens = [512], out = 10, input = 224, drop = 2, dim = 3, window = 9):
    res = 0
    prev = 1
    i = 1
    for d in depths:
        res += prev*dim*window*d*(input/i)**2
        i *= drop
        prev = d
    if len(hiddens) > 0:
        res += (input/(drop**len(depths)))**2*prev*hiddens[0]
        prev = hiddens[0]
    for h in hiddens[1:]:
        res += h * prev
        prev = h
    res += out * prev
    return int(res)

def calcNbDays(nbiter, nbsample, epochs=100, freq=3.7, nbcycle=10):
    return (nbiter * nbsample * epochs / (freq * 1e9 / nbcycle)) / (24 * 3600)

def calcGpuRatio(dimcnn = 224, dimmlp = 25088, nbcore = 2048):
    w = nbcore ** 0.5
    k = (dimcnn / w) ** 2
    k2 = dimmlp / w
    return (k ** 2 + k2) ** 0.5

nbcat = 10

print()
print("Transfert strategy with VGG16")
print(f"CVCTDK{16-4}-TCNN-VGG16-KNN-10O")
print(f"    VGG16-CNN|KNN|{nbcat}")
print(f"    Nb days={calcNbDays(config.datasetSize * config.datasetSize * 25088, config.datasetSize, 1, nbcycle=1):.1f}")
print(f"CVCTDK{16-4}-TCNN-VGG16-RF-10O")
print(f"    VGG16-CNN|RF|{nbcat}")
print(f"CVCTDK{16-4}-TCNN-VGG16-XGB-10O")
print(f"    VGG16-CNN|XGB|{nbcat}")
print(f"CVCTDK{16-1}-TCNN-VGG16-2H-10O")
print(f"    VGG16-CNN|4096|1024|{nbcat}")
print(f"    Nb kweigths={int(calcNbW([],[4096,1024],10,25088)/1000)}")
d = calcNbDays(calcNbIter([],[4096,1024],10,25088) / 1000, config.datasetSize)
print(f"    Nb days={d:.1f} GPU={d / calcGpuRatio(1,25088):.1f}")
print(f"CVCTDK{16-1}-TCNN-VGG16-CIFAR10-10O")
print(f"    VGG16-CNN|3136|512|{nbcat}")
print(f"    Nb kweigths={int(calcNbW([],[3136,512],10,25088)/1000)}")
d = calcNbDays(calcNbIter([],[3136,512],10,25088) / 1000, config.datasetSize)
print(f"    Nb days={d:.1f} GPU={d / calcGpuRatio(1,25088):.1f}")
print(f"CVCTDK{16-2}-TCNN-VGG16-1H-10O")
print(f"    VGG16-CNN|512|{nbcat}")
print(f"    Nb kweigths={int(calcNbW([],[512],10,25088)/1000)}")
d = calcNbDays(calcNbIter([],[512],10,25088) / 1000, config.datasetSize)
print(f"    Nb days={d:.1f} GPU={d / calcGpuRatio(1,25088):.1f}")

print()
print("Transfert strategy with NASNet")
print(f"CVCTDK1040-TCNN-NASNet-KNN-10O")
print(f"    NASNet-CNN|KNN|{nbcat}")
print(f"    Nb days={calcNbDays(config.datasetSize * config.datasetSize * 1056, config.datasetSize, 1, nbcycle=1):.1f}")
print(f"CVCTDK1040-TCNN-NASNet-RF-10O")
print(f"    NASNet-CNN|RF|{nbcat}")
print(f"CVCTDK1040-TCNN-NASNet-XGB-10O")
print(f"    NASNet-CNN|XGB|{nbcat}")
print(f"CVCTDK{1041+1}-TCNN-NASNet-1H-10O")
print(f"    NASNet-CNN|1024|{nbcat}")
print(f"    Nb kweigths={int(calcNbW([],[1024],10,1056)/1000)}")
d = calcNbDays(calcNbIter([],[1024],10,1056) / 1000, config.datasetSize)
print(f"    Nb days={d:.1f} GPU={d / calcGpuRatio(1,1056):.1f}")

print()
print("Transfert strategy with InceptionResNet")
print(f"CVCTDK781-TCNN-NASNet-KNN-10O")
print(f"    NASNet-CNN|KNN|{nbcat}")
print(f"    Nb days={calcNbDays(config.datasetSize * config.datasetSize * 1056, config.datasetSize, 1, nbcycle=1):.1f}")
print(f"CVCTDK781-TCNN-NASNet-RF-10O")
print(f"    NASNet-CNN|RF|{nbcat}")
print(f"CVCTDK781-TCNN-NASNet-XGB-10O")
print(f"    NASNet-CNN|XGB|{nbcat}")
print(f"CVCTDK{782+1}-TCNN-NASNet-1H-10O")
print(f"    NASNet-CNN|1000|{nbcat}")
print(f"    Nb kweigths={int(calcNbW([],[1000],10,1536)/1000)}")
d = calcNbDays(calcNbIter([],[1000],10,1536) / 1000, config.datasetSize)
print(f"    Nb days={d:.1f} GPU={d / calcGpuRatio(1,1536):.1f}")

print()
print("Alternative with NasNetMobile(771)")
print("Alternative with MobileNet_V2(157) but input 112x112")

def calcInputSizeBottleneck(res, nbBottleneck, drop = 2):
    return (int(resdefectmin / drop ** nbBottleneck) + 1) * drop ** nbBottleneck

inputcnn4bottleneck = calcInputSizeBottleneck(resdefectmin, 4, 2)
j = inputcnn4bottleneck
print()
print("Full strategy")
print(f"CVCTDK{1+4*2+1+3+1}-CNN-{j}x2I-4BN-2H-10O")
j = min(j, calcInputSizeBottleneck(resdefectmin, 3, 2))
print(f"    {j}*{j}*2|{j}*{j}*32|{int(j/2)}*{int(j/2)}*64|{int(j/4)}*{int(j/4)}*128|{int(j/8)}*{int(j/8)}*256|{int(j/16)}*{int(j/16)}*256|{int((j/16)**2*256)}|4096|1024|{nbcat}")
print(f"    Nb kweigths={int(calcNbW([32,64,128,256],[4096,1024],10,j,2,2)/1000)}")
d = calcNbDays(calcNbIter([32,64,128,256],[4096,1024],10,j,2,2), config.datasetSize)
print(f"    Nb days={d:.1f} GPU={d / calcGpuRatio(j,12000):.1f}")
print(f"CVCTDK{1+3*2+1+1+1}-CNN-{j}x2I-3BN-1H-10O")
print(f"    {j}*{j}*2|{j}*{j}*32|{int(j/2)}*{int(j/2)}*32|{int(j/4)}*{int(j/4)}*64|{int(j/8)}*{int(j/8)}*64|{int((j/8)**2*64)}|1024|{nbcat}")
print(f"    Nb kweigths={int(calcNbW([32,32,64],[1024],10,j,2,2)/1000)}")
d = calcNbDays(calcNbIter([32,32,64],[1024],10,j,2,2), config.datasetSize)
print(f"    Nb days={d:.1f} GPU={d / calcGpuRatio(j,12000):.1f}")
j = min(j, int((resdefectmin / 2 ** 2) / 3) * 2 ** 2 * 3)
print(f"CVCTDK{1+3*2+1+1+1}-CNN-{j}x2I-3BN-DROPLAST3-1H-10O")
print(f"    {j}*{j}*2|{j}*{j}*32|{int(j/2)}*{int(j/2)}*32|{int(j/4)}*{int(j/4)}*64|{int(int(j/4)/3)}*{int(int(j/4)/3)}*64|{int(int(int(j/4)/3)**2*64)}|512|{nbcat}")
print(f"    Nb kweigths={int((2*3*3*32+32*2*3*3*32+32*2*3*3*64+int(int(int(j/4)/3)**2*64)*512+512*10)/1000)}")
inputcnn2bottleneck = calcInputSizeBottleneck(resdefectmin, 2, 3)
j = min(inputcnn2bottleneck, j)
print(f"CVCTDK{1+2*2+1+1+1}-CNN-{j}x2I-2BN-DROP3-1H-10O")
print(f"    {j}*{j}*2|{j}*{j}*32|{int(j/3)}*{int(j/3)}*64|{int(j/9)}*{int(j/9)}*64|{int((j/9)**2*64)}|512|{nbcat}")
print(f"    Nb kweigths={int(calcNbW([32,64],[512],10,j,3,2)/1000)}")
d = calcNbDays(calcNbIter([32,64],[512],10,j,3,2), config.datasetSize)
print(f"    Nb days={d:.1f} GPU={d / calcGpuRatio(j,9000):.1f}")
print(f"CVCTDK{1+2*2+1+1+1}-CNN-{j}I-2BN-DROP3-1H-10O")
print(f"    {j}*{j}*1|{j}*{j}*32|{int(j/3)}*{int(j/3)}*64|{int(j/9)}*{int(j/9)}*64|{int((j/9)**2*64)}|256|{nbcat}")
print(f"    Nb kweigths={int(calcNbW([32,64],[256],10,j,3,1)/1000)}")
d = calcNbDays(calcNbIter([32,64],[256],10,j,3,1), config.datasetSize)
print(f"    Nb days={d:.1f} GPU={d / calcGpuRatio(j,9000):.1f}")
j = resdefectmin
print(f"CVCTDK{1+2+1}-MLP-{(j**2)*1}I-1H-10O") # Concat layers not flatten
print(f"    {j}*{j}*1|{(j**2)*1}|128|{nbcat}")
print(f"    Nb kweigths={int(calcNbW([],[128],10,(j**2)*1)/1000)}")
d = calcNbDays(calcNbIter([],[128],10,(j**2)*1), config.datasetSize)
print(f"    Nb days={d:.1f} GPU={d / calcGpuRatio(j,(j**2)*2):.1f}")

print()
print("From scratch strategy alternative with MobileNetV2 like")
print("Replace Flatten by GlobaleAveragePool 7x7 to obtain 1280 outputs")

print()
print("Mosaic strategy")
dimmosaic = 4
datasetsize = 1000
print(f"Mosaic width={dimmosaic}")
print(f"Nb mosaic={dimmosaic ** 2}")
print(f"Nb stride={(dimmosaic * 2 - 1) ** 2}")
res = int(res / dimmosaic)
print(f"Mosaic resolution={res}px")
resdefectmin = int(res / config.sizeDefectcenter)+1
print(f"Resolution min for detect defect={resdefectmin}px")
inputcnn2bottleneck = int(resdefectmin / 2 ** 2) * 2 ** 2
j = inputcnn2bottleneck
print()
print("Transfert strategy like no mosaic")
print("From scratch strategy")
print(f"CVCTDK{1+2*2+1+1+1}-CNN-{j}I-2BN-1H-10O")
print(f"    {j}*{j}*1|{j}*{j}*32|{int(j/2)}*{int(j/2)}*64|{int(j/4)}*{int(j/4)}*64|{int((j/4)**2*64)}|512|{nbcat}")
print(f"    Nb kweigths={int(calcNbW([32,64],[512],10,j,2,1)/1000)}")
d = calcNbDays(calcNbIter([32,64],[512],10,j,2,1), config.datasetSize * 16)
print(f"    Nb days={d:.1f} GPU={d / calcGpuRatio(j,5000):.1f}")
inputcnn2bottleneck = calcInputSizeBottleneck(resdefectmin, 2, 2)
j = min(j, inputcnn2bottleneck)
print(f"CVCTDK{1+2*2+1+1+1}-CNN-{j}I-2BN-DROP3-1H-10O")
print(f"    {j}*{j}*1|{j}*{j}*32|{int(j/3)}*{int(j/3)}*64|{int(j/9)}*{int(j/9)}*64|{int((j/9)**2*64)}|512|{nbcat}")
print(f"    Nb kweigths={int(calcNbW([32,64],[512],10,j,3,1)/1000)}")
d = calcNbDays(calcNbIter([32,64],[512],10,j,3,1), config.datasetSize * 16)
print(f"    Nb days={d:.1f} GPU={d / calcGpuRatio(j,5000):.1f}")
inputcnn1bottleneck = calcInputSizeBottleneck(resdefectmin, 2, 3)
j = min(j, inputcnn1bottleneck)
print(f"CVCTDK{1+1*2+1+1+1}-CNN-{j}I-1BN-DROP3-1H-10O")
print(f"    {j}*{j}*1|{j}*{j}*32|{int(j/3)}*{int(j/3)}*32|{int((j/3)**2*32)}|512|{nbcat}")
print(f"    Nb kweigths={int(calcNbW([32],[512],10,j,3,1)/1000)}")
d = calcNbDays(calcNbIter([32],[512],10,j,3,1), config.datasetSize * 16)
print(f"    Nb days={d:.1f} GPU={d / calcGpuRatio(j,5000):.1f}")
j = min(j, calcInputSizeBottleneck(resdefectmin, 1, 4))
print(f"CVCTDK{1+1*2+1+1+1}-CNN-{j}I-1BN-DROP4-1H-10O")
print(f"    {j}*{j}*1|{j}*{j}*32|{int(j/4)}*{int(j/4)}*32|{int((j/4)**2*32)}|256|{nbcat}")
print(f"    Nb kweigths={int(calcNbW([32],[256],10,j,4,1)/1000)}")
d = calcNbDays(calcNbIter([32],[256],10,j,4,1), config.datasetSize * 16)
print(f"    Nb days={d:.1f} GPU={d / calcGpuRatio(j,5000):.1f}")
j = resdefectmin
print(f"CVCTDK{1+2+1}-MLP-{(j**2)*1}I-1H-10O") # Concat layers not flatten
print(f"    {j}*{j}*1|{(j**2)*1}|128|{nbcat}")
print(f"    Nb kweigths={int(calcNbW([],[128],10,(j**2)*1)/1000)}")
d = calcNbDays(calcNbIter([],[128],10,(j**2)*1), config.datasetSize)
print(f"    Nb days={d:.1f} GPU={d / calcGpuRatio(j,(j**2)*2):.1f}")

print()
print("For info VGG16")
j = 224
print(f"VGG16-{1+5*2+1+3+1}-CNN-{j}x3I-5BN-2H-1000O")
print(f"    {j}*{j}*2|{j}*{j}*32|{int(j/2)}*{int(j/2)}*64|{int(j/4)}*{int(j/4)}*128|{int(j/8)}*{int(j/8)}*256|{int(j/16)}*{int(j/16)}*256|{int(j/32)}*{int(j/32)}*512|{int((j/32)**2*512)}|4096|4096|1024|1000")
print(f"    Nb kweigths={int((3*3*3*32+32*3*3*3*64+64*3*3*3*128+128*3*3*3*256+256*3*3*3*512+int((j/32)**2*512)*4096+4096*4096+4096*1024+1024*1000)/1000)}")
print(f"    Nb Miter={int((3*3*3*32*j**2+32*3*3*3*64*(j/2)**2+64*3*3*3*128*(j/4)**2+128*3*3*3*256*(j/8)**2+256*3*3*3*512*(j/16)**2+int((j/32)**2*512)*4096+4096*4096+4096*1024+1024*1000)/1000000)}")
d = calcNbDays(calcNbIter([32,64,128,256,512],[4096,4096,1024],1000), 10000)
print(f"    Nb days={int(d)} GPU={int(d / calcGpuRatio())}")

vgg16w = calcNbW([32,64,128,256,512],[4096,4096,1024],1000)
print(f"    Verif nb weigths={vgg16w}")
vgg16i = calcNbIter([32,64,128,256,512],[4096,4096,1024],1000)
print(f"    Verif nb iter={vgg16i}")
