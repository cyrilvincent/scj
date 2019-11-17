import skimage
import skimage.io
import skimage.filters
import skimage.color
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageStat

path = "tests/img/die-0-OSD185236-Brutes-01-01-crop.bmp"
#path = "tests/img/OSD185247-Brutes-07-05.bmp"
outpath = "tests/img/OSD185236-Brutes-01-01-sato.png"
#outpath = "tests/img/OSD185247-Brutes-07-05-sato.png"
# pim = Image.open(path)
# im = pim.getdata()
im = skimage.io.imread(path,as_gray=True)
# im = skimage.filters.meijering(im) #OK
# plt.imshow(im)
# plt.show()
im = skimage.filters.sato(im) #OK++
plt.imshow(im)
plt.show()
plt.imsave(outpath, im)
im = Image.open(outpath)
im = im.convert("L")
stat = ImageStat.Stat(im)
median = stat.median[0]
print(median)
#im = im.point(lambda x : np.clip(x - lum + 127, 0, 255))
im = im.point(lambda x : 0 if x < 52 else 255)
im.show()

# Detect corner
import skimage.feature
harris = skimage.feature.corner_harris(im)
res = skimage.feature.corner_peaks(harris)
print(res)
print(len(res))


