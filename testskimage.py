import skimage
import skimage.io
import skimage.filters
import skimage.color
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageStat

path = "tests/img/die-0-OSD185236-Brutes-01-01-crop.bmp"
#path = "tests/img/die0-OSD185236-Brutes-01-01-norm-c4-224.bmp"
outpath = "tests/img/OSD185236-Brutes-01-01-sato.png"
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

# plt.imshow(im)
# plt.show()
# skimage.io.imsave(outpath, im)
# pim = Image.fromarray(im)
# pim.show()
# im = skimage.filters.frangi(im) #TOIMPROVE
# plt.imshow(im)
# plt.show()

