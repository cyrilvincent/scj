from PIL import Image, ImageStat, ImageEnhance, ImageFilter
import numpy as np
import math

radius = 875

path = "tests/img/OSD185236-Analysees-01-01.jpg"
path = "tests/img/OSD185236-Brutes-01-01.bmp"
im = Image.open(path)
print(im.mode)
#im.show()

im =  im.crop(((im.size[0] / 2) - radius, (im.size[1] / 2) - radius, (im.size[0] / 2) + radius, (im.size[1] / 2) + radius))
#im.show()

stat = ImageStat.Stat(im)
lum = stat.mean[0]
std = stat.stddev[0]
min2 = stat.extrema[0][0]
max2 = stat.extrema[0][1]
print(lum, std, min2, max2) #140.8660502857143 10.847677919700216 104 208

norm = im.point(lambda x : np.clip(x - lum + 127, 0, 255))
stat = ImageStat.Stat(norm)
lum = stat.mean[0]
std = stat.stddev[0]
min2 = stat.extrema[0][0]
max2 = stat.extrema[0][1]
pxs = list(norm.getdata())
nb0px = len([p for p in pxs if p == 0])
print(lum, std, min2, max2, nb0px)
#norm.show()

# for c in range(1,15):
#     ie = ImageEnhance.Contrast(norm)
#     c1 = ie.enhance(c)
#     stat = ImageStat.Stat(c1)
#     lum = stat.mean[0]
#     std = stat.stddev[0]
#     min = stat.extrema[0][0]
#     max = stat.extrema[0][1]
#     pxs = list(c1.getdata())
#     nb0px = len([p for p in pxs if p == 0])
#     print(c,lum,std,min,max,nb0px)
#     c1.show()
#
#

ie = ImageEnhance.Contrast(norm)
c10 = ie.enhance(10)
stat = ImageStat.Stat(c10)
lum = stat.mean[0]
std = stat.stddev[0]
min2 = stat.extrema[0][0]
max2 = stat.extrema[0][1]
pxs = list(c10.getdata())
nb0px = len([p for p in pxs if p == 0]) #4:458 5:9403 6:33295
print(lum, std, min2, max2, nb0px)
#c10.show()

ie = ImageEnhance.Contrast(norm)
c4 = ie.enhance(4)

radius = 400
im2 =  im.crop(((im.size[0] / 2) - radius, (im.size[1] / 2) - radius, (im.size[0] / 2) + radius, (im.size[1] / 2) + radius))
im2 = im2.resize((100,100), Image.LANCZOS) # super
#im2 = im2.resize((500,500), Image.NEAREST)
ie = ImageEnhance.Contrast(im2)
im2 = ie.enhance(40)
pxs = list(im2.getdata())
nb0px = len([p for p in pxs if p == 0])
#im2 = im2.point(lambda x : 255 if x == 0 else x)
print(nb0px)
#im2.show() # resulat pas terrible

imload = im2.load()
for x in range(1,99):
    for y in range(1,99):
        if imload[x,y] == 0:
            # imload[x,y] = min(imload[x - 1, y],
            #                    imload[x + 1, y],
            #                    imload[x - 1, y - 1],
            #                    imload[x + 1, y + 1],
            #                    imload[x - 1, y + 1],
            #                    imload[x + 1, y - 1],
            #                    imload[x, y - 1],
            #                    imload[x, y + 1]
            #                    )
            imload[x,y] = min(imload[x + 1, y],
                               imload[x + 2, y],
                               imload[x + 3, y],
                               imload[x + 4, y]
                               )
            if imload[x, y] !=0:
                imload[x, y] = 255
pxs = list(im2.getdata())
nb0px = len([p for p in pxs if p == 0])
print(nb0px)
#im2.show()

stdmean = 11.65
stsstd = 2.17
stdmax = 21.42
print(127 / stdmean, 127/stdmax)
stdtestboucle = 4
# Tester avec 4, 6 et 10

radius = 990
for x in range(im.size[0]):
    for y in range(im.size[1]):
        d = math.sqrt(((x - (im.size[0] / 2)) ** 2) + ((y - (im.size[1] / 2)) ** 2))
        if d > radius:
            im.putpixel((x, y), (0))
#c4.show()
