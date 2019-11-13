from PIL import Image

radius = 875

path = "tests/img/OSD185236-Analysees-01-01.jpg"
im = Image.open(path)
im.show()

im =  im.crop(((im.size[0] / 2) - radius, (im.size[1] / 2) - radius, (im.size[0] / 2) + radius, (im.size[1] / 2) + radius))
im.show()

lum = 132.05
std = 11.65
