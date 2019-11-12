from PIL import Image, ImageEnhance, ImageStat

path= "../usbkey.bak/OSD185236/01/OSD185236-Brutes-01-01.bmp"

with Image.open(path) as im:
    print(f"size: {im.size}")
    print(f"format: {im.format}")
    print(f"mode: {im.mode}")
    stat = ImageStat.Stat(im)
    print(f"lum: {stat.mean}")
    print(f"std: {stat.stddev}")
