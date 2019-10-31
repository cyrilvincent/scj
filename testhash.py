import hashlib
import os
import base64
import json
from PIL import Image, ImageEnhance, ImageStat
import math

image = (2500, 2000)
nbPix = image[0]*image[1]
print(nbPix)
res = 13E-6
inceptioninput = 299
vgg16iput = 224

ratio = image[0]*0.9/vgg16iput
print(ratio) #10
precision = res * ratio
print(precision) #130µ
# Center precision
print((image[0]*0.5 / vgg16iput) * res) #73µ

h = hashlib.blake2b(b"Hello World")
print(h.hexdigest())
l = os.listdir("img")
print(l)
i = 0
db = []
for e in l:
    if os.path.isfile(f"img/{e}"):
        o = {"id": i, "name": e}
        i += 1
        with open(f"img/{e}", "rb") as f:
            b = f.read()
            h = hashlib.blake2b(b)
            d = h.digest()
            b64 = base64.b64encode(d).decode('utf-8')
            o["h"] = b64
        with Image.open(f"img/{e}") as im:
            o["size"] = im.size
            o["format"] = im.format
            o["mode"] = im.mode
            region = im.crop((100,100,im.size[0]-100,im.size[1]-100))
            region.save(f"img/crop/{e}")
            out = region.transpose(Image.FLIP_LEFT_RIGHT)
            out.save(f"img/crop/flr-{e}")
            out = region.transpose(Image.FLIP_TOP_BOTTOM)
            out.save(f"img/crop/ftb-{e}")
            out = region.transpose(Image.ROTATE_90)
            out.save(f"img/crop/r90-{e}")
            out = region.transpose(Image.ROTATE_180)
            out.save(f"img/crop/r180-{e}")
            out = region.transpose(Image.ROTATE_270) # On peut ajouter TRANSPOSE (diag) TRANSVERSE (oppositediag) FLIP+ROTATE
            # TRANSPOSE ET TRANSERVE ne servent à rien : https://github.com/python-pillow/Pillow/blob/master/Tests/test_image_transpose.py
            out.save(f"img/crop/r270-{e}")
            ie = ImageEnhance.Contrast(region)
            out = ie.enhance(1.3)
            out.save(f"img/crop/contrast30-{e}")
            ie = ImageEnhance.Brightness(region)
            out = ie.enhance(1.2)
            out.save(f"img/crop/bright20-{e}")
            b20 = out
            out = region.convert("L")
            out.save(f"img/crop/gray-{e}")

            gray = out

            stat = ImageStat.Stat(region)
            o["lum"] = stat.mean[0]
            o["std"] = stat.stddev[0]
            #stat = ImageStat.Stat(b20)
            #o["lum2"] = stat.mean[0]

            out = region.point(lambda i: i * 1.2)
            out.save(f"img/crop/dot12-{e}")
            #stat = ImageStat.Stat(out)
            #o["lum3"] = stat.mean[0]

            stat = ImageStat.Stat(region)
            out = region.point(lambda i: int(min(255, max(0, i - stat.mean[0] + 128))))
            out.save(f"img/crop/lcenter-{e}")

            stat = ImageStat.Stat(region)
            out = region.point(lambda i: min(255, int(max(0, ((i - stat.mean[0]) / stat.stddev[0]) * (128 / 3) + 128))))
            out.save(f"img/crop/autoadjust-{e}")

            r = 300
            for x in range(im.size[0]):
                for y in range(im.size[1]):
                    d = math.sqrt(((x - (im.size[0] / 2)) ** 2) + ((y - (im.size[1] / 2)) ** 2))
                    if d > r:
                        im.putpixel((x, y), (255,255,255))
            im.save(f"img/crop/circlecrop-{e}")

            out = gray.point(lambda i : 255 - i)
            out.save(f"img/crop/inv-{e}")

            stat = ImageStat.Stat(gray)
            contrast = gray.point(lambda i: min(255, int(max(0, ((i - stat.mean[0])*1.5)))))
            contrast.save(f"img/crop/contrast50-{e}")

            zoom = gray.crop(((gray.size[0] / 2) - 100, (gray.size[1] / 2) - 100, (gray.size[0] / 2) + 100, (gray.size[1] / 2) + 100))
            zoom = zoom.resize(gray.size)
            zoom.save(f"img/crop/zoom-{e}")

            merge = Image.merge("RGB", (gray, contrast, zoom))
            merge.save(f"img/crop/merge-{e}")



        db.append(o)

l = os.listdir("img/crop")
for e in l:
    if os.path.isfile(f"img/crop/{e}"):
        o = {"id": i, "name": e}
        i += 1
        with open(f"img/crop/{e}", "rb") as f:
            b = f.read()
            h = hashlib.blake2b(b)
            d = h.digest()
            b64 = base64.b64encode(d).decode('utf-8')
            o["h"] = b64
        with Image.open(f"img/crop/{e}") as im:
            o["size"] = im.size
            o["format"] = im.format
            o["mode"] = im.mode
            stat = ImageStat.Stat(im)
            o["lum"] = stat.mean[0]
            o["std"] = stat.stddev[0]
            stat = ImageStat.Stat(b20)
        db.append(o)

print(db)
with open("db.json", "w") as f:
    f.write(json.dumps(db, indent=4))

#https://stackoverflow.com/questions/51995977/how-can-i-use-a-pre-trained-neural-network-with-grayscale-images

#https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/71179
#https://www.guru99.com/keras-tutorial.html

#https://blog.octo.com/classification-dimages-les-reseaux-de-neurones-convolutifs-en-toute-simplicite/
#https://www.rsipvision.com/wafer-macro-defects-detection-classification/
#https://www.mdpi.com/2076-3417/9/3/597/htm

#https://blog.octo.com/classification-dimages-les-reseaux-de-neurones-convolutifs-en-toute-simplicite/
#http://www.datacorner.fr/xgboost/
#http://cs231n.github.io/transfer-learning/
#https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

#Code de Francois Chollet
#Little CNN avec 2 categories : https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
#BottleNeeck strategie
#VGG16 + Petit Dense https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
#Fine Tuning SDG + LR + Momentum
#https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975



