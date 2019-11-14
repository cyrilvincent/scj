import config
import json
import math
from PIL import Image
import skimage.filters
import skimage.io
import matplotlib.pyplot as plt
import os

print("Data worflow step 5.2: Sato")
print("===========================")
path = f"{config.path}-small/03-split"
path = f"{config.path}/03-split"
outpath = f"{config.path}-small/05-ccrop/sato-224"
outpath = f"{config.path}/05-ccrop/sato-224"

def ccrop(im, radius = config.radiusGlobal):
    radius = radius * im.size[0] * 0.5 / config.radius
    for x in range(im.size[0]):
        for y in range(im.size[1]):
            d = math.sqrt(((x - (im.size[0] / 2)) ** 2) + ((y - (im.size[1] / 2)) ** 2))
            if d > radius:
                im.putpixel((x, y), 0)

def transform(s="train", size=224, method = Image.LANCZOS):
    print(f"Open {s}/db.json")
    with open(f"{path}/{s}/db.json", "r") as f:
        db = json.loads(f.read())
    print(db)
    db["size"] = size
    db["name"] += "-sato"
    db["path"] = f"{outpath}/{s}"
    db["format"] = "PNG"
    for die in db["dies"]:
        im = skimage.io.imread(die["path"], as_gray=True)
        im = skimage.filters.sato(im)
        plt.imshow(im)
        plt.imsave(db["path"]+"/temp.png", im)
        im = Image.open(db["path"]+"/temp.png")
        im = im.convert("L")
        im = im.point(lambda x: 0 if x < 52 else 255)
        ccrop(im, config.radiusGlobal - 20)
        die["path"] = die["path"].replace("/02-crop/",f"/05-ccrop/sato-224/{s}/").replace("-crop.bmp","-sato.png")
        print(f"Creating {die['path']}")
        im = im.resize((size,size),method)
        pxs = list(im.getdata())
        nb255px = len([p for p in pxs if p > 127])
        die["sato52"] = nb255px
        im.save(die["path"])


    db["sato52"] = int(sum([d["sato52"] for d in db["dies"]]) / len(db["dies"]))
    os.remove(db["path"] + "/temp.png")
    print(f"Create {db['path']}/db.json")
    with open(f"{db['path']}/db.json", "w") as f:
        f.write(json.dumps(db, indent=4))

transform("test")
transform("train")
