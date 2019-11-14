import config
import json
import numpy as np
import math
from PIL import Image

print("Data worflow step 6 BETA: RGB")
print("=============================")
path = f"{config.path}-small/04-norm"
#path = f"{config.path}/04-norm/c4-224"
rpath = "c4-224"
gpath = "c10-224"
bpath = "center-c4-224"
outpath = f"{config.path}-small/06-rvg"
#outpath = f"{config.path}/06-rvg"

def ccrop(im, radius = config.radiusGlobal):
    radius = radius * im.size[0] * 0.5 / config.radius
    for x in range(im.size[0]):
        for y in range(im.size[1]):
            d = math.sqrt(((x - (im.size[0] / 2)) ** 2) + ((y - (im.size[1] / 2)) ** 2))
            if d > radius:
                im.putpixel((x, y), (0,0,0))

def transform(s="train", inv = True):
    print(f"Open {s}/db.json")
    db = None
    with open(f"{path}/{rpath}/{s}/db.json", "r") as f:
        db = json.loads(f.read())
    print(db)
    db["name"] += "-rvg"
    db["path"] = f"{outpath}/{rpath}/{s}"
    rdies = db["dies"]
    with open(f"{path}/{gpath}/{s}/db.json", "r") as f:
        gdies = json.loads(f.read())
        gdies = gdies["dies"]
    with open(f"{path}/{bpath}/{s}/db.json", "r") as f:
        bdies = json.loads(f.read())
        bdies = bdies["dies"]
    dies = zip(rdies, gdies, bdies)
    for t in dies:
        die = t[0]
        im = Image.open(die["path"])
        img =  Image.open(t[1]["path"])
        imb = Image.open(t[2]["path"])
        im = Image.merge("RGB", (im, img, imb))
        if inv:
            im = im.point(lambda x : 255- x)
        ccrop(im)
        die["path"] = die["path"].replace("/04-norm","/06-rvg").replace("-norm-","-rvg-")
        print(f"Creating {die['path']}")
        im.save(die["path"])

    print(f"Create {db['path']}/db.json")
    with open(f"{db['path']}/db.json", "w") as f:
        f.write(json.dumps(db, indent=4))

transform("test")
transform("train")
