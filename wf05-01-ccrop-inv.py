import config
import json
import math
from PIL import Image

print("Data worflow step 5: CircleCrop & inverse")
print("=========================================")
path = f"{config.path}-small/04-norm/c4-224"
path = f"{config.path}/04-norm/c4-224"
outpath = f"{config.path}-small/05-ccrop/c4-224"
outpath = f"{config.path}/05-ccrop/c4-224"

def ccrop(im, radius = config.radiusGlobal):
    radius = radius * im.size[0] * 0.5 / config.radius
    for x in range(im.size[0]):
        for y in range(im.size[1]):
            d = math.sqrt(((x - (im.size[0] / 2)) ** 2) + ((y - (im.size[1] / 2)) ** 2))
            if d > radius:
                im.putpixel((x, y), 0)

def transform(s="train", inv = True):
    print(f"Open {s}/db.json")
    with open(f"{path}/{s}/db.json", "r") as f:
        db = json.loads(f.read())
    print(db)
    db["name"] += "-ccrop"
    db["path"] = f"{outpath}/{s}"
    for die in db["dies"]:
        im = Image.open(die["path"])
        if inv:
            im = im.point(lambda x : 255- x)
        ccrop(im)
        die["path"] = die["path"].replace("/04-norm","/05-ccrop").replace("-norm-","-ccrop-")
        print(f"Creating {die['path']}")
        im.save(die["path"])

    print(f"Create {db['path']}/db.json")
    with open(f"{db['path']}/db.json", "w") as f:
        f.write(json.dumps(db, indent=4))

transform("test")
transform("train")
