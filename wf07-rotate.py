import config
import json
import numpy as np
import math
from PIL import Image

print("Data worflow step 7: Rotation")
print("=========================================")
path = f"{config.path}-small/05-ccrop/c6-224"
path = f"{config.path}/05-ccrop/c6-224"
outpath = f"{config.path}-small/07-rotate/c6-224"
outpath = f"{config.path}/07-rotate/c6-224"

def rotate(die,db,im,angle,flip=False):
    rdie = dict(die)
    rdie["path"] = die["path"].replace("-r0-", f"-rf{angle}-" if flip else f"-r{angle}-")
    db["dies"].append(rdie)
    rim = im.rotate(angle)
    rim.save(rdie["path"])

def transform():
    s = "train"
    print(f"Open {s}/db.json")
    with open(f"{path}/{s}/db.json", "r") as f:
        db = json.loads(f.read())
    db["name"] += "-rotate"
    db["path"] = f"{outpath}/{s}"
    l = list(db["dies"])
    for die in l:
        im = Image.open(die["path"])
        die["path"] = die["path"].replace("/05-ccrop","/07-rotate").replace("-ccrop-","-r0-")
        print(f"Rotating {die['path']}")
        im.save(die["path"])
        for i in range(1,4):
            rotate(die,db,im,i * 90)
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
        for i in range(0, 4):
            rotate(die, db, im, i * 90, True)

    db["count"] = len(db["dies"])
    print(f"Create {db['path']}/db.json")
    with open(f"{db['path']}/db.json", "w") as f:
        f.write(json.dumps(db, indent=4))

transform()
