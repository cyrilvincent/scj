import config
import json
import skimage.filters
import skimage.io
import os
import skimage.feature

print("Data worflow step 5.3: Harris")
print("=============================")
path = f"{config.path}-small/05-ccrop/sato-224"
path = f"{config.path}/05-ccrop/sato-224"

def compute(s="train"):
    print(f"Open {s}/db.json")
    with open(f"{path}/{s}/db.json", "r") as f:
        db = json.loads(f.read())
    print(db)
    for die in db["dies"]:
        im = skimage.io.imread(die["path"], as_gray=True)
        harris = skimage.feature.corner_harris(im)
        res = skimage.feature.corner_peaks(harris)
        die["harris"] = min(len(res), die["sato52"])
    db["harris"] = int(sum([d["harris"] for d in db["dies"]]) / len(db["dies"]))
    print(f"Create {path}/{s}/db.json")
    with open(f"{path}/{s}/db.json", "w") as f:
        f.write(json.dumps(db, indent=4))

compute("test")
compute("train")
