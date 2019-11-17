import config
import json
import csv
from PIL import Image

print("Data worflow step 8: Make ML.csv")
print("================================")
path = f"{config.path}-small/05-ccrop/c6-28"
path = f"{config.path}/05-ccrop/c6-28"
dbpath = f"{config.path}-small/05-ccrop/sato-224"
dbpath = f"{config.path}/05-ccrop/sato-224"
outpath = f"{config.path}-small/08-ml"
outpath = f"{config.path}/08-ml"

def makecsv(s = "train"):
    print(f"Open {dbpath}/{s}/db.json")
    with open(f"{dbpath}/{s}/db.json", "r") as f:
        db = json.loads(f.read())
    l = list(db["dies"])
    with open(f"{path}/{s}/db.json", "r") as f:
        db2 = json.loads(f.read())
    l2 = list(db2["dies"])
    for die, die2 in zip(l,l2):
        im = Image.open(die2["path"])
        pxs = list(im.getdata())
        for i, px in enumerate(pxs):
            die[f"px{i}"] = px
    file = f"{outpath}/ML-{s}.csv"
    print(f"Create {file}")
    with open(file, "w", newline='') as f:
        keys = ["id", "lot", "wafer", "die", "y", "lum", "lummin", "lummax", "std", "sato52", "harris"]
        for i in range(28*28):
            keys.append(f"px{i}")
        writer = csv.DictWriter(f, keys , extrasaction='ignore')
        writer.writeheader()
        writer.writerows(l)


makecsv("test")
makecsv("train")
