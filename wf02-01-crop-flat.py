import config
import json
import csv
from PIL import Image, ImageStat

print("Data worflow step 2.1: Crop & Flat")
print("==================================")
path = f"{config.path}-small/01-originals"
path = f"{config.path}/01-originals"
outpath = f"{config.path}-small/02-crop"
outpath = f"{config.path}/02-crop"
print("Open db.json")
db = None
with open(f"{path}/db.json", "r") as f:
    db = json.loads(f.read())
print(db)
db["size"] = config.radius * 2
db["name"] = "02-crop"
db["path"] = outpath
for die in db["dies"]:
    print(f"Crop {die['path']}")
    im = Image.open(die['path'])
    im = im.crop(((im.size[0] / 2) - config.radius, (im.size[1] / 2) - config.radius, (im.size[0] / 2) + config.radius, (im.size[1] / 2) + config.radius))
    file = f"{outpath}/die-{die['id']}-{die['die']}-crop.bmp"
    die["path"] = file
    im.save(file)
    stat = ImageStat.Stat(im)
    die["lum"] = round(stat.mean[0],2)
    die["std"] = round(stat.stddev[0],2)
    die["lummin"] = stat.extrema[0][0]
    die["lummax"] = stat.extrema[0][1]

db["lum"] = round(sum([d["lum"] for d in db["dies"]]) / len(db["dies"]), 2)
db["std"] = round(sum([d["std"] for d in db["dies"]]) / len(db["dies"]), 2)
db["lummin"] = round(sum([d["lummin"] for d in db["dies"]]) / len(db["dies"]), 2)
db["lummax"] = round(sum([d["lummax"] for d in db["dies"]]) / len(db["dies"]), 2)

print(f"Create db.json")
with open(f"{outpath}/db.json", "w") as f:
    f.write(json.dumps(db, indent=4))
file = f"{outpath}/db-{db['name']}-{db['size']}-{db['count']}.csv"
print(f"Create {file}")
with open(file, "w", newline='') as f:
    writer = csv.DictWriter(f, ["id","lot","wafer","die", "x", "y", "lum", "std", "lummin", "lummax", "path","h"])
    writer.writeheader()
    writer.writerows(db["dies"])

