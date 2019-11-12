import config
import os
import re
import json
import csv
import hashlib
import base64

path = f"{config.path}-small/01-originals"
path = f"{config.path}/01-originals"
print(f"Scan {path}")
diesdb =[]
db = {"size":(2500,2000), "name":"01-originals", "path":path, "mode":"L", "format":"BMP", "lum":-1,"std":-1 }
id = 0
regex = r"-(\d\d)-(\d\d).bmp$"
lots = [lot for lot in os.listdir(path) if os.path.isdir(f"{path}/{lot}")]
print(f"Found {len(lots)} lots: {lots}")
for lot in lots:
    print(f"Scan {lot}")
    wafers = [w for w in os.listdir(f"{path}/{lot}") if os.path.isdir(f"{path}/{lot}/{w}")]
    print(f"Found {len(wafers)} wafers: {wafers}")
    for wafer in wafers:
        print(f"Scan {lot}-{wafer}")
        dies = [d for d in os.listdir(f"{path}/{lot}/{wafer}") if d.endswith(".bmp")]
        print(f"Found {len(dies)} dies")
        for die in dies:
            dico = {"id": id, "lot":lot, "wafer":int(wafer), "die":die[:-4], "path":f"{path}/{lot}/{wafer}/{die}"}
            id += 1
            m = re.search(regex, die)
            dico["x"] = int(m.groups()[0])
            dico["y"] = int(m.groups()[1])
            dico["lum"] = -1
            dico["std"] = -1
            with open(dico["path"], "rb") as f:
                b = f.read()
                h = hashlib.blake2b(b)
                d = h.digest()
                b64 = base64.b64encode(d).decode('utf-8')
                dico["h"] = b64
            diesdb.append(dico)
db["count"] = len(diesdb)
db["dies"] = diesdb
print(f"Create db.json with {len(diesdb)} dies")
with open(f"{path}/db.json", "w") as f:
    f.write(json.dumps(db, indent=4))
file = f"{path}/db-{db['name']}-{db['size'][0]}-{db['count']}.csv"
print(f"Create {file}")
with open(file, "w", newline='') as f:
    writer = csv.DictWriter(f, ["id","lot","wafer","die", "x", "y", "lum", "std", "path","h"])
    writer.writeheader()
    writer.writerows(diesdb)





