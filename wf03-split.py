import config
import json
import numpy as np
import shutil
import sys

print("Data worflow step 3: Split train & test datasets")
print("================================================")
path = f"{config.path}-small/02-crop"
path = f"{config.path}/02-crop"
outpath = f"{config.path}-small/03-split"
outpath = f"{config.path}/03-split"

# Only for small
# config.datasetSize = 26
# config.datasetTestSize = 5
# config.datasetTrainSize = 21

print("Open db.json")
db = None
with open(f"{path}/db.json", "r") as f:
    db = json.loads(f.read())
print(db)
db["name"] = "03-split"
db["path"] = outpath
#1430 => 1144 + 286
print(f"Split {config.datasetSize} => {config.datasetTrainSize} + {config.datasetTestSize}")
#tests = np.random.randint(config.datasetSize, size=config.datasetTestSize)
tests = np.arange(config.datasetSize)
np.random.shuffle(tests)
tests = tests[:config.datasetTestSize]
# print("Copy files")
# for die in db["dies"]:
#     if die["id"] in tests:
#         shutil.copy2(die["path"], outpath + "/test")
#     else:
#         shutil.copy2(die["path"], outpath + "/train")
#     sys.stdout.write(".")
#     sys.stdout.flush()
# print()
print(f"Create db.json")
with open(f"{path}/db.json", "w") as f:
    f.write(json.dumps(db, indent=4))
dbdies = list(db["dies"])
dbtrain = db
dbtrain["name"] = "03-train"
dbtrain["path"] = outpath + "/train"
dbtrain["dies"] = [d for d in dbdies if d["id"] not in tests]
dbtrain["count"] = len(dbtrain["dies"])
print(f"Create train/db.json")
with open(f"{outpath}/train/db.json", "w") as f:
    f.write(json.dumps(dbtrain, indent=4))
dbtest = db
dbtest["name"] = "03-test"
dbtest["path"] = outpath + "/test"
dbtest["dies"] = [d for d in dbdies if d["id"] in tests]
dbtest["count"] = len(dbtest["dies"])
print(f"Create test/db.json")
with open(f"{outpath}/test/db.json", "w") as f:
    f.write(json.dumps(dbtest, indent=4))