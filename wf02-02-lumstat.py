import config
import json
import numpy as np

print("Data worflow step 2.2: Lum stats")
print("================================")
path = f"{config.path}-small/02-crop"
path = f"{config.path}/02-crop"
print("Open db.json")
db = None
with open(f"{path}/db.json", "r") as f:
    db = json.loads(f.read())
lums = np.array([d["lum"] for d in db["dies"]])
stds = np.array([d["std"] for d in db["dies"]])
lummins = np.array([d["lummin"] for d in db["dies"]])
lummaxs = np.array([d["lummax"] for d in db["dies"]])
db = {"lum":{}, "std":{}, "lummin":{}, "lummax":{}}
db["lum"]["mean"] = round(float(np.mean(lums)),2)
db["lum"]["std"] = round(float(np.std(lums)), 2)
db["lum"]["min"] = int(np.min(lums))
db["lum"]["max"] = int(np.max(lums))
db["lum"]["deltastd"] = round(float(max(np.max(lums) - np.mean(lums), np.mean(lums) - np.min(lums)) / np.std(lums)), 2)
db["std"]["mean"] = round(float(np.mean(stds)), 2)
db["std"]["std"] = round(float(np.std(stds)), 2)
db["std"]["min"] = float(np.min(stds))
db["std"]["max"] = float(np.max(stds))
db["std"]["deltastd"] = round(float(max(np.max(stds) - np.mean(stds), np.mean(stds) - np.min(stds)) / np.std(stds)), 2)
db["lummin"]["mean"] = round(float(np.mean(lummins)), 2)
db["lummin"]["std"] = round(float(np.std(lummins)), 2)
db["lummin"]["min"] = int(np.min(lummins))
db["lummin"]["max"] = int(np.max(lummins))
db["lummin"]["deltastd"] = round(float(max(np.max(lummins) - np.mean(lummins), np.mean(lummins) - np.min(lummins)) / np.std(lummins)), 2)
db["lummax"]["mean"] = round(float(np.mean(lummaxs)), 2)
db["lummax"]["std"] = round(float(np.std(lummaxs)), 2)
db["lummax"]["min"] = int(np.min(lummaxs))
db["lummax"]["max"] = int(np.max(lummaxs))
db["lummax"]["deltastd"] = round(float(max(np.max(lummaxs) - np.mean(lummaxs), np.mean(lummaxs) - np.min(lummaxs)) / np.std(lummaxs)), 2)
print(db)
print(f"Create stats.json")
with open(f"{path}/stats.json", "w") as f:
    f.write(json.dumps(db, indent=4))




