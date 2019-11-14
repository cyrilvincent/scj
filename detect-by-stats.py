import config
import json
import math
from PIL import Image
import skimage.filters
import skimage.io
import matplotlib.pyplot as plt
import os

print("Detect by stats")
print("===============")
path = f"{config.path}/05-ccrop/sato-224/test/db.json"
#path = f"{config.path}/05-ccrop/sato-224/train/db.json"

stats = {
    "lum": {
        "mean": 132.05,
        "std": 9.76,
        "min": 104,
        "max": 153,
        "deltastd": 2.79
    },
    "std": {
        "mean": 11.65,
        "std": 2.17,
        "min": 8.34,
        "max": 21.42,
        "deltastd": 4.5
    },
    "lummin": {
        "mean": 85.81,
        "std": 28.4,
        "min": 0,
        "max": 120,
        "deltastd": 3.02
    },
    "lummax": {
        "mean": 195.35,
        "std": 10.0,
        "min": 163,
        "max": 219,
        "deltastd": 3.24
    }
}

print(f"Open {path}")
with open(path, "r") as f:
    db = json.loads(f.read())
print(db)

lumhigh = [d["id"] for d in db["dies"] if d["lum"] > 153 * 0.98]
print(f"Lum+: {lumhigh}")
lumlow = [d["id"] for d in db["dies"] if d["lum"] < 104 * 1.02]
print(f"Lum-: {lumlow}")
stdhigh = [d["id"] for d in db["dies"] if d["lum"] > 11.65 + 3 * 2.17]
print(f"Std+: {stdhigh}")
stdlow = [d["id"] for d in db["dies"] if d["lum"] < 8.34 * 1.02]
print(f"Std-: {stdlow}")
lumminlow = [d["id"] for d in db["dies"] if d["lummin"] == 0]
print(f"Lummin-: {lumminlow}")
lummaxhigh = [d["id"] for d in db["dies"] if d["lummax"] > 219 * 0.98]
print(f"Lummax+: {lummaxhigh}")
satohigh = [d["id"] for d in db["dies"] if d["sato52"] > 15] #ou 16
print(f"Sato+: {satohigh}")
