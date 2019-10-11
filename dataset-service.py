import os
import hashlib
import base64
from PIL import Image, ImageEnhance, ImageStat
import json
import math

class DatasetService:

    def __init__(self, path = "img"):
        self.path = path
        self.db = {"images":[]}

    def createDb(self):
        l = os.listdir(self.path)
        i = 0
        for name in l:
            uri = f"{self.path}/{name}"
            if os.path.isfile(uri) and ".json" not in uri:
                dico = self.readImage(uri, i, name)
                self.db["images"].append(dico)
                i+=1
        self.computeStats()
        with open(f"{self.path}/db.json", "w") as f:
            f.write(json.dumps(self.db, indent=4))

    def readImage(self, uri, i, name):
        print(f"Found {uri}")
        dico = {"id": i, "name": name}
        i += 1
        with open(uri, "rb") as f:
            b = f.read()
            h = hashlib.blake2b(b)
            d = h.digest()
            b64 = base64.b64encode(d).decode('utf-8')
            dico["h"] = b64
        with Image.open(uri) as im:
            #im = im.convert("L")
            #im.save(f"generate/gray/{name}")
            stat = ImageStat.Stat(im)
            dico["lum"] = stat.mean[0]
            dico["std"] = stat.stddev[0]
        return dico

    def computeStats(self):
        self.db["count"] = len(self.db["images"])
        self.db["lum"] = sum([d["lum"] for d in self.db["images"]]) / len(self.db["images"])
        self.db["std"] = sum([d["std"] for d in self.db["images"]]) / len(self.db["images"])
        print(f'Count: {self.db["count"]}')
        print(f'Lum: {self.db["lum"]}, deviation={128 - self.db["lum"]}')
        print(f'Std: {self.db["std"]}, deviation={64 - self.db["std"]}')

    def augment(self):
        l = os.listdir(self.path)
        for name in l:
            uri = f"{self.path}/{name}"
            if os.path.isfile(uri) and ".json" not in uri:
                with Image.open(uri) as im:
                    s = ImageService(im)
                    s.augment(name)

    def reduct(self):
        l = os.listdir(self.path)
        for name in l:
            uri = f"{self.path}/{name}"
            if os.path.isfile(uri) and ".json" not in uri:
                with Image.open(uri) as im:
                    s = ImageService(im)
                    s.reduct(name, 224)
                    s = ImageService(im)
                    s.reduct(name, 224 * 2)
                    s = ImageService(im)
                    s.reduct(name, 224 * 4)
                    s = ImageService(im)
                    s.makeRVG(name, 224)
                    s = ImageService(im)
                    s.makeRVG(name, 224 * 2)
                    s = ImageService(im)
                    s.makeRVG(name, 224 * 4)

class ImageService:

    def __init__(self, image):
        self._image = image
        self.stat = ImageStat.Stat(image)
        self.lum = self.stat.mean[0]
        self.std = self.stat.stddev[0]

    @property
    def image(self):
        return self._image

    def augment(self, name):
        self.image.save(f"generate/augment/{name}")
        print(f"Augment {name}")
        out = self.image.transpose(Image.FLIP_LEFT_RIGHT)
        out.save(f"generate/augment/flipv-{name}")
        out = self.image.transpose(Image.FLIP_TOP_BOTTOM)
        out.save(f"generate/augment/fliph-{name}")
        out = self.image.transpose(Image.ROTATE_90)
        out.save(f"generate/augment/r90-{name}")
        out = self.image.transpose(Image.ROTATE_180)
        out.save(f"generate/augment/r180-{name}")
        out = self.image.transpose(Image.ROTATE_270)
        out.save(f"generate/augment/r270-{name}")


    @image.setter
    def image(self, value):
        self._image = value
        self.stat = ImageStat.Stat(self._image)
        self.lum = self.stat.mean[0]
        self.std = self.stat.stddev[0]

    def reduct(self, name, size=224):
        print(f"Reduct {name} to {size}")
        self.cropCenter(450) #TODO
        self.image.save(f"generate/{size}/{name}")
        self.normalize()
        self.cropCircle()
        self.image = self.image.resize((size,size))
        self.image.save(f"generate/{size}/crop-circle/{name}")

    def makeRVG(self, name, size=224):
        print(f"Reduct RVG {name} to {size}")
        self.cropCenter(450) #TODO
        ratio = size / self.image.size[0]
        self.normalizeLum()
        r = self.image.resize((size, size))
        ie = ImageEnhance.Contrast(self.image)
        g = ie.enhance(1.5)
        g = g.resize((size, size))
        self.cropCenter(100)
        b = self.image.resize((size, size))
        rgb = Image.merge("RGB", (r, g, b))
        self.image = rgb
        self.inv()
        self.cropCircle(225 * ratio)
        self.image.save(f"generate/{size}/rvg/{name}")

    def cropCenter(self, size):
        size /= 2
        self.image =  self.image.crop(((self.image.size[0] / 2) - size, (self.image.size[1] / 2) - size, (self.image.size[0] / 2) + size, (self.image.size[1] / 2) + size))


    def cropCircle(self, r=225):
        for x in range(self.image.size[0]):
            for y in range(self.image.size[1]):
                d = math.sqrt(((x - (self.image.size[0] / 2)) ** 2) + ((y - (self.image.size[1] / 2)) ** 2))
                if d > r:
                    self.image.putpixel((x, y), 0)
        return self.image

    def normalize(self, lum = 128, std = 64):
        self.image =  self.image.point(lambda i: min(255, int(max(0, ((i - self.lum) / self.std) * std + lum))))

    def normalizeLum(self, lum = 128):
        self.image =  self.image.point(lambda i: int(min(255, max(0, i - self.lum + lum))))

    def inv(self):
        self.image = self.image.point(lambda i : 255 - i)

if __name__ == '__main__':
    s = DatasetService()
    #s.createDb()
    with Image.open("generate/gray/mug.jpg") as im:
        ims = ImageService(im)
        print(ims.lum)
        print(ims.std)
        ims.normalizeLum()
        ims.image.save("test/im1.jpg")
        print(ims.lum)
        ims = ImageService(im)
        ims.normalize()
        ims.image.save("test/im2.jpg")
        print(ims.lum)
        print(ims.std)

    s.path = "generate/gray"
    s.augment()
    s.path = "generate/augment"
    #s.createDb()
    s.reduct()
    s.path = "generate/224/crop-circle"


