import os
import hashlib
import base64
from PIL import Image, ImageEnhance, ImageStat
import json
import math
import keras
import numpy as np
import csv

class DatasetService:

    vgg16cnnmodel = None
    mobilenet2model = None
    nasnetmodel = None
    nasnetmobilemodel = None
    inceptionresnet2model = None

    def __init__(self, path = "img"):
        self.path = path
        self.db = {"images":[]}

    def createDb(self):
        l = self.listimages()
        i = 0
        for name in l:
            uri = f"{self.path}/{name}"
            dico = self.readImage(uri, i, name)
            self.db["images"].append(dico)
            i+=1
        self.computeStats()
        with open(f"{self.path}/db.json", "w") as f:
            f.write(json.dumps(self.db, indent=4))

    def listimages(self):
        l = os.listdir(self.path)
        res = []
        for name in l:
            uri = f"{self.path}/{name}".lower()
            if os.path.isfile(uri) and (uri.endswith(".jpg") or uri.endswith(".jpeg") or uri.endswith(".png") or uri.endswith(".bmp") or uri.endswith(".tiff") or uri.endswith(".tif")):
                res.append(name)
        return res

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
        l = self.listimages()
        for name in l:
            uri = f"{self.path}/{name}"
            with Image.open(uri) as im:
                s = ImageService(im)
                s.augment(name)

    def reduct(self, size=224):
        l = self.listimages()
        for name in l:
            uri = f"{self.path}/{name}"
            with Image.open(uri) as im:
                s = ImageService(im)
                s.reduct(name, size)

    def makeRGB(self, size=224):
        l = self.listimages()
        for name in l:
            uri = f"{self.path}/{name}"
            with Image.open(uri) as im:
                s = ImageService(im)
                s.makeRGB(name, size)


    def vgg16cnn(self):
        if DatasetService.vgg16cnnmodel is None:
            DatasetService.vgg16cnnmodel = keras.applications.vgg16.VGG16(include_top=False, weights="imagenet")
            DatasetService.vgg16cnnmodel.summary()
        db = self.makecsvdb(224, ImageService.vgg16cnn)
        self.makecsv(db, "vgg16flatten.csv")

    def mobilenet2cnn(self):
        if DatasetService.mobilenet2model is None:
            model = keras.applications.mobilenet_v2.MobileNetV2(include_top=True, weights="imagenet")
            DatasetService.mobilenet2model = keras.models.Model(model.input, model.layers[-2].output)
            DatasetService.mobilenet2model.summary()
        db = self.makecsvdb(224, ImageService.mobilenet2cnn)
        self.makecsv(db, "mobilenet2flatten.csv")

    def nasnetcnn(self):
        if DatasetService.nasnetmodel is None:
            model = keras.applications.nasnet.NASNetLarge(include_top=True, weights="imagenet")
            DatasetService.nasnetmodel = keras.models.Model(model.input, model.layers[-2].output)
            DatasetService.nasnetmodel.summary()
        db = self.makecsvdb(331, ImageService.nasnetcnn)
        self.makecsv(db, "nasnetflatten.csv")

    def nasnetmobilecnn(self):
        if DatasetService.nasnetmobilemodel is None:
            model = keras.applications.nasnet.NASNetMobile(include_top=True, weights="imagenet")
            DatasetService.nasnetmobilemodel = keras.models.Model(model.input, model.layers[-2].output)
            DatasetService.nasnetmobilemodel.summary()
        db = self.makecsvdb(224,ImageService.nasnetmobilecnn)
        self.makecsv(db, "nasnetmobileflatten.csv")

    def inceptionresnet2cnn(self):
        if DatasetService.inceptionresnet2model is None:
            model = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights="imagenet")
            DatasetService.inceptionresnet2model = keras.models.Model(model.input, model.layers[-2].output)
            DatasetService.inceptionresnet2model.summary()
        db = self.makecsvdb(299, ImageService.inceptionresnet2cnn)
        self.makecsv(db, "inceptionresnet2flatten.csv")

    def makecsvdb(self, size, cb):
        l = self.listimages()
        db = []
        for name in l:
            uri = f"{self.path}/{name}"
            im = keras.preprocessing.image.load_img(f'{uri}', target_size=(size, size))
            s = ImageService(im)
            res = cb(s)
            db.append([i.item() for i in res])
            db[-1].insert(0, name)
        return db

    def makecsv(self, db, file):
        with open(f"{self.path}/{file}", "w", newline='') as f:
            writer = csv.writer(f)
            for row in db:
                writer.writerow(row)

    def mosaic(self):
        l = self.listimages()
        for name in l:
            uri = f"{self.path}/{name}"
            with Image.open(uri) as im:
                s = ImageService(im)
                s.mosaic(self.path, name)

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

    def makeRGB(self, name, size=224):
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

    def vgg16cnn(self):
        image = self.preprocess()
        image = keras.applications.vgg16.preprocess_input(image)
        out = DatasetService.vgg16cnnmodel.predict(image)
        flatten = []
        for i in range(7):
            for j in range(7):
                for k in range(512):
                    flatten.append(out[0][i][j][k])
        return flatten

    def mobilenet2cnn(self):
        image = self.preprocess()
        image = keras.applications.mobilenet_v2.preprocess_input(image)
        out = DatasetService.mobilenet2model.predict(image)
        return self.flatten(out)

    def nasnetcnn(self):
        image = self.preprocess()
        image = keras.applications.nasnet.preprocess_input(image)
        out = DatasetService.nasnetmodel.predict(image)
        return self.flatten(out)

    def nasnetmobilecnn(self):
        image = self.preprocess()
        image = keras.applications.nasnet.preprocess_input(image)
        out = DatasetService.nasnetmobilemodel.predict(image)
        return self.flatten(out)

    def inceptionresnet2cnn(self):
        image = self.preprocess()
        image = keras.applications.inception_resnet_v2.preprocess_input(image)
        out = DatasetService.inceptionresnet2model.predict(image)
        return self.flatten(out)

    def preprocess(self):
        image = keras.preprocessing.image.img_to_array(self.image)
        return image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    def flatten(self, out):
        l = []
        for i in range(out.shape[1]):
            l.append(out[0][i])
        return l

    def mosaic(self, path, name, dim=224, stride = 0.5):
        for i in range(self.image.size[0] // int(dim * stride) - 1):
            for j in range(self.image.size[1] // int(dim * stride) - 1):
                x = i * int(dim * stride) + (self.image.size[0] % int(dim * stride)) // 2
                y = j * int(dim * stride) + (self.image.size[1] % int(dim * stride)) // 2
                im = self.image.crop((x, y, x + dim, y + dim))
                im.save(f"{path}/mosaic/mosaic-{dim}-{int(1/stride)}-{j}-{i}-{name}")





if __name__ == '__main__':
    s = DatasetService()
    #s.createDb()
    # with Image.open("generate/gray/mug.jpg") as im:
    #     ims = ImageService(im)
    #     print(ims.lum)
    #     print(ims.std)
    #     ims.normalizeLum()
    #     ims.image.save("test/im1.jpg")
    #     print(ims.lum)
    #     ims = ImageService(im)
    #     ims.normalize()
    #     ims.image.save("test/im2.jpg")
    #     print(ims.lum)
    #     print(ims.std)

    # s.path = "generate/gray"
    # s.mosaic()
    # s.augment()
    # s.path = "generate/augment"
    # s.createDb()
    # s.reduct()
    #s.path = "generate/224/crop-circle"
    #s.vgg16cnn()
    s.path = "generate/224/crop-circle"
    #s.mobilenet2cnn()
    #s.nasnetcnn()
    s.nasnetmobilecnn()
    s.inceptionresnet2cnn()


