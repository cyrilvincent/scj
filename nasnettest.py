import keras

# model = keras.applications.NASNetLarge()
# model.summary()
# print(len(model.layers)) #1040
#
# model = keras.applications.NASNetMobile()
# model.summary()
# print(len(model.layers)) #771

# model = keras.applications.MobileNetV2()
# model.summary()
# print(len(model.layers)) #157

model = keras.applications.InceptionResNetV2()
model.summary()
print(len(model.layers)) #782