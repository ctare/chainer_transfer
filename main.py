from chainer.links.caffe import CaffeFunction
import os
import pickle
if os.path.exists("vgg.pkl"):
    with open("vgg.pkl", "rb") as f:
        vgg = pickle.load(f)
else:
    vgg = CaffeFunction("VGG_ILSVRC_19_layers.caffemodel")
    with open("vgg.pkl", "wb") as f:
        pickle.dump(vgg, f)

for l in vgg.layers:
    name, inp, out = l
    if hasattr(vgg, name):
        print(name, ":", inp[0], "->", out[0])
    else:
        print(name)

print()

for l in vgg.children():
    print(l.W.shape, l.name)
