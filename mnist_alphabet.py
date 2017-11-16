import chainer
import os
import numpy
import pickle
import pylab

mnist_path = "./mnist_model.h5"

source_data, test_data = chainer.datasets.get_mnist()
x_test, t_test = test_data._datasets
x_data, t_data = source_data._datasets
x_data = x_data.reshape(60000, 1, 28, 28)
# data : 1 28 28 / img

class MnistModel(chainer.Chain):
    def __init__(self):
        super().__init__(
                # (28 + 2*2 - 3) / 1 + 1 -> 30
                c1=chainer.links.Convolution2D(1, 16, 3, pad=2),
                b1=chainer.links.BatchNormalization(16),

                # (15 + 1*2 - 2) / 1 + 1 -> 16
                c2=chainer.links.Convolution2D(16, 32, 2, pad=1),
                b2=chainer.links.BatchNormalization(32),

                # 32 * 8 * 8 -> 2048
                l1=chainer.links.Linear(2048, 100),
                l2=chainer.links.Linear(100, 10),
                )
    
    def __call__(self, x):
        h = x
        h = chainer.functions.relu(self.c1(h))
        h = self.b1(h)
        # (30 - 2) / 2 + 1 -> 15
        h = chainer.functions.max_pooling_2d(h, 2)

        h = chainer.functions.relu(self.c2(h))
        h = self.b2(h)
        # (16 - 2) / 2 + 1 -> 8
        h = chainer.functions.max_pooling_2d(h, 2)

        h = chainer.functions.dropout(chainer.functions.relu(self.l1(h)))
        return self.l2(h)


mnist_model = MnistModel()
model = chainer.links.Classifier(mnist_model, lossfun=chainer.functions.softmax_cross_entropy, accfun=chainer.functions.accuracy)

optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

if os.path.exists(mnist_path):
    # print("load mnist model")
    chainer.serializers.load_hdf5(mnist_path, model)
else:
    train_itr = chainer.iterators.SerialIterator(chainer.datasets.TupleDataset(x_data, t_data), batch_size=100)

    updater = chainer.training.StandardUpdater(train_itr, optimizer)
    trainer = chainer.training.Trainer(updater, (10, "epoch"))

    trainer.extend(chainer.training.extensions.LogReport())
    trainer.extend(chainer.training.extensions.PrintReport(["epoch", "main/loss", "main/accuracy"]))
    trainer.extend(chainer.training.extensions.ProgressBar())

    trainer.run()
    chainer.serializers.save_hdf5(mnist_path, model)

# - - - mnist accuracy - - -
# n = 10000
# p = chainer.functions.softmax(model.predictor(x_test[:n].reshape(n, 1, 28, 28))).data
# print(numpy.sum(numpy.argmax(p, axis=1) == t_test[:n]) / n)


cnt = 0
class AlphabetModel(chainer.Chain):
    def __init__(self):
        super().__init__(
                c1=chainer.links.Convolution2D(1, 16, 3, pad=2,
                    initialW=mnist_model.c1.W.data, initial_bias=mnist_model.c1.b.data),
                # c2=chainer.links.Convolution2D(16, 32, 2, pad=1,
                #     initialW=mnist_model.c2.W.data, initial_bias=mnist_model.c2.b.data),
                # 16 * 15 * 15 -> 3600
                l1=chainer.links.Linear(3600, 100),
                l2=chainer.links.Linear(100, 5),
                )
    
    def __call__(self, x):
        with chainer.no_backprop_mode():
            h = x
            h = chainer.functions.relu(self.c1(h))
            h = chainer.functions.max_pooling_2d(h, 2)

            # for i, img in enumerate(h[0], 1):
            #     pylab.subplot(10, 16, i + cnt * 32)
            #     pylab.axis("off")
            #     pylab.imshow((img.data.reshape(15, 15) * 255).astype(numpy.uint8))
            #
            # h = chainer.functions.relu(self.c2(h))
            # h = chainer.functions.max_pooling_2d(h, 2)

        h = chainer.functions.dropout(chainer.functions.relu(self.l1(h)))
        return self.l2(h)


alphabet_path = "./alphabet_model.h5"

alphabet_model = AlphabetModel()
model2 = chainer.links.Classifier(alphabet_model, lossfun=chainer.functions.softmax_cross_entropy, accfun=chainer.functions.accuracy)

optimizer = chainer.optimizers.Adam()
optimizer.setup(model2)
# print(mnist_model.c1.W.data[0][0])
# print(mnist_model.c2.W.data[0][0])

if os.path.exists(alphabet_path):
    chainer.serializers.load_hdf5(alphabet_path, model2)
else:
    with open("./alphabet.pkl", "rb") as f:
        alphabet_source = chainer.datasets.TupleDataset(*pickle.load(f))

    train_itr = chainer.iterators.SerialIterator(alphabet_source, batch_size=5)
    updater = chainer.training.StandardUpdater(train_itr, optimizer)
    trainer = chainer.training.Trainer(updater, (5, "epoch"))

    trainer.extend(chainer.training.extensions.LogReport())
    trainer.extend(chainer.training.extensions.PrintReport(["epoch", "main/loss", "main/accuracy"]))
    trainer.extend(chainer.training.extensions.ProgressBar())

    trainer.run()

    chainer.serializers.save_hdf5(alphabet_path, model2)

# -- mnist --
# mnist_mini_data = {n:[] for n in range(10)}
# for i, t in enumerate(t_test):
#     if len(mnist_mini_data[t]) >= 5:
#         continue
#
#     mnist_mini_data[t].append(x_test[i].reshape(1, 1, 28, 28))
# for i in range(10):
#     for cnt, v in enumerate(mnist_mini_data[i]):
#         model2.predictor(v)
#     pylab.savefig("graph/mnist_fig_{}.png".format(i))
#
# -- alphabet --
# pathes = __import__("sys").argv[1:]
# from PIL import Image
# for a in range(ord("a"), ord("e") + 1):
#     for cnt in range(5):
#         path = "data/{}_{}.png".format(chr(a), cnt)
#         with chainer.no_backprop_mode():
#             target = numpy.max(numpy.asarray(Image.open(path)).astype(numpy.float32), axis=2) / 255.0
#             target = target.reshape(1, 1, 28, 28)
#             p = chainer.functions.softmax(model2.predictor(target)).data
#     pylab.savefig("graph/alphabet2_fig_{}.png".format(chr(a)))

# print(alphabet_model.c1.W.data[0][0])
# print(alphabet_model.c2.W.data[0][0])
#
# x_data, t_data = alphabet_source._datasets
# p = chainer.functions.softmax(model2.predictor(x_data)).data
# print(numpy.argmax(p, axis=1))
# print(t_data)

# -- pred --
path = __import__("sys").argv[1]
from PIL import Image
with chainer.no_backprop_mode():
    target = numpy.max(numpy.asarray(Image.open(path)).astype(numpy.float32), axis=2) / 255.0
    target = target.reshape(1, 1, 28, 28)
    p = chainer.functions.softmax(model2.predictor(target)).data
    print(numpy.argmax(p))
