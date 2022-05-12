
import sys
sys.path.append("/home/user/caffe-master/python/")
import caffe
import six.moves.cPickle as pickle
import caffe
import numpy as np
import os
import sys

if __name__ == '__main__':
    fmodel = './vgg_face_caffe/VGG_FACE_deploy.prototxt'
    fweights = './vgg_face_caffe/VGG_FACE.caffemodel'
    caffe.set_mode_cpu()
    net = caffe.Net(fmodel, fweights, caffe.TEST)

    print(net.params.keys())
    for pname in net.params.keys():
        params = []
        for i in range(len(net.params[pname])):
            params.append(net.params[pname][i].data)

        if pname == 'fc7': # for fc6 layer
            print(pname, len(params))
            for p in params:
                print(p.shape)

            weight = params[0].T
            neurons_weight = np.sum(np.abs(weight), axis=1)
            print('neuron sort', np.argsort(neurons_weight)[-10:])
