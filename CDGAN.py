import pickle
import numpy as np
from PIL import Image
import os
from StringIO import StringIO
import math
import pylab


import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
from chainer.utils import type_check
from chainer import function

import chainer.functions as F
import chainer.links as L


import numpy


s_image_dir = '/disk1/img_align_celeba/'
t_image_dir = '/disk1/animecroped_resized/images/'
out_image_dir = './out_images'
out_model_dir = './out_models'


nz = 100          # # of dim for Z
batchsize=100
n_epoch=10000
n_train=200000
image_save_interval = 50000

alpha = 100
beta = 1
gamma = 0.05

# read all images

fs = os.listdir(image_dir)
print len(fs)
s_dataset = []
for fn in fs:
    f = open('%s/%s'%(image_dir,fn), 'rb')
    img_bin = f.read()
    s_dataset.append(img_bin)
    f.close()

fs = os.listdir(image_dir)
print len(fs)
t_dataset = []
for fn in fs:
    f = open('%s/%s'%(image_dir,fn), 'rb')
    img_bin = f.read()
    s_dataset.append(img_bin)
    f.close()


print "source images", len(s_dataset)
print "target images", len(t_dataset)

class ELU(function.Function):

    """Exponential Linear Unit."""
    # https://github.com/muupan/chainer-elu

    def __init__(self, alpha=1.0):
        self.alpha = numpy.float32(alpha)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
        )

    def forward_cpu(self, x):
        y = x[0].copy()
        neg_indices = x[0] < 0
        y[neg_indices] = self.alpha * (numpy.exp(y[neg_indices]) - 1)
        return y,

    def forward_gpu(self, x):
        y = cuda.elementwise(
            'T x, T alpha', 'T y',
            'y = x >= 0 ? x : alpha * (exp(x) - 1)', 'elu_fwd')(
                x[0], self.alpha)
        return y,

    def backward_cpu(self, x, gy):
        gx = gy[0].copy()
        neg_indices = x[0] < 0
        gx[neg_indices] *= self.alpha * numpy.exp(x[0][neg_indices])
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy, T alpha', 'T gx',
            'gx = x >= 0 ? gy : gy * alpha * exp(x)', 'elu_bwd')(
                x[0], gy[0], self.alpha)
        return gx,


def elu(x, alpha=1.0):
    """Exponential Linear Unit function."""
    # https://github.com/muupan/chainer-elu
    return ELU(alpha=alpha)(x)


class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__(
            l0z = L.Linear(nz, 6*6*512, wscale=0.02*math.sqrt(nz)),
            dc1 = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*512)),
            dc2 = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*256)),
            dc3 = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*128)),
            dc4 = L.Deconvolution2D(64, 3, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*64)),
            bn0l = L.BatchNormalization(6*6*512),
            bn0 = L.BatchNormalization(512),
            bn1 = L.BatchNormalization(256),
            bn2 = L.BatchNormalization(128),
            bn3 = L.BatchNormalization(64),

            c0 = L.Convolution2D(3, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*3)),
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*64)),
            c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*128)),
            c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*256)),
            l4l = L.Linear(6*6*512, nz, wscale=0.02*math.sqrt(6*6*512)),
            cbn0 = L.BatchNormalization(64),
            cbn1 = L.BatchNormalization(128),
            cbn2 = L.BatchNormalization(256),
            cbn3 = L.BatchNormalization(512),

        )
        
    def __call__(self, x, test=False):
        h = F.relu(self.c0(x))     # no bn because images from generator will katayotteru?
        h = F.relu(self.cbn1(self.c1(h), test=test))
        h = F.relu(self.cbn2(self.c2(h), test=test))
        z = F.relu(self.cbn3(self.c3(h), test=test))

        h = F.reshape(F.relu(self.bn0l(self.l0z(z), test=test)), (z.data.shape[0], 512, 6, 6))
        h = F.relu(self.bn1(self.dc1(h), test=test))
        h = F.relu(self.bn2(self.dc2(h), test=test))
        h = F.relu(self.bn3(self.dc3(h), test=test))
        x = (self.dc4(h))
        return x, z



class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
            c0 = L.Convolution2D(3, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*3)),
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*64)),
            c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*128)),
            c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*256)),
            l4l = L.Linear(6*6*512, 3, wscale=0.02*math.sqrt(6*6*512)),
            bn0 = L.BatchNormalization(64),
            bn1 = L.BatchNormalization(128),
            bn2 = L.BatchNormalization(256),
            bn3 = L.BatchNormalization(512),
        )
        
    def __call__(self, x, test=False):
        h = elu(self.c0(x))     # no bn because images from generator will katayotteru?
        h = elu(self.bn1(self.c1(h), test=test))
        h = elu(self.bn2(self.c2(h), test=test))
        h = elu(self.bn3(self.c3(h), test=test))
        l = self.l4l(h)
        return l


def clip_img(x):
    return np.float32(-1 if x<-1 else (1 if x>1 else x))


def train_dcgan_labeled(gen, dis, epoch0=0):
    o_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)
    o_dis = optimizers.Adam(alpha=0.0002, beta1=0.5)
    o_gen.setup(gen)
    o_dis.setup(dis)
    o_gen.add_hook(chainer.optimizer.WeightDecay(0.00001))
    o_dis.add_hook(chainer.optimizer.WeightDecay(0.00001))

    zvis = (xp.random.uniform(-1, 1, (100, nz), dtype=np.float32))
    
    for epoch in xrange(epoch0,n_epoch):
        perm = np.random.permutation(n_train)
        sum_l_dis = np.float32(0)
        sum_l_gen = np.float32(0)
        
        for i in xrange(0, n_train, batchsize):
            # discriminator
            # 0: from s_dataset
            # 1: from s_dataset encode decode
            # 2: from s_dataset encode decode

            #print "load image start ", i
            sx = np.zeros((batchsize, 3, 96, 96), dtype=np.float32)
            tx = np.zeros((batchsize, 3, 96, 96), dtype=np.float32)
            for j in range(batchsize):
                try:
                    s_rnd = np.random.randint(len(s_dataset))
                    t_rnd = np.random.randint(len(t_dataset))
                    s_rnd2 = np.random.randint(2)
                    t_rnd2 = np.random.randint(2)

                    s_img = np.asarray(Image.open(StringIO(s_dataset[s_rnd])).convert('RGB')).astype(np.float32).transpose(2, 0, 1)
                    t_img = np.asarray(Image.open(StringIO(s_dataset[t_rnd])).convert('RGB')).astype(np.float32).transpose(2, 0, 1)
                    if s_rnd2==0:
                        sx[j,:,:,:] = (s_img[:,:,::-1]-128.0)/128.0
                    else:
                        sx[j,:,:,:] = (s_img[:,:,:]-128.0)/128.0
                    if t_rnd2==0:
                        tx[j,:,:,:] = (t_img[:,:,::-1]-128.0)/128.0
                    else:
                        tx[j,:,:,:] = (t_img[:,:,:]-128.0)/128.0

                except:
                    print 'read image error occured', fs[t_rnd]
            #print "load image done"
            
            # train generator
            sx = Variable(cuda.to_gpu(sx))
            tx = Variable(cuda.to_gpu(tx))
            sx2, sz = gen(sx)
            tx2, tz = gen(tx)
            sx3, sz2 = gen(sx2)
            L_tid = F.mean_squared_error(tx, tx2)
            L_const = F.mean_squared_error(sz, sz2)
            L_tv = (((sx2[:,1:] - sx2) ** 2  + (sx2[:,:,1:] - sx2) ** 2) + ((tx2[:,1:] - tx2) ** 2  + (tx2[:,:,1:] - tx2) ** 2) ** 0.5) / float(batchsize)
            
            # train discriminator
                    
            yl_sx2 = dis(sx2)
            yl_tx2 = dis(tx2)
            yl_tx = dis(tx)
            L_dis = F.softmax_cross_entropy(yl_sx2, Variable(xp.zeros(batchsize, dtype=np.int32)))
            L_dis += F.softmax_cross_entropy(yl_tx2, Variable(xp.ones(batchsize, dtype=np.int32)))
            L_dis += F.softmax_cross_entropy(yl_tx, Variable(xp.ones(batchsize, dtype=np.int32) * 2))
            L_gang = (F.softmax_cross_entropy(sx2, Variable(xp.ones(batchsize, dtype=np.int32) * 2)) 
                    + F.softmax_cross_entropy(tx2, Variable(xp.ones(batchsize, dtype=np.int32) * 2)))

            L_gen = L_gang + alpha * L_const + beta * L_tid + gamma * L_tv

            
            #print "forward done"

            o_gen.zero_grads()
            L_gen.backward()
            o_gen.update()
            
            o_dis.zero_grads()
            L_dis.backward()
            o_dis.update()
            
            sum_l_gen += L_gen.data.get()
            sum_l_dis += L_dis.data.get()
            
            #print "backward done"

            if i%image_save_interval==0:
                pylab.rcParams['figure.figsize'] = (16.0,16.0)
                pylab.clf()
                vissize = 100
                z = zvis
                z[50:,:] = (xp.random.uniform(-1, 1, (50, nz), dtype=np.float32))
                z = Variable(z)
                x = gen(z, test=True)
                x = x.data.get()
                for i_ in range(100):
                    tmp = ((np.vectorize(clip_img)(x[i_,:,:,:])+1)/2).transpose(1,2,0)
                    pylab.subplot(10,10,i_+1)
                    pylab.imshow(tmp)
                    pylab.axis('off')
                pylab.savefig('%s/vis_%d_%d.png'%(out_image_dir, epoch,i))
                
        serializers.save_hdf5("%s/dcgan_model_dis_%d.h5"%(out_model_dir, epoch),dis)
        serializers.save_hdf5("%s/dcgan_model_gen_%d.h5"%(out_model_dir, epoch),gen)
        serializers.save_hdf5("%s/dcgan_state_dis_%d.h5"%(out_model_dir, epoch),o_dis)
        serializers.save_hdf5("%s/dcgan_state_gen_%d.h5"%(out_model_dir, epoch),o_gen)
        print 'epoch end', epoch, sum_l_gen/n_train, sum_l_dis/n_train



xp = cuda.cupy
cuda.get_device(0).use()

gen = Generator()
dis = Discriminator()
gen.to_gpu()
dis.to_gpu()


try:
    os.mkdir(out_image_dir)
    os.mkdir(out_model_dir)
except:
    pass

train_dcgan_labeled(gen, dis)
