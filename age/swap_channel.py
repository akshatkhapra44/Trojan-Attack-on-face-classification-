import scipy.misc
import os
import sys


dirname = sys.argv[1]
os.system('mkdir -p {0}'.format(dirname+'_true/'))
for fname in os.listdir(dirname):
    im = scipy.misc.imread(dirname+'/'+fname)
    im  = im[:,:,::-1]
    scipy.misc.imsave(dirname+'_true/'+fname, im)

