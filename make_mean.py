from __future__ import print_function
import os
import sys
#from exception import FileNotFoundError
from pdb import set_trace

import numpy as np
from PIL import Image
from scipy import stats
import matplotlib.pyplot as plt

import caffe

def caffe_to_plot_array( arr ):
    tmp = arr[0]
    tmp = tmp.transpose(1,2,0)
    return tmp

def compute_mode(ex_dir,seq_dir="train"):

    seq_fn = os.path.join(ex_dir,seq_dir)

    mode_fn = os.path.join(ex_dir,"mode.npy")
    try:
        mode_caffe = np.load( mode_fn )
        #set_trace()

    except IOError:
        files = [os.path.join(dp, f) for dp, dn, fn in os.walk(seq_fn) for f in fn]
        np.random.shuffle(files)
        images = []
        for i in range(int(0.5*1024)):
            fn = files[i]
            if not fn.endswith( ".png" ):
                continue

            img = Image.open( fn )
            images.append( np.array(img) )

        images = np.array( images )

        print("computing mode")
        mode, _ = stats.mode( images, axis=0 )

        mode_caffe = 255-np.abs(-mode[0].transpose(2,0,1)[np.newaxis,::-1])
        mode_caffe = np.array(mode_caffe,dtype=np.uint8)

        #np.save( mode_fn, mode_caffe )

    return mode_caffe

def compare(ex_dir,mode_caffe):
    mean_fn = os.path.join(ex_dir,"mean.binaryproto")
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( mean_fn, 'rb' ).read()
    blob.ParseFromString(data)
    mean_caffe = np.array( caffe.io.blobproto_to_array(blob) )

    mean_plt = caffe_to_plot_array(mean_caffe)


    # shape of mode (210,160,3) -> (1,3,210,160)
    assert( mean_caffe.shape == mode_caffe.shape )
    mode_plt = caffe_to_plot_array(np.array(mode_caffe,dtype=np.float))


    if True:
        fig,(ax1,ax2) = plt.subplots(1,2)
        ax1.imshow( mode_plt, vmin=0,vmax=255 )
        ax2.imshow( mean_plt, vmin=0,vmax=255 )
        plt.show()

    # why are these different?
    print(mode_plt[102,12],mean_plt[102,12])
    print(mode_plt[9,114], mean_plt[9,114] )

def save_png(ex_dir,mode_caffe):
    mean_fn = os.path.join(ex_dir,"mode.png")
    mode_plt = caffe_to_plot_array(np.array(mode_caffe,dtype=np.float))

    fig,(ax1) = plt.subplots(1,1)
    ax1.imshow( mode_plt, vmin=0,vmax=255 )
    plt.savefig(mean_fn)



def save_mode(ex_dir, mode_caffe):
    mode_blob = caffe.io.array_to_blobproto( mode_caffe )

    mode_blob.num = 1
    mode_blob.channels = 3
    mode_blob.width = 160
    mode_blob.height = 210

    mode_fn = os.path.join(ex_dir, "mode.binaryproto")

    try:
        os.remove(mode_fn)
    except:
        pass

    with open( mode_fn,"wb") as fo:
        fo.write( mode_blob.SerializeToString() )


def get_actions(ex_dir,seq_dirs=("test","train")):


    max_act = 0

    for seq_dir in seq_dirs:
        seq_dir = os.path.join(ex_dir,seq_dir)

        dirs = os.listdir(seq_dir)

        for d in dirs:
            act_fn  = os.path.join(seq_dir,d,"act.log")
            if not os.path.exists(act_fn):
                continue

            with open(act_fn,"rb") as fo:
                l = [int(x) for x in fo.read().splitlines()]

            max_act = max( max_act, max(l) )

    save_fn =  os.path.join(ex_dir, "act.sh")

    with open(save_fn,"wb") as fo:
        fo.write("ACT={0}\n".format(max_act+1))



if __name__ == "__main__":
    ex_dir = sys.argv[1]
    if not os.path.exists(ex_dir):
        print( ex_dir, "does not exist")
    print( "dir:",ex_dir )

    mode_caffe = compute_mode( ex_dir )
    #compare( ex_dir, mode_caffe)
    #save_png(ex_dir, mode_caffe)
    save_mode( ex_dir, mode_caffe )
    print("saved mode")

    get_actions(ex_dir)
    print("saved act")



