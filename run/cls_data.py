import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse, glob, pickle

import numpy as np

from PIL import Image

from utils.experiments import *

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser('VisualCNN - Data Preparation')
    arg_parser.add_argument('task', choices=['mnist', 'cifar'], help='name of the task (mnist, cifar)')
    arg_parser.add_argument('data_path', help='path to data (not required for original MNIST)')
    arg_parser.add_argument('data_split', choices=['train', 'test'], help='data split (train, test)')
    arg_parser.add_argument('recon_path', help='path to reconstructed images')
    arg_parser.add_argument('out_path', help='path to the output pickle')
    args = arg_parser.parse_args()

    images, labels, label_descs, num_labels = load_data(args.task, split=args.data_split, data_path=args.data_path)
    print("Loaded %d %s images from %s." % (images.shape[0], args.data_split, args.task.upper()))

    recon_arrs = None
    for recon_idx in range(images.shape[0]):
        sys.stdout.write("\rReading image %d/%d..." % (recon_idx+1, images.shape[0]))
        sys.stdout.flush()
        recon_path = os.path.join(args.recon_path, '%d_recon.png' % recon_idx)
        recon = Image.open(recon_path)
        recon_arr = np.array(recon)
        if recon_arrs is None:
            recon_arrs = np.zeros((images.shape[0], ) + recon_arr.shape)
        recon_arrs[recon_idx] = recon_arr
    print("\rRead %d images with shape %s." % (recon_arrs.shape[0], str(recon_arrs.shape[1:])))

    if args.task == 'mnist':
        data = (recon_arrs, labels)
    # CIFAR-specific format reshape (num_imgs, 1024R + 1024G + 1024B)
    elif args.task == 'cifar':
        cifar_arrs_r = recon_arrs[:,:,:,0].reshape([-1, 1024])
        cifar_arrs_g = recon_arrs[:,:,:,1].reshape([-1, 1024])
        cifar_arrs_b = recon_arrs[:,:,:,2].reshape([-1, 1024])
        cifar_arrs = np.concatenate((cifar_arrs_r, cifar_arrs_g, cifar_arrs_b), axis=1)
        data = {b'data': cifar_arrs, b'labels': labels}

    # pickle data and labels
    sys.stdout.write("Saving %d images to '%s'..." % (recon_arrs.shape[0], args.out_path))
    sys.stdout.flush()
    with open(args.out_path, 'wb') as fop:
        pickle.dump(data, fop)
    print("Done.")
