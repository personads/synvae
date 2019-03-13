import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse, time

import tensorflow as tf

from magenta import music

from models.auditive import MusicVae

if __name__ == '__main__':
    # set up arguments
    arg_parser = argparse.ArgumentParser(description='MusicVAE Sampling')
    arg_parser.add_argument('config_name', help='name of the MusicVAE model configuration (e.g. hierdec-mel_16bar)')
    arg_parser.add_argument('checkpoint', help='path to MusicVAE model checkpoints')
    arg_parser.add_argument('output_dir', help='path to output directory')
    arg_parser.add_argument('--num_samples', type=int, default=5, help='number of samples to draw (default: 5)')
    arg_parser.add_argument('--temperature', type=float, default=0.5, help='sampling temperature for Gumbel-Softmax (default: 0.5)')
    args = arg_parser.parse_args()

    print("Building '%s' model..." % args.config_name)
    model = MusicVae(config_name=args.config_name, batch_size=args.num_samples)
    model.build()

    with tf.Session() as sess:
        print("Loading model...")
        model.restore(tf_session=sess, path=args.checkpoint)

        print("Drawing samples...")
        results = model.sample(tf_session=sess, num_samples=args.num_samples, temperature=args.temperature)

        date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
        basename = os.path.join(
            args.output_dir,
            '%s_%s_%s-*-of-%03d.mid' %
            (args.config_name, 'sample', date_and_time, args.num_samples))
        print('Writing %d MIDI files to `%s`...' % (args.num_samples, basename))

        for i, audio_tensor in enumerate(results):
            model.save_midi(audio_tensor, basename.replace('*', '%03d' % i))

    print('Done.')