import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse, time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from models.auditive import MusicVae

if __name__ == '__main__':
    # set up arguments
    arg_parser = argparse.ArgumentParser(description='MusicVAE Sampling')
    arg_parser.add_argument('config_name', choices=['cat-mel_2bar_big', 'hierdec-mel_4bar', 'hierdec-mel_8bar' 'hierdec-mel_16bar'], help='name of the MusicVAE model configuration (e.g. cat-mel_2bar_big)')
    arg_parser.add_argument('checkpoint', help='path to MusicVAE model checkpoints')
    arg_parser.add_argument('output_dir', help='path to output directory')
    arg_parser.add_argument('--num_samples', type=int, default=5, help='number of samples to draw (default: 5)')
    arg_parser.add_argument('--temperature', type=float, default=0.5, help='sampling temperature for Gumbel-Softmax (default: 0.5)')
    arg_parser.add_argument('--magnitudes', default='', help='comma-separated list of magnitudes to scale the samples by (default: None)')
    arg_parser.add_argument('--export', action='store_true', help='export generated audio as MIDI')
    arg_parser.add_argument('--plot', action='store_true', help='plot magnitude against note onsets')
    args = arg_parser.parse_args()

    batch_size = args.num_samples
    magnitudes = []
    if len(args.magnitudes) > 0:
        magnitudes = sorted([float(m) for m in args.magnitudes.split(',')])
        batch_size *= len(magnitudes)

    print("Building '%s' model..." % args.config_name)
    model = MusicVae(config_name=args.config_name, batch_size=batch_size)
    model.build()

    with tf.Session() as sess:
        print("Loading model...")
        model.restore(tf_session=sess, path=args.checkpoint)

        latents = np.random.randn(args.num_samples, model.latent_dim).astype(np.float32)
        print("Sampled %d latent vectors." % args.num_samples)
        if len(magnitudes) > 0:
            rnd_latents = np.copy(latents)
            latents = np.zeros([batch_size, model.latent_dim])
            latent_idx = 0
            for i in range(args.num_samples):
                for magnitude in magnitudes:
                    latents[latent_idx] = rnd_latents[i] * magnitude
                    latent_idx += 1
            print("Scaled latent vectors by %s." % magnitudes)

        print("Generating music samples...")
        results = model.sample(tf_session=sess, latents=latents, temperature=args.temperature)

        if args.export or args.plot:
            num_notes = np.zeros([len(magnitudes), model.music_length + 1])
            str_datetime = time.strftime('%Y-%m-%d_%H%M%S')

            for i, audio_tensor in enumerate(results):
                if args.export:
                    latent_idx = np.floor(i / args.num_samples) if len(magnitudes) > 0 else i
                    midi_path = '%s_%s-%03d.mid' % (args.config_name, str_datetime, latent_idx)
                    if len(magnitudes) > 0:
                        midi_path = '%s_%s-%03d-mag%.2f.mid' % (args.config_name, str_datetime, latent_idx, magnitudes[(i % len(magnitudes))])
                    model.save_midi(audio_tensor, os.path.join(args.output_dir, midi_path))
                if args.plot:
                    note_seq = model._config.data_converter.to_items([audio_tensor])[0]
                    num_notes[(i % len(magnitudes)), len(note_seq.notes)] += 1
                    # print("%d-%.2f: %d notes" % (i, magnitudes[(i % len(magnitudes))], len(note_seq.notes)))
            if args.export: print("Saved %d MIDI files to '%s'." % (len(results), args.output_dir))

            if args.plot:
                print("Plotting...")
                fig, ax = plt.subplots(ncols=1, nrows=len(magnitudes), figsize=[num_notes.shape[1]/2, len(magnitudes)])
                ims = []
                for mi, magnitude in enumerate(magnitudes):
                    ims.append(ax[mi].imshow(np.reshape(num_notes[mi]/np.sum(num_notes[mi]), [1, num_notes.shape[1]]), origin='lower', aspect='auto', cmap='inferno'))
                    for n in range(num_notes.shape[1]):
                        value = round(num_notes[mi][n]/np.sum(num_notes[mi]), 2)
                        color = 'w' if num_notes[mi][n] < np.max(num_notes[mi]) * .85 else 'indigo'
                        ax[mi].text(n, 0, value, va='center', ha='center', color=color)
                    ax[mi].label_outer()
                    ax[mi].get_yaxis().set_visible(False)
                    ax[mi].set_xticks([t for t in range(num_notes.shape[1]) if t%4 == 0])
                    if magnitude == 1.:
                        ax[mi].set_title(str(magnitude), loc='right', x=-.01, y=.3, weight='bold')
                    else:
                        ax[mi].set_title(str(magnitude), loc='right', x=-.01, y=.3, fontsize=12)

                fig.tight_layout()
                fig.savefig(os.path.join(args.output_dir, 'num_notes-%s.pdf' % str_datetime))
                plt.show()