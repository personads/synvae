import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time

import tensorflow as tf

from magenta import music

from models.auditive import MusicVae

# set up arguments
config_name = 'hierdec-mel_16bar'
checkpoint_path = '/mnt/d/thesis/models/hierdec-mel_16bar.ckpt'
output_dir = '/mnt/d/thesis/exp/music_vae_dbg_mel16'
num_samples = 5
temperature = 0.5

print("Building model...")
model = MusicVae(config_name=config_name, batch_size=num_samples)
model.build()

with tf.Session() as sess:
    print("Loading model...")
    model.restore(tf_session=sess, path=checkpoint_path)

    print("Drawing samples...")
    results = model.sample(tf_session=sess, num_samples=num_samples, temperature=temperature)

    date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
    basename = os.path.join(
        output_dir,
        '%s_%s_%s-*-of-%03d.mid' %
        (config_name, 'sample', date_and_time, num_samples))
    print('Outputting %d files as `%s`...' % (num_samples, basename))

    for i, ns in enumerate(results):
        music.sequence_proto_to_midi_file(ns, basename.replace('*', '%03d' % i))

print('Done.')