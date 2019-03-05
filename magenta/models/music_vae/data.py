# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MusicVAE data library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import copy
import functools
import itertools

import magenta.music as mm
from magenta.music import chords_lib
from magenta.music import drums_encoder_decoder
from magenta.music import sequences_lib
from magenta.protobuf import music_pb2
import numpy as np
import tensorflow as tf

PIANO_MIN_MIDI_PITCH = 21
PIANO_MAX_MIDI_PITCH = 108
MIN_MIDI_PITCH = 0
MAX_MIDI_PITCH = 127
MIDI_PITCHES = 128

MAX_INSTRUMENT_NUMBER = 127

MEL_PROGRAMS = range(0, 32)  # piano, chromatic percussion, organ, guitar
BASS_PROGRAMS = range(32, 40)
ELECTRIC_BASS_PROGRAM = 33

# 9 classes: kick, snare, closed_hh, open_hh, low_tom, mid_tom, hi_tom, crash,
# ride
REDUCED_DRUM_PITCH_CLASSES = drums_encoder_decoder.DEFAULT_DRUM_TYPE_PITCHES
# 61 classes: full General MIDI set
FULL_DRUM_PITCH_CLASSES = [
    [p] for c in drums_encoder_decoder.DEFAULT_DRUM_TYPE_PITCHES for p in c]

OUTPUT_VELOCITY = 80

CHORD_SYMBOL = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL


def _maybe_pad_seqs(seqs, dtype):
  """Pads sequences to match the longest and returns as a numpy array."""
  if not len(seqs):  # pylint:disable=g-explicit-length-test,len-as-condition
    return np.zeros((0, 0, 0), dtype)
  lengths = [len(s) for s in seqs]
  if len(set(lengths)) == 1:
    return np.array(seqs, dtype)
  else:
    length = max(lengths)
    return (np.array([np.pad(s, [(0, length - len(s)), (0, 0)], mode='constant')
                      for s in seqs], dtype))


def _extract_instrument(note_sequence, instrument):
  extracted_ns = copy.copy(note_sequence)
  del extracted_ns.notes[:]
  extracted_ns.notes.extend(
      n for n in note_sequence.notes if n.instrument == instrument)
  return extracted_ns


def np_onehot(indices, depth, dtype=np.bool):
  """Converts 1D array of indices to a one-hot 2D array with given depth."""
  onehot_seq = np.zeros((len(indices), depth), dtype=dtype)
  onehot_seq[np.arange(len(indices)), indices] = 1.0
  return onehot_seq


class NoteSequenceAugmenter(object):
  """Class for augmenting NoteSequences.

  Args:
    transpose_range: A tuple containing the inclusive, integer range of
        transpose amounts to sample from. If None, no transposition is applied.
    stretch_range: A tuple containing the inclusive, float range of stretch
        amounts to sample from.
  Returns:
    The augmented NoteSequence.
  """

  def __init__(self, transpose_range=None, stretch_range=None):
    self._transpose_range = transpose_range
    self._stretch_range = stretch_range

  def augment(self, note_sequence):
    """Python implementation that augments the NoteSequence.

    Args:
      note_sequence: A NoteSequence proto to be augmented.

    Returns:
      The randomly augmented NoteSequence.
    """
    transpose_min, transpose_max = (
        self._transpose_range if self._transpose_range else (0, 0))
    stretch_min, stretch_max = (
        self._stretch_range if self._stretch_range else (1.0, 1.0))

    return sequences_lib.augment_note_sequence(
        note_sequence,
        stretch_min,
        stretch_max,
        transpose_min,
        transpose_max,
        delete_out_of_range_notes=True)

  def tf_augment(self, note_sequence_scalar):
    """TF op that augments the NoteSequence."""
    def _augment_str(note_sequence_str):
      note_sequence = music_pb2.NoteSequence.FromString(note_sequence_str)
      augmented_ns = self.augment(note_sequence)
      return [augmented_ns.SerializeToString()]

    augmented_note_sequence_scalar = tf.py_func(
        _augment_str,
        [note_sequence_scalar],
        tf.string,
        stateful=False,
        name='augment')
    augmented_note_sequence_scalar.set_shape(())
    return augmented_note_sequence_scalar


class ConverterTensors(collections.namedtuple(
    'ConverterTensors', ['inputs', 'outputs', 'controls', 'lengths'])):
  """Tuple of tensors output by `to_tensors` method in converters.

  Attributes:
    inputs: Input tensors to feed to the encoder.
    outputs: Output tensors to feed to the decoder.
    controls: (Optional) tensors to use as controls for both encoding and
        decoding.
    lengths: Length of each input/output/control sequence.
  """

  def __new__(cls, inputs=None, outputs=None, controls=None, lengths=None):
    if inputs is None:
      inputs = []
    if outputs is None:
      outputs = []
    if lengths is None:
      lengths = [len(i) for i in inputs]
    if not controls:
      controls = [np.zeros([l, 0]) for l in lengths]
    return super(ConverterTensors, cls).__new__(
        cls, inputs, outputs, controls, lengths)


class BaseConverter(object):
  """Base class for data converters between items and tensors.

  Inheriting classes must implement the following abstract methods:
    -`_to_tensors`
    -`_to_items`
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self, input_depth, input_dtype, output_depth, output_dtype,
               control_depth=0, control_dtype=np.bool, end_token=None,
               max_tensors_per_item=None,
               str_to_item_fn=lambda s: s, length_shape=()):
    """Initializes BaseConverter.

    Args:
      input_depth: Depth of final dimension of input (encoder) tensors.
      input_dtype: DType of input (encoder) tensors.
      output_depth: Depth of final dimension of output (decoder) tensors.
      output_dtype: DType of output (decoder) tensors.
      control_depth: Depth of final dimension of control tensors, or zero if not
          conditioning on control tensors.
      control_dtype: DType of control tensors.
      end_token: Optional end token.
      max_tensors_per_item: The maximum number of outputs to return for each
          input.
      str_to_item_fn: Callable to convert raw string input into an item for
          conversion.
      length_shape: Shape of length returned by `to_tensor`.
    """
    self._input_depth = input_depth
    self._input_dtype = input_dtype
    self._output_depth = output_depth
    self._output_dtype = output_dtype
    self._control_depth = control_depth
    self._control_dtype = control_dtype
    self._end_token = end_token
    self._max_tensors_per_input = max_tensors_per_item
    self._str_to_item_fn = str_to_item_fn
    self._is_training = False
    self._length_shape = length_shape

  @property
  def is_training(self):
    return self._is_training

  @property
  def str_to_item_fn(self):
    return self._str_to_item_fn

  @is_training.setter
  def is_training(self, value):
    self._is_training = value

  @property
  def max_tensors_per_item(self):
    return self._max_tensors_per_input

  @max_tensors_per_item.setter
  def max_tensors_per_item(self, value):
    self._max_tensors_per_input = value

  @property
  def end_token(self):
    """End token, or None."""
    return self._end_token

  @property
  def input_depth(self):
    """Dimension of inputs (to encoder) at each timestep of the sequence."""
    return self._input_depth

  @property
  def input_dtype(self):
    """DType of inputs (to encoder)."""
    return self._input_dtype

  @property
  def output_depth(self):
    """Dimension of outputs (from decoder) at each timestep of the sequence."""
    return self._output_depth

  @property
  def output_dtype(self):
    """DType of outputs (from decoder)."""
    return self._output_dtype

  @property
  def control_depth(self):
    """Dimension of control inputs at each timestep of the sequence."""
    return self._control_depth

  @property
  def control_dtype(self):
    """DType of control inputs."""
    return self._control_dtype

  @property
  def length_shape(self):
    """Shape of length returned by `to_tensor`."""
    return self._length_shape

  @abc.abstractmethod
  def _to_tensors(self, item):
    """Implementation that converts item into encoder/decoder tensors.

    Args:
     item: Item to convert.

    Returns:
      A ConverterTensors struct containing encoder inputs, decoder outputs,
      (optional) control tensors used for both encoding and decoding, and
      sequence lengths.
    """
    pass

  @abc.abstractmethod
  def _to_items(self, samples, controls=None):
    """Implementation that decodes model samples into list of items."""
    pass

  def _maybe_sample_outputs(self, outputs):
    """If should limit outputs, returns up to limit (randomly if training)."""
    if (not self.max_tensors_per_item or
        len(outputs) <= self.max_tensors_per_item):
      return outputs
    if self.is_training:
      indices = set(np.random.choice(
          len(outputs), size=self.max_tensors_per_item, replace=False))
      return [outputs[i] for i in indices]
    else:
      return outputs[:self.max_tensors_per_item]

  def to_tensors(self, item):
    """Python method that converts `item` into list of tensors."""
    tensors = self._to_tensors(item)
    sampled_results = self._maybe_sample_outputs(list(zip(*tensors)))
    if sampled_results:
      return ConverterTensors(*zip(*sampled_results))
    else:
      return ConverterTensors()

  def _combine_to_tensor_results(self, to_tensor_results):
    """Combines the results of multiple to_tensors calls into one result."""
    results = []
    for result in to_tensor_results:
      results.extend(zip(*result))
    sampled_results = self._maybe_sample_outputs(results)
    if sampled_results:
      return ConverterTensors(*zip(*sampled_results))
    else:
      return ConverterTensors()

  def to_items(self, samples, controls=None):
    """Python method that decodes samples into list of items."""
    if controls is None:
      return self._to_items(samples)
    else:
      return self._to_items(samples, controls)

  def tf_to_tensors(self, item_scalar):
    """TensorFlow op that converts item into output tensors.

    Sequences will be padded to match the length of the longest.

    Args:
      item_scalar: A scalar of type tf.String containing the raw item to be
          converted to tensors.

    Returns:
      inputs: A Tensor, shaped [num encoded seqs, max(lengths), input_depth],
          containing the padded input encodings.
      outputs: A Tensor, shaped [num encoded seqs, max(lengths), output_depth],
          containing the padded output encodings resulting from the input.
      controls: A Tensor, shaped
          [num encoded seqs, max(lengths), control_depth], containing the padded
          control encodings.
      lengths: A tf.int32 Tensor, shaped [num encoded seqs], containing the
        unpadded lengths of the tensor sequences resulting from the input.
    """
    def _convert_and_pad(item_str):
      item = self.str_to_item_fn(item_str)  # pylint:disable=not-callable
      tensors = self.to_tensors(item)
      inputs = _maybe_pad_seqs(tensors.inputs, self.input_dtype)
      outputs = _maybe_pad_seqs(tensors.outputs, self.output_dtype)
      controls = _maybe_pad_seqs(tensors.controls, self.control_dtype)
      return inputs, outputs, controls, np.array(tensors.lengths, np.int32)
    inputs, outputs, controls, lengths = tf.py_func(
        _convert_and_pad,
        [item_scalar],
        [self.input_dtype, self.output_dtype, self.control_dtype, tf.int32],
        stateful=False,
        name='convert_and_pad')
    inputs.set_shape([None, None, self.input_depth])
    outputs.set_shape([None, None, self.output_depth])
    controls.set_shape([None, None, self.control_depth])
    lengths.set_shape([None] + list(self.length_shape))
    return inputs, outputs, controls, lengths


def preprocess_notesequence(note_sequence, presplit_on_time_changes):
  """Preprocesses a single NoteSequence, resulting in multiple sequences."""
  if presplit_on_time_changes:
    note_sequences = sequences_lib.split_note_sequence_on_time_changes(
        note_sequence)
  else:
    note_sequences = [note_sequence]

  return note_sequences


class BaseNoteSequenceConverter(BaseConverter):
  """Base class for NoteSequence data converters.

  Inheriting classes must implement the following abstract methods:
    -`_to_tensors`
    -`_to_notesequences`
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self, input_depth, input_dtype, output_depth, output_dtype,
               control_depth=0, control_dtype=np.bool, end_token=None,
               presplit_on_time_changes=True,
               max_tensors_per_notesequence=None):
    """Initializes BaseNoteSequenceConverter.

    Args:
      input_depth: Depth of final dimension of input (encoder) tensors.
      input_dtype: DType of input (encoder) tensors.
      output_depth: Depth of final dimension of output (decoder) tensors.
      output_dtype: DType of output (decoder) tensors.
      control_depth: Depth of final dimension of control tensors, or zero if not
          conditioning on control tensors.
      control_dtype: DType of control tensors.
      end_token: Optional end token.
      presplit_on_time_changes: Whether to split NoteSequence on time changes
        before converting.
      max_tensors_per_notesequence: The maximum number of outputs to return
        for each NoteSequence.
    """
    super(BaseNoteSequenceConverter, self).__init__(
        input_depth, input_dtype, output_depth, output_dtype,
        control_depth, control_dtype, end_token,
        max_tensors_per_item=max_tensors_per_notesequence,
        str_to_item_fn=music_pb2.NoteSequence.FromString)

    self._presplit_on_time_changes = presplit_on_time_changes

  @property
  def max_tensors_per_notesequence(self):
    return self.max_tensors_per_item

  @max_tensors_per_notesequence.setter
  def max_tensors_per_notesequence(self, value):
    self.max_tensors_per_item = value

  @abc.abstractmethod
  def _to_notesequences(self, samples, controls=None):
    """Implementation that decodes model samples into list of NoteSequences."""
    pass

  def to_notesequences(self, samples, controls=None):
    """Python method that decodes samples into list of NoteSequences."""
    return self._to_items(samples, controls)

  def to_tensors(self, note_sequence):
    """Python method that converts `note_sequence` into list of tensors."""
    note_sequences = preprocess_notesequence(
        note_sequence, self._presplit_on_time_changes)

    results = []
    for ns in note_sequences:
      results.append(super(BaseNoteSequenceConverter, self).to_tensors(ns))
    return self._combine_to_tensor_results(results)

  def _to_items(self, samples, controls=None):
    """Python method that decodes samples into list of NoteSequences."""
    if controls is None:
      return self._to_notesequences(samples)
    else:
      return self._to_notesequences(samples, controls)


class LegacyEventListOneHotConverter(BaseNoteSequenceConverter):
  """Converts NoteSequences using legacy OneHotEncoding framework.

  Quantizes the sequences, extracts event lists in the requested size range,
  uniquifies, and converts to encoding. Uses the OneHotEncoding's
  output encoding for both the input and output.

  Args:
    event_list_fn: A function that returns a new EventSequence.
    event_extractor_fn: A function for extracing events into EventSequences. The
      sole input should be the quantized NoteSequence.
    legacy_encoder_decoder: An instantiated OneHotEncoding object to use.
    add_end_token: Whether or not to add an end token. Recommended to be False
      for fixed-length outputs.
    slice_bars: Optional size of window to slide over raw event lists after
      extraction.
    steps_per_quarter: The number of quantization steps per quarter note.
      Mututally exclusive with `steps_per_second`.
    steps_per_second: The number of quantization steps per second.
      Mututally exclusive with `steps_per_quarter`.
    quarters_per_bar: The number of quarter notes per bar.
    pad_to_total_time: Pads each input/output tensor to the total time of the
      NoteSequence.
    max_tensors_per_notesequence: The maximum number of outputs to return
      for each NoteSequence.
    presplit_on_time_changes: Whether to split NoteSequence on time changes
      before converting.
    chord_encoding: An instantiated OneHotEncoding object to use for encoding
      chords on which to condition, or None if not conditioning on chords.
  """

  def __init__(self, event_list_fn, event_extractor_fn,
               legacy_encoder_decoder, add_end_token=False, slice_bars=None,
               slice_steps=None, steps_per_quarter=None, steps_per_second=None,
               quarters_per_bar=4, pad_to_total_time=False,
               max_tensors_per_notesequence=None,
               presplit_on_time_changes=True, chord_encoding=None):
    if (steps_per_quarter, steps_per_second).count(None) != 1:
      raise ValueError(
          'Exactly one of `steps_per_quarter` and `steps_per_second` should be '
          'provided.')
    if (slice_bars, slice_steps).count(None) == 0:
      raise ValueError(
          'At most one of `slice_bars` and `slice_steps` should be provided.')
    self._event_list_fn = event_list_fn
    self._event_extractor_fn = event_extractor_fn
    self._legacy_encoder_decoder = legacy_encoder_decoder
    self._chord_encoding = chord_encoding
    self._steps_per_quarter = steps_per_quarter
    if steps_per_quarter:
      self._steps_per_bar = steps_per_quarter * quarters_per_bar
    self._steps_per_second = steps_per_second
    if slice_bars:
      self._slice_steps = self._steps_per_bar * slice_bars
    else:
      self._slice_steps = slice_steps
    self._pad_to_total_time = pad_to_total_time

    depth = legacy_encoder_decoder.num_classes + add_end_token
    control_depth = (
        chord_encoding.num_classes if chord_encoding is not None else 0)
    super(LegacyEventListOneHotConverter, self).__init__(
        input_depth=depth,
        input_dtype=np.bool,
        output_depth=depth,
        output_dtype=np.bool,
        control_depth=control_depth,
        control_dtype=np.bool,
        end_token=legacy_encoder_decoder.num_classes if add_end_token else None,
        presplit_on_time_changes=presplit_on_time_changes,
        max_tensors_per_notesequence=max_tensors_per_notesequence)

  def _to_tensors(self, note_sequence):
    """Converts NoteSequence to unique, one-hot tensor sequences."""
    try:
      if self._steps_per_quarter:
        quantized_sequence = mm.quantize_note_sequence(
            note_sequence, self._steps_per_quarter)
        if (mm.steps_per_bar_in_quantized_sequence(quantized_sequence) !=
            self._steps_per_bar):
          return ConverterTensors()
      else:
        quantized_sequence = mm.quantize_note_sequence_absolute(
            note_sequence, self._steps_per_second)
    except (mm.BadTimeSignatureError, mm.NonIntegerStepsPerBarError,
            mm.NegativeTimeError) as e:
      return ConverterTensors()

    if self._chord_encoding and not any(
        ta.annotation_type == CHORD_SYMBOL
        for ta in quantized_sequence.text_annotations):
      # We are conditioning on chords but sequence does not have chords. Try to
      # infer them.
      try:
        mm.infer_chords_for_sequence(quantized_sequence)
      except mm.ChordInferenceError:
        return ConverterTensors()

    event_lists, unused_stats = self._event_extractor_fn(quantized_sequence)
    if self._pad_to_total_time:
      for e in event_lists:
        e.set_length(len(e) + e.start_step, from_left=True)
        e.set_length(quantized_sequence.total_quantized_steps)
    if self._slice_steps:
      sliced_event_lists = []
      for l in event_lists:
        for i in range(self._slice_steps, len(l) + 1, self._steps_per_bar):
          sliced_event_lists.append(l[i - self._slice_steps: i])
    else:
      sliced_event_lists = event_lists

    if self._chord_encoding:
      try:
        sliced_chord_lists = chords_lib.event_list_chords(
            quantized_sequence, sliced_event_lists)
      except chords_lib.CoincidentChordsError:
        return ConverterTensors()
      sliced_event_lists = [zip(el, cl) for el, cl in zip(sliced_event_lists,
                                                          sliced_chord_lists)]

    # TODO(adarob): Consider handling the fact that different event lists can
    # be mapped to identical tensors by the encoder_decoder (e.g., Drums).

    unique_event_tuples = list(set(tuple(l) for l in sliced_event_lists))
    unique_event_tuples = self._maybe_sample_outputs(unique_event_tuples)

    if not unique_event_tuples:
      return ConverterTensors()

    control_seqs = []
    if self._chord_encoding:
      unique_event_tuples, unique_chord_tuples = zip(
          *[zip(*t) for t in unique_event_tuples if t])
      for t in unique_chord_tuples:
        try:
          chord_tokens = [self._chord_encoding.encode_event(e) for e in t]
          if self.end_token:
            # Repeat the last chord instead of using a special token; otherwise
            # the model may learn to rely on the special token to detect
            # endings.
            if chord_tokens:
              chord_tokens.append(chord_tokens[-1])
            else:
              chord_tokens.append(
                  self._chord_encoding.encode_event(mm.NO_CHORD))
        except (mm.ChordSymbolError, mm.ChordEncodingError):
          return ConverterTensors()
        control_seqs.append(
            np_onehot(chord_tokens, self.control_depth, self.control_dtype))

    seqs = []
    for t in unique_event_tuples:
      seqs.append(np_onehot(
          [self._legacy_encoder_decoder.encode_event(e) for e in t] +
          ([] if self.end_token is None else [self.end_token]),
          self.output_depth, self.output_dtype))

    return ConverterTensors(inputs=seqs, outputs=seqs, controls=control_seqs)

  def _to_notesequences(self, samples, controls=None):
    output_sequences = []
    for i, sample in enumerate(samples):
      s = np.argmax(sample, axis=-1)
      if self.end_token is not None and self.end_token in s.tolist():
        end_index = s.tolist().index(self.end_token)
      else:
        end_index = len(s)
      s = s[:end_index]
      event_list = self._event_list_fn()
      for e in s:
        assert e != self.end_token
        event_list.append(self._legacy_encoder_decoder.decode_event(e))
      if self._steps_per_quarter:
        qpm = mm.DEFAULT_QUARTERS_PER_MINUTE
        seconds_per_step = 60.0 / (self._steps_per_quarter * qpm)
        sequence = event_list.to_sequence(velocity=OUTPUT_VELOCITY, qpm=qpm)
      else:
        seconds_per_step = 1.0 / self._steps_per_second
        sequence = event_list.to_sequence(velocity=OUTPUT_VELOCITY)
      if self._chord_encoding and controls is not None:
        chords = [self._chord_encoding.decode_event(e)
                  for e in np.argmax(controls[i], axis=-1)[:end_index]]
        chord_times = [step * seconds_per_step for step in event_list.steps]
        chords_lib.add_chords_to_sequence(sequence, chords, chord_times)
      output_sequences.append(sequence)
    return output_sequences


class OneHotMelodyConverter(LegacyEventListOneHotConverter):
  """Converter for legacy MelodyOneHotEncoding.

  Args:
    min_pitch: The minimum pitch to model. Those below this value will be
      ignored.
    max_pitch: The maximum pitch to model. Those above this value will be
      ignored.
    valid_programs: Optional set of program numbers to allow.
    skip_polyphony: Whether to skip polyphonic instruments. If False, the
      highest pitch will be taken in polyphonic sections.
    max_bars: Optional maximum number of bars per extracted melody, before
      slicing.
    slice_bars: Optional size of window to slide over raw Melodies after
      extraction.
    gap_bars: If this many bars or more of non-events follow a note event, the
       melody is ended. Disabled when set to 0 or None.
    steps_per_quarter: The number of quantization steps per quarter note.
    quarters_per_bar: The number of quarter notes per bar.
    pad_to_total_time: Pads each input/output tensor to the total time of the
      NoteSequence.
    add_end_token: Whether to add an end token at the end of each sequence.
    max_tensors_per_notesequence: The maximum number of outputs to return
      for each NoteSequence.
    chord_encoding: An instantiated OneHotEncoding object to use for encoding
      chords on which to condition, or None if not conditioning on chords.
  """

  def __init__(self, min_pitch=PIANO_MIN_MIDI_PITCH,
               max_pitch=PIANO_MAX_MIDI_PITCH, valid_programs=None,
               skip_polyphony=False, max_bars=None, slice_bars=None,
               gap_bars=1.0, steps_per_quarter=4, quarters_per_bar=4,
               add_end_token=False, pad_to_total_time=False,
               max_tensors_per_notesequence=5, presplit_on_time_changes=True,
               chord_encoding=None):
    self._min_pitch = min_pitch
    self._max_pitch = max_pitch
    self._valid_programs = valid_programs
    steps_per_bar = steps_per_quarter * quarters_per_bar
    max_steps_truncate = steps_per_bar * max_bars if max_bars else None

    def melody_fn():
      return mm.Melody(
          steps_per_bar=steps_per_bar, steps_per_quarter=steps_per_quarter)
    melody_extractor_fn = functools.partial(
        mm.extract_melodies,
        min_bars=1,
        gap_bars=gap_bars or float('inf'),
        max_steps_truncate=max_steps_truncate,
        min_unique_pitches=1,
        ignore_polyphonic_notes=not skip_polyphony,
        pad_end=True)
    super(OneHotMelodyConverter, self).__init__(
        melody_fn,
        melody_extractor_fn,
        mm.MelodyOneHotEncoding(min_pitch, max_pitch + 1),
        add_end_token=add_end_token,
        slice_bars=slice_bars,
        pad_to_total_time=pad_to_total_time,
        steps_per_quarter=steps_per_quarter,
        quarters_per_bar=quarters_per_bar,
        max_tensors_per_notesequence=max_tensors_per_notesequence,
        presplit_on_time_changes=presplit_on_time_changes,
        chord_encoding=chord_encoding)

  def _to_tensors(self, note_sequence):
    def is_valid(note):
      if (self._valid_programs is not None and
          note.program not in self._valid_programs):
        return False
      return self._min_pitch <= note.pitch <= self._max_pitch
    notes = list(note_sequence.notes)
    del note_sequence.notes[:]
    note_sequence.notes.extend([n for n in notes if is_valid(n)])
    return super(OneHotMelodyConverter, self)._to_tensors(note_sequence)


class DrumsConverter(BaseNoteSequenceConverter):
  """Converter for legacy drums with either pianoroll or one-hot tensors.

  Inputs/outputs are either a "pianoroll"-like encoding of all possible drum
  hits at a given step, or a one-hot encoding of the pianoroll.

  The "roll" input encoding includes a final NOR bit (after the optional end
  token).

  Args:
    max_bars: Optional maximum number of bars per extracted drums, before
      slicing.
    slice_bars: Optional size of window to slide over raw Melodies after
      extraction.
    gap_bars: If this many bars or more follow a non-empty drum event, the
      drum track is ended. Disabled when set to 0 or None.
    pitch_classes: A collection of collections, with each sub-collection
      containing the set of pitches representing a single class to group by. By
      default, groups valid drum pitches into 9 different classes.
    add_end_token: Whether or not to add an end token. Recommended to be False
      for fixed-length outputs.
    steps_per_quarter: The number of quantization steps per quarter note.
    quarters_per_bar: The number of quarter notes per bar.
    pad_to_total_time: Pads each input/output tensor to the total time of the
      NoteSequence.
    roll_input: Whether to use a pianoroll-like representation as the input
      instead of a one-hot encoding.
    roll_output: Whether to use a pianoroll-like representation as the output
      instead of a one-hot encoding.
    max_tensors_per_notesequence: The maximum number of outputs to return
      for each NoteSequence.
    presplit_on_time_changes: Whether to split NoteSequence on time changes
      before converting.
  """

  def __init__(self, max_bars=None, slice_bars=None, gap_bars=1.0,
               pitch_classes=None, add_end_token=False, steps_per_quarter=4,
               quarters_per_bar=4, pad_to_total_time=False, roll_input=False,
               roll_output=False, max_tensors_per_notesequence=5,
               presplit_on_time_changes=True):
    self._pitch_classes = pitch_classes or REDUCED_DRUM_PITCH_CLASSES
    self._pitch_class_map = {
        p: i for i, pitches in enumerate(self._pitch_classes) for p in pitches}

    self._steps_per_quarter = steps_per_quarter
    self._steps_per_bar = steps_per_quarter * quarters_per_bar
    self._slice_steps = self._steps_per_bar * slice_bars if slice_bars else None
    self._pad_to_total_time = pad_to_total_time
    self._roll_input = roll_input
    self._roll_output = roll_output

    self._drums_extractor_fn = functools.partial(
        mm.extract_drum_tracks,
        min_bars=1,
        gap_bars=gap_bars or float('inf'),
        max_steps_truncate=self._steps_per_bar * max_bars if max_bars else None,
        pad_end=True)

    num_classes = len(self._pitch_classes)

    self._pr_encoder_decoder = mm.PianorollEncoderDecoder(
        input_size=num_classes + add_end_token)
    # Use pitch classes as `drum_type_pitches` since we have already done the
    # mapping.
    self._oh_encoder_decoder = mm.MultiDrumOneHotEncoding(
        drum_type_pitches=[(i,) for i in range(num_classes)])

    if self._roll_output:
      output_depth = num_classes + add_end_token
    else:
      output_depth = self._oh_encoder_decoder.num_classes + add_end_token

    if self._roll_input:
      input_depth = num_classes + 1 + add_end_token
    else:
      input_depth = self._oh_encoder_decoder.num_classes + add_end_token

    super(DrumsConverter, self).__init__(
        input_depth=input_depth,
        input_dtype=np.bool,
        output_depth=output_depth,
        output_dtype=np.bool,
        end_token=output_depth - 1 if add_end_token else None,
        presplit_on_time_changes=presplit_on_time_changes,
        max_tensors_per_notesequence=max_tensors_per_notesequence)

  def _to_tensors(self, note_sequence):
    """Converts NoteSequence to unique sequences."""
    try:
      quantized_sequence = mm.quantize_note_sequence(
          note_sequence, self._steps_per_quarter)
      if (mm.steps_per_bar_in_quantized_sequence(quantized_sequence) !=
          self._steps_per_bar):
        return ConverterTensors()
    except (mm.BadTimeSignatureError, mm.NonIntegerStepsPerBarError,
            mm.NegativeTimeError) as e:
      return ConverterTensors()

    new_notes = []
    for n in quantized_sequence.notes:
      if not n.is_drum:
        continue
      if n.pitch not in self._pitch_class_map:
        continue
      n.pitch = self._pitch_class_map[n.pitch]
      new_notes.append(n)
    del quantized_sequence.notes[:]
    quantized_sequence.notes.extend(new_notes)

    event_lists, unused_stats = self._drums_extractor_fn(quantized_sequence)

    if self._pad_to_total_time:
      for e in event_lists:
        e.set_length(len(e) + e.start_step, from_left=True)
        e.set_length(quantized_sequence.total_quantized_steps)
    if self._slice_steps:
      sliced_event_tuples = []
      for l in event_lists:
        for i in range(self._slice_steps, len(l) + 1, self._steps_per_bar):
          sliced_event_tuples.append(tuple(l[i - self._slice_steps: i]))
    else:
      sliced_event_tuples = [tuple(l) for l in event_lists]

    unique_event_tuples = list(set(sliced_event_tuples))
    unique_event_tuples = self._maybe_sample_outputs(unique_event_tuples)

    rolls = []
    oh_vecs = []
    for t in unique_event_tuples:
      if self._roll_input or self._roll_output:
        if self.end_token is not None:
          t_roll = list(t) + [(self._pr_encoder_decoder.input_size - 1,)]
        else:
          t_roll = t
        rolls.append(np.vstack([
            self._pr_encoder_decoder.events_to_input(t_roll, i).astype(np.bool)
            for i in range(len(t_roll))]))
      if not (self._roll_input and self._roll_output):
        labels = [self._oh_encoder_decoder.encode_event(e) for e in t]
        if self.end_token is not None:
          labels += [self._oh_encoder_decoder.num_classes]
        oh_vecs.append(np_onehot(
            labels,
            self._oh_encoder_decoder.num_classes + (self.end_token is not None),
            np.bool))

    if self._roll_input:
      input_seqs = [
          np.append(roll, np.expand_dims(np.all(roll == 0, axis=1), axis=1),
                    axis=1) for roll in rolls]
    else:
      input_seqs = oh_vecs

    output_seqs = rolls if self._roll_output else oh_vecs

    return ConverterTensors(inputs=input_seqs, outputs=output_seqs)

  def _to_notesequences(self, samples):
    output_sequences = []
    for s in samples:
      if self._roll_output:
        if self.end_token is not None:
          end_i = np.where(s[:, self.end_token])
          if len(end_i):  # pylint: disable=g-explicit-length-test,len-as-condition
            s = s[:end_i[0]]
        events_list = [frozenset(np.where(e)[0]) for e in s]
      else:
        s = np.argmax(s, axis=-1)
        if self.end_token is not None and self.end_token in s:
          s = s[:s.tolist().index(self.end_token)]
        events_list = [self._oh_encoder_decoder.decode_event(e) for e in s]
      # Map classes to exemplars.
      events_list = [
          frozenset(self._pitch_classes[c][0] for c in e) for e in events_list]
      track = mm.DrumTrack(
          events=events_list, steps_per_bar=self._steps_per_bar,
          steps_per_quarter=self._steps_per_quarter)
      output_sequences.append(track.to_sequence(velocity=OUTPUT_VELOCITY))
    return output_sequences


class TrioConverter(BaseNoteSequenceConverter):
  """Converts to/from 3-part (mel, drums, bass) multi-one-hot events.

  Extracts overlapping segments with melody, drums, and bass (determined by
  program number) and concatenates one-hot tensors from OneHotMelodyConverter
  and OneHotDrumsConverter. Takes the cross products from the sets of
  instruments of each type.

  Args:
    slice_bars: Optional size of window to slide over full converted tensor.
    gap_bars: The number of consecutive empty bars to allow for any given
      instrument. Note that this number is effectively doubled for internal
      gaps.
    max_bars: Optional maximum number of bars per extracted sequence, before
      slicing.
    steps_per_quarter: The number of quantization steps per quarter note.
    quarters_per_bar: The number of quarter notes per bar.
    max_tensors_per_notesequence: The maximum number of outputs to return
      for each NoteSequence.
    chord_encoding: An instantiated OneHotEncoding object to use for encoding
      chords on which to condition, or None if not conditioning on chords.
  """

  class InstrumentType(object):
    UNK = 0
    MEL = 1
    BASS = 2
    DRUMS = 3
    INVALID = 4

  def __init__(
      self, slice_bars=None, gap_bars=2, max_bars=1024, steps_per_quarter=4,
      quarters_per_bar=4, max_tensors_per_notesequence=5, chord_encoding=None):
    self._melody_converter = OneHotMelodyConverter(
        gap_bars=None, steps_per_quarter=steps_per_quarter,
        pad_to_total_time=True, presplit_on_time_changes=False,
        max_tensors_per_notesequence=None, chord_encoding=chord_encoding)
    self._drums_converter = DrumsConverter(
        gap_bars=None, steps_per_quarter=steps_per_quarter,
        pad_to_total_time=True, presplit_on_time_changes=False,
        max_tensors_per_notesequence=None)
    self._slice_bars = slice_bars
    self._gap_bars = gap_bars
    self._max_bars = max_bars
    self._steps_per_quarter = steps_per_quarter
    self._steps_per_bar = steps_per_quarter * quarters_per_bar
    self._chord_encoding = chord_encoding

    self._split_output_depths = (
        self._melody_converter.output_depth,
        self._melody_converter.output_depth,
        self._drums_converter.output_depth)
    output_depth = sum(self._split_output_depths)

    self._program_map = dict(
        [(i, TrioConverter.InstrumentType.MEL) for i in MEL_PROGRAMS] +
        [(i, TrioConverter.InstrumentType.BASS) for i in BASS_PROGRAMS])

    super(TrioConverter, self).__init__(
        input_depth=output_depth,
        input_dtype=np.bool,
        output_depth=output_depth,
        output_dtype=np.bool,
        control_depth=self._melody_converter.control_depth,
        control_dtype=self._melody_converter.control_dtype,
        end_token=False,
        presplit_on_time_changes=True,
        max_tensors_per_notesequence=max_tensors_per_notesequence)

  def _to_tensors(self, note_sequence):
    try:
      quantized_sequence = mm.quantize_note_sequence(
          note_sequence, self._steps_per_quarter)
      if (mm.steps_per_bar_in_quantized_sequence(quantized_sequence) !=
          self._steps_per_bar):
        return ConverterTensors()
    except (mm.BadTimeSignatureError, mm.NonIntegerStepsPerBarError,
            mm.NegativeTimeError):
      return ConverterTensors()

    if self._chord_encoding and not any(
        ta.annotation_type == CHORD_SYMBOL
        for ta in quantized_sequence.text_annotations):
      # We are conditioning on chords but sequence does not have chords. Try to
      # infer them.
      try:
        mm.infer_chords_for_sequence(quantized_sequence)
      except mm.ChordInferenceError:
        return ConverterTensors()

      # The trio parts get extracted from the original NoteSequence, so copy the
      # inferred chords back to that one.
      for qta in quantized_sequence.text_annotations:
        if qta.annotation_type == CHORD_SYMBOL:
          ta = note_sequence.text_annotations.add()
          ta.annotation_type = CHORD_SYMBOL
          ta.time = qta.time
          ta.text = qta.text

    total_bars = int(
        np.ceil(quantized_sequence.total_quantized_steps / self._steps_per_bar))
    total_bars = min(total_bars, self._max_bars)

    # Assign an instrument class for each instrument, and compute its coverage.
    # If an instrument has multiple classes, it is considered INVALID.
    instrument_type = np.zeros(MAX_INSTRUMENT_NUMBER + 1, np.uint8)
    coverage = np.zeros((total_bars, MAX_INSTRUMENT_NUMBER + 1), np.bool)
    for note in quantized_sequence.notes:
      i = note.instrument
      if i > MAX_INSTRUMENT_NUMBER:
        tf.logging.warning('Skipping invalid instrument number: %d', i)
        continue
      if note.is_drum:
        inferred_type = self.InstrumentType.DRUMS
      else:
        inferred_type = self._program_map.get(
            note.program, self.InstrumentType.INVALID)
      if not instrument_type[i]:
        instrument_type[i] = inferred_type
      elif instrument_type[i] != inferred_type:
        instrument_type[i] = self.InstrumentType.INVALID

      start_bar = note.quantized_start_step // self._steps_per_bar
      end_bar = int(np.ceil(note.quantized_end_step / self._steps_per_bar))

      if start_bar >= total_bars:
        continue
      coverage[start_bar:min(end_bar, total_bars), i] = True

    # Group instruments by type.
    instruments_by_type = collections.defaultdict(list)
    for i, type_ in enumerate(instrument_type):
      if type_ not in (self.InstrumentType.UNK, self.InstrumentType.INVALID):
        instruments_by_type[type_].append(i)
    if len(instruments_by_type) < 3:
      # This NoteSequence doesn't have all 3 types.
      return ConverterTensors()

    # Encode individual instruments.
    # Set total time so that instruments will be padded correctly.
    note_sequence.total_time = (
        total_bars * self._steps_per_bar *
        60 / note_sequence.tempos[0].qpm / self._steps_per_quarter)
    encoded_instruments = {}
    encoded_chords = None
    for i in (instruments_by_type[self.InstrumentType.MEL] +
              instruments_by_type[self.InstrumentType.BASS]):
      tensors = self._melody_converter.to_tensors(
          _extract_instrument(note_sequence, i))
      if tensors.outputs:
        encoded_instruments[i] = tensors.outputs[0]
        if encoded_chords is None:
          encoded_chords = tensors.controls[0]
        elif not np.array_equal(encoded_chords, tensors.controls[0]):
          tf.logging.warning('Trio chords disagreement between instruments.')
      else:
        coverage[:, i] = False
    for i in instruments_by_type[self.InstrumentType.DRUMS]:
      tensors = self._drums_converter.to_tensors(
          _extract_instrument(note_sequence, i))
      if tensors.outputs:
        encoded_instruments[i] = tensors.outputs[0]
      else:
        coverage[:, i] = False

    # Fill in coverage gaps up to self._gap_bars.
    og_coverage = coverage.copy()
    for j in range(total_bars):
      coverage[j] = np.any(
          og_coverage[
              max(0, j-self._gap_bars):min(total_bars, j+self._gap_bars) + 1],
          axis=0)

    # Take cross product of instruments from each class and compute combined
    # encodings where they overlap.
    seqs = []
    control_seqs = []
    for grp in itertools.product(
        instruments_by_type[self.InstrumentType.MEL],
        instruments_by_type[self.InstrumentType.BASS],
        instruments_by_type[self.InstrumentType.DRUMS]):
      # Consider an instrument covered within gap_bars from the end if any of
      # the other instruments are. This allows more leniency when re-encoding
      # slices.
      grp_coverage = np.all(coverage[:, grp], axis=1)
      grp_coverage[:self._gap_bars] = np.any(coverage[:self._gap_bars, grp])
      grp_coverage[-self._gap_bars:] = np.any(coverage[-self._gap_bars:, grp])
      for j in range(total_bars - self._slice_bars + 1):
        if (np.all(grp_coverage[j:j + self._slice_bars]) and
            all(i in encoded_instruments for i in grp)):
          start_step = j * self._steps_per_bar
          end_step = (j + self._slice_bars) * self._steps_per_bar
          seqs.append(np.concatenate(
              [encoded_instruments[i][start_step:end_step] for i in grp],
              axis=-1))
          if encoded_chords is not None:
            control_seqs.append(encoded_chords[start_step:end_step])

    return ConverterTensors(inputs=seqs, outputs=seqs, controls=control_seqs)

  def _to_notesequences(self, samples, controls=None):
    output_sequences = []
    dim_ranges = np.cumsum(self._split_output_depths)
    for i, s in enumerate(samples):
      mel_ns = self._melody_converter.to_notesequences(
          [s[:, :dim_ranges[0]]],
          [controls[i]] if controls is not None else None)[0]
      bass_ns = self._melody_converter.to_notesequences(
          [s[:, dim_ranges[0]:dim_ranges[1]]])[0]
      drums_ns = self._drums_converter.to_notesequences(
          [s[:, dim_ranges[1]:]])[0]

      for n in bass_ns.notes:
        n.instrument = 1
        n.program = ELECTRIC_BASS_PROGRAM
      for n in drums_ns.notes:
        n.instrument = 9

      ns = mel_ns
      ns.notes.extend(bass_ns.notes)
      ns.notes.extend(drums_ns.notes)
      ns.total_time = max(
          mel_ns.total_time, bass_ns.total_time, drums_ns.total_time)
      output_sequences.append(ns)
    return output_sequences


def count_examples(examples_path, data_converter,
                   file_reader=tf.python_io.tf_record_iterator):
  """Counts the number of examples produced by the converter from files."""
  filenames = tf.gfile.Glob(examples_path)

  num_examples = 0

  for f in filenames:
    tf.logging.info('Counting examples in %s.', f)
    reader = file_reader(f)
    for item_str in reader:
      item = data_converter.str_to_item_fn(item_str)
      tensors = data_converter.to_tensors(item)
      num_examples += len(tensors.inputs)
  tf.logging.info('Total examples: %d', num_examples)
  return num_examples


def get_dataset(
    config,
    num_threads=1,
    tf_file_reader=tf.data.TFRecordDataset,
    prefetch_size=4,
    is_training=False):
  """Get input tensors from dataset for training or evaluation.

  Args:
    config: A Config object containing dataset information.
    num_threads: The number of threads to use for pre-processing.
    tf_file_reader: The tf.data.Dataset class to use for reading files.
    prefetch_size: The number of batches to prefetch. Disabled when 0.
    is_training: Whether or not the dataset is used in training. Determines
      whether dataset is shuffled and repeated, etc.

  Returns:
    A tf.data.Dataset containing input, output, control, and length tensors.

  Raises:
    ValueError: If no files match examples path.
  """
  batch_size = config.hparams.batch_size
  examples_path = (
      config.train_examples_path if is_training else config.eval_examples_path)
  note_sequence_augmenter = (
      config.note_sequence_augmenter if is_training else None)
  data_converter = config.data_converter
  data_converter.is_training = is_training

  tf.logging.info('Reading examples from: %s', examples_path)

  num_files = len(tf.gfile.Glob(examples_path))
  if not num_files:
    raise ValueError(
        'No files were found matching examples path: %s' %  examples_path)
  files = tf.data.Dataset.list_files(examples_path)
  if is_training:
    files = files.apply(
        tf.contrib.data.shuffle_and_repeat(buffer_size=num_files))

  reader = files.apply(
      tf.contrib.data.parallel_interleave(
          tf_file_reader,
          cycle_length=num_threads,
          sloppy=True))

  def _remove_pad_fn(padded_seq_1, padded_seq_2, padded_seq_3, length):
    if length.shape.ndims == 0:
      return (padded_seq_1[0:length], padded_seq_2[0:length],
              padded_seq_3[0:length], length)
    else:
      # Don't remove padding for hierarchical examples.
      return padded_seq_1, padded_seq_2, padded_seq_3, length

  dataset = reader
  if note_sequence_augmenter is not None:
    dataset = dataset.map(note_sequence_augmenter.tf_augment)
  dataset = (dataset
             .map(data_converter.tf_to_tensors,
                  num_parallel_calls=num_threads)
             .flat_map(lambda *t: tf.data.Dataset.from_tensor_slices(t))
             .map(_remove_pad_fn))
  if is_training:
    dataset = dataset.shuffle(buffer_size=batch_size * 4)

  dataset = dataset.padded_batch(batch_size, dataset.output_shapes)

  if prefetch_size:
    dataset = dataset.prefetch(prefetch_size)

  return dataset


class GrooveConverter(BaseNoteSequenceConverter):
  """Converts to and from hit/velocity/offset representations.

  In this setting, we represent drum sequences and performances
  as triples of (hit, velocity, offset). Each timestep refers to a fixed beat
  on a grid, which is by default spaced at 16th notes.  Drum hits that don't
  fall exactly on beat are represented through the offset value, which refers
  to the relative distance from the nearest quantized step.

  Hits are binary [0, 1].
  Velocities are continuous values in [0, 1].
  Offsets are continuous values in [-0.5, 0.5], rescaled to [-1, 1] for tensors.

  Each timestep contains this representation for each of a fixed list of
  drum categories, which by default is the list of 9 categories defined in
  drums_encoder_decoder.py.  With the default categories, the input and output
  at a single timestep is of length 9x3 = 27. So a single measure of drums
  at a 16th note grid is a matrix of shape (16, 27).

  Args:
    split_bars: Optional size of window to slide over full converted tensor.
    steps_per_quarter: The number of quantization steps per quarter note.
    quarters_per_bar: The number of quarter notes per bar.
    pitch_classes: A collection of collections, with each sub-collection
      containing the set of pitches representing a single class to group by. By
      default, groups valid drum pitches into 9 different classes.
    humanize: If True, flatten all input velocities and microtiming. The model
      then learns to map from a flattened input to the original sequence.
    tapify: If True, squash all drums at each timestep to the open hi-hat
      channel.
    add_instruments: A list of strings matching drums in DRUM_LIST.
      These drums are removed from the inputs but not the outputs.
    num_velocity_bins: The number of bins to use for representing velocity as
      one-hots.  If not defined, the converter will use continuous values.
    num_offset_bins: The number of bins to use for representing timing offsets
      as one-hots.  If not defined, the converter will use continuous values.
    split_instruments: Whether to produce outputs for each drum at a given
      timestep across multiple steps of the model output. With 9 drums, this
      makes the sequence 9 times as long. A one-hot control sequence is also
      created to identify which instrument is to be output at each step.
    hop_size: Number of steps to slide window.
    hits_as_controls: If True, pass in hits with the conditioning controls
      to force model to learn velocities and offsets.
    fixed_velocities: If True, flatten all input velocities.
  """

  def __init__(self, split_bars=None, steps_per_quarter=4, quarters_per_bar=4,
               max_tensors_per_notesequence=8, pitch_classes=None,
               humanize=False, tapify=False, add_instruments=None,
               num_velocity_bins=None, num_offset_bins=None,
               split_instruments=False, hop_size=None,
               hits_as_controls=False, fixed_velocities=False):

    self._split_bars = split_bars
    self._steps_per_quarter = steps_per_quarter
    self._steps_per_bar = steps_per_quarter * quarters_per_bar

    self._humanize = humanize
    self._tapify = tapify
    self._add_instruments = add_instruments
    self._fixed_velocities = fixed_velocities

    self._num_velocity_bins = num_velocity_bins
    self._num_offset_bins = num_offset_bins
    self._categorical_outputs = num_velocity_bins and num_offset_bins

    self._split_instruments = split_instruments

    self._hop_size = hop_size
    self._hits_as_controls = hits_as_controls

    self._pitch_classes = pitch_classes or REDUCED_DRUM_PITCH_CLASSES
    self._pitch_class_map = {
        p: i for i, pitches in enumerate(self._pitch_classes) for p in pitches}
    self._num_drums = len(self._pitch_classes)

    if bool(num_velocity_bins) ^ bool(num_offset_bins):
      raise ValueError(
          'Cannot define only one of num_velocity_vins and num_offset_bins.')

    if split_bars is None and hop_size is not None:
      raise ValueError(
          'Cannot set hop_size without setting split_bars')

    drums_per_output = 1 if self._split_instruments else self._num_drums
    # Each drum hit is represented by 3 numbers - on/off, velocity, and offset
    if self._categorical_outputs:
      output_depth = (
          drums_per_output * (1 + num_velocity_bins + num_offset_bins))
    else:
      output_depth = drums_per_output * 3

    control_depth = 0
    # Set up controls for passing hits as side information.
    if self._hits_as_controls:
      if self._split_instruments:
        control_depth += 1
      else:
        control_depth += self._num_drums
    # Set up controls for cycling through instrument outputs.
    if self._split_instruments:
      control_depth += self._num_drums

    super(GrooveConverter, self).__init__(
        input_depth=output_depth,
        input_dtype=np.float32,
        output_depth=output_depth,
        output_dtype=np.float32,
        control_depth=control_depth,
        control_dtype=np.bool,
        end_token=False,
        presplit_on_time_changes=False,
        max_tensors_per_notesequence=max_tensors_per_notesequence)

  def _get_feature(self, note, feature, step_length=None):
    """Compute numeric value of hit/velocity/offset for a note.

    For now, only allow one note per instrument per quantization time step.
    This means at 16th note resolution we can't represent some drumrolls etc.
    We just take the note with the highest velocity if there are multiple notes.

    Args:
      note: A Note object from a NoteSequence.
      feature: A string, either 'hit', 'velocity', or 'offset'.
      step_length: Time duration in seconds of a quantized step. This only needs
        to be defined when the feature is 'offset'.

    Raises:
      ValueError: Any feature other than 'hit', 'velocity', or 'offset'.

    Returns:
      The numeric value of the feature for the note.
    """

    def _get_offset(note, step_length):
      true_onset = note.start_time
      quantized_onset = step_length * note.quantized_start_step
      diff = quantized_onset - true_onset
      return diff/step_length

    if feature == 'hit':
      if note:
        return 1.
      else:
        return 0.

    elif feature == 'velocity':
      if note:
        return note.velocity/127.  # Switch from [0, 127] to [0, 1] for tensors.
      else:
        return 0.  # Default velocity if there's no note is 0

    elif feature == 'offset':
      if note:
        offset = _get_offset(note, step_length)
        return offset*2  # Switch from [-0.5, 0.5] to [-1, 1] for tensors.
      else:
        return 0.  # Default offset if there's no note is 0

    else:
      raise ValueError('Unlisted feature: ' + feature)

  def _to_tensors(self, note_sequence):

    def _get_steps_hash(note_sequence):
      """Partitions all Notes in a NoteSequence by quantization step and drum.

      Creates a hash with each hash bucket containing a dictionary
      of all the notes at one time step in the sequence grouped by drum/class.
      If there are no hits at a given time step, the hash value will be {}.

      Args:
        note_sequence: The NoteSequence object

      Returns:
        The fully constructed hash

      Raises:
        ValueError: If the sequence is not quantized
      """
      if not mm.sequences_lib.is_quantized_sequence(note_sequence):
        raise ValueError('NoteSequence must be quantized')

      h = collections.defaultdict(lambda: collections.defaultdict(list))

      for note in note_sequence.notes:
        step = int(note.quantized_start_step)
        drum = self._pitch_class_map[note.pitch]
        h[step][drum].append(note)

      return h

    def _remove_drums_from_tensors(to_remove, tensors):
      """Drop hits in drum_list and set velocities and offsets to 0."""
      for t in tensors:
        t[:, to_remove] = 0.
      return tensors

    def _convert_vector_to_categorical(vectors, min_value, max_value, num_bins):
      # Avoid edge case errors by adding a small amount to max_value
      bins = np.linspace(min_value, max_value+0.0001, num_bins)
      return np.array([np.concatenate(
          np_onehot(np.digitize(v, bins, right=True), num_bins, dtype=np.int32))
                       for v in vectors])

    def _extract_windows(tensor, window_size, hop_size):
      """Slide a window across the first dimension of a 2D tensor."""
      return [tensor[i:i+window_size, :] for i in range(
          0, len(tensor) - window_size  + 1, hop_size)]

    try:
      quantized_sequence = sequences_lib.quantize_note_sequence(
          note_sequence, self._steps_per_quarter)
      if (mm.steps_per_bar_in_quantized_sequence(quantized_sequence) !=
          self._steps_per_bar):
        return ConverterTensors()
      if not quantized_sequence.time_signatures:
        quantized_sequence.time_signatures.add(numerator=4, denominator=4)
    except (mm.BadTimeSignatureError, mm.NonIntegerStepsPerBarError,
            mm.NegativeTimeError, mm.MultipleTimeSignatureError,
            mm.MultipleTempoError):
      return ConverterTensors()

    beat_length = 60. / quantized_sequence.tempos[0].qpm
    step_length = beat_length / (
        quantized_sequence.quantization_info.steps_per_quarter)

    steps_hash = _get_steps_hash(quantized_sequence)

    max_start_step = np.max(
        [note.quantized_start_step for note in quantized_sequence.notes])

    # Round up so we pad to the end of the bar.
    total_bars = int(np.ceil((max_start_step + 1) / self._steps_per_bar))
    max_step = self._steps_per_bar * total_bars

    # Each of these stores a (total_beats, num_drums) matrix.
    hit_vectors = np.zeros((max_step, self._num_drums))
    velocity_vectors = np.zeros((max_step, self._num_drums))
    offset_vectors = np.zeros((max_step, self._num_drums))

    # Loop through timesteps.
    for step in range(max_step):
      notes = steps_hash[step]

      # Loop through each drum instrument.
      for drum in range(self._num_drums):
        drum_notes = notes[drum]
        if len(drum_notes) > 1:
          note = max(drum_notes, key=lambda n: n.velocity)
        elif len(drum_notes) == 1:
          note = drum_notes[0]
        else:
          note = None

        hit_vectors[step, drum] = self._get_feature(note, 'hit')
        velocity_vectors[step, drum] = self._get_feature(note, 'velocity')
        offset_vectors[step, drum] = self._get_feature(
            note, 'offset', step_length)

    # These are the input tensors for the encoder.
    in_hits = copy.deepcopy(hit_vectors)
    in_velocities = copy.deepcopy(velocity_vectors)
    in_offsets = copy.deepcopy(offset_vectors)

    if self._tapify:
      argmaxes = np.argmax(in_velocities, axis=1)
      in_hits[:] = 0
      in_velocities[:] = 0
      in_offsets[:] = 0
      in_hits[:, 3] = hit_vectors[np.arange(max_step), argmaxes]
      in_velocities[:, 3] = velocity_vectors[np.arange(max_step), argmaxes]
      in_offsets[:, 3] = offset_vectors[np.arange(max_step), argmaxes]

    if self._humanize:
      in_velocities[:] = 0
      in_offsets[:] = 0

    if self._fixed_velocities:
      in_velocities[:] = 0

    # If learning to add drums, remove the specified drums from the inputs.
    if self._add_instruments:
      in_hits, in_velocities, in_offsets = _remove_drums_from_tensors(
          self._add_instruments, [in_hits, in_velocities, in_offsets])

    if self._categorical_outputs:
      # Convert continuous velocity and offset to one hots.
      velocity_vectors = _convert_vector_to_categorical(
          velocity_vectors, 0., 1., self._num_velocity_bins)
      in_velocities = _convert_vector_to_categorical(
          in_velocities, 0., 1., self._num_velocity_bins)

      offset_vectors = _convert_vector_to_categorical(
          offset_vectors, -1., 1., self._num_offset_bins)
      in_offsets = _convert_vector_to_categorical(
          in_offsets, -1., 1., self._num_offset_bins)

    if self._split_instruments:
      # Split the outputs for each drum into separate steps.
      total_length = max_step * self._num_drums
      hit_vectors = hit_vectors.reshape([total_length, -1])
      velocity_vectors = velocity_vectors.reshape([total_length, -1])
      offset_vectors = offset_vectors.reshape([total_length, -1])
      in_hits = in_hits.reshape([total_length, -1])
      in_velocities = in_velocities.reshape([total_length, -1])
      in_offsets = in_offsets.reshape([total_length, -1])
    else:
      total_length = max_step

    # Now concatenate all 3 vectors into 1, eg (16, 27).
    seqs = np.concatenate(
        [hit_vectors, velocity_vectors, offset_vectors], axis=1)

    input_seqs = np.concatenate(
        [in_hits, in_velocities, in_offsets], axis=1)

    # Controls section.
    controls = []
    if self._hits_as_controls:
      controls.append(hit_vectors.astype(np.bool))
    if self._split_instruments:
      # Cycle through instrument numbers.
      controls.append(np.tile(
          np_onehot(
              np.arange(self._num_drums), self._num_drums, np.bool),
          (max_step, 1)))
    controls = np.concatenate(controls, axis=-1) if controls else None

    if self._split_bars:
      window_size = self._steps_per_bar * self._split_bars
      hop_size = self._hop_size or window_size
      if self._split_instruments:
        window_size *= self._num_drums
        hop_size *= self._num_drums
      seqs = _extract_windows(seqs, window_size, hop_size)
      input_seqs = _extract_windows(input_seqs, window_size, hop_size)
      if controls is not None:
        controls = _extract_windows(controls, window_size, hop_size)
    else:
      # Output shape will look like (1, 64, output_depth).
      seqs = [seqs]
      input_seqs = [input_seqs]
      if controls is not None:
        controls = [controls]

    return ConverterTensors(inputs=input_seqs, outputs=seqs, controls=controls)

  def _to_notesequences(self, samples, controls=None):

    def _zero_one_to_velocity(val):
      output = int(np.round(val*127))
      return np.clip(output, 0, 127)

    def _minus_1_1_to_offset(val):
      output = val/2
      return np.clip(output, -0.5, 0.5)

    def _one_hot_to_velocity(v):
      return int((np.argmax(v) / len(v)) * 127)

    def _one_hot_to_offset(v):
      return (np.argmax(v) / len(v)) - 0.5

    output_sequences = []

    for sample in samples:
      n_timesteps = (sample.shape[0] // (
          self._num_drums if self._categorical_outputs else 1))

      note_sequence = music_pb2.NoteSequence()
      note_sequence.tempos.add(qpm=120)
      beat_length = 60. / note_sequence.tempos[0].qpm
      step_length = beat_length / self._steps_per_quarter

      # Each timestep should be a (1, output_depth) vector
      # representing n hits, n velocities, and n offsets in order.

      for i in range(n_timesteps):
        if self._categorical_outputs:
          # Split out the categories from the flat output.
          if self._split_instruments:
            hits, velocities, offsets = np.split(  # pylint: disable=unbalanced-tuple-unpacking
                sample[i*self._num_drums: (i+1)*self._num_drums],
                [1, self._num_velocity_bins+1],
                axis=1)
          else:
            hits, velocities, offsets = np.split(  # pylint: disable=unbalanced-tuple-unpacking
                sample[i],
                [self._num_drums, self._num_drums*(self._num_velocity_bins+1)]
                )
            # Split out the instruments.
            velocities = np.split(velocities, self._num_drums)
            offsets = np.split(offsets, self._num_drums)
        else:
          if self._split_instruments:
            hits, velocities, offsets = sample[
                i*self._num_drums: (i+1)*self._num_drums].T
          else:
            hits, velocities, offsets = np.split(  # pylint: disable=unbalanced-tuple-unpacking
                sample[i], 3)

        # Loop through the drum instruments: kick, snare, etc.
        for j in range(len(hits)):
          # Create a new note
          if hits[j] > 0.5:
            note = note_sequence.notes.add()
            note.instrument = 9  # All drums are instrument 9
            note.is_drum = True
            pitch = self._pitch_classes[j][0]
            note.pitch = pitch
            if self._categorical_outputs:
              note.velocity = _one_hot_to_velocity(velocities[j])
              offset = _one_hot_to_offset(offsets[j])
            else:
              note.velocity = _zero_one_to_velocity(velocities[j])
              offset = _minus_1_1_to_offset(offsets[j])
            note.start_time = (i - offset) * step_length
            note.end_time = note.start_time + step_length

      output_sequences.append(note_sequence)

    return output_sequences
