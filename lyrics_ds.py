import random
import tensorflow as tf
from genius import Genius



class LyricsGenerator:
    """A dataset of song lyrics.

    A LyricsGenerator object represents a dataset where each sample is a
    sequence of unicode characters taken from song lyrics. More precisely,
    each example in a LyricsGenerator is a character sequence that maps to a
    sequence with same length but shifted content: for example, there could be
    a LyricsGenerator where "Hello wo" maps to "ello wor", and "ello wor" maps
    to "llo worl".
    """

    default_vocab = ['\n', ' ', '!', '"', '$', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '}', '~', '\xa0', '¡', 'Å', 'É', 'á', 'ã', 'ä', 'ç', 'è', 'é', 'ë', 'í', 'ï', 'ñ', 'ó', 'ô', 'ö', 'ú', 'ü', 'ē', 'Ι', 'ا', 'ح', 'د', 'ل', 'م', 'ه', '\u2005', '\u200e', '…', '\u2060']

    def __init__(self, artists: list = [], per_artist: int = 5, vocabulary: list = None, token: str = None):
        """Create a LyricsGenerator object.

        Create a LyricsGenerator object that will provide lyrics from a
        specified set of artists, and will filter those lyrics to only include
        characters from a vocabulary of unicode characters.

        :param artists: artists whose songs should be included
        :param per_artist: number of songs to include per artist
        :param vocabulary: the unicode characters accepted by the object
        :param token: a token to access the Genius API
        """

        # declare what characters the dataset can accept:
        self.vocabulary = vocabulary or LyricsGenerator.default_vocab
        keys_tensor = tf.constant(self.vocabulary)
        vals_tensor = tf.range(tf.size(keys_tensor))
        self.char2idx = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)

        # create a Genius object to fetch lyrics:
        self.genius = Genius(token)

        # mark the songs that should be downloaded:
        self.songs = []
        for a in artists:
            for s in self.genius.popular_songs(a, per_artist):
                print('LyricsGenerator:',
                    'adding {} by {} to wishlist'.format(s, a))
                self.songs.append(s)

        # shuffle the list of songs to be downloaded:
        random.shuffle(self.songs)

    def as_dataset(self, batch_size: int = 1) -> tf.data.Dataset:
        """Get a TensorFlow dataset equivalent to this object.

        Get a TensorFlow dataset whose samples are yielded by a generator
        that queries a Genius object, and where each sample is the lyrics of
        a song processed for (predictive) supervised learning.

        :param batch_size: the batch size of the dataset
        :return: a TensorFlow Dataset object
        """

        def get_song_lyrics(s):
            l = tf.py_function(self.genius.get_song_lyrics, [s], tf.string)
            l = tf.reshape(l, ())
            return tf.data.Dataset.from_tensors(l)

        songs_dataset = tf.data.Dataset.from_tensor_slices(self.songs)

        lyrics_dataset = songs_dataset.interleave(
            get_song_lyrics,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
            ).filter(
            lambda x: tf.size(tf.strings.bytes_split(x)) > 0
            )

        processed_dataset = lyrics_dataset.map(
            self.preprocess
            ).unbatch()

        dataset = processed_dataset.map(
            lambda chunk: (chunk[:-1], chunk[1:])
            ).batch(
            batch_size,
            drop_remainder=True).cache()

        return dataset

    @tf.function
    def preprocess(self, text: str, seq_length: int = 100) -> list:
        """Preprocess a string of text containing lyrics.

        Preprocess a string containing lyrics, extracting substrings that
        have a fixed, specified size.

        :param text: a string containing lyrics
        :param seq_length: fixed size of each output sequence
        :return: a list of substrings extracted from the lyrics
        """

        # convert the string to a list of integers:
        text_as_chars = tf.strings.bytes_split(text)
        text_as_int = tf.map_fn(
            fn=lambda c: self.char2idx.lookup(c),
            elems=text_as_chars,
            dtype=tf.int32)
        text_as_int = tf.boolean_mask(text_as_int, text_as_int > -1)

        # compute the number of characters in the text:
        text_size = tf.size(text_as_int)

        # increase the sequence length by 1, for character-level prediction:
        seq_length += 1

        # sequences of characters are just batches of characters:
        trail  = tf.truncatemod(text_size, seq_length)
        n_seqs = tf.truncatediv(text_size, seq_length)
        to_keep = text_size - trail
        sequences = tf.reshape(text_as_int[:to_keep], [n_seqs, seq_length])

        # shuffle the sequences (batches) of characters:
        sequences = tf.random.shuffle(sequences)

        return sequences
