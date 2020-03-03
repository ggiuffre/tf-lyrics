import os, warnings
from time import time
from collections import Counter
import tensorflow as tf



class Poet:
    """An artificial poet.

    A Poet object is an object that wraps a recurrent predictive TensorFlow
    model, and can learn how to predict the next character in a sequence of
    characters extracted from a possibly large text corpus.
    """

    def __init__(self, name: str = None, embedding_dim: int = 256, rnn_units: int = 1024):
        """Create a Poet.

        Create a Poet, optionally specifying its name, embedding
        dimensionality, and number of recurrent hidden units.

        :param name: string that uniquely identifies the poet
        :param embedding_dim: output dimensionality of the hidden layer
        :param rnn_units: number of units in the hidden layer
        """

        self.name = name or str(time()).replace('.', '')
        self.vocabulary = []
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.model = None
        self.weights = None
        self.checkpoint_dir = None

    def build_model(self, batch_size: int = 1) -> None:
        """Build the Poet's internal model of the world.

        :param batch_size: the number of inputs to be fed at once to the model
        """

        vocab_size = len(self.vocabulary)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, self.embedding_dim,
                batch_input_shape=[batch_size, None]),
            tf.keras.layers.LSTM(self.rnn_units, # GRU / LSTM
                return_sequences=True,
                stateful=True,
                recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(self.rnn_units, # GRU / LSTM
                return_sequences=True,
                stateful=True,
                recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(vocab_size)
            ])

    def preprocess(self, text: str, batch_size: int = 32) -> object:
        """Preprocess a text corpus for supervised learning.

        Given a UTF-8 text string, map each possible sequence of 100 contiguous
        characters to the sequence made of the successors of each of these
        characters (where the "successor" of a character is the character that
        comes after it in the text). Optionally provide a batch size that will
        be used when training the model to predict the next character in the
        text corpus.

        This method returns a tf.data.Dataset that relates each character to
        its successor, in "windows" of 100 characters at a time. If the text
        corpus is empty, it returns None.

        :param text: a text corpus made of UTF-8 characters
        :param batch_size: the number of sequences fed at once to the model
        :return: a tf.data.Dataset that relates each character to its successor
        """

        # if the text corpus is empty, return None:
        if len(text) == 0:
            return None

        # sequences of characters are just batches of characters:
        text_as_int = [self.char2idx[c] for c in text]
        char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
        seq_length = 100 # window size of each character sequence
        sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

        # associate training features to labels:
        split_input_label = lambda chunk: (chunk[:-1], chunk[1:])
        dataset = sequences.map(split_input_label)

        # shuffle the sequences (batches) of characters:
        dataset = dataset.shuffle(len(text))

        # create bacthes of sequences ("batches of batches"):
        dataset = dataset.batch(batch_size, drop_remainder=True)

        return dataset

    def train_on(self, text: str, n_epochs: int = 1, validation_split: float = 0.0, checkpoints: bool = False) -> None:
        """Train the poet's internal model on a text corpus.

        Train the poet's internal model (with gradient-based optimitazion)
        on a UTF-8 text corpus, for a specified number of epochs. The model
        can be trained with or wihout splitting the dataset into training and
        validation, and with or without regularly saving checkpoints of the
        model's parameters.

        :param text: a text corpus made of UTF-8 characters
        :param n_epochs: the number of epochs to train the model
        :param validation_split: the fraction of text to use as validation data
        :param checkpoints: whether to save the weighs on disk after each epoch
        """

        # declare the model's vocabulary, and sort it by character occurence:
        hist = Counter(text)
        self.vocabulary = sorted(hist, key=hist.get, reverse=True)

        # create a mapping from unique characters to indices:
        self.char2idx = {u: i for i, u in enumerate(self.vocabulary)}

        # split the text corpus into training and validation corpora:
        training_batch_size = 64
        split_index = int(validation_split * len(text))
        val_text, train_text = text[:split_index], text[split_index:]

        # preprocess the two corpora:
        train_dataset = self.preprocess(train_text, training_batch_size)
        val_dataset = self.preprocess(val_text, training_batch_size)

        # build a TensorFlow model with adequate training batch size:
        self.build_model(batch_size=training_batch_size)

        callbacks = []
        if checkpoints:
            # ensure that checkpoints are saved during training:
            self.checkpoint_dir = './checkpoints' + '_' + self.name
            ckpt_prefix = os.path.join(self.checkpoint_dir, 'ckpt_{epoch}')
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=ckpt_prefix,
                save_weights_only=True)
            callbacks.append(checkpoint_callback)

        # set optimization hyper-parameters:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(
            optimizer='adam',
            loss=loss)

        # train the model:
        history = self.model.fit(train_dataset,
            epochs=n_epochs,
            validation_data=val_dataset,
            callbacks=callbacks)

        # save the weights just learned by the model:
        self.weights = self.model.weights

    def restore(self) -> None:
        """Restore the state of the Poet's model from the latest checkpoint.

        If checkpoints have been saved during training, set the Poet's model
        parameters (weights) to the state of the latest checkpoint taken.

        :raises: ResourceWarning
        """

        if self.checkpoint_dir is None:
            warnings.warn('No checkpoints have been saved for this poet.',
                warnings.ResourceWarning)
        else:
            latest_state = tf.train.latest_checkpoint(self.checkpoint_dir)
            self.model.load_weights(latest_state)

    def generate(self, start_string: str, n_gen_chars: int = 1000, temperature: float = 1.0) -> str:
        """Generate text using the poet's internal model.

        Generate a specified number of characters by feeding an initial string
        to the model and using this string as a starting point to sample a
        character that is likely to appear after the previously generated
        character.

        :param start_string: the initial string fed to the model
        :param n_gen_chars: the number of characters to generate
        :param temperature: how surprising vs. predictable characters should be
        :return: a new text
        """

        # build model with unit batch_size:
        self.build_model(batch_size=1)

        # restore the model parameters:
        self.model.set_weights([w.numpy() for w in self.weights])

        # encode the starting string to numbers:
        model_input = [self.char2idx[s] for s in start_string]
        model_input = tf.expand_dims(model_input, 0)

        model_output = []
        self.model.reset_states()
        for c in range(n_gen_chars):

            predictions = self.model(model_input)

            # remove the batch dimension:
            predictions = tf.squeeze(predictions, 0)

            # sample the next character:
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, 1)[-1, 0].numpy()
            model_output.append(self.vocabulary[predicted_id])

            # pass the predicted character as the next input to the model,
            # along with the previous hidden state:
            model_input = tf.expand_dims([predicted_id], 0)

        return start_string + ''.join(model_output)
