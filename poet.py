import os
from time import time
import numpy as np
import tensorflow as tf



class Poet:
    """An artificial poet."""

    def __init__(self, name: str = None, embedding_dim: int = 256, rnn_units: int = 1024):
        """Create a Poet.

        Create a Poet, optionally specifying its name, embedding
        dimensionality, and number of recurrent hidden units."""

        self.name = name or str(time()).replace('.', '')
        self.vocab = []
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.model = None
        self.weights = None
        self.checkpoint_dir = './training_checkpoints' + '_' + self.name

    def build_model(self, batch_size: int = 1) -> None:
        """Build the Poet's internal model of the world."""

        vocab_size = len(self.vocab)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, self.embedding_dim,
                batch_input_shape=[batch_size, None]),
            tf.keras.layers.LSTM(self.rnn_units, # GRU / LSTM
                return_sequences=True,
                stateful=True,
                recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(vocab_size)
            ])

    def preprocess(self, text: str, batch_size: int) -> tf.data.Dataset:
        """Preprocess a text corpus for supervised learning."""

        # create mappings from unique characters to indices, and viceversa:
        self.vocab = sorted(set(text))
        self.char2idx = {u: i for i, u in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)

        # create training examples & targets:
        text_as_int = np.array([self.char2idx[c] for c in text])
        char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

        # sequences of characters are just batches:
        max_seq_length = 100 # max length of a sentence for a single input
        sequences = char_dataset.batch(max_seq_length + 1, drop_remainder=True)

        # from each sequence, form the input and target text:
        split_input_target = lambda chunk: (chunk[:-1], chunk[1:])
        dataset = sequences.map(split_input_target)

        # shuffle the sequences of characters, and create bacthes of them:
        dataset = dataset.shuffle(len(text))
        dataset = dataset.batch(batch_size, drop_remainder=True)

        return dataset

    def train_on(self, text: str, n_epochs: int = 1, checkpoints: bool = False) -> None:
        """Train the poet's internal model on a text corpus."""

        # preprocess the data and set a couple of other things:
        training_batch_size = 64
        dataset = self.preprocess(text, training_batch_size)

        # build model with adequate training batch_size:
        self.build_model(batch_size=training_batch_size)

        callbacks = []
        if checkpoints:
            # ensure that checkpoints are saved during training:
            ckpt_prefix = os.path.join(self.checkpoint_dir, 'ckpt_{epoch}')
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=ckpt_prefix,
                save_weights_only=True)
            callbacks.append(checkpoint_callback)

        # set various optimization hyper-parameters:
        loss = lambda t, y: tf.keras.losses.sparse_categorical_crossentropy(t, y, from_logits=True)
        self.model.compile(
            optimizer='adam',
            loss=loss)

        # train the model:
        history = self.model.fit(dataset,
            epochs=n_epochs,
            callbacks=callbacks)

        self.weights = self.model.weights

    def restore(self):
        """Restore the state of the Poet's model from the latest checkpoint."""

        latest_state = tf.train.latest_checkpoint(self.checkpoint_dir)
        self.model.load_weights(latest_state)

    def generate(self, start_string: str, n_gen_chars: int = 1000, temperature: float = 1.0) -> str:
        """Generate text using the poet's internal model."""

        # build model with unit batch_size:
        self.build_model(batch_size=1)

        # restore the model parameters:
        self.model.set_weights([w.numpy() for w in self.weights])

        # encode the starting string:
        input_eval = [self.char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        generated_text = []
        self.model.reset_states()
        for c in range(n_gen_chars):

            predictions = self.model(input_eval)

            # remove the batch dimension:
            predictions = tf.squeeze(predictions, 0)

            # use a categorical distribution to sample the next character:
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

            # pass the predicted character as the next input to the model,
            # along with the previous hidden state:
            input_eval = tf.expand_dims([predicted_id], 0)
            generated_text.append(self.idx2char[predicted_id])

        return start_string + ''.join(generated_text)
