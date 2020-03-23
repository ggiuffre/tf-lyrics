import os, warnings
from time import time
import tensorflow as tf
from tflyrics.constants import default_vocab



class Poet:
    """An artificial poet.

    A Poet object is an object that wraps a recurrent predictive TensorFlow
    model, and with it can predict the next character in a sequence of
    unicode characters extracted from a possibly large text corpus.
    """

    def __init__(self, name: str = None, vocabulary: list = None,
        embedding_dim: int = 256, rnn_units: int = 1024):
        """Create a Poet.

        Create a Poet, optionally specifying its name, embedding
        dimensionality, and number of recurrent hidden units.

        :param name: string that uniquely identifies the poet
        :param vocabulary: list of unicode characters accepted by the model
        :param embedding_dim: output dimensionality of the hidden layer
        :param rnn_units: number of units in the hidden layer
        """

        # assign an identifier to the Poet object:
        self.name = name or str(time()).replace('.', '')

        # create the poet's vocabulary and its inverse map:
        self.vocabulary = vocabulary or default_vocab
        self.char2idx = {ch: idx for idx, ch in enumerate(self.vocabulary)}

        # remember the model's architecture:
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.build_model()

    @property
    def batch_size(self) -> int:
        """Get the batch size of the poet's internal model.

        Get the model's batch size, i.e. the number of inputs that can be fed
        at once to the model for traning or evaluation.

        :return: the current batch size
        """

        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_size: int) -> None:
        """Set the batch size of the poet's internal model.

        Set the model's batch size, i.e. the number of inputs that can be fed
        at once to the model for traning or evaluation.

        :param size: the new batch size
        """

        self.build_model(batch_size=new_size)
        self._batch_size = new_size

        # restore the model parameters:
        if hasattr(self, 'weights'):
            self.model.set_weights([w.numpy() for w in self.weights])

    def build_model(self, batch_size: int = 1) -> None:
        """Build the Poet's internal model.

        :param batch_size: the number of inputs to be fed at once to the model
        """

        vocab_size = len(self.vocabulary)

        if vocab_size < 1:
            return

        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, self.embedding_dim,
                batch_input_shape=[batch_size, None]),
            tf.keras.layers.LSTM(self.rnn_units,
                return_sequences=True,
                stateful=True,
                recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(self.rnn_units,
                return_sequences=True,
                stateful=True,
                recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(vocab_size)
            ])

    def train_on(self, train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset = None, n_epochs: int = 1,
        checkpoints: bool = False) -> dict:
        """Train the Poet's internal model on a dataset.

        Train the Poet's internal model on a TensorFlow Dataset containing
        batches of sequences of text (encoded as integers). Optionally specify
        the number of epochs, and whether a checkpoint of the model should be
        saved at the end of each training epoch. Returns the training history,
        as a dictionary whose values are lists; each list represents a metric,
        and each element in the list is the value of that metric at a certain
        epoch.

        :param train_dataset: TensorFlow Dataset of batches of sequences
        :param n_epochs: number of training epochs
        :param checkpoints: whether or not to save checkpoints of the model
        :return: the training history, as a dictionary of lists
        """

        # change the internal model to have adequate training batch size:
        self.batch_size = train_dataset.element_spec[0].shape[0]

        # declare a list of callbacks to be run afer each epoch:
        callbacks = []

        # optionally ensure that checkpoints be saved during training:
        if checkpoints:
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
        history_obj = self.model.fit(train_dataset,
            epochs=n_epochs,
            validation_data=val_dataset,
            callbacks=callbacks)

        # make a copy of the weights just learned by the model:
        self.weights = self.model.weights

        return history_obj.history

    def restore(self) -> None:
        """Restore the state of the Poet's model from a checkpoint.

        If checkpoints have been saved during training, set the Poet's model
        parameters (weights) to the state of the latest checkpoint saved.

        :raises: ResourceWarning
        """

        if hasattr(self, 'checkpoint_dir'):
            latest_state = tf.train.latest_checkpoint(self.checkpoint_dir)
            self.model.load_weights(latest_state)
        else:
            warnings.warn('No checkpoints have been saved for this poet.',
                ResourceWarning)

    def generate(self, start_string: str, n_gen_chars: int = 1000,
        temperature: float = 1.0) -> str:
        """Generate text using the poet's internal model.

        Generate a specified number of characters by feeding an initial string
        to the model and using this string as a starting point to repeatedly
        sample a character that is likely to appear after the previously
        generated character.

        :param start_string: initial string fed to the model
        :param n_gen_chars: number of characters to generate
        :param temperature: how surprising vs. predictable characters should be
        :return: a new, generated text
        """

        # change the internal model to have unit batch_size:
        self.batch_size = 1

        # convert the starting string to a list of integers:
        model_input = [self.char2idx[s] for s in start_string]
        model_input = tf.expand_dims(model_input, 0)

        # predict characters:
        model_output = []
        self.model.reset_states()
        for c in range(n_gen_chars):

            # get predictions over characters from the model:
            predictions = self.model(model_input)

            # remove the batch dimension:
            predictions = tf.squeeze(predictions, 0)

            # sample the next character:
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, 1)[-1, 0].numpy()
            model_output.append(self.vocabulary[predicted_id])

            # pass the predicted character as the next input to the model:
            model_input = tf.expand_dims([predicted_id], 0)

        return start_string + ''.join(model_output)
