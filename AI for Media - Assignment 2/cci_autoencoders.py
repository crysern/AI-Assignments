##adpated from https://keras.io/examples/generative/vae/

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt

#editted from https://machinelearningmastery.com/how-to-interpolate-and-perform-vector-arithmetic-with-faces-using-a-generative-adversarial-network/
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from numpy import linspace
from keras.models import load_model

 # generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=10):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    return z_input

# uniform interpolation between two points in latent space
def interpolate_points(p1, p2, n_steps=10):
    # interpolate ratios between the points
    ratios = linspace(0, 1, num=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
    return asarray(vectors)

# create a plot of generated images
def plot_generated(examples):
    # plot images
    num_to_show = len(examples)
    plt.figure(figsize=(12,12))
    cols = 3
    rows = int(np.ceil(num_to_show / cols))
    for i in range(num_to_show):
        # define subplot
        plt.subplot(rows, cols, i+1)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(examples[i])
    plt.show()

def plot_label_clusters(encoder, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean = encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()    
    
def plot_label_clusters_vae(encoder, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()

def plot_latent_space(decoder, n=30, figsize=15, scale = (-5,5)):
    # display a n*n 2D manifold of digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(scale[0], scale[1], n)
    grid_y = np.linspace(scale[0], scale[1], n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data, reconstruction)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }    
    
def load_from_file(path = ""):
    encoder = keras.models.load_model(path+'encoder.tf')
    decoder = keras.models.load_model(path+'decoder.tf')
    vae = VAE(encoder, decoder)
    return vae
    
def init_VAE(input_dims = (28,28,1), latent_dim = 32):    
    
    encoder_inputs = keras.Input(shape=input_dims)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    
    quarter = int(input_dims[0]/4)
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(quarter * quarter * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((quarter, quarter, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(input_dims[2], 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")



    vae = VAE(encoder, decoder)
    return vae

# def init_VAE(input_dims = (28,28,1)):    
    
#     latent_dim = 2

#     encoder_inputs = keras.Input(shape=input_dims)
#     x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
#     x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
#     x = layers.Flatten()(x)
#     x = layers.Dense(16, activation="relu")(x)
#     z_mean = layers.Dense(latent_dim, name="z_mean")(x)
#     z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
#     z = Sampling()([z_mean, z_log_var])
#     encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

#     latent_inputs = keras.Input(shape=(latent_dim,))
#     x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
#     x = layers.Reshape((7, 7, 64))(x)
#     x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
#     x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
#     decoder_outputs = layers.Conv2DTranspose(input_dims[2], 3, activation="sigmoid", padding="same")(x)
#     decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

#     class VAE(keras.Model):
#         def __init__(self, encoder, decoder, **kwargs):
#             super(VAE, self).__init__(**kwargs)
#             self.encoder = encoder
#             self.decoder = decoder
#             self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
#             self.reconstruction_loss_tracker = keras.metrics.Mean(
#                 name="reconstruction_loss"
#             )
#             self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

#         @property
#         def metrics(self):
#             return [
#                 self.total_loss_tracker,
#                 self.reconstruction_loss_tracker,
#                 self.kl_loss_tracker,
#             ]

#         def train_step(self, data):
#             with tf.GradientTape() as tape:
#                 z_mean, z_log_var, z = self.encoder(data)
#                 reconstruction = self.decoder(z)
#                 print(data.shape, reconstruction.shape)
#                 reconstruction_loss = tf.reduce_mean(
#                     tf.reduce_sum(
#                         keras.losses.mean_squared_error(data, reconstruction)
#                     )
#                 )
#                 kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
#                 kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
#                 total_loss = reconstruction_loss + kl_loss
#             grads = tape.gradient(total_loss, self.trainable_weights)
#             self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
#             self.total_loss_tracker.update_state(total_loss)
#             self.reconstruction_loss_tracker.update_state(reconstruction_loss)
#             self.kl_loss_tracker.update_state(kl_loss)
#             return {
#                 "loss": self.total_loss_tracker.result(),
#                 "reconstruction_loss": self.reconstruction_loss_tracker.result(),
#                 "kl_loss": self.kl_loss_tracker.result(),
#             }

#     vae = VAE(encoder, decoder)
#     return vae
