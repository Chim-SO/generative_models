import numpy as np
from keras.datasets import mnist
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from src.models.vanilla_gan.model import generator, discriminator, gan

if __name__ == '__main__':
    # Define hyperparameters
    img_shape = (28, 28, 1)
    latent_dim = 100
    optimizer = Adam(lr=0.0002, beta_1=0.5)

    # Build the generator and discriminator models
    generator = generator(latent_dim)
    discriminator = discriminator(img_shape)

    # Compile the discriminator model
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Build the GAN model
    gan = gan(generator, discriminator, latent_dim)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)

    # Load MNIST dataset
    (X_train, _), (_, _) = mnist.load_data()

    # Normalize input images to [-1, 1]
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)

    # Define training parameters
    epochs = 10000
    batch_size = 128

    # Train the GAN model
    for epoch in range(epochs):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        # Select a random half of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]
        # Generate a batch of fake images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_imgs = generator.predict(noise)
        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------
        # Generate a batch of noise vectors
        noise = np.random.normal(0, 1, (batch_size,latent_dim))
        # Train the generator (to have the discriminator label samples as real)
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        # Print the progress
        if epoch % 100 == 0:
            print("Epoch %d: Discriminator Loss: %f, Generator Loss: %f" % (epoch, d_loss[0], g_loss))

        # Save generated images every 1000 epochs
        if epoch % 1000 == 0:
            # Generate images from noise, using the generator model
            noise = np.random.normal(0, 1, (25, latent_dim))
            gen_imgs = generator.predict(noise)
            # Rescale images to 0-1
            gen_imgs = 0.5 * gen_imgs + 0.5
            # Save generated images to file
            for i in range(gen_imgs.shape[0]):
                plt.subplot(5, 5, i + 1)
                plt.imshow(gen_imgs[i, :, :, 0], cmap='gray')
                plt.axis('off')
            plt.savefig("gan_mnist_%d.png" % epoch)
            plt.close()
