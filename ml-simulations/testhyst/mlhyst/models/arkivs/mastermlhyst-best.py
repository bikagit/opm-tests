import numpy as np
import os, sys
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import set_random_seed

set_random_seed(1234)

sys.path.insert(0, '/Users/macbookn/activopmwkspc/edgedev/opm-common/python/opm/ml')
from ml_tools import export_model, MinMaxScalerLayer, MinMaxUnScalerLayer

# Function to create the generator model with dense layers
def create_generator(input_dim, output_dim):
    generator = Sequential()
    generator.add(Dense(128, input_dim=input_dim, activation='relu'))
    generator.add(Dense(256, activation='relu'))
    generator.add(Dense(output_dim, activation='linear'))
    return generator

# Function to create the discriminator model with dense layers
def create_discriminator(input_dim):
    discriminator = Sequential()
    discriminator.add(Dense(256, input_dim=input_dim, activation='relu'))
    discriminator.add(Dense(128, activation='relu'))
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy', metrics=['accuracy'])
    return discriminator

# Function to create the GAN model
def create_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(generator.input_shape[1],))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    gan.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy')
    return gan

# Preprocess data function (unchanged)
def PreprocessData(Sh, data, wp):
    SandSmax = []
    with open(path+'/sat.csv', 'w', newline='') as file:
        for S in Sh:
            smax = .0
            for s in S:
                smax = max(s,smax)
                file.write(str(s)+"\n")
                SandSmax.append([s, smax])
    pathCall = "/Users/macbookn/activopmwkspc/edgedev/build/opm-common/bin/hysteresis " + data +".DATA sat.csv relperms.csv " + wp + " 0"
    os.system(pathCall)
    krw, krnw, krM, sT, S2 = [], [], [], [], []
    with open(path+'/relperms.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            S2.append(float(row[0]))
            krnw.append(float(row[1]))
            krw.append(float(row[2]))
            krM.append(float(row[3]))
            sT.append([float(row[4])])
    return krnw, krw, SandSmax

# Training function for GAN
def train_gan(generator, discriminator, gan, data, epochs=100, batch_size=64):
    for epoch in range(epochs):
        # Train discriminator
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = data[idx]
        noise = np.random.normal(0, 1, (batch_size, generator.input_shape[1]))
        fake_data = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train generator
        noise = np.random.normal(0, 1, (batch_size, generator.input_shape[1]))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
        
        # Print the progress
        if epoch % 1000 == 0:
            print(f"{epoch} [D loss: {d_loss[0]}] [G loss: {g_loss}]")

# Main execution
swl = 0.05
path = os.getcwd()
S = np.linspace(0.0, 1.0-swl, 700)
SS = [np.linspace(smax, 0, 100) for smax in S]
krnw, krw, SandSmax = PreprocessData(SS, "/Users/macbookn/activopmwkspc/pyDavid/pyopmnearwell/examples/hysteresis_models/Killough/CO2_KILLOUGH", "GW")

# Prepare data for GAN
data = np.array(SandSmax)

# Create models
# generator = create_generator(input_dim=data.shape[1], output_dim=1)
# discriminator = create_discriminator(input_dim=1)
# gan = create_gan(generator, discriminator)

# Train GAN
# train_gan(generator, discriminator, gan, data)

# Save the generator model
export_model(generator, 'generator_model.model')

plt.savefig("newKillough.png")
