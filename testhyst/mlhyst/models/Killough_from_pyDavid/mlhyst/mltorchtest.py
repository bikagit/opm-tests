import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

# Define the neural network model
class NeuralNetworkModel(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetworkModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 8)
        self.layer2 = nn.Linear(4, 4)
        self.layer3 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        return x

def preprocess_data(Sh, data, wp):
    path = os.getcwd()
    SandSmax = []

    with open(path+'/sat.csv', 'w', newline='') as file:
        for S in Sh:
            smax = .0
            for s in S:
                smax = max(s, smax)
                file.write(str(s)+"\n")
                SandSmax.append([s, smax])

    pathCall = "/Users/macbookn/activopmwkspc/edgedev/build/opm-common/bin/hysteresis " + data +".DATA sat.csv relperms.csv " + wp + " 0"
    os.system(pathCall)

    krw = []
    krnw = []
    krM = []
    sT = []
    S2 = []

    with open(path+'/relperms.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            S2.append(float(row[0]))
            krnw.append(float(row[1]))
            krw.append(float(row[2]))
            krM.append(float(row[3]))
            sT.append([float(row[4])])

    return krnw, krw, SandSmax

def train_nn(krnw, Smax):
    x = np.array(Smax)
    y = np.array(krnw)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = NeuralNetworkModel(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Convert data to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Training loop
    num_epochs = 30
    losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test)
            val_losses.append(val_loss.item())

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    # Convert the model to TorchScript
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, 'neural_network_model.pt')

    print("Model has been converted to TorchScript and saved.")

    # Plot the losses
    plt.plot(range(num_epochs), losses, label='Training Loss')
    plt.plot(range(num_epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.savefig("loss.png")

    return model

swl = 0.05
S = np.linspace(0.0, 1.0-swl, 700)
SS = [np.linspace(smax, 0, 100) for smax in S]

krnw, krw, SandSmax = preprocess_data(SS, "../CO2_KILLOUGH", "GW")
modelnonwett = train_nn(krnw, SandSmax)

satspace = np.linspace(0.1, 0.95, 30)
satMaxspace = np.linspace(0.5, 0.8, 2)

for valsatmax in satMaxspace:
    for valsat in satspace:
        if valsatmax > valsat:
            xhat = np.array([[valsat, valsatmax]])
            xhat_tensor = torch.tensor(xhat, dtype=torch.float32)
            yhatkrnw = modelnonwett(xhat_tensor).detach().numpy()

# Save the model using export_model function
# export_model(modelnonwett, 'oldmodelkrnw.model')

plt.savefig("newKillough.png")
