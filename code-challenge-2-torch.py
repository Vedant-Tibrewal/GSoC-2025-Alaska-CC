import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import argparse
import os


# Define the MLP model
class MLP(nn.Module):
    def __init__(self, leaking_coeff=0.0):
        super(MLP, self).__init__()
        # Input layer: 28*28 = 784 -> 100 hidden nodes
        self.fc1 = nn.Linear(784, 100)
        # Hidden layer: 100 -> 10 output nodes
        self.fc2 = nn.Linear(100, 10)
        self.leaking_coeff = leaking_coeff
        
        # Initialize weights with normal distribution (0, 0.1)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.1)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        
    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the input
        s1 = self.fc1(x)
        # Apply leaky ReLU
        mask = (s1 > 0).float() + (self.leaking_coeff * (s1 < 0).float())
        a1 = s1 * mask
        s2 = self.fc2(a1)
        # No softmax here as it's included in CrossEntropyLoss
        return s2

# Load MNIST dataset
def load_mnist():
    file = np.load('./mnist.npz', 'r')
    x_train = file['train_data']
    y_train = file['train_labels']
    x_test = file['test_data']
    y_test = file['test_labels']
    file.close()
    return x_train, y_train, x_test, y_test

def main(args):
    print("PyTorch implementation of MNIST MLP")
    
    # is_training = bool(args.is_training)
    # leaking_coeff = args.leaking_coeff
    batchsize = args.minibatch_size
    lr = args.learning_rate
    num_epoch = args.num_epoch
    
    # if is_training:
        # Load MNIST data
    x_train, y_train, x_test, y_test = load_mnist()
    
    split = args.split
    val_split = args.val_split
    x_train = x_train[:split]
    y_train = y_train[:split]
    x_val = x_test[:val_split]
    y_val = y_test[:val_split]
    
    print(f'#={split}, batch={batchsize}, lr={lr}')
    
    # Convert to PyTorch tensors
    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.LongTensor(np.argmax(y_train, axis=1))
    x_val_tensor = torch.FloatTensor(x_val)
    y_val_tensor = torch.LongTensor(np.argmax(y_val, axis=1))
    
    # Create model
    model = MLP()
    
    # Load weights if available
    if os.path.isfile("./weightin.npz"):
        print("using ./weightin.npz")
        randfile = np.load("./weightin.npz", "r")
        W1 = randfile["W1"]
        W2 = randfile["W2"]
        b1 = W1[0, :]
        b2 = W2[0, :]
        W1 = W1[1:, :].T
        W2 = W2[1:, :].T

        randfile.close()
        
        # Copy weights to model (excluding bias row)
        with torch.no_grad():
            model.fc1.weight.data = torch.FloatTensor(W1)
            model.fc1.bias.data = torch.FloatTensor(b1)
            model.fc2.weight.data = torch.FloatTensor(W2)
            model.fc2.bias.data = torch.FloatTensor(b2)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # Performance tracking
    train_accuracy = np.zeros(num_epoch)
    val_accuracy = np.zeros(num_epoch)
    
    start_time = time.process_time()
    
    # Training loop
    for epoch in range(num_epoch):
        print(f'At Epoch {1 + epoch}:')
        model.train()
        
        # Training batches
        for mbatch in range(int(split / batchsize)):
            start = mbatch * batchsize
            x_batch = x_train_tensor[start:(start + batchsize)]
            y_batch = y_train_tensor[start:(start + batchsize)]
            
            # Forward pass
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Evaluate on training set
        model.eval()
        correct_count = 0
        with torch.no_grad():
            for mbatch in range(int(split / batchsize)):
                start = mbatch * batchsize
                x_batch = x_train_tensor[start:(start + batchsize)]
                y_batch = y_train_tensor[start:(start + batchsize)]
                
                outputs = model(x_batch)
                _, predicted = torch.max(outputs, 1)
                correct_count += (predicted == y_batch).sum().item()
        
        accuracy = correct_count / split
        print(f"train-set accuracy at epoch {1 + epoch}: {accuracy}")
        train_accuracy[epoch] = 100 * accuracy
        
        # Evaluate on validation set
        correct_count = 0
        with torch.no_grad():
            for mbatch in range(int(val_split / batchsize)):
                start = mbatch * batchsize
                x_batch = x_val_tensor[start:(start + batchsize)]
                y_batch = y_val_tensor[start:(start + batchsize)]
                
                outputs = model(x_batch)
                _, predicted = torch.max(outputs, 1)
                correct_count += (predicted == y_batch).sum().item()
        
        accuracy = correct_count / val_split
        print(f"Val-set accuracy at epoch {1 + epoch}: {accuracy}")
        val_accuracy[epoch] = 100 * accuracy
    
    print(f"elapsed time={time.process_time()-start_time}")
    
    # Save model weights in the same format as original code
    with torch.no_grad():
        W1_out = np.zeros((785, 100))
        W1_out[1:, :] = model.fc1.weight.cpu().numpy().T
        W1_out[0, :] = model.fc1.bias.cpu().numpy()
        
        W2_out = np.zeros((101, 10))
        W2_out[1:, :] = model.fc2.weight.cpu().numpy().T
        W2_out[0, :] = model.fc2.bias.cpu().numpy()
        
        np.savez_compressed("./weightout_pytorch.npz", W1=W1_out, W2=W2_out)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='PyTorch MNIST MLP')
    parser.add_argument('--minibatch_size', type=int, default=1, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--num_epoch', type=int, default=7, help='number of epochs')
    parser.add_argument('--split', type=int, default=50, help='training split size')
    parser.add_argument('--val_split', type=int, default=50, help='validation split size')
    args = parser.parse_args()
    print(args)

    main(args)
