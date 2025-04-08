import tensorflow as tf
import numpy as np
import time
import os
import argparse

# Parse arguments
# Custom softmax function to match the original implementation
def custom_softmax(inputs):
    max_vals = tf.reduce_max(inputs, axis=1, keepdims=True)
    exp_values = tf.exp(inputs - max_vals)
    sum_exp = tf.reduce_sum(exp_values, axis=1, keepdims=True)
    return exp_values / sum_exp

# Custom model class
class MLP(tf.keras.Model):
    def __init__(self, leaking_coeff=0.0):
        super(MLP, self).__init__()
        self.leaking_coeff = leaking_coeff
        
        # Initialize weights with normal distribution (0, 0.1)
        self.W1 = tf.Variable(tf.random.normal([785, 100], mean=0, stddev=0.1), name='W1')
        self.W2 = tf.Variable(tf.random.normal([101, 10], mean=0, stddev=0.1), name='W2')
    
    def call(self, x, training=False):
        # Add ones for bias
        batch_size = tf.shape(x)[0]
        ones = tf.ones([batch_size, 1])
        
        # First layer
        x_with_bias = tf.concat([ones, x], axis=1)
        s1 = tf.matmul(x_with_bias, self.W1)
        
        # Apply leaky ReLU
        mask = tf.cast(s1 > 0, tf.float32) + (self.leaking_coeff * tf.cast(s1 < 0, tf.float32))
        a1 = s1 * mask
        
        # Second layer
        a1_with_bias = tf.concat([ones, a1], axis=1)
        s2 = tf.matmul(a1_with_bias, self.W2)
        
        return s2, a1, mask, x_with_bias, a1_with_bias

def main(args):
    print("TensorFlow implementation of MNIST MLP")
    
    # is_training = bool(args.is_training)
    # leaking_coeff = args.leaking_coeff
    batchsize = args.minibatch_size
    lr = args.learning_rate
    num_epoch = args.num_epoch
    # lambda_ = args.lambda_
    
    # Load MNIST data
    file = np.load('./mnist.npz', 'r')
    x_train = file['train_data']
    y_train = file['train_labels']
    x_test = file['test_data']
    y_test = file['test_labels']
    file.close()
    
    split = args.split
    val_split = args.val_split
    x_train = x_train[:split]
    y_train = y_train[:split]
    x_val = x_test[:val_split]
    y_val = y_test[:val_split]
    
    print(f'#={split}, batch={batchsize}, lr={lr}')
    
    # Create model
    model = MLP()
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    
    # Load weights if available
    if os.path.isfile("./weightin.npz"):
        print("using ./weightin.npz")
        randfile = np.load("./weightin.npz", "r")
        W1 = randfile["W1"]
        W2 = randfile["W2"]
        randfile.close()
        
        # Assign weights to model
        model.W1.assign(tf.convert_to_tensor(W1, dtype=tf.float32))
        model.W2.assign(tf.convert_to_tensor(W2, dtype=tf.float32))
    
    # Performance tracking
    train_accuracy = np.zeros(num_epoch)
    val_accuracy = np.zeros(num_epoch)
    
    start_time = time.process_time()
    
    # Training loop
    for epoch in range(num_epoch):
        print(f'At Epoch {1 + epoch}:')
        
        # Training batches
        for mbatch in range(int(split / batchsize)):
            start = mbatch * batchsize
            x_batch = x_train[start:(start + batchsize)]
            y_batch = y_train[start:(start + batchsize)]
            
            x_batch_tensor = tf.convert_to_tensor(x_batch, dtype=tf.float32)
            y_batch_tensor = tf.convert_to_tensor(y_batch, dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                # Forward pass
                s2, a1, mask, x_with_bias, a1_with_bias = model(x_batch_tensor, training=True)
                
                # Apply softmax
                a2 = custom_softmax(s2)
                
                # Calculate loss (cross-entropy)
                loss = -tf.reduce_sum(y_batch_tensor * tf.math.log(a2 + 1e-10)) / batchsize
                
                # Add L2 regularization
                # if lambda_ > 0:
                #     l2_loss = lambda_ * (tf.reduce_sum(tf.square(model.W1)) + tf.reduce_sum(tf.square(model.W2)))
                #     loss += l2_loss
            
            # Calculate gradients manually to match the original implementation
            grad_s2 = (a2 - y_batch_tensor) / batchsize
            grad_a1 = tf.matmul(grad_s2, tf.transpose(model.W2[1:]))
            delta_W2 = tf.matmul(tf.transpose(a1_with_bias), grad_s2)
            grad_s1 = mask * grad_a1
            delta_W1 = tf.matmul(tf.transpose(x_with_bias), grad_s1)
            
            gradients = [delta_W1] 
            optimizer.apply_gradients(zip(gradients, [model.W1, model.W2]))
        
        # Evaluate on training set
        correct_count = 0
        for mbatch in range(int(split / batchsize)):
            start = mbatch * batchsize
            x_batch = x_train[start:(start + batchsize)]
            y_batch = y_train[start:(start + batchsize)]
            
            x_batch_tensor = tf.convert_to_tensor(x_batch, dtype=tf.float32)
            
            s2, _, _, _, _ = model(x_batch_tensor)
            predicted = tf.argmax(s2, axis=1).numpy()
            true_labels = np.argmax(y_batch, axis=1)
            correct_count += np.sum(predicted == true_labels)
        
        accuracy = correct_count / split
        print(f"train-set accuracy at epoch {1 + epoch}: {accuracy}")
        train_accuracy[epoch] = 100 * accuracy
        
        # Evaluate on validation set
        correct_count = 0
        for mbatch in range(int(val_split / batchsize)):
            start = mbatch * batchsize
            x_batch = x_val[start:(start + batchsize)]
            y_batch = y_val[start:(start + batchsize)]
            
            x_batch_tensor = tf.convert_to_tensor(x_batch, dtype=tf.float32)
            
            s2, _, _, _, _ = model(x_batch_tensor)
            predicted = tf.argmax(s2, axis=1).numpy()
            true_labels = np.argmax(y_batch, axis=1)
            correct_count += np.sum(predicted == true_labels)
        
        accuracy = correct_count / val_split
        print(f"Val-set accuracy at epoch {1 + epoch}: {accuracy}")
        val_accuracy[epoch] = 100 * accuracy
        
        print(f"elapsed time={time.process_time()-start_time}")
        
        # Save model weights in the same format as original code
        W1_np = model.W1.numpy()
        W2_np = model.W2.numpy()
        np.savez_compressed("./weightout_tensorflow.npz", W1=W1_np, W2=W2_np)

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
