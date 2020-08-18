import numpy as np
import tensorflow as tf


def Sampling(X, Y, n_way, n_shot, n_query, class_sampling=False):
    """
    Randomly select support and query examples.
    
    Inputs:
    -X: images; numpy array of shape (N,H,W,C) where N is size, H height, W width and C color channel
    -Y: labels; numpy array of shape (N,)
    -n_way: number of classes to sample
    -n_shot: number of support examples in each class for calculating the prototype (mean vector)
    -n_query: number of query examples in each class for testing
    -class_sampling: boolean, whether or not randomly sample n_way classes
    
    Outputs:
    -X_support: support examples; numpy array of shape (n_way*n_shot,H,W,C)
    -X_query: query examples; numpy array of shape(n_way*n_query,H,W,C)
    """
    X_support, X_query = [], []
    
    if class_sampling:
        classes = np.random.choice(np.unique(Y), n_way, replace=False)
    else:
        classes = np.unique(Y)
        
    for c in classes:
        D_c = X[Y == c]
        D_c = np.random.permutation(D_c)
        D_c = D_c[:n_shot+n_query]
        
        S_c = D_c[:n_shot]
        X_support.append(S_c)
        
        Q_c = D_c[n_shot:]
        X_query.append(Q_c)
        
    X_support, X_query = np.concatenate(X_support, axis=0), np.concatenate(X_query, axis=0)
    
    return X_support, X_query
    

def conv_bn_relu_pool(X, num_filters, filter_size=3, stride=1, pad='same', pool_size=2):
    """
    Build a block with Convolution-BatchNormalization-ReLu-MaxPooling layers in TensorFlow Keras. 
    
    Inputs:
    -X: images; numpy array of shape (N,H,W,C)
    -num_filters: number of filters in the convolution layer
    -filter_size: filter size in the convolution layer
    -stride: stride in the convolution layer
    -pad: 'same' or 'valid' padding in the convolution layer
    -pool_size: pooling size in the max pooling layer (note that stride number is the same as the pooling size)
    
    Output:
    -output: the block output of X
    """
    X = tf.keras.layers.Conv2D(num_filters, filter_size, stride, pad)(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    output = tf.keras.layers.MaxPool2D(pool_size)(X)
    
    return output


def encoder(input_shape=(84,84,3)):
    """
    Build a function that embeds from images to vectors, which is a TensorFlow Keras model 
    with 4 blocks such that each block defined above uses 64 filters.
    
    Input:
    -input_shape: input shape of an image (H,W,C)
    
    Output:
    -model
    """
    X_input = tf.keras.Input(shape=input_shape)
    X = conv_bn_relu_pool(X_input, 64)
    X = conv_bn_relu_pool(X, 64)
    X = conv_bn_relu_pool(X, 64)
    X = conv_bn_relu_pool(X, 64)
    output = tf.keras.layers.Flatten()(X)
    
    model = tf.keras.Model(X_input, output, name='Encoder')
    
    return model
    
    
def D2(X1, X2):
    """
    Compute the (vectorized) squared euclidean distance.
    
    Inputs:
    -X1: tf.Tensor matrix of shape (N1, num_columns)
    -X2: tf.Tensor matrix of shape (N2, num_columns)
    
    Output:
    -D: squared distance matrix from X1 to X2; tf.Tensor matrix of shape (N1, N2)
    """
    N1, N2 = X1.shape[0], X2.shape[0]
    D = np.zeros((N1,N2))
    
    X1_SumSquare = tf.reduce_sum(tf.math.square(X1), axis=1) #(N1,)
    X2_SumSquare = tf.reduce_sum(tf.math.square(X2), axis=1) #(N2,)
    mul = tf.linalg.matmul(X1, X2, transpose_b=True)
    D += X1_SumSquare[:,None] - 2*mul  + X2_SumSquare
    
    return D    
    
    
 def forward(encoder, X_support, X_query, n_way, n_shot, n_query, is_training):
    """
    Given the embedding function, compute the forward pass of an epsiode (or step, iteration) 
    in a prototypical network.
    
    Inputs:
    -encoder: TensorFlow Keras model defined above for embedding images to vectors
    -X_support: support examples obtained by the function Sampling()
    -X_query: query examples obtained by the function Sampling()
    -n_way: number of classes to select
    -n_shot: number of support examples in each class
    -n_query: number of query examples in each class
    -is_training: boolean, whether or not shuffle encoded X_query
    
    Outputs:
    -y_true: ground true labels, same order as X_query
    -scores: the negative distance matrix from encoded X_query to prototypes
    """
    fX_support, fX_query = encoder(X_support), encoder(X_query)
    fX_s = tf.reshape(fX_support, (n_way, n_shot, fX_support.shape[1]))
    
    prototype = tf.reduce_mean(fX_s, axis=1)
    
    y_true = tf.constant(np.arange(n_way).repeat(n_query))
    
    if is_training:
        mask = np.random.permutation(n_way*n_query) # fX_query.shape[0]
        fX_query = tf.gather(fX_query, mask)
        y_true = tf.gather(y_true, mask)
    
    scores = -1 * D2(fX_query, prototype) # (n_way*n_query, n_way)
    
    return y_true, scores  
    
    
class protoNet(tf.keras.Model):
    """
    Build a prototypical network using TensorFlow Keras model.
    
    Inputs:
    -input_shape: images input shape (H,W,C) in the model encoder()
    -X_support, X_query, n_way, n_shot, n_query, is_training: see description in forward()
    
    Outputs:
    -y_true, score: see description in forward()
    """
    def __init__(self, input_shape, name='ProtoNet'):
        super(protoNet, self).__init__(name=name)
        self.encoder = encoder(input_shape)
          
    def call(self, X_support, X_query, n_way, n_shot, n_query, is_training=True):       
        y_true, scores = forward(self.encoder, X_support, X_query, n_way, n_shot, n_query, is_training)
        return y_true, scores    
    
    
