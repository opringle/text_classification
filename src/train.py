import mxnet as mx

batch_size = 30
#data iterator to read from csv file
train_iter = mx.io.CSVIter(data_csv='../data/x_train.csv',
                           data_shape=(2,),
                           label_csv = "../data/y_train.csv",
                           label_shape = (1,),
                           batch_size=batch_size,
                           label_name='lin_reg_label')
val_iter = mx.io.CSVIter(data_csv='../data/x_test.csv',
                           data_shape=(2,),
                           label_csv="../data/y_test.csv",
                           label_shape=(1,),
                           batch_size=batch_size,
                           label_name='lin_reg_label')

# placeholder for input data and labels
data = mx.sym.Variable('data')
Y = mx.symbol.Variable('lin_reg_label')

# The first fully-connected layer and the corresponding activation function
fc1 = mx.sym.FullyConnected(data=data, num_hidden=128)
act1 = mx.sym.Activation(data=fc1, act_type="relu")

# The second fully-connected layer and the corresponding activation function
fc2 = mx.sym.FullyConnected(data=act1, num_hidden=64)
act2 = mx.sym.Activation(data=fc2, act_type="relu")

# Since this is regression ouput layer has 1 neuron
fc3 = mx.sym.FullyConnected(data=act2, num_hidden=1)
# compute MSE of output
mlp = mx.sym.LinearRegressionOutput(data=fc3, label = Y, name='mlp')

import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

# create a trainable module on CPU
#mlp_model = mx.mod.Module(symbol=mlp, context=mx.cpu())

mlp_model = mx.mod.Module(
    context=mx.cpu(),
    symbol=mlp,
    data_names=['data'],
    label_names=['lin_reg_label']  # network structure
)

#save network architecure image 
mx.viz.plot_network(symbol=mlp)

#fit model to training data
mlp_model.fit(train_iter, val_iter,
          optimizer_params={'learning_rate': 0.005, 'momentum': 0.9},
          num_epoch=50,
          eval_metric='mse',
          batch_end_callback=mx.callback.Speedometer(batch_size, 2))

#now model is trained make prediction
prob = mlp_model.predict(val_iter)

# get MSE on reserved test set
rmse = mx.metric.RMSE()
mlp_model.score(val_iter, rmse)
print(rmse)
