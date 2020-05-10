import nni

def run_trial(params):
    mnist = input_data.read_data_sets(params['data_dir'], one_hot=True)

    mnist_network = MnistNetwork(channel_1_num=params['channel_1_num'], channel_2_num=params['channel_2_num'], conv_size=params['conv_size'], hidden_size=params['hidden_size'], pool_size=params['pool_size'], learning_rate=params['learning_rate'])
    mnist_network.build_network()

    with tf.Session() as sess:
        mnist_network.train(sess, mnist)
        test_acc = mnist_network.evaluate(mnist)

        nni.report_final_result(test_acc)

if __name__ == '__main__':

   params = {'data_dir': '/tmp/tensorflow/mnist/input_data', 'dropout_rate': 0.5, 'channel_1_num': 32, 'channel_2_num': 64,
   'conv_size': 5, 'pool_size': 2, 'hidden_size': 1024, 'learning_rate': 1e-4, 'batch_num': 2000, 'batch_size': 32}
   params = nni.get_next_parameter()
   run_trial(params)
