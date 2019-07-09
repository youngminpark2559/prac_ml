WARNING:tensorflow:From 02_ConvolutionalNeuralNetwork-OCR.py:47: read_data_sets (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
WARNING:tensorflow:From /home/young/anaconda3/envs/py36gputorch100/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:260: maybe_download (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
Instructions for updating:
Please write your own downloading logic.
WARNING:tensorflow:From /home/young/anaconda3/envs/py36gputorch100/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:262: extract_images (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use tf.data to implement this functionality.
WARNING:tensorflow:From /home/young/anaconda3/envs/py36gputorch100/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:267: extract_labels (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use tf.data to implement this functionality.
WARNING:tensorflow:From /home/young/anaconda3/envs/py36gputorch100/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:110: dense_to_one_hot (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use tf.one_hot on tensors.
WARNING:tensorflow:From /home/young/anaconda3/envs/py36gputorch100/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:290: DataSet.__init__ (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
xkbcommon: ERROR: failed to add default include path 
Qt: Failed to create XKB context!
Use QT_XKB_CONFIG_ROOT environmental variable to provide an additional search path, add ':' as separator to provide several search paths and/or make sure that XKB configuration data directory contains recent enough contents, to update please see http://cgit.freedesktop.org/xkeyboard-config/ .
WARNING:tensorflow:From /home/young/anaconda3/envs/py36gputorch100/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
WARNING:tensorflow:From 02_ConvolutionalNeuralNetwork-OCR.py:291: calling argmax (from tensorflow.python.ops.math_ops) with dimension is deprecated and will be removed in a future version.
Instructions for updating:
Use the `axis` argument instead
2019-07-09 18:36:20.286316: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-07-09 18:36:20.535475: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:998] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-07-09 18:36:20.538205: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x56108f879840 executing computations on platform CUDA. Devices:
2019-07-09 18:36:20.538246: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): GeForce GTX 1060 6GB, Compute Capability 6.1
2019-07-09 18:36:20.561239: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3399680000 Hz
2019-07-09 18:36:20.562195: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x56108f8e3500 executing computations on platform Host. Devices:
2019-07-09 18:36:20.562215: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
2019-07-09 18:36:20.562723: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties: 
name: GeForce GTX 1060 6GB major: 6 minor: 1 memoryClockRate(GHz): 1.7335
pciBusID: 0000:1f:00.0
totalMemory: 5.93GiB freeMemory: 5.14GiB
2019-07-09 18:36:20.562749: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-07-09 18:36:20.567064: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-07-09 18:36:20.567104: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-07-09 18:36:20.567119: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-07-09 18:36:20.568596: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 4961 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1060 6GB, pci bus id: 0000:1f:00.0, compute capability: 6.1)
2019-07-09 18:36:21.111657: I tensorflow/stream_executor/dso_loader.cc:152] successfully opened CUDA library libcublas.so.10.0 locally
Extracting data/MNIST/train-images-idx3-ubyte.gz
Extracting data/MNIST/train-labels-idx1-ubyte.gz
Extracting data/MNIST/t10k-images-idx3-ubyte.gz
Extracting data/MNIST/t10k-labels-idx1-ubyte.gz
Accuracy on Test-Set: 9.0% (904 / 10000)
Optimization Iteration:      1, Training Accuracy:   6.2%
Time usage: 0:00:00
Accuracy on Test-Set: 8.5% (853 / 10000)
Time usage: 0:00:00
Accuracy on Test-Set: 69.5% (6951 / 10000)
Example errors:
Optimization Iteration:    101, Training Accuracy:  67.2%
Optimization Iteration:    201, Training Accuracy:  79.7%
Optimization Iteration:    301, Training Accuracy:  85.9%
Optimization Iteration:    401, Training Accuracy:  87.5%
Optimization Iteration:    501, Training Accuracy:  87.5%
Optimization Iteration:    601, Training Accuracy:  95.3%
Optimization Iteration:    701, Training Accuracy:  93.8%
Optimization Iteration:    801, Training Accuracy:  92.2%
Optimization Iteration:    901, Training Accuracy:  96.9%
Time usage: 0:00:03
Accuracy on Test-Set: 93.5% (9345 / 10000)
Example errors:
