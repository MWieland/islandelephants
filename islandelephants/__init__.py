import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

version = "0.0.1"


import tensorflow as tf

print(tf.__version__)
print(tf.config.list_physical_devices())
exit()
