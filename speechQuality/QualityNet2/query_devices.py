import tensorflow as tf
# 查看gpu和cpu的数量
gpus = tf.test.is_gpu_available()
print(gpus,tf.test.gpu_device_name())
