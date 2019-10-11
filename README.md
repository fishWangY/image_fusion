# image_fusion
This repository contains the implementation of 《基于深度堆叠卷积神经网络的图像融合》

# Prerequisites and Installation
* tensorflow > 1.4
* scipy==1.2.1
* numpy==1.16.4
* opencv-python==4.1.0.25

# Getting Started
Clone this repository:
--
```python
git clone https://github.com/fishyinwang/image_fusion.git
cd nets
vim fusion.py
double click initializers jump to tensorflow/contrib/layers/python/layers/initializers.py
```
Add function on initializers.py
--
```python
def gaussian(uniform=False, dtype=dtypes.float32):
    # gaussian and laplacian filter
    if not dtype.is_floating:
        raise TypeError('Cannot create initializer for non-floating point type.')

    def _initializer(shape):
        """Initializer function."""
        if not uniform:
            # gaussian filter
            kernel = tf.constant([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
                                  [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
                                  [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],
                                  [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
                                  [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]])
            gaussian_kernel = tf.get_variable("gaussian_kernel", dtype=tf.float32, initializer=kernel)
            gaussian_kernels = tf.concat([shape[-1] * [gaussian_kernel]], axis=-1)
            gaussian_kernel = tf.reshape(gaussian_kernels, shape)

            return gaussian_kernel

        else:
            # laplacian filter
            kernel = tf.constant([[0.0, 0.0, 0.059761, 0.0, 0.0],
                                  [0.0, 0.059761, 0.119522, 0.059761, 0.0],
                                  [0.059761, 0.119522, -0.956182, 0.119522, 0.059761],
                                  [0.0, 0.059761, 0.119522, 0.059761, 0.0],
                                  [0.0, 0.0, 0.059761, 0.0, 0.0]])
            gaussian_kernel = tf.get_variable("laplacian_kernel", dtype=tf.float32, initializer=kernel)
            gaussian_kernels = tf.concat([shape[-1] * [gaussian_kernel]], axis=-1)
            gaussian_kernel = tf.reshape(gaussian_kernels, shape)

            return gaussian_kernel

    return _initializer
```
# Running the code
* prepare train/val dataset
```python
cd datasets
python convert2tfRecord.py
```
* train and eval
```python
python train.py
cd scripts
sh test.sh
```
# Citation
If you use our code, please consider citing our repo.
