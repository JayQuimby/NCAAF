import tensorflow as tf
from tensorflow.python.platform import build_info as tf_build_info
import ctypes

def check_cudnn_version():
    print("TensorFlow version:", tf.__version__)
    
    # Check CuDNN version using ctypes to access CuDNN library
    try:
        cudnn_lib = ctypes.CDLL("libcudnn.so")
        cudnn_version = ctypes.c_int()
        cudnn_lib.cudnnGetVersion(ctypes.byref(cudnn_version))
        print("CuDNN version (from CuDNN library):", cudnn_version.value)
    except Exception as e:
        print("Error accessing CuDNN version:", e)

def check_cuda_config():
    print("\nCUDA config:")
    print("Is CUDA available:", tf.test.is_built_with_cuda())
    
    # Print CUDA version using alternative method
    try:
        for device in tf.config.list_physical_devices('GPU'):
            device_name = device.name
            print(f"GPU Device Name: {device_name}")
            cuda_version = tf.strings.split(tf.strings.split(device_name, ':')[1], '.')[0]
            print(f"CUDA version: {cuda_version}")
    except Exception as e:
        print("Error accessing CUDA version:", e)

def check_cudnn_config():
    print("\nCuDNN config:")
    # Check if CuDNN is available
    try:
        cuda_device = tf.config.list_physical_devices('GPU')[0]
        cuda_device_attributes = tf.python.client.device_lib.DeviceAttributes(device=cuda_device)
        cudnn_info = tf.python.client.device_lib.CudnnVersionAttributes(cuda_device_attributes)
        print("Is CuDNN available:", cudnn_info.cudnn)
        
        # Print CuDNN version if available
        if cudnn_info.cudnn:
            print("CuDNN version:", cudnn_info.cudnn_version)
    except Exception as e:
        print("Error checking CuDNN config:", e)

def check_warnings():
    print("\nTensorFlow Warnings:")
    # Print TensorFlow warnings
    for warning in tf.get_logger().warning_messages:
        print(warning)

if __name__ == "__main__":
    check_cudnn_version()
    check_cuda_config()
    check_cudnn_config()
    #check_warnings()

