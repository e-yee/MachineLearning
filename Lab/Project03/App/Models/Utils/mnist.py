import struct
import numpy as np
        
class MNIST:
    def __init__(self): pass
        
    def load_mnist_images(self, file_path):
        with open(file_path, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            if magic != 2051:
                print(f"Magic number {magic} không hợp lệ trong file ảnh MNIST.")
                return None
            
            buf = f.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols)
            return data
        
    def load_mnist_labels(self, file_path):
        with open(file_path, 'rb') as f:
            magic, num_labels = struct.unpack(">II", f.read(8))
            if magic != 2049:
                print(f"Magic number {magic} không hợp lệ trong file nhãn MNIST.")
                return None
            
            buf = f.read(num_labels)
            labels = np.frombuffer(buf, dtype=np.uint8)
            return labels
        
    def load_data(self):
        train_images_path = "../Data/Input/Train/train-images.idx3-ubyte"
        train_labels_path = "../Data/Input/Train/train-labels.idx1-ubyte"
        test_images_path = "../Data/Input/Test/t10k-images.idx3-ubyte"
        test_labels_path = "../Data/Input/Test/t10k-labels.idx1-ubyte"
        
        X_train = self.load_mnist_images(train_images_path)
        y_train = self.load_mnist_labels(train_labels_path)
        X_test = self.load_mnist_images(test_images_path)
        y_test = self.load_mnist_labels(test_labels_path)
        return (X_train, y_train), (X_test, y_test)