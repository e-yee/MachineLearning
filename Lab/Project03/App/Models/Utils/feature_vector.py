import numpy as np
import cv2

class FeatureVector:
    def normalized_pixels(self, X):
        X_flatten = (X / 255.0).reshape(X.shape[0], -1).astype(dtype=np.float32)
        return X_flatten
    
    def sobel_edge(self, X):
        # Pixel Normalized
        X = X.astype(np.float32) / 255.0
        
        X = np.array([cv2.GaussianBlur(img, (3, 3), 0) for img in X], dtype=np.float32)
        
        edges = np.array([
            np.sqrt(cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)**2 +
                    cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)**2)
            for img in X
        ], dtype=np.float32)

        # Normalize
        edges = np.array([e / (e.max() + 1e-8) for e in edges], dtype=np.float32)
        
        # Stack with origin images
        X_new = np.stack([X, edges], axis=-1).reshape(X.shape[0], -1)
        return X_new
    
    def block_average(self, X, block_size=4):
        # Pixel Normalized
        X = X.astype(np.float32) / 255.0
        N, H, W = X.shape
        H_b, W_b = H // block_size, W // block_size

        # Reshape and calculate mean
        X_reshaped = X.reshape(N, H_b, block_size, W_b, block_size)
        blocks = X_reshaped.mean(axis=(2, 4))  # (N, H_b, W_b)
        
        # Normalize
        blocks = np.array([b / (b.max() + 1e-8) for b in blocks], dtype=np.float32)
        return blocks.reshape(N, -1)