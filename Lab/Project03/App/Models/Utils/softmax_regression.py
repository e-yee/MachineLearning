import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, epochs=100, batch_size=128):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.losses = []
    
    # Training functions
    def softmax(self, z):
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)
    
    def cross_entropy(self, pred, true):
        pred_clipped = np.clip(pred, 1e-7, 1 - 1e-7)
        return -np.mean(np.sum(true * np.log(pred_clipped), axis=1))
    
    def fit(self, X, y):
        N, D = X.shape
        C = y.shape[1]

        # Weights Initialization
        self.weights = np.random.randn(D, C).astype(dtype=np.float32) / D**0.5 # Kaiming init
        self.bias = np.zeros((1, C), dtype=np.float32)

        for epoch in range(self.epochs):
            indices = np.arange(N)
            np.random.shuffle(indices)
            X_itr = X[indices]
            y_itr = y[indices]
            
            epoch_loss = 0
            
            # Batch training
            for i in range(0, N, self.batch_size):
                X_batch = X_itr[i:i+self.batch_size]
                y_batch = y_itr[i:i+self.batch_size]
                batch_size_actual = len(X_batch)
                
                # Forward pass
                logits = X_batch @ self.weights + self.bias
                y_pred = self.softmax(logits)

                # Adjusted prediction
                y_pred_adjusted = y_pred - y_batch
                
                # Backward pass
                dw = (X_batch.T @ y_pred_adjusted) / batch_size_actual
                db = np.sum(y_pred_adjusted, axis=0) / batch_size_actual
                
                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
                # Compute loss
                batch_loss = self.cross_entropy(y_pred, y_batch)
                epoch_loss += batch_loss * batch_size_actual
            
            epoch_loss /= N
            self.losses.append(epoch_loss)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
    
    def predict(self, X):
        logits = X @ self.weights + self.bias
        y_pred = self.softmax(logits)
        return y_pred
    
    # Evaluating functions
    def accuracy(self, pred, true):
        pred_digit = np.argmax(pred, axis=1)
        true_digit = np.argmax(true, axis=1)
        return np.mean(pred_digit == true_digit)
    
    def precision(self, pred, true, average="none"):
        cm = self.confusion_matrix(pred, true)
        num_classes = true.shape[1]
        
        TPs = np.array([cm[i, i] for i in range(num_classes)])
        pred_counts = np.sum(cm, axis=0) # TPs + FPs 
        precisions = np.array(TPs / pred_counts, dtype=float)
        
        match average:
            case "none": return precisions
            case "macro": return np.mean(precisions)
            case "micro": return np.sum(TPs) / np.sum(pred_counts)
            case "weighted":
                true_counts = np.sum(cm, axis=1)
                weights = true_counts / np.sum(true_counts)
                return np.sum(precisions * weights)      
        return None
        
    def recall(self, pred, true, average="none"):
        cm = self.confusion_matrix(pred, true)
        num_classes = true.shape[1]
        
        TPs = np.array([cm[i, i] for i in range(num_classes)])
        true_counts = np.sum(cm, axis=1) # TPs + FNs
        recalls = np.array(TPs / true_counts, dtype=float)
        
        match average:
            case "none": return recalls
            case "macro": return np.mean(recalls)
            case "micro": return np.sum(TPs) / np.sum(true_counts)         
            case "weighted":
                weights = true_counts / np.sum(true_counts)
                return np.sum(recalls * weights)
        return None
    
    def F1_score(self, pred, true, average="none"):
        cm = self.confusion_matrix(pred, true)
        num_classes = true.shape[1]
        
        precisions = self.precision(pred, true, "none")
        recalls = self.recall(pred, true, "none")
        F1_scores = np.array(2 * (precisions * recalls / (precisions + recalls)), dtype=float)
        
        match average:
            case "none": return F1_scores
            case "macro": return np.mean(F1_scores)
            case "micro":
                TPs = np.array([cm[i, i] for i in range(num_classes)])
                true_counts = np.sum(cm, axis=1)
                pred_counts = np.sum(cm, axis=0)
                return 2 * np.sum(TPs) / np.sum(true_counts + pred_counts)
            case "weighted":
                true_counts = np.sum(cm, axis=1)
                weights = true_counts / np.sum(true_counts)
                return np.sum(F1_scores * weights)              
        return None
    
    def confusion_matrix(self, pred, true):
        pred_digit = np.argmax(pred, axis=1)
        true_digit = np.argmax(true, axis=1) 
        num_classes = len(np.unique(pred_digit))
        
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(pred_digit, true_digit):
            cm[t, p] += 1
        return cm
    
    def eval(self, X, y, average="none"):
        logits = X @ self.weights + self.bias
        y_pred = self.softmax(logits)
        
        print(f"Cross-entropy loss: {self.cross_entropy(y_pred, y):.4f}")
        self.plot_confusion_matrix(y_pred, y)
        self.plot_metrics(y_pred, y, average)
        print(f"Accuracy: {self.accuracy(y_pred, y):.4f}")
        
    # Plotting functions
    def plot_metrics(self, pred, true, average="none"):
        precisions = self.precision(pred, true, average)
        recalls = self.recall(pred, true, average)
        F1_scores = self.F1_score(pred, true, average)
        
        if average == "none":
            df = pd.DataFrame({
                "Class": np.arange(true.shape[1]),
                "Precision": precisions,
                "Recall": recalls,
                "F1-score": F1_scores
            })
            print(df.round(4).to_string(index=False))    
        elif average != "none"\
            and precisions is not None\
            and recalls is not None\
            and F1_scores is not None:
            
            print(f"Precision ({average}): {precisions:.4f}")
            print(f"Recall ({average}): {recalls:.4f}")
            print(f"F1-score ({average}): {F1_scores:.4f}")           
        
    def plot_confusion_matrix(self, pred, true):
        cm = self.confusion_matrix(pred, true)
        num_classes = pred.shape[1]
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True, linewidths=0.5)
        
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("True", fontsize=12)
        plt.xticks(ticks=np.arange(num_classes)+0.5, labels=range(num_classes))
        plt.yticks(ticks=np.arange(num_classes)+0.5, labels=range(num_classes), rotation=0)
        
        plt.title("Confusion Matrix (0-9)", fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_loss(self):
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, self.epochs + 1), self.losses, label="Loss")
        
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        
        plt.title("Epoch vs Loss")
        plt.legend()
        plt.show()