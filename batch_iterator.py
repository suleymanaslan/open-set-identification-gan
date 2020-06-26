import numpy as np


class BatchIterator:
    def __init__(self, inputs, labels, batch_size):
        self.inputs = inputs
        self.labels = labels
        self.batch_size = batch_size
        self.size = self.inputs.shape[0]
        self.epochs = 0
        self.cursor = 0
        self.shuffle()
    
    def shuffle(self):
        self.indices = np.random.permutation(self.size)
        self.cursor = 0
    
    def next_batch(self):
        batch_inputs, batch_labels = self.shuffled_batch()
        self.cursor += self.batch_size
        if self.cursor + self.batch_size - 1 >= self.size:
            self.epochs += 1
            self.shuffle()
        return batch_inputs, batch_labels
    
    def shuffled_batch(self):
        batch_indices = self.indices[self.cursor:self.cursor + self.batch_size]
        return self.inputs[batch_indices], self.labels[batch_indices]
