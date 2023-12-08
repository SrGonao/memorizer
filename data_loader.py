import numpy as np


class DataLoader:
    def __init__(self, data_path, batch_size, seq_length,overlap):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.overlap = overlap

        # Load data
        data = open(data_path, 'r').read()
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        # Create dictionaries
        self.char_to_ix = {ch: i for i, ch in enumerate(chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(chars)}

        # Convert data to indices
        self.data = [self.char_to_ix[ch] for ch in data]
        self.data = np.array(self.data)

        # Create overlapping sequences
        self.x_batches = []
        self.y_batches = []
        for i in range(0, self.data.size - self.seq_length, self.overlap):
            if i + self.batch_size * self.seq_length > self.data.size:
                break
            x = [self.data[i+j*self.seq_length:i+(j+1)*self.seq_length] for j in range(self.batch_size)]
            y = [self.data[i+j*self.seq_length+1:i+(j+1)*self.seq_length+1] for j in range(self.batch_size)]
            self.x_batches.append(x)
            self.y_batches.append(y)
        
        
        self.num_batches = len(self.x_batches)
        self.x_batches = np.array(self.x_batches)
        self.y_batches = np.array(self.y_batches)

        self.pointer = 0

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0


if __name__ == '__main__':
    data_loader = DataLoader('pi.dat', 50, 50,5)
    x, y = data_loader.next_batch()
    print(x.shape, y.shape)
    print(x[0, :10])
    print(y[0, :10])
    print(data_loader.num_batches)