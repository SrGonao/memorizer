import numpy as np
import torch as t

class DataLoader:
    def __init__(self, data_path, seq_length,digits):
        self.seq_length = seq_length
        
        self.digits = digits

        # Load data
        data = open(data_path, 'r').read()
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        # Create dictionaries
        self.char_to_ix = {"0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,".":10}
        self.ix_to_char = {0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",10:"."}

        # Convert data to indices
        self.data = [self.char_to_ix[ch] for ch in data]
        self.data = np.array(self.data)

        # Create overlapping sequences
        possible_data = self.data[:self.digits]

        self.x_batches = []
        #self.y_batches = []


        for i in range(0, self.digits-self.seq_length, 5):
            
            x = self.data[i:i + self.seq_length]
            #y = self.data[i+1:i+1 + self.seq_length]
            if len(x) == self.seq_length:
                self.x_batches.append(x)
            #self.y_batches.append(y)

        
        
        self.num_batches = len(self.x_batches)
        self.x_batches = np.array(self.x_batches)
        self.shuffled = self.x_batches

        self.pointer = 0

    def shuffle(self,seed):
        np.random.seed(seed)
        self.shuffled = np.random.permutation(self.x_batches)

    def __iter__(self,seed=0):
    
        self.pointer = 0
        return self
    
    def __next__(self):
        if self.pointer >= self.num_batches:
            raise StopIteration
        x = self.shuffled[self.pointer]
        #y = self.y_batches[self.pointer]
        self.pointer += 1
        x = t.tensor(x, dtype=t.long).unsqueeze(0).cuda()
        #y = t.tensor(y, dtype=t.long).unsqueeze(0)
        
        return x


if __name__ == '__main__':
    data_loader = DataLoader('pi.dat', 50,100)
  