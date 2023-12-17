# Create overlapping sequences
x_batches = []
#self.y_batches = []


for i in range(0, 100000-20, 20//4):
    
    x = self.data[i:i + self.seq_length]
    #y = self.data[i+1:i+1 + self.seq_length]
    if len(x) == self.seq_length:
        self.x_batches.append(x)
    #self.y_batches.append(y)
