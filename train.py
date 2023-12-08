import torch as t
import torch.nn as nn
import model
import data_loader
import numpy as np

data_load = data_loader.DataLoader('pi.dat', 500, 50,5)
number_of_classes = len(data_load.char_to_ix)

model = model.Memorizer(number_of_classes)

# Define loss function and optimizer
criterion = t.nn.CrossEntropyLoss()
optimizer = t.optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(100):
    data_load.reset_batch_pointer()
    for i in range(data_load.num_batches):
        # Get batch of data
        x, y = data_load.next_batch()
        x = t.tensor(x, dtype=t.long)
        y = t.tensor(y, dtype=t.long)

        # Forward pass
        outputs = model(x,y)
        outputs=outputs.transpose(0,1)
        outputs=outputs.transpose(1,2)
        loss = criterion(outputs, y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                  .format(epoch + 1, 100, i + 1, data_load.num_batches, loss.item(), np.exp(loss.item())))

