import torch as t
import torch.nn as nn
from model import Memorizer
import numpy as np
import wandb
from tqdm import tqdm
import data_loader
class MemorizerTrainingArgs:
    seq_length: int = 20
    digits : int = 100000
    epochs : int = 1000
    max_steps_per_epoch = 1000
    data_loader = data_loader.DataLoader('pi.dat', seq_length,digits)
    lr = 0.00001
    wandb_project = "memorizer"
    wandb_name = None
    seed = 0
def get_log_probs(
    logits, 
    tokens
):

    log_probs = logits.log_softmax(dim=-1)
    # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    log_probs_for_tokens = log_probs[:, 10:].gather(dim=-1, index=tokens[:,11:].unsqueeze(-1)).squeeze(-1)

    return -log_probs_for_tokens.mean()



class Memorize_Trainer:
    def __init__(self,args: MemorizerTrainingArgs,model: Memorizer):
        super().__init__()
        self.model=model
        self.args=args
        self.data_loader=args.data_loader
        self.optimizer = t.optim.Adam(self.model.parameters(), lr=args.lr,weight_decay=0.00001)
        self.step = 0
        self.scheduler = t.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.95)
    def training_step(self, batch):
        input = batch[:,:-1]
        
        
        logits = self.model(input)
        if isinstance(logits, tuple):
            logits = logits.logits
        loss = get_log_probs(logits, batch)
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += 1
        wandb.log({"loss": loss.item()})
        #print(loss_entropy.item())
        return loss.item()  
    
    def validation_step(self, batch):
        input = batch[:,:-1]
        
        outputs = self.model(input)
        if isinstance(outputs, tuple):
            outputs = outputs.logits
        predicted = t.argmax(outputs, dim=-1)
        
        correct = (predicted[0][10:] == batch[0][11:]).flatten()
        return correct


    def train(self):
        #wandb.watch(self.model,get_log_probs,log="all",log_freq=1000)
        accuracy = 0
        progress_bar = tqdm(total = min(self.args.data_loader.num_batches,self.args.max_steps_per_epoch )* self.args.epochs)
        
        for epoch in range(self.args.epochs):
            np.random.seed(self.args.seed)
            self.args.seed += 1
            batches = []
            for i, batch in enumerate(self.train_loader(self.args.seed)):
                
                loss = self.training_step(batch)
                progress_bar.update()
                progress_bar.set_description(f"Epoch {epoch+1}, loss: {loss:.3f}, accuracy: {accuracy:.2f}")
                batches.append(batch)
                if i >= self.args.max_steps_per_epoch:
                    break

            correct_predictions = t.concat([self.validation_step(batch) for batch in batches])
            accuracy = correct_predictions.float().mean().item()
            self.scheduler.step()
            #print(accuracy)
            wandb.log({"accuracy": accuracy}, step=self.step)

        wandb.finish()
    
    def train_loader(self,seed):
        self.data_loader.shuffle(seed)
        return iter(self.data_loader)
 

#print(loss_e)
