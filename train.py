import torch as t
import torch.nn as nn
from model import Memorizer
import data_loader
import numpy as np
import wandb
from tqdm import tqdm


def get_log_probs(
    logits, 
    tokens
):

    log_probs = logits.log_softmax(dim=-1)
    # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    log_probs_for_tokens = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

    return log_probs_for_tokens

class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 11
    init_range: float = 0.02
    n_ctx: int = 100000
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12



class MemorizerTrainingArgs:
    seq_length: int = 999
    digits : int = 1000
    epochs : int = 1000
    max_steps_per_epoch = 50
    data_loader = data_loader.DataLoader('pi.dat', seq_length,digits)
    lr = 0.01
    momentum = 0.9
    wandb_project = "memorizer"
    wandb_name = None

class Memorize_Trainer:
    def __init__(self,args: MemorizerTrainingArgs,model: Memorizer):
        super().__init__()
        self.model=model
        self.args=args
        self.data_loader=args.data_loader
        self.optimizer = t.optim.SGD(self.model.parameters(), lr=args.lr,momentum=self.args.momentum)
        self.step = 0
        self.scheduler = t.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)
    def training_step(self, batch):
        input = batch
        
        
        logits = self.model(input)
        loss = -get_log_probs(logits, input).mean()
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += 1
        wandb.log({"loss": loss.item()})
        #print(loss_entropy.item())
        return loss.item()
    
    def validation_step(self, batch):
        input = batch
        
        #print(tokens.shape)
        outputs = self.model(input)
        #print(outputs.shape)
        predicted = t.argmax(outputs, dim=-1)
        #print(input)
        #print(predicted)
        #print(target)
        correct = (predicted[0][:-1] == input[0][1:]).flatten()
        return correct


    def train(self):
        wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=self.args)
        accuracy = np.nan
        progress_bar = tqdm(total = min(self.args.data_loader.num_batches,self.args.max_steps_per_epoch )* self.args.epochs)
        for epoch in range(self.args.epochs):
            for i, batch in enumerate(self.train_loader()):
                
                loss = self.training_step(batch)
                progress_bar.update()
                progress_bar.set_description(f"Epoch {epoch+1}, loss: {loss:.3f}, accuracy: {accuracy:.2f}")
                if i >= self.args.max_steps_per_epoch:
                    break
            correct_predictions = t.concat([self.validation_step(batch) for batch in self.train_loader()])
            accuracy = correct_predictions.float().mean().item()
            self.scheduler.step()
            #print(accuracy)
            wandb.log({"accuracy": accuracy}, step=self.step)

        wandb.finish()
    
    def train_loader(self):
        return iter(self.data_loader)
 
number_of_classes = 11

cfg = Config()
model = Memorizer(cfg)

args = MemorizerTrainingArgs()
trainer = Memorize_Trainer(args,model)
trainer.train()

#print(loss_e)

input = ["3","."]
input = [args.data_loader.char_to_ix[i] for i in input]
input = t.tensor(input).unsqueeze(0)
for i in range(30):
    output = model(input,input[:-1])
    next_digit = t.argmax(output, dim=-1)
    #print("answer")
    #print(next_digit)
    next_digit = next_digit[0][-1]
    input = t.cat((input,t.tensor([next_digit]).unsqueeze(-1)),dim=1)
    #print("next")
    #print(input)    

digits = input.numpy().flatten()

pi = [args.data_loader.ix_to_char[i] for i in digits]

print("".join(pi))

real = next(iter(args.data_loader))[0].numpy().flatten()[:32]
digits = [args.data_loader.ix_to_char[i] for i in real]

print("".join(digits))

equal = [i==j for i,j in zip(pi,digits)]
print(sum(equal)/len(equal))