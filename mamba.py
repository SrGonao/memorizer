from model import Memorizer
from train import Memorize_Trainer
import data_loader
import wandb
import torch as t
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

class MemorizerTrainingArgs:
    seq_length: int = 10000
    digits : int = 100000
    epochs : int = 3
    max_steps_per_epoch = 10000
    data_loader = data_loader.DataLoader('pi.dat', seq_length,digits)
    lr = 0.00001
    wandb_project = "memorizer"
    wandb_name = "mamba"
    seed = 0

batch, length, dim = 2, 64, 16
d_model = 300
n_layer = 4
vocab_size = 11
model = MambaLMHeadModel(d_model,n_layer,vocab_size).cuda()

number_of_classes = 11

#model.load_state_dict(t.load("mamba.pt"))
#t.save(model.state_dict(),"mamba.pt")
args = MemorizerTrainingArgs()
trainer = Memorize_Trainer(args,model)
with wandb.init(project=args.wandb_project, name=args.wandb_name, config=args):
    config = wandb.config
    trainer.train()
        
inputs = next(iter(trainer.train_loader(trainer.args.seed)))

model(inputs)
