from model import Memorizer
from train import Memorize_Trainer
import data_loader
import wandb
import torch as t

class Config:
    d_model: int = 300
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 11
    init_range: float = 0.02
    n_ctx: int = 100000
    d_head: int = 32
    d_mlp: int = 1200
    n_heads: int = 4
    n_layers: int = 4

class MemorizerTrainingArgs:
    seq_length: int = 1000
    digits : int = 100000
    epochs : int = 500
    max_steps_per_epoch = 2000
    data_loader = data_loader.DataLoader('pi.dat', seq_length,digits)
    lr = 0.0001
    wandb_project = "memorizer"
    wandb_name = "Small"
    seed = 0

number_of_classes = 11
cfg = Config()
model = Memorizer(cfg).cuda()
model.load_state_dict(t.load("small.pt"))
#t.save(model.state_dict(),"small.pt")
args = MemorizerTrainingArgs()
trainer = Memorize_Trainer(args,model)
with wandb.init(project=args.wandb_project, name=args.wandb_name, config=args):
    config = wandb.config
    trainer.train()
        
