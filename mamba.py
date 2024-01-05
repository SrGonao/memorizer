from model import Memorizer
from train import Memorize_Trainer
import data_loader
import wandb
import torch as t
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

class MemorizerTrainingArgs:
    seq_length: int = 30
    digits : int = 100000
    epochs : int = 100
    max_steps_per_epoch = 100000
    data_loader = data_loader.CurriculumLoader('pi.dat', seq_length,digits)
    lr = 0.00001
    wandb_project = "memorizer"
    wandb_name = "mamba"
    seed = 0

batch, length, dim = 2, 64, 16
d_model = 768
n_layer = 4
vocab_size = 11
model = MambaLMHeadModel(d_model,n_layer,vocab_size).cuda()

number_of_classes = 11

model.load_state_dict(t.load("curriculum-30-100k-100-n5-6.pt"))
#t.save(model.state_dict(),"mamba.pt")
args = MemorizerTrainingArgs()
trainer = Memorize_Trainer(args,model)
with wandb.init(project=args.wandb_project, name=args.wandb_name, config=args):
    config = wandb.config
    trainer.train()
t.save(model.state_dict(),"curriculum-50-100k-100-n5-2(8).pt")
loader = iter(trainer.train_loader(trainer.args.seed))
inputs = next(loader)

output = model(inputs)
next_digit = t.argmax(output.logits, dim=-1)
print(inputs)
print(next_digit)