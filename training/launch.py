from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer
import sys

with initialize(version_base=None, config_path="config"):
    cfg = compose(config_name=sys.argv[2] if len(sys.argv) > 2 else "default", overrides=["+defaults=[]"])

trainer = Trainer(**cfg)
trainer.run()
