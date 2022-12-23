import wandb
from omegaconf import OmegaConf

_logging_enabled = False


def wandb_init(config):
    global _logging_enabled
    _ = wandb.init(
        entity=config["entity"],
        project=config["project"],
        settings=wandb.Settings(start_method="thread"),
        config=wandb.config,
    )
    _logging_enabled = True


def wandb_log(args):
    if _logging_enabled:
        wandb.log(args)
