import warnings

from invoke import task

warnings.filterwarnings("ignore")

@task
def train(ctx):
    from model.trainer import SMPTrainer

    config = ctx.segmentation_model
    if not config.wandb["logging"]:
        print(
            "Wandb logging doesn't work. You must specify 'wandb.logging=true' in invoke.yml to run logging."
        )

    trainer = SMPTrainer(config)
    trainer.run()

@task
def inference(ctx):
    from model.trainer import SMPTrainer

    config = ctx.segmentation_model
    if not config.wandb["logging"]:
        print(
            "Wandb logging doesn't work. You must specify 'wandb.logging=true' in invoke.yml to run logging."
        )
    print()

    print('Make sure that you have specified the path to the trained model and "setup.inference" is True in the "invoke.yml".')


    trainer = SMPTrainer(config)
    trainer.inference()
