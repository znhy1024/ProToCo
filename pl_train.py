import time
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DeepSpeedStrategy

from data import FinetuneDataModule, get_dataset_reader
from models.EncoderDecoder import EncoderDecoder
from models.modify_model import modify_transformer
from utils.Config import Config
from utils.util import ParseKwargs, set_seeds


def get_transformer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.origin_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.origin_model, low_cpu_mem_usage=True)
    tokenizer.model_max_length = config.max_seq_len
    model = modify_transformer(model, config)
    return tokenizer, model


def main(config):
    """
    Trains the model

    :param config:
    :return:
    # """

    tokenizer, model = get_transformer(config)
    dataset_reader = get_dataset_reader(config)
    datamodule = FinetuneDataModule(config, tokenizer, dataset_reader)
    model = EncoderDecoder(config, tokenizer, model, dataset_reader)

    logger = TensorBoardLogger(config.exp_dir, name="log") 

    trainer = Trainer(
        enable_checkpointing=False,
        gpus=torch.cuda.device_count(),
        amp_backend="native",
        strategy=config.compute_strategy,
        logger=logger,
        log_every_n_steps=4,
        max_steps=config.num_steps,
        min_steps=config.num_steps,
        num_sanity_val_steps=-1 if config.eval_before_training else 0,
        check_val_every_n_epoch=config.eval_epoch_interval,
        accumulate_grad_batches=config.grad_accum_factor,
        gradient_clip_val=config.grad_clip_norm,
    )
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_files", default="default.json",required=False)
    parser.add_argument("-k", "--kwargs", nargs="*", action=ParseKwargs, default={})
    args = parser.parse_args()

    config = Config(args.config_files, args.kwargs)

    
    print(f"Start experiment {config.exp_name}")
    # Setup config
    assert config.compute_strategy in ["none", "ddp", "deepspeed_stage_3_offload", "deepspeed_stage_3"]

    print(config.to_json())

    set_seeds(config.seed)
    main(config)
    print("Done")
    print(time.strftime("*****%Y-%m-%d %H:%M:%S*****",time.localtime()))
