"""Evaluate a trained Conformer checkpoint on val and test sets."""
import sys, logging, torch, pytorch_lightning as pl
from omegaconf import OmegaConf
from hydra.utils import instantiate
from hydra import compose, initialize_config_dir
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config
config_dir = str(Path(__file__).parent / "config")
with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
    cfg = compose(config_name="discrete_gestures_transfer")

pl.seed_everything(cfg.seed, workers=True)

# Load the best checkpoint
ckpt_path = "D:/emg/method1_conformer/logs/2026-05-22/16-37-30/lightning_logs/version_0/checkpoints/epoch=64-step=227890.ckpt"
logger.info(f"Loading checkpoint from {ckpt_path}")

# Instantiate module and load checkpoint
module_class = instantiate(cfg.lightning_module, _convert_="all").__class__
module = module_class.load_from_checkpoint(ckpt_path)
module.eval()

# Set up data module
datamodule = instantiate(cfg.data_module, _convert_="all")

# Validate
trainer_val = pl.Trainer(accelerator="gpu", devices=1)
val_results = trainer_val.validate(model=module, datamodule=datamodule)
logger.info(f"Validation: {val_results}")

# Test (on CPU for long sequences)
trainer_test = pl.Trainer(accelerator="cpu", devices=1)
test_results = trainer_test.test(model=module, datamodule=datamodule)
logger.info(f"Test: {test_results}")
