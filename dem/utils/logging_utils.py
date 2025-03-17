from io import BytesIO
from typing import Any, Dict

import PIL
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf, DictConfig
import yaml

from dem.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")
    hparams["energy"] = cfg["energy"]

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        # print(f"logging hparams to {logger} = {logger.__class__.__name__}")
        logger.log_hyperparams(hparams)


def fig_to_image(fig):
    try:
        buffer = BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
        buffer.seek(0)

        return PIL.Image.open(buffer)

    except Exception as e:
        fig.canvas.draw()
        return PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.renderer.buffer_rgba()
        )


def get_name_from_config(cfg: DictConfig) -> str:
    """Human-readable name for wandb. Pretty janky."""
    name = ""
    energy = cfg["energy"]["_target_"].split(".")[-1]
    if "DoubleWell" in energy:
        energy_name = "DW"
    elif "AlDi" in energy:
        energy_name = "AlDi"
    elif "GMM" in energy:
        energy_name = "GMM"
    else:
        energy_name = energy
    name += f"{energy_name} "
    if "GAD" in energy:
        name += "GAD"
    elif "Pseudo" in energy:
        if cfg["energy"]["term_aggr"] in ["cond_force", "cond_force_proj"]:
            name += f"(|F|^{cfg['energy']['force_exponent']})"
        else:
            name += f"({cfg['energy']['force_activation']}{cfg['energy']['force_scale']} {cfg['energy']['hessian_eigenvalue_penalty']}{cfg['energy']['hessian_scale']} {cfg['energy']['term_aggr']})"
    if "temperature" in cfg["energy"] and cfg["energy"]["temperature"] != 1.0:
        name += f"T{cfg['energy']['temperature']}"
    return name
