import shutil
import sys
import time
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from .logger import LOG_PATH, init_logging

logger = init_logging(__name__)


def init_hydra_run(cfg):
    """_summary_

    Parameters
    ----------
    cfg : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("")
    logger.info(OmegaConf.to_yaml(cfg))

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir_str = hydra_cfg.runtime.output_dir
    output_dir = Path(output_dir_str)
    logger.info(output_dir)

    save_latest_dir = Path(cfg.save_latest_dir)
    if save_latest_dir.exists():
        shutil.rmtree(save_latest_dir)
    save_latest_dir.mkdir(parents=True)

    return output_dir


def change_hydra_output_dir(output_dir: str):
    current_date_time = time.strftime("%Y-%m-%d/%H-%M-%S")
    output_dir = output_dir + f"/{current_date_time}"
    sys.argv.append(f"hydra.run.dir={output_dir}")
