import shutil

import hydra
from omegaconf import DictConfig

from utils.hydra_utils import init_hydra_run
from utils.logger import LOG_PATH, init_logging

logger = init_logging(__name__)


def main():
    pass


@hydra.main(version_base="1.2", config_path="config", config_name="main")
def main_cli(cfg: DictConfig) -> None:
    output_dir = init_hydra_run(cfg)

    main()

    shutil.copy(LOG_PATH, f"{cfg.save_latest_dir}/{__name__}.log")
    shutil.copytree(cfg.save_latest_dir, output_dir.joinpath("latest"))
    logger.info(f"log file copied to %s/latest/{__name__}.log", output_dir)


if __name__ == "__main__":
    main_cli()
