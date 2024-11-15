import hydra
import numpy as np
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="./config", config_name="config-preprocess")
def preprocessing(cfg):
    data = instantiate(cfg.data_loader).load()

    print("Process finished...")


if __name__ == "__main__":
    preprocessing()
