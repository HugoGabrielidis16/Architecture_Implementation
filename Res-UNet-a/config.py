import yaml


def load_config():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


config = load_config()

architecture = config["model"]["architecture"]
training_params = config["training"]
