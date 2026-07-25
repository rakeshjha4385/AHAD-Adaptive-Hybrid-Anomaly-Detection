import yaml


class Config:

    def __init__(self, config_path="config.yaml"):

        with open(config_path, "r") as file:

            self.config = yaml.safe_load(file)

    def get(self, *keys):

        value = self.config

        for key in keys:

            value = value[key]

        return value