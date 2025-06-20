import yaml

class ConfigLoader:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get_param_grid(self, model_name):
        return self.config.get(model_name, {}).get('param_grid', {})

    def get_best_params(self, model_name):
        return self.config.get(model_name, {}).get('best_params', {}) 