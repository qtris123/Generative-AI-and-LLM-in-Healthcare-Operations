class ModelManager:
    def __init__(self):
        self.models = {}

    def register(self, name, adapter):
        self.models[name] = adapter

    def get(self, name):
        return self.models.get(name)

    def list_models(self):
        return list(self.models.keys()) 