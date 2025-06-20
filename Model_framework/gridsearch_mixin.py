from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

class GridSearchMixin:
    def grid_search(self, X, y, param_grid, cv=5, scoring=None):
        gs = HalvingGridSearchCV(self.model, param_grid, cv=cv, factor=2, verbose = 2, scoring=scoring)
        gs.fit(X, y)
        self.model = gs.best_estimator_
        return gs.best_params_, gs.best_score_ 
    
    