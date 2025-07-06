import joblib
import os

class GalaxyClassifier:
    def __init__(self):
        self.models = {
            'morph': self._load_model('morphology.pkl'),
            'redshift': self._load_model('redshift.pkl'),
            'agn': self._load_model('agn.pkl')
        }
    
    def _load_model(self, model_name):
        path = os.path.join('models', model_name)
        return joblib.load(path) if os.path.exists(path) else None
    
    def predict(self, features):
        return {
            'morphology': self.models['morph'].predict([features])[0],
            'redshift': round(self.models['redshift'].predict([features])[0], 4) if self.models['redshift'] else None,
            'is_agn': bool(self.models['agn'].predict([features])[0]) if self.models['agn'] else None
        }