from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from imblearn.over_sampling import SMOTE
import joblib

def train_models(df):
    """Train all models and save to disk"""
    X_morph = df.filter(regex='color|u|g|r|i|z')
    y_morph = df['morphology']
    
    X_train, X_test, y_train, y_test = train_test_split(X_morph, y_morph, test_size=0.2)
    X_res, y_res = SMOTE().fit_resample(X_train, y_train)
    
    morph_model = RandomForestClassifier().fit(X_res, y_res)
    joblib.dump(morph_model, 'models/morphology.pkl')
    
    rs_model = GradientBoostingRegressor().fit(X_morph, df['z'])
    joblib.dump(rs_model, 'models/redshift.pkl')
    
    if 'AGN' in df.columns:
        agn_model = RandomForestClassifier().fit(X_morph, df['AGN'])
        joblib.dump(agn_model, 'models/agn.pkl')
    
    return {
        'morph_accuracy': accuracy_score(y_test, morph_model.predict(X_test)),
        'rs_mae': mean_absolute_error(df['z'], rs_model.predict(X_morph)),
        'agn_accuracy': accuracy_score(df['AGN'], agn_model.predict(X_morph)) if 'AGN' in df.columns else None
    }