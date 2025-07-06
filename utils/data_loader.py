import kagglehub
import pandas as pd

def load_data():
    """Load and preprocess SDSS data from Kaggle API"""
    df = kagglehub.load_dataset(
        "bryancimo/sdss-galaxy-classification-dr18",
        file_path="sdss_dr18.csv"
    ).rename(columns={
        'class': 'morphology',
        'redshift': 'z'
    })
    
    bands = ['u', 'g', 'r', 'i', 'z']
    for i in range(len(bands)-1):
        df[f'color_{bands[i]}{bands[i+1]}'] = df[bands[i]] - df[bands[i+1]]
    
    if all(col in df.columns for col in ['h_alpha_flux', 'h_beta_flux', 'oiii_5007_flux', 'nii_6584_flux']):
        df['OIII_HB'] = df['oiii_5007_flux'] / df['h_beta_flux']
        df['NII_HA'] = df['nii_6584_flux'] / df['h_alpha_flux']
        df['AGN'] = (df['OIII_HB'] > 3) & (df['NII_HA'] > 0.6)
    
    return df.dropna()