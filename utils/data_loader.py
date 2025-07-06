import kagglehub
import pandas as pd
from pathlib import Path

def load_data():
    """
    Load and preprocess SDSS data from Kaggle with fallback options
    Returns:
        pd.DataFrame: Processed SDSS galaxy data with morphology, redshift, and AGN features
    """
    try:
        # Attempt new KaggleHub API syntax first
        try:
            df_path = kagglehub.model_download(
                'bryancimo/sdss-galaxy-classification-dr18',
                'sdss_dr18.csv',
                force_download=True
            )
            df = pd.read_csv(df_path) if isinstance(df_path, str) else df_path
        except:
            # Fallback to legacy API
            df = kagglehub.load_dataset(
                "bryancimo/sdss-galaxy-classification-dr18",
                file_name="sdss_dr18.csv"
            )

        # Standardize column names
        df = df.rename(columns={
            'class': 'morphology',
            'redshift': 'z'
        })

        # Feature engineering - color indices
        bands = ['u', 'g', 'r', 'i', 'z']
        for i in range(len(bands)-1):
            df[f'color_{bands[i]}{bands[i+1]}'] = df[bands[i]] - df[bands[i+1]]

        # AGN detection (if emission lines exist)
        if all(col in df.columns for col in ['h_alpha_flux', 'h_beta_flux', 'oiii_5007_flux', 'nii_6584_flux']):
            df['OIII_HB'] = df['oiii_5007_flux'] / df['h_beta_flux']
            df['NII_HA'] = df['nii_6584_flux'] / df['h_alpha_flux']
            df['AGN'] = (df['OIII_HB'] > 3) & (df['NII_HA'] > 0.6).astype(int)

        return df.dropna()

    except Exception as e:
        print(f"❌ Data loading failed: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Verify kagglehub installation: !pip show kagglehub")
        print("2. Check authentication: !kagglehub configure")
        print("3. Test direct download:")
        print("   !wget https://raw.githubusercontent.com/ItzmeOmeGa/Sdss_galaxy_classification/main/data/sdss_dr18.csv")
        raise
