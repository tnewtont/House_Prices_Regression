import pandas as pd
import numpy as np
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import Lasso

# — custom FeatureGenerator to replicate all of your manual steps —
class FeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, f1f2_pairs):
        self.f1f2_pairs = f1f2_pairs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # (1) All your numeric feature‐engineering:
        df['MSSubClass'] = df['MSSubClass'].astype(str)
        df['MSSubClass']        = df['MSSubClass'].astype(str)
        df['LotDepth']          = df['LotArea'] / df['LotFrontage']
        df['Frontage_Area_Ratio'] = df['LotFrontage'] / df['LotArea']
        df['LogLotArea']        = np.log1p(df['LotArea'])
        df['LogLotFrontage']    = np.log1p(df['LotFrontage'])
        df['HouseAge']          = df['YrSold'] - df['YearBuilt']
        df['Remodeled']         = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)
        df['MasVnrArea']        = df['MasVnrArea'].fillna(0)
        df['HasMasVnr']         = (df['MasVnrArea'] > 0).astype(int)
        df['LogMasVnrArea']     = np.log1p(df['MasVnrArea'])
        df['VeneerAreaRatio']   = df['MasVnrArea'] / df['GrLivArea']
        df[['BsmtFinSF1','BsmtFinSF2']] = df[['BsmtFinSF1','BsmtFinSF2']].fillna(0)
        df['TotalBsmtFinSF']    = df['BsmtFinSF1'] + df['BsmtFinSF2']
        df['TotalBsmtSF']       = df['TotalBsmtSF'].fillna(0)
        df['BsmtUnfSF']         = df['BsmtUnfSF'].fillna(0)
        df['FinishedBsmtRatio'] = df['TotalBsmtFinSF'] / df['TotalBsmtSF'].replace(0,np.nan)
        df['UnfinishedBsmtRatio']= df['BsmtUnfSF'] / df['TotalBsmtSF'].replace(0,np.nan)
        df['Has2ndFlr']         = (df['2ndFlrSF']>0).astype(int)
        df['TotalFlrSF']        = df['1stFlrSF'] + df['2ndFlrSF']
        df['Prop2ndFlr']        = df['2ndFlrSF'] / df['TotalFlrSF']
        df['FlrRatio']          = df['2ndFlrSF'] / df['1stFlrSF']
        df['HasLowQualFin']     = (df['LowQualFinSF']>0).astype(int)
        df['OverallQual_LowQual_Interact'] = df['LowQualFinSF'] * df['OverallQual']
        df['LogGrLivArea']      = np.log1p(df['GrLivArea'])
        df['Qual_GrLivArea']    = df['GrLivArea'] * df['OverallQual']
        df['GrLivAreaBin'] = pd.qcut(
        df['GrLivArea'],
        q=4,
        labels=False,
        duplicates='drop')
        df['TotalBsmtBath']     = df['BsmtFullBath'] + 0.5*df['BsmtHalfBath']
        df['BedroomAbvGr']      = df['BedroomAbvGr'].clip(upper=6)
        df.loc[df['KitchenAbvGr']==0,'KitchenAbvGr']=1
        df['Fireplaces']        = (df['Fireplaces']>0).astype(int)
        df[['GarageCars','GarageArea']] = df[['GarageCars','GarageArea']].fillna(0)
        df['HasGarage']         = df['GarageType'].notnull().astype(int)
        df[['GarageQual','GarageCond','GarageType','GarageFinish']] = df[['GarageQual','GarageCond','GarageType','GarageFinish']].fillna('None')
        df['LogWoodDeckSF']     = np.log1p(df['WoodDeckSF'])
        df['HasWoodDeck']       = (df['WoodDeckSF']>0).astype(int)
        for c in ['OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch']:
            df[f"{c}_log"] = np.log1p(df[c])
            df[f"Has_{c}"] = (df[c]>0).astype(int)
        df['TotalPorchSF']      = df[['OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch']].sum(axis=1)
        df['TotalPorchSF_log']  = np.log1p(df['TotalPorchSF'])
        df['HasPool']           = (df['PoolArea']>0).astype(int)
        df.drop(columns=['PoolArea'], inplace=True)
        df.drop(columns=['MiscVal'],inplace=True)
        df['MoSold_sin']        = np.sin(2*np.pi*df['MoSold']/12)
        df['MoSold_cos']        = np.cos(2*np.pi*df['MoSold']/12)
        for f1, f2 in self.f1f2_pairs:
            df[f"{f1}_x_{f2}"] = df[f1] * df[f2]

        # (2) Drop columns you never use downstream (e.g., raw IDs, misc)
        df = df.drop(columns=['Id','MiscVal','PoolQC','Alley'], errors='ignore')

        return df