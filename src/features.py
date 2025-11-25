import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class BikeSharingPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.feature_names = None
        self.numerical_cols = ['temp', 'atemp', 'humidity', 'windspeed', 'year', 'month', 'day', 'hour', 'weekday']
        self.categorical_cols = ['season', 'weather', 'holiday', 'workingday']
        
        self.pipeline = None

    def _extract_date_features(self, df):
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['weekday'] = df['datetime'].dt.weekday
        df['hour'] = df['datetime'].dt.hour
        return df

    def fit_transform(self, df, y=None):
        df = self._extract_date_features(df)
        
        # 不要な列の削除
        drop_cols = ['datetime', 'casual', 'registered', 'count']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        
        # カテゴリカルと数値の分離
        # hour, month, weekday, season, weather はカテゴリカルとして扱う
        cat_features = ['season', 'weather', 'holiday', 'workingday', 'month', 'hour', 'weekday']
        # year, temp, atemp, humidity, windspeed は数値として扱う
        num_features = ['temp', 'atemp', 'humidity', 'windspeed', 'year']
        
        # データフレームに存在しない列がある場合の対策
        cat_features = [c for c in cat_features if c in X.columns]
        num_features = [c for c in num_features if c in X.columns]
        
        # Preprocessing Pipeline
        preprocessor_step1 = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', num_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
            ],
            verbose_feature_names_out=False
        )
        
        # Step 2: StandardScaler for everything
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor_step1),
            ('scaler', StandardScaler())
        ])
        
        X_transformed = self.pipeline.fit_transform(X)
        
        try:
            feature_names_step1 = preprocessor_step1.get_feature_names_out()
            self.feature_names = feature_names_step1
        except:
            self.feature_names = [f"feat_{i}" for i in range(X_transformed.shape[1])]
            
        return X_transformed

    def transform(self, df):
        df = self._extract_date_features(df)
        drop_cols = ['datetime', 'casual', 'registered', 'count']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        
        X_transformed = self.pipeline.transform(X)
        return X_transformed

def get_target(df):
    """ターゲット変数を取得し、対数変換して返す"""
    if 'count' in df.columns:
        y = df['count']
        y_log = np.log1p(y)
        return y_log
    return None
