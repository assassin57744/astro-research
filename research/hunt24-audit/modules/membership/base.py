from abc import ABC, abstractmethod
import pandas as pd

class BaseDisambiguation(ABC):
    """阶段一基类：负责将全量天区划分为星团系统与背景野星"""
    @abstractmethod
    def fit_predict(self, df_all: pd.DataFrame, df_seeds: pd.DataFrame, features: list) -> pd.DataFrame:
        pass

class BaseSubstructure(ABC):
    """阶段二基类：负责对精筛后的成员星进行动力学子结构解剖"""
    @abstractmethod
    def analyze(self, df_members: pd.DataFrame, features: list) -> pd.DataFrame:
        pass