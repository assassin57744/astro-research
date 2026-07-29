# __init__.py
import logging
import pandas as pd

# 内部绝对路径导入
from modules.pipelines.core.disambiguation.bayesian import BayesianGmmDisambiguation
from modules.pipelines.core.disambiguation.threshold import ThresholdGmmDisambiguation
from modules.pipelines.core.disambiguation.blind import BlindGmmDisambiguation

# from .substructure.identity import IdentitySubstructure
# from .substructure.dual_component import DualCompSubstructure
# from .substructure.triple_component import TripleCompSubstructure

logger = logging.getLogger("hunt24.membership")

class AstroMembershipPipeline:
    """
    统一的天文成员星净化与结构解剖流水线控制器。
    完美解耦阶段一（Disambiguation）与阶段二（Substructure）。
    """
    DISAMBIGUATION_MAP = {
        "bayesian_gmm": BayesianGmmDisambiguation,
        "threshold_gmm": ThresholdGmmDisambiguation,
        "blind_gmm": BlindGmmDisambiguation
    }
    
    # SUBSTRUCTURE_MAP = {
    #     "identity": IdentitySubstructure,
    #     "dual_comp": DualCompSubstructure,
    #     "triple_comp": TripleCompSubstructure
    # }

    def __init__(self, disambiguation_mode: str, substructure_mode: str, **kwargs):
        dis_cls = self.DISAMBIGUATION_MAP.get(disambiguation_mode, BayesianGmmDisambiguation)
        sub_cls = self.SUBSTRUCTURE_MAP.get(substructure_mode, IdentitySubstructure)
        
        self.disambiguation_strategy = dis_cls(**kwargs)
        self.substructure_strategy = sub_cls(**kwargs)

    def process(self, df_all: pd.DataFrame, df_seeds: pd.DataFrame, features: list = None) -> pd.DataFrame:
        if features is None:
            features = ['ra', 'dec', 'plx', 'pmra', 'pmdec']
            
        # 1. 运行背景消除
        df_res = self.disambiguation_strategy.fit_predict(df_all, df_seeds, features)
        
        is_member = df_res['is_member'] == True
        df_members = df_res[is_member].copy()
        df_field = df_res[~is_member].copy()
        df_field['substructure_label'] = 'Field'
        
        # 2. 运行子结构解剖
        if not df_members.empty:
            df_members_analyzed = self.substructure_strategy.analyze(df_members, features)
            final_df = pd.concat([df_members_analyzed, df_field], axis=0)
        else:
            final_df = df_field
            
        return final_df.sort_index()