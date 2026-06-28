import logging
from .engine import AstroEngine
from .operators import AssetManager, SchemaAligner, IdentityRegistry

class AstroDBFacade:
    """
    鬼魂 DB 门面 (Facade)。
    Workflow 仅与此类对话，屏蔽底层 DuckDB 复杂度和算子实现细节。
    """
    def __init__(self, cluster_id: str):
        # 1. 基础配置与日志上下文绑定
        self.cluster_id = cluster_id
        self.logger = logging.getLogger(f"AstroDB.{cluster_id}")
        
        # 2. 初始化基础设施引擎
        self._engine = AstroEngine()
        
        # 3. 依赖注入
        self._asset_mgr = AssetManager(self._engine)
        self._aligner = SchemaAligner(self._engine)
        self._registry = IdentityRegistry(self._engine)
        
        self.logger.info(f"✨ AstroDB 门面已就绪，准备处理集群: {cluster_id}")

    # --- 上下文管理器协议支持 ---

    def __enter__(self) -> "AstroDBFacade":
        """进入 with 语句块时触发，返回自身以供 as db 语法使用"""
        # 如果进入时需要初始化连接，可以写在这里
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """离开 with 语句块时自动触发，确保连接被关闭，防止锁死"""
        try:
            self.close()
        except Exception as e:
            self.logger.error(f"关闭数据库连接时发生异常: {e}")
        # 返回 False 或不返回，让异常正常向上抛出
        return False

    # --- 核心业务 API ---

    def prepare_assets(self):
        """Stage 1.2: 挂载所有原始数据资产"""
        self.logger.info(f"正在进行资产挂载...")
        self._asset_mgr.mount_all(self.cluster_id)

    def align_schemas(self):
        """Stage 1.5: 资产清洗与物理视图对齐"""
        self.logger.info(f"正在进行 schema 对齐与视图注册...")
        self._aligner.align_all(self.cluster_id)

    def get_kinematic_identity(self, mode="static"):
        """获取/重构星团物理真身 DNA"""
        self.logger.info(f"正在获取/重构星团真身 DNA (模式: {mode})...")
        return self._registry.resolve(self.cluster_id, mode)

    def update_kinematic_identity(self, data):
        """滚动更新真身数据"""
        self.logger.info(f"正在持久化星团真身更新...")
        self._registry.update(self.cluster_id, data)

    # --- 辅助与基础设施接口 ---

    def get_view(self, view_name: str):
        """提供受控的数据读取通道"""
        return self._engine.query(f"SELECT * FROM {view_name}")

    def close(self):
        """关闭数据库连接"""
        self._engine.close()
        self.logger.info(f"💾 AstroDB 连接已关闭。")