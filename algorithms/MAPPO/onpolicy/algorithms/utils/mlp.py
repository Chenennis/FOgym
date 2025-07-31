import torch.nn as nn
import torch
import logging
from .util import init, get_clones

"""MLP modules."""

# 获取日志记录器
logger = logging.getLogger(__name__)

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.use_orthogonal = use_orthogonal
        self.use_ReLU = use_ReLU
        self.dimension_changes = 0  # 跟踪维度变化次数

        self._create_first_layer()
        
        # Create the rest of the layers
        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc2 = nn.ModuleList([nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size)) for i in range(self._layer_N)])
    
    def _create_first_layer(self, old_layer=None):
        """Create the first layer with current input_dim"""
        active_func = [nn.Tanh(), nn.ReLU()][self.use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self.use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][self.use_ReLU])
        
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        
        # 创建新的第一层
        new_fc1_linear = nn.Linear(self.input_dim, self.hidden_size)
        init_(new_fc1_linear)
        
        # 如果存在旧层，尝试保留部分权重
        if old_layer is not None:
            try:
                # 获取旧的线性层
                old_linear = None
                for module in old_layer:
                    if isinstance(module, nn.Linear):
                        old_linear = module
                        break
                
                if old_linear is not None:
                    old_input_dim = old_linear.weight.size(1)
                    # 如果新维度更大，保留旧权重
                    if self.input_dim > old_input_dim:
                        new_fc1_linear.weight.data[:, :old_input_dim] = old_linear.weight.data
                        new_fc1_linear.bias.data = old_linear.bias.data
                        logger.info(f"保留原始权重并扩展: {old_input_dim} → {self.input_dim}")
                    # 如果新维度更小，截断旧权重
                    else:
                        new_fc1_linear.weight.data = old_linear.weight.data[:, :self.input_dim]
                        new_fc1_linear.bias.data = old_linear.bias.data
                        logger.info(f"截断原始权重: {old_input_dim} → {self.input_dim}")
            except Exception as e:
                logger.warning(f"迁移权重失败: {e}")
        
        self.fc1 = nn.Sequential(
            new_fc1_linear, active_func, nn.LayerNorm(self.hidden_size))

    def update_input_dim(self, new_dim):
        """更新输入维度并重建第一层"""
        if new_dim == self.input_dim:
            return False
            
        self.dimension_changes += 1
        logger.warning(f"🔧 MLP输入维度变化: {self.input_dim} → {new_dim} (第{self.dimension_changes}次变化)")
        
        # 保存旧层以尝试迁移权重
        old_fc1 = self.fc1
        
        # 更新维度并重建
        self.input_dim = new_dim
        self._create_first_layer(old_fc1)
        
        return True

    def forward(self, x):
        # 检查输入维度是否变化
        input_dim = x.size(-1)
        if input_dim != self.input_dim:
            # 更新输入维度并重建第一层
            self.update_input_dim(input_dim)
            # 确保新层在正确的设备上
            self.fc1 = self.fc1.to(x.device)
        
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class MLPBase(nn.Module):
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size
        self.dimension_changes = 0  # 跟踪维度变化次数

        obs_dim = obs_shape[0]
        self.obs_dim = obs_dim  # Store the original observation dimension

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(obs_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        # 检查输入维度是否变化
        input_dim = x.size(-1)
        if self._use_feature_normalization and input_dim != self.obs_dim:
            # 更新feature_norm层以匹配新的输入维度
            self.obs_dim = input_dim
            old_feature_norm = self.feature_norm
            self.feature_norm = nn.LayerNorm(input_dim).to(x.device)
            self.dimension_changes += 1
            logger.warning(f"🔧 MLPBase输入维度变化: {self.obs_dim} → {input_dim} (第{self.dimension_changes}次变化)")
        
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x