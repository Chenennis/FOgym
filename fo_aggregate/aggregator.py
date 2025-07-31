from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import pandas as pd
import math
import logging
import sys
import os
from dataclasses import dataclass
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# 添加项目根目录到系统路径以便导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入标准FlexOffer结构
from fo_common.flexoffer import FlexOffer, FOSlice

# 创建日志记录器
logger = logging.getLogger(__name__)

@dataclass
class AggregatedFlexOffer:
    """聚合后的FlexOffer (AFO)"""
    afo_id: str                           # 聚合FlexOffer ID
    source_fo_ids: List[str]              # 参与聚合的FO ID列表
    aggregated_fo: FlexOffer              # 聚合后的FO
    aggregation_method: str               # 聚合方法 ("LP" 或 "DP")
    total_energy_min: float = 0.0         # 总最小能量
    total_energy_max: float = 0.0         # 总最大能量
    power_profile_rmse: float = 0.0       # 功率轮廓RMSE
    power_profile_cv: float = 0.0         # 功率轮廓变异系数
    slice_count: int = 0                  # 时间片数量
    
    def __post_init__(self):
        """初始化后计算属性"""
        if self.aggregated_fo:
            self.total_energy_min = self.aggregated_fo.total_energy_min
            self.total_energy_max = self.aggregated_fo.total_energy_max
            self.slice_count = len(self.aggregated_fo.slices)
            self._calculate_power_metrics()
    
    def _calculate_power_metrics(self):
        """计算功率相关指标"""
        if not self.aggregated_fo.slices:
            return
        
        # 获取功率轮廓
        p_min, p_max = self.aggregated_fo.get_power_profile()
        avg_power = [(p_min[i] + p_max[i]) / 2 for i in range(len(p_min))]
        
        # 过滤掉可能的NaN值
        valid_power = [p for p in avg_power if not np.isnan(p)]
        
        # 如果没有有效值，设置默认值
        if not valid_power:
            self.power_profile_rmse = 0.0
            self.power_profile_cv = 0.0
            return
            
        # 计算RMSE（相对于目标功率阈值100kW）
        spt = 100.0  # 目标功率阈值
        self.power_profile_rmse = float(np.sqrt(np.mean([(p - spt) ** 2 for p in valid_power])))
        
        # 计算变异系数CV
        std_dev = float(np.std(valid_power))
        mean_power = float(np.mean(valid_power))
        self.power_profile_cv = std_dev / mean_power if mean_power != 0 else 0.0

class FOAggregator(ABC):
    """FlexOffer聚合器抽象基类"""
    
    def __init__(self, spt: float = 100.0, ppt: int = 23, tf_threshold: float = 1.0, 
                 power_deviation: float = 5.0):
        """
        初始化聚合器
        
        Args:
            spt: 切片功率阈值 (kW)
            ppt: 功率轮廓阈值 (小时)
            tf_threshold: 时间灵活性阈值
            power_deviation: 允许的功率偏差 (kW)
        """
        self.spt = spt  # Slice Power Threshold
        self.ppt = ppt  # Power Profile Threshold  
        self.tf_threshold = tf_threshold
        self.power_deviation = power_deviation
        self.results: List[AggregatedFlexOffer] = []
    
    @abstractmethod
    def initialize(self, flex_offers: List[FlexOffer]) -> Tuple[List[FlexOffer], List[FlexOffer], Optional[FlexOffer], int]:
        """
        初始化聚合过程
        
        Args:
            flex_offers: 输入的FlexOffer列表
            
        Returns:
            Tuple[PF, UF, fini, round]: 处理集合, 未处理集合, 初始FO, 轮次
        """
        pass
    
    def binary_aggregation(self, fo1: FlexOffer, fo2: FlexOffer) -> Optional[FlexOffer]:
        """
        二元聚合操作
        
        Args:
            fo1: 第一个FlexOffer
            fo2: 第二个FlexOffer
            
        Returns:
            聚合后的FlexOffer或None
        """
        # 检查兼容性
        if not fo1.is_compatible_with(fo2, self.tf_threshold):
            return None
        
        # 确保两个FO有相同的时间片数量
        max_slices = max(len(fo1.slices), len(fo2.slices))
        
        # 创建聚合后的时间片
        aggregated_slices = []
        for i in range(max_slices):
            # 获取两个FO在该时间片的能量
            e1_min, e1_max = fo1.get_energy_bounds(i) if i < len(fo1.slices) else (0.0, 0.0)
            e2_min, e2_max = fo2.get_energy_bounds(i) if i < len(fo2.slices) else (0.0, 0.0)
            
            # 聚合能量（简单相加）
            agg_e_min = e1_min + e2_min
            agg_e_max = e1_max + e2_max
            
            # 使用第一个FO的时间信息作为基准
            base_slice = fo1.slices[i] if i < len(fo1.slices) else fo2.slices[i]
            
            aggregated_slice = FOSlice(
                slice_id=i,
                start_time=base_slice.start_time,
                end_time=base_slice.end_time,
                energy_min=agg_e_min,
                energy_max=agg_e_max,
                duration_minutes=base_slice.duration_minutes,
                device_type="aggregated",
                device_id=f"agg_{fo1.device_id}_{fo2.device_id}"
            )
            aggregated_slices.append(aggregated_slice)
        
        # 创建聚合的FlexOffer
        aggregated_fo = FlexOffer(
            fo_id=f"agg_{fo1.fo_id}_{fo2.fo_id}",
            hour=fo1.hour,
            start_time=fo1.start_time,
            end_time=fo1.end_time,
            device_id=f"agg_{fo1.device_id}_{fo2.device_id}",
            device_type="aggregated",
            slices=aggregated_slices
        )
        
        return aggregated_fo
    
    def calculate_rmse(self, fo: FlexOffer) -> float:
        """计算FlexOffer相对于目标功率的RMSE"""
        p_min, p_max = fo.get_power_profile()
        avg_power = [(p_min[i] + p_max[i]) / 2 for i in range(len(p_min))]
        
        # 过滤掉可能的NaN或无穷大值
        valid_power = [p for p in avg_power if not np.isnan(p) and np.isfinite(p)]
        
        if not valid_power:
            return 0.0
            
        return np.sqrt(np.mean([(p - self.spt) ** 2 for p in valid_power]))
    
    def calculate_cv(self, fo: FlexOffer) -> float:
        """计算FlexOffer功率轮廓的变异系数"""
        p_min, p_max = fo.get_power_profile()
        avg_power = [(p_min[i] + p_max[i]) / 2 for i in range(len(p_min))]
        
        # 过滤掉可能的NaN或无穷大值
        valid_power = [p for p in avg_power if not np.isnan(p) and np.isfinite(p)]
        
        if not valid_power:
            return 0.0
            
        std_dev = float(np.std(valid_power))
        mean_power = float(np.mean(valid_power))
        
        if mean_power == 0:
            return 0.0
            
        return std_dev / mean_power
    
    def process(self, PF: List[FlexOffer], fini: FlexOffer) -> FlexOffer:
        """
        处理阶段 - 执行二元聚合操作
        
        Args:
            PF: 处理集合
            fini: 初始FlexOffer
            
        Returns:
            最终聚合的FlexOffer
        """
        current_fo = fini
        used_fos = [fini.fo_id]
        
        # 按时间灵活性降序排列PF
        PF_sorted = sorted(PF, key=lambda fo: fo.tf(), reverse=True)
        
        for candidate_fo in PF_sorted:
            if candidate_fo.fo_id in used_fos:
                continue
                
            # 尝试二元聚合
            aggregated = self.binary_aggregation(current_fo, candidate_fo)
            
            if aggregated:
                # 计算聚合后的质量指标
                new_rmse = self.calculate_rmse(aggregated)
                current_rmse = self.calculate_rmse(current_fo)
                
                # 如果RMSE改善，接受聚合
                if new_rmse < current_rmse:
                    new_cv = self.calculate_cv(aggregated)
                    current_cv = self.calculate_cv(current_fo)
                    
                    # 进一步检查CV是否改善
                    if new_cv <= current_cv:
                        current_fo = aggregated
                        used_fos.append(candidate_fo.fo_id)
                        logger.debug(f"聚合成功: {candidate_fo.fo_id}, RMSE: {new_rmse:.2f}, CV: {new_cv:.2f}")
        
        return current_fo
    
    def aggregate(self, flex_offers: List[FlexOffer]) -> List[AggregatedFlexOffer]:
        """
        聚合FlexOffer列表
        
        Args:
            flex_offers: 输入的FlexOffer列表
            
        Returns:
            聚合结果列表
        """
        self.results = []
        
        if not flex_offers:
            return self.results
        
        # 增强日志：记录聚合开始信息
        logger.info(f"开始聚合 - 方法: {self.__class__.__name__}, 输入FO数量: {len(flex_offers)}")
        logger.info(f"输入FO特征 - 平均轮廓尺寸: {sum(fo.profile_size() for fo in flex_offers) / len(flex_offers):.2f}, "
                   f"平均时间灵活性: {sum(fo.tf() for fo in flex_offers) / len(flex_offers):.2f}")
        
        # 初始化
        PF, UF, fini, round_num = self.initialize(flex_offers)
        
        logger.info(f"聚合初始化完成: PF={len(PF)}, UF={len(UF)}, 算法={self.__class__.__name__}")
        
        # 处理阶段 - 只有当fini不为None且PF不为空时才处理
        if fini is not None and PF:
            aggregated_fo = self.process(PF, fini)
            
            # 创建聚合结果
            source_fo_ids = [fini.fo_id] + [fo.fo_id for fo in PF]
            afo = AggregatedFlexOffer(
                afo_id=f"AFO_{round_num}_{self.__class__.__name__}",
                source_fo_ids=source_fo_ids,
                aggregated_fo=aggregated_fo,
                aggregation_method=self.__class__.__name__
            )
            
            self.results.append(afo)
            # 增强日志：记录详细的聚合结果信息
            logger.info(f"聚合完成: 方法={self.__class__.__name__}, AFO包含{len(source_fo_ids)}个FO, "
                       f"总能量范围[{afo.total_energy_min:.2f}, {afo.total_energy_max:.2f}], "
                       f"轮廓尺寸={afo.aggregated_fo.profile_size()}, "
                       f"时间灵活性={afo.aggregated_fo.tf():.2f}, "
                       f"功率RMSE={afo.power_profile_rmse:.2f}, "
                       f"功率CV={afo.power_profile_cv:.2f}")
        elif fini is not None:
            # 如果只有fini，没有PF，直接使用fini作为结果
            afo = AggregatedFlexOffer(
                afo_id=f"AFO_{round_num}_{self.__class__.__name__}_single",
                source_fo_ids=[fini.fo_id],
                aggregated_fo=fini,
                aggregation_method=self.__class__.__name__
            )
            self.results.append(afo)
            # 增强日志：记录单个FO聚合结果
            logger.info(f"单个FO聚合: 方法={self.__class__.__name__}, FO_ID={fini.fo_id}, "
                       f"总能量范围[{afo.total_energy_min:.2f}, {afo.total_energy_max:.2f}], "
                       f"轮廓尺寸={afo.aggregated_fo.profile_size()}")
        
        # 处理未处理集合中的FO
        for unused_fo in UF:
            afo = AggregatedFlexOffer(
                afo_id=f"AFO_unused_{unused_fo.fo_id}",
                source_fo_ids=[unused_fo.fo_id],
                aggregated_fo=unused_fo,
                aggregation_method=f"{self.__class__.__name__}_unused"
            )
            self.results.append(afo)
        
        # 增强日志：记录聚合最终结果
        logger.info(f"聚合结果统计: 方法={self.__class__.__name__}, 结果数量={len(self.results)}, "
                   f"处理FO数量={len(flex_offers) - len(UF)}, 未处理FO数量={len(UF)}")
        
        return self.results

class LongestProfileAggregator(FOAggregator):
    """Longest Profile (LP) 聚合算法"""
    
    def initialize(self, flex_offers: List[FlexOffer]) -> Tuple[List[FlexOffer], List[FlexOffer], Optional[FlexOffer], int]:
        """
        LP初始化方法
        
        基于算法2的实现：
        1. 找出所有具有最大轮廓尺寸的FO
        2. 在最长FO中选择时间灵活性最高的作为fini
        3. 所有其他FO加入处理集合PF
        """
        if not flex_offers:
            return [], [], None, 1
        
        # 步骤1：找出最大轮廓尺寸
        max_profile_size = max(fo.profile_size() for fo in flex_offers)
        longest_fos = [fo for fo in flex_offers if fo.profile_size() == max_profile_size]
        
        logger.info(f"LP初始化: 最大轮廓尺寸={max_profile_size}, 最长FO数量={len(longest_fos)}")
        
        # 步骤2：在最长FO中选择时间灵活性最高的作为fini
        fini = max(longest_fos, key=lambda fo: fo.tf())
        
        # 步骤3：所有其他FO加入处理集合PF
        PF = [fo for fo in flex_offers if fo.fo_id != fini.fo_id]
        UF = []  # LP方法中未处理集合为空
        
        logger.info(f"LP初始化完成: fini={fini.fo_id}(profile_size={fini.profile_size()}, tf={fini.tf():.2f})")
        
        return PF, UF, fini, 1

class DynamicProfileAggregator(FOAggregator):
    """Dynamic Profile (DP) 聚合算法"""
    
    def initialize(self, flex_offers: List[FlexOffer]) -> Tuple[List[FlexOffer], List[FlexOffer], Optional[FlexOffer], int]:
        """
        DP初始化方法
        
        基于算法3的实现：
        1. 计算轮廓尺寸的上围栏
        2. 过滤FO集合，排除异常值
        3. 在过滤后的集合中选择最长且最灵活的FO
        """
        if not flex_offers:
            return [], [], None, 1
        
        # 步骤1：计算上围栏
        profile_sizes = [fo.profile_size() for fo in flex_offers]
        uf = self._upper_fence_profile_size(profile_sizes)
        
        logger.info(f"DP初始化: 轮廓尺寸范围[{min(profile_sizes)}, {max(profile_sizes)}], 上围栏={uf:.2f}")
        
        # 步骤2：过滤FO集合
        PF_candidates = [fo for fo in flex_offers if fo.profile_size() <= uf]
        UF = [fo for fo in flex_offers if fo.profile_size() > uf]  # 异常值
        
        logger.info(f"DP过滤: 候选FO={len(PF_candidates)}, 异常值FO={len(UF)}")
        
        if not PF_candidates:
            # 如果所有FO都被过滤掉，回退到最小的FO
            fini = min(flex_offers, key=lambda fo: fo.profile_size())
            PF = [fo for fo in flex_offers if fo.fo_id != fini.fo_id]
            UF = []
            logger.warning("DP初始化: 所有FO都被过滤，回退到最小轮廓FO")
            return PF, UF, fini, 1
        
        # 步骤3：在过滤后的集合中选择最长且最灵活的FO
        max_size_in_pf = max(fo.profile_size() for fo in PF_candidates)
        longest_in_pf = [fo for fo in PF_candidates if fo.profile_size() == max_size_in_pf]
        fini = max(longest_in_pf, key=lambda fo: fo.tf())
        
        # 步骤4：从处理集合中移除fini
        PF = [fo for fo in PF_candidates if fo.fo_id != fini.fo_id]
        
        logger.info(f"DP初始化完成: fini={fini.fo_id}(profile_size={fini.profile_size()}, tf={fini.tf():.2f})")
        
        return PF, UF, fini, 1
    
    def _upper_fence_profile_size(self, sizes: List[int]) -> float:
        """
        使用四分位数方法计算上围栏
        Upper Fence = Q3 + 1.5 * IQR
        """
        if not sizes:
            return 0.0
        
        sorted_sizes = sorted(sizes)
        n = len(sorted_sizes)
        
        # 计算四分位数
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        
        q1 = sorted_sizes[q1_idx] if q1_idx < n else sorted_sizes[-1]
        q3 = sorted_sizes[q3_idx] if q3_idx < n else sorted_sizes[-1]
        
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        
        logger.debug(f"四分位数计算: Q1={q1}, Q3={q3}, IQR={iqr}, Upper Fence={upper_fence}")
        
        return upper_fence

class FOAggregatorFactory:
    """FlexOffer聚合器工厂"""
    
    @staticmethod
    def create_aggregator(method: str, **kwargs) -> FOAggregator:
        """
        创建聚合器
        
        Args:
            method: 聚合方法 ("LP" 或 "DP")
            **kwargs: 聚合器参数
            
        Returns:
            聚合器实例
        """
        if method.upper() == "LP":
            return LongestProfileAggregator(**kwargs)
        elif method.upper() == "DP":
            return DynamicProfileAggregator(**kwargs)
        else:
            raise ValueError(f"不支持的聚合方法: {method}. 支持的方法: LP, DP")
    
    @staticmethod
    def get_available_methods() -> List[str]:
        """获取可用的聚合方法列表"""
        return ["LP", "DP"]

# 便利函数
def aggregate_flex_offers(flex_offers: List[FlexOffer], method: str = "DP", **kwargs) -> List[AggregatedFlexOffer]:
    """
    聚合FlexOffer的便利函数
    
    Args:
        flex_offers: FlexOffer列表
        method: 聚合方法 ("LP" 或 "DP")
        **kwargs: 聚合器参数
        
    Returns:
        聚合结果列表
    """
    aggregator = FOAggregatorFactory.create_aggregator(method, **kwargs)
    return aggregator.aggregate(flex_offers) 