# FlexOffer系统数据文件说明

本文档详细说明FlexOffer多智能体强化学习系统所使用的所有数据文件格式、结构和使用方法。所有数据文件均为CSV格式，使用UTF-8编码。

## 📋 目录

- [1. 数据文件概览](#1-数据文件概览)
- [2. 天气数据](#2-天气数据)
- [3. 电价数据](#3-电价数据)
- [4. 用户与管理器配置](#4-用户与管理器配置)
- [5. 设备配置数据](#5-设备配置数据)
- [6. 能源需求数据](#6-能源需求数据)
- [7. 电池系统数据](#7-电池系统数据)
- [8. 热泵系统数据](#8-热泵系统数据)
- [9. 不确定性能源数据](#9-不确定性能源数据)
- [10. 数据加载与使用](#10-数据加载与使用)
- [11. 数据获取与更新建议](#11-数据获取与更新建议)

## 1. 数据文件概览

| 文件名 | 描述 | 优先级 | 必要性 |
|--------|------|--------|--------|
| weather_data.csv | 天气数据（温度、辐照度、风速） | 高 | 必需 |
| grid_price.csv | 丹麦电网电价数据 | 高 | 必需 |
| user_config_36users.csv | 36个用户的配置信息 | 高 | 必需 |
| manager_config_36users.csv | 4个管理器的配置信息 | 高 | 必需 |
| device_config_36users.csv | 所有设备的配置信息 | 高 | 必需 |
| scenario_config_36users.csv | 场景配置信息 | 高 | 必需 |
| user_demands.csv | 用户能源需求数据 | 中 | 必需 |
| battery_base_parameters.csv | 电池基本参数 | 中 | 必需 |
| battery_dfo_input.csv | 电池FlexOffer输入参数 | 中 | 必需 |
| heat_pump_system.csv | 热泵系统参数 | 中 | 必需 |
| uncertain_energy_data.csv | 不确定性能源数据 | 低 | 可选 |

## 2. 天气数据

### 文件名: weather_data.csv

### 数据维度：4维
- **timestamp**: 时间戳 (ISO 8601格式)
- **temperature**: 温度 (摄氏度)
- **solar_irradiance**: 太阳辐照度 (W/m²)
- **wind_speed**: 风速 (m/s)

### 示例数据
```csv
timestamp,temperature,solar_irradiance,wind_speed
2024-01-15T00:00:00,2.5,0,8.2
2024-01-15T01:00:00,2.1,0,7.8
2024-01-15T02:00:00,1.8,0,7.5
...
```

### 数据特点
- 丹麦冬季温度通常在-5°C到10°C之间
- 夏季温度通常在10°C到25°C之间
- 太阳辐照度在冬季较低（最高400-500 W/m²），夏季较高（最高800-1000 W/m²）
- 风速通常在5-15 m/s之间

### 使用场景
- 光伏发电预测
- 热泵能效计算
- 建筑物热损失估计
- 可再生能源产量预测

## 3. 电价数据

### 文件名: grid_price.csv

### 数据维度：6维
- **timestamp**: 时间戳 (YYYY-MM-DD HH:MM:SS)
- **hour**: 小时 (0-23)
- **day_type**: 日期类型 (weekday/weekend)
- **price_dkk_kwh**: 丹麦克朗电价 (DKK/kWh)
- **price_usd_kwh**: 美元电价 (USD/kWh)
- **price_level**: 价格水平 (low/rising/valley/high/peak/falling/medium)

### 示例数据
```csv
timestamp,hour,day_type,price_dkk_kwh,price_usd_kwh,price_level
2024-12-06 00:00:00,0,weekday,0.85,0.12,low
2024-12-06 01:00:00,1,weekday,0.82,0.12,low
...
```

### 丹麦电价特征模式

#### 工作日电价模式 (weekday)
```
时间段          | 电价特征        | 价格水平      | USD/kWh范围
0:00-5:00      | 较低           | low          | 0.10-0.12
6:00-9:00      | 上升到较高      | rising/high  | 0.16-0.23
10:00-16:00    | 电价低谷        | valley       | 0.13-0.15
17:00-21:00    | 高峰           | peak         | 0.24-0.27
22:00-23:00    | 下降           | falling      | 0.15-0.19
```

#### 休息日电价模式 (weekend)
```
时间段          | 电价特征        | 价格水平      | USD/kWh范围
0:00-5:00      | 较低           | low          | 0.10-0.11
6:00-9:00      | 缓慢上升        | rising       | 0.12-0.16
10:00-16:00    | 中等偏低        | medium       | 0.15-0.17
17:00-21:00    | 高峰(较工作日低) | peak         | 0.22-0.25
22:00-23:00    | 下降           | falling      | 0.14-0.18
```

### 价格水平说明
- **low**: < 0.12 USD/kWh (夜间低价)
- **medium**: 0.12-0.16 USD/kWh (日间平价)
- **high**: 0.16-0.20 USD/kWh (日间高价)
- **peak**: > 0.20 USD/kWh (高峰电价)
- **rising**: 价格上升期
- **falling**: 价格下降期
- **valley**: 价格低谷期

### 电价优先级体系
1. **优先级1**: `grid_price.csv` - 丹麦实际电价数据
2. **优先级2**: 传统 `price_data.csv` 文件
3. **优先级3**: 动态电价预测（基于丹麦电价模式）

### 使用方法
```python
from fo_generate.price_loader import PriceLoader
from datetime import datetime

# 初始化电价加载器
price_loader = PriceLoader("data")

# 获取24小时电价数据
start_time = datetime.now()
price_data = price_loader.get_price_data(start_time, 24)

# 当前电价信息
current_price = price_loader.get_current_price(datetime.now())
print(f"当前电价: {current_price['price']:.4f} USD/kWh")
print(f"价格水平: {current_price['price_level']}")
```

## 4. 用户与管理器配置

### 管理器配置文件: manager_config_36users.csv

#### 数据维度：6维
- **manager_id**: Manager ID
- **location_x**: X坐标 (km)
- **location_y**: Y坐标 (km)
- **coverage_area**: 覆盖面积 (km²)
- **user_count**: 用户数量
- **district_type**: 小区类型 ("residential", "commercial", "mixed")

#### 示例数据
```csv
manager_id,location_x,location_y,coverage_area,user_count,district_type
manager_1,2.5,3.2,1.5,6,residential
manager_2,5.8,7.1,2.3,10,mixed
manager_3,8.2,4.6,1.8,8,residential
manager_4,11.5,9.3,3.1,12,commercial
```

### 用户配置文件: user_config_36users.csv

#### 数据维度：8维
- **user_id**: 用户ID
- **manager_id**: 所属Manager ID
- **location_x**: X坐标 (km)
- **location_y**: Y坐标 (km)
- **user_type**: 用户类型 ("prosumer", "consumer", "producer")
- **economic_pref**: 经济性偏好 (0-1)
- **comfort_pref**: 舒适性偏好 (0-1)
- **environmental_pref**: 环保性偏好 (0-1)

#### 示例数据
```csv
user_id,manager_id,location_x,location_y,user_type,economic_pref,comfort_pref,environmental_pref
user_1,manager_1,2.3,3.1,prosumer,0.3,0.4,0.3
user_2,manager_1,2.7,3.3,consumer,0.5,0.3,0.2
...
```

### 场景配置文件: scenario_config_36users.csv

#### 数据维度：3维
- **scenario_id**: 场景ID
- **description**: 场景描述
- **parameters**: 场景参数（JSON格式）

#### 使用场景
- 定义Manager-User层级关系
- 设置用户偏好权重
- 配置地理分布情况
- 定义用户类型分布

## 5. 设备配置数据

### 文件名: device_config_36users.csv

### 数据维度：10维
- **device_id**: 设备ID
- **user_id**: 所属用户ID
- **device_type**: 设备类型 ("battery", "heat_pump", "ev", "pv", "dishwasher")
- **capacity**: 容量 (kWh或其他单位)
- **max_power**: 最大功率 (kW)
- **efficiency**: 效率 (0-1)
- **initial_state**: 初始状态 (SOC、温度等)
- **param1**: 设备特定参数1
- **param2**: 设备特定参数2
- **param3**: 设备特定参数3

### 示例数据
```csv
device_id,user_id,device_type,capacity,max_power,efficiency,initial_state,param1,param2,param3
battery_1,user_1,battery,10.0,5.0,0.95,0.5,0.1,0.9,
heatpump_1,user_1,heat_pump,0.0,3.0,3.5,20.0,18.0,26.0,0.1
ev_1,user_2,ev,60.0,7.0,0.9,0.3,0.1,0.95,
pv_1,user_1,pv,0.0,5.0,0.18,0.0,30.0,180.0,28.0
dishwasher_1,user_3,dishwasher,3.0,1.2,0.85,0.0,3.5,0.5,6.0
```

### 设备类型参数说明

#### 电池储能系统 (Battery)
- **capacity**: 电池容量 (kWh)
- **max_power**: 最大充放电功率 (kW)
- **efficiency**: 充放电效率 (0.8-0.95)
- **initial_state**: 初始SOC (0.1-0.9)
- **param1**: 最小SOC (0.1)
- **param2**: 最大SOC (0.9)

#### 热泵系统 (Heat Pump)
- **capacity**: 不适用 (0.0)
- **max_power**: 最大功率 (kW)
- **efficiency**: COP系数 (3.0-4.5)
- **initial_state**: 初始温度 (°C)
- **param1**: 最低温度 (°C)
- **param2**: 最高温度 (°C)
- **param3**: 热损失系数 (0.1-0.2)

#### 电动汽车 (EV)
- **capacity**: 电池容量 (kWh)
- **max_power**: 最大充电功率 (kW)
- **efficiency**: 充电效率 (0.85-0.92)
- **initial_state**: 初始SOC (0.1-0.9)
- **param1**: 最小SOC (0.1)
- **param2**: 最大SOC (0.95)

#### 光伏系统 (PV)
- **capacity**: 不适用 (0.0)
- **max_power**: 最大发电功率 (kW)
- **efficiency**: 转换效率 (0.15-0.22)
- **initial_state**: 不适用 (0.0)
- **param1**: 倾斜角度 (°)
- **param2**: 方位角 (°)
- **param3**: 面板面积 (m²)

#### 洗碗机 (Dishwasher)
- **capacity**: 总能量需求 (kWh)
- **max_power**: 额定功率 (kW)
- **efficiency**: 能效 (0.8-0.9)
- **initial_state**: 不适用 (0.0)
- **param1**: 运行时长 (h)
- **param2**: 最小启动延迟 (h)
- **param3**: 最大启动延迟 (h)

### 设备部署率
- 热泵系统: 100% (36/36用户)
- 洗碗机: 100% (36/36用户)
- 电池储能系统: 67% (24/36用户)
- 电动汽车: 39% (14/36用户)
- 光伏系统: 22% (8/36用户)

## 6. 能源需求数据

### 文件名: user_demands.csv

### 数据维度
- **user_id**: 用户ID
- **timestamp**: 时间戳
- **heating_demand**: 供暖需求 (kWh)
- **electricity_demand**: 电力需求 (kWh)
- **hot_water_demand**: 热水需求 (kWh)
- **ev_charging_need**: 电动车充电需求 (kWh)

### 使用场景
- 热泵供暖控制
- 电动车充电规划
- 用户负荷预测
- 能源消耗模式分析

## 7. 电池系统数据

### 基本参数文件: battery_base_parameters.csv

#### 数据维度
- **battery_type**: 电池类型
- **nominal_voltage**: 额定电压 (V)
- **cycle_life**: 循环寿命
- **energy_density**: 能量密度 (Wh/kg)
- **self_discharge_rate**: 自放电率 (%/month)
- **calendar_life**: 日历寿命 (年)
- **cost_per_kwh**: 每千瓦时成本 (USD/kWh)

### FlexOffer输入文件: battery_dfo_input.csv

#### 数据维度
- **battery_id**: 电池ID
- **time_step**: 时间步
- **min_power**: 最小功率 (kW)
- **max_power**: 最大功率 (kW)
- **min_energy**: 最小能量 (kWh)
- **max_energy**: 最大能量 (kWh)
- **flexibility_factor**: 灵活性因子 (0-1)

## 8. 热泵系统数据

### 文件名: heat_pump_system.csv

### 数据维度
- **heat_pump_id**: 热泵ID
- **heat_pump_type**: 热泵类型
- **cop_reference**: 参考COP值
- **thermal_capacity**: 热容量 (kW)
- **temperature_lift**: 温度提升 (°C)
- **min_part_load**: 最小部分负载 (%)
- **max_flow_temp**: 最大流动温度 (°C)
- **defrost_energy**: 除霜能耗 (kWh)

### 使用场景
- 热泵效率计算
- 温度控制优化
- 供暖能耗预测
- FlexOffer生成

## 9. 不确定性能源数据

### 文件名: uncertain_energy_data.csv

### 数据维度
- **timestamp**: 时间戳
- **energy_type**: 能源类型
- **expected_value**: 期望值
- **uncertainty_low**: 低不确定性边界
- **uncertainty_high**: 高不确定性边界
- **confidence_level**: 置信水平

### 使用场景
- 鲁棒性优化
- 风险评估
- 不确定性建模
- 场景生成

## 10. 数据加载与使用

### 数据加载器
FlexOffer系统提供了一系列数据加载器，用于处理不同类型的数据：

```python
from fo_generate.data_loader import DataLoader
from fo_generate.price_loader import PriceLoader
from fo_generate.weather_loader import WeatherLoader

# 初始化数据加载器
data_loader = DataLoader("data")
price_loader = PriceLoader("data")
weather_loader = WeatherLoader("data")

# 加载用户配置
users = data_loader.load_users("user_config_36users.csv")

# 加载设备配置
devices = data_loader.load_devices("device_config_36users.csv")

# 加载天气数据
weather = weather_loader.load_weather_data()

# 加载电价数据
prices = price_loader.load_price_data()
```

### 集成到FlexOffer Pipeline
```python
from run_fo_pipeline import FOPipeline

# 创建FlexOffer Pipeline
pipeline = FOPipeline({
    'data_dir': 'data',
    'rl_algorithm': 'fomappo',
    'num_episodes': 100,
    'log_verbosity': 'brief'
})

# 运行Pipeline
pipeline.run()
```

## 11. 数据获取与更新建议

### 丹麦数据源
1. **天气数据**: DMI (Danish Meteorological Institute) - https://www.dmi.dk/
2. **电价数据**: Energinet - https://www.energinet.dk/
3. **光伏数据**: PVGIS (European Commission) - https://re.jrc.ec.europa.eu/pvg_tools/en/

### 数据更新频率
- 天气数据：每小时更新
- 电价数据：每小时更新（日前市场）
- 光伏预测：每小时更新，提供24小时预测
- 工作日数据：年度更新

### 数据质量要求
- 时间戳必须连续，无缺失
- 数值范围必须合理
- 预测数据需要包含不确定性信息
- 所有时间戳使用UTC+1（丹麦时间）

### 默认数据生成
如果外部数据不可用，系统将使用内置的丹麦天气和电价模型生成模拟数据。模拟数据基于丹麦的典型气候和电力市场特征。 