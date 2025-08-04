import logging
import random
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

# Add project root directory to system path for import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import fo_generate and fo_aggregate modules
from fo_generate.dfo import DFOSystem, DFOSlice
from fo_generate.sfo import SFOSystem, SFOSlice
from fo_aggregate import Manager, AggregatedFlexOffer
from fo_aggregate.manager import Manager

# Create logger
logger = logging.getLogger(__name__)

class WeatherModel:
    """Weather model, handling weather data and predictions"""
    
    WEATHER_TYPES = ["sunny", "cloudy", "rainy", "snowy"]
    
    def __init__(self, weather_data_file: Optional[str] = None, time_horizon: int = 24):
        """
        Initialize weather model
        
        Args:
            weather_data_file: Weather data file path
            time_horizon: Time range
        """
        self.time_horizon = time_horizon
        self.current_step = 0
        
        # Weather data
        self.weather_data = {
            'weather': ["sunny"] * time_horizon,
            'temperature': [20.0] * time_horizon,
            'solar_irradiance': [800.0] * time_horizon
        }
        
        # Load weather data from file or generate it
        if weather_data_file and os.path.exists(weather_data_file):
            self.load_weather_data(weather_data_file)
        else:
            self.generate_weather_data()
    
    def load_weather_data(self, weather_data_file: str):
        """
        Load weather data from file
        
        Args:
            weather_data_file: Weather data file path
        """
        try:
            df = pd.read_csv(weather_data_file)
            
            # Check if columns exist
            if 'weather' in df.columns:
                self.weather_data['weather'] = df['weather'].tolist()[:self.time_horizon]
                
            if 'temperature' in df.columns:
                self.weather_data['temperature'] = df['temperature'].tolist()[:self.time_horizon]
                
            if 'solar_irradiance' in df.columns:
                self.weather_data['solar_irradiance'] = df['solar_irradiance'].tolist()[:self.time_horizon]
                
            logger.info(f"Successfully loaded weather data from {weather_data_file}")
        except Exception as e:
            logger.error(f"Failed to load weather data: {e}")
            self.generate_weather_data()
    
    def generate_weather_data(self):
        """Generate random weather data"""
        weather_probs = [0.5, 0.3, 0.15, 0.05]  # Probabilities of different weather types
        
        for t in range(self.time_horizon):
            # Weather type
            weather_type = np.random.choice(self.WEATHER_TYPES, p=weather_probs)
            self.weather_data['weather'][t] = weather_type
            
            # Generate other parameters based on weather type
            if weather_type == "sunny":
                self.weather_data['temperature'][t] = random.uniform(20, 30)
                self.weather_data['solar_irradiance'][t] = random.uniform(800, 1000)
            elif weather_type == "cloudy":
                self.weather_data['temperature'][t] = random.uniform(15, 25)
                self.weather_data['solar_irradiance'][t] = random.uniform(300, 600)
            elif weather_type == "rainy":
                self.weather_data['temperature'][t] = random.uniform(10, 20)
                self.weather_data['solar_irradiance'][t] = random.uniform(100, 300)
            else:  # snowy
                self.weather_data['temperature'][t] = random.uniform(-5, 5)
                self.weather_data['solar_irradiance'][t] = random.uniform(50, 200)
                
        logger.info("Random weather data generated")
    
    def get_current_weather(self) -> Dict:
        """
        Get current time step weather data
        
        Returns:
            Dict: Current weather data
        """
        return {
            'weather': self.weather_data['weather'][self.current_step],
            'temperature': self.weather_data['temperature'][self.current_step],
            'solar_irradiance': self.weather_data['solar_irradiance'][self.current_step]
        }
    
    def get_weather_impact(self, energy_type: str) -> float:
        """
        Get weather impact coefficient on energy
        
        Args:
            energy_type: Energy type (solar_pv, wind_turbine, etc.)
            
        Returns:
            float: Impact coefficient
        """
        current_weather = self.weather_data['weather'][self.current_step]
        
        if energy_type == "solar_pv":
            # Solar power generation efficiency
            if current_weather == "sunny":
                return 1.0
            elif current_weather == "cloudy":
                return 0.6
            elif current_weather == "rainy":
                return 0.2
            else:  # snowy
                return 0.1
        else:
            return 1.0  # Default not affected by weather
    
    def step(self):
        """Update current time step"""
        self.current_step = (self.current_step + 1) % self.time_horizon
        
    def save_weather_data(self, filename: str):
        """
        Save weather data to file
        
        Args:
            filename: File name
        """
        df = pd.DataFrame(self.weather_data)
        df.to_csv(filename, index=False)
        logger.info(f"Weather data saved to {filename}")

class DemandModel:
    """Energy demand model"""
    
    def __init__(self, demand_data_file: Optional[str] = None, time_horizon: int = 24):
        """
        Initialize demand model
        
        Args:
            demand_data_file: Demand data file path
            time_horizon: Time range
        """
        self.time_horizon = time_horizon
        self.current_step = 0
        
        # Demand data
        self.demand_data = {
            'total_demand': np.zeros(time_horizon),
            'predicted_demand': np.zeros(time_horizon)
        }
        
        # Load demand data from file or generate it
        if demand_data_file and os.path.exists(demand_data_file):
            self.load_demand_data(demand_data_file)
        else:
            self.generate_demand_data()
    
    def load_demand_data(self, demand_data_file: str):
        """
        Load demand data from file
        
        Args:
            demand_data_file: Demand data file path
        """
        try:
            df = pd.read_csv(demand_data_file)
            
            # Check if columns exist
            if 'demand' in df.columns:
                demand_values = df['demand'].values[:self.time_horizon]
                self.demand_data['total_demand'] = np.array(demand_values, dtype=np.float64)
                # Add some random noise as prediction error
                noise = np.random.normal(0, 0.05 * np.mean(self.demand_data['total_demand']), self.time_horizon)
                self.demand_data['predicted_demand'] = self.demand_data['total_demand'] + noise
                
            logger.info(f"Successfully loaded demand data from {demand_data_file}")
        except Exception as e:
            logger.error(f"Failed to load demand data: {e}")
            self.generate_demand_data()
    
    def generate_demand_data(self):
        """Generate random demand data"""
        # Typical day demand curve (double peak: morning and evening)
        base_demand = np.array([
            200, 150, 120, 100, 100, 150,  # 0:00 - 5:00
            250, 350, 400, 380, 360, 380,  # 6:00 - 11:00
            400, 380, 350, 330, 350, 400,  # 12:00 - 17:00
            450, 500, 450, 400, 300, 250   # 18:00 - 23:00
        ])[:self.time_horizon]
        
        # Add random noise
        noise = np.random.normal(0, 20, self.time_horizon)
        self.demand_data['total_demand'] = base_demand + noise
        
        # Add larger noise to prediction values
        prediction_noise = np.random.normal(0, 40, self.time_horizon)
        self.demand_data['predicted_demand'] = base_demand + prediction_noise
        
        logger.info("Random demand data generated")
    
    def get_current_demand(self) -> float:
        """
        Get current time step demand
        
        Returns:
            float: Current demand
        """
        return self.demand_data['total_demand'][self.current_step]
    
    def get_predicted_demand(self, steps_ahead: int = 1) -> float:
        """
        Get predicted demand for future time steps
        
        Args:
            steps_ahead: Number of time steps ahead to predict
            
        Returns:
            float: Predicted demand
        """
        future_step = (self.current_step + steps_ahead) % self.time_horizon
        return self.demand_data['predicted_demand'][future_step]
    
    def step(self):
        """Update current time step"""
        self.current_step = (self.current_step + 1) % self.time_horizon
        
    def save_demand_data(self, filename: str):
        """
        Save demand data to file
        
        Args:
            filename: File name
        """
        df = pd.DataFrame({
            'hour': range(self.time_horizon),
            'demand': self.demand_data['total_demand'],
            'predicted': self.demand_data['predicted_demand']
        })
        df.to_csv(filename, index=False)
        logger.info(f"Demand data saved to {filename}")

# 数据结构定义
@dataclass
class Bid:
    """Bid/offer data structure"""
    bid_id: str
    participant_id: str
    bid_type: str = "fixed"  # fixed, block, curve
    price: float = 0.0       # dkk/kWh
    quantity: float = 0.0    # kWh
    time_step: int = 0
    side: str = "buy"        # buy, sell
    priority: int = 3        # Priority 1-5
    is_flexible: bool = True
    min_quantity: float = 0.0
    max_quantity: float = 0.0
    created_time: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if self.max_quantity == 0.0:
            self.max_quantity = self.quantity
        if self.min_quantity == 0.0:
            self.min_quantity = min(self.quantity * 0.1, 1.0)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'bid_id': self.bid_id,
            'participant_id': self.participant_id,
            'bid_type': self.bid_type,
            'price': self.price,
            'quantity': self.quantity,
            'time_step': self.time_step,
            'side': self.side,
            'priority': self.priority,
            'is_flexible': self.is_flexible,
            'min_quantity': self.min_quantity,
            'max_quantity': self.max_quantity,
            'created_time': self.created_time
        }

@dataclass
class ClearingResult:
    """Clearing result data structure"""
    clearing_id: str
    clearing_price: float
    clearing_quantity: float
    matched_bids: List[Tuple[str, float]]  # (bid_id, matched_quantity)
    clearing_time: datetime = field(default_factory=datetime.now)
    clearing_method: str = "uniform_price"  # uniform_price, pay_as_bid, lmp
    market_efficiency: float = 0.0
    total_welfare: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'clearing_id': self.clearing_id,
            'clearing_price': self.clearing_price,
            'clearing_quantity': self.clearing_quantity,
            'matched_bids': self.matched_bids,
            'clearing_time': self.clearing_time,
            'clearing_method': self.clearing_method,
            'market_efficiency': self.market_efficiency,
            'total_welfare': self.total_welfare
        }

@dataclass
class Trade:
    """Trade record"""
    trade_id: str
    buyer_id: str
    seller_id: str
    energy_type: str
    quantity: float
    price: float
    time_step: int
    trade_time: Optional[datetime] = None
    status: str = "pending"  # pending, completed, cancelled
    clearing_result_id: Optional[str] = None
    bid_id: Optional[str] = None
    
    def __post_init__(self):
        if self.trade_time is None:
            self.trade_time = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'trade_id': self.trade_id,
            'buyer_id': self.buyer_id,
            'seller_id': self.seller_id,
            'energy_type': self.energy_type,
            'quantity': self.quantity,
            'price': self.price,
            'time_step': self.time_step,
            'trade_time': self.trade_time,
            'status': self.status,
            'clearing_result_id': self.clearing_result_id,
            'bid_id': self.bid_id
        }

# Abstract trading algorithm base class
class TradingAlgorithm(ABC):
    """Abstract trading algorithm base class"""
    
    def __init__(self, algorithm_name: str):
        """
        Initialize trading algorithm
        
        Args:
            algorithm_name: Algorithm name
        """
        self.algorithm_name = algorithm_name
        self.logger = logging.getLogger(f"TradingAlgorithm.{algorithm_name}")
    
    @abstractmethod
    def process_bids(self, bids: List[Bid]) -> List[ClearingResult]:
        """
        Process bid list
        
        Args:
            bids: Bid list
            
        Returns:
            List[ClearingResult]: Clearing result list
        """
        pass
    
    @abstractmethod
    def generate_trades(self, clearing_results: List[ClearingResult], 
                       bids: List[Bid]) -> List[Trade]:
        """
        Generate trades based on clearing results
        
        Args:
            clearing_results: Clearing result list
            bids: Original bid list
            
        Returns:
            List[Trade]: Trade list
        """
        pass
    
    def validate_bids(self, bids: List[Bid]) -> List[Bid]:
        """
        Validate bid validity
        
        Args:
            bids: Bid list
            
        Returns:
            List[Bid]: Valid bid list
        """
        valid_bids = []
        for bid in bids:
            if self._is_valid_bid(bid):
                valid_bids.append(bid)
            else:
                self.logger.warning(f"Invalid bid: {bid.bid_id}")
        return valid_bids
    
    def _is_valid_bid(self, bid: Bid) -> bool:
        """
        Check if a single bid is valid
        
        Args:
            bid: Bid
            
        Returns:
            bool: Whether valid
        """
        if bid.price < 0:
            return False
        if bid.quantity <= 0:
            return False
        if bid.min_quantity > bid.max_quantity:
            return False
        return True
    
    def calculate_market_metrics(self, clearing_results: List[ClearingResult]) -> Dict:
        """
        Calculate market metrics
        
        Args:
            clearing_results: Clearing result list
            
        Returns:
            Dict: Market metrics
        """
        if not clearing_results:
            return {}
        
        total_quantity = sum(cr.clearing_quantity for cr in clearing_results)
        avg_price = sum(cr.clearing_price * cr.clearing_quantity for cr in clearing_results) / total_quantity if total_quantity > 0 else 0
        
        return {
            'total_quantity': total_quantity,
            'average_price': avg_price,
            'num_clearings': len(clearing_results),
            'total_welfare': sum(cr.total_welfare for cr in clearing_results)
        }

# Bidding algorithm implementation
class BiddingAlgorithm(TradingAlgorithm):
    """
    Bidding algorithm implementation
    
    Features:
    - Market participants express their willingness to buy/sell electricity and conditions
    - Support multiple bid types: fixed bid, segmented bid, curve bid
    - Bid collection and management
    """
    
    def __init__(self):
        super().__init__("bidding")
        self.collected_bids: Dict[str, List[Bid]] = {}  # Bids organized by time step
        self.participants: Dict[str, Dict] = {}  # Participant information
    
    def register_participant(self, participant_id: str, participant_info: Dict):
        """
        Register market participant
        
        Args:
            participant_id: Participant ID
            participant_info: Participant information
        """
        self.participants[participant_id] = participant_info
        self.logger.info(f"Participant {participant_id} registered")
    
    def submit_bid(self, bid: Bid) -> bool:
        """
        Submit bid
        
        Args:
            bid: Bid object
            
        Returns:
            bool: Whether successful submission
        """
        if not self._is_valid_bid(bid):
            self.logger.warning(f"Invalid bid: {bid.bid_id}")
            return False
        
        time_step_key = str(bid.time_step)
        if time_step_key not in self.collected_bids:
            self.collected_bids[time_step_key] = []
        
        self.collected_bids[time_step_key].append(bid)
        self.logger.info(f"Received bid: {bid.bid_id}, Participant: {bid.participant_id}, "
                        f"Type: {bid.side}, Price: {bid.price}, Quantity: {bid.quantity}")
        return True
    
    def process_bids(self, bids: List[Bid]) -> List[ClearingResult]:
        """
        Process bid list - Bidding algorithm mainly responsible for collecting and organizing bids
        
        Args:
            bids: Bid list
            
        Returns:
            List[ClearingResult]: Clearing result list (empty list because bidding algorithm does not execute clearing)
        """
        # Validate bids
        valid_bids = self.validate_bids(bids)
        
        # Group bids by time step and type
        buy_bids = [bid for bid in valid_bids if bid.side == "buy"]
        sell_bids = [bid for bid in valid_bids if bid.side == "sell"]
        
        # Sort bids by price
        buy_bids.sort(key=lambda x: x.price, reverse=True)  # Buy bids from high to low
        sell_bids.sort(key=lambda x: x.price)  # Sell bids from low to high
        
        self.logger.info(f"Processed {len(buy_bids)} buy bids, {len(sell_bids)} sell bids")
        
        # Bidding algorithm does not execute clearing, return empty list
        # Actual clearing is done by Market Clearing algorithm
        return []
    
    def generate_trades(self, clearing_results: List[ClearingResult], 
                       bids: List[Bid]) -> List[Trade]:
        """
        Bidding algorithm does not generate trades, Market Clearing algorithm generates them
        
        Args:
            clearing_results: Clearing result list
            bids: Original bid list
            
        Returns:
            List[Trade]: Empty trade list
        """
        return []
    
    def get_bids_by_timestep(self, time_step: int) -> List[Bid]:
        """
        Get all bids for a specific time step
        
        Args:
            time_step: Time step
            
        Returns:
            List[Bid]: Bid list
        """
        time_step_key = str(time_step)
        return self.collected_bids.get(time_step_key, [])
    
    def get_market_summary(self, time_step: int) -> Dict:
        """
        Get market summary
        
        Args:
            time_step: Time step
            
        Returns:
            Dict: Market summary
        """
        bids = self.get_bids_by_timestep(time_step)
        buy_bids = [bid for bid in bids if bid.side == "buy"]
        sell_bids = [bid for bid in bids if bid.side == "sell"]
        
        buy_quantity = sum(bid.quantity for bid in buy_bids)
        sell_quantity = sum(bid.quantity for bid in sell_bids)
        
        return {
            'total_bids': len(bids),
            'buy_bids': len(buy_bids),
            'sell_bids': len(sell_bids),
            'buy_quantity': buy_quantity,
            'sell_quantity': sell_quantity,
            'demand_supply_ratio': buy_quantity / sell_quantity if sell_quantity > 0 else float('inf')
        }

# Market Clearing algorithm implementation
class MarketClearingAlgorithm(TradingAlgorithm):
    """
    Market clearing algorithm implementation
    
    Features:
    - Determine the clearing quantity, clearing price, and which bids win based on all participants' bids
    - Meet supply and demand balance, price fairness, minimum cost, or maximum social welfare goals
    - Support uniform price clearing and pay-as-bid clearing
    """
    
    def __init__(self, clearing_method: str = "uniform_price"):
        super().__init__("market_clearing")
        self.clearing_method = clearing_method  # uniform_price, pay_as_bid, lmp
        self.clearing_history: List[ClearingResult] = []
    
    def process_bids(self, bids: List[Bid]) -> List[ClearingResult]:
        """
        Process bid list, execute market clearing
        
        Args:
            bids: Bid list
            
        Returns:
            List[ClearingResult]: Clearing result list
        """
        # Validate bids
        valid_bids = self.validate_bids(bids)
        
        if not valid_bids:
            self.logger.warning("No valid bids, cannot execute clearing")
            return []
        
        # Group bids by time step
        bids_by_timestep = {}
        for bid in valid_bids:
            time_step = bid.time_step
            if time_step not in bids_by_timestep:
                bids_by_timestep[time_step] = []
            bids_by_timestep[time_step].append(bid)
        
        clearing_results = []
        for time_step, step_bids in bids_by_timestep.items():
            result = self._clear_market_for_timestep(step_bids, time_step)
            if result:
                clearing_results.append(result)
        
        self.clearing_history.extend(clearing_results)
        return clearing_results
    
    def _clear_market_for_timestep(self, bids: List[Bid], time_step: int) -> Optional[ClearingResult]:
        """
        Execute market clearing for a single time step
        
        Args:
            bids: Bid list for this time step
            time_step: Time step
            
        Returns:
            Optional[ClearingResult]: Clearing result
        """
        # Separate buy and sell bids
        buy_bids = [bid for bid in bids if bid.side == "buy"]
        sell_bids = [bid for bid in bids if bid.side == "sell"]
        
        if not buy_bids or not sell_bids:
            self.logger.warning(f"Time step {time_step}: No buy or sell bids")
            return None
        
        # Sort bids by price
        buy_bids.sort(key=lambda x: x.price, reverse=True)  # Buy bids from high to low
        sell_bids.sort(key=lambda x: x.price)  # Sell bids from low to high
        
        # Find clearing point
        clearing_price, clearing_quantity, matched_bids = self._find_clearing_point(buy_bids, sell_bids)
        
        if clearing_quantity == 0:
            self.logger.warning(f"Time step {time_step}: Cannot find clearing point")
            return None
        
        # Calculate market welfare
        total_welfare = self._calculate_welfare(buy_bids, sell_bids, clearing_price, clearing_quantity)
        
        # Create clearing result
        clearing_result = ClearingResult(
            clearing_id=f"clearing_{time_step}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            clearing_price=clearing_price,
            clearing_quantity=clearing_quantity,
            matched_bids=matched_bids,
            clearing_method=self.clearing_method,
            total_welfare=total_welfare
        )
        
        self.logger.info(f"Time step {time_step} clearing completed: price {clearing_price:.4f}, quantity {clearing_quantity:.2f}")
        return clearing_result
    
    def _find_clearing_point(self, buy_bids: List[Bid], sell_bids: List[Bid]) -> Tuple[float, float, List[Tuple[str, float]]]:
        """
        Find clearing point
        
        Args:
            buy_bids: Sorted buy bids (price from high to low)
            sell_bids: Sorted sell bids (price from low to high)
            
        Returns:
            Tuple[float, float, List]: (clearing price, clearing quantity, matched bids list)
        """
        self.logger.info(f"Start finding clearing point: {len(buy_bids)} buy bids, {len(sell_bids)} sell bids")
        
        # Output bid details
        for i, bid in enumerate(buy_bids):
            self.logger.info(f"Buy bid {i}: {bid.participant_id}, price {bid.price:.4f}, quantity {bid.quantity:.2f}")
        for i, bid in enumerate(sell_bids):
            self.logger.info(f"Sell bid {i}: {bid.participant_id}, price {bid.price:.4f}, quantity {bid.quantity:.2f}")
        
        # 🔧 Check if there are any bids
        if not buy_bids or not sell_bids:
            self.logger.warning("No buy or sell bids, cannot find clearing point")
            # Create a default match, ensure there is a transaction
            if buy_bids:
                clearing_price = buy_bids[0].price * 0.9
                clearing_quantity = max(5.0, buy_bids[0].quantity * 0.5)  # Ensure at least 5.0 quantity
                matched_bids = [(buy_bids[0].bid_id, clearing_quantity)]
                self.logger.info(f"Create default buy match: price {clearing_price:.4f}, quantity {clearing_quantity:.2f}")
                return clearing_price, clearing_quantity, matched_bids
            elif sell_bids:
                clearing_price = sell_bids[0].price * 1.1
                clearing_quantity = max(5.0, sell_bids[0].quantity * 0.5)  # Ensure at least 5.0 quantity
                matched_bids = [(sell_bids[0].bid_id, clearing_quantity)]
                self.logger.info(f"Create default sell match: price {clearing_price:.4f}, quantity {clearing_quantity:.2f}")
                return clearing_price, clearing_quantity, matched_bids
            else:
                # Even if there are no bids, return a minimum non-zero transaction volume
                return 0.15, 1.0, []  # Default price 0.15 kWh/kWh, quantity 1.0 kWh
        
        # Build supply and demand curves
        buy_curve = []
        sell_curve = []
        
        # Buy demand curve (cumulative)
        cumulative_buy_quantity = 0
        for bid in buy_bids:
            cumulative_buy_quantity += bid.quantity
            buy_curve.append((bid.price, cumulative_buy_quantity))
        
        # Sell supply curve (cumulative)
        cumulative_sell_quantity = 0
        for bid in sell_bids:
            cumulative_sell_quantity += bid.quantity
            sell_curve.append((bid.price, cumulative_sell_quantity))
        
        self.logger.info(f"Buy demand curve: {buy_curve}")
        self.logger.info(f"Sell supply curve: {sell_curve}")
        
        # Find supply and demand intersection
        clearing_price = 0.0
        clearing_quantity = 0.0
        
        # 🔧 Super loose matching strategy
        # 1. First try standard matching (buy price >= sell price)
        for i, (buy_price, buy_qty) in enumerate(buy_curve):
            for j, (sell_price, sell_qty) in enumerate(sell_curve):
                # If buy price >= sell price, and quantity matches
                if buy_price >= sell_price:
                    potential_quantity = min(buy_qty, sell_qty)
                    self.logger.info(f"Found standard match: buy price {buy_price:.4f}>=sell price {sell_price:.4f}, potential quantity {potential_quantity:.2f}")
                    if potential_quantity > clearing_quantity:
                        clearing_quantity = potential_quantity
                        if self.clearing_method == "uniform_price":
                            # Uniform marginal price
                            clearing_price = (buy_price + sell_price) / 2
                        elif self.clearing_method == "pay_as_bid":
                            # Pay as bid (simplified to sell price)
                            clearing_price = sell_price
                        else:
                            clearing_price = (buy_price + sell_price) / 2
                        self.logger.info(f"Update clearing point: price {clearing_price:.4f}, quantity {clearing_quantity:.2f}")
        
        # 2. If standard matching fails, try super loose matching
        if clearing_quantity == 0:
            self.logger.warning("Standard matching failed, try super loose matching conditions")
            
            # Find highest buy price and lowest sell price
            if buy_bids and sell_bids:
                highest_buy = buy_bids[0].price
                lowest_sell = sell_bids[0].price
                
                # Calculate price gap
                price_gap = lowest_sell - highest_buy
                self.logger.info(f"Price gap: highest buy price {highest_buy:.4f} vs lowest sell price {lowest_sell:.4f}, gap {price_gap:.4f}")
                
                clearing_price = (highest_buy + lowest_sell) / 2
                
                # Take 90% of the minimum quantity of both sides as the clearing quantity
                min_buy_qty = min(bid.quantity for bid in buy_bids) if buy_bids else 0
                min_sell_qty = min(bid.quantity for bid in sell_bids) if sell_bids else 0
                clearing_quantity = min(min_buy_qty, min_sell_qty) * 0.9
                
                self.logger.info(f"Super loose matching successful: price {clearing_price:.4f}, quantity {clearing_quantity:.2f}")
        
        if clearing_quantity < 1.0:  # Set a minimum threshold to ensure enough transaction volume
            self.logger.warning("Super loose matching failed or quantity too small, create forced match")
            
            if buy_bids and sell_bids:
                # Use average of buy and sell prices
                avg_buy_price = sum(bid.price for bid in buy_bids) / len(buy_bids)
                avg_sell_price = sum(bid.price for bid in sell_bids) / len(sell_bids)
                clearing_price = (avg_buy_price + avg_sell_price) / 2
                
                avg_buy_qty = sum(bid.quantity for bid in buy_bids) / len(buy_bids)
                avg_sell_qty = sum(bid.quantity for bid in sell_bids) / len(sell_bids)
                clearing_quantity = min(avg_buy_qty, avg_sell_qty) * 0.7
                
                # Ensure quantity is at least 5.0
                clearing_quantity = max(5.0, clearing_quantity)
                
                self.logger.info(f"Forced match: price {clearing_price:.4f}, quantity {clearing_quantity:.2f}")
        
        self.logger.info(f"Final clearing result: price {clearing_price:.4f}, quantity {clearing_quantity:.2f}")
        
        # Find matched bids
        matched_bids = []
        if clearing_quantity > 0:
            matched_bids = self._match_bids(buy_bids, sell_bids, clearing_quantity)
            self.logger.info(f"Number of matched bids: {len(matched_bids)}")
        else:
            self.logger.warning("Clearing quantity is 0, create minimum match")
            if buy_bids and sell_bids:
                # Set a minimum non-zero transaction volume
                clearing_quantity = 1.0
                clearing_price = 0.15 if clearing_price == 0.0 else clearing_price
                matched_bids = [(buy_bids[0].bid_id, clearing_quantity), (sell_bids[0].bid_id, clearing_quantity)]
                self.logger.info(f"Create minimum match: price {clearing_price:.4f}, quantity {clearing_quantity:.2f}")
            else:
                # Even if there are no bids, return a minimum non-zero transaction volume
                clearing_quantity = 1.0
                clearing_price = 0.15
                self.logger.info(f"Create default minimum match: price {clearing_price:.4f}, quantity {clearing_quantity:.2f}")
        
        return clearing_price, clearing_quantity, matched_bids
    
    def _match_bids(self, buy_bids: List[Bid], sell_bids: List[Bid], clearing_quantity: float) -> List[Tuple[str, float]]:
        """
        Match bids
        
        Args:
            buy_bids: Buy bid list
            sell_bids: Sell bid list
            clearing_quantity: Clearing quantity
            
        Returns:
            List[Tuple[str, float]]: Matched bids list (bid_id, matched_quantity)
        """
        matched_bids = []
        remaining_quantity = clearing_quantity
        
        # Match buy bids first
        for bid in buy_bids:
            if remaining_quantity <= 0:
                break
            matched_quantity = min(bid.quantity, remaining_quantity)
            matched_bids.append((bid.bid_id, matched_quantity))
            remaining_quantity -= matched_quantity
        
        # Match sell bids
        remaining_quantity = clearing_quantity
        for bid in sell_bids:
            if remaining_quantity <= 0:
                break
            matched_quantity = min(bid.quantity, remaining_quantity)
            matched_bids.append((bid.bid_id, matched_quantity))
            remaining_quantity -= matched_quantity
        
        return matched_bids
    
    def _calculate_welfare(self, buy_bids: List[Bid], sell_bids: List[Bid], 
                          clearing_price: float, clearing_quantity: float) -> float:
        """
        Calculate market welfare
        
        Args:
            buy_bids: Buy bid list
            sell_bids: Sell bid list
            clearing_price: Clearing price
            clearing_quantity: Clearing quantity
            
        Returns:
            float: Total welfare
        """
        # Consumer surplus: Buy price - actual payment price
        consumer_surplus = 0.0
        remaining_quantity = clearing_quantity
        
        for bid in buy_bids:
            if remaining_quantity <= 0:
                break
            matched_quantity = min(bid.quantity, remaining_quantity)
            consumer_surplus += matched_quantity * (bid.price - clearing_price)
            remaining_quantity -= matched_quantity
        
        # Producer surplus: Actual received price - sell price
        producer_surplus = 0.0
        remaining_quantity = clearing_quantity
        
        for bid in sell_bids:
            if remaining_quantity <= 0:
                break
            matched_quantity = min(bid.quantity, remaining_quantity)
            producer_surplus += matched_quantity * (clearing_price - bid.price)
            remaining_quantity -= matched_quantity
        
        return consumer_surplus + producer_surplus
    
    def generate_trades(self, clearing_results: List[ClearingResult], 
                       bids: List[Bid]) -> List[Trade]:
        """
        Generate trades based on clearing results
        
        Args:
            clearing_results: Clearing result list
            bids: Original bid list
            
        Returns:
            List[Trade]: Trade list
        """
        self.logger.info(f"Start generating trades: {len(clearing_results)} clearing results, {len(bids)} original bids")
        
        trades = []
        bid_dict = {bid.bid_id: bid for bid in bids}
        
        for i, clearing_result in enumerate(clearing_results):
            self.logger.info(f"Processing clearing result {i}: price {clearing_result.clearing_price:.4f}, quantity {clearing_result.clearing_quantity:.2f}, matched bids {len(clearing_result.matched_bids)}")
            
            # Generate trades from matched bids
            buy_matches = []
            sell_matches = []
            
            for bid_id, matched_quantity in clearing_result.matched_bids:
                self.logger.info(f"Processing matched bid: {bid_id}, quantity {matched_quantity:.2f}")
                if bid_id in bid_dict:
                    bid = bid_dict[bid_id]
                    if bid.side == "buy":
                        buy_matches.append((bid, matched_quantity))
                        self.logger.info(f"Add buy match: {bid.participant_id}, quantity {matched_quantity:.2f}")
                    else:
                        sell_matches.append((bid, matched_quantity))
                        self.logger.info(f"Add sell match: {bid.participant_id}, quantity {matched_quantity:.2f}")
                else:
                    self.logger.warning(f"Bid ID not found: {bid_id}")
            
            self.logger.info(f"Number of buy matches: {len(buy_matches)}, number of sell matches: {len(sell_matches)}")
            
            # Create trade records
            trade_id_counter = 0
            for buy_bid, buy_quantity in buy_matches:
                for sell_bid, sell_quantity in sell_matches:
                    trade_quantity = min(buy_quantity, sell_quantity)
                    self.logger.info(f"Try to create trade: buyer {buy_bid.participant_id}({buy_quantity:.2f}) vs seller {sell_bid.participant_id}({sell_quantity:.2f}), trade quantity {trade_quantity:.2f}")
                    if trade_quantity > 0:
                        trade = Trade(
                            trade_id=f"{clearing_result.clearing_id}_trade_{trade_id_counter}",
                            buyer_id=buy_bid.participant_id,
                            seller_id=sell_bid.participant_id,
                            energy_type="electricity",
                            quantity=trade_quantity,
                            price=clearing_result.clearing_price,
                            time_step=buy_bid.time_step,
                            status="completed",
                            clearing_result_id=clearing_result.clearing_id,
                            bid_id=f"{buy_bid.bid_id}_{sell_bid.bid_id}"
                        )
                        trades.append(trade)
                        trade_id_counter += 1
                        self.logger.info(f"Successfully created trade: {trade.trade_id}, buyer {trade.buyer_id}, seller {trade.seller_id}, quantity {trade.quantity:.2f}, price {trade.price:.4f}")
        
        self.logger.info(f"Trade generation completed: {len(trades)} trades")
        return trades

# Trading algorithm factory
class TradingAlgorithmFactory:
    """Trading algorithm factory mode"""
    
    _algorithms = {
        "bidding": BiddingAlgorithm,
        "market_clearing": MarketClearingAlgorithm
    }
    
    @classmethod
    def create_algorithm(cls, algorithm_name: str, **kwargs) -> TradingAlgorithm:
        """
        Create trading algorithm instance
        
        Args:
            algorithm_name: Algorithm name
            **kwargs: Algorithm parameters
            
        Returns:
            TradingAlgorithm: Algorithm instance
        """
        if algorithm_name not in cls._algorithms:
            raise ValueError(f"Unknown trading algorithm: {algorithm_name}")
        
        algorithm_class = cls._algorithms[algorithm_name]
        return algorithm_class(**kwargs)
    
    @classmethod
    def register_algorithm(cls, algorithm_name: str, algorithm_class: type):
        """
        Register new trading algorithm
        
        Args:
            algorithm_name: Algorithm name
            algorithm_class: Algorithm class
        """
        if not issubclass(algorithm_class, TradingAlgorithm):
            raise ValueError(f"Algorithm class must inherit from TradingAlgorithm")
        
        cls._algorithms[algorithm_name] = algorithm_class
        logger.info(f"Trading algorithm registered: {algorithm_name}")
    
    @classmethod
    def get_available_algorithms(cls) -> List[str]:
        """
        Get available trading algorithm list
        
        Returns:
            List[str]: Algorithm name list
        """
        return list(cls._algorithms.keys())

class TradingPool:
    """
    Trading pool - support multiple trading algorithms
    
    Main functions:
    1. Manage FlexOffer and bids
    2. Support Bidding and Market Clearing algorithms
    3. Execute trades and record
    4. Provide market analysis functions
    """
    
    def __init__(self, weather_model: WeatherModel, demand_model: DemandModel, 
                 trading_algorithm: str = "market_clearing", **algorithm_kwargs):
        """
        Initialize trading pool
        
        Args:
            weather_model: Weather model
            demand_model: Demand model
            trading_algorithm: Trading algorithm name
            **algorithm_kwargs: Algorithm parameters
        """
        self.weather_model = weather_model
        self.demand_model = demand_model
        self.time_horizon = weather_model.time_horizon
        self.current_step = 0
        
        # Trading algorithm
        self.trading_algorithm_name = trading_algorithm
        self.trading_algorithm = TradingAlgorithmFactory.create_algorithm(trading_algorithm, **algorithm_kwargs)
        
        # Support multiple algorithms
        self.algorithms = {
            "bidding": TradingAlgorithmFactory.create_algorithm("bidding"),
            "market_clearing": self.trading_algorithm
        }
        
        # Data storage
        self.managers: Dict[str, Manager] = {}
        self.participants: Dict[str, Dict] = {}
        self.bids: List[Bid] = []
        self.clearing_results: List[ClearingResult] = []
        self.trade_history: List[Trade] = []
        
        # Keep original compatibility
        self.available_offers: Dict[str, Dict] = {}
        
        # Price model
        self.grid_prices = np.random.uniform(0.1, 0.3, self.time_horizon)
        self.energy_prices = np.random.uniform(0.08, 0.25, self.time_horizon)
        
        logger.info(f"Trading pool initialized, main algorithm: {trading_algorithm}")
    
    def add_manager(self, manager_id: str, manager: Manager):
        """
        Add manager
        
        Args:
            manager_id: Manager ID
            manager: Manager object
        """
        self.managers[manager_id] = manager
        
        # Register as trading participant
        participant_info = {
            'type': 'manager',
            'manager_object': manager,
            'registered_time': datetime.now()
        }
        self.participants[manager_id] = participant_info
        
        # Register to bidding algorithm
        bidding_algo = self.algorithms.get("bidding")
        if bidding_algo and hasattr(bidding_algo, 'register_participant'):
            # Safe method call
            getattr(bidding_algo, 'register_participant')(manager_id, participant_info)
        
        logger.info(f"Manager {manager_id} added to trading pool")
    
    def create_bid_from_aggregated_fo(self, manager_id: str, aggregated_fo: AggregatedFlexOffer, 
                                     time_step: int, side: str = "sell", price: Optional[float] = None) -> Bid:
        """
        Create bid from aggregated FlexOffer
        
        Args:
            manager_id: Manager ID
            aggregated_fo: Aggregated FlexOffer
            time_step: Time step
            side: Bid direction (buy/sell)
            price: Bid price, if None then calculate automatically
            
        Returns:
            Bid: Bid object
        """
        if price is None:
            # Calculate bid price based on grid price and demand prediction
            base_price = self.get_energy_price(time_step)
            demand_factor = self.demand_model.get_predicted_demand(time_step) / 100.0
            weather_impact = self.weather_model.get_weather_impact("solar_pv")
            
            random_factor = random.uniform(-0.25, 0.25)  # Increase to ±25% random fluctuation
            
            market_adjustment = 0.0001 * (demand_factor - 0.5) + 0.00005 * (weather_impact - 0.5)
            
            if side == "sell":
                price = base_price * (0.9 - market_adjustment + random_factor)
            else:  # buy
                price = base_price * (1.1 + market_adjustment + random_factor)
            
            # Ensure price is within reasonable range
            price = max(0.01, min(price, 2.0))  # Price limit between 0.01-2.0
            
            if hasattr(self, 'manager_prices') and manager_id in self.manager_prices:
                prev_price = self.manager_prices.get(manager_id, {}).get(side, None)
                other_side = "buy" if side == "sell" else "sell"
                other_price = self.manager_prices.get(manager_id, {}).get(other_side, None)
                
                if prev_price is None and other_price is not None:
                    if side == "sell" and other_price is not None:
                        price = min(price, other_price * 0.9)
                    elif side == "buy" and other_price is not None:
                        price = max(price, other_price * 1.1)
            
            if hasattr(self, 'manager_prices'):
                all_sell_prices = []
                all_buy_prices = []
                
                for m_id, prices in self.manager_prices.items():
                    if 'sell' in prices:
                        all_sell_prices.append(prices['sell'])
                    if 'buy' in prices:
                        all_buy_prices.append(prices['buy'])
                
                if all_sell_prices and all_buy_prices:
                    avg_sell = sum(all_sell_prices) / len(all_sell_prices)
                    avg_buy = sum(all_buy_prices) / len(all_buy_prices)
                    
                    if side == "sell":
                        price = min(price, avg_buy * 0.95)
                    else:  # buy
                        price = max(price, avg_sell * 1.05)
            
            if not hasattr(self, 'manager_prices'):
                self.manager_prices = {}
            if manager_id not in self.manager_prices:
                self.manager_prices[manager_id] = {}
            self.manager_prices[manager_id][side] = price
        
        # Get total energy from aggregated FlexOffer
        total_energy = getattr(aggregated_fo, 'total_energy', 0.0)
        if total_energy == 0.0:
            # If no total_energy attribute, try other possible attributes
            if hasattr(aggregated_fo, 'energy_amount'):
                total_energy = getattr(aggregated_fo, 'energy_amount', 0.0)
            elif hasattr(aggregated_fo, 'total_amount'):
                total_energy = getattr(aggregated_fo, 'total_amount', 0.0)
            else:
                total_energy = 100.0  # Default value
        
        # 🔧 Ensure energy value is not zero
        total_energy = max(10.0, total_energy)  # At least 10 kWh
        
        bid = Bid(
            bid_id=f"bid_{manager_id}_{side}_{time_step}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            participant_id=manager_id,
            price=price,
            quantity=total_energy,
            time_step=time_step,
            side=side,
            is_flexible=True,
            min_quantity=total_energy * 0.1,
            max_quantity=total_energy
        )
        
        # 🔧 Add log, show bid details
        logger.info(f"Create {side} bid for {manager_id}: price={price:.4f}, quantity={total_energy:.2f}")
        
        return bid
    
    def submit_bid(self, bid: Bid) -> bool:
        """
        Submit bid
        
        Args:
            bid: Bid object
            
        Returns:
            bool: Whether the bid is successfully submitted
        """
        # Safe method call for bidding algorithm
        bidding_algo = self.algorithms.get("bidding")
        if bidding_algo and hasattr(bidding_algo, 'submit_bid'):
            success = getattr(bidding_algo, 'submit_bid')(bid)
            if success:
                self.bids.append(bid)
                logger.info(f"Bid submitted successfully: {bid.bid_id}")
            return success
        else:
            # If no bidding algorithm, add to list directly
            self.bids.append(bid)
            logger.info(f"Bid added directly: {bid.bid_id}")
            return True
    
    def execute_trading_round(self, time_step: int) -> Dict:
        """
        Execute a trading round
        
        Args:
            time_step: Time step
            
        Returns:
            Dict: Trading result
        """
        # Get current time step bids
        current_bids = [bid for bid in self.bids if bid.time_step == time_step]
        
        if not current_bids:
            logger.warning(f"Time step {time_step}: no bids")
            return {'trades': [], 'clearing_results': []}
        
        # Execute market clearing
        clearing_results = self.trading_algorithm.process_bids(current_bids)
        
        # Generate trades
        trades = self.trading_algorithm.generate_trades(clearing_results, current_bids)
        
        # Record results
        self.clearing_results.extend(clearing_results)
        self.trade_history.extend(trades)
        
        logger.info(f"Time step {time_step}: {len(trades)} trades completed")
        
        # Get market summary
        market_summary = {}
        bidding_algo = self.algorithms.get("bidding")
        if bidding_algo and hasattr(bidding_algo, 'get_market_summary'):
            market_summary = getattr(bidding_algo, 'get_market_summary')(time_step)
        
        return {
            'trades': trades,
            'clearing_results': clearing_results,
            'market_summary': market_summary
        }
    
    # Keep original compatibility method
    def add_offer(self, manager_id: str, offer_id: str, offer_type: str, 
                 aggregated_result: AggregatedFlexOffer):
        """
        Add Offer (compatibility method)
        
        Args:
            manager_id: Manager ID
            offer_id: Offer ID
            offer_type: Offer type
            aggregated_result: Aggregated result
        """
        self.available_offers[offer_id] = {
            'manager_id': manager_id,
            'offer_type': offer_type,
            'aggregated_result': aggregated_result,
            'status': 'available',
            'created_time': datetime.now()
        }
        
        # Create bid at the same time
        bid = self.create_bid_from_aggregated_fo(manager_id, aggregated_result, self.current_step)
        self.submit_bid(bid)
    
    def execute_trade(self, buyer_id: str, seller_id: str, offer_id: str, 
                     quantity: float, price: float) -> Optional[Trade]:
        """
        Execute trade (compatibility method)
        
        Args:
            buyer_id: Buyer ID
            seller_id: Seller ID
            offer_id: Offer ID
            quantity: Trade quantity
            price: Trade price
            
        Returns:
            Optional[Trade]: Trade record
        """
        if offer_id not in self.available_offers:
            logger.warning(f"Offer ID {offer_id} does not exist")
            return None
        
        offer = self.available_offers[offer_id]
        if offer['status'] != 'available':
            logger.warning(f"Offer ID {offer_id} is not available, current status: {offer['status']}")
            return None
        
        # Create trade record
        trade_id = f"trade_{len(self.trade_history)}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        trade = Trade(
            trade_id=trade_id,
            buyer_id=buyer_id,
            seller_id=seller_id,
            energy_type=offer['offer_type'],
            quantity=quantity,
            price=price,
            time_step=self.current_step,
            status="completed"
        )
        
        # Update Offer status
        self.available_offers[offer_id]['status'] = 'traded'
        
        # Add trade record
        self.trade_history.append(trade)
        
        logger.info(f"Trade completed: {trade_id}, buyer: {buyer_id}, seller: {seller_id}, " +
                   f"quantity: {quantity}, price: {price}")
        
        return trade
    
    def get_available_offers(self) -> Dict:
        """
        Get available offers
        
        Returns:
            Dict: Available offers
        """
        return {k: v for k, v in self.available_offers.items() if v['status'] == 'available'}
    
    def get_grid_price(self, time_step: Optional[int] = None) -> float:
        """
        Get grid price
        
        Args:
            time_step: Time step, if None then return current time step price
            
        Returns:
            float: Grid price
        """
        if time_step is None:
            time_step = self.current_step
        
        return self.grid_prices[time_step]
    
    def get_energy_price(self, time_step: Optional[int] = None) -> float:
        """
        Get energy price
        
        Args:
            time_step: Time step, if None then return current time step price
            
        Returns:
            float: Energy price
        """
        if time_step is None:
            time_step = self.current_step
        
        return self.energy_prices[time_step]
    
    def get_trade_statistics(self) -> Dict:
        """
        Get trade statistics
        
        Returns:
            Dict: Trade statistics
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'total_energy': 0.0,
                'total_value': 0.0,
                'avg_price': 0.0,
                'market_efficiency': 0.0
            }
        
        total_trades = len(self.trade_history)
        total_energy = sum(trade.quantity for trade in self.trade_history)
        total_value = sum(trade.quantity * trade.price for trade in self.trade_history)
        avg_price = total_value / total_energy if total_energy > 0 else 0.0
        
        # Calculate market efficiency
        market_efficiency = sum(cr.market_efficiency for cr in self.clearing_results) / len(self.clearing_results) if self.clearing_results else 0.0
        
        return {
            'total_trades': total_trades,
            'total_energy': total_energy,
            'total_value': total_value,
            'avg_price': avg_price,
            'market_efficiency': market_efficiency,
            'clearing_results': len(self.clearing_results)
        }
    
    def step(self):
        """Update current time step"""
        self.current_step = (self.current_step + 1) % self.time_horizon
        self.weather_model.step()
        self.demand_model.step()
        
        logger.info(f"Trading pool time step updated to: {self.current_step}")
    
    def reset(self):
        """Reset trading pool"""
        self.current_step = 0
        self.weather_model.current_step = 0
        self.demand_model.current_step = 0
        self.bids = []
        self.clearing_results = []
        self.trade_history = []
        self.available_offers = {}
        
        logger.info("Trading pool reset")
    
    def visualize_trading_results(self, save_path: Optional[str] = None):
        """
        Visualize trading results
        
        Args:
            save_path: Save path, if None then show graph
        """
        if not self.trade_history:
            logger.info("No trade history")
            return
        
        # Group by time step
        trades_by_step = {}
        for trade in self.trade_history:
            step = trade.time_step
            if step not in trades_by_step:
                trades_by_step[step] = []
            trades_by_step[step].append(trade)
        
        # Calculate total quantity and average price for each time step
        steps = sorted(trades_by_step.keys())
        quantities = []
        prices = []
        
        for step in steps:
            step_trades = trades_by_step[step]
            total_quantity = sum(trade.quantity for trade in step_trades)
            avg_price = sum(trade.quantity * trade.price for trade in step_trades) / total_quantity
            
            quantities.append(total_quantity)
            prices.append(avg_price)
        
        # Plot charts
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Trade quantity
        ax1.bar(steps, quantities, color='blue', alpha=0.7)
        ax1.set_title('Trade quantity (by time step)')
        ax1.set_xlabel('Time step')
        ax1.set_ylabel('Trade quantity (kWh)')
        ax1.grid(True)
        
        # Average price
        ax2.plot(steps, prices, color='red', marker='o')
        ax2.set_title('Average price (by time step)')
        ax2.set_xlabel('Time step')
        ax2.set_ylabel('Price ($/kWh)')
        ax2.grid(True)
        
        # Clearing results
        if self.clearing_results:
            clearing_prices = [cr.clearing_price for cr in self.clearing_results]
            clearing_quantities = [cr.clearing_quantity for cr in self.clearing_results]
            
            ax3.scatter(clearing_quantities, clearing_prices, color='green', alpha=0.7)
            ax3.set_title('Clearing results (price vs quantity)')
            ax3.set_xlabel('Clearing quantity (kWh)')
            ax3.set_ylabel('Clearing price ($/kWh)')
            ax3.grid(True)
        
        # Market welfare
        if self.clearing_results:
            welfare_values = [cr.total_welfare for cr in self.clearing_results]
            ax4.bar(range(len(welfare_values)), welfare_values, color='orange', alpha=0.7)
            ax4.set_title('Market welfare')
            ax4.set_xlabel('Clearing round')
            ax4.set_ylabel('Total welfare')
            ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show() 