# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from typing import Dict, List, Optional, Tuple
import numpy as np
import cupy as cp
from scipy import stats
import pandas as pd
import logging
import torch
from torch import nn
from torch.cuda.amp import autocast
import traceback
import yfinance as yf  # Yahoo Finance cho dữ liệu chứng khoán
from hmmlearn.hmm import GaussianHMM  # HMM cho trạng thái tiêu dùng

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', 
                    handlers=[logging.FileHandler("votranh_abyss_micro.log"), logging.StreamHandler()])

class QuantumResonanceLayer(nn.Module):
    def __init__(self, d_model=8192):  # Tăng chiều sâu để xử lý chính sách vĩ mô và mạng phản xạ
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.phase_shift = nn.Parameter(torch.randn(d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        try:
            with autocast():
                x = self.linear(x) + torch.sin(self.phase_shift) * x
                return self.norm(torch.tanh(x))
        except Exception as e:
            logging.error(f"Error in QuantumResonanceLayer: {e}")
            return x

class MicroEconomicPredictor(nn.Module):
    def __init__(self, input_dim=72, hidden_dim=8192, num_layers=24):  # Tăng input và tầng cho chính sách
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=5, batch_first=True)  # Tăng layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=8, batch_first=True)  # Tăng layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=128, dim_feedforward=32768, batch_first=True), 
            num_layers=num_layers)  # Tăng nhead và dim_feedforward
        self.quantum_resonance = QuantumResonanceLayer(hidden_dim)
        self.fc_short = nn.Linear(hidden_dim, 1)  # Short-term (months)
        self.fc_mid = nn.Linear(hidden_dim, 1)    # Mid-term (10 years)
        self.dropout = nn.Dropout(0.01)  # Giảm dropout để giữ thông tin chính sách

    def forward(self, x):
        try:
            with autocast():
                _, h_gru = self.gru(x)
                _, (h_lstm, _) = self.lstm(x)
                x = self.transformer(h_lstm)
                x = self.quantum_resonance(x)
                x = self.dropout(x)
                short_pred = self.fc_short(x[-1])
                mid_pred = self.fc_mid(x[-1])
            return short_pred, mid_pred
        except Exception as e:
            logging.error(f"Error in MicroEconomicPredictor: {e}")
            return torch.zeros(1).to(x.device), torch.zeros(1).to(x.device)

class HyperAgent:
    def __init__(self, id: str, nation: str, role: str, wealth: float, innovation: float, trade_flow: float, resilience: float):
        self.id = id
        self.nation = nation
        self.role = role
        self.wealth = max(0, wealth)
        self.innovation = max(0, min(1, innovation))
        self.trade_flow = max(0, trade_flow)
        self.resilience = max(0, min(1, resilience))
        self.psychology_state = "Hope"
        self.fear_index = 0.0
        self.greed_index = 0.0
        self.complacency_index = 0.0
        self.hope_index = 0.0
        self.credit_growth = 0.0
        self.real_income = wealth
        self.market_momentum = 0.0
        self.real_income_history = [wealth] * 30  # Lịch sử 30 ngày
        self.consumption_state = "normal"  # Trạng thái tiêu dùng: low, normal, high
        self.hmm_model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=200)  # Tăng iter cho chính xác
        self.policy_response = {"tax_reduction": 0.0, "interest_rate": 0.0, "subsidy": 0.0}  # Phản ứng với chính sách

    def update_psychology(self, global_context: Dict[str, float], nation_space: Dict[str, float], 
                          volatility_history: List[float], gdp_history: List[float], sentiment: float, 
                          market_momentum: float) -> None:
        """Update psychological state based on real market data"""


        try:
            volatility_30d = np.std(volatility_history[-30:]) if len(volatility_history) >= 30 else 0.0
            volatility_year = np.mean(volatility_history[-365:]) if len(volatility_history) >= 365 else 1e-6
            gdp_drop_3m = (gdp_history[-90] - gdp_history[-1]) / gdp_history[-90] if len(gdp_history) >= 90 and gdp_history[-90] != 0 else 0.0
            self.fear_index = (volatility_30d / volatility_year) + max(0, gdp_drop_3m) - market_momentum * 0.15

            gdp_growth_6m = (gdp_history[-1] - gdp_history[-180]) / gdp_history[-180] if len(gdp_history) >= 180 and gdp_history[-180] != 0 else 0.0
            trade_flow_growth = (nation_space["trade"] - nation_space.get("trade_prev", nation_space["trade"])) / nation_space.get("trade_prev", 1e-6)
            self.greed_index = max(0, gdp_growth_6m) * max(0, trade_flow_growth) + market_momentum * 0.25

            low_volatility_months = sum(1 for v in volatility_history[-180:] if v < 0.05) / 30 if len(volatility_history) >= 180 else 0
            self.complacency_index = low_volatility_months * max(0, self.credit_growth) * (1 - market_momentum * 0.2)

            gdp_recovery = max(0, gdp_history[-1] - min(gdp_history[-180:])) / abs(min(gdp_history[-180:]) + 1e-6) if len(gdp_history) >= 180 else 0
            sentiment_shift = max(0, sentiment - nation_space.get("sentiment_prev", 0.0))
            self.hope_index = gdp_recovery * sentiment_shift + market_momentum * 0.3

            indices = {"Fear": self.fear_index, "Greed": self.greed_index, "Complacency": self.complacency_index, "Hope": self.hope_index}
            self.psychology_state = max(indices, key=indices.get)
            nation_space["trade_prev"] = nation_space["trade"]
            nation_space["sentiment_prev"] = sentiment
            self.market_momentum = market_momentum
        except Exception as e:
            logging.error(f"Error in update_psychology for {self.id}: {e}")
            self.psychology_state = "Hope"

    def update_real_income(self, inflation: float, interest_rate: float, tax_rate: float):
        """Update real income based on macroeconomic policies""" 
        try:
            self.real_income = self.wealth / (1 + inflation + interest_rate + tax_rate * (1 - self.policy_response["tax_reduction"]))
            self.real_income_history.append(self.real_income)
            self.real_income_history = self.real_income_history[-30:]
            real_income_drop = (self.real_income_history[0] - self.real_income) / (self.real_income_history[0] + 1e-6) if len(self.real_income_history) >= 30 else 0.0
            real_income_rise = (self.real_income - self.real_income_history[0]) / (self.real_income_history[0] + 1e-6) if len(self.real_income_history) >= 30 else 0.0
            
            if real_income_drop > 0.15:
                self.fear_index += 0.25
                self.wealth += self.wealth * 0.2  # Tăng tiết kiệm
            elif real_income_rise > 0.10:
                self.hope_index += 0.20
        except Exception as e:
            logging.error(f"Error in update_real_income for {self.id}: {e}")

    def update_consumption_state(self):
        """Predict consumption state using HMM"""
        try:
            if len(self.real_income_history) >= 30:
                X = np.array(self.real_income_history).reshape(-1, 1)
                indices = {"Fear": self.fear_index, "Greed": self.greed_index, "Complacency": self.complacency_index, "Hope": self.hope_index}
                psych_value = max(indices.values())
                X = np.hstack([X, np.full((X.shape[0], 1), psych_value)])
                self.hmm_model.fit(X)
                state = self.hmm_model.predict(X)[-1]
                self.consumption_state = ["low", "normal", "high"][state]
        except Exception as e:
            logging.error(f"Error in update_consumption_state for {self.id}: {e}")
            self.consumption_state = "normal"

    def apply_policy_effects(self, policy: Dict[str, float]):
        """Apply macro policy impact on agents"""
        try:
            action = policy.get("action")
            param = policy.get("param", 0.0)
            target = policy.get("target", "all")

            if self.role in [target, "all"]:
                if action == "reduce_tax":
                    self.policy_response["tax_reduction"] = param
                    self.real_income *= 1 + 0.15 * param
                    self.hope_index += 0.25
                    self.fear_index -= 0.1
                elif action == "raise_interest_rate":
                    self.credit_growth *= 1 - 0.25 * param
                    self.greed_index -= 0.15
                    self.resilience += 0.05
                elif action == "increase_subsidy":
                    self.wealth += self.wealth * 0.1 * param
                    self.hope_index += 0.25
                elif action == "tighten_credit":
                    self.credit_growth *= 1 - 0.3 * param
                    self.resilience += 0.1
                    self.wealth *= 0.8  # Giảm luxury spending
                elif action == "export_incentive":
                    self.trade_flow += self.trade_flow * 0.2 * param
                    self.market_momentum += 0.1
                elif action == "infrastructure_investment":
                    self.resilience += 0.08 * param
                    self.innovation += 0.1
                elif action == "education_reform":
                    self.innovation += 0.15 * param
                    self.resilience += 0.08
                elif action == "currency_devaluation":
                    self.trade_flow += self.trade_flow * 0.25 * param
        except Exception as e:
            logging.error(f"Error in apply_policy_effects for {self.id}: {e}")

    def interact(self, agents: List['HyperAgent'], global_context: Dict[str, float], nation_space: Dict[str, float], 
                 volatility_history: List[float], gdp_history: List[float], market_data: Dict[str, float], 
                 policy: Optional[Dict[str, float]] = None) -> None:
        """Interact with market data and macroeconomic policies"""
        try:
            sentiment = global_context["market_sentiment"]
            market_momentum = market_data.get("market_momentum", 0.0)
            self.update_psychology(global_context, nation_space, volatility_history, gdp_history, sentiment, market_momentum)
            self.update_real_income(nation_space["inflation"], global_context.get("interest_rate", 0.02), global_context.get("tax_rate", 0.1))
            if policy:
                self.apply_policy_effects(policy)
            self.update_consumption_state()
            pmi = global_context.get("pmi", 0.5)

            if self.role == "citizen":
                trade_gain = sum(a.trade_flow * a.wealth for a in agents if a.role == "business" and a.nation == self.nation) * 0.004
                self.wealth += trade_gain * global_context.get("global_trade", 1.0)
                self.resilience -= global_context.get("geopolitical_tension", 0.0) * 0.006 * (1 - nation_space.get("institutions", 0.5))
                real_income_drop = (self.real_income_history[0] - self.real_income) / (self.real_income_history[0] + 1e-6) if len(self.real_income_history) >= 30 else 0.0
                if self.psychology_state == "Fear" or pmi < 0.3:
                    self.wealth *= (1 - self.fear_index) ** 3 * (1 - real_income_drop * 0.5)
                    self.resilience += 0.25
                elif self.psychology_state == "Greed" or pmi > 0.7:
                    self.wealth *= 1.35
                    self.trade_flow += math.log(self.wealth + 1) * self.greed_index * 2 * (1 + market_momentum * 0.3)
                elif self.psychology_state == "Complacency":
                    self.wealth *= 1.3
                    self.credit_growth += self.wealth * self.complacency_index * 1.0 * (1 + self.credit_growth * 0.6)
                elif self.psychology_state == "Hope":
                    self.wealth *= 1.2 + market_data.get("Stock_Volatility", 0.0) * 0.2

            elif self.role == "business":
                tax = sum(a.wealth for a in agents if a.role == "government" and a.nation == self.nation) * 0.035
                self.wealth -= tax
                self.innovation += global_context.get("global_growth", 0.03) * 0.08 if nation_space["market_sentiment"] > 0 else -0.006
                self.trade_flow += self.innovation * nation_space.get("trade", 1.0) * 0.18
                self.resilience -= sum(a.resilience for a in agents if a.nation != self.nation) * 0.0025
                if self.psychology_state == "Fear" or pmi < 0.3:
                    self.trade_flow *= 0.65
                    self.wealth += 0.25 * tax
                elif self.psychology_state == "Greed" or pmi > 0.7:
                    self.trade_flow *= 1.4 + market_momentum * 0.18
                    self.innovation -= 0.07
                elif self.psychology_state == "Complacency":
                    self.credit_growth += 0.3
                    self.resilience -= 0.1
                elif self.psychology_state == "Hope":
                    self.innovation += 0.28 + market_momentum * 0.07

            elif self.role == "government":
                revenue = sum(a.wealth * 0.045 for a in agents if a.nation == self.nation)
                conflict_cost = sum(a.resilience for a in agents if a.nation != self.nation) * 0.006
                self.wealth += revenue - conflict_cost * global_context["geopolitical_tension"]
                self.trade_flow -= global_context["climate_impact"] * 0.025
                self.innovation += 0.035 if nation_space.get("cultural_economic_factor", 0.8) > 0.7 else -0.005
                if self.psychology_state == "Fear" or pmi < 0.3:
                    self.wealth *= 1.3
                    self.trade_flow -= 0.07 * tax
                elif self.psychology_state == "Greed" or pmi > 0.7:
                    self.trade_flow *= 0.75
                elif self.psychology_state == "Complacency":
                    self.wealth *= 0.8
                    self.trade_flow += 0.25 * revenue
                elif self.psychology_state == "Hope":
                    self.trade_flow += 0.3 * revenue + market_momentum * 0.07

            self.wealth = max(0, self.wealth)
            self.innovation = max(0, min(1, self.innovation))
            self.trade_flow = max(0, self.trade_flow)
            self.resilience = max(0, min(1, self.resilience))
            self.credit_growth = max(0, min(1, self.credit_growth))
        except Exception as e:
            logging.error(f"Agent interaction error for {self.id}: {e}")

# Real-time market data retrieval function via API
def fetch_market_data(nation_name: str) -> Dict[str, float]:
    try:
        ticker = "^VNINDEX.VN" if nation_name == "Vietnam" else "^GSPC"
        data = yf.download(ticker, period="7d", interval="1m")
        volatility_7d = np.std(data["Close"].pct_change()[-10080:]) if len(data) >= 10080 else 0.0
        growth_7d = (data["Close"][-1] - data["Close"][-10080]) / data["Close"][-10080] if len(data) >= 10080 else 0.0
        market_momentum = growth_7d / (volatility_7d + 1e-6)
        return {
            "market_momentum": market_momentum,
            "Stock_Volatility": volatility_7d,
            "Gold_Price": 1800.0 + random.uniform(-50, 50),
            "Oil_Price": 80.0 + random.uniform(-5, 5),
            "Currency_Rate": 23000.0 + random.uniform(-500, 500) if nation_name == "Vietnam" else 1.0
        }
    except Exception as e:
        logging.error(f"Error in fetch_market_data: {e}")
        return {"market_momentum": 0.0, "Stock_Volatility": 0.0, "Gold_Price": 1800.0, "Oil_Price": 80.0, "Currency_Rate": 23000.0}

# Function to calculate Market Psychology Momentum (PMI)
def calculate_pmi(volatility_history: List[float], gdp_history: List[float], trade_flow_history: List[float], 
                  sentiment: float, geopolitical_tension: float, credit_growth: float) -> float:
    try:
        volatility_90d = np.mean(volatility_history[-90:]) if len(volatility_history) >= 90 else 1e-6
        volatility_inverse = 1 / (volatility_90d + 1e-6)
        sideways_days = sum(1 for i in range(-90, -1) if abs((gdp_history[i] - gdp_history[i-1]) / gdp_history[i-1]) < 0.02 
                            and abs((trade_flow_history[i] - trade_flow_history[i-1]) / trade_flow_history[i-1]) < 0.02 
                            if i >= -len(gdp_history) and i >= -len(trade_flow_history)) if len(gdp_history) >= 90 else 0
        pmi = (volatility_inverse * sideways_days * credit_growth + sentiment - geopolitical_tension) / 5.0
        return max(0, min(1, pmi))
    except Exception as e:
        logging.error(f"Error in calculate_pmi: {e}")
        return 0.5
    # Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from typing import Dict, List, Optional, Tuple
import numpy as np
import cupy as cp
from scipy import stats, fft
import pandas as pd
from datetime import datetime
import networkx as nx
import hashlib
import logging
import json
import requests
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from multiprocessing import Pool
import traceback
import tweepy
from filterpy.kalman import ExtendedKalmanFilter
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from collections import deque
from part1 import QuantumResonanceLayer, MicroEconomicPredictor, HyperAgent, fetch_market_data, calculate_pmi

#Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', 
                    handlers=[logging.FileHandler("votranh_abyss_micro.log"), logging.StreamHandler()])

class VoTranhAbyssCoreMicro:
    def __init__(self, nations: List[Dict[str, Dict]], t: float = 0.0, 
                 initial_omega: float = 40.0, k_constant: float = 2.0,  # Tăng omega và k cho độ nhạy cao hơn
                 transcendence_key: str = "Cauchyab12", resonance_factor: float = 3.0,  # Tăng resonance
                 deterministic: bool = False, api_keys: Dict[str, str] = {}, 
                 agent_scale: int = 5000000):  # Tăng agent scale lên 5M
        self.nations = {n["name"]: {"observer": n["observer"], "space": n["space"]} for n in nations}
        self.t = t
        self.initial_omega = max(1e-6, initial_omega)
        self.k = max(0.1, k_constant)
        self.transcendence_key = transcendence_key
        self.resonance_factor = max(0.5, resonance_factor)
        self.deterministic = deterministic
        self.noise = cp.array(0) if deterministic else cp.random.uniform(0, 0.01)  # Giảm noise tối đa
        self.global_data = self.load_hyper_data(api_keys)
        self.initialize_nations()
        self.frequency_weights = {"short": 0.5, "mid": 0.3, "long": 0.2}
        self.cycle_periods = {"short": 2.0, "mid": 9.0, "long": 50.0}
        self.history = {name: [] for name in self.nations}
        self.axioms = {name: [] for name in self.nations}
        self.solutions = {name: [] for name in self.nations}
        self.reflection_network = nx.DiGraph()  # Mạng phản xạ chiến lược
        self.global_context = {
            "global_trade": 1.0, "global_inflation": 0.02, "global_growth": 0.03, 
            "geopolitical_tension": 0.2, "climate_impact": 0.1, "market_sentiment": 0.0,
            "pmi": 0.5, "credit_growth": 0.0, "regime_shift_probability": 0.0,
            "interest_rate": 0.02, "tax_rate": 0.1
        }
        self.eternal_pulse = self._activate_eternal_pulse()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = MicroEconomicPredictor(input_dim=72).to(self.device)  # Tăng input cho chính sách
        self.optimizer = torch.optim.AdamW(self.predictor.parameters(), lr=0.00001, weight_decay=1e-7)
        self.scaler = GradScaler()
        self.agents = []
        self.volatility_history = {name: [] for name in self.nations}
        self.gdp_history = {name: [] for name in self.nations}
        self.trade_flow_history = {name: [] for name in self.nations}
        self.error_history = {name: [] for name in self.nations}
        self.systemic_risk_network = nx.DiGraph()
        self.crowd_rf_model = RandomForestRegressor(n_estimators=300, max_depth=15)  # Tăng sức mạnh RF
        # Macroeconomic policy
        self.policy_memory = {name: deque(maxlen=100) for name in self.nations}
        self.successful_policies = {name: [] for name in self.nations}
        self.failed_policies = {name: [] for name in self.nations}
        self.policy_stats = {name: {} for name in self.nations}
        self.q_table = {name: {} for name in self.nations}
        self.gpr = GaussianProcessRegressor(kernel=C(1.0) * RBF(1.0), alpha=0.01)  # GPR cho success_score
        for n in nations:
            for i in range(agent_scale):
                role = random.choice(["citizen", "business", "government"])
                self.agents.append(HyperAgent(f"{n['name']}_{i}", n["name"], role, random.uniform(1e3, 1e10), 
                                              random.uniform(0, 0.3), random.uniform(0, 3.0), random.uniform(0, 0.6)))
        self.resonance_threshold = 4.0  # Tăng threshold cho độ nhạy cao hơn
        self.ekf = self._initialize_ekf()
        logging.info("VoTranh-Abyss-Core-Micro initialized with ultimate policy generator and reflexive network")

    def _initialize_ekf(self) -> ExtendedKalmanFilter:
        ekf = ExtendedKalmanFilter(dim_x=2, dim_z=1)
        ekf.x = np.array([1.0, 3.0])  # Tăng resonance_factor ban đầu
        ekf.P = np.eye(2) * 0.03  # Giảm covariance ban đầu
        ekf.Q = np.eye(2) * 0.0003
        ekf.R = np.array([[0.003]])
        return ekf

    def load_hyper_data(self, api_keys: Dict[str, str]) -> Dict:
        global_data = {}
        try:
            for ind in ["NY.GDP.MKTP.CD", "NE.TRD.GNFS.ZS", "FP.CPI.TOTL.ZG"]:
                url = f"http://api.worldbank.org/v2/country/all/indicator/{ind}?format=json&per_page=10000&key={api_keys.get('worldbank', 'default')}"
                response = requests.get(url, timeout=10).json()
                for entry in response[1]:
                    country = entry["country"]["value"]
                    year = int(entry["date"])
                    value = entry["value"] or 0
                    if country not in global_data:
                        global_data[country] = {}
                    if year not in global_data[country]:
                        global_data[country][year] = {}
                    global_data[country][year][ind] = value
            for country in global_data:
                for year in global_data[country]:
                    global_data[country][year].update({
                        "Climate_Risk": random.uniform(0, 1),
                        "Sentiment": random.uniform(-1, 1),
                        "Geopolitical_Stability": random.uniform(0, 1)
                    })
            logging.info("Loaded enhanced 10-year historical hyper-dimensional data")
        except Exception as e:
            logging.warning(f"Failed to load historical hyper data: {e}, using defaults")
        return global_data

    def load_realtime_data(self, nation_name: str, api_keys: Dict[str, str]) -> Dict[str, float]:
        try:
            crowd_sentiment = self.fetch_crowd_wisdom(nation_name, api_keys)
            market_data = fetch_market_data(nation_name)
            return {
                "GDP": self.nations[nation_name]["observer"]["GDP"],
                "Trade": self.nations[nation_name]["space"]["trade"],
                "Sentiment": crowd_sentiment,
                "Climate_Risk": random.uniform(0, 1),
                "Geopolitical_Tension": random.uniform(0, 1),
                "Market_Momentum": market_data["market_momentum"],
                "Stock_Volatility": market_data["Stock_Volatility"],
                "Gold_Price": market_data["Gold_Price"],
                "Oil_Price": market_data["Oil_Price"],
                "Currency_Rate": market_data["Currency_Rate"]
            }
        except Exception as e:
            logging.warning(f"Failed to load realtime data for {nation_name}: {e}, using defaults")
            return {
                "GDP": self.nations[nation_name]["observer"]["GDP"],
                "Trade": self.nations[nation_name]["space"]["trade"],
                "Sentiment": 0.0,
                "Climate_Risk": 0.5,
                "Geopolitical_Tension": 0.2,
                "Market_Momentum": 0.0,
                "Stock_Volatility": 0.0,
                "Gold_Price": 1800.0,
                "Oil_Price": 80.0,
                "Currency_Rate": 23000.0
            }

    def fetch_crowd_wisdom(self, nation_name: str, api_keys: Dict[str, str]) -> float:
        try:
            auth = tweepy.OAuthHandler(api_keys["twitter_consumer_key"], api_keys["twitter_consumer_secret"])
            auth.set_access_token(api_keys["twitter_access_token"], api_keys["twitter_access_token_secret"])
            twitter_api = tweepy.API(auth)
            tweets = twitter_api.search_tweets(q=f"{nation_name} economy #finance #crypto", count=1000, result_type="recent")
            custom_lexicon = {"bullish": 0.7, "bearish": -0.7, "crash": -0.8, "rally": 0.7}
            vader = SentimentIntensityAnalyzer()
            vader.lexicon.update(custom_lexicon)
            twitter_sentiment = np.mean([vader.polarity_scores(tweet.text)["compound"] for tweet in tweets]) or 0.0
            trends_score = random.uniform(0, 100)  # Placeholder Google Trends
            prediction_prob = random.uniform(0, 1)  # Placeholder Prediction Markets
            market_momentum = fetch_market_data(nation_name)["market_momentum"]
            crowd_sentiment = (0.45 * twitter_sentiment + 
                               0.30 * (trends_score / 100) + 
                               0.25 * prediction_prob * (1 + market_momentum * 0.2))
            return max(-1, min(1, crowd_sentiment))
        except Exception as e:
            logging.error(f"Error in fetch_crowd_wisdom for {nation_name}: {e}")
            return 0.0

    def generate_policy(self, nation_name: str, context: Dict[str, float]) -> Dict[str, float]:
        """Generate macroeconomic policy using Q-learning"""
        try:
            state = [context["pmi"], context["fear_index"], context["greed_index"], 
                     context.get("systemic_risk_score", 0.0), context.get("market_momentum", 0.0), 
                     context.get("Stock_Volatility", 0.0)]
            state_hash = str(round(sum(state) * 1000) / 1000)  # Hash đơn giản
            q_values = self.q_table[nation_name].get(state_hash, {})
            
            candidates = []
            volatility = context.get("Stock_Volatility", 0.0)
            duration_factor = 24 if volatility < 0.2 else 12 if volatility < 0.4 else 3  # Duration động

            if context["pmi"] < 0.4 or context["fear_index"] > 0.6:
                candidates.extend([
                    {"action": "reduce_tax", "param": min(0.15, 0.1 + (0.4 - context["pmi"]) * 0.5), "duration": duration_factor, "target": "citizen", "prob": 0.4},
                    {"action": "increase_subsidy", "param": min(0.1, 0.05 + (0.4 - context["pmi"]) * 0.3), "duration": duration_factor, "target": "citizen", "prob": 0.3},
                    {"action": "infrastructure_investment", "param": 0.03, "duration": duration_factor + 6, "target": "all", "prob": 0.2},
                    {"action": "currency_devaluation", "param": 0.1, "duration": duration_factor, "target": "all", "prob": 0.1}
                ])
            elif context["pmi"] > 0.8 or context["greed_index"] > 0.7:
                candidates.extend([
                    {"action": "raise_interest_rate", "param": min(0.015, 0.005 + (context["greed_index"] - 0.7) * 0.05), "duration": duration_factor, "target": "business", "prob": 0.5},
                    {"action": "tighten_credit", "param": min(0.25, 0.15 + (context["pmi"] - 0.8) * 0.5), "duration": duration_factor, "target": "business", "prob": 0.4},
                    {"action": "export_incentive", "param": 0.1, "duration": duration_factor, "target": "business", "prob": 0.1}
                ])
            elif context.get("systemic_risk_score", 0.0) > 0.9:
                candidates.extend([
                    {"action": "tighten_credit", "param": 0.2, "duration": duration_factor, "target": "business", "prob": 0.6},
                    {"action": "infrastructure_investment", "param": 0.05, "duration": duration_factor + 6, "target": "all", "prob": 0.25},
                    {"action": "education_reform", "param": 0.03, "duration": duration_factor + 12, "target": "all", "prob": 0.15}
                ])
            elif context.get("market_momentum", 0.0) < -1:
                candidates.extend([
                    {"action": "increase_subsidy", "param": 0.07, "duration": duration_factor, "target": "citizen", "prob": 0.5},
                    {"action": "reduce_tax", "param": 0.08, "duration": duration_factor, "target": "citizen", "prob": 0.3},
                    {"action": "currency_devaluation", "param": 0.15, "duration": duration_factor, "target": "all", "prob": 0.2}
                ])
            elif volatility > 0.5:
                candidates.extend([
                    {"action": "tighten_credit", "param": 0.25, "duration": duration_factor, "target": "business", "prob": 0.7},
                    {"action": "raise_interest_rate", "param": 0.01, "duration": duration_factor, "target": "business", "prob": 0.2},
                    {"action": "education_reform", "param": 0.02, "duration": duration_factor + 12, "target": "all", "prob": 0.1}
                ])
            else:
                candidates.append({"action": "education_reform", "param": 0.02, "duration": 24, "target": "all", "prob": 1.0})

            # """15% random exploration"""
            if random.random() < 0.15 or not q_values:
                return random.choice(candidates)

            # Select action based on Q-value
            q_values = {c["action"]: q_values.get(c["action"], 0) for c in candidates}
            best_action = max(q_values, key=q_values.get) if q_values else random.choice(candidates)["action"]
            return next(c for c in candidates if c["action"] == best_action)
        except Exception as e:
            logging.error(f"Error in generate_policy for {nation_name}: {e}")
            return {"action": "education_reform", "param": 0.02, "duration": 24, "target": "all"}

    def apply_policy_impact(self, nation_name: str, policy: Dict[str, float]):
        """Apply the effect of macroeconomic policy"""
        try:
            action = policy["action"]
            param = policy["param"]
            target = policy["target"]
            space = self.nations[nation_name]["space"]
            observer = self.nations[nation_name]["observer"]

            if action == "reduce_tax":
                self.global_context["tax_rate"] = max(0.01, self.global_context.get("tax_rate", 0.1) * (1 - param))
                for agent in self.agents:
                    if agent.nation == nation_name and agent.role == target:
                        agent.real_income *= 1 + 0.15 * param
                        agent.hope_index += 0.25
                        agent.fear_index -= 0.1
            elif action == "raise_interest_rate" and space["inflation"] >= 0.015:
                self.global_context["interest_rate"] += param
                for agent in self.agents:
                    if agent.nation == nation_name and agent.role in [target, "government"]:
                        agent.credit_growth *= 1 - 0.25 * param
                        agent.greed_index -= 0.15
                        agent.resilience += 0.05
            elif action == "increase_subsidy":
                for agent in self.agents:
                    if agent.nation == nation_name and agent.role == target:
                        agent.wealth += agent.wealth * 0.1 * param
                        agent.hope_index += 0.25
                        agent.fear_index -= 0.05
            elif action == "tighten_credit":
                for agent in self.agents:
                    if agent.nation == nation_name and agent.role == target:
                        agent.credit_growth *= 1 - 0.3 * param
                        agent.resilience += 0.1
                        agent.wealth *= 0.8
            elif action == "export_incentive":
                for agent in self.agents:
                    if agent.nation == nation_name and agent.role == target:
                        agent.trade_flow += agent.trade_flow * 0.2 * param
                space["inflation"] -= 0.05 * param
            elif action == "infrastructure_investment":
                observer["Material_Strength"] += 0.08 * param
                space["trade"] += 0.15 * param
                for agent in self.agents:
                    if agent.nation == nation_name:
                        agent.innovation += 0.1
            elif action == "education_reform":
                observer["Cultural_Depth"] += 0.05 * param
                for agent in self.agents:
                    if agent.nation == nation_name:
                        agent.innovation += 0.15 * param
                        agent.resilience += 0.08
            elif action == "currency_devaluation" and space["trade"] < 1.3 * self.trade_flow_history[nation_name][-90] if len(self.trade_flow_history[nation_name]) >= 90 else True:
                market_data = fetch_market_data(nation_name)
                market_data["Currency_Rate"] += market_data["Currency_Rate"] * 0.2 * param
                space["trade"] += 0.25 * param
                space["inflation"] += 0.15 * param

            # Lưu policy vào memory
            self.policy_memory[nation_name].append({
                "policy": policy, "t": self.t, "trigger_condition": context.copy(),
                "expected_outcome": {"resilience": space["resilience"], "cash_flow": 0.0, "consumption": 0.0}
            })
        except Exception as e:
            logging.error(f"Error in apply_policy_impact for {nation_name}: {e}")

    def evaluate_policy_impact(self, nation_name: str) -> None:
        """Evaluate the success or failure of the policy"""
        try:
            for policy_entry in list(self.policy_memory[nation_name]):
                t_start = policy_entry["t"]
                if self.t - t_start >= policy_entry["policy"]["duration"]:
                    past_idx = next(i for i, h in enumerate(self.history[nation_name]) if h["t"] >= t_start)
                    past = self.history[nation_name][past_idx]
                    now = self.history[nation_name][-1]
                    delta_resilience = now["resilience"] - past["resilience"]
                    delta_cash = now["Predicted_Value"]["short_term"] - past["Predicted_Value"]["short_term"]
                    delta_volatility = past["Volatility"] - now["Volatility"]
                    delta_consumption = (now["consumption"] + now["luxury_spending"] + now["debt_spending"] + now["durable_goods"]) - \
                                        (past["consumption"] + past["luxury_spending"] + past["debt_spending"] + past["durable_goods"])
                    delta_srs = past.get("srs", 0.0) - now.get("srs", 0.0)
                    
                    success_score = 0.3 * delta_resilience + 0.3 * delta_cash + 0.15 * delta_volatility + \
                                    0.15 * delta_consumption + 0.1 * delta_srs
                    
                    policy = policy_entry["policy"]
                    policy["success_score"] = success_score
                    if success_score > 0.02:
                        self.successful_policies[nation_name].append({"policy": policy, "score": success_score})
                        if success_score > 0.05:
                            policy["param"] = min(policy["param"] * 2, 0.25)  # Tăng param tối đa gấp đôi
                    elif success_score < -0.015:
                        self.failed_policies[nation_name].append(policy)
                    
                    # Cập nhật Q-table
                    state_hash = str(round(sum([policy_entry["trigger_condition"][k] for k in ["pmi", "fear_index", "greed_index"]]) * 1000) / 1000)
                    if state_hash not in self.q_table[nation_name]:
                        self.q_table[nation_name][state_hash] = {}
                    q = self.q_table[nation_name][state_hash].get(policy["action"], 0)
                    self.q_table[nation_name][state_hash][policy["action"]] = q + 0.1 * (success_score * 100 + 0.9 * max(self.q_table[nation_name][state_hash].values(), default=0) - q)
                    
                    # Cập nhật policy_stats
                    if policy["action"] not in self.policy_stats[nation_name]:
                        self.policy_stats[nation_name][policy["action"]] = []
                    self.policy_stats[nation_name][policy["action"]].append(success_score)
                    self.policy_memory[nation_name].remove(policy_entry)  # Xóa policy đã đánh giá
        except Exception as e:
            logging.error(f"Error in evaluate_policy_impact for {nation_name}: {e}")

    def initialize_nations(self):
        for name in self.nations:
            observer = self.nations[name]["observer"]
            space = self.nations[name]["space"]
            latest_year = max(self.global_data.get(name, {2023: {"NY.GDP.MKTP.CD": 450e9}}).keys())
            observer.update({
                "GDP": self.global_data.get(name, {}).get(latest_year, {}).get("NY.GDP.MKTP.CD", observer.get("GDP", 450e9)),
                "Population": observer.get("population", 1e6),
                "Material_Strength": random.uniform(0.5, 1),  # Tăng baseline
                "Cultural_Depth": random.uniform(0.6, 1)  # Tăng baseline
            })
            space.update({
                "trade": self.global_data.get(name, {}).get(latest_year, {}).get("NE.TRD.GNFS.ZS", space.get("trade", 1.0)) / 100,
                "inflation": self.global_data.get(name, {}).get(latest_year, {}).get("FP.CPI.TOTL.ZG", space.get("inflation", 0.0)) / 100,
                "institutions": space.get("institutions", 0.85),  # Tăng baseline
                "cultural_economic_factor": space.get("cultural_economic_factor", 1.2),  # Tăng baseline
                "market_sentiment": 0.0,
                "resilience": random.uniform(0.5, 1),  # Tăng baseline
                "fear_index": 0.0, "greed_index": 0.0, "complacency_index": 0.0, "hope_index": 0.0
            })
            self.nations[name]["amplitude"] = {
                "GDP": observer["GDP"] * self.resonance_factor,
                "Population": observer["Population"],
                "Material_Strength": observer["Material_Strength"],
                "Cultural_Resonance": observer["Cultural_Depth"]
            }
            self.nations[name]["resonance"] = self._initialize_resonance(space)
            self.gdp_history[name].append(observer["GDP"])
            self.trade_flow_history[name].append(space["trade"])
            self.systemic_risk_network.add_node(name, resilience=space["resilience"], trade_flow=space["trade"])

    def _activate_eternal_pulse(self) -> str:
        try:
            pulse_seed = hashlib.sha256(self.transcendence_key.encode()).hexdigest()
            return f"VoTranh-EternalPulse-{pulse_seed[:16]}"
        except Exception as e:
            logging.error(f"Error in eternal pulse: {e}")
            return "VoTranh-EternalPulse-Default"

    def _initialize_resonance(self, space: Dict[str, float]) -> Dict[str, float]:
        return {
            "Trade": space.get("trade", 1.0),
            "Inflation": space.get("inflation", 0.0),
            "Institutions": space.get("institutions", 0.85),
            "Innovation": space.get("innovation", 0.06),  # Tăng baseline
            "Labor_Participation": space.get("labor_participation", 0.95),  # Tăng baseline
            "Cultural_Economic_Factor": space.get("cultural_economic_factor", 1.2),
            "Resilience": space.get("resilience", 0.5),
            "Market_Sentiment": space.get("market_sentiment", 0.0)
        }

    def update_cycle_weights(self, nation_name: str):
        try:
            volatility_90d = np.std(self.volatility_history[nation_name][-90:]) if len(self.volatility_history[nation_name]) >= 90 else 0.0
            volatility_5y = np.mean(self.volatility_history[nation_name][-1825:]) if len(self.volatility_history[nation_name]) >= 1825 else 1e-6
            gdp_5y = (self.gdp_history[nation_name][-1] - self.gdp_history[nation_name][-1825]) / self.gdp_history[nation_name][-1825] if len(self.gdp_history[nation_name]) >= 1825 else 0.0
            gdp_10y = (self.gdp_history[nation_name][-1] - self.gdp_history[nation_name][-3650]) / self.gdp_history[nation_name][-3650] if len(self.gdp_history[nation_name]) >= 3650 else 0.0
            innovation_20y = np.mean([a.innovation for a in self.agents if a.nation == nation_name]) if self.t > 7300 else 0.06
            population_growth = (self.nations[nation_name]["amplitude"]["Population"] - self.nations[nation_name]["amplitude"]["Population"] * 0.97) / self.nations[nation_name]["amplitude"]["Population"]

            w1 = min(0.6, 0.5 * (volatility_90d / volatility_5y))
            w2 = min(0.4, 0.3 * (gdp_5y / (gdp_10y + 1e-6)))
            w3 = min(0.3, 0.2 * (innovation_20y + population_growth))
            total = w1 + w2 + w3
            if total > 0:
                self.frequency_weights["short"] = w1 / total
                self.frequency_weights["mid"] = w2 / total
                self.frequency_weights["long"] = w3 / total
            else:
                self.frequency_weights = {"short": 0.5, "mid": 0.3, "long": 0.2}
        except Exception as e:
            logging.error(f"Error in update_cycle_weights for {nation_name}: {e}")

    def compute_frequency(self) -> float:
        try:
            w1, w2, w3 = self.frequency_weights["short"], self.frequency_weights["mid"], self.frequency_weights["long"]
            p1, p2, p3 = self.cycle_periods["short"], self.cycle_periods["mid"], self.cycle_periods["long"]
            return 12.0 * (w1 * math.cos(2 * math.pi * self.t / p1) + 
                           w2 * math.cos(2 * math.pi * self.t / p2) + 
                           w3 * math.cos(2 * math.pi * self.t / p3)) * random.uniform(0.9, 1.6)
        except Exception as e:
            logging.error(f"Error in compute_frequency: {e}")
            return 12.0

    def compute_resonance(self, nation_name: str, global_context: Optional[Dict[str, float]] = None) -> float:
        try:
            amplitude = {k: cp.array(v) for k, v in self.nations[nation_name]["amplitude"].items()}
            resonance = {k: cp.array(v) for k, v in self.nations[nation_name]["resonance"].items()}
            cultural_factor = resonance["Cultural_Economic_Factor"]
            material_strength = amplitude["Material_Strength"]
            pmi_factor = 1 + global_context.get("pmi", 0.5) * 0.3
            market_momentum = global_context.get("market_momentum", 0.0) * 0.2
            L_t = (resonance["Trade"] * (1 - resonance["Inflation"]) * 
                   resonance["Institutions"] * resonance["Innovation"] * 
                   resonance["Labor_Participation"] * 
                   cp.log1p(amplitude["GDP"] / (amplitude["Population"] + 1e-6)) *
                   cultural_factor * resonance["Resilience"] * material_strength * pmi_factor * (1 + market_momentum) * 1.5)
            if global_context:
                L_t *= (1 + 0.5 * (global_context.get("global_trade", 1.0) - 1) + 
                        0.4 * global_context.get("global_growth", 0.03) - 
                        0.08 * global_context.get("global_inflation", 0.0) - 
                        0.05 * global_context.get("geopolitical_tension", 0.0))
            frequency = self.compute_frequency()
            return float(L_t + cp.sin(self.t / frequency) * self.noise * 2.5)
        except Exception as e:
            logging.error(f"Error in compute_resonance for {nation_name}: {e}")
            return 0.0

    def update_systemic_risk_network(self):
        try:
            for nation_a in self.nations:
                for nation_b in self.nations:
                    if nation_a != nation_b:
                        trade_flow_ab = sum(a.trade_flow for a in self.agents if a.nation == nation_a and a.role == "business") * 0.18
                        total_trade_a = sum(a.trade_flow for a in self.agents if a.nation == nation_a and a.role == "business") + 1e-6
                        dependency = trade_flow_ab / total_trade_a
                        self.systemic_risk_network.add_edge(nation_a, nation_b, weight=dependency)
        except Exception as e:
            logging.error(f"Error in update_systemic_risk_network: {e}")

    def calculate_systemic_risk_score(self, nation_name: str) -> float:
        try:
            self.update_systemic_risk_network()
            centrality = nx.betweenness_centrality(self.systemic_risk_network, k=800)  # Tăng k cho độ chính xác
            vulnerability = 1 - self.nations[nation_name]["space"]["resilience"]
            srs = sum(centrality.get(n, 0) * (1 - self.nations[n]["space"]["resilience"]) for n in self.nations) * vulnerability
            return min(1.0, max(0.0, srs))
        except Exception as e:
            logging.error(f"Error in calculate_systemic_risk_score for {nation_name}: {e}")
            return 0.0

    def project_pulse(self, nation_name: str, delta_t: float, new_space: Dict[str, float], 
                      external_shock: float = 0.0, global_context: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        try:
            nation = self.nations[nation_name]
            nation["resonance"].update(new_space)
            L_t_old = self.compute_resonance(nation_name, global_context)

            T_k = cp.where(nation["amplitude"]["Material_Strength"] > 0.5, 1.5, -0.5)  # Tăng biên độ
            delta_S_t = cp.sum(cp.array([(nation["resonance"].get(k, 0) - v) * w for k, v, w in [
                ("Trade", 1.0, 0.5), ("Inflation", 0.0, 0.4), ("Institutions", 0.5, 0.45),
                ("Innovation", 0.06, 0.4), ("Labor_Participation", 0.7, 0.35),
                ("Cultural_Economic_Factor", 0.8, 0.3), ("Resilience", 0.5, 0.4)
            ]])) + external_shock
            T = T_k + delta_S_t

            tau_t = cp.cos(2 * cp.pi * self.t / self.compute_frequency()) * nation["resonance"]["Resilience"] * 1.4
            L_t_new = self.compute_resonance(nation_name, global_context)
            integral_L = sum(h.get("L_t", 0) for h in self.history[nation_name][-60:]) + (L_t_new + L_t_old) * delta_t / 2
            omega_t = max(self.initial_omega * math.exp(integral_L * 1.4), 1e-6)
            a_t = (L_t_new - L_t_old) / delta_t if delta_t != 0 else 0
            s_loeh = a_t * math.log(omega_t + 1e-6) * 1.5

            pred_growth = 0.08 if len(self.history[nation_name]) < 60 else stats.linregress(
                [h["t"] for h in self.history[nation_name][-60:]], 
                [h["growth"] for h in self.history[nation_name][-60:]]
            ).slope * self.t + np.mean([h["growth"] for h in self.history[nation_name][-60:]])

            R_i = {
                "growth": float(T * tau_t * (1 + pred_growth + (0 if self.deterministic else cp.random.uniform(-self.noise, self.noise)))),
                "cash_flow": float(T * tau_t * nation["resonance"]["Trade"] * 1.5),
                "resilience": float(abs(T * tau_t) * nation["resonance"]["Resilience"] * 1.4),
                "L_t": float(L_t_new),
                "s_loeh": float(s_loeh),
                "t": self.t
            }
            self.history[nation_name].append(R_i.copy())
            self.reflection_network.add_node(f"{nation_name}_{self.t}", **R_i)
            return R_i
        except Exception as e:
            logging.error(f"Error in project_pulse for {nation_name}: {e}")
            return {"growth": 0.0, "cash_flow": 0.0, "resilience": 0.0, "L_t": 0.0, "s_loeh": 0.0, "t": self.t}

    def get_result_domain(self, nation_name: str) -> List[Dict[str, float]]:
        try:
            R_set = []
            base_R_i = self.project_pulse(nation_name, 1.0, self.nations[nation_name]["resonance"])
            domain_size = int(math.log(self.nations[nation_name]["amplitude"]["GDP"] + 1) * 8 + 12)  # Tăng domain size
            core_prob = 0.995
            step_prob = (1 - core_prob) / max(1, domain_size - 1)

            R_set.append({**base_R_i, "probability": core_prob})
            for i in range(1, domain_size):
                scale = 1 - i * 0.05
                R_set.append({
                    "growth": base_R_i["growth"] * scale,
                    "cash_flow": base_R_i["cash_flow"] * scale,
                    "resilience": base_R_i["resilience"] * scale,
                    "L_t": base_R_i["L_t"] * scale,
                    "s_loeh": base_R_i["s_loeh"] * scale,
                    "t": base_R_i["t"],
                    "probability": step_prob
                })
            return R_set
        except Exception as e:
            logging.error(f"Error in get_result_domain for {nation_name}: {e}")
            return []

    def update_amplitude(self, nation_name: str, feedback: Dict[str, float]) -> None:
        try:
            for key, value in feedback.items():
                self.nations[nation_name]["amplitude"][key] = max(0, self.nations[nation_name]["amplitude"].get(key, 0) + value * 0.5)
        except Exception as e:
            logging.error(f"Error in update_amplitude for {nation_name}: {e}")

    def compute_entropy(self, nation_name: str, L_t: float, delta_t: float, omega_t: float) -> float:
        try:
            a_t = (L_t - self.history[nation_name][-1]["L_t"]) / delta_t if self.history[nation_name] and delta_t != 0 else 0
            entropy = a_t * math.log(max(omega_t, 1e-6)) * 1.4
            return entropy if not math.isnan(entropy) and not math.isinf(entropy) else 0.0
        except Exception as e:
            logging.error(f"Error in compute_entropy for {nation_name}: {e}")
            return 0.0

    def update_entropy(self, nation_name: str, observer: Dict[str, float], space: Dict[str, float], 
                       delta_t: float, global_context: Optional[Dict[str, float]] = None) -> float:
        try:
            L_t = self.compute_resonance(nation_name, global_context)
            if global_context:
                self.global_context.update(global_context)
                L_t *= (1 + 0.55 * (self.global_context.get("global_trade", 1.0) - 1) - 
                        0.06 * self.global_context.get("global_inflation", 0.0))
            integral_L = sum(h["L_t"] for h in self.history[nation_name][-120:]) + L_t * delta_t if self.history[nation_name] else L_t * delta_t  # Tăng window
            omega_t = self.initial_omega * math.exp(integral_L * 1.4)
            return self.compute_entropy(nation_name, L_t, delta_t, omega_t)
        except Exception as e:
            logging.error(f"Error in update_entropy for {nation_name}: {e}")
            return 0.0
        # Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from typing import Dict, List, Optional, Tuple
import numpy as np
import cupy as cp
from scipy import stats, optimize
import pandas as pd
from datetime import datetime
import networkx as nx
import hashlib
import logging
import json
import requests
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from multiprocessing import Pool
import traceback
import tweepy
from filterpy.kalman import ExtendedKalmanFilter
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from collections import deque
import pickle
from annoy import AnnoyIndex  # Approximate Nearest Neighbors
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # Thêm import cho VADER
from part1 import QuantumResonanceLayer, MicroEconomicPredictor, HyperAgent, fetch_market_data, calculate_pmi
from part2 import VoTranhAbyssCoreMicro as VoTranhBase

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', 
                    handlers=[logging.FileHandler("votranh_abyss_micro.log"), logging.StreamHandler()])

class VoTranhAbyssCoreMicro(VoTranhBase):
    def __init__(self, nations: List[Dict[str, Dict]], t: float = 0.0, 
                 initial_omega: float = 40.0, k_constant: float = 2.0, 
                 transcendence_key: str = "Cauchyab12", resonance_factor: float = 3.0, 
                 deterministic: bool = False, api_keys: Dict[str, str] = {}, 
                 agent_scale: int = 5000000):
        super().__init__(nations, t, initial_omega, k_constant, transcendence_key, resonance_factor, 
                         deterministic, api_keys, agent_scale)
        self.portfolio_weights = {n["name"]: {"stocks": 0.4, "bonds": 0.3, "gold": 0.2, "cash": 0.1} for n in nations}
        self.policy_impact_matrix = {}
        self.pca = PCA(n_components=0.98)
        self.reflexive_cache = {name: deque(maxlen=10) for name in self.nations}  # Cache top 10 chính sách hiệu quả
        self.macro_strategies = {name: [] for name in self.nations}  # Chuỗi chiến lược vĩ mô
        self.ann_index = {name: AnnoyIndex(5, 'angular') for name in self.nations}  # ANN cho reflexive network
        self.ann_built = {name: False for name in self.nations}
        logging.info("VoTranh-Abyss-Core-Micro fully initialized with ultimate policy generator and reflexive strategic network")

    def optimize_portfolio(self, nation_name: str, short_pred: float, mid_pred: float, volatility: float, crowd_sentiment: float) -> Dict[str, float]:
        try:
            ER = 0.55 * short_pred + 0.45 * mid_pred * (1 + crowd_sentiment * 0.2)
            risk_free_rate = self.global_context["interest_rate"]
            history_short = [h["pred_value"]["short_term"] for h in self.history[nation_name][-90:]] if len(self.history[nation_name]) >= 90 else [0] * 90
            history_mid = [h["pred_value"]["mid_term"] for h in self.history[nation_name][-90:]] if len(self.history[nation_name]) >= 90 else [0] * 90
            history_gold = [h.get("Gold_Price", 1800.0) for h in self.history[nation_name][-90:]] if len(self.history[nation_name]) >= 90 else [1800.0] * 90
            history_vol = [h["volatility"] for h in self.history[nation_name][-90:]] if len(self.history[nation_name]) >= 90 else [0] * 90
            corr_matrix = np.corrcoef([history_short, history_mid, history_gold, history_vol])
            
            def sharpe_ratio(weights):
                risk = volatility * math.sqrt(
                    weights[0]**2 + weights[1]**2 + weights[2]**2 + weights[3]**2 +
                    2 * weights[0] * weights[1] * corr_matrix[0, 1] +
                    2 * weights[0] * weights[2] * corr_matrix[0, 2] +
                    2 * weights[1] * weights[2] * corr_matrix[1, 2]
                )
                return -(ER - risk_free_rate) / (risk + 1e-6)

            constraints = (
                {'type': 'eq', 'fun': lambda w: sum(w) - 1},
                {'type': 'ineq', 'fun': lambda w: 0.8 - (w[0] + w[2])}
            )
            bounds = [(0, 1)] * 4
            initial_weights = [0.4, 0.3, 0.2, 0.1]
            result = optimize.minimize(sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
            optimized_weights = result.x if result.success else initial_weights

            return {
                "stocks": optimized_weights[0], "bonds": optimized_weights[1], 
                "gold": optimized_weights[2], "cash": optimized_weights[3],
                "sharpe_ratio": -result.fun if result.success else 1.5
            }
        except Exception as e:
            logging.error(f"Error in optimize_portfolio for {nation_name}: {e}")
            return {"stocks": 0.4, "bonds": 0.3, "gold": 0.2, "cash": 0.1, "sharpe_ratio": 1.5}

    def suggest_reflexive_policy(self, nation_name: str, context: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Policy recommendation from the Strategic Reflex Network (SRN)"""
        try:
            current_state = [context["pmi"], context["market_sentiment"], context.get("market_momentum", 0.0), 
                             context.get("systemic_risk_score", 0.0), context.get("Stock_Volatility", 0.0)]
            if not self.ann_built[nation_name]:
                for i, (node, attr) in enumerate(self.reflection_network.nodes(data=True)):
                    if "policy" in node:
                        node_state = [attr["pmi"], attr["sentiment"], attr["context_snapshot"].get("market_momentum", 0.0),
                                      attr["context_snapshot"].get("systemic_risk_score", 0.0), attr["Volatility"]]
                        self.ann_index[nation_name].add_item(i, node_state)
                self.ann_index[nation_name].build(10)  # Build ANN index
                self.ann_built[nation_name] = True

            nearest = self.ann_index[nation_name].get_nns_by_vector(current_state, 5, include_distances=True)
            matches = []
            for idx, dist in zip(nearest[0], nearest[1]):
                node = list(self.reflection_network.nodes())[idx]
                if "policy" in node:
                    attr = self.reflection_network.nodes[node]
                    similarity = -0.35 * abs(current_state[0] - attr["pmi"]) - 0.25 * abs(current_state[1] - attr["sentiment"]) - \
                                 0.2 * abs(current_state[2] - attr["context_snapshot"].get("market_momentum", 0.0)) - \
                                 0.15 * abs(current_state[3] - attr["context_snapshot"].get("systemic_risk_score", 0.0)) - \
                                 0.05 * abs(current_state[4] - attr["Volatility"])
                    if similarity > -0.2:
                        edges = self.reflection_network[node]
                        for next_node in edges:
                            if edges[next_node].get("effect") == "positive":
                                matches.append((similarity + 0.1 if edges[next_node]["policy_success_score"] > 0.05 else similarity, 
                                               attr["action"], edges[next_node]["policy_success_score"]))

            if matches:
                matches.sort(reverse=True)
                top_matches = matches[:5]
                avg_score = sum(m[2] for m in top_matches) / len(top_matches)
                best_action = max(top_matches, key=lambda x: x[2])[1]
                confidence = max(m[0] for m in top_matches) * (avg_score / 0.05)
                if confidence > 0.5:
                    policy = {"action": best_action, "param": 0.05, "duration": 6, "target": "all", "confidence": min(1, confidence)}
                    self.reflexive_cache[nation_name].append(policy)
                    return policy
            return None
        except Exception as e:
            logging.error(f"Error in suggest_reflexive_policy for {nation_name}: {e}")
            return None

    def reflect_economy(self, t: float, observer: Dict[str, float], space: Dict[str, float], 
                        R_set: List[Dict[str, float]], nation_name: str, external_shock: float = 0.0) -> Dict[str, object]:
        try:
            # Load realtime data
            realtime = self.load_realtime_data(nation_name, api_keys)
            observer.update(realtime)
            space["market_sentiment"] = realtime["Sentiment"]
            entropy = self.update_entropy(nation_name, observer, space, 1.0)
            cash_flows = cp.array([r["cash_flow"] * observer.get("GDP", 1e9) for r in R_set])
            mean_cash_flow = float(cp.mean(cash_flows)) if cash_flows.size > 0 else 0.0
            volatility = float(cp.std(cash_flows)) if cash_flows.size > 0 else 0.0
            stability = 1 / (1 + volatility / (abs(mean_cash_flow) + 1e-6)) if mean_cash_flow != 0 else 0.0
            cultural_depth = observer.get("Cultural_Depth", 0.5)
            resilience = space.get("resilience", 0.5)

            # Store historical data
            self.volatility_history[nation_name].append(volatility)
            self.gdp_history[nation_name].append(observer["GDP"])
            self.trade_flow_history[nation_name].append(space["trade"])

            # Compute PMI and SRS
            credit_growth = np.mean([a.credit_growth for a in self.agents if a.nation == nation_name])
            self.global_context["credit_growth"] = credit_growth
            pmi = calculate_pmi(self.volatility_history[nation_name], self.gdp_history[nation_name], 
                                self.trade_flow_history[nation_name], space["market_sentiment"], 
                                self.global_context["geopolitical_tension"], credit_growth)
            self.global_context["pmi"] = pmi
            if space["market_sentiment"] - space.get("sentiment_prev", 0.0) > 0.3:
                self.global_context["pmi"] += 0.15
            elif space["market_sentiment"] - space.get("sentiment_prev", 0.0) < -0.3:
                self.global_context["pmi"] -= 0.20
            self.update_cycle_weights(nation_name)
            srs = self.calculate_systemic_risk_score(nation_name)
            self.global_context["systemic_risk_score"] = srs
            self.global_context["Stock_Volatility"] = volatility
            self.global_context["market_momentum"] = realtime["Market_Momentum"]

            # Forecast economic indicators
            pred_input = torch.tensor([[t] + list(observer.values()) + list(space.values()) + 
                                      [space["market_sentiment"], space["trade"], space["resilience"],
                                       space["fear_index"], space["greed_index"], space["complacency_index"], space["hope_index"],
                                       realtime["Market_Momentum"], realtime["Stock_Volatility"], realtime["Gold_Price"], 
                                       realtime["Oil_Price"], realtime["Currency_Rate"], 
                                       self.global_context["tax_rate"], srs]], 
                                      dtype=torch.float32).to(self.device)
            short_pred, mid_pred = self.predictor(pred_input.unsqueeze(0))
            pred_value = {"short_term": short_pred.item(), "mid_term": mid_pred.item()}
            self.train_predictor(nation_name)

            # Execute Market Feedback Loop via Extended Kalman Filter (EKF)
            actual_growth = realtime["Market_Momentum"] * 0.15
            error = pred_value["short_term"] - actual_growth
            self.error_history[nation_name].append(error)
            if len(self.error_history[nation_name]) >= 30 or volatility > 0.25:
                self.ekf.update(np.array([error]))
                self.ekf.predict()
                pmi_factor, new_resonance = self.ekf.x
                self.resonance_factor = max(0.8, min(1.2, new_resonance))
                logging.info(f"EKF updated for {nation_name}: pmi_factor={pmi_factor:.3f}, resonance_factor={self.resonance_factor:.3f}")

            # Consumer Behavior Prediction
            base_consumption = sum(a.wealth for a in self.agents if a.nation == nation_name and a.role == "citizen") * 0.015
            consumption = base_consumption
            luxury_spending = 0.0
            debt_spending = 0.0
            durable_goods = 0.0
            real_income_drop = (sum(a.real_income_history[0] for a in self.agents if a.nation == nation_name and a.role == "citizen") - 
                                sum(a.real_income for a in self.agents if a.nation == nation_name and a.role == "citizen")) / \
                               (sum(a.real_income_history[0] for a in self.agents if a.nation == nation_name and a.role == "citizen") + 1e-6) if len(self.history[nation_name]) >= 30 else 0.0
            if space["fear_index"] > 0.6:
                consumption *= (1 - space["fear_index"]) ** 3 * (1 - real_income_drop * 0.5)
            elif space["greed_index"] > 0.7:
                luxury_spending = math.log(sum(a.wealth for a in self.agents if a.nation == nation_name) + 1) * space["greed_index"] * 2 * (1 + realtime["Market_Momentum"] * 0.3)
            elif space["complacency_index"] > 0.5:
                debt_spending = sum(a.wealth for a in self.agents if a.nation == nation_name) * space["complacency_index"] * 1.0 * (1 + credit_growth * 0.6)
            elif space["hope_index"] > 0.6:
                durable_goods = base_consumption * space["hope_index"] * 1.5 * (1 + realtime["Stock_Volatility"] * 0.2)
            delta_consumption = consumption + luxury_spending + debt_spending + durable_goods - base_consumption

            # Portfolio Optimization
            portfolio = self.optimize_portfolio(nation_name, pred_value["short_term"], pred_value["mid_term"], volatility, space["market_sentiment"])
            self.portfolio_weights[nation_name] = portfolio

            # Policy Impact & Reflexive Network
            policy = None
            reflexive_policy = self.suggest_reflexive_policy(nation_name, self.global_context)
            if reflexive_policy and random.random() < 0.8:  # 80% ưu tiên phản xạ
                policy = reflexive_policy
            else:
                policy = self.generate_policy(nation_name, self.global_context)
            if policy:
                self.apply_policy_impact(nation_name, policy)
                self.policy_memory[nation_name][-1]["expected_outcome"] = {"resilience": resilience, "cash_flow": pred_value["short_term"], 
                                                                          "consumption": consumption + luxury_spending + debt_spending + durable_goods}
            self.evaluate_policy_impact(nation_name)

            # Add a policy node into the reflection_network
            if policy:
                self.reflection_network.add_node(f"policy_{t}", 
                    action=policy["action"], param=policy["param"], duration=policy["duration"], target=policy["target"],
                    context_snapshot=self.global_context.copy(), sentiment=space["market_sentiment"], pmi=pmi,
                    resilience=resilience, Volatility=volatility, crowd_sentiment=space["market_sentiment"],
                    psychology_dominant=max(['fear', 'greed', 'complacency', 'hope'], key=lambda k: space[k + '_index']),
                    market_state=[realtime["Stock_Volatility"], realtime["Gold_Price"], realtime["Oil_Price"], realtime["Currency_Rate"]],
                    consumption_state=consumption + luxury_spending + debt_spending + durable_goods)
                self.reflection_network.add_edge(f"policy_{t}", f"{nation_name}_{t}", 
                    result_cashflow=pred_value["short_term"], volatility=volatility, srs=srs, 
                    delta_consumption=delta_consumption, time_lag=1, external_shock=external_shock,
                    effect="pending", policy_success_score=0.0)  # Success_score sẽ cập nhật sau

            # Practical Output
            domain = self.get_result_domain(nation_name)
            growth_p5 = np.percentile([r["growth"] for r in domain], 5)
            growth_p95 = np.percentile([r["growth"] for r in domain], 95)
            risk = abs(growth_p5) if growth_p5 < 0 else 0
            reward = growth_p95 if growth_p95 > 0 else 0

            policy_rec = policy if policy else {"action": "Maintain current policy", "success_prob": 0.99}
            invest_signal = {"action": f"Portfolio: {portfolio['stocks']*100:.0f}% stocks, {portfolio['bonds']*100:.0f}% bonds", 
                             "reward": reward * 100, "risk": risk * 100}
            psych_insight = f"Market in {nation_name}: PMI={pmi:.3f}, SRS={srs:.3f}, Consumption State={max([a.consumption_state for a in self.agents if a.nation == nation_name], key=lambda x: x.count(x))}"
            if pmi > 0.8 and space["complacency_index"] > 0.7:
                psych_insight += " - Complacency high, bubble risk 75%."
            elif pmi < 0.3 and space["fear_index"] > 0.8:
                psych_insight += " - Fear dominating, contraction imminent."
            elif srs > 1.0:
                psych_insight += " - Systemic risk critical, domino effect 85%."

            self.axioms[nation_name].append({"Statement": "Cash flow is the eternal pulse; resilience binds the land.", "Confidence": 0.999})
            self.solutions[nation_name].append({"trade": "Maximize trade flows to amplify resilience over 10 years."})

            result = {
                "Axiom": {"Statement": "Cash flow is the eternal pulse; resilience binds the land.", "Confidence": 0.999},
                "Solution": {"Policy": policy_rec, "Investment": invest_signal},
                "Insight": {"Psychology": psych_insight, "Systemic_Risk_Score": srs, "Consumption": {
                    "base": consumption, "luxury": luxury_spending, "debt": debt_spending, "durable": durable_goods}},
                "Predicted_Value": pred_value,
                "Volatility": volatility,
                "Stability": stability,
                "Entropy": entropy,
                "Resilience": resilience,
                "Cultural_Depth": cultural_depth,
                "Portfolio": portfolio,
                "Eternal_Pulse": self.eternal_pulse,
                "Network_Depth": len(self.reflection_network.nodes)
            }
            self.history[nation_name][-1].update({
                "pred_value": pred_value, "consumption": consumption, "luxury_spending": luxury_spending, 
                "debt_spending": debt_spending, "durable_goods": durable_goods, "srs": srs, 
                "pmi": pmi, "observer": observer.copy(), "space": space.copy()
            })
            logging.info(f"Reflection for {nation_name} at t={t:.1f}: Resilience={resilience:.3f}, PMI={pmi:.3f}, SRS={srs:.3f}")
            return result
        except Exception as e:
            logging.error(f"Critical error in reflect_economy for {nation_name}: {e}\n{traceback.format_exc()}")
            return {
                "Insight": {"Psychology": f"In {nation_name} at t={t:.1f}, chaos emerges from {str(e)}, revealing fragility."},
                "Entropy": float('inf'),
                "Resilience": 0.0,
                "Eternal_Pulse": self.eternal_pulse
            }

    def simulate_nation_step(self, args):
        try:
            nation_name, t, delta_t, space, R_set, global_context = args
            nation_agents = [a for a in self.agents if a.nation == nation_name]
            market_data = fetch_market_data(nation_name)
            policy_frequency = 5 if abs(global_context["market_momentum"]) > 1.5 or global_context["Stock_Volatility"] > 0.4 else \
                              15 if global_context["pmi"] < 0.3 or global_context["systemic_risk_score"] > 1.0 else 30
            policy = self.generate_policy(nation_name, global_context) if int(t) % policy_frequency == 0 else None
            for agent in nation_agents:
                agent.interact(nation_agents, global_context, space, 
                               self.volatility_history[nation_name], self.gdp_history[nation_name], market_data, policy)
            self.project_pulse(nation_name, delta_t, space, external_shock=0.0, global_context=global_context)
            return {nation_name: self.reflect_economy(t, self.nations[nation_name]["observer"], space, R_set, nation_name)}
        except Exception as e:
            logging.error(f"Error in simulate_nation_step for {nation_name}: {e}")
            return {nation_name: {"Error": str(e), "t": t}}

    def simulate_system(self, steps: int, delta_t: float, space_sequence: List[Dict], 
                        R_set_sequence: List[List[Dict]], global_context_sequence: List[Dict] = None) -> Dict[str, List[Dict[str, object]]]:
        results = {name: [] for name in self.nations}
        current_t = self.t
        try:
            with Pool(processes=len(self.nations) * 2) as p:
                for step in range(steps):
                    args = [(name, current_t, delta_t, space_sequence[step % len(space_sequence)], 
                             R_set_sequence[step % len(R_set_sequence)], 
                             global_context_sequence[step % len(global_context_sequence)] if global_context_sequence else self.global_context)
                            for name in self.nations]
                    step_results = p.map(self.simulate_nation_step, args)
                    for res in step_results:
                        for name, data in res.items():
                            results[name].append(data)
                    current_t += delta_t
                    self.t = current_t

                    # Evolution graph every 200 steps
                    if step % 200 == 0 and step > 0:
                        pr = nx.pagerank(self.reflection_network)
                        nodes_to_remove = [n for n, attr in self.reflection_network.nodes(data=True) if "policy" in n and 
                                          (pr[n] < 0.002 and attr.get("policy_success_score", 0) < 0.03)]
                        for node in nodes_to_remove[:int(len(nodes_to_remove) * 0.1)]:  # Xóa 10%
                            self.reflection_network.remove_node(node)
                        # Detect macro-strategies
                        for name in self.nations:
                            sequences = {}
                            for n1, n2 in self.reflection_network.edges():
                                if "policy" in n1 and "timestep" in n2 and n2.startswith(name):
                                    edge_data = self.reflection_network[n1][n2]
                                    if edge_data.get("policy_success_score", 0) > 0.025:
                                        seq_key = f"{n1}->{n2}"
                                        sequences[seq_key] = sequences.get(seq_key, 0) + 1
                                        if sequences[seq_key] > 3:
                                            self.macro_strategies[name].append({
                                                "sequence": [self.reflection_network.nodes[n1]["action"], 
                                                             self.reflection_network.nodes[n2]["action"] if "policy" in n2 else "timestep"],
                                                "score": edge_data["policy_success_score"],
                                                "trigger_condition": self.reflection_network.nodes[n1]["context_snapshot"]
                                            })
                        # Compress graph if too large
                        if len(self.reflection_network.nodes) > 50000:
                            with open(f"reflection_network_{step}.pkl", "wb") as f:
                                pickle.dump(self.reflection_network, f)
                            self.reflection_network = nx.DiGraph()
                            for name in self.nations:
                                self.ann_built[name] = False
            return results
        except Exception as e:
            logging.error(f"Error in simulate_system: {e}\n{traceback.format_exc()}")
            return results

    def forecast_system(self, steps: int, delta_t: float, space_sequence: List[Dict], 
                        R_set_sequence: List[List[Dict]], global_context_sequence: List[Dict] = None) -> Dict[str, List[Dict[str, object]]]:
        forecast = {name: [] for name in self.nations}
        current_t = self.t
        try:
            with Pool(processes=len(self.nations) * 2) as p:
                for step in range(steps):
                    args = [(name, current_t, delta_t, space_sequence[step % len(space_sequence)], 
                             R_set_sequence[step % len(R_set_sequence)], 
                             global_context_sequence[step % len(global_context_sequence)] if global_context_sequence else self.global_context)
                            for name in self.nations]
                    step_results = p.map(self.simulate_nation_step, args)
                    for res in step_results:
                        for name, data in res.items():
                            data["Forecast_Confidence"] = 0.999 - 0.0001 * step  # Giảm decay rate
                            forecast[name].append(data)
                    current_t += delta_t
            return forecast
        except Exception as e:
            logging.error(f"Error in forecast_system: {e}\n{traceback.format_exc()}")
            return forecast

    def export_data(self, filename: str = "votranh_abyss_micro.csv") -> None:
        try:
            for nation_name in self.nations:
                data = {
                    "Time": [h["t"] for h in self.history[nation_name]],
                    "Short_Term_Prediction": [h["pred_value"]["short_term"] for h in self.history[nation_name]],
                    "Mid_Term_Prediction": [h["pred_value"]["mid_term"] for h in self.history[nation_name]],
                    "Volatility": [h["volatility"] for h in self.history[nation_name]],
                    "Stability": [h["stability"] for h in self.history[nation_name]],
                    "Entropy": [h["entropy"] for h in self.history[nation_name]],
                    "Resilience": [h["resilience"] for h in self.history[nation_name]],
                    "Cultural_Depth": [h["cultural_depth"] for h in self.history[nation_name]],
                    "PMI": [h.get("pmi", 0.5) for h in self.history[nation_name]],
                    "SRS": [h.get("srs", 0.0) for h in self.history[nation_name]],
                    "Consumption": [h["consumption"] for h in self.history[nation_name]],
                    "Luxury_Spending": [h["luxury_spending"] for h in self.history[nation_name]],
                    "Debt_Spending": [h["debt_spending"] for h in self.history[nation_name]],
                    "Durable_Goods": [h["durable_goods"] for h in self.history[nation_name]]
                }
                df = pd.DataFrame(data)
                nation_file = filename.replace(".csv", f"_{nation_name}.csv")
                df.to_csv(nation_file, index=False)
                with open(nation_file.replace(".csv", ".json"), "w") as f:
                    json.dump(self.history[nation_name], f, indent=2)
                logging.info(f"Micro data exported to {nation_file}")
        except Exception as e:
            logging.error(f"Error in export_data: {e}\n{traceback.format_exc()}")

    def train_predictor(self, nation_name: str):
        try:
            if len(self.history[nation_name]) > 500:  # Tăng dữ liệu huấn luyện
                X = torch.tensor([[h["t"]] + list(h["observer"].values()) + list(h["space"].values()) + 
                                 [h["space"]["market_sentiment"], h["space"]["trade"], h["space"]["resilience"],
                                  h["space"]["fear_index"], h["space"]["greed_index"], h["space"]["complacency_index"], h["space"]["hope_index"],
                                  h.get("Market_Momentum", 0.0), h["Volatility"], h.get("Gold_Price", 1800.0), 
                                  h.get("Oil_Price", 80.0), h.get("Currency_Rate", 23000.0), 
                                  self.global_context["tax_rate"], h.get("srs", 0.0)] 
                                 for h in self.history[nation_name][-500:]], dtype=torch.float32).to(self.device)
                y_short = torch.tensor([h["growth"] for h in self.history[nation_name][-250:]], 
                                      dtype=torch.float32).to(self.device)
                y_mid = torch.tensor([h["cash_flow"] for h in self.history[nation_name][-125:]], 
                                    dtype=torch.float32).to(self.device)
                with autocast():
                    short_pred, mid_pred = self.predictor(X.unsqueeze(0))
                    loss = nn.MSELoss()(short_pred.squeeze(), y_short[-1]) + nn.MSELoss()(mid_pred.squeeze(), y_mid.mean())
                    loss += 0.0003 * sum(p.pow(2).sum() for p in self.predictor.parameters())  # Regularization mạnh hơn
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                logging.info(f"Trained MicroEconomicPredictor for {nation_name}")
        except Exception as e:
            logging.error(f"Error in train_predictor for {nation_name}: {e}")

if __name__ == "__main__":
    nations = [
        {"name": "Vietnam", "observer": {"GDP": 450e9, "population": 100e6}, 
         "space": {"trade": 0.8, "inflation": 0.04, "institutions": 0.7, "cultural_economic_factor": 0.85}},
        {"name": "USA", "observer": {"GDP": 26e12, "population": 331e6}, 
         "space": {"trade": 1.2, "inflation": 0.03, "institutions": 0.85, "cultural_economic_factor": 0.75}}
    ]
    api_keys = {
        "worldbank": "your_worldbank_api_key",
        "twitter_consumer_key": "your_twitter_consumer_key",
        "twitter_consumer_secret": "your_twitter_consumer_secret",
        "twitter_access_token": "your_twitter_access_token",
        "twitter_access_token_secret": "your_twitter_access_token_secret"
    }
    core = VoTranhAbyssCoreMicro(nations, transcendence_key="Cauchyab12", deterministic=False, api_keys=api_keys)

    space_sequence = [
        {"trade": 0.8, "inflation": 0.04, "institutions": 0.7, "cultural_economic_factor": 0.85},
        {"trade": 0.78, "inflation": 0.045, "institutions": 0.7, "cultural_economic_factor": 0.82},
        {"trade": 0.76, "inflation": 0.05, "institutions": 0.68, "cultural_economic_factor": 0.80}
    ]
    R_set_sequence = [
        [{"growth": 0.03, "cash_flow": 0.5}, {"growth": -0.02, "cash_flow": 0.3}, {"growth": 0.01, "cash_flow": 0.4}],
        [{"growth": 0.02, "cash_flow": 0.45}, {"growth": -0.01, "cash_flow": 0.25}, {"growth": 0.005, "cash_flow": 0.35}],
        [{"growth": 0.015, "cash_flow": 0.4}, {"growth": -0.015, "cash_flow": 0.2}, {"growth": 0.0, "cash_flow": 0.3}]
    ]
    global_context_sequence = [
        {"global_trade": 1.0, "global_inflation": 0.02, "global_growth": 0.03, "geopolitical_tension": 0.2, "climate_impact": 0.1},
        {"global_trade": 0.95, "global_inflation": 0.025, "global_growth": 0.025, "geopolitical_tension": 0.25, "climate_impact": 0.12},
        {"global_trade": 0.9, "global_inflation": 0.03, "global_growth": 0.02, "geopolitical_tension": 0.3, "climate_impact": 0.15}
    ]
    results = core.simulate_system(3650, 1.0, space_sequence, R_set_sequence, global_context_sequence)

    for nation_name, nation_results in results.items():
        print(f"\nResults for {nation_name}:")
        for i, res in enumerate(nation_results[:5]):
            print(f"Day {i+1}: Short-Term={res['Predicted_Value']['short_term']:.2e}, Mid-Term={res['Predicted_Value']['mid_term']:.2e}, Resilience={res['Resilience']:.3f}")
            print(f"Solution: {res['Solution']}")
            print(f"Insight: {res['Insight']}\n")

    core.export_data("votranh_abyss_micro_vietnam_2034.csv")
