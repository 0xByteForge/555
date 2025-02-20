import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy, IntParameter
import tensorflow as tf
from tensorflow.keras.models import load_model
import talib.abstract as ta
from datetime import datetime
from typing import Optional

class LSTMStrategy(IStrategy):
    INTERFACE_VERSION = 3
    
    # Temel parametreler
    timeframe = '5m'
    minimal_roi = {
        "0": 0.05,    # 5 dakika içinde %5 kar
        "30": 0.025,  # 30 dakika sonra %2.5 kar
        "60": 0.015,  # 1 saat sonra %1.5 kar
        "120": 0.01   # 2 saat sonra %1 kar
    }
    
    # Dinamik stoploss için başlangıç değeri
    stoploss = -0.02  # Başlangıç stop-loss %2
    
    # Trailing stop (dinamik stop) ayarları
    trailing_stop = True
    trailing_stop_positive = 0.01  # %1 kar sonrası trailing aktif
    trailing_stop_positive_offset = 0.02  # Trailing stop mesafesi %2
    trailing_only_offset_is_reached = True  # Sadece offset'e ulaşınca trailing başlat
    
    # Risk yönetimi parametreleri
    position_adjustment_enable = True  # Pozisyon büyüklüğü ayarı
    max_entry_position_adjustment = 8  # Maksimum giriş sayısı
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.model = load_model('user_data/models/best_model_combined_20250220_010540.h5')
        self.WINDOW_SIZE = 80
        
        # Trend takibi için
        self.trend_periods = 14
        self.min_trend_strength = 0.02  # %2 trend gücü
        
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Dinamik stop-loss hesaplama
        """
        # Kar arttıkça stop-loss seviyesini yukarı çek
        if current_profit > 0.03:  # %3 kar
            return current_profit - 0.02  # %2 altında stop
        elif current_profit > 0.02:  # %2 kar
            return current_profit - 0.015  # %1.5 altında stop
        
        return -0.02  # Varsayılan stop-loss
        
    def custom_entry_price(self, pair: str, current_time: datetime, proposed_rate: float,
                          entry_tag: Optional[str], **kwargs) -> float:
        """
        Özel giriş fiyatı belirleme
        """
        # Volatiliteye göre giriş fiyatını ayarla
        dataframe = self.dp.get_pair_dataframe(pair, self.timeframe)
        volatility = self.get_volatility(dataframe)
        
        if volatility > 0.02:  # Yüksek volatilite
            return proposed_rate * 0.995  # %0.5 daha düşük giriş
        return proposed_rate
        
    def get_volatility(self, dataframe: pd.DataFrame) -> float:
        """
        Volatilite hesaplama
        """
        return np.std(dataframe['close'].pct_change().dropna()) * np.sqrt(288)  # Günlük volatilite
        
    def get_trend_strength(self, dataframe: pd.DataFrame) -> float:
        """
        Trend gücünü hesapla
        """
        close_prices = dataframe['close'].values
        trend = (close_prices[-1] - close_prices[-self.trend_periods]) / close_prices[-self.trend_periods]
        return trend
        
    def feature_engineering(self, dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Özellik mühendisliği - modelde kullandığınız aynı özellikleri oluşturun
        """
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        
        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['bb_middle'] = bollinger['middleband']
        
        # Stochastic RSI
        stoch = ta.STOCHRSI(dataframe)
        dataframe['stoch_rsi'] = stoch['fastd']
        
        # Williams %R
        dataframe['williams_r'] = ta.WILLR(dataframe)
        
        return dataframe
        
    def create_sequence(self, df: pd.DataFrame) -> np.ndarray:
        """
        Model için gerekli sequence'i oluştur
        """
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd',
                       'bb_upper', 'bb_lower', 'bb_middle', 'stoch_rsi', 'williams_r']
        
        # Son WINDOW_SIZE kadar veriyi al
        sequence = df[feature_cols].values[-self.WINDOW_SIZE:]
        
        # Boyut kontrolü
        if len(sequence) < self.WINDOW_SIZE:
            return None
            
        return sequence.reshape(1, self.WINDOW_SIZE, len(feature_cols))
        
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe = self.feature_engineering(dataframe)
        
        # Ek indikatörler
        dataframe['volatility'] = dataframe['close'].rolling(window=20).std()
        dataframe['trend_strength'] = self.get_trend_strength(dataframe)
        
        return dataframe
        
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        if len(dataframe) >= self.WINDOW_SIZE:
            sequence = self.create_sequence(dataframe)
            if sequence is not None:
                prediction = self.model.predict(sequence)[0]
                current_close = dataframe['close'].iloc[-1]
                trend_strength = self.get_trend_strength(dataframe)
                volatility = self.get_volatility(dataframe)
                
                future_price = prediction[0]
                price_change = (future_price - current_close) / current_close
                
                # Long pozisyon koşulları
                if (price_change > 0.01 and  # %1 üzeri artış beklentisi
                    trend_strength > self.min_trend_strength and  # Trend yeterince güçlü
                    volatility < 0.03):  # Volatilite makul seviyede
                    dataframe.loc[dataframe.index[-1], 'enter_long'] = 1
                
                # Short pozisyon koşulları
                elif (price_change < -0.01 and  # %1 üzeri düşüş beklentisi
                      trend_strength < -self.min_trend_strength and
                      volatility < 0.03):
                    dataframe.loc[dataframe.index[-1], 'enter_short'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Çıkış sinyallerini oluştur
        """
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        
        # ROI ve stoploss kullanacağız
        return dataframe 