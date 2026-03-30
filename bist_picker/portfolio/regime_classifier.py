
import datetime
import logging
import pandas as pd
from sqlalchemy.orm import Session
from bist_picker.db.schema import Company, DailyPrice

logger = logging.getLogger(__name__)

class MarketRegimeClassifier:
    """Classifies market regime based on XU100 price, SMA200, and volatility.
    
    Regimes:
      - BULL_LOW_VOL:  Price > SMA200 and 3m volatility < 1y median
      - BULL_HIGH_VOL: Price > SMA200 and 3m volatility >= 1y median
      - BEAR:          Price <= SMA200
    """
    
    def __init__(self, session: Session):
        self.session = session

    def _get_xu100_prices(self, end_date: datetime.date) -> pd.DataFrame:
        """Fetch XU100 prices up to end_date."""
        xu100 = (
            self.session.query(Company.id)
            .filter(Company.ticker == "XU100")
            .first()
        )
        if xu100 is None:
            logger.warning("XU100 benchmark company not found in database")
            return pd.DataFrame()
        xu100_id = xu100[0]

        query = (
            self.session.query(DailyPrice.date, DailyPrice.close, DailyPrice.adjusted_close)
            .filter(DailyPrice.company_id == xu100_id, DailyPrice.date <= end_date)
            .order_by(DailyPrice.date.asc())
        )
        df = pd.read_sql(query.statement, self.session.bind)
        if df.empty:
            return df
        df['price'] = df['adjusted_close'].fillna(df['close'])
        return df

    def classify(self, scoring_date: datetime.date) -> str:
        """Determines the market regime using a responsive multi-indicator approach."""
        df = self._get_xu100_prices(scoring_date)
        
        if len(df) < 200:
            return "BULL_LOW_VOL"
        
        latest_price = df['price'].iloc[-1]
        sma200 = df['price'].rolling(200).mean().iloc[-1]
        sma50 = df['price'].rolling(50).mean().iloc[-1]
        
        # Calculate returns and 3-month (63 trading days) volatility
        df['returns'] = df['price'].pct_change()
        df['vol'] = df['returns'].rolling(63).std()
        
        latest_vol = df['vol'].iloc[-1]
        median_vol = df['vol'].tail(252).median()
        
        # Responsiveness: Even if below SMA200, if Price > SMA50 and 1-month momentum is positive, it's a potential recovery.
        one_month_ret = (df['price'].iloc[-1] / df['price'].iloc[-21]) - 1
        
        is_bullish = latest_price > sma200 or (latest_price > sma50 and one_month_ret > 0.05)
        
        if is_bullish:
            if latest_vol < median_vol:
                return "BULL_LOW_VOL"
            else:
                return "BULL_HIGH_VOL"
        else:
            return "BEAR"
