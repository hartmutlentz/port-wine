import pandas as pd
import yfinance as yf

class Asset:
    """Basic class for a single asset. A simplified Wrapper around a yfinance Ticker.

        Attributes
        ----------
        identifier : str
            Unique identifier of the asset as used in Yahoo! Finance.

    """

    def __init__(self, identifier: str, period: str = "6Mo"):
        """
        Parameters
        ----------
        identifier : str
            Unique identifier

        period : str
            Considered time period. Interval [period, now].
            Allowed values: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo, 6Mo
        """
        self.identifier = identifier
        self.Ticker = yf.Ticker(identifier)
        self.period = period

        self.currency = self.Ticker.info["currency"]
        self.name = self.Ticker.info["longName"]

        self.prices = self.get_prices()
        self.returns = self.get_returns()

        print("This class is temporary.s")

    def get_returns(self, price_time="Close"):
        """Returns as time series"""
        X = self.Ticker.history(period=self.period)[price_time].diff()
        X.rename("Returns", inplace=True)
        return X.iloc[1:]

    def get_prices(self, price_time="Close"):
        """Prices as time series. Parameter price_time can be open, closing, high, low."""
        return self.Ticker.history(period=self.period)[price_time]

    def get_expected_return(self):
        """Return expectation value of returns."""
        pass

    def get_sigma(self):
        """Return standard deviation of returns."""
        pass


if __name__ == "__main__":
    A = Asset("PFE", "6Mo")
    print(A.prices)
