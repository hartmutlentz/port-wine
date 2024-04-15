import yfinance as yf
import numpy as np
from Asset import Asset


class Portfolio:
    """
    Class for multiple assets data. Wrapper for yFinance-Tickers.

    """

    def __init__(self, assets: list, period: str = "6Mo", weights: list = None):
        """
        Parameters
        ----------
        assets : list
            Identifiers given as list of strings.

        period : str (optional)
            Considered time period. Interval [period, now]. Resolution is 1 day.
            Allowed values: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo, 6Mo

        weights : list (optional)
            Weights of the assets given as float numbers. If not given, assets weights are equally distributed.

        """
        self.asset_names = assets
        self.period = period
        if len(assets) > 1:
            self.is_single_asset = False
            self.Tickers = yf.Tickers(self.asset_names)
        else:
            self.is_single_asset = True
            self.Tickers = yf.Ticker(self.asset_names[0])
        self.prices = self.get_prices()
        self.returns = self.get_returns()

        if weights is None:
            self.weights = [1.0 / len(assets) for _ in range(len(assets))]
        else:
            self.weights = weights

    def get_prices(self, price_time: str = "Close"):
        """Return price data."""
        tickers_hist = self.Tickers.history(period=self.period)[price_time]
        return tickers_hist

    def get_returns(self, price_time: str = "Close"):
        """Return all returns as inter-day price differences."""
        X = self.Tickers.history(period=self.period)[price_time].diff()
        if self.is_single_asset:
            X.rename("Returns", inplace=True)

        return X.iloc[1:]

    def weighted_prices(self, delete_unweighted=False):
        """Prices including weights."""
        X = self.prices.copy()
        for a, w in zip(self.asset_names, self.weights):
            X[a + "_weighted"] = X[a] * w

        if delete_unweighted:
            X.drop(self.asset_names, axis=1, inplace=True)
        return X

    def get_expected_return(self):
        """Expectation value of returns."""
        return self.returns.mean()

    def get_correlation_matrix(self, as_numpy=False):
        """Correlation matrix between all assets as numpy matrix."""
        if self.is_single_asset:
            raise NotImplementedError('No (auto-)correlation for a single asset.')
        if as_numpy:
            return self.returns.corr().to_numpy()
        else:
            return self.returns.corr()

    def get_weight_vector(self):
        return np.matrix(self.weights)

    def total_value(self):
        """Return the total value of the portfolio as time series."""
        return self.weighted_prices(delete_unweighted=True).sum(axis=1)

    def current_value(self):
        """Current value of the portfolio"""
        return self.total_value().iloc[-1]


if __name__ == "__main__":
    P = Portfolio(['AMZN', 'GOOG', 'WMT', 'TSLA', 'META'], weights=[2.5, 1, 1, 1, 1])
    #P = Portfolio(assets=['AMZN'])
    print(P.returns)

    # Variance of Portfolio return
    sigma = P.get_correlation_matrix(True)
    w = P.get_weight_vector()
    print(np.matmul(np.matmul(w, sigma), w.T))


