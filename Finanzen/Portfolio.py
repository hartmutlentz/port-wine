import yfinance as yf
import numpy as np
from Asset import Asset
from pprint import pprint


class Portfolio:
    """
    Class for multiple assets data. Wrapper for yFinance-Tickers.

    """

    def __init__(self, assets: list, weights_amounts: list = None, period: str = "6Mo"):
        """
        Parameters
        ----------
        assets : list
            Identifiers given as list of strings.

        period : str (optional)
            Considered time period. Interval [period, now]. Resolution is 1 day.
            Allowed values: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo, 6Mo

        weights_amounts : list (optional)
            Weights of the assets given as float numbers. If not given, assets weights are equally distributed.

        """
        # basic properties
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

        # weights and amounts
        if weights_amounts:
            if abs(sum(weights_amounts) - 1.0) < 1.e-8:
                self.weights = weights_amounts
                self.amounts = self.estimate_amounts_from_weights()
            else:
                self.amounts = weights_amounts
                self.weights = [float(i) / sum(self.amounts) for i in self.amounts]
        else:
            self.weights = [1.0 / len(assets) for _ in range(len(assets))]
            self.amounts = [1 for _ in range(len(assets))]


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

    def total_prices(self, delete_unweighted=False):
        """Prices including amounts."""
        X = self.prices.copy()
        for a, w in zip(self.asset_names, self.amounts):
            X[a + "_total"] = X[a] * w

        if delete_unweighted:
            X.drop(self.asset_names, axis=1, inplace=True)
        return X

    def estimate_amounts_from_weights(self, x):
        """Given a list of weights, return a list of integers, such that the weight remains the same."""
        factor = round(1.0/min(x))
        amounts = [i * factor for i in x]
        return amounts

    def expected_return(self):
        """Expectation value of returns."""
        return self.returns.mean()

    def correlation_matrix(self, as_numpy=False):
        """Correlation matrix between all assets as numpy matrix."""
        if self.is_single_asset:
            raise NotImplementedError('No (auto-)correlation for a single asset.')
        if as_numpy:
            return self.returns.corr().to_numpy()
        else:
            return self.returns.corr()

    def weight_vector(self):
        return np.matrix(self.weights)

    def total_value(self):
        """Return the total value of the portfolio as time series."""
        return self.total_prices(delete_unweighted=True).sum(axis=1)

    def current_value(self):
        """Current value of the portfolio"""
        return self.total_value().iloc[-1]


if __name__ == "__main__":
    a = ['AMZN', 'GOOG', 'WMT', 'TSLA', 'META']
    a = ['AMZN', 'GOOG']
    P = Portfolio(assets=a, weights_amounts=[2.5, 1, 1, 1, 1])
    #P = Portfolio(assets=['AMZN'])
    print(P.returns)

    # Variance of Portfolio return
    sigma = P.correlation_matrix(True)
    w = P.weight_vector()
    #print(np.matmul(np.matmul(w, sigma), w.T))

    pprint(P.total_prices())


