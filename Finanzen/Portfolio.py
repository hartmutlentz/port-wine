import yfinance as yf
import numpy as np
import pandas as pd
from scipy.linalg import inv
import Finanzen.Tools as tl
from pprint import pprint


class Portfolio:
    """
    Class for multiple assets data. Wrapper for yFinance-Tickers.

    """

    def __init__(self, assets, period: str = "6Mo"):
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
        # input cases
        if isinstance(assets, list):
            self.asset_names = assets
            self.weights = [1.0 / len(assets) for _ in range(len(assets))]
            self.amounts = [1 for _ in range(len(assets))]
        elif isinstance(assets, dict):
            x, y = tl.dict2lists(assets)
            self.asset_names = x
            if abs(sum(y) - 1.0) < 1.e-8:
                # weights-input is assumed as normalized to one
                self.weights = y
                self.amounts = self.estimate_amounts_from_weights()
            else:
                # weights-input is assumed as total numbers
                self.amounts = y
                self.weights = [float(i) / sum(self.amounts) for i in self.amounts]
        else:
            raise ImportError("Portfolio must be given as list or dict.")

        # basic properties
        self.period = period
        if len(assets) > 1:
            self.is_single_asset = False
            self.Tickers = yf.Tickers(self.asset_names)
        else:
            self.is_single_asset = True
            self.Tickers = yf.Ticker(self.asset_names[0])

        self.prices = self.get_prices()
        self.returns = self.get_returns()
        self.size = len(assets)

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
        """Returns the expectation value of returns for each asset separately."""
        return self.returns.mean()

    def excpected_risk(self):
        """Standard deviation for each asset."""
        return self.returns.var()

    def risk_profile(self):
        """"""
        df = pd.concat([self.expected_return(), self.excpected_risk()], axis=1)
        df.columns=["Expected Return", "Risk"]
        return df

    def correlation_matrix(self, as_numpy=True):
        """Correlation matrix between all assets as numpy matrix."""
        if self.is_single_asset:
            raise NotImplementedError('No (auto-)correlation for a single asset.')
        if as_numpy:
            return self.returns.corr().to_numpy()
        else:
            return self.returns.corr()

    def weight_vector(self):
        return np.array(self.weights)

    def expected_return_vector(self):
        return self.expected_return().to_numpy()

    def global_minimum_variance_portfolio(self):
        """Return the portfolio weights with global minimum.
            Global minimum means minimum variance with constraint that weight add up to one, but
            expected return is irrelevant.
        """
        sigma = self.correlation_matrix()
        S2 = np.vstack((2. * sigma, np.ones((1, sigma.shape[0]))))
        S2 = np.hstack((S2, np.ones((sigma.shape[0] + 1, 1))))
        S2[-1, -1] = 0.

        unit_vec = np.zeros((sigma.shape[0] + 1, 1))
        unit_vec[-1, 0] = 1.

        rslt = np.dot(inv(S2), unit_vec)[:-1]
        return rslt.reshape((self.size, ))

    def minimum_variance_portfolio(self, r):
        """Return weights for the minimum portfolio given a desired return rate r."""
        assert isinstance(r, float), "Return rate must be of float type."

        sigma = self.correlation_matrix()

        # stack weight vector on the right
        S2 = np.hstack((2. * sigma, P.weight_vector().reshape((P.size, 1))))

        # stack weight vector below. Define elongated weight vecor first.
        wt = np.append(P.weight_vector(), np.zeros((1, 1)))
        S2 = np.vstack((S2, wt))

        # stack ones-vector on the right
        S2 = np.hstack((S2, np.ones((sigma.shape[0] + 1, 1))))

        # stack ones-vector below
        S2 = np.vstack((S2, np.ones((1, sigma.shape[0] + 2))))

        # replace lower right block with zeros
        S2[-1, -1] = 0
        S2[-2, -2] = 0
        S2[-1, -2] = 0
        S2[-2, -1] = 0

        # compute result
        unit_vec = np.zeros((sigma.shape[0] + 2, 1))
        unit_vec[-1, 0] = 1.
        unit_vec[-2, 0] = r

        rslt = np.dot(inv(S2), unit_vec)[:-2]

        return rslt.reshape((self.size, ))

    def total_value(self):
        """Return the total value of the portfolio as time series."""
        return self.total_prices(delete_unweighted=True).sum(axis=1)

    def current_value(self):
        """Current value of the portfolio"""
        return self.total_value().iloc[-1]

if __name__ == "__main__":
    a = ['AMZN', 'GOOG', 'WMT', 'TSLA', 'META']
    weights = [1, 2, 3, 4, 4]
    weighted_assets = tl.lists2dict(a, weights)

    #a = ['AMZN', 'GOOG']
    P = Portfolio(assets=a)
    print(P.asset_names,'\n', P.weights, '\n', P.amounts)
    print(P.risk_profile())
    print(P.weight_vector())

    #P = Portfolio(assets=['AMZN'])
    #print(P.returns)

    # Variance of Portfolio return
    #sigma = P.correlation_matrix(True)
    #w = P.weight_vector()
    #print(np.matmul(np.matmul(w, sigma), w.T))

    #pprint(P.total_prices())
    #print(P.global_minimum_variance_portfolio())
    print(P.minimum_variance_portfolio(r=1.0))

