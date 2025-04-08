import numpy as np
import pandas as pd
from logging import Logger
from quool import setup_logger
from quool import Evaluator as QuoolEvaluator


class Evaluator:

    def __init__(
        self, factor: pd.DataFrame, price: pd.DataFrame, logger: Logger = None
    ):
        self._factor = factor
        self._price = price
        self._logger = logger or setup_logger("FactorEvaluator", level="DEBUG")
        if (_factor_diff := self._factor.index.difference(self._price.index)).size:
            self._logger.warning(
                f"Index {_factor_diff} in factor without price, these dates will be dropped"
            )
            self._factor = self._factor.drop(index=_factor_diff)
        if (_price_diff := self._price.index.difference(self._factor.index)).size:
            self._logger.warning(
                f"Index {_price_diff} in price without factor, these dates will apply latest factor"
            )
            self._factor = self._factor.reindex(index=self._price.index, method="ffill")
        self._shifted = self._price.shift(-1)

    def calc_info_coef(self, freq: int = 1, method: str = "spearman"):
        future = self._shifted.shift(-freq) / self._shifted - 1
        self.ic = self._factor.corrwith(future, axis=1, method=method)
        self._direction = np.sign(self.ic.mean())
        return self

    def calc_topk_returns(
        self,
        k: int = 100,
        freq: int = 1,
        weight: pd.DataFrame = None,
        feasible: pd.DataFrame = None,
        benchmark: pd.Series = None,
        commission: float = 0.0005,
    ):
        direction = getattr(self, "_direction", 1)
        future = self._shifted.shift(-freq) / self._shifted - 1
        rank = (self._factor * direction).rank(axis=1, ascending=False)
        feasible = (
            feasible
            if feasible is not None
            else pd.DataFrame(
                np.ones_like(future),
                index=future.index,
                columns=future.columns,
            )
        )
        weight = (
            weight
            if weight is not None
            else pd.DataFrame(
                np.ones_like(future),
                index=future.index,
                columns=future.columns,
            )
        )
        weight = weight.where(feasible, 0)
        # select top k weight
        weight = weight.where(rank <= k, 0)
        # unify weight, divide into freq parts to avoid difference when starting point differs
        weight = weight.div(weight.sum(axis=1), axis=0) / freq
        # calculate turnover
        turnover = weight.diff(freq).abs()
        turnover.iloc[:freq] = weight.iloc[:freq]
        turnover = turnover.sum(axis=1)
        # calculate returns
        returns = (future * weight).sum(axis=1) - commission * turnover
        self.value_topk = (returns.shift(1 + freq).fillna(0) + 1).cumprod()
        self.turnover_topk = turnover
        self.evaluation_topk = QuoolEvaluator.evaluate(
            self.value_topk, benchmark, turnover
        )
        return self

    def calc_ngroup_returns(
        self,
        n: int = 10,
        freq: int = 1,
        weight: pd.DataFrame = None,
        feasible: pd.DataFrame = None,
        benchmark: pd.Series = None,
        commission: float = 0.0005,
    ):
        direction = getattr(self, "_direction", 1)
        try:
            groups = (self._factor * direction).apply(
                lambda x: pd.qcut(x, n, labels=False, duplicates="raise"), axis=1
            ) + 1

            future = self._shifted.shift(-freq) / self._shifted - 1
            feasible = (
                feasible
                if feasible is not None
                else pd.DataFrame(
                    np.ones_like(future),
                    index=future.index,
                    columns=future.columns,
                )
            )
            weight = (
                weight
                if weight is not None
                else pd.DataFrame(
                    np.ones_like(future), index=future.index, columns=future.columns
                )
            )
            weight = weight.where(feasible, 0)

            values = []
            turnovers = []
            evaluations = []
            for i in range(1, n + 1):
                # filter weight
                _weight = weight.where(groups == i, 0)
                _weight = _weight.div(_weight.sum(axis=1), axis=0) / freq
                # calculate turnover
                turnover = _weight.diff(freq).abs()
                turnover.iloc[:freq] = _weight.iloc[:freq]
                turnover = turnover.sum(axis=1) / freq
                # calculate returns
                returns = (future * _weight).sum(axis=1) - commission * turnover
                value = (returns.shift(1 + freq).fillna(0) + 1).cumprod()
                values.append(value)
                turnovers.append(turnover)
                evaluations.append(
                    QuoolEvaluator.evaluate(
                        value, benchmark=benchmark, turnover=turnover
                    )
                )

            self.value_ngroup = pd.concat(
                values, axis=1, keys=[f"Group_{i}" for i in range(1, n + 1)]
            )
            self.turnover_ngroup = pd.concat(
                turnovers, axis=1, keys=[f"Group_{i}" for i in range(1, n + 1)]
            )
            self.evaluation_ngroup = pd.concat(
                evaluations, axis=1, keys=[f"Group_{i}" for i in range(1, n + 1)]
            )

        except Exception as e:
            for idx, row in self._factor.iterrows():
                try:
                    pd.qcut(row, n, labels=False, duplicates="raise")
                except Exception as e:
                    self._logger.critical(f"Error on {idx}: {e}")
            self.value_ngroup = pd.DataFrame(
                np.ones((self._price.shape[0], n)),
                index=self._price.index,
                columns=[f"Group_{i}" for i in range(1, n + 1)],
            )
            self.turnover_ngroup = pd.DataFrame(
                np.zeros((self._price.shape[0], n)),
                index=self._price.index,
                columns=[f"Group_{i}" for i in range(1, n + 1)],
            )
            self.evaluation_ngroup = pd.DataFrame()

        return self

    def __call__(
        self,
        method: str = "spearman",
        n: int = 10,
        k: int = 100,
        freq: int = 1,
        weight: pd.DataFrame = None,
        feasible: pd.DataFrame = None,
        benchmark: pd.Series = None,
        commission: float = 0.0005,
    ):
        return (
            self.calc_info_coef(freq, method)
            .calc_topk_returns(k, freq, weight, feasible, benchmark, commission)
            .calc_ngroup_returns(n, freq, weight, feasible, benchmark, commission)
        )

    def __str__(self):
        return (
            "Factor Evaluator(\n"
            f"\tdirection: {self._direction}\n"
            f"\tinfo_coef: \n{self.ic}\n"
            f"\tvalue_topk: \n{self.value_topk}\n"
            f"\tturnover_topk: \n{self.turnover_topk}\n"
            f"\tevaluation_topk: \n{self.evaluation_topk}\n"
            f"\tvalue_ngroup: \n{self.value_ngroup}\n"
            f"\tturnover_ngroup: \n{self.turnover_ngroup}\n"
            f"\tevaluation_ngroup: \n{self.evaluation_ngroup}\n"
            ")"
        )

    def __repr__(self):
        return self.__str__()
