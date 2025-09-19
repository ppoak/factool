import numpy as np
import pandas as pd
from logging import Logger
from parquool import setup_logger
from joblib import Parallel, delayed
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

    def evaluate_info_coef(self, freq: int = 1, method: str = "spearman"):
        future = self._shifted.shift(-freq) / self._shifted - 1
        self.ic = self._factor.corrwith(future, axis=1, method=method)
        self._direction = np.sign(self.ic.mean())
        return self

    def evaluate_topk(
        self,
        k: int = 100,
        freq: int = 1,
        rebalance: bool = True,
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
                dtype="bool",
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
        weight = weight.div(weight.sum(axis=1), axis=0)
        if rebalance:
            self.topk_result = QuoolEvaluator.evaluate_rebalance(
                weight,
                self._price,
                freq,
                benchmark,
                commission,
            )
        else:
            self.topk_result = QuoolEvaluator.evaluate_index(
                weight.shift(1).iloc[1:], self._price, freq, benchmark, commission
            )
        return self

    def evaluate_ngroup(
        self,
        n: int = 10,
        freq: int = 1,
        rebalance: bool = True,
        weight: pd.DataFrame = None,
        feasible: pd.DataFrame = None,
        benchmark: pd.Series = None,
        commission: float = 0.0005,
        n_jobs: int = -1,
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
                    dtype="bool",
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

            def _calc_group(i: int):
                # filter weight
                _weight = weight.where(groups == i, 0)
                _weight = _weight.div(_weight.sum(axis=1), axis=0)
                if rebalance:
                    return QuoolEvaluator.evaluate_rebalance(
                        _weight,
                        self._price,
                        freq,
                        benchmark,
                        commission,
                    )
                return QuoolEvaluator.evaluate_index(
                    _weight.shift(1).iloc[1:],
                    self._price,
                    freq,
                    benchmark,
                    commission,
                )

            self.ngroup_result = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_calc_group)(i) for i in range(1, n + 1)
            )

        except ValueError as e:
            for idx, row in self._factor.iterrows():
                try:
                    pd.qcut(row, n, labels=False, duplicates="raise")
                except Exception as e:
                    self._logger.critical(f"Error on {idx}: {e}")
            self.ngroup_result = []

        return self

    def __call__(
        self,
        method: str = "spearman",
        n: int = 10,
        k: int = 100,
        freq: int = 1,
        rebalance: bool = True,
        weight: pd.DataFrame = None,
        feasible: pd.DataFrame = None,
        benchmark: pd.Series = None,
        commission: float = 0.0005,
    ):
        return (
            self.evaluate_info_coef(freq, method)
            .evaluate_topk(k, freq, rebalance, weight, feasible, benchmark, commission)
            .evaluate_ngroup(
                n, freq, rebalance, weight, feasible, benchmark, commission
            )
        )

    def __str__(self):
        return (
            "Factor Evaluator(\n"
            f"\tdirection: {self._direction}\n"
            f"\tinfo_coef: \n{self.ic}\n"
            f"\ttopk_result: \n{self.topk_result}\n"
            f"\tngroup_result: \n{self.ngroup_result}\n"
            ")"
        )

    def __repr__(self):
        return self.__str__()
