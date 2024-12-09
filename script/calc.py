from pathlib import Path
from factorlab import Factor, quotes_day


def calc_factor(
    path: Path,
    name: str,
    begin: str,
    end: str,
    partition: str = "month",
    njobs: int = 1,
) -> None:
    factor = Factor(path, partition=partition)
    trading_days = quotes_day.get_trading_days(begin, end)
    data = factor.calc(name=name, trading_days=trading_days, njobs=njobs)
    data = data.stack().reset_index()
    data.columns = ["date", "code", "value"]
    data["name"] = name
    factor.upsert(data, partition=lambda x: x["date"].dt.strftime("%Y-%m"))


if __name__ == "__main__":
    path = Path("data/price_volume")
    begin = "2024-01-01"
    end = "2024-12-05"
    name = "volume_weighted_price"
    partition = "month"
    njobs = -1
    calc_factor(
        path=path,
        name=name,
        begin=begin,
        end=end,
        partition=partition,
        njobs=njobs
    )