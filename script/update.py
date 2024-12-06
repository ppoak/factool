from pathlib import Path
from factorlab import Factor, quotes_day


def update_factor(
    path: Path,
    name: str,
    begin: str,
    end: str,
    partition: str = "month",
    n_jobs: int = 1,
) -> None:
    factor = Factor(path, partition=partition)
    trading_days = quotes_day.read(
        index="date", code="000001.XSHG", date__ge=begin, date__le=end
    ).index.sort_values()
    data = factor.get(name=name, trading_days=trading_days, n_jobs=n_jobs)
    data = data.stack().reset_index()
    data.columns = ["date", "code", "value"]
    data["name"] = name
    factor.upsert(data, partition=lambda x: x["date"].dt.strftime("%Y-%m"))


if __name__ == "__main__":
    path = Path("data/price_volume")
    begin = "2010-01-01"
    end = "2024-12-05"
    name = "volume_weighted_price"
    partition = "month"
    n_jobs = -1
    update_factor(path=path, name=name, begin=begin, end=end, partition=partition, n_jobs=n_jobs)