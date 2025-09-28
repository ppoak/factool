# Factor

This is a framework for factor mining and evaluating. The data backend is from anthor quantative tool repository named `parquool`, see [parquool](https://github.com/ppoak/parquool) for more information.

## Installation

You can simple click download file in zip format on github, or you can use git command line if you want:

```
git clone git@github.com:ppoak/factool
```

## Quick Start

### Factor Source

Factor source is managed by `parquool.DuckParquet`, which provides a easy and simple way to manage factor data completely offline by your own computer. DuckParquet provides a series of sql-like interfaces, you can simple create a DuckParquet directory by using `DuckParquet(path).upsert_from_df(df)`. And you can also select with `where` clause, `order by` clause, etc.

### Factor Computing

You can simply create a function starts with `calc_` and add the real factor script name to it in a python file. For example, if you want to create a `market_size` function, which calculate the `log market size` factor and `non-linear market size` factor. You can just create one script named `market_size.py` with one function called `calc_market_size`. Then by running `calc` function in `calc.py`, pass the path to the `market_size.py` to it, it will compute and save the result to the directory where your environment variable `FACTOR_DATA_PATH` points to.

### Factor Evaluating

To evaluate a factor, you need to tell the `evaluate` funciton where your factor `DuckParquet` path is and your k-line data for computing returns of the market. Moreover, if you want to see the performance of the benchmark, you can also assign the benchmark code to the function. If the parameters above are not set, the funciont will automatically find them in environment variable: `QUOTESDAY_PATH`, `INDEXQUOTESDAY_PATH`. And the result output will be `EVAL_PATH`. The result is composed with one excel file with two sheets, one for ic time series, and one for evaluation result. And the other result is ic time-series image and net value of grouping result with benchmark value if set.

### Automation

All the processes above can be set in an automatic way. The `scritp/agent.py` and `script/web.py` file provide a terminal and web interface for using AI models in calculating and evaluating factors.

For the cli, you can simply input a factor definition markdown file path to the command. All the process will be run automatically. As for the web, just explore it!
