import argparse
from pathlib import Path
from .update import update_factor


def update_subcommand(args):
    path = Path(args.path)
    update_factor(
        path=path,
        name=args.name,
        begin=args.begin,
        end=args.end,
        partition=args.partition,
        n_jobs=args.n_jobs,
    )


def main():
    parser = argparse.ArgumentParser(
        description="FactorLab Command Line Tool",
        usage="factorlab <command> [<args>]",
    )

    subparsers = parser.add_subparsers(help="Available commands")

    parser_update = subparsers.add_parser("update", help="Update factor data")
    parser_update.add_argument("--path", type=str, required=True, help="Path to factor data")
    parser_update.add_argument("--name", type=str, required=True, help="Factor name to update")
    parser_update.add_argument("--begin", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser_update.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser_update.add_argument("--partition", type=str, default="month", help="Partitioning method (default: 'month')")
    parser_update.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs (default: 1)")
    parser_update.set_defaults(func=update_subcommand)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
