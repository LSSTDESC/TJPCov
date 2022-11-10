#!/usr/bin/python

from tjpcov.covariance_calculator import CovarianceCalculator


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute the covariance matrix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("INPUT", type=str, help="Input YAML data file")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="summary_statistics.sacc",
        help="Output file name",
    )
    args = parser.parse_args()

    cov = CovarianceCalculator(args.INPUT)
    cov.create_sacc_cov(args.output)
