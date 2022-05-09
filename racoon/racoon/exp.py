import argparse
from racoon.experiment import ExpManager

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")

    args = parser.parse_args()

    manger = ExpManager()

    manger.int_exp_dir(
        name=args.name
    )

if __name__ == '__main__':
    main()