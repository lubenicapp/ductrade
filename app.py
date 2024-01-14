import argparse
from core.models.model import Model


def main():
    parser = argparse.ArgumentParser(description="Stock Prediction and Plotting")
    parser.add_argument("stock_symbol", type=str, help="Stock symbol (e.g., TSLA)")
    args = parser.parse_args()

    stock_symbol = args.stock_symbol

    model = Model(stock_symbol)
    print(model.predict())
    model.plot()


if __name__ == "__main__":
    main()
