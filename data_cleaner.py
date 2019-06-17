import pandas as pd

CSV_PATH="./data/raw.csv"
X_CSV_PATH = "./data/x_train.csv"
Y_CSV_PATH = "./data/y_train.csv"


def main():
    raw_data = pd.read_csv(CSV_PATH)
    raw_data.drop(raw_data.index[0])
    raw_data.drop(['ID_code', 'target'], axis=1)
    raw_data.to_csv(X_CSV_PATH)

    raw_data = pd.read_csv(CSV_PATH)
    raw_data[['target']]
    raw_data.to_csv(Y_CSV_PATH)


if __name__ == "__main__":
    main()
