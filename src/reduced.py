import pandas as pd
import random as rand


#
# The following script reduces the dataset, proportionally to 'target' ratio
#

CSV_PATH = 'data/raw.csv'
OUT_PATH = 'data/reduced.csv'


def main():
    data = open(CSV_PATH, 'r')
    output = open(OUT_PATH, 'w')

    header = data.readline()
    output.write(header)

    for line in data:
        if rand.random() < 0.05:
            output.write(line)

    output.close()
    data.close()


if __name__ == '__main__':
    main()
