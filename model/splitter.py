import pandas as pd


def split_into_batches(batch_size):
    in_csv = 'data/weatherAUS_to_cut.csv'
    number_lines = sum(1 for row in (open(in_csv)))
    for i in range(0, number_lines, batch_size):
        df = pd.read_csv(in_csv,
                         nrows=batch_size,
                         skiprows=i)
        out_csv = 'data/batches/weatherAUS_' + str(int(i / batch_size)) + '.csv'

        df.to_csv(out_csv,
                  index=False,
                  header=True,
                  mode='w',
                  chunksize=batch_size)

    return number_lines // batch_size