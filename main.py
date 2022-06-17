from model import reader, trainModel, monitor, splitter
from model.drift import drift
import warnings

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    no_of_batches = splitter.split_into_batches(500)
    reader.initialize()
    reader.process_data()
    trainModel.train_new()
    for batch_no in range(0, no_of_batches):
        monitor.monitor(batch_no)
        if drift.should_retrain('rain') or drift.should_retrain('humidity'):
            print(f'Model is being retrained on batch {batch_no}')
            trainModel.train_new()
        else:
            print(f'On batch {batch_no} the model will not be retrained')

