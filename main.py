# Press the green button in the gutter to run the script.
from model import readModel, trainModel, evaluateModel, monitoringModel, splitter, appendingData
from model.drift import drift

if __name__ == '__main__':
    #splitting second file
    splitter.splitting(500)
    #first read
    readModel.initialize()
    readModel.read()
    #train
    trainModel.train()
    # evaluateModel.evaluate()
    for batch_no in range(0, 112):
        monitoringModel.monitoring(batch_no)
        # updateTrainingData(batch_no)
        #for...
        #appendingData.read_and_append('data/cutted/filename.csv')
        #trainModel.train()?
        if drift.should_retrain('rain') or drift.should_retrain('humidity'):
            print(f'Model is being retrained on batch {batch_no}')
            trainModel.train()
        else:
            print(f'On batch {batch_no} model will not be retrained')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
