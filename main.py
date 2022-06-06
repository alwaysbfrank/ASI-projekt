

# Press the green button in the gutter to run the script.
from model import readModel, trainModel, evaluateModel, monitoringModel, splitter, appendingData
from model.drift import drift

if __name__ == '__main__':
    #splitting second file
    splitter.splitting(500)
    #first read
    readModel.read()
    #train
    trainModel.train()
    evaluateModel.evaluate()
    monitoringModel.monitoring()
    #for...
    #appendingData.read_and_append('data/cutted/filename.csv')
    #trainModel.train()?
    should_retrain = drift.should_retrain()
    print(f'drift evaluation suggested that should retrain is {should_retrain}')
    if should_retrain:
        trainModel.train()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
