import data
import model

if __name__ == "__main__":
    print("Building Model")
    m = model.Model()
    m.construct()

    # Get training and testing data
    data_set = 'Misc'
    train_path = './Datasets/' + data_set + '_Data/Train/'
    test_path = './Datasets/' + data_set + '_Data/Test/'
    output_path = './Results/' + data_set + '/'
    train_data = data.Data(train_path)
    test_data = data.Data(test_path)

    # Train and test model
    m.train(train_data)

    print("Done training model")
    print("Testing model")
    m.test(test_data, output_path)
    print("Done Testing Model")
