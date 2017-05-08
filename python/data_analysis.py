import matplotlib.pyplot as plt
import kddcup

def graph_distributions(data):
    data[data.dtypes[(data.dtypes=="float64")|(data.dtypes=="int64")].index.values].hist(figsize=[20,20])
    plt.show()

if __name__ == "__main__":
    dataset = kddcup.load_data_10_percent()
    graph_distributions(dataset.data)