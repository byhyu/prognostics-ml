import matplotlib.pyplot as plt
import featureeng

def presentData(data_frame, columns=[]):
    if isinstance(data_frame, featureeng.Frame):
        data_frame = data_frame.get_panda_frame()

    indices = range(len(data_frame.index))
    plt.title('Chart')
    for column in columns:
        data = map(float, list(data_frame[column]))
        plt.plot(indices, data)

    plt.legend(columns, loc='upper right')
    plt.show()

def saveChart(pandas_frame, columns=[], file_name='figure.png'):
    indices = range(len(pandas_frame.index))
    plt.title('Chart')
    for column in columns:
        data = map(float, list(pandas_frame[column]))
        plt.plot(indices, data)

    plt.legend(columns, loc='upper left')
    plt.savefig(file_name)
    plt.close()


