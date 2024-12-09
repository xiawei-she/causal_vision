from utils.train_plot import TrainPlot

if __name__ == '__main__':
    graph_dir = '../graph'
    log_dir = '../log'
    experiment = 'backbone_cnn'
    xlabel='step'
    update_freq=5
    train_plot = TrainPlot(graph_dir=graph_dir,
                           log_dir=log_dir,
                           experiment=experiment,
                           xlabel=xlabel,
                           xlim=None,
                           update_freq=update_freq)

    train_plot.start_plot()
    '''
    graph_dir = '../graph'
    log_dir = '../log'
    experiment = 'test_plot'
    xlabel='step'
    update_freq=5
    train_plot = TrainPlot(graph_dir=graph_dir,
                           log_dir=log_dir,
                           experiment=experiment,
                           xlabel=xlabel,
                           xlim=None,
                           update_freq=update_freq)

    train_plot.start_plot()
    # '''