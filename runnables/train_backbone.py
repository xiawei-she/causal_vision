import os.path
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=UserWarning)
    from torch.utils.data import DataLoader, random_split
    import torch.nn as nn
    import torch.optim as optim
    import torch
    from models.backbone import Backbone
    from preprocessing.dataset import CarDataset
    from utils.train_plot import PlotWriter

if __name__ == '__main__':
    print("------------------------Preparation------------------------")
    # writer for log of training process
    writer = PlotWriter(log_dir='../log', experiment='backbone_cnn')
    # save check point
    model_name = 'backbone_cnn'
    checkpoint_dir = os.path.join('../checkpoints', model_name+'.pt')

    # path of data folder
    print("preparing data......")
    data_path = 'E:/Projet_CausalVision/images/classification/total'
    car_data = CarDataset(data_dir=data_path)
    train_dataset, test_dataset = random_split(car_data, [91, 20])
    print(f'Training data volume = {train_dataset.__len__()}, Test data volume = {test_dataset.__len__()}, ')
    batch_size_train = 2
    batch_size_test = 2
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=4,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True, num_workers=4, pin_memory=True)

    examples = iter(train_dataloader)
    example = next(examples)

    backbone_type = "cnn"
    print(f"load backbone, type = {backbone_type}......")
    model_backone1 = Backbone(backbone_type=backbone_type)

    # definition of loss_function
    criterion = nn.BCELoss()
    # definition of optimizer
    optimizer = optim.SGD(model_backone1.parameters(), lr=0.001, momentum=0.9)

    # check if there are avalaible GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    #### training process ####
    print("------------------------start training------------------------")
    epoch_num = 2000
    for epoch in range(epoch_num):
        print(f'training and testing Epoch {epoch}......')
        model_backone1.train()
        for b, data in enumerate(train_dataloader):
            inputs, targets = data
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model_backone1(inputs).squeeze(-1)
            if torch.cuda.is_available():
                outputs = outputs.cuda()
                targets = targets.cuda()
            # print(outputs)
            # print(targets)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            print(f'[Epoch: {epoch + 1}, Batch: {b + 1}]-->training_loss: {loss.item():.3f}')
            writer.write(curve_name='training_loss', y=round(loss.item(),3), x=epoch * len(train_dataloader) + b,
                         epoch=epoch,batch=b)

        model_backone1.eval()
        with torch.no_grad():
            for b_test, data_test in enumerate(test_dataloader):
                images, labels = data_test
                outputs = model_backone1(images).squeeze(-1)
                if torch.cuda.is_available():
                    outputs = outputs.cuda()
                    targets = targets.cuda()
                loss = criterion(outputs, targets)
                print(f'[Epoch: {epoch + 1}, Batch: {b_test + 1}]-->test_loss: {loss.item():.3f}')
                writer.write(curve_name='test_loss', y=round(loss.item(),3), x=epoch * len(test_dataloader) + b_test,
                             epoch=epoch, batch=b_test)
        # save model
        if epoch % epoch_num/10 == 0:
            torch.save(model_backone1.state_dict(), 'model.pth')
        print(f'Finished Epoch {epoch}')