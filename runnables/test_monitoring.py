import math
import sys

import numpy as np
import matplotlib.pyplot as plt
import time

from torch.utils.tensorboard import SummaryWriter


def generate_loss(x:float,loss_name:str):
    if loss_name == 'train':
        loss = 20/(x+0.2)+np.random.randn()*0.1
    elif loss_name == 'test':
        loss = 30 /(x+0.2) + np.random.randn() * 0.1
    else:
        loss = math.tanh(x)+ np.random.randn() * 0.1

    # data_write = f"{x} {loss}\n"
    # with open(f'E:\pyCausalVision\log\simultrain/{loss_name}.txt', 'a', encoding='utf-8') as file:
    #     file.write(data_write)
    return loss


if __name__ == '__main__':
    from utils.train_plot import PlotWriter

    x =np.arange(start=0, stop=20, step=1)
    print('training starts ************')

    writer= PlotWriter(log_dir='../log', experiment='test_plot')

    for epoch in range(0,100):
        for b in x:
            time.sleep(3)
            loss_train = generate_loss(b,"train")
            writer.write(curve_name='loss_train',y=round(loss_train,3),x=epoch*20+b,epoch=epoch,batch=b)
            print('training finished')

            time.sleep(3)
            loss_test = generate_loss(b, "test")
            writer.write(curve_name='loss_test',y=round(loss_test,3),x=epoch*20+b,epoch=epoch,batch=b)
            print('test finished')

            time.sleep(3)
            acc = generate_loss(b, "acc")
            writer.write(curve_name='acc',y=round(acc,3),x=epoch*20+b,epoch=epoch,batch=b)
            print('accuracy finished')
    # tp.finish()
    print('training ends ************')



