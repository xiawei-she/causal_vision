import time

import matplotlib.pyplot as plt
import os


class PlotWriter:

    def __init__(self, log_dir,experiment:str):
        self.experiment = experiment
        self.log_dir = os.path.join(log_dir, self.experiment)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            print("log folder has been built")
        else:
            print("log folder already exists")
            file_list = os.listdir(self.log_dir)
            for file in file_list:
                file_path = os.path.join(self.log_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    def write(self,curve_name:str,x:int,y:float,epoch:int,batch:int):
        file_path = os.path.join(self.log_dir,curve_name+'.txt')
        data = str(x)+' '+str(y)+' '+str(epoch)+' '+str(batch)+'\n'
        # 检查文件是否存在
        if os.path.exists(file_path) and os.path.isfile(file_path):
            # 文件存在，写入字符串 "aaa"
            with open(file_path, 'a') as file:  # 使用 'a' 模式追加内容
                file.write(data)  # 追加写入 "aaa" 并换行，你可以根据需要修改是否换行
        else:
            # 文件不存在，创建文件并写入字符串 "aaa"
            with open(file_path, 'w') as file:  # 使用 'w' 模式写入内容
                file.write(data)


class TrainPlot:

    def __init__(self,graph_dir,log_dir,experiment:str,xlabel:str,xlim:float,update_freq:float):

        self.experiment = experiment
        self.graph_dir = os.path.join(graph_dir,self.experiment)
        if not os.path.exists(self.graph_dir):
            os.makedirs(self.graph_dir)
            print("graph folder has been built")
        self.log_dir = os.path.join(log_dir,self.experiment)
        self.xlabel = xlabel
        self.xlim = xlim
        self.update_freq = update_freq

    def _plot_curve_(self,data:str,curve_name:str,fig_idx,linewidth):
        items = data.split('\n')
        x_array = []
        y_array = []
        epoch = None
        batch = None
        step = None
        count = 0
        for item in items:
            count += 1
            if len(item) > 0:
                x_array.append(int(item.split(' ')[0]))
                y_array.append(float(item.split(' ')[1]))
            if count == len(items)-1:
                step = int(item.split(' ')[0])
                epoch = int(item.split(' ')[2])
                batch = int(item.split(' ')[3])
        graph_name = self.experiment+'_'+curve_name+'_epoch:'+str(epoch)+'_batch:'+str(batch)+'_step:'+str(step)
        plt.figure(fig_idx)
        plt.grid()
        plt.title(graph_name)
        plt.xlabel(self.xlabel)
        plt.ylabel(curve_name)
        plt.plot(x_array,y_array,color='red',linewidth=linewidth)
        graph_path = os.path.join(self.graph_dir,curve_name)
        plt.savefig(f'{graph_path}.png')

    def _delete_curve_(self,curve_name:str):
        file_list = os.listdir(self.graph_dir)
        if len(file_list) > 0:
            for file_name in file_list:
                if curve_name in file_name:
                    file_path = os.path.join(self.graph_dir, file_name)
                    try:
                        os.remove(file_path)
                        break
                    except Exception:
                        print("this graph is being using")
                        pass

    def _plot_(self):
        if not os.path.exists(self.log_dir):
            raise Exception('log dir does not exist')
        else:
            file_list = os.listdir(self.log_dir)
            fig_idx = 0
            for file_name in file_list:
                file_path = os.path.join(self.log_dir, file_name)
                with open(file_path, "r") as f:
                    data = f.read()
                curve_name = file_name.split('.')[0]
                self._delete_curve_(curve_name)
                self._plot_curve_(data=data,curve_name=curve_name,fig_idx=fig_idx,linewidth=1.5)
                fig_idx+=1
        return

    def start_plot(self):
        print('-------start_ploting------')
        while True:
            self._plot_()
            time.sleep(self.update_freq)





