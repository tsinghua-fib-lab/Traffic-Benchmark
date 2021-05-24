import torch.optim as optim
import math
from net import *
import util


class Trainer():
    def __init__(self,
                 model,
                 lrate,
                 wdecay,
                 clip,
                 step_size,
                 seq_out_len,
                 scaler,
                 device,
                 cl=True,
                 new_training_method=False):
        self.scaler = scaler
        self.model = model
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lrate,
                                    weight_decay=wdecay)
        self.loss = util.masked_mae
        self.clip = clip
        self.step = step_size

        self.iter = 0
        self.task_level = 1
        self.seq_out_len = seq_out_len
        self.cl = cl
        self.new_training_method = new_training_method

    def train(self, input, real_val, ycl, idx=None, batches_seen=None):
        self.iter += 1

        if self.iter % self.step == 0 and self.task_level < self.seq_out_len:
            self.task_level += 1
            if self.new_training_method:
                self.iter = 0

        self.model.train()
        self.optimizer.zero_grad()
        if self.cl:
            output = self.model(input,
                                idx=idx,
                                ycl=ycl,
                                batches_seen=self.iter,
                                task_level=self.task_level)
        else:
            output = self.model(input,
                                idx=idx,
                                ycl=ycl,
                                batches_seen=self.iter,
                                task_level=self.seq_out_len)

        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)

        if self.cl:

            loss = self.loss(predict[:, :, :, :self.task_level],
                             real[:, :, :, :self.task_level], 0.0)
            mape = util.masked_mape(predict[:, :, :, :self.task_level],
                                    real[:, :, :, :self.task_level],
                                    0.0).item()
            rmse = util.masked_rmse(predict[:, :, :, :self.task_level],
                                    real[:, :, :, :self.task_level],
                                    0.0).item()
        else:
            loss = self.loss(predict, real, 0.0)
            mape = util.masked_mape(predict, real, 0.0).item()
            rmse = util.masked_rmse(predict, real, 0.0).item()

        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()

        return loss.item(), mape, rmse

    def eval(self, input, real_val, ycl):
        self.model.eval()
        with torch.no_grad():
            output = self.model(input, ycl=ycl)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse