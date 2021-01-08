# 
# Local client training in FedReID for decentralised person re-identification
#

import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from lib.kl_loss import KLLoss

import time
from tqdm import tqdm

import pdb


# local client dataset partition
class DatasetSplitLM(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]

# local client model training and current model evaluation
class LocalUpdateLM(object):
    def __init__(self, args, dataset, idxs, nround, user):
        self.args = args
        self.round = nround
        self.user = user
        self.data_loader = DataLoader(DatasetSplitLM(dataset, list(idxs)), batch_size=self.args.batchsize, shuffle=True, num_workers=4)
        self.criterion_ce = nn.CrossEntropyLoss() # local supervision
        self.criterion_kl = KLLoss().cuda() # server supervision
        #shitong
        self.criterion_mse = nn.MSELoss()

    # updating local model parameters
    def update_weights(self, model, cur_epoch, idx_client, model_sv):
        # mapping network parameters
        # when local epoch > 1, the copy server model should also be updated shared the merit of mutual learning
        ignored_params = list(map(id, model.model.fc.parameters() )) + list(map(id, model.classifier_1.parameters() ))\
                            + list(map(id, model.classifier_2.parameters())) + list(map(id, model.classifier_3.parameters() ))\
                            + list(map(id, model.classifier_4.parameters()))
        ignored_params_sv = list(map(id, model_sv.model.fc.parameters() )) + list(map(id, model_sv.classifier_1.parameters() ))\
                            + list(map(id, model_sv.classifier_2.parameters())) + list(map(id, model_sv.classifier_3.parameters() ))\
                            + list(map(id, model_sv.classifier_4.parameters()))


        # feature embedding network parameters
        # shitong filter: first parameter: return True/False; filter will choose from the second parameter which satisfy the first
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        base_params_sv = filter(lambda p: id(p) not in ignored_params_sv, model_sv.parameters())

        # optimiser setting
        # global LR sheduler, hyperparameters can be fine-tuned
        # lr_init: 0.01 for embedding network and 0.1 for mapping network
        decay_factor = 1.0 if cur_epoch < 20 else 0.1
       #shitong
        args=self.args
        optimizer = optim.SGD([
             {'params': base_params, 'lr': args.lr_init*decay_factor},
             {'params': model.model.fc.parameters(), 'lr': args.lr_init*10*decay_factor},
             {'params': model.classifier_1.parameters(), 'lr': args.lr_init*10*decay_factor},
             {'params': model.classifier_2.parameters(), 'lr': args.lr_init*10*decay_factor},
             {'params': model.classifier_3.parameters(), 'lr': args.lr_init*10*decay_factor},
             {'params': model.classifier_4.parameters(), 'lr': args.lr_init*10*decay_factor},
             {'params': base_params_sv, 'lr': args.lr_init*decay_factor},
             {'params': model_sv.model.fc.parameters(), 'lr': args.lr_init*10*decay_factor},
             {'params': model_sv.classifier_1.parameters(), 'lr': args.lr_init*10*decay_factor},
             {'params': model_sv.classifier_2.parameters(), 'lr': args.lr_init*10*decay_factor},
             {'params': model_sv.classifier_3.parameters(), 'lr': args.lr_init*10*decay_factor},
             {'params': model_sv.classifier_4.parameters(), 'lr': args.lr_init*10*decay_factor},
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)

        # local LR scheduler
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        # train and update
        list_loss = []
        KL_loss_list=[] #shitong

        for epoch in range(self.args.local_ep):
            print('Local Training Epoch {}/{}'.format(epoch+1, self.args.local_ep))
            print('-' * 10)

            model,model_sv = model.cuda(), model_sv.cuda()
            model.train(True)
            model_sv.train(True)

            running_loss, running_loss_sv = 0.0, 0.0
            running_corrects, running_corrects_sv = 0.0, 0.0

            for data in tqdm(self.data_loader):
                # get the inputs
                inputs, labels = data
                now_batch_size, c, h, w = inputs.shape

                ## skip the last small batch
                #if now_batch_size<self.args.batchsize:
                #   continue

                # wrap data in Variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs, idx_client)
                outputs_sv = model_sv(inputs, idx_client)

                # compute loss
                _, preds = torch.max(outputs.data, 1)
                loss = self.criterion_ce(outputs, labels) # local model loss
                loss_sv = self.criterion_ce(outputs_sv, labels) # copy central model loss
                loss_kl = self.criterion_kl(outputs, outputs_sv, T=self.args.T) # server supervision loss

                loss = loss + loss_sv + loss_kl # optimisation objective

                # backward
                loss.backward()
                optimizer.step()

                # compute accuracy and loss of current batch
                list_loss.append(loss.item())
                running_loss += loss.item() * now_batch_size
                running_corrects += float(torch.sum(preds == labels.data))

                #shitong
                KL_loss_list.append(loss_kl.item())

            # compute accuracy and loss of current epoch
            epoch_loss = running_loss / (len(self.data_loader)*self.args.batchsize)
            epoch_acc = running_corrects / (len(self.data_loader)*self.args.batchsize)

            scheduler.step()
            
            print('-'*20)
            print('Local client {} Training Loss: {:.4f} Acc: {:.4f}'.format(
                idx_client, epoch_loss, epoch_acc))

        # return model parameters and training loss
        return{'params': model.cpu().state_dict(),
               'loss': sum(list_loss) / len(list_loss),
               'KL_loss': sum(KL_loss_list) / len(KL_loss_list)}

        
    # evaluating current model
    def evaluate(self, data_loader, model, idx_client=0,disEN=False):
        model = model.cuda()       
        model.train(False)  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0.0

        # Iterate over data.
        for data in data_loader:
            # get the inputs
            inputs, labels = data
            now_batch_size,c,h,w = inputs.shape

            # skip the last small batch
            # if now_batch_size<self.args.batchsize: 
                # continue
            
            # wrap data in Variable           
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            
            # forward
            if disEN:
                outputs = model(inputs,idx_client)
            else:
                outputs = model(inputs, idx_client)
            # compute accuracy and loss of current batch

            _, preds = torch.max(outputs.data, 1)
            pdb.set_trace()
            loss = self.criterion_ce(outputs, labels)

            running_loss += loss.item() * now_batch_size
            running_corrects += float(torch.sum(preds == labels.data))

        # compute accuracy and loss of current epoch
        epoch_loss = running_loss / (len(data_loader)*self.args.batchsize)
        epoch_acc = running_corrects / (len(data_loader)*self.args.batchsize)


        # return evaluation loss and accuracy
        return epoch_loss, epoch_acc

    # updating local model parameters for disentanglement case
    # shiong
    def update_weights_disEN(self, model, cur_epoch, idx_client, model_sv):

        #shitong

        # mapping network parameters
        # when local epoch > 1, the copy server model should also be updated shared the merit of mutual learning
        ignored_params = list(map(id, model.model.fc.parameters())) + list(map(id, model.classifier_1.parameters())) \
                         + list(map(id, model.classifier_2.parameters())) + list(map(id, model.classifier_3.parameters())) \
                         + list(map(id, model.classifier_4.parameters()))
        ignored_params_sv = list(map(id, model_sv.model.fc.parameters())) + list(map(id, model_sv.classifier_1.parameters())) \
                            + list(map(id, model_sv.classifier_2.parameters())) + list(map(id, model_sv.classifier_3.parameters())) \
                            + list(map(id, model_sv.classifier_4.parameters()))

        # feature embedding network parameters
        # shitong filter: first parameter: return True/False; filter will choose from the second parameter which satisfy the first
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        base_params_sv = filter(lambda p: id(p) not in ignored_params_sv, model_sv.parameters())

        # optimiser setting
        # global LR sheduler, hyperparameters can be fine-tuned
        # lr_init: 0.01 for embedding network and 0.1 for mapping network
        decay_factor = 1.0 if cur_epoch < 20 else 0.1
        # shitong
        args = self.args
        optimizer = optim.SGD([
            {'params': base_params, 'lr': args.lr_init * decay_factor},
            {'params': model.model.fc.parameters(), 'lr': args.lr_init * 10 * decay_factor},
            {'params': model.classifier_1.parameters(), 'lr': args.lr_init * 10 * decay_factor},
            {'params': model.classifier_2.parameters(), 'lr': args.lr_init * 10 * decay_factor},
            {'params': model.classifier_3.parameters(), 'lr': args.lr_init * 10 * decay_factor},
            {'params': model.classifier_4.parameters(), 'lr': args.lr_init * 10 * decay_factor},
            {'params': base_params_sv, 'lr': args.lr_init * decay_factor},
            {'params': model_sv.model.fc.parameters(), 'lr': args.lr_init * 10 * decay_factor},
            {'params': model_sv.classifier_1.parameters(), 'lr': args.lr_init * 10 * decay_factor},
            {'params': model_sv.classifier_2.parameters(), 'lr': args.lr_init * 10 * decay_factor},
            {'params': model_sv.classifier_3.parameters(), 'lr': args.lr_init * 10 * decay_factor},
            {'params': model_sv.classifier_4.parameters(), 'lr': args.lr_init * 10 * decay_factor},
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)

        # local LR scheduler
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        # train and update
        list_loss = []
        mse_list_loss=[]
        id_list_loss=[]

        for epoch in range(self.args.local_ep):
            print('-' * 10)
            start=time.time()
            model, model_sv = model.cuda(), model_sv.cuda()
            model.train(True)
            model_sv.train(True)

            running_loss, running_loss_sv = 0.0, 0.0
            running_corrects, running_corrects_sv = 0.0, 0.0

            for data in tqdm(self.data_loader):
                # get the inputs
                inputs, labels = data
                now_batch_size, c, h, w = inputs.shape

                ## skip the last small batch
                # if now_batch_size<self.args.batchsize:
                #   continue

                # wrap data in Variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs,globel_r = model(inputs, idx_client)
                globel_r_sv = model_sv(inputs, idx_client)

                # compute loss
                # _, preds = torch.max(outputs.data, 1)
                _,preds = torch.max(outputs.data, 1)

                loss_id = self.criterion_ce(outputs, labels)  # local model loss
                loss_mse = self.criterion_mse(globel_r,globel_r_sv)

                loss = loss_id + loss_mse # todo: can add weight in furture
                #
                # backward
                loss.backward()
                optimizer.step()

                # compute accuracy and loss of current batch
                list_loss.append(loss.item())
                running_loss += loss.item() * now_batch_size
                running_corrects += float(torch.sum(preds == labels.data))

                #shitong
                id_list_loss.append(loss_id.item())
                mse_list_loss.append(loss_mse.item())


            # compute accuracy and loss of current epoch
            epoch_loss = running_loss / (len(self.data_loader) * self.args.batchsize)
            epoch_acc = running_corrects / (len(self.data_loader) * self.args.batchsize)

            scheduler.step()

            print('-' * 20)
            print('Local Training Epoch {}/{}: Local client {} Training Loss: {:.4f} Acc: {:.4f} Time : {:.2f}'
                  .format(epoch + 1, self.args.local_ep,idx_client, epoch_loss, epoch_acc,time.time()-start))

        # return model parameters and training loss
        return {'params': model.cpu().state_dict(),
                'loss': sum(list_loss) / len(list_loss),
                'acc': epoch_acc,
                'id_loss': sum(id_list_loss) / len(id_list_loss),
                'mse_loss':sum(mse_list_loss) / len(mse_list_loss) }


