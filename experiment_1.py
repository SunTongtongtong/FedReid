# experiment 1: test AAAI FedReID: if it work well on local dataset
# prove that the performance of AAAI FedReID on local data is worse than the model train on market1501 and test on it


from lib.model import embedding_net
from lib.localUpdate import DatasetSplitLM
from utils.get_dataset import get_dataset
from config import opt
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn as nn
import torch
import os
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/centerial_experiment_1')


local_datasets, dict_users, dataloaders_val = get_dataset(opt, is_training=True)
num_ids_client = []  # number of training ids in each local client
for i in range(opt.nusers):
    num_ids_client.append(len(local_datasets[i].classes))


for idx,dataset in enumerate(local_datasets):

    data_loader = DataLoader(DatasetSplitLM(dataset, list(dict_users[idx][0])), batch_size=opt.batchsize, shuffle=True,
                              num_workers=4)
    model = embedding_net(num_ids_client) # a fresh model with initialization
    ignored_params = list(map(id, model.model.fc.parameters())) + list(map(id, model.classifier_1.parameters())) \
                     + list(map(id, model.classifier_2.parameters())) + list(map(id, model.classifier_3.parameters())) \
                     + list(map(id, model.classifier_4.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    decay_factor = 1.0
    optimizer = optim.SGD([
             {'params': base_params, 'lr': opt.lr_init*decay_factor},
             {'params': model.model.fc.parameters(), 'lr': opt.lr_init*10*decay_factor},
             {'params': model.classifier_1.parameters(), 'lr': opt.lr_init*10*decay_factor},
             {'params': model.classifier_2.parameters(), 'lr': opt.lr_init*10*decay_factor},
             {'params': model.classifier_3.parameters(), 'lr': opt.lr_init*10*decay_factor},
             {'params': model.classifier_4.parameters(), 'lr': opt.lr_init*10*decay_factor} ], weight_decay=5e-4, momentum=0.9, nesterov=True)  # in AAAI ACM use different scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion_ce = nn.CrossEntropyLoss()  # local supervision

    model.train(True)
    model.cuda()

    list_loss=[]
    for epoch in range(opt.local_ep):
        epoch= epoch+1
        running_loss= 0.0
        running_corrects = 0.0

        for data in data_loader:
            # get the inputs
            inputs, labels = data
            now_batch_size, c, h, w = inputs.shape

            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()

            # forward
            outputs = model(inputs, idx)  # idx should correspond with classifier and dataset ids

            # compute loss
            _, preds = torch.max(outputs.data, 1)
            loss = criterion_ce(outputs, labels)  # local model loss

            # backward
            loss.backward()
            optimizer.step()
            list_loss.append(loss.item())
            running_loss += loss.item() * now_batch_size
            running_corrects += float(torch.sum(preds == labels.data))

        epoch_loss = running_loss / (len(data_loader) * opt.batchsize)
        epoch_acc = running_corrects / (len(data_loader) * opt.batchsize)
        scheduler.step()

        print('Local Training Epoch {}/{}: Training Loss: {:.4f} Acc: {:.4f}'.format(epoch, opt.local_ep, epoch_loss, epoch_acc))
        model_pth = 'model_{}_epoch{}.pth'.format(dataset.root.split('/')[-3],epoch)
        model_saved = os.path.join(opt.logs_dir,os.path.join('centralize',model_pth)) # model directory
        if epoch %10 ==0:# change to 0 todo
            with open(model_saved, 'wb') as f:
                torch.save(model.state_dict(), f)

        writer.add_scalar(dataset.root.split('/')[-3]+'_training loss',
                          epoch_loss,
                          epoch)




