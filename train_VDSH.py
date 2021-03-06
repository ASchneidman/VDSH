import os
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from datasets import *
from utils import *
from models.VDSH import VDSH
import argparse
from datetime import datetime
import boto3
import botocore

def end():
    with open('logs/VDSH/result.txt', 'a') as handle:
        handle.write('{},{},{}\n'.format(dataset, data_fmt, args.nbits))

    if not os.path.isdir('trained_models'):
        os.mkdir('trained_models')

    now = datetime.now()

    torch.save(model.state_dict(), 'trained_models/' + model.get_name() + '_' + now.strftime("%d:%m:%Y_%H:%M") + '.pth')

##################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpunum", help="GPU number to train the model.", default='0')
parser.add_argument("-d", "--dataset", help="Name of the dataset.", default='darknet.tfidf')
parser.add_argument("-b", "--nbits", help="Number of bits of the embedded vector.", type=int, default=8)
parser.add_argument("--dropout", help="Dropout probability (0 means no dropout)", default=0.1, type=float)
parser.add_argument("--train_batch_size", default=100, type=int)
parser.add_argument("--test_batch_size", default=100, type=int)
parser.add_argument("--transform_batch_size", default=100, type=int)
parser.add_argument("--num_epochs", default=30, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--download", help="1 to download from s3 bucket", type=int, default=0)

args = parser.parse_args()

if not args.gpunum:
    parser.error("Need to provide the GPU number.")
    
if not args.dataset:
    parser.error("Need to provide the dataset.")

if not args.nbits:
    parser.error("Need to provide the number of bits.")
        
##################################################################################################

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpunum
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#########################################################################################################

if args.download:
    print("Downloading")
    s3_client = boto3.client('s3')
    #s3_client.download_file('15405finalprojectcsvdata', 'pickled_split_data/test.tf.df.pkl', 'dataset/darknet/test.tf.df.pkl')
    #print("Done test")
    s3_client.download_file('15405finalprojectcsvdata', 'pickled_split_data/train.tf.df.pkl', 'dataset/darknet/train.tf.df.pkl')
    print("Done train")

dataset, data_fmt = args.dataset.split('.')

if dataset in ['reuters', 'tmc', 'rcv1']:
    single_label_flag = False
else:
    single_label_flag = True
        
if single_label_flag:
    train_set = SingleLabelTextDataset('dataset/{}'.format(dataset), subset='train', bow_format=data_fmt, download=True)
    #test_set = SingleLabelTextDataset('dataset/{}'.format(dataset), subset='test', bow_format=data_fmt, download=True)
else:
    train_set = MultiLabelTextDataset('dataset/{}'.format(dataset), subset='train', bow_format=data_fmt, download=True)
    test_set = MultiLabelTextDataset('dataset/{}'.format(dataset), subset='test', bow_format=data_fmt, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.train_batch_size, shuffle=True)
#test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.test_batch_size, shuffle=True)

#########################################################################################################
y_dim = train_set.num_classes()
num_bits = args.nbits
num_features = train_set[0][0].size(0)

print("Train VDSH model ...")
print("dataset: {}".format(args.dataset))
print("numbits: {}".format(args.nbits))
print("gpu id:  {}".format(args.gpunum))
print("dropout probability: {}".format(args.dropout))
if single_label_flag:
    print("single-label prediction.")
else:
    print("multi-label prediction.")
print("num epochs: {}".format(args.num_epochs))
print("learning rate: {}".format(args.lr))
#print("num train: {} num test: {}".format(len(train_set), len(test_set)))

#########################################################################################################

model = VDSH(dataset, num_features, num_bits, dropoutProb=0.1, device=device)
model.to(device)
model.train()

num_epochs = args.num_epochs

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5,amsgrad=True)
#kl_weight = 0.
kl_weight = 0.1
kl_step = 1 / 5000.

if not os.path.isdir('logs'):
    os.mkdir('logs')
if not os.path.isdir('logs/' + model.get_name()):
    os.mkdir('logs/' + model.get_name())

try:
    with open('logs/VDSH/loss.log.txt', 'w') as log_handle:
        log_handle.write('epoch,step,loss,reconstr_loss,kl_loss\n')
        
        for epoch in range(num_epochs):
            avg_loss = []
            for step, (xb, yb) in enumerate(train_loader):
                xb = xb.to(device)
                yb = yb.to(device)

                logprob_w, mu, logvar = model(xb)
                kl_loss = VDSH.calculate_KL_loss(mu, logvar)
                reconstr_loss = VDSH.compute_reconstr_loss(logprob_w, xb)
                
                loss = reconstr_loss + kl_weight * kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                kl_weight = min(kl_weight + kl_step, 1.)
                avg_loss.append(loss.item())
                
                log_handle.write('{},{},{:.4f},{:.4f},{:.4f}'.format(epoch, step, loss.item(), 
                                                                    reconstr_loss.item(), kl_loss.item()))
            print('{} epoch:{} loss:{:.4f}'.format(model.get_name(), epoch+1, np.mean(avg_loss)))
            
            '''
            with torch.no_grad():
                train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader)
                retrieved_indices = retrieve_topk(test_b.to(device), train_b.to(device), topK=100)
                prec = compute_precision_at_k(retrieved_indices, test_y.to(device), train_y.to(device), topK=100, is_single_label=single_label_flag)
                print("precision at 100: {:.4f}".format(prec.item()))

                if prec.item() > best_precision:
                    best_precision = prec.item()
                    best_precision_epoch = epoch + 1
            '''
            
    #########################################################################################################
    end()
except KeyboardInterrupt:
    print("Ending prematurely")
    end()

