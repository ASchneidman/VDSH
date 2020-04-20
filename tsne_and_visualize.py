import torch, os
from models.VDSH import VDSH
from datasets import *
from utils import *
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
        from tsnecuda import TSNE
except ImportError:
        from sklearn.manifold import TSNE

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpunum", help="GPU number to train the model.", default='0')
parser.add_argument("-d", "--dataset", help="Name of the dataset.", default='darknet.tfidf')
parser.add_argument("-b", "--nbits", help="Number of bits of the embedded vector.", type=int, default=8)
parser.add_argument("--train_batch_size", default=100, type=int)
parser.add_argument("-m", "--model", help="File name of the pretrained model (located in trained_models/)", default='VDSH_15:04:2020_03:28.pth')
args = parser.parse_args()

if not args.gpunum:
        parser.error("Need to provide the GPU number.")

if not args.dataset:
        parser.error("Need to provide the dataset.")

if not args.nbits:
        parser.error("Need to provide the number of bits.")


os.environ["CUDA_VISIBLE_DEVICES"]=args.gpunum
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset, data_fmt = args.dataset.split('.')

test_set = SingleLabelTextDataset('dataset/{}'.format(dataset), subset='train', bow_format=data_fmt, download=True)

test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=True)

y_dim = test_set.num_classes()
num_bits = args.nbits
num_features = test_set[0][0].size(0)

# load model
model = VDSH(None, num_features, num_bits, dropoutProb=0.1, device=device)
model.to(device)

model.load_state_dict(torch.load('trained_models_32/' + args.model,  map_location=torch.device(device)))
model.eval()

codes = []

# encode all of test
for step, (xb, yb) in enumerate(test_loader):
        with torch.no_grad():
                code = model.get_code(xb).numpy()[0]
                #print("Sample " + str(step) + ": " + str(code))
                codes.append(code)

codes = np.array(codes)
embedded = TSNE(n_components = 2).fit_transform(codes)

df_embedded = pd.DataFrame()
df_embedded['tsne_x'] = embedded[:,0]
df_embedded['tsne_y'] = embedded[:,1]

df_embedded.to_pickle('df_embedded_result.pkl')

sns.scatterplot(data=df_embedded, alpha=0.3, x='tsne_x', y='tsne_y')
plt.show()