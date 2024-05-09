import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import (LinearRegression, LogisticRegression, ElasticNet, Lasso)
from sklearn.model_selection import (train_test_split, KFold, GridSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from ISLP import load_data
from ISLP.models import ModelSpec as MS

import torch
from torch import nn
from torch.optim import RMSprop
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import (MeanAbsoluteError, R2Score)
from torchinfo import summary

from torchvision.io import read_image
from torchvision.datasets import (MNIST, CIFAR100)
from torchvision.models import (resnet50, ResNet50_Weights)
from torchvision.transforms import (Resize, Normalize, CenterCrop, ToTensor)

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import seed_everything

seed_everything (0)
torch.use_deterministic_algorithms(True, warn_only=True)

from ISLP.torch import (SimpleDataModule, SimpleModule, ErrorTracker, rec_num_workers)
from ISLP.torch.imdb import (load_lookup, load_tensor, load_sparse, load_sequential)

from glob import glob
import json


Working on it....


hitters=load_data('Hitters').dropna()
hitters.shape
n=hitters.shape[0]
# 263*20  n=number of rows
hitters.head()
y=hitters.pop('Salary')
X=MS(hitters,intercept=False).fit_transform(hitters)
X.head()
X.shape
# MS creates dummies and delete the 1st columns. 
# 3 binary categorical features, cols=19. Intercept well be crated directly by sklearn model.
X_train,  X_test, y_train, y_test=train_test_split(X.values, y.values, random_state=1, test_size=1/3)

# I Linear Regression
hit_lm=LinearRegression()
hit_lm.fit(X_train, y_train)
pred_lm=hit_lm.predict(X_test)
mae_lm=np.abs(pred_lm-y_test).mean()
mae_lm
# 259.71528833146317

# II lasso
scaler=StandardScaler(with_mean=True, with_std=True)
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)
n=X_train_scaled.shape[0]
n
# 175 rows
cv=KFold(random_state=1, n_splits=10, shuffle=True)
lambda_max=np.fabs(X_train_scaled.T.dot(y_train-y_train.mean())).max()/n
lambda_max
lambdas=np.exp(np.linspace(0, np.log(0.01),100))*lambda_max
lambdas
# 255.65755026491283
lasso=ElasticNet(l1_ratio=1, warm_start=True, max_iter=30000)
# lambdas 100 nums bw 0.01 and 1
grid_lasso=GridSearchCV(lasso, param_grid={'alpha':lambdas},cv=cv, scoring='neg_mean_absolute_error')
grid_lasso.fit(X_train, y_train)
grid_lasso.best_params_
# 176.21483255243294
pred_lasso=grid_lasso.best_estimator_.predict(X_test)
mae_lasso=np.abs(pred_lasso-y_test).mean()
mae_lasso
# 257.23820107995016

# III Single Layer Network on Hitters Data 
# Did not finish this part: output is different from book, even different for each run
# Transform X and y into 32FLOAT, then tensor, then combine X and y as dataset. Treat similarly for training set.
# Convert Input and Output data to Tensors and create a TensorDataset
X_train_ten=torch.tensor(X_train.astype(np.float32))
y_train_ten=torch.tensor(y_train.astype(np.float32))
hit_train=TensorDataset(X_train_ten, y_train_ten)

X_test.dtype
X_test_ten=torch.tensor(X_test_scaled.astype(np.float32))
y_test_ten=torch.tensor(y_test.astype(np.float32))
hit_test=TensorDataset(X_test_ten, y_test_ten)

# Create Dataloader to read the data within batch sizes and put into memory. 
train_loader = DataLoader(hit_train, batch_size = 32, shuffle = True) 
validate_loader = DataLoader(hit_test, batch_size = 1) 
test_loader = DataLoader(hit_test, batch_size = 1)

# Define model parameters 
input_size=X_train.shape[1]
output_size=1

# Define neural network structure
class HittersModel (nn.Module):
    def __init__(self, input_size, output_size):
        super(HittersModel, self).__init__()
        self.flatten = nn.Flatten()
        self.sequential=nn.Sequential(
            nn.Linear(in_features=input_size, out_features=50, bias=True), 
            nn.ReLU(), 
            nn.Dropout(0.4), 
            nn.Linear(in_features=50,out_features=output_size,bias=True))
    def forward(self, x):
        x=self.flatten(x)
        return torch.flatten(self.sequential(x))
    
# Instantiate the model 
hit_model=HittersModel(input_size, output_size)

# Visualize model structure
summary(hit_model, input_size=X_train.shape, col_names=['input_size','output_size','num_params'])

# Define your execution device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
print("The model will be running on", device, "device\n") 
hit_model.to(device)    

# Function to save the model 
def saveModel(): 
    path = "C:/Users/yxm02/.ipynb_checkpoints/NetModel.pth" 
    torch.save(hit_model.state_dict(), path) 

max_num_workers = rec_num_workers()
hit_dm = SimpleDataModule(hit_train, hit_test, batch_size=32,validation=hit_test)
hit_module= SimpleModule.regression(hit_model, metrics={'mae':MeanAbsoluteError()})
hit_logger=CSVLogger('logs', name='hitters')

hit_trainer= Trainer(deterministic=True,accelerator="cpu",max_epochs=50, log_every_n_steps=5, logger=hit_logger, callbacks=[ErrorTracker()])
hit_trainer.fit(hit_module, datamodule=hit_dm)

hit_trainer.test(hit_module, datamodule=hit_dm)

hit_results = pd.read_csv(hit_logger.experiment.metrics_file_path)

def summary_plot(results,
                 ax,
                 col='loss',
                 valid_legend='Validation',
                 training_legend='Training',
                 ylabel='Loss',
                 fontsize=20):
    for (column,
         color,
         label) in zip([f'train_{col}_epoch',
                        f'valid_{col}'],
                       ['black',
                        'red'],
                       [training_legend,
                        valid_legend]):
        results.plot(x='epoch',
                     y=column,
                     label=label,
                     marker='o',
                     color=color,
                     ax=ax)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    return ax
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax = summary_plot(hit_results,
                  ax,
                  col='mae',
                  ylabel='MAE',
                  valid_legend='Validation (=Test)')
ax.set_ylim([0, 400])
ax.set_xticks(np.linspace(0, 50, 11).astype(int));

hit_model.eval() 
preds = hit_module(X_test_ten)
torch.abs(y_test_ten - preds).mean()

del(hitters,hit_model, hit_dm, hit_logger,hit_test, hit_train, X, Y,X_test, X_train,y_test, y_train, X_test_ten, y_train_ten,hit_trainer, hit_module)


# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
