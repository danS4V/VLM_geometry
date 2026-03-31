import torch
import torch.nn.functional as F

######################
### LOSS FUNCTIONS ###

def CrossEnt(r,hastarget):
  '''Cross entropy, assumes r to be sigmoid output'''
  ### Con sigmoide è equivalente a F.binary_cross_entropy_with_logits(r,hastarget,reduction='sum')
  assert (hastarget.shape == r.shape), f"{hastarget.shape},{r.shape}"
  crossent = -torch.sum(hastarget*torch.log(r)+(1-hastarget)*torch.log(1-r))
  return (crossent)

def ExpRegCrossEnt(y_hat,y):
  '''A cross entropy regularized on logits to avoid having answers close to 0.5.
  
  y_hat : torch.Tensor
    logits output from the model
  y : torch.Tensor
    Ground truth (0-1)
  '''
  return CrossEnt(F.sigmoid(y_hat),y)+0.5*torch.sum(torch.exp(-torch.abs(y_hat)))

def ExpRegTorch(y_hat,y):
  return F.binary_cross_entropy_with_logits(y_hat,y)+0.5*torch.sum(torch.exp(-torch.abs(y_hat)))/y_hat.shape[0]

def GetLoss(LossName):
  'LossName may also be F.something'
  return eval(LossName)

################
### DATASETS ###

class HasColorShapeMultiLabelDataset(torch.utils.data.Dataset):
  '''
  Multi-label dataset for joint probe training.
  Returns hidden states and a (K,) vector of binary labels, one per concept.
  colshapes must be an ordered list of K concept strings (e.g. ['redsquare', ...]);
  the same order must be preserved when loading the trained joint probe.
  '''
  def __init__(self, metadata, colshapes, hidden_states):
    self.labels = torch.tensor(metadata[colshapes].values, dtype=torch.float32)  # (N, K)
    self.data = hidden_states[metadata['ID'].astype(int).tolist()].cuda()
    self.ids = metadata['ID'].tolist()

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]


class HasColorShapeDataset(torch.utils.data.Dataset):
  '''
  In this dataset, the target labels are 0-1 values for a single colorshape
  '''
  def __init__(self,metadata,targetcol,hidden_states):
    self.labels = torch.tensor(metadata[targetcol].tolist(),dtype=torch.float32)
    self.data = hidden_states[metadata['ID'].astype(int).tolist()].cuda()
    self.ids = metadata['ID'].tolist()
    
  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self,idx):
    return self.data[idx], self.labels[idx]
  
  def get_ids(self,idx):
    return self.data[idx],self.labels[idx],self.ids[idx]
