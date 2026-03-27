import torch
import lightning as pl


class LAttnProbeJoint(pl.LightningModule):
  """Joint attention probe predicting K concepts simultaneously.

  Architecture mirrors LAttnProbeSingle but uses Linear(D, K) so all concepts
  share the same forward pass. Each concept has its own per-token softmax
  attention and its own affine calibration (last_mult, last_bias).

  Args:
    in_dim:     hidden dimension D of input token embeddings.
    n_concepts: number of concepts K to predict jointly.
    Loss:       callable(logits, targets) returning a scalar loss.
    optimizer:  partial optimizer constructor (receives params).
    scheduler:  partial lr-scheduler constructor (receives optimizer), optional.
  """

  def __init__(self, in_dim: int, n_concepts: int, Loss,
               optimizer: torch.optim.Optimizer,
               scheduler: torch.optim.lr_scheduler = None):
    super().__init__()
    self.n_concepts = n_concepts
    self.attn_proj  = torch.nn.Linear(in_dim, n_concepts)
    self.softmax    = torch.nn.Softmax(dim=-2)  # softmax over token dimension
    self.last_mult  = torch.nn.Parameter(torch.ones(n_concepts))
    self.last_bias  = torch.nn.Parameter(torch.full((n_concepts,), 1e-2))
    self.Loss = Loss

    self.optimizer    = optimizer(params=self.parameters())
    self.lr_scheduler = scheduler(optimizer=self.optimizer) if scheduler is not None else None

  def forward(self, x):
    # x: (B, T, D)
    alog = self.attn_proj(x)           # (B, T, K)
    a    = self.softmax(alog)          # (B, T, K) — softmax over T per concept
    s    = torch.sum(a * alog, dim=-2) # (B, K)
    return s * self.last_mult + self.last_bias  # (B, K)

  def training_step(self, batch, batch_idx):
    x, y = batch  # y: (B, K)
    y_hat = self(x)
    loss = self.Loss(y_hat, y)
    with torch.no_grad():
      acc = (y == (y_hat > 0)).float().mean()
      self.log("training_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
      self.log("training_acc",  acc,  on_step=False, on_epoch=True, prog_bar=True, logger=True)
    return loss

  def validation_step(self, batch, batch_idx):
    with torch.no_grad():
      x, y = batch
      y_hat = self(x)
      loss = self.Loss(y_hat, y)
      acc  = (y == (y_hat > 0)).float().mean()
      self.log('val_loss',     loss, prog_bar=True, on_step=False, on_epoch=True)
      self.log('val_accuracy', acc,  prog_bar=True, on_step=False, on_epoch=True)
    return acc.detach()

  def configure_optimizers(self):
    if self.lr_scheduler is None:
      return self.optimizer
    else:
      return {'optimizer': self.optimizer,
              "lr_scheduler": {
                  "scheduler": self.lr_scheduler,
                  "monitor":   "training_loss",
                  "interval":  "epoch",
                  "frequency": 1,
                  "strict":    True,
              }}


class LAttnProbeSingle(pl.LightningModule):

  def __init__(self,in_dim : int, Loss, optimizer : torch.optim.Optimizer, 
               scheduler : torch.optim.lr_scheduler = None):
    super().__init__()
    self.attn_proj = torch.nn.Linear(in_dim,1)
    self.softmax = torch.nn.Softmax(dim=-2)
    self.last_mult = torch.nn.parameter.Parameter(torch.tensor([1.],requires_grad=True))
    self.last_bias = torch.nn.parameter.Parameter(torch.tensor([1e-2],requires_grad=True))
    self.Loss = Loss
    
    self.optimizer = optimizer(params=self.parameters())
    self.lr_scheduler = scheduler(optimizer=self.optimizer) if scheduler is not None else None
    #Does not apply the sigmoid by default

  def forward(self, x, output_attentions=False,output_attn_logits=False):
    alog = self.attn_proj(x)
    a = self.softmax(alog)
    x=a*alog
    x = torch.sum(x,axis=-2)
    x= x*self.last_mult+self.last_bias
    if output_attentions:
      return x.squeeze(),a
    elif output_attn_logits:
      return x.squeeze(), alog
    else:
      return x.squeeze()

  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = self.Loss(y_hat,y)
    with torch.no_grad():
      acc = (y==(y_hat>0)).sum()/y.shape[0]
      self.log("training_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
      self.log("training_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    return loss

  def validation_step(self, batch, batch_idx):
    with torch.no_grad():
      x, y = batch
      y_hat = self(x)
      acc = (y==(y_hat>0)).sum()/y.shape[0]
      loss = self.Loss(y_hat,y)
      self.log('val_loss',loss,prog_bar=True,on_step=False,on_epoch=True)
      self.log('val_accuracy',acc,prog_bar=True,on_step=False,on_epoch=True)
    return acc.detach()
  
  def configure_optimizers(self):
    if self.lr_scheduler is None:
      return self.optimizer
    else:
      return {'optimizer': self.optimizer, 
              "lr_scheduler": {
                    "scheduler": self.lr_scheduler,
                    "monitor": "training_loss",
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": True
                }
      }
