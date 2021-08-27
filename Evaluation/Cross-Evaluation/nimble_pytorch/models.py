import os, sys, random, string
import numpy as np
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from torch.cuda.amp import autocast as ampautocast
from torch.cuda.amp import GradScaler as AmpGradScaler
from . import BASE_METRICS, BASE_LOSSES, BASE_SCHEDULERS, BASE_OPTIMIZERS

class NimbleSequential(nn.Module):
    def __init__(self, *layers, name=None):
        super().__init__()
        self.__set_default(name)
        if layers:
            self.sequential = nn.Sequential(*layers)
            self.reset_parameters()        
        
    def __set_default(self, name=None):
        ## General Cofigs
        self.optimizer = None
        self.scheduler = None
        self.loss = None
        self.metrics = []
        self.sequential = None
        
        ## Train + Valid History
        self.train_history = {} 
        self.valid_history = {}
        
        ## Auto save best model
        self.set_autosave()
        
        ## Helper
        self.current_epoch = 0
        self.name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=12))
        if name is not None:
            self.name = name
        self.device = None

        ## Set Torch APM
        self.use_mixed_precision = False
        self.amp_gradient_scaler = None
        
        
    def forward(self, X):
        #TODO: change to args + kwargs
        ## Execute simple forward pass
        for layer in self.sequential:
            X = layer(X)
        return X
    
    
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            return self.forward(X)
    
    
    def set_optimizer(self, optimizer, *args, **kwargs):
        if isinstance(optimizer, str) and optimizer in BASE_OPTIMIZERS:
            self.optimizer = BASE_OPTIMIZERS[optimizer](self.sequential.parameters(), *args, **kwargs)
            
        elif isinstance(optimizer, str) and not optimizer in BASE_OPTIMIZERS:
            raise Exception('Optimizer string mus be one of the following: {}'.format(BASE_OPTIMIZERS.keys()))
        
        else:
            self.optimizer = optimizer(self.sequential.parameters(), *args, **kwargs)
        
        if not isinstance(self.optimizer, torch.optim.Optimizer):
            raise Exception('Optimizer must be of instance \
                                torch.optim.Optimizer but is {}'.format(type(self.optimizer)))
            
            
    def set_scheduler(self, lr_scheduler, *args, **kwargs):    
        if self.optimizer is None:
            raise Exception('Optimizer must be set first!')
        
        if isinstance(lr_scheduler, str) and lr_scheduler in BASE_SCHEDULERS:
            self.scheduler = BASE_SCHEDULERS[lr_scheduler](self.optimizer, *args, **kwargs)
            
        elif isinstance(lr_scheduler, str) and not lr_scheduler in BASE_SCHEDULERS:
            raise Exception('Scheduler string mus be one of the following: {}'.format(BASE_SCHEDULERS.keys()))
            
        else: 
            self.scheduler = lr_scheduler(self.optimizer, *args, **kwargs)
        
        
    def set_loss(self, loss, *args, **kwargs):
        if isinstance(loss, str) and loss in BASE_LOSSES:
            self.loss = BASE_LOSSES[loss](*args, **kwargs)
            
        elif isinstance(loss, str) and not loss in BASE_LOSSES:
            raise Exception('Loss string mus be one of the following: {}'.format(BASE_LOSSES.keys()))
            
        else:  
            self.loss = loss(*args, **kwargs)
        
        if not isinstance(self.loss, torch.nn.modules.loss._Loss):
            raise Exception('Loss must be of instance \
                                torch.nn.modules.loss._Loss but is {}'.format(type(self.loss)))
            
            
    def add_metric(self, metric, *args, **kwargs):
        if isinstance(metric, str) and metric in BASE_METRICS:
            m = BASE_METRICS[metric](*args, **kwargs)
            
        elif isinstance(metric, str) and not metric in BASE_METRICS:
            raise Exception('Metric string mus be one of the following: {}'.format(BASE_METRICS.keys()))
            
        else:
            m = metric(*args, **kwargs)
            
        self.metrics.append(m)
        
        
    def set_amp_scaler(self, enable=True, **kwargs):
        if enable:
            self.use_mixed_precision = True
            self.amp_gradient_scaler = AmpGradScaler(**kwargs)
        else:
            self.use_mixed_precision = False
            self.amp_gradient_scaler = None
    
    
    def set_autosave(self, best_valid_label='loss', best_valid_selector=min, best_valid_postfix='_best'):
        self.best_valid = None
        self.best_valid_label = best_valid_label
        self.best_valid_selector = best_valid_selector
        self.best_valid_postfix = best_valid_postfix
                     
        
    def reset_parameters(self):
        for module in self.sequential:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
                
                
    def save(self, path, name=None):
        if name is None and self.name is None:
            torch.save(self, path)
        else:
            name = self.name + '.pt' if name is None else name
            path = os.path.join(path, name)
            torch.save(self, path)
            
    
    def set_device(self, device):
        self.device = device
        return self.to(device=device)
        
    def cuda(self):
        return self.set_device('cuda')
        
    def cpu(self):
        return self.set_device('cpu')
        
    def fit_numpy(self, X_train, y_train, X_valid=None, y_valid=None, valid_split=0., epochs=1, warm_up=0, 
            auto_save_path=None, **kwargs):
        if (X_valid is None or y_valid is None) and valid_split > 0.:
            train_ds, valid_ds = self._create_tensor_from_numpy(X_train, y_train, valid_split=valid_split)
        elif X_valid is not None and y_valid is not None:
            train_ds,_ = self._create_tensor_from_numpy(X_train, y_train)
            valid_ds,_ = self._create_tensor_from_numpy(X_valid, y_valid)
        else:
            train_ds,_ = self._create_tensor_from_numpy(X_train, y_train)
            valid_ds = None
            
        train_loader = torch.utils.data.DataLoader(train_ds, **kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_ds, **kwargs) if valid_ds is not None else None
        
        self.fit(train_loader, valid_loader, epochs, warm_up)
        
    def fit(self, train_loader, valid_loader=None, epochs=1, warm_up=0, 
            auto_save_path=None, disabled_tqdm=False, **kwargs):
        
        ## Append train infos to kwargs
        kwargs['train_loader'] = train_loader
        kwargs['valid_loader'] = valid_loader
        kwargs['epochs'] = epochs
        kwargs['warm_up'] = warm_up
        kwargs['auto_save_path'] = auto_save_path
        
        start_epoch = self.current_epoch + 1
        stop_epoch = epochs + self.current_epoch + 1
        
        if isinstance(warm_up, int) and warm_up > 0 and self.current_epoch == 0:
            warm_up_steps = min(len(train_loader), warm_up)
            lam_warmup = lambda step: min(1., (step+1) / warm_up_steps)
            kwargs['warmup_scheduler'] = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lam_warmup)
        else: 
            kwargs['warmup_scheduler'] = None
        
        for epoch in range(start_epoch, stop_epoch):
            self.current_epoch = epoch
            print('Epoch {}/{}:'.format(epoch, stop_epoch-1), flush=True)
            
            ### Start Train ###
            self.train()
            kwargs = self._on_train_epoch_start(**kwargs)
            
            with tqdm(train_loader, unit="batch", desc='Train: ', colour='#1f77b4',\
                      file=sys.stdout, disable=disabled_tqdm) as tepoch:
                for batch in tepoch:
                    #LR logger 
                    lr_logs = {'lr_{}'.format(i) : float(param_group["lr"]) \
                               for i, param_group in enumerate(self.optimizer.param_groups)}
                    self.log_train(**lr_logs)
                        
                    self.execute_batch_train(batch, **kwargs)
                    tepoch.set_postfix(self.get_train_metrics())
                
            kwargs = self._on_train_epoch_end(**kwargs)
            self.eval()
            ### End Train ###
            
            
            ### Start Valid ###
            with torch.no_grad():
                kwargs = self._on_valid_epoch_start(**kwargs)
                if valid_loader is not None:
                    with tqdm(valid_loader, unit="batch", desc='Valid: ', colour='#ff7f0e', \
                              file=sys.stdout, disable=disabled_tqdm) as tepoch:
                        for batch in tepoch:
                            self.execute_batch_valid(batch, **kwargs)
                            tepoch.set_postfix(self.get_valid_metrics())
                    
                kwargs = self._on_valid_epoch_end(**kwargs)
            ### End Valid ###
            
            
            ### Autosave Model
            current_valids = self.get_valid_metrics()
            if auto_save_path is not None and self.best_valid_label in current_valids.keys() and self.name is not None:
                current = current_valids[self.best_valid_label]
                current_better = self.best_valid_selector(current, self.best_valid) if self.best_valid is not None else current
                
                if self.best_valid is None or current_better == current:
                    self.best_valid = current_better
                    name = self.name + self.best_valid_postfix + '.pt'
                    self.save(auto_save_path, name)
            
        ### Safe final Model
        if auto_save_path is not None and self.name is not None:
            name = self.name + '_final.pt'
            self.save(auto_save_path, name)

        ### Tidy up cuda memory!
        torch.cuda.empty_cache()
        
    
    def execute_batch_train(self, batch, warmup_scheduler=None, **kwargs):
        batch = self._batch_to_device(batch)
        
        if isinstance(batch, (list, tuple)):
            X, y = batch[0], batch[1]
            batch_kwargs = batch[2] if len(batch) >= 3 else {} 
        elif isinstance(batch, dict):
            X, y, batch_kwargs = batch, None, {}
        else:
            X, y, batch_kwargs = batch, None, {}
            
        
        with ampautocast(enabled=self.use_mixed_precision):
            loss = self._train_step(X, y, batch_kwargs, **kwargs)
        
        self.log_train(loss=loss)
        
        if self.use_mixed_precision and self.amp_gradient_scaler.is_enabled():
            self.amp_gradient_scaler.scale(loss).backward()
            self.amp_gradient_scaler.step(self.optimizer)
            self.amp_gradient_scaler.update()
            scale = self.amp_gradient_scaler.get_scale()
            self.log_train(amp_scale=scale)
        else:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if warmup_scheduler is not None:
            warmup_scheduler.step()
            
        del X, y, batch_kwargs, loss

        
    def execute_batch_valid(self, batch, **kwargs):
        batch = self._batch_to_device(batch)
        
        if isinstance(batch, (list, tuple)):
            X, y = batch[0], batch[1]
            batch_kwargs = batch[2] if len(batch) >= 3 else {} 
        elif isinstance(batch, dict):
            X, y, batch_kwargs = batch, None, {}
        else:
            X, y, batch_kwargs = batch, None, {}
        
        with ampautocast(enabled=self.use_mixed_precision):
            loss = self._valid_step(X, y, batch_kwargs, **kwargs)
            
        self.log_valid(loss=loss)
        del X, y, batch_kwargs, loss

    
    def _on_train_epoch_start(self, **kwargs):
        self.train_history[self.current_epoch] = {'loss' : []}
        return kwargs
    
    def _on_valid_epoch_start(self, **kwargs):
        self.valid_history[self.current_epoch] = {'loss' : []}
        return kwargs
    
    def _on_train_epoch_end(self, **kwargs):
        kwargs['warmup_scheduler'] = None
        return kwargs
    
    def _on_valid_epoch_end(self, **kwargs):
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            mean_loss = self.get_valid_metrics()['loss']
            self.scheduler.step(mean_loss)
            
        elif isinstance(self.scheduler, torch.optim.lr_scheduler._LRScheduler):
            self.scheduler.step()
        return kwargs
    
    
    def _train_step(self, X, y, batch_kwargs={}, **kwargs):
        y_pred = self(X)
        for m in self.metrics:
            with torch.no_grad():
                self.log_train(**m(y_pred, y, **batch_kwargs, **kwargs))

        return self.loss(y_pred, y, **batch_kwargs, **kwargs)
        
    
    def _valid_step(self, X, y, batch_kwargs={}, **kwargs):
        y_pred = self(X)
        for m in self.metrics:
            self.log_valid(**m(y_pred, y, **batch_kwargs, **kwargs))
        
        return self.loss(y_pred, y, **batch_kwargs, **kwargs)
        
        
    def log_train(self, **kwargs):
        for key, value in kwargs.items():
            self._log_metric(self.train_history, key, value)
            
    def log_valid(self, **kwargs):
        for key, value in kwargs.items():
            self._log_metric(self.valid_history, key, value)
        
    def _log_metric(self, history, key, value, epoch=None):
        if epoch is None:
            epoch = self.current_epoch
            
        if torch.is_tensor(value):
            value = value.detach().cpu().item()
        
        if key in history[epoch]:
            history[epoch][key].append(value)
        else:
            history[epoch][key] = [value]
    
    def get_train_metrics(self, epoch=None):
        return self._get_metrics(self.train_history, epoch)
    
    def get_valid_metrics(self, epoch=None):
        return self._get_metrics(self.valid_history, epoch)
    
    def _get_metrics(self, history, epoch=None):
        if epoch is None:
            epoch = self.current_epoch
        
        metrics = {}
        for key, values in history[epoch].items():
            if key[:3] == 'lr_' or key[:2] == '__': continue
            metrics[key] = np.array(values).mean()
        
        return metrics
    
    def _create_tensor_from_numpy(self, X, y, *args, valid_split=0.):
        if valid_split > 0.:
            idx = np.arange(X.shape[0])
            idx = np.random.permutation(idx)
            X = X[idx]
            y = y[idx]
            args = args[idx] if args is not None else None
        
        valid_split = int(valid_split * X.shape[0])
        X_train, y_train = torch.from_numpy(X[valid_split:]), torch.from_numpy(y[valid_split:])
        if valid_split > 0:
            X_valid, y_valid = torch.from_numpy(X[:valid_split]), torch.from_numpy(y[:valid_split])
        
        if args:
            m_train = [torch.from_numpy(arg[valid_split:]) for arg in args]
            m_valid = [torch.from_numpy(arg[:valid_split]) for arg in args]
            train_data = TensorDataset(X_train, y_train, *m_train)
            valid_data = TensorDataset(X_valid, y_valid, *m_valid) if valid_split > 0 else None
        else:
            train_data = torch.utils.data.TensorDataset(X_train, y_train)
            valid_data = torch.utils.data.TensorDataset(X_valid, y_valid) if valid_split > 0 else None
        return train_data, valid_data
    
    
    def _batch_to_device(self, batch):
        if torch.is_tensor(batch):
            return batch.to(self.device)

        elif isinstance(batch, list):
            return [self._batch_to_device(b) for b in batch]

        elif isinstance(batch, tuple):
            return tuple(self._batch_to_device(b) for b in batch)

        elif isinstance(batch, dict):
            return {k: self._batch_to_device(v) for k,v in batch.items()}
    
        else:
            return batch