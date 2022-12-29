import os, sys, random, string
import numpy as np
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from torch.cuda.amp import autocast as ampautocast
from torch.cuda.amp import GradScaler as AmpGradScaler
from . import BASE_METRICS, BASE_LOSSES, BASE_SCHEDULERS, BASE_OPTIMIZERS
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

class NimbleModule(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.sequential = self.__model_builder(layers)
        self.reset_parameters()
        self.name = None

    def forward(self, X):
        ## Execute simple forward pass
        X = self._batch_to_device(X)
        return self.__model_exec(self.sequential, X)

    def predict(self, X):
        self.eval()
        X = self._batch_to_device(X)
        with torch.no_grad():
            return self.forward(X)

    def set_device(self, device):
        self.device = device
        return self.to(device=device)

    def cuda(self):
        return self.set_device('cuda')

    def cpu(self):
        return self.set_device('cpu')

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

    def __model_builder(self, layers):
        if isinstance(layers, dict):
            return nn.ModuleDict({key: self.__model_builder(layer) for key, layer in layers.items()})
        elif isinstance(layers, (list, tuple)):
            return nn.ModuleList([self.__model_builder(layer) for layer in layers])
        else:
            return layers

    def __model_exec(self, layers, X):
        if isinstance(layers, nn.ModuleDict) and isinstance(X, dict):
            return {key: self.__model_exec(layer, X[key]) for key, layer in layers.items()}

        elif isinstance(layers, nn.ModuleDict):
            return {key: self.__model_exec(layer, X) for key, layer in layers.items()}

        elif isinstance(layers, nn.ModuleList):
            for layer in layers:
                X = self.__model_exec(layer, X)
            return X

        elif isinstance(layers, nn.Module) and isinstance(X, dict):
            return layers(**X)

        elif isinstance(layers, nn.Module) and isinstance(X, (list, tuple)):
            return layers(*X)

        elif isinstance(layers, nn.Module):
            return layers(X)

        else:
            return X

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


class NimbleSequential(NimbleModule):
    def __init__(self, *layers, name=None):
        super().__init__(*layers)
        self.__set_default(name)

    def __set_default(self, name=None):
        ## General Cofigs
        self.optimizer = None
        self.scheduler = None
        self.loss = None
        self.metrics = []

        ## Train + Valid History
        self.train_history = {}
        self.valid_history = {}
        self.evald_history = {}

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


    def evaluate(self, eval_loader, disabled_tqdm=False, **kwargs):
        self.evald_history = {'loss': []}
        with torch.no_grad():
            with tqdm(eval_loader, unit="batch", desc='Evaluate: ', colour='#2ca02c', \
                      file=sys.stdout, disable=disabled_tqdm) as tepoch:
                for batch in tepoch:
                    self.execute_batch_valid(batch, is_evald=True, **kwargs)
                    tepoch.set_postfix(self.get_evald_metrics())

        return self.get_evald_metrics()


    def evaluate_numpy(self, X_eval, y_eval, kw_eval=None, disabled_tqdm=False, loader_kwargs={}, **kwargs):
        evald_ds, _ = self._create_tensor_from_numpy(X_train, y_train, kw_train)

        eval_loader = torch.utils.data.DataLoader(evald_ds, **loader_kwargs)
        return self.evaluate(self, eval_loader, disabled_tqdm, **kwargs)


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


    def enable_debugging(self, enable=True, record_shapes_emit_nvtx=False, **kwargs):
        torch.autograd.set_detect_anomaly(enable)
        torch.autograd.profiler.emit_nvtx(enabled=enable, record_shapes=record_shapes_emit_nvtx)
        torch.autograd.profiler.profile(enabled=enable, **kwargs)


    def enable_cudnn(self, enable=True, benchmark=False, allow_tf32=False, deterministic=False):
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.enabled = enable
            torch.backends.cudnn.benchmark = benchmark
            torch.backends.cudnn.allow_tf32 = allow_tf32
            torch.backends.cudnn.deterministic = deterministic
        else:
            print('Warning: cudnn not available!')


    def save(self, path, name=None):
        if name is None and self.name is None:
            torch.save(self, path)
        else:
            name = self.name + '.pt' if name is None else name
            path = os.path.join(path, name)
            torch.save(self, path)


    def plot_metric(self, metric='loss', hlines=np.min, hlines_alignment=('top', 'bottom'),
                    kw_figure={"figsize": (6, 5), "dpi": 128},
                    y_limit=None, quantile=(0.1, 0.9), baseline=None, title=None):
        fig = figure(**kw_figure)
        if title is None:
            _ = plt.title(self.name)
        else:
            _ = plt.title(title)

        x = np.arange(self.current_epoch+1)[1:]
        if y_limit is not None:
            _ = plt.ylim(*y_limit)
        _ = plt.ylabel(metric, size='large')
        _ = plt.xlabel('Epoch', size='large')
        _ = plt.xticks(np.arange(self.current_epoch + 1, step=max(self.current_epoch // 10, 1)))

        train_ = self._selected_history(self.train_history, metric, np.mean)
        if train_ is not None:
            _ = plt.plot(x, train_, color='C0')
            _ = plt.fill_between(x, self._selected_history(self.train_history, metric, np.quantile, quantile[1]),
                                     self._selected_history(self.train_history, metric, np.quantile, quantile[0]),
                                 color='C0', alpha=0.25, linewidth=0.)

        valid_ = self._selected_history(self.valid_history, metric, np.mean)
        if valid_ is not None:
            _ = plt.plot(x, valid_, color='C1')
            _ = plt.fill_between(x, self._selected_history(self.valid_history, metric, np.quantile, quantile[1]),
                                     self._selected_history(self.valid_history, metric, np.quantile, quantile[0]),
                                 color='C1', alpha=0.25, linewidth=0.)

        if hlines is not None:
            train_best = hlines(train_)
            valid_best = hlines(valid_)
            if train_best <= valid_best:
                hlines_alignment = (hlines_alignment[0], hlines_alignment[1])
            else:
                hlines_alignment = (hlines_alignment[1], hlines_alignment[0])

            _ = plt.hlines(train_best, 1, self.current_epoch, color='C0', linestyles='dotted')
            _ = plt.hlines(valid_best, 1, self.current_epoch, color='C1', linestyles='dotted')

            _ = plt.text(1, train_best, f'{metric}: {train_best.round(2)}',
                         color='C0', size='x-small', verticalalignment=hlines_alignment[0])
            _ = plt.text(1, valid_best, f'{metric}: {valid_best.round(2)}',
                         color='C1', size='x-small', verticalalignment=hlines_alignment[1])

        if baseline is not None:
            _ = plt.hlines(baseline, 1, self.current_epoch, color='r', linestyles='dotted')
            _ = plt.text(1, baseline, f'Baseline: {baseline}',
                         color='r', size='x-small', verticalalignment='bottom')
        return fig


    def fit_numpy(self, X_train, y_train, kw_train=None,
                  X_valid=None, y_valid=None, kw_valid=None,
                  valid_split=0., epochs=1, warm_up=0, auto_save_path=None, disabled_tqdm=False,
                  loader_kwargs={}, **kwargs):

        if (X_valid is None or y_valid is None) and valid_split > 0.:
            train_ds, valid_ds = self._create_tensor_from_numpy(X_train, y_train, kw_train, valid_split=valid_split)
        elif X_valid is not None and y_valid is not None:
            train_ds,_ = self._create_tensor_from_numpy(X_train, y_train, kw_train)
            valid_ds,_ = self._create_tensor_from_numpy(X_valid, y_valid, kw_valid)
        else:
            train_ds, _ = self._create_tensor_from_numpy(X_train, y_train, kw_train)
            valid_ds = None

        train_loader = torch.utils.data.DataLoader(train_ds, **loader_kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_ds, **loader_kwargs) if valid_ds is not None else None

        self.fit(train_loader, valid_loader, epochs, warm_up, auto_save_path, disabled_tqdm, **kwargs)


    def fit(self, train_loader, valid_loader=None, epochs=1, warm_up=0,
            auto_save_path=None, disabled_tqdm=False, clip_norm_value=None, accumulate_gradients=1, **kwargs):

        ## Append train infos to kwargs
        kwargs['train_loader'] = train_loader
        kwargs['valid_loader'] = valid_loader
        kwargs['epochs'] = epochs
        kwargs['warm_up'] = warm_up
        kwargs['auto_save_path'] = auto_save_path
        kwargs['clip_norm_value'] = clip_norm_value
        kwargs['accumulate_gradients'] = accumulate_gradients
        kwargs['accumulate_gradients_counter'] = 0
        
        ## Create autosave path
        os.makedirs(auto_save_path, exist_ok=True)
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
                    kwargs['accumulate_gradients_counter'] += 1
                    is_final_iter = (kwargs['accumulate_gradients_counter'] >= len(train_loader))
                    self.execute_batch_train(batch, is_final_iter=is_final_iter, **kwargs)
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


        ### Safe final Model
        if auto_save_path is not None and self.name is not None:
            name = self.name + '_final.pt'
            self.save(auto_save_path, name)

        ### Tidy up cuda memory!
        torch.cuda.empty_cache()


    def execute_batch_train(self, batch, is_final_iter=False, warmup_scheduler=None, **kwargs):
        batch = self._batch_to_device(batch)
        #LR logger
        lr_logs = {'lr_{}'.format(i) : float(param_group["lr"]) \
                   for i, param_group in enumerate(self.optimizer.param_groups)}
        self.log_train(**lr_logs)

        if isinstance(batch, (list, tuple)):
            X, y = batch[0], batch[1]
            batch_kwargs = batch[2] if len(batch) >= 3 else {}
        elif isinstance(batch, dict):
            X, y, batch_kwargs = batch, None, {}
        else:
            X, y, batch_kwargs = batch, None, {}


        with ampautocast(enabled=self.use_mixed_precision):
            loss = self._train_step(X, y, batch_kwargs, **kwargs)

        if not isinstance(loss, dict):
            loss = {'loss': loss}
        self.log_train(**loss)
        loss['loss'] = loss['loss'] / kwargs['accumulate_gradients']

        if self.use_mixed_precision and self.amp_gradient_scaler.is_enabled():
            self.amp_gradient_scaler.scale(loss['loss']).backward()
            self.log_norm()
            if is_final_iter or kwargs['accumulate_gradients_counter'] % kwargs['accumulate_gradients'] == 0:
                if 'clip_norm_value' in kwargs and kwargs['clip_norm_value'] is not None:
                    self.amp_gradient_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.sequential.parameters(), kwargs['clip_norm_value'])

                self.amp_gradient_scaler.step(self.optimizer)
                self.amp_gradient_scaler.update()
                ##self.optimizer.zero_grad()
                self._zero_grad(self.sequential)
                self.log_train(__bw_pass=1.)

                scale = self.amp_gradient_scaler.get_scale()
                self.log_train(amp_scale=scale)
        else:
            loss['loss'].backward()
            self.log_norm()
            if is_final_iter or kwargs['accumulate_gradients_counter'] % kwargs['accumulate_gradients'] == 0:
                if 'clip_norm_value' in kwargs and kwargs['clip_norm_value'] is not None:
                    torch.nn.utils.clip_grad_norm_(self.sequential.parameters(), kwargs['clip_norm_value'])
                self.optimizer.step()
                ##self.optimizer.zero_grad()
                self._zero_grad(self.sequential)
                self.log_train(__bw_pass=1.)

        if warmup_scheduler is not None and kwargs['accumulate_gradients_counter'] % kwargs['accumulate_gradients'] == 0:
            warmup_scheduler.step()

        del X, y, batch_kwargs, loss


    def execute_batch_valid(self, batch, is_evald=False, **kwargs):
        batch = self._batch_to_device(batch)

        if isinstance(batch, (list, tuple)):
            X, y = batch[0], batch[1]
            batch_kwargs = batch[2] if len(batch) >= 3 else {}
        elif isinstance(batch, dict):
            X, y, batch_kwargs = batch, None, {}
        else:
            X, y, batch_kwargs = batch, None, {}

        with ampautocast(enabled=self.use_mixed_precision):
            loss = self._valid_step(X, y, batch_kwargs, is_evald=is_evald, **kwargs)

        if not isinstance(loss, dict):
            loss = {'loss': loss}

        if not is_evald:
            self.log_valid(**loss)
        else:
            self.log_evald(**loss)

        del X, y, batch_kwargs, loss


    ## Custom On Event Functions
    def on_train_epoch_start(self, **kwargs):
        return kwargs

    def on_valid_epoch_start(self, **kwargs):
        return kwargs

    def on_train_epoch_end(self, **kwargs):
        return kwargs

    def on_valid_epoch_end(self, **kwargs):
        return kwargs

    ## Buildin On Event Functions
    def _on_train_epoch_start(self, **kwargs):
        self.train_history[self.current_epoch] = {'loss' : []}
        kwargs['accumulate_gradients_counter'] = 0
        ##self.optimizer.zero_grad()
        self._zero_grad(self.sequential)
        return self.on_train_epoch_start(**kwargs)

    def _on_valid_epoch_start(self, **kwargs):
        self.valid_history[self.current_epoch] = {'loss' : []}
        return self.on_valid_epoch_start(**kwargs)

    def _on_train_epoch_end(self, **kwargs):
        kwargs['warmup_scheduler'] = None
        self._zero_grad(self.sequential)
        return self.on_train_epoch_end(**kwargs)

    def _on_valid_epoch_end(self, **kwargs):
        ### Autosave Model
        current_valids = self.get_valid_metrics()
        if kwargs['auto_save_path'] is not None and self.best_valid_label in current_valids.keys() and self.name is not None:
            current = current_valids[self.best_valid_label]
            current_better = self.best_valid_selector(current, self.best_valid) if self.best_valid is not None else current

            if self.best_valid is None or current_better == current:
                self.best_valid = current_better
                name = self.name + self.best_valid_postfix + '.pt'
                self.save(kwargs['auto_save_path'], name)

        ### Scheduler Step
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            mean_loss = self.get_valid_metrics()['loss']
            self.scheduler.step(mean_loss)

        elif isinstance(self.scheduler, torch.optim.lr_scheduler._LRScheduler):
            self.scheduler.step()
        return self.on_valid_epoch_end(**kwargs)



    def _train_step(self, X, y, batch_kwargs={}, **kwargs):
        if isinstance(X, (tuple, list)):
            y_pred = self.forward(*X)
        elif isinstance(X, dict):
            y_pred = self.forward(**X)
        else:
            y_pred = self.forward(X)

        for m in self.metrics:
            with torch.no_grad():
                self.log_train(**m(y_pred, y, **batch_kwargs, training=True, **kwargs))

        return self.loss(y_pred, y, **batch_kwargs, **kwargs)


    def _valid_step(self, X, y, batch_kwargs={}, is_evald=False, **kwargs):
        if isinstance(X, (tuple, list)):
            y_pred = self.forward(*X)
        elif isinstance(X, dict):
            y_pred = self.forward(**X)
        else:
            y_pred = self.forward(X)

        for m in self.metrics:
            if not is_evald:
                self.log_valid(**m(y_pred, y, **batch_kwargs, training=False, **kwargs))
            else:
                self.log_evald(**m(y_pred, y, **batch_kwargs, training=False, **kwargs))

        return self.loss(y_pred, y, **batch_kwargs, **kwargs)


    def _zero_grad(self, model):
        for param in model.parameters():
            if param.requires_grad:
                param.grad = None


    ## Plot Gradient Norm
    def log_norm(self):
        total_norm = 0.
        for p in self.sequential.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.log_train(__grad_norm_2=total_norm)


    def log_train(self, **kwargs):
        for key, value in kwargs.items():
            self._log_metric(self.train_history, key, value)

    def log_valid(self, **kwargs):
        for key, value in kwargs.items():
            self._log_metric(self.valid_history, key, value)

    def log_evald(self, **kwargs):
        for key, value in kwargs.items():
            if torch.is_tensor(value):
                value = value.detach().cpu().item()

            if key in self.evald_history:
                self.evald_history[key].append(value)
            else:
                self.evald_history[key] = [value]


    def _log_metric(self, history, key, value, epoch=None):
        if epoch is None:
            epoch = self.current_epoch

        if torch.is_tensor(value):
            value = value.detach().cpu().tolist()

        if key in history[epoch] and isinstance(value, list):
            history[epoch][key].extend(value)
        elif key in history[epoch] and not isinstance(value, list):
            history[epoch][key].append(value)
        elif key not in history[epoch] and isinstance(value, list):
            history[epoch][key] = value
        else:
            history[epoch][key] = [value]


    def get_train_metrics(self, epoch=None):
        return self._get_metrics(self.train_history, epoch)

    def get_valid_metrics(self, epoch=None):
        return self._get_metrics(self.valid_history, epoch)

    def get_evald_metrics(self):
        return {k: np.array(v).mean() for k, v in self.evald_history.items()}

    def _get_metrics(self, history, epoch=None):
        if epoch is None:
            epoch = self.current_epoch

        metrics = {}
        for key, values in history[epoch].items():
            if key[:3] == 'lr_' or key[:2] == '__': continue
            metrics[key] = np.array(values).mean()

        return metrics

    def get_train_history(self, key, agg=np.mean, *args, **kwargs):
        return self._selected_history(self.train_history, key, agg, *args, **kwargs)

    def get_valid_history(self, key, agg=np.mean, *args, **kwargs):
        return self._selected_history(self.valid_history, key, agg, *args, **kwargs)

    def _selected_history(self, history, key, agg=np.mean, *args, **kwargs):
        metrics = []
        for i in range(1, len(history.keys())+1):
            if key in history[i]:
                v = np.array(history[i][key])
                metrics.append(agg(v, *args, **kwargs))
        return np.array(metrics) if len(metrics) >= 1 else None


    def _create_tensor_from_numpy(self, X, y, kwargs=None, valid_split=0.):
        if valid_split > 0.:
            idx = np.arange(X.shape[0])
            idx = np.random.permutation(idx)
            X = X[idx]
            y = y[idx]
            kwargs = kwargs[idx] if kwargs is not None else None

        valid_split = int(valid_split * X.shape[0])
        X_train, y_train = torch.from_numpy(X[valid_split:]), torch.from_numpy(y[valid_split:])
        X_valid, y_valid = torch.from_numpy(X[:valid_split]), torch.from_numpy(y[:valid_split]) \
                                                                                            if valid_split > 0. else None, None

        if kwargs is not None and isinstance(kwargs, dict):
            kw_train = {key:torch.from_numpy(value[valid_split:]) for key, value in kwargs.items()}
            kw_valid = {key:torch.from_numpy(value[:valid_split]) for key, value in kwargs.items()}
            train_data = TensorDataset(X_train, y_train, kw_train)
            valid_data = TensorDataset(X_valid, y_valid, kw_valid) if valid_split > 0. else None
        else:
            train_data = torch.utils.data.TensorDataset(X_train, y_train)
            valid_data = torch.utils.data.TensorDataset(X_valid, y_valid) if valid_split > 0. else None
        return train_data, valid_data
