import torch
from torch import nn
import numpy as np
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
from multiprocessing import Pool
from tqdm.notebook import tqdm

from collections import defaultdict


def train(model, criterion, optimizer,
          train_batch_generator,
          val_batch_generator,
          num_epochs=100, device='cpu',
          save_path='out/model.model',
          ylim_loss=None, ylim_mape=None, verbose=True):
    """
        Обучает нейронную сеть
        model: nn.Module -- сеть
        criterion: nn.Loss -- функция ошибки
        optimizer: torch.optim.Optimizer -- оптимизатор (SGD, Adam, ...)
        train_batch_generator: torch.utils.data.DataLoader -- генератор батчей для теста
        val_batch_generator: torch.utils.data.DataLoader -- генератор батчей для теста
        num_epochs: int -- количество этапов обучения
        device: str -- устройство для обучения
        save_path: str -- путь для сохранения наилучшей модели
        ylim: (int, int) or None -- параметр отображения графиков ошибки
        verbose: bool -- нужен ли дебажный вывод
    """
    history = defaultdict(lambda: defaultdict(list))

    best_val_mape = 1.0

    for epoch in range(num_epochs):
        if epoch % 50 == 0 and epoch > 0:
            for g in optimizer.param_groups:
                g['lr'] /= 2.0

        train_loss = 0
        val_loss = 0
        train_mape = 0
        val_mape = 0

        start_time = time.time()

        # Устанавливаем поведение dropout / batch_norm  в обучение
        model.train(True)

        # На каждой "эпохе" делаем полный проход по данным
        for X_batch, y_batch in train_batch_generator:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Обучаемся на батче (одна "итерация" обучения нейросети)
            logits = model.forward(X_batch)
            loss = criterion.forward(logits, y_batch)
            # Обратный проход, шаг оптимизатора и зануление градиентов
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.detach().cpu().numpy()
            target = y_batch.cpu().numpy()
            train_mape += np.mean(np.abs(logits.detach().cpu().numpy() - target) / target)

        # Подсчитываем лоссы и сохраням в "историю"
        train_loss /= len(train_batch_generator)
        train_mape /= len(train_batch_generator)
        history['loss']['train'].append(train_loss)
        history['mape']['train'].append(train_mape)

        # Устанавливаем поведение dropout / batch_norm  в тестирование
        model.train(False)

        # Полный проход по валидации
        with torch.no_grad():  # Отключаем подсчет градиентов, то есть detach не нужен
            for X_batch, y_batch in val_batch_generator:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                logits = model.forward(X_batch)
                loss = criterion.forward(logits, y_batch)

                val_loss += loss.cpu().numpy()
                target = y_batch.cpu().numpy()
                val_mape += np.mean(np.abs(logits.cpu().numpy() - target) / target)

        # Подсчитываем лоссы и сохраням в "историю"
        val_loss /= len(val_batch_generator)
        val_mape /= len(val_batch_generator)
        history['loss']['val'].append(val_loss)
        history['mape']['val'].append(val_mape)

        if val_mape < best_val_mape:
            best_val_mape = val_mape
            torch.save(model.state_dict(), save_path)

        # Печатаем результаты после каждой эпохи
        if verbose:
            clear_output(wait=True)
            print('Time elapsed: {:.3f}s'.format(time.time() - start_time))
            plt.figure(figsize=(18, 7))
            plt.subplot(1, 2, 1)
            plt.plot(history['loss']['train'], label='train')
            plt.plot(history['loss']['val'], label='validation')
            plt.xlabel('Номер итерации обучения')
            plt.ylabel('MSE')
            plt.legend()
            plt.ylim(ylim_loss)
            plt.subplot(1, 2, 2)
            plt.plot(history['mape']['train'], label='train')
            plt.plot(history['mape']['val'], label='validation')
            plt.xlabel('Номер итерации обучения')
            plt.ylabel('MAPE')
            plt.legend()
            plt.ylim(ylim_mape)
            plt.show()
    return history


def accumulate_histories(train_step, num_executions=100):
    """
        Производит серию запусков обучения с целью анализа ошибки.
        train_step -- функция обучения
        num_executions -- количество запусков фцнкции обучения
    """
    histories = []
    for no in range(num_executions):
        histories.append(train_step(no))

    loss_val = np.array([x['loss']['val'] for x in histories])
    loss_train = np.array([x['loss']['train'] for x in histories])
    mape_val = np.array([x['mape']['val'] for x in histories])
    mape_train = np.array([x['mape']['train'] for x in histories])
    return {
        'loss': {
            'mean': {'val': loss_val.mean(axis=0), 'train': loss_train.mean(axis=0)},
            'std': {'val': loss_val.std(axis=0), 'train': loss_train.std(axis=0)},
            'all': {'val': loss_val, 'train': loss_train}
        },
        'mape': {
            'mean': {'val': mape_val.mean(axis=0), 'train': mape_train.mean(axis=0)},
            'std': {'val': mape_val.std(axis=0), 'train': mape_train.std(axis=0)},
            'all': {'val': mape_val, 'train': mape_train}
        }
    }


def visualize_histories(history, figsize=(14, 7), q=(0.05, 0.95), ylim_loss=None, ylim_mape=None):
    """
        Визуализирует данные, полученные функцией accumulate_histories
        figsize: (int, int) -- размер фигуры для графиков
        q: (float, float) -- уровни квантилей
    """
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.plot(history['loss']['mean']['train'], label='train')
    plt.fill_between(
        np.arange(len(history['loss']['mean']['train'])),
        np.quantile(history['loss']['all']['train'], q[0], axis=0),
        np.quantile(history['loss']['all']['train'], q[1], axis=0),
        color='C0', alpha=0.5)
    plt.plot(history['loss']['mean']['val'], label='validation')
    plt.fill_between(
        np.arange(len(history['loss']['mean']['val'])),
        np.quantile(history['loss']['all']['val'], q[0], axis=0),
        np.quantile(history['loss']['all']['val'], q[1], axis=0),
        color='C1', alpha=0.5)
    plt.legend()
    plt.ylim(ylim_loss)
    plt.xlabel('Номер итерации')
    plt.ylabel('MSE')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['mape']['mean']['train'], label='train')
    plt.fill_between(
        np.arange(len(history['mape']['mean']['train'])),
        np.quantile(history['mape']['all']['train'], q[0], axis=0),
        np.quantile(history['mape']['all']['train'], q[1], axis=0),
        color='C0', alpha=0.5)
    plt.plot(history['mape']['mean']['val'], label='validation')
    plt.fill_between(
        np.arange(len(history['mape']['mean']['val'])),
        np.quantile(history['mape']['all']['val'], q[0], axis=0),
        np.quantile(history['mape']['all']['val'], q[1], axis=0),
        color='C1', alpha=0.5)
    plt.legend()
    plt.ylim(ylim_mape)
    plt.xlabel('Номер итерации')
    plt.ylabel('MAPE')
