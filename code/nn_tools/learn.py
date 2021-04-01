import torch
from torch import nn
import numpy as np
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
from multiprocessing import Pool
from tqdm.notebook import tqdm

from collections import defaultdict


def batch_generator(X, y, batch_size, device='cpu', shuffle=True):
    """
        Генерирует tuple из батча объектов и их меток
        X: torch.tensor -- выборка
        y: torch.tensor -- таргет
        batch_size: int -- размер батча
        device: str -- утсройство, на котором будут производиться вычисления
        shuffle: bool -- перемешивать выборку или нет
    """

    indices = np.arange(len(X))

    # Во время обучения перемешиваем, во время тестирования - нет
    if shuffle:
        indices = np.random.permutation(indices)

    # Идем по всем данным с шагом batch_size.
    # Возвращаем start: start + batch_size объектов на каждой итерации
    for start in range(0, len(indices), batch_size):
        ix = indices[start: start + batch_size]

        # Переведем массивы в соотв. тензоры.
        # Для удобства переместим выборку на наше устройство (GPU).
        yield X[ix], y[ix]


def train(model, criterion, optimizer,
          X_train, y_train, X_val, y_val,
          batch_size=64, num_epochs=100,
          save_path='out/model.model',
          ylim=None, verbose=True):
    """
        Обучает нейронную сеть
        model: nn.Module -- сеть
        criterion: nn.Loss -- функция ошибки
        optimizer: torch.optim.Optimizer -- оптимизатор (SGD, Adam, ...)
        X_train: torch.tensor -- обучающая выборка
        y_train: torch.tensor -- обучающий таргет
        X_val: torch.tensor -- валидационная выборка
        y_val: torch.tensor -- валидационный таргет
        batch_size: int -- размер батча (64, 128, ...)
        num_epochs: int -- количество этапов обучения
        save_path: str -- путь для сохранения наилучшей модели
        ylim: (int, int) or None -- параметр отображения графиков ошибки
        verbose: bool -- нужен ли дебажный вывод
    """
    num_train_batches = len(X_train) // batch_size
    num_val_batches = len(X_val) // batch_size

    history = defaultdict(lambda: defaultdict(list))

    best_val_loss = 1000

    for epoch in range(num_epochs):
        if epoch % 50 == 0 and epoch > 0:
            for g in optimizer.param_groups:
                g['lr'] /= 2.0

        train_loss = 0
        val_loss = 0

        start_time = time.time()

        # Устанавливаем поведение dropout / batch_norm  в обучение
        model.train(True)

        # На каждой "эпохе" делаем полный проход по данным
        for X_batch, y_batch in batch_generator(X_train, y_train, batch_size):
            # Обучаемся на батче (одна "итерация" обучения нейросети)
            logits = model.forward(X_batch)
            loss = criterion.forward(logits, y_batch)
            # Обратный проход, шаг оптимизатора и зануление градиентов
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.detach().cpu().numpy()

        # Подсчитываем лоссы и сохраням в "историю"
        train_loss /= num_train_batches
        history['loss']['train'].append(train_loss)

        # Устанавливаем поведение dropout / batch_norm  в тестирование
        model.eval()

        # Полный проход по валидации
        with torch.no_grad():  # Отключаем подсчет градиентов, то есть detach не нужен
            for X_batch, y_batch in batch_generator(X_val, y_val, batch_size):
                logits = model.forward(X_batch)
                loss = criterion.forward(logits, y_batch)

                val_loss += loss.cpu().numpy()

        # Подсчитываем лоссы и сохраням в "историю"
        val_loss /= num_val_batches
        history['loss']['val'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        # Печатаем результаты после каждой эпохи
        if verbose:
            clear_output(wait=True)
            plt.figure(figsize=(14, 7))
            plt.plot(history['loss']['val'], label='validation')
            plt.plot(history['loss']['train'], label='train')
            plt.xlabel('Номер итерации обучения')
            plt.ylabel('Loss function')
            plt.legend()
            plt.ylim(ylim)
            plt.show()
    return history


def accumulate_histories(train_step, num_executions=100):
    """
        Производит серию запусков обучения с целью анализа ошибки.
        train_step -- функция обучения
        num_executions -- количество запусков фцнкции обучения
    """
    histories = []
    for no in tqdm(np.arange(num_executions)):
        histories.append(train_step(no))

    loss_val = np.array([x['loss']['val'] for x in histories])
    loss_train = np.array([x['loss']['train'] for x in histories])
    return {
        'mean': {'val': loss_val.mean(axis=0), 'train': loss_train.mean(axis=0)},
        'std': {'val': loss_val.std(axis=0), 'train': loss_train.std(axis=0)}
    }
