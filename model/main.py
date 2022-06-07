import os
import time
from tqdm import tqdm

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler, SequentialSampler
from torch_geometric.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW, lr_scheduler

from config import parser
from evaluate import evaluate

from DatasetLoader import DatasetLoader
from ModelFramework import EnvEnhancedFramework


if __name__ == "__main__":
    args = parser.parse_args()

    if args.debug:
        args.save = './ckpts/debug'
        args.epochs = 2

    if os.path.exists(args.save):
        os.system('rm -r {}'.format(args.save))
    if not os.path.exists(args.save):
        os.mkdir(args.save)

    print('\n{} Experimental Dataset: {} {}\n'.format(
        '=' * 20, args.dataset, '=' * 20))
    print('save path: ', args.save)
    print('Start time:', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    print('Loading data...')
    start = time.time()

    train_dataset = DatasetLoader(args, 'train')
    val_dataset = DatasetLoader(args, 'val')
    test_dataset = DatasetLoader(args, 'test')

    if not args.evaluate:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = SequentialSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    test_sampler = SequentialSampler(test_dataset)

    if not args.evaluate:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=8,
            pin_memory=(torch.cuda.is_available()),
            drop_last=False,
            sampler=train_sampler
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=(torch.cuda.is_available()),
            drop_last=False,
            sampler=train_sampler
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=(torch.cuda.is_available()),
        drop_last=False,
        sampler=val_sampler
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=(torch.cuda.is_available()),
        drop_last=False,
        sampler=test_sampler
    )

    print('Loading data time: {:.2f}s\n'.format(time.time() - start))

    print('-----------------------------------------\nLoading model...\n')
    start = time.time()
    model = EnvEnhancedFramework(args)
    print(model)
    print('\nLoading model time: {:.2f}s\n-----------------------------------------\n'.format(
        time.time() - start))

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = AdamW(filter(lambda p: p.requires_grad,
                             model.parameters()), lr=args.lr)

    if args.fp16:
        scaler = GradScaler()

    model = model.cuda()

    if args.resume != '':
        resume_dict = torch.load(args.resume)
        model.load_state_dict(resume_dict['state_dict'])
        optimizer.load_state_dict(resume_dict['optimizer'])
        args.start_epoch = resume_dict['epoch'] + 1

    with open(os.path.join(args.save, 'args.txt'), 'w') as f:
        print('\n---------------------------------------------------\n')
        print('[Arguments] \n')
        for arg in vars(args):
            v = getattr(args, arg)
            s = '{}\t{}'.format(arg, v)
            f.write('{}\n'.format(s))
            print(s)
        print('\n---------------------------------------------------\n')
        f.write('\n{}\n'.format(model))

    # Only evaluate
    if args.evaluate:
        if not args.resume:
            print('No trained .pt file loaded.\n')

        print('Start Evaluating... local_rank=', args.local_rank)
        args.current_epoch = args.start_epoch

        train_losses, _ = evaluate(
            args, train_loader, model, criterion, 'train', inference_analysis=args.inference_analysis)
        val_losses, _ = evaluate(args, val_loader, model, criterion,
                                 'val', inference_analysis=args.inference_analysis)
        test_losses, _ = evaluate(
            args, test_loader, model, criterion, 'test', inference_analysis=args.inference_analysis)

        exit()

    last_epoch = args.start_epoch if args.start_epoch != 0 else -1

    # Training
    if args.local_rank in [-1, 0]:
        print('Start training...')

    # Best results on validation dataset
    best_val_result = 0
    best_val_epoch = -1

    start = time.time()
    args.global_step = 0
    for epoch in range(args.start_epoch, args.epochs):
        args.current_epoch = epoch
        print('\n------------------------------------------------\n')
        print('Start Training Epoch', epoch, ':', time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
        model.train()

        train_loss = 0.

        lr = optimizer.param_groups[0]['lr']
        print_step = int(len(train_loader) / 20)
        print_step = 10
        for step, (idxs, labels) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            with autocast():
                # (bs, category_num)
                out, h_mac, h_mic = model(idxs, train_dataset)

                if args.model == 'EANN':
                    out, event_out = out
                    event_labels = train_dataset.event_labels[idxs]

                labels = labels.long().to(args.device)
                CEloss = criterion(out, labels)

                if args.model == 'EANN':
                    event_loss = criterion(event_out, event_labels)
                    event_loss = args.eann_weight_of_event_loss * event_loss
                    CEloss += event_loss

                loss = CEloss

                if torch.any(torch.isnan(loss)):
                    print('out: ', out)
                    print('loss = {:.4f}\n'.format(loss.item()))
                    exit()

                if step % print_step == 0:
                    print('\n\nEpoch: {}, Step: {}, CELoss = {:.4f}'.format(
                        epoch, step, CEloss.item()))

            if args.fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            args.global_step += 1

        train_loss /= len(train_loader)
        val_loss, val_result = evaluate(
            args, val_loader, model, criterion, 'val')
        test_loss, test_result = evaluate(
            args, test_loader, model, criterion, 'test')

        print('='*10, 'Epoch: {}/{}'.format(epoch, args.epochs),
              'lr: {}'.format(lr), '='*10)
        print('\n[Loss]\nTrain: {:.6f}\tVal: {:.6f}\tTest: {:.6f}'.format(
            train_loss, val_loss, test_loss))
        print('-'*10)
        print('\n[Macro F1]\nVal: {:.6f}\tTest: {:.6f}\n'.format(
            val_result, test_result))
        print('-'*10)

        if val_result >= best_val_result:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            },
                os.path.join(args.save, '{}.pt'.format(epoch))
            )

            if best_val_epoch != -1:
                os.system('rm {}'.format(os.path.join(
                    args.save, '{}.pt'.format(best_val_epoch))))

            best_val_result = val_result
            best_val_epoch = epoch

    print('Training Time: {:.2f}s'.format(time.time() - start))
    print('End time:', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
