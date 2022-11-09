import os
import time
import tqdm
import torch
import argparse
import numpy as np
import os.path as osp
from loss import DiceLoss
from manet import MANet
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from dataset.vaihingen_dataset import VaihingenDataset
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu, poly_lr_scheduler


def val(args, model, dataloader):
    print('###---start val---###')
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))

        for i, results in enumerate(dataloader):
            if torch.cuda.is_available() and args.use_gpu:
                data = results['img'].cuda()
                label = results['gt_semantic_seg'].cuda()

            predict = model(data)

            predict = torch.nn.Softmax(dim=1)(predict)
            predict = predict.argmax(dim=1)

            # predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # if args.loss == 'dice':
            #     label = reverse_one_hot(label)
            label = np.array(label.cpu())

            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            precision_record.append(precision)

        mean_precision = np.mean(precision_record)

        iou_list = per_class_iu(hist)

        miou = np.mean(iou_list)

        print('mean precision per pixel for test : %.3f' % mean_precision)
        print('mIoU for test : %.3f' % miou)
        print('IoU per class for test : {}' .format(np.round(iou_list, 3)))

    return mean_precision, miou

def train(args, model, optimizer, dataloader_train, dataloader_val):

    folder_name = '%s_%s' % (time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time())),
                            args.net_name)
    save_folder = osp.join('results/', args.dataset, folder_name)
    save_pkl = osp.join(save_folder, 'checkpoint')
    if not osp.exists('results'):
        os.mkdir('results')
    if not osp.exists(osp.join('results/', args.dataset)):
        os.mkdir(osp.join('results/', args.dataset))
    if not osp.exists(save_folder):
        os.mkdir(save_folder)
    if not osp.exists(save_pkl):
        os.mkdir(save_pkl)

    writer = SummaryWriter(log_dir=save_folder)
    writer.add_text(folder_name, 'Args:%s,' % args)

    if args.loss == 'dice':
        loss_func = DiceLoss(smooth=0.05, ignore_index=None)

    max_iou = 0

    for epoch in range(args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))

        loss_record = []
        for i, results in enumerate(dataloader_train):
            if torch.cuda.is_available() and args.use_gpu:
                data = results['img'].cuda()
                label = results['gt_semantic_seg'].cuda()

            output = model(data)

            loss = loss_func(output, label)

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_record.append(loss.item())
        
        tq.close()
        mean_train_loss = np.mean(loss_record)
        writer.add_scalar('epoch/mean_train_loss', float(mean_train_loss), epoch)
        print('mean_train_loss : %f' % (mean_train_loss))

        if epoch % args.checkpoint_step == 0 and epoch != 0:
            if not osp.isdir(save_pkl):
                os.mkdir(save_pkl)
            torch.save(model.module.state_dict(), osp.join(save_pkl, 'latest_model.pth'))

        if epoch % args.validation_step == 0:
            mean_precision, miou = val(args, model, dataloader_val)
            if miou > max_iou:
                max_iou = miou
                torch.save(model.module.state_dict(), osp.join(save_pkl, 'best_miou_model.pth'))

            writer.add_scalar('epoch/mean_val_precision', mean_precision, epoch)
            writer.add_scalar('epoch/val_miou', miou, epoch)



def main(params):
        # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=10, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="vaihingen", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of images in each batch')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--data', type=str, default='../DATASET/convert_vaihingen/', help='path of training data')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=12, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='dice', help='loss function, dice or crossentropy')
    parser.add_argument('--net-name', type=str, default='manet', help='net name: fcn')


    args = parser.parse_args(params)

    train_dataset = VaihingenDataset(data_root=args.data, mode='train')
    val_dataset = VaihingenDataset(data_root=args.data, mode='test')

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
        )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
        )

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    model = MANet(num_classes=args.num_classes)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None



    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)
        model.module.load_state_dict(torch.load(args.pretrained_model_path))
        print('Done!')


    train(args, model, optimizer, train_dataloader, val_dataloader)


if __name__ == '__main__':
    params = [
        '--num_epochs', '50',
        '--learning_rate', '2.5e-3',
        '--num_workers', '4',
        '--num_classes', '7',
        '--cuda', '0',
        '--batch_size', '4',
        '--save_model_path', './results/',
        '--optimizer', 'sgd',

    ]
    main(params)
