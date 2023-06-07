import argparse
import time as t
from commons import *
import torch.nn.functional as F
from info_nce import InfoNCE

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./dataset/modelnet40/')
    parser.add_argument('--save_root', type=str, default='results/ULD-Net/train/')
    parser.add_argument('--restart_path', type=str, default='restore/')
    parser.add_argument('--comment', type=str, default='train')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed for reproducability.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers.')
    parser.add_argument('--restart', action='store_true', default=False)
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--num_points', type=int, default=2048, help='num of points to use')
    parser.add_argument('--noise', type=float, default=0, help='noise deviation in robustness test')
    parser.add_argument('--dataset', type=str, default='modelnet')

    # Train.
    parser.add_argument('--res1', type=int, default=2048, help='Input size scale from.')
    parser.add_argument('--res2', type=int, default=2048, help='Input size scale to.')
    parser.add_argument('--batch_size_eval', type=int, default=16)
    parser.add_argument('--batch_size_train', type=int, default=24)
    parser.add_argument('--batch_size_test', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optim_type', type=str, default='Adam')
    parser.add_argument('--emb_dims', type=int, default=64)
    parser.add_argument('--normal_channel', default=False)
    parser.add_argument('--knn', type=int, default=20)
    parser.add_argument('--decay_epoch', type=int, default=20)

    # Loss. 
    parser.add_argument('--metric_train', type=str, default='cosine')

    # Dataset. 
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--equiv', action='store_true', default=True)
    parser.add_argument('--min_scale', type=float, default=0.5)

    # inv augment.
    parser.add_argument('--scale', action='store_true', default=True)
    parser.add_argument('--rotateperturbation', action='store_true', default=True)
    parser.add_argument('--jitter', action='store_true', default=True)
    parser.add_argument('--normalize', action='store_true', default=True)
    parser.add_argument('--translate', action='store_true', default=True)
    # eqv augment.
    parser.add_argument('--randomcrop', action='store_true', default=True)
    parser.add_argument('--val_type', type=str, default='train')
    # Eval-only
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--eval_path', type=str, default= 'checkpoint_best_svm.pth.tar')

    return parser.parse_args()





def train(args, logger, dataloader, model, classifier, optimizer, epoch):
    losses = AverageMeter()
    simsiam = nn.CosineSimilarity(1)
    ce = InfoNCE()

    # switch to train mode
    model.train()
    classifier.train()

    for i, data in enumerate(dataloader):
        (indice, input1, input2) = data
        input1, _ = eqv_transform_if_needed(args, dataloader, indice, input1.cuda(non_blocking=True), None)
        input2 = input2.cuda(non_blocking=True)
        p1, p2, feat1, feat2, pp1, pp2, featmap1, featmap2, x_z1, x_z2 = model(input1.transpose(1,2), input2.transpose(1, 2))
        _, featmap2 = eqv_transform_if_needed(args, dataloader, indice, input2, featmap2.transpose(1, 2))
        _, x_z2 = eqv_transform_if_needed(args, dataloader, indice, input2, x_z2.transpose(1, 2))
        input2, pp2 = eqv_transform_if_needed(args, dataloader, indice, input2, pp2.transpose(1, 2))
        B, N, P = x_z1.size()[:3]
        _, C, _ = featmap1.size()[:3]
        x_z1 = torch.matmul(featmap1, x_z1.transpose(1,2))
        x_z2 = torch.matmul(featmap2.transpose(1,2), x_z2)
        region1_pred, region2_pred, region1, region2 = classifier(x_z1, x_z2)
        del x_z1, x_z2
        featmap1, pp1 = featmap1.transpose(1, 2).reshape(-1, C), pp1.transpose(1, 2).reshape(-1, C)
        region1_pred, region2_pred, region1, region2 = region1_pred.reshape(-1,N), region2_pred.reshape(-1,N), region1.reshape(-1,N), region2.reshape(-1,N)
        featmap2, pp2 = featmap2.reshape(-1, C), pp2.reshape(-1, C)
        if i == 0:
            logger.info('Batch input size   : {}'.format(list(input1.shape)))
            logger.info('Batch region size   : {}'.format(list(region1.shape)))
            logger.info('Batch feature size : {}\n'.format(list(featmap1.shape)))
        
        if args.metric_train == 'cosine':
            p1 = F.normalize(p1, dim=1, p=2)
            p2 = F.normalize(p2, dim=1, p=2)
            feat1 = F.normalize(feat1, dim=1, p=2)
            feat2 = F.normalize(feat2, dim=1, p=2)
            featmap1 = F.normalize(featmap1, dim=1, p=2)
            featmap2 = F.normalize(featmap2, dim=1, p=2)
            pp1 = F.normalize(pp1, dim=1, p=2)
            pp2 = F.normalize(pp2, dim=1, p=2)
            region1_pred, region2_pred, region1, region2 = F.normalize(region1_pred, dim=1, p=2),F.normalize(region2_pred, dim=1, p=2),F.normalize(region1, dim=1, p=2),F.normalize(region2, dim=1, p=2)

        loss_simsiam = 1 - (simsiam(p1, feat2).mean() + simsiam(p2, feat1).mean()) * 0.5

        within_sim = 1 - simsiam(pp1, featmap2).mean()
        cross_sim = 1 - simsiam(pp2, featmap1).mean()
        loss_sim = 0.5 * within_sim + 0.5 * cross_sim

        loss_region = 0.5*ce(region1_pred, region2) + 0.5*ce(region2_pred, region1)

        if epoch > -1:
            loss =  loss_sim * 100 + loss_simsiam * 100 + loss_region/B * 10
        else:
            loss = loss_sim * 100 + loss_simsiam * 100
        
        # record loss
        losses.update(loss.item(), B)
        if (i % 20) == 0:
            logger.info('{0} / {1}\t'.format(i, len(dataloader)))
            logger.info('loss:{:.4f} | sim:{:.4f} | region:{:.4f} | simsiam:{:.4f}'.format(loss.item(), loss_sim, loss_region.item(), loss_simsiam.item()))
        del loss_sim, loss_simsiam, loss_region, within_sim, cross_sim

        # compute gradient and do step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del loss,featmap1,featmap2, feat1, feat2, region1, region2


    return losses.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = args.lr * (0.7 ** (epoch // args.decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main(args, logger):
    logger.info(args)

    # Use random seed.
    if args.seed > 0:
         fix_seed_for_reproducability(args.seed)

    # Start time.
    t_start = t.time()

    # Get model and optimizer.
    model, optimizer, classifier = get_model_and_optimizer(args, logger)

    # New trainset inside for-loop.
    if args.dataset == 'shapenet':
        args.data_root = './dataset/'
    inv_list, eqv_list = get_transform_params(args)
    trainset = get_dataset(args, mode='train', inv_list=inv_list, eqv_list=eqv_list)

    trainset.mode = 'compute'

    args.dataset = 'modelnet'
    args.data_root = './dataset/modelnet40/'
    evalset = get_dataset(args, mode='train', inv_list=inv_list, eqv_list=eqv_list)

    evalloader = torch.utils.data.DataLoader(evalset,
                                                batch_size=args.batch_size_eval,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True,
                                                collate_fn=collate_eval,
                                                worker_init_fn=worker_init_fn(args.seed))
    evalset.mode = 'compute'

    testset = get_dataset(args, mode='train_val')
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size_test,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=True,
                                             collate_fn=collate_eval,
                                             worker_init_fn=worker_init_fn(args.seed))

    if os.path.exists(args.restart_path):
        checkpoint = torch.load(args.restart_path)
        model.load_state_dict(checkpoint)

    best_acc = 0.0
    if not args.eval_only:
        # Train start.
        for epoch in range(args.start_epoch, args.num_epoch):

            # Adjust lr if needed. 
            adjust_learning_rate(optimizer, epoch, args)

            # Set-up train loader.
            trainloader_loop = torch.utils.data.DataLoader(trainset,
                                                            batch_size=args.batch_size_train, 
                                                            shuffle=True,
                                                            num_workers=args.num_workers,
                                                            pin_memory=True,
                                                            collate_fn=collate_train,
                                                            worker_init_fn=worker_init_fn(args.seed))
            trainloader_loop.dataset.reshuffle(args.num_points)
            trainloader_loop.dataset.mode = 'train'
            logger.info('Start training ...')
            train_loss = train(args, logger, trainloader_loop, model.train(), classifier.train(), optimizer, epoch)

            logger.info('============== Epoch [{}] =============='.format(epoch))
            logger.info('  Training Total Loss  : {:.5f}'.format(train_loss))
            if epoch % 1 == 0 or epoch > 50:
                evalloader.dataset.mode = 'compute'
                acc1 = evaluate(args, logger, evalloader, testloader, model.module.encoder.eval())
                if acc1 > best_acc:
                    best_acc = acc1
                logger.info('ACC: {:.4f}   Best ACC: {:.4f}'.format(acc1, best_acc))
                logger.info('========================================\n')
            

            torch.save({'epoch': epoch+1, 
                        'args' : args,
                        'state_dict': model.state_dict(),
                        'classifier_state_dict' : classifier.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        },
                        os.path.join(args.save_model_path, 'checkpoint_{}.pth.tar'.format(epoch)))
            
            torch.save({'epoch': epoch+1, 
                        'args' : args,
                        'state_dict': model.state_dict(),
                        'classifier_state_dict' : classifier.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        },
                        os.path.join(args.save_model_path, 'checkpoint.pth.tar'))
    else:
        # Evaluate.
        evalloader.dataset.reshuffle(args.num_points)
        evalloader.dataset.mode = 'compute'
        evaluate(args, logger, evalloader, testloader, model.module.encoder.eval())
        logger.info('Experiment done. [{}]\n'.format(get_datetime(int(t.time())-int(t_start))))
        
        
if __name__=='__main__':

    args = parse_arguments()

    # Setup the path to save.
    args.save_model_path = os.path.join(args.save_root, args.comment)
    args.save_eval_path = args.save_model_path
    if not os.path.exists(args.save_eval_path):
        os.makedirs(args.save_eval_path)

    # Setup logger.
    logger = set_logger(os.path.join(args.save_eval_path, 'train.log'))
    
    # Start.
    main(args, logger)
