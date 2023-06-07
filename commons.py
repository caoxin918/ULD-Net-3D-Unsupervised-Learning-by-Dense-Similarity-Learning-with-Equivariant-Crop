from modules.ContrastModel import DenseSimSiam
from utils import *
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def get_model_and_optimizer(args, logger):

    # Init model 
    model = DenseSimSiam(args=args)
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()

    # Init classifier (for eval only.)
    classifier = initialize_classifier(args)

    # Init optimizer 
    if args.optim_type == 'SGD':
        logger.info('SGD optimizer is used.')
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.module.parameters()), lr=args.lr, \
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim_type == 'Adam':
        logger.info('Adam optimizer is used.')
        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, list(model.module.parameters())+list(classifier.module.parameters())), lr=args.lr)
   
    # optional restart. 
    args.start_epoch  = 0
    if args.restart or args.eval_only:
        load_path = os.path.join(args.restart_path, args.eval_path)

        if os.path.isfile(load_path):
            checkpoint  = torch.load(load_path)
            # state_dict = model.module.encoder.state_dict()
            # pretrained_dict = checkpoint["state_dict"]
            # # print('====pretrain')
            # # for key in pretrained_dict:
            # #     print(key)
            # # print('====model')
            # for key in state_dict:
            #     # print(key)
            # #     if "online_network." + key in pretrained_dict:
            #     if "module.encoder." + key in pretrained_dict:
            #             # and (not "classifier"in key) and (not "conv8.0"in key):
            #         print('====load===='+key)
            #         state_dict[key] = pretrained_dict["module.encoder." + key]
            #         # state_dict[key] = pretrained_dict["online_network." + key]
            # #         print(key)
            #     else:
            #         print('====no'+ key +'in pretrained model, ignore it.====' )
            # model.module.encoder.load_state_dict(state_dict, strict=True)
            # trans_model(model, checkpoint)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            # classifier.load_state_dict(checkpoint['classifier_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info('Loaded checkpoint at [epoch {}] from {}'.format(args.start_epoch, load_path))
        else:
            logger.info('No checkpoint found at [{}].\nStart from beginning...\n'.format(load_path))
    
    return model, optimizer, classifier
def load_model(weight_path, model):
    state_dict = model.state_dict()

    ckpt = torch.load(weight_path, map_location="cpu")
    pretrained_dict = ckpt["state_dict"]

    for key in state_dict:
        if "online_network." + key in pretrained_dict:
            state_dict[key] = pretrained_dict["online_network."+key]
            print(key)
        # if "online_netwrok."+key in pretrained_dict:
        #     state_dict[key] = pretrained_dict["online_netwrok."+key]

    model.load_state_dict(state_dict, strict=True)
    return model
def trans_model(model, checkpoint):
    state_dict = model.module.encoder.state_dict()
    # model_
    pretrained_dict = checkpoint["state_dict"]
    print('====pretrain')
    for key in pretrained_dict:
        print(key)
    print('====model')
    for key in state_dict:
        # print(key)
    #     if "online_network." + key in pretrained_dict:
    #     if "module.encoder." + key in pretrained_dict:
    #     "module." +
        if key in pretrained_dict:
                # and (not "classifier"in key) and (not "conv8.0"in key):
            print('====load===='+key)
            state_dict[key] = pretrained_dict[key]
            # state_dict[key] = pretrained_dict["online_network." + key]
        else:
            print('====no'+ key +'in pretrained model, ignore it.====' )
    model.module.encoder.load_state_dict(state_dict, strict=True)
    return model

def fetch_represent(logger, dataloader, model):
    representations = None
    labels = None
    model.eval()
    with torch.no_grad():
        for i, (cls, image) in enumerate(dataloader):
            image = image.cuda(non_blocking=True)
            _,feats = model(image.transpose(1, 2), get_feature=True)
            # B, C, N = feats.size()
            if i == 0:
                logger.info('Batch input size   : {}'.format(list(image.shape)))
            representation = feats.data.cpu().numpy()
            label = np.array(cls)
            if representations is None:
                representations = representation
                labels = label
            else:
                representations = np.concatenate([representations, representation], 0)
                labels = np.concatenate([labels, label], 0)
    model.train()
    return representations, labels

def filter_MN10(represent, label):
    ModelNet10lbl = [1, 2, 8, 12, 14, 22, 23, 30, 33, 35]
    res_rep = []
    res_label = []
    for lbl in ModelNet10lbl:
        res_rep.append(represent[label==lbl])
        res_label.append(label[label==lbl])
    res_rep = np.concatenate(res_rep, 0)
    res_label = np.concatenate(res_label, 0)
    return res_rep, res_label

# SVM evaluate
def evaluate(args, logger, eval_loader, test_loader, model):
    if args.noise > 0:
        logger.info("robustness test with std:{}".format(args.noise))
    print("Fetch Train Data Representation")
    train_represent, train_label = fetch_represent(logger, eval_loader, model)

    print("Fetch Val Data Representation")
    test_represent, test_label = fetch_represent(logger, test_loader, model)

    if args.dataset == "ModelNet10":
        train_represent, train_label = filter_MN10(train_represent, train_label)
        test_represent, test_label = filter_MN10(test_represent, test_label)

    from sklearn.svm import LinearSVC
    clf = LinearSVC()
    clf.fit(train_represent, train_label)
    pred = clf.predict(test_represent)
    score = np.sum(test_label == pred) * 1. / pred.shape[0]
    logger.info("LinearSVC Val Accuracy:{}".format(score) )
    from sklearn.svm import SVC
    svc = SVC(kernel="linear")
    svc.fit(train_represent, train_label)

    score2 = svc.score(test_represent, test_label)
    logger.info("STRL SVC Val Accuracy:{}".format(score2))
    model.train()
    return max(score, score2)