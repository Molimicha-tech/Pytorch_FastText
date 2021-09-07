import errno

import pandas as pd
import numpy as np
import argparse
import sys
import os

from utils import word2id
from model import FastText

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

def train_model(train_loader,test_loader,model,args):
    '''
    train model
    :param train_loader:
    :param test_loader:
    :param model:
    :param args:
    :return:
    '''
    # optimization scheme
    if args.optimizer == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    # loss function
    criterion = nn.CrossEntropyLoss()

    # continue training from checkpoint model
    if args.continue_from:
        print("=> loading checkpoint from '{}'".format(args.continue_from))
        assert os.path.isfile(args.continue_from), "=> no checkpoint found at '{}'".format(args.continue_from)
        checkpoint = torch.load(args.continue_from)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint.get('best_acc', None)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        start_epoch = 1
        best_acc = None

    model.train()

    epoch = start_epoch
    while epoch <= args.eopch:
        for i_batch, batch in enumerate(train_loader):
            #print(i_batch)
            input_ids = batch[0]
            labels = batch[1]

            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            #loss = Variable(loss, requires_grad=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # training log
            if i_batch % args.log_interval == 0:
                corrects = torch.as_tensor((torch.argmax(outputs, dim=1) == labels)).sum().numpy()
                accuracy = 100.0 * corrects / args.batch_size
                print('Epoch[{}] Batch[{}] = loss: {:.5f}  lr: {:.5f}  acc: {:.2f}%  {}/{}'.format(epoch,
                                                                                                  i_batch,
                                                                                                  loss.detach().numpy(),
                                                                                                  args.lr,
                                                                                                  accuracy,
                                                                                                  corrects,
                                                                                                  args.batch_size,
                                                                                                  ))
        # validation
        #if i_batch % args.val_interval == 0:
        print('\nTest model:')
        val_loss, val_acc = test_model(test_loader, model, epoch, criterion, args)
        if best_acc is None or val_acc > best_acc:
            file_path = '%s/SelfAttnSent_best.pth.tar' % (args.save_folder)
            print("\r=> found better validated model, saving to %s" % file_path)
            save_checkpoint(model,
                            {'epoch': epoch,
                             'optimizer': optimizer.state_dict(),
                             'best_acc': best_acc},
                            file_path)
            best_acc = val_acc

        if args.checkpoint and epoch % args.save_interval == 0:
            file_path = '%s/SelfAttnSent_epoch_%d.pth.tar' % (args.save_folder, epoch)
            print("\r=> saving checkpoint model to %s" % file_path)
            save_checkpoint(model, {'epoch': epoch,
                                    'optimizer': optimizer.state_dict(),
                                    'best_acc': best_acc},
                            file_path)
        print('\n')
        epoch += 1

def save_checkpoint(model,checkpoint,file_path):
    checkpoint['state_dict'] = model.state_dict()
    torch.save(checkpoint,file_path)

def test_model(test_loader, model, epoch, criterion, args):
    '''
    :param test_loader:
    :param model:
    :param epoch:
    :param criterion:
    :param args:
    :return:
    '''
    model.eval()
    corrects, avg_loss, accumulated_loss, size = 0, 0, 0, 0

    predicates_all, target_all = [], []
    for i_batch, batch in enumerate(test_loader):
        input_ids = batch[0]
        labels = batch[1]

        size += len(labels)
        outputs = model(input_ids)
        accumulated_loss += criterion(outputs, labels).detach().numpy()

        predicates = torch.argmax(outputs, dim=1)
        corrects += torch.as_tensor((torch.argmax(outputs, dim=1) == labels)).sum().numpy()
        predicates_all += predicates.cpu().numpy().tolist()
        target_all += labels.cpu().numpy().tolist()

    avg_loss = accumulated_loss / size
    accuracy = 100.0 * corrects / size
    model.train()
    print('Evaluation - loss: {:.5f}  lr: {:.5f}  acc: {:.2f} ({}/{}) error: {:.2f}'.format(avg_loss,
                                                                                              args.lr,
                                                                                              accuracy,
                                                                                              corrects,
                                                                                              size,
                                                                                              100.0 - accuracy))
    if args.log_result:
        with open(os.path.join(args.save_folder, 'result.csv'), 'a') as r:
            r.write('\n{:d},{:.5f},{:.2f},{:f}'.format(epoch,
                                                            avg_loss,
                                                            accuracy,
                                                            args.lr))
    return avg_loss,accuracy

def data2loader(train_path,test_path,dataset_name,batch_size,bi_gram,bi_gram_min_count):
    '''
    :param train_path:
    :param test_path:
    :return: train and test dataloader,label_count,vocab_len
    '''
    #加载数据
    data_train = pd.read_csv(train_path,header=None)
    data_train[3] = data_train[[1,2]].apply(lambda x: " ".join(x),axis=1)
    data_test = pd.read_csv(test_path,header=None)
    data_test[3] = data_test[[1,2]].apply(lambda x: " ".join(x),axis=1)

    #获取全部文本数据
    texts = data_train[3].values.tolist()
    texts.extend(data_test[3].values.tolist())


    w2id = word2id.word2id("./middle_data/"+dataset_name+"_dic.pkl",bi_gram = bi_gram, Min_count = bi_gram_min_count)
    w2id.get_dic(texts)
    #print(w2id.vocab_len)
    #print(np.mean(w2id.text_len))
    vocab_len = w2id.vocab_len
    text_len_mean = int(np.mean(w2id.text_len))+2

    #训练和测试数据编码
    input_ids_train = w2id.get_id(data_train[3].values.tolist(), text_len_mean)
    input_ids_test = w2id.get_id(data_test[3].values.tolist(),text_len_mean)

    x_train_tensor = torch.LongTensor(input_ids_train)
    x_test_tensor = torch.LongTensor(input_ids_test)

    y_train_tensor = torch.LongTensor(np.array(data_train[0].values.tolist())-1)
    y_test_tensor = torch.LongTensor(np.array(data_test[0].values.tolist())-1)

    #装载数据
    dataset = TensorDataset(x_train_tensor,y_train_tensor)
    train_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True)

    #统计类别标签频次
    label_count = []
    y = data_train[0].values.tolist()
    for i in list(set(y)):
        label_count.append((i,y.count(i)))

    return train_loader,test_loader,label_count,vocab_len

def main():
    args = parser.parse_args()

    if args.dataset == "AG":
        train_path = "./data/ag_news_csv/train.csv"
        test_path = "./data/ag_news_csv/test.csv"
    elif args.dataset == "Sogou":
        train_path = "./data/sogou_news_csv/train.csv"
        test_path = "./data/sogou_news_csv/test.csv"
    elif args.dataset == "DBP":
        train_path = "./data/dbpedia_csv/train.csv"
        test_path = "./data/dbpedia_csv/test.csv"
    else:
        print("You just can choose [AG, Sogou, DBP, Yelp P., Yelp F., Yah. A., Amz. F., Amz. P.] as dataset!")
        sys.exit()

    # 加载数据
    print("loading data ......")
    train_loader,test_loader,label_count,vocab_len = \
        data2loader(train_path, test_path, args.dataset, args.batch_size,args.bi_gram,args.bi_gram_min_count)
    print("\n")

    # 数据说明
    print("Dataset declaration：")
    print("Dataset name: %s"%args.dataset)
    print("Train data statistics：")
    print(label_count)
    print("\n")

    # 创建训练结果存储文件夹
    print("Create folder......")
    try:
        os.makedirs(args.save_folder)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise
    print("\n")

    # 参数
    print("Configuration:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}:".format(attr.capitalize().replace('_', ' ')).ljust(25) + "{}".format(value))

    # 训练结果日志
    if args.log_result:
        with open(os.path.join(args.save_folder, 'result.csv'), 'w') as r:
            r.write('{:s},{:s},{:s},{:s}'.format('Epoch', 'Loss', 'ACC', 'lr'))

    # 模型
    divice = torch.device('cuda' if torch.cuda.is_available() and args.cuda == True else 'cpu')
    model = FastText.FastText(
        vocab_len,
        label_count,
        hidden_num=args.hidden_num,
        embedding_dim=args.embedding_dim
    ).to(divice)

    print("\n")
    print(model)

    # 训练
    train_model(train_loader, test_loader, model, args)

if __name__=="__main__":
    SEED = 1
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    np.random.seed(SEED)

    parser = argparse.ArgumentParser(description='参数')
    # dataset
    parser.add_argument('--dataset', metavar='dataset',
                        help='choose a dataset: AG, Sogou, DBP, Yelp P., Yelp F., Yah. A., Amz. F., Amz. P.',
                        default='AG')
    parser.add_argument("--bi_gram", type=bool, default=False, help="是否使用bigram(bool)")
    parser.add_argument("--bi_gram_min_count", type=int, default=10, help="使用bigram时，最小频次(int)")

    # train
    train = parser.add_argument_group("Training options")
    train.add_argument("--batch_size", type=int, default=128, help="训练批度(int)")
    train.add_argument("--eopch", type=int, default=5, help="训练迭代次数(int)")
    train.add_argument("--lr", type=float, default=0.00001, help="学习率(float)")
    train.add_argument("--optimizer", type=str, default="Adam", help="优化函数：Adam，Adagrad，AdamW (str)")

    # model
    model_args = parser.add_argument_group("Model options")
    model_args.add_argument("--hidden_num", type=int, default=10, help="FastText隐层数量(int)")
    model_args.add_argument("--embedding_dim", type=int, default=256, help="FastText词嵌入维度(int)")

    # device
    device = parser.add_argument_group('Device options')
    device.add_argument('--cuda', action='store_true', default=True, help='是否使用GPU(bool)')
    device.add_argument('--gpu', type=int, default=None)

    # experiment
    experiment = parser.add_argument_group("Experiment options")
    experiment.add_argument("--continue_from", type=str, default=None, help="断点恢复：断点文件路径(str)")
    experiment.add_argument("--log_interval", type=int, default=100, help="每多少个batch更新日志(int)")
    experiment.add_argument("--val_interval", type=int, default=100, help="每多少个batch验证模型(int)")
    experiment.add_argument("--log_result", type=bool, default=True, help="是否保存训练日志(bool)")
    experiment.add_argument("--save_folder", type=str, default="./output", help="存储训练中的模型以及日志文件夹路径(str)")
    experiment.add_argument("--checkpoint", type=bool, default=True, help="是否进行断点恢复(bool)")
    experiment.add_argument("--save_interval", type=int, default=5, help="每隔多少个epoch进行一次断点保存(int)")

    main()