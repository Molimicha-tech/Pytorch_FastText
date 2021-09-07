import torch
import torch.nn as nn

from utils import get_huffman_tree

class H_Softmax(nn.Module):
    '''
    层次softmax
    :input 隐层输出
    :output 预测结果
    '''
    def __init__(self,Huffman_tree,embedding):
        super(H_Softmax,self).__init__()
        self.Huffman_tree = Huffman_tree
        self.tree_pro = get_huffman_tree.tree_process(None)
        self.huffman,self.nodes = self.tree_pro.deepFirst(Huffman_tree)
        self.Huffman_coding = self.tree_pro.huffmanEncoding(self.nodes,Huffman_tree)

        self.thetas = [nn.Linear(embedding, 2) for item in self.huffman if isinstance(item[0], str)]

        self.m = nn.Softmax(dim=1)

    def forward(self, x):
        output = []
        for c in self.Huffman_coding:
            cp = torch.ones((x.size()[0], 1))
            for path in c:
                cp = self.m(self.thetas[path[0]](x)).t()[path[1]].unsqueeze(1) * cp
            output.append(cp)
        output = torch.cat(output, 1)
        return output


class FastText(nn.Module):
    '''
        FastText模型
    '''
    def __init__(self,vocab_size,label_count,hidden_num=10,embedding_dim=512):
        super(FastText,self).__init__()
        self.hidden_num = hidden_num

        self.embedding = nn.Embedding(vocab_size,embedding_dim) #embedding

        self.hidden_layer = nn.Linear(embedding_dim,embedding_dim)
        #构造huffman
        tree_pro = get_huffman_tree.tree_process(None)
        sourceData = [get_huffman_tree.Huffman_tree(x[0], x[1]) for x in label_count]
        Huffman_tree = tree_pro.makeHuffman(sourceData)

        self.H_softmax = H_Softmax(Huffman_tree, embedding_dim)

    def forward(self,input_ids):
        x = self.embedding(input_ids)
        x = torch.mean(x, 1)
        LN = nn.LayerNorm(x.shape[1:])
        x = LN(x)

        for _ in range(self.hidden_num):
            x = self.hidden_layer(x)

        x = self.H_softmax(x)
        return x