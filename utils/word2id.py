import pickle as pkl
import os
import re


class word2id():
    '''
        input: （文本列表，N-gram）
        output： 编码后的文本列表
    '''

    def __init__(self, dic_path, bi_gram=False, Min_count=10):
        self.dic_path = dic_path
        self.N_gram = bi_gram
        self.Min_count = Min_count
        self.vocab_len = None
        self.text_len = None

    def get_dic(self, texts):
        '''
        组建字典
        :return:Null
        '''
        if os.path.exists(self.dic_path):
            # 读取字典
            with open(self.dic_path, "rb") as f:
                dictionary = pkl.load(f)

            self.vocab_len = dictionary["vocab_len"]
            self.text_len = dictionary["text_len"]
            return 0

        vocabulary = []
        dictionary = {}
        lens = []
        if self.N_gram == False:
            for text in texts:
                words = text.split(" ")
                lens.append(len(words))
                vocabulary.extend(list(set(words)))
            vocabulary = list(set(vocabulary))
            for index, word in enumerate(vocabulary):
                dictionary[word] = index + 1
        else:
            bigram = {}
            vocabulary = []
            for text in texts:
                words = text.split(" ")
                vocabulary.extend(words)
                lens.append(len(words))
                for x, y in zip(words[0:-2], words[1:-1]):
                    if x + " " + y in bigram:
                        bigram[x + " " + y] += 1
                    else:
                        bigram[x + " " + y] = 1
            for bi in bigram.keys():
                if bigram[bi] > self.Min_count:
                    vocabulary.append(bi)
                else:
                    vocabulary.extend(bi.split(" "))
            vocabulary = list(set(vocabulary))
            for index, word in enumerate(vocabulary):
                dictionary[word] = index + 1

        dictionary["[padding]"] = 0
        with open(self.dic_path, "wb") as f:
            pkl.dump({"dictionary": dictionary,
                      "vocab_len": len(dictionary),
                      "text_len": lens}, f)

        self.vocab_len = len(dictionary)
        self.text_len = lens

    def get_id(self, texts, squence_len):
        '''
        :parameter texts 待转换文本 dic_path 字典路径
        :return:ids,字典大小
        '''
        # 读取字典
        with open(self.dic_path, "rb") as f:
            dictionary = pkl.load(f)

        # print(dictionary.keys())
        self.vocab_len = dictionary["vocab_len"]
        self.text_len = dictionary["text_len"]
        dictionary = dictionary["dictionary"]
        # print(dictionary.keys())

        ids = []

        if self.N_gram is False:
            for text in texts:
                id = []
                for word in text.split(" ")[:squence_len]:
                    try:
                        id.append(dictionary[word])
                    except:
                        print("字典中缺少部分词语！")
                len_id = len(id)
                for _ in range(0, squence_len - len_id):
                    id.append(0)
                ids.append(id)
        else:
            lens = []
            idss = []
            for text in texts:
                id = self.FMM(text, dictionary)
                lens.append(len(id))
                idss.append(id)
            squence_len = max(lens)
            for id in idss:
                idd = id
                for _ in range(0, squence_len - len(id)):
                    idd.append(0)
                ids.append(idd)

        return ids

    def FMM(self, sentence, word_dic):
        '''
        实现最大正向匹配
        :param sentence: 待分词的文本
        :return: 最大正向匹配的分词结果，list格式
        '''
        s = sentence.split(" ")
        result = []
        # 从前往后取词遍历句子
        while len(s) > 0:
            # 如果句长小于字典最大词长，则初始化取词长度为句子长度
            if len(s) < 2:
                i = len(s)
            else:
                i = 2
            while i > 0:
                word = " ".join(s[:i])
                # 当取出的为单字，或取出的词在词典中，则将取出的结果保存并结束循环
                if i == 1 or word in word_dic:
                    try:
                        result.append(word_dic[word])
                    except:
                        print("字典中缺少部分词语！")
                    s = s[i:]
                    break
                else:
                    i = i - 1
        return result