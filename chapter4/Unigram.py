import re
from collections import Counter
import copy
from collections import defaultdict
from math import log
class Unigramtokenize:
    def __init__(self,text_data,vocab_size=100):
        self.vocab_size=vocab_size
        self.text_data=text_data
    def get_initalvocab(self):
        '''构建初始词表，此处用语料库所有可能的字符串作为初始词表'''
        token_counter = Counter()
        char_counter=Counter()
        subtoken_counter=Counter()
        #构建初始语料库的词频字典
        for item in self.text_data:
            tokens = '_'+item
            token_counter[tokens] += 1
        #对所有可能字符串构建词频字典
        for token,count in token_counter.items():
            for i in range(len(token)):
                char_counter[token[i]]+=count
                #构建长度至少为2的子词的词频字典
                for j in range(i+2,len(token)+1):
                    subtoken_counter[token[i:j]]+=count
        vocab_initial={**char_counter,**subtoken_counter}
        vocab_initial=sorted(vocab_initial.items(),key=lambda x:x[1],reverse=True)
        vocab_initial={k:v for k,v in vocab_initial}
        return vocab_initial,token_counter
    def get_log(self):
        '''构建词到概率的log值的映射字典'''
        total_sum=sum([count for token,count in self.vocab_initial.items()])
        log_dict={token:-log(float(count)/total_sum) for token, count in self.vocab_initial.items()}
        return log_dict
    def encode_word(self,word,log_dict):
        best_segmentations=[{"start":0,"score":1}]+[{"strat":None,"score":None} for _ in range(len(word))]
        for start_idx in range(len(word)):
            #遍历一个word所有的字符串组合，并计算累计得分score,应用动态规划的方法，获取得分最高的字符串组合，即分词方法
            best_score_at_start=best_segmentations[start_idx]["score"]
            for end_idx in range(start_idx+1,len(word)+1):
                token=word[start_idx:end_idx]
                if token in log_dict and best_score_at_start is not None:
                    score=log_dict[token]+best_score_at_start
                    # 如果找到了一个更好的、在 end_idx 结束的分词方式，就更新记录
                    if (best_segmentations[end_idx]["score"] is None or best_segmentations[end_idx]["score"]>score):
                        best_segmentations[end_idx]={"start": start_idx, "score": score}
        segmentation = best_segmentations[-1]
        if segmentation["score"] is None:
            # 如果没有找到有效的分词方式，就将整个单词标记为未知
            return ["<unk>"], None
        score=segmentation["score"]
        start=segmentation["start"]
        end=len(word)
        tokens=[]
        while start!=0:
            tokens.insert(0,word[start:end])
            next_start = best_segmentations[start]["start"]
            end = start
            start = next_start
        tokens.insert(0, word[start:end])
        return tokens,score
    def compute_loss(self,log_dict):
        loss=0
        for word,count in self.data_counter.items():
            _,word_loss=self.encode_word(word,log_dict)
            loss+=count*word_loss
        return loss
    def compute_scores(self,log_dict):
        '''计算删除词元后的损失值'''
        scores={}
        model_loss=self.compute_loss(log_dict)
        for token,score in log_dict.items():
            if len(token)==1:
                continue
            log_dict_without_token=copy.deepcopy(log_dict)
            _=log_dict_without_token.pop(token)
            scores[token]=self.compute_loss(log_dict_without_token)-model_loss
        return scores
    def tokenize(self,percent_to_remove=0.1):
        self.vocab_initial,self.data_counter=self.get_initalvocab()
        log_dict=self.get_log()
        while len(log_dict) > self.vocab_size:
            scores=self.compute_scores(log_dict)
            sorted_scores=sorted(scores.items(),key=lambda x:x[1])
            # 每次迭代移除 percent_to_remove 比例的低分词元
            for i in range(int(len(log_dict) * percent_to_remove)):
                _= self.vocab_initial.pop(sorted_scores[i][0])
            total_sum=sum([count for token,count in self.vocab_initial.items()])
            log_dict={token:-log(float(count)/total_sum) for token, count in self.vocab_initial.items()}
        return self.vocab_initial
            

                    
        

if __name__=='__main__':
    import nltk
    from nltk.corpus import gutenberg
    from collections import Counter
    # 下载gutenberg语料库
    nltk.download('gutenberg')
    # 获取文本
    text = gutenberg.words('austen-emma.txt')
    Unigram=Unigramtokenize(text[:30])
    print(Unigram.tokenize())
            
            