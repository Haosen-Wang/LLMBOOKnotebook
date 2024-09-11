import re
from collections import Counter
class BPEtokenize:
    #处理文本text数据
    def __init__(self,text_data,vocab_size:int=2048,merge_num:int=1000000):
        #vocab_size:构建词表最终大小 merge_num最大合并次数
        self.vocab_size=vocab_size
        self.merge_num=merge_num
        self.text_data=text_data
    def extract_frequencies(self,sequence):
        '''给定序列文本,返回词频字典'''
        token_counter = Counter()
        for item in sequence:
            tokens = ' '.join(list(item)) + ' </w>'
            token_counter[tokens] += 1
        return token_counter
    def frequency_of_pairs(self,frequencies):
        '''给定一个词频率词典，返回一个字符对到频率的映射字典'''
        pair_count=Counter()
        for token,count in frequencies.items():
            chars=token.split()
            for i in range(len(chars)-1):
                pair=(chars[i],chars[i+1])#字符对，并不是真的字符串，如（ab,cd）不是abcd
                pair_count[pair]+=count
        return pair_count
    def merge_vocab(self,merge_pair,vocab:dict):
        '''给定一对相邻词元和一个词频字典，将相邻词元合并为新的词元，并返回新的词表'''
        re_pattern = re.escape(' '.join(merge_pair))#re.escape()消除正则表达式的影响
        pattern = re.compile(r'(?<!\S)' + re_pattern + r'(?!\S)')
        #re.complile()将其转换为一个正则表达式字符串，r'(?<!\S)'代表只有re_pattern前是空白字符串才能查找，对应上一行中‘  ‘.join()
        update_tokens={pattern.sub(''.join(merge_pair),token): freq for token,freq in vocab.items()}
        #pattern.sub(''.join(merge_pair),token)代表将merge_pair变成字符串，并替换token，且作为token：freq的新键
        return update_tokens
        #返回一个新的词频字典
    def encode_with_bpe(self):
        '''给定待分词的数据以及最大合并次数，返回合并后的词表（词频字典）（按出现次数降值排序）'''
        vocab_map=self.extract_frequencies(self.text_data)
        for _ in range(self.merge_num):
            pair_freqs=self.frequency_of_pairs(vocab_map)
            if not pair_freqs:
                break
            most_common_pair=pair_freqs.most_common(1)[0][0]
            vocab_map =self.merge_vocab(most_common_pair, vocab_map)
            if len(vocab_map)>=self.vocab_size:
                break
        sorted_items = sorted(vocab_map.items(), key=lambda x: x[1],reverse=True)
        vocab_map = {k: v for k, v in sorted_items}
        return vocab_map
if __name__=='__main__':
    import nltk
    from nltk.corpus import gutenberg
    from collections import Counter
    # 下载gutenberg语料库
    nltk.download('gutenberg')
    # 获取文本
    text = gutenberg.words('austen-emma.txt')
    #print(text)
    BPE=BPEtokenize(text[:300])
    print(BPE.encode_with_bpe())
    #print(BPE.tokenize())
     
        
    

