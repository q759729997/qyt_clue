from gensim.models import KeyedVectors


if __name__ == "__main__":
    """
    预训练模型下载： https://github.com/Embedding/Chinese-Word-Vectors
    安装gensim
    参考代码：https://radimrehurek.com/gensim/models/word2vec.html
    """
    # word2vec_model_file = './data/embed/sgns.weibo.word/sgns.weibo.word'
    word2vec_model_file = './data/UCAS_NLP_TC/train_words_embedding.txt'
    model = KeyedVectors.load_word2vec_format(word2vec_model_file, binary=False)
    results = model.most_similar('扎克伯格', topn=5)
    print(results)
    results = model.most_similar('舍友', topn=5)
    print(results)
    results = model.most_similar('客户', topn=5)
    print(results)
    """
    [('孙正义', 0.6995826959609985), ('贝索斯', 0.6919196844100952), ('AOL', 0.6872508525848389), ('鲍尔默', 0.6810077428817749), ('贝佐斯', 0.67955482006073)]
    [('自习课', 0.691378653049469), ('同宿舍', 0.6856659650802612), ('高中都', 0.6823875904083252), ('同室', 0.6806875467300415), ('卧谈', 0.6784518361091614)]
    [('顾客', 0.6356246471405029), ('客户关系', 0.6265754699707031), ('售前', 0.6204451322555542), ('最终用户', 0.6057270765304565), ('老客户', 0.6033730506896973)]
    """
