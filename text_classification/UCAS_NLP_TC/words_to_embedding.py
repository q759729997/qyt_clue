"""提取train中的所有单词的embedding"""
import codecs
import sys
sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import myClue  # noqa
print('myClue module path :{}'.format(myClue.__file__))  # 输出测试模块文件位置
from myClue.core import logger  # noqa
from myClue.tools.file import read_file_texts  # noqa
from myClue.tools.file import init_file_path  # noqa


if __name__ == "__main__":
    # train_words_file = './data/UCAS_NLP_TC/train_words.txt'
    # output_file = './data/UCAS_NLP_TC/train_words_embedding.txt'
    train_words_file = './data/UCAS_NLP_TC/train_baidu_basicwords.txt'
    output_file = './data/UCAS_NLP_TC/train_baidu_basicwords_embedding.txt'
    words = set(read_file_texts(train_words_file))
    print('words len:{}'.format(len(words)))
    with codecs.open(output_file, mode='w', encoding='utf8') as fw:
        fw.write('{} 200\n'.format(len(words)))  # 完成后根据实际个数修改计数
        with codecs.open(r'e:\下载专区\Tencent_AILab_ChineseEmbedding\Tencent_AILab_ChineseEmbedding.txt', mode='r', encoding='utf8') as fr:
            line_index = 0
            line_count = 8824330
            for line in fr:
                line_index += 1
                print('当前：{}，合计：{}'.format(line_index, line_count), end='\r')
                row_words = line.split(' ')
                if row_words[0] in words:
                    fw.write(line)
