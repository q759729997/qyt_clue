def decode_ner_tags(chars, tags):
    """ 解码实体识别数据encoding_type='BIO'.

        @params:
            chars - 字符列表，如：['你', '好', '中', '国'…….
            tags - 标签列表，如：['O', 'O', 'B-ORG', 'I-ORG'…….

        @return:
            On success - 实体列表,[{'word': '中国', 'type': 'ORG', 'offset': 2, 'length': 2}, ……].
    """
    # 去掉句子中的关键信息标识
    ne_words = list()
    ner_words_ = list()
    ner_tag_ = None
    for char_i in range(len(chars)):
        char_data_ = chars[char_i]
        char_tag_ = tags[char_i]
        if char_tag_.startswith('B-'):
            if len(ner_words_) > 0:
                ner_word_ = ''.join(ner_words_)
                length_ = len(ner_word_)
                offset_ = len(''.join(chars[:char_i])) - length_
                ne_words.append({'word': ner_word_, 'type': ner_tag_, 'offset': offset_, 'length': length_})
                ner_words_ = list()
            ner_tag_ = char_tag_.replace('B-', '')
            ner_words_.append(char_data_)
        elif char_tag_.startswith('I-'):
            ner_words_.append(char_data_)
        else:
            if len(ner_words_) > 0:
                ner_word_ = ''.join(ner_words_)
                length_ = len(ner_word_)
                offset_ = len(''.join(chars[:char_i])) - length_
                ne_words.append({'word': ner_word_, 'type': ner_tag_, 'offset': offset_, 'length': length_})
                ner_words_ = list()
    if len(ner_words_) > 0:
        ner_word_ = ''.join(ner_words_)
        length_ = len(ner_word_)
        offset_ = len(''.join(chars[:char_i + 1])) - length_
        ne_words.append({'word': ner_word_, 'type': ner_tag_, 'offset': offset_, 'length': length_})
        ner_words_ = list()
    return ne_words