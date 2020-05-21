# Weibo NER

### Task
Named Entity Recognition
### Description
**Tags**: PER(人名), LOC(地点名), GPE(行政区名), ORG(机构名)   

|Label|Tag|Meaning|
|:-:|:-:|:--|
|PER|PER.NAM|名字（张三）|
||PER.NOM|代称、类别名（穷人）|
|LOC|LOC.NAM|特指名称（紫玉山庄）|
||LOC.NOM|泛称（大峡谷、宾馆）|
|GPE|GPE.NAM|行政区的名称（北京）|
|ORG|ORG.NAM|特定机构名称（通惠医院）|
||ORG.NOM|泛指名称、统称（文艺公司）|

**Tag Strategy**：BIO  
**Split**:   
'\t' in raw data (北\tB-LOC)  
'*space*'in transformed data (北 B-LOC)  
**Data Size**:  
Train data set ( train.conll ):  

|句数|字符数|PER.NAM数|PER.NOM数|LOC.NAM数|LOC.NOM数|GPE.NAM数|ORG.NAM数|ORG.NOM数|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|1350|73778|574|766|56|51|205|183|42|

Dev data set ( dev.conll ):  

|句数|字符数|PER.NAM数|PER.NOM数|LOC.NAM数|LOC.NOM数|GPE.NAM数|ORG.NAM数|ORG.NOM数|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|270|14509|90|208|6|6|26|47|5|

Test data set ( test.conll )

|句数|字符数|PER.NAM数|PER.NOM数|LOC.NAM数|LOC.NOM数|GPE.NAM数|ORG.NAM数|ORG.NOM数|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|270|14842|111|170|19|9|47|39|17|

## 参考文献：

- Peng N, Dredze M. Named entity recognition for chinese social media with jointly trained embeddings[C]//Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing. 2015: 548-554.
- Peng N, Dredze M. Improving named entity recognition for chinese social media with word segmentation representation learning[J]. arXiv preprint arXiv:1603.00786, 2016.

**Reference**:   
[Named Entity Recognition for Chinese Social Media
with Jointly Trained Embeddings
](http://aclweb.org/anthology/D15-1064)  
[Improving Named Entity Recognition for Chinese Social Media
with Word Segmentation Representation Learning](http://www.aclweb.org/anthology/P16-2025)  
<https://github.com/hltcoe/golden-horse>  
<https://github.com/OYE93/Chinese-NLP-Corpus>