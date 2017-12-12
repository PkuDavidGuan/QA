# QA
This is the final project for `Web data mining class 2017 FALL-PKU`. We made a Chinese QA system, using <a href="https://dumps.wikimedia.org/zhwiki/20171020/zhwiki-20171020-pages-articles-multistream.xml.bz2">zh-wikipedia data</a>. 
## File Organization
- code
  - xml2Raw.py: Transform zh-wikipedia data into raw data.
  - glance.py [fileAddr][beginLine] [endLine]: View the raw data in the teriminal, since the raw data file is huge.
  - cut.py: cut the wiki corpus into several files using `jieba`.
  - tfidf.py: calculate the TF-IDF of each document.
  - query.py: select top k related documents.
  - build-inverted-index.py: optimize the query process.
  - question2dict.py: build up a dict containing (question, answer) pairs.
  - vector2dict.py: build up a dict containing (Chinese word, word_vector) pairs.
  - answer_class.py: predict the answer's pos tag. 
  - keyword.py: analyse the key word of the question.
- results
