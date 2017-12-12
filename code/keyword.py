import re
import jieba
import pickle
import jieba.posseg as pseg

def is_num(question):
    pattern = re.compile(u'.*几.*|.*多少.*')
    match = pattern.match(question)
    if match:
        return True
    else:
        return False

def is_person(question):
    pattern = re.compile(u'.*谁.*|.*哪位.*')
    match = pattern.match(question)
    if match:
        return True
    else:
        return False
def qw_pos(question):
    pattern = re.compile(u'哪.*|什么.*')
    return pattern.search(question)

def axis_split(question):
    pattern = re.compile(u'是|叫|名叫|为|称为|作为')
    return re.split(pattern, question)

def default_process(question):
    res_pos = pseg.cut(axis_split(question)[0])
    tmp_list = []
    for w,f in res_pos:
        tmp_list.append([w,f])
    for l in reversed(tmp_list):
        if(l[1][0] == 'n'):
            return w
    return None

def keyword(question):
    '''Return the keyword of the question, 
    generally a Chinese word is returned,
    'm' is returned when the answer should be a number, 
    None is returned when the answer fails to be analysed.
    '''
    if is_num(question):
        return 'm'
    elif is_person(question):
        return u'人'
    else:
        match = qw_pos(question)
        if match:
            res_pos = pseg.cut(match.group(0))
            for w,f in res_pos:
                if(f[0] == 'n'):
                    return w
        return default_process(question)
        



if __name__ == '__main__':
    with open("question.pkl", "rb") as infile:
        data = pickle.load(infile)
        for q,a in data:
            print('=============================================')
            print(q)
            print(keyword(q))
            print(a)
