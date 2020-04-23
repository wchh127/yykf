import pkuseg
import os
import jieba


file_path = os.path.dirname(__file__)
print("文件目录：")
print(file_path)

default_user_dict_path = os.path.join(file_path, "user_seg_dict.txt")
print(default_user_dict_path)

seg = pkuseg.pkuseg(user_dict=default_user_dict_path)
text = seg.cut('娱乐模板里欢乐斗怎么玩')
print(text)


class SegmentClass(object):
    def __init__(self, user_dict_path = ""):
        if (len(user_dict_path) > 0):
            self.user_dict_path = user_dict_path
        else:
            self.user_dict_path = default_user_dict_path
        self.seg = pkuseg.pkuseg(user_dict=self.user_dict_path)

    def cut(self, sentence):
        if (len(sentence) > 0):
            return self.seg.cut(sentence)
        return []