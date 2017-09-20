import numpy as np
encoding='unicode'
import os, ntpath, cPickle as pickle, io


class Node:
    def __init__(self):
        self.source = None
        self.target = None
        self.pref_len = -1
        self.suf_len = -1
        self.flag = -1
        self.left = None
        self.right = None

class Node_with_level:
    def __init__(self, n1, level1):
        self.n = n1
        self.level = level1

class LCS_finder:
    def find(self, s1, s2):
        if s1 == "" or s2 == "":
            return ""

        A = list(s1)
        B = list(s2)
        LCS = np.zeros((len(A)+1, len(B)+1))

        for i in range(1, len(A)+1):
            for j in range(1, len(B)+1):
                if A[i-1] == B[j-1]:
                    LCS[i][j] = LCS[i-1][j-1]+1
                else:
                    LCS[i][j] = 0

        result = -1
        start_index = -1

        for i in range(0, len(A)+1):
            for j in range(0, len(B)+1):
                if result < LCS[i][j]:
                    result = LCS[i][j]
                    start_index = i

        if result == 0:
            return ""

        string = u""

        for i in range(int(start_index-result), int(start_index)):
            string += A[i]

        return string

class Edit_tree_creator:
   def build_tree(self, input, output):
       start = None
       if input == "" and output == "":
           return None

       start = Node()
       l = LCS_finder()
       lcs = l.find(input, output)
       if input == "" or output == "" or lcs == "":
           start.source = input
           start.target = output
           start.flag = 1
           return start

       ss = input.find(lcs)
       es = ss + len(lcs)
       st = output.find(lcs)
       et = st + len(lcs)

       start.pref_len = ss
       start.suf_len = len(input)-es
       start.flag = 0
       start.left = self.build_tree(input[0:ss], output[0:st])
       start.right = self.build_tree(input[int(es):len(input)], output[int(et):len(output)])
       return start

   def apply(self, tree, word):
       if tree == None and word == "":
           return ""
       if tree == None and word == None:
           print ('apply error => tree is None word is None')
           return "INVALID"
       if tree == None and word != None:
           return "INVALID"
       if tree.flag == -1:
           print ('apply error: tree.flag = -1')
           return ""
       if tree.flag == 0:
           if len(word) < tree.pref_len + tree.suf_len:
               return "INVALID"
           P = self.apply(tree.left, word[0:tree.pref_len])
           if P == "INVALID":
               return  "INVALID"
           S = self.apply(tree.right, word[len(word)-tree.suf_len:len(word)])
           if S == "INVALID":
               return  "INVALID"
           return P + word[tree.pref_len:len(word)-tree.suf_len] + S

       else:
           if tree.source == word:
               return tree.target
           return "INVALID"

       return ""

   def depth(self, tree):
       if tree == None:
           return 0
       depth = 1
       left_depth = self.depth(tree.left)
       right_depth = self.depth(tree.right)
       if left_depth > right_depth:
           depth += left_depth
       else:
           depth += right_depth
       return depth

   def print_node(self, tree):
       if tree is None:
           return ""
       printing = ""
       if tree.flag == 0:
           printing = "" + str(tree.flag) + "(" + str(tree.pref_len) + "," + str(tree.suf_len) + ")"
       else:
           printing = "" + str(tree.flag) + "(" + tree.source + "," + tree.target + ")"
       return printing

   def print_tree(self, tree):
       arlst = []
       if tree is None:
           return ''
       root = Node_with_level(tree, 1)
       arlst.append(root)

       while arlst:
           nl = arlst[0]
           print("[" + str(self.print_node(nl.n)) + "lev=" + str(nl.level) + "]\t"),
           if nl.n.left is not None:
               nl_left = Node_with_level(nl.n.left, nl.level+1)
               arlst.append(nl_left)
           if nl.n.right is not None:
               nl_right = Node_with_level(nl.n.right, nl.level+1)
               arlst.append(nl_right)
           del arlst[0]
       print ('')
       return ''

   def equal_node(self, first, second):
       if first is None and second is None:
           return True
       if ((first is not None and second is None) or (first is None and second is not None)):
           return False
       if first.flag != second.flag:
           return False
       if first.flag == 0:
           if first.pref_len == second.pref_len and first.suf_len == second.suf_len:
               return True
           else:
               return False
       elif first.flag == 1:
           if first.source == second.source and first.target == second.target:
               return True
           else:
               return False
       print ('in equalnode ERROR: suspicious')
       return True

   def equal_tree(self, first, second):
       if first is None and second is None:
           return True
       if ((first is not None and second is None) or (first is None and second is not None)):
           return False

       A = self.equal_node(first, second)
       B = self.equal_tree(first.left, second.left)
       C = self.equal_tree(first.right, second.right)
       if A == True and B == True and C == True:
           return True
       return False

   def generate_file_and_reset_array(self, file_substring, sr_num, all_vec_array):
       f_name = file_substring + "_" + str(sr_num)
       tagFile = open(f_name, "wb")
       pickle.dump(all_vec_array, tagFile)
       tagFile.close()
       print('dumped arrays to file : ', f_name)
       return np.array([])



   def generate_class_annotated_file_and_unique_edit_trees(self, file_name_with_full_path, divide_file_factor = 2000, where_to_dump = './'):
       file_substring = str(file_name_with_full_path) + '_class_annotated'
       path2 = file_name_with_full_path + '_unique_edit_trees'

       with io.open(file_name_with_full_path, 'r', encoding="utf-8") as f:
           lines = f.readlines()
       f.close()

       lines1 = [line for line in lines if line.strip() != '']
       lines = lines1
       lines1 = None
       del lines1

       unique_edit_trees_set = []
       class_annotation = []

       for line in lines:
           if line == '\n':
               continue
           while '\t\t' in line:
               line = line.replace('\t\t', '\t')
           line = line.strip()
           fields = line.split('\t')
           word = fields[0]
           form = fields[1]
           tree = self.build_tree(word, form)

           flag = 0; index = 0
           for trees in unique_edit_trees_set:
               index += 1
               if self.equal_tree(tree, trees):
                   flag = 1
                   break
           if flag == 0:
               unique_edit_trees_set.append(tree)
               class_annotation.append(len(unique_edit_trees_set)-1)
           else:
               class_annotation.append(index-1)

       no_of_classes = len(unique_edit_trees_set)
       all_class_annotation = np.array([])

       if os.path.isfile(path2):
           return no_of_classes

       count = 0; k = 0
       for i in class_annotation:
           vec = np.zeros(no_of_classes)
           vec[i] = 1
           all_class_annotation = np.append(all_class_annotation, vec)

           if (count % divide_file_factor == 0 and count != 0) or count == len(class_annotation) - 1:
               if os.path.isfile(file_substring + '_' + str(k)) is False:
                   all_class_annotation = self.generate_file_and_reset_array(file_substring, k, all_class_annotation)
                   k += 1
               else:
                   all_class_annotation = np.array([])
                   k += 1
           count += 1

       f = open(path2, 'wb')
       pickle.dump(unique_edit_trees_set, f)
       f.close()
       return no_of_classes

   def applicable_edit_trees_generation(self, file_name_with_full_path, unique_edit_trees_set_dumped_file, divide_file_factor = 2000, where_to_dump = './'):

       file_substring = file_name_with_full_path + "_applicable_edit_trees_using_" + ntpath.basename(unique_edit_trees_set_dumped_file)

       with io.open(file_name_with_full_path, 'r', encoding="utf-8") as f:
           lines = f.readlines()
       f.close()

       lines1 = [line for line in lines if line.strip() != '']
       lines = lines1
       lines1 = None
       del lines1

       f = open(unique_edit_trees_set_dumped_file, 'rb')
       unique_edit_trees_set = pickle.load(f)
       f.close()

       no_of_unique_edit_trees = len(unique_edit_trees_set)
       all_applicable_trees_vec = np.array([])

       count = 0; k = 0
       for line in lines:
           if line == '\n':
               continue
           while '\t\t' in line:
               line = line.replace('\t\t', '\t')
           line = line.strip()
           fields = line.split('\t')
           word = fields[0]

           applicable_trees_vec = np.zeros(no_of_unique_edit_trees)
           for i in range(0, len(unique_edit_trees_set)):
               if self.apply(unique_edit_trees_set[i], word) != 'INVALID':
                   applicable_trees_vec[i] = 1

           all_applicable_trees_vec = np.append(all_applicable_trees_vec, applicable_trees_vec)

           if (count % divide_file_factor == 0 and count != 0) or count == len(lines) - 1:
               if os.path.isfile(file_substring + '_' + str(k)) is False:
                  all_applicable_trees_vec  = self.generate_file_and_reset_array(file_substring, k, all_applicable_trees_vec)
                  k += 1
               else:
                   all_applicable_trees_vec = np.array([])
                   k += 1
           count += 1
       return no_of_unique_edit_trees

