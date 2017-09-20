import numpy as np
encoding='unicode'
import os, io, cPickle as pickle, ntpath

input_lines = None

class Preprocessing:
    def __init__(self, file_name_with_full_path, maximum_word_length = 25):
        self.file_name_with_full_path = file_name_with_full_path
        self.maximum_word_length = maximum_word_length

    def build_char_dic(self):
       char_dic_file = self.file_name_with_full_path + '_char_dictionary'

       if os.path.isfile(char_dic_file):
           return

       dump_char_dic_file = open(char_dic_file, "wb")
       char_dic = {}
       value = 1;

       flag = 0
       with io.open(self.file_name_with_full_path, 'r', encoding="utf-8") as f1:
           input_lines = f1.readlines()
       f1.close()

       for line in input_lines:
            if line == '\n':
                continue
            line = line.strip()
            fields = line.split('\t')
            word = fields[0]
            form = fields[1]

            for ch in word:
                if char_dic.get(ch, None) is None:
                    char_dic[ch] = value
                    value += 1

            for ch in form:
                if char_dic.get(ch, None) is None:
                    char_dic[ch] = value
                    value += 1


       pickle.dump(char_dic, dump_char_dic_file)
       dump_char_dic_file.close()

    def get_char_dic(self):
        if not os.path.isfile(self.file_name_with_full_path + '_char_dictionary'):
            print ('in get_char_dic() of preprocessing.py : character dictionary is not found')
            return

        dump_char_dic_file = open( self.file_name_with_full_path + '_char_dictionary', "rb")
        char_dic = pickle.load(dump_char_dic_file)
        dump_char_dic_file.close()
        return char_dic

    def generate_char_one_hot_vec_and_num_encoding_for_single_word(self, word):
        char_dic = self.get_char_dic()

        word_length = 0
        char_vec_dim = len(char_dic.keys())

        char_array = np.array([])
        char_num_array = np.array([])

        for ch in word:
            word_length += 1
            char_vec = np.zeros(char_vec_dim)
            if(word_length > self.maximum_word_length):
                break
            if char_dic.get(ch, None) is not None:
                vec=int(char_dic[ch])-1
                char_vec[vec] = 1
                char_array = np.append(char_array,char_vec)
                char_num_array = np.append(char_num_array,int(char_dic[ch]))
            else:
                char_array = np.append(char_array,char_vec)
                char_num_array = np.append(char_num_array,0)

        if(word_length < self.maximum_word_length):
            char_vec = np.zeros(char_vec_dim)
            while(word_length != self.maximum_word_length):
                char_array = np.append(char_array,char_vec)
                char_num_array = np.append(char_num_array, 0)
                word_length = word_length+1

        return char_array, char_num_array


    def generate_file_and_reset_array(self, file_substring, sr_num, all_vec_array):
        f_name = file_substring + "_" + str(sr_num)
        tagFile = open(f_name, "wb")
        pickle.dump(all_vec_array, tagFile)
        tagFile.close()
        print('dumped arrays to file : ', f_name)
        return np.array([])

    def generate_char_one_hot_vec_and_num_encoding_for_file(self, file, divide_file_factor = 2000, where_to_dump = './'):

        file_substring1 = str(file) + "_char_one_hot_encoded"
        file_substring2 = str(file) + "_char_num_encoded"

        with io.open(file, 'r', encoding="utf-8") as f1:
            input_lines = f1.readlines()
        f1.close()

        input_lines1 = [line for line in input_lines if line.strip() != '']
        input_lines = input_lines1
        input_lines1 = None
        del input_lines1


        all_char_vec_array = np.array([])
        all_char_num_vec_array = np.array([])
        count = 0; k = 0

        for line in input_lines:
            if line == '\n':
                continue
            while '\t\t' in line:
                line = line.replace('\t\t', '\t')
            line = line.strip()

            fields = line.split('\t')
            word = fields[0]
            char_array, char_num_array = self.generate_char_one_hot_vec_and_num_encoding_for_single_word(word)

            all_char_vec_array = np.append(all_char_vec_array, char_array)
            all_char_num_vec_array = np.append(all_char_num_vec_array, char_num_array)

            if (count % divide_file_factor == 0 and count != 0) or count == len(input_lines)-1:
                if os.path.isfile(file_substring1 + '_' + str(k)) is False:
                   all_char_vec_array =  self.generate_file_and_reset_array(file_substring1, k, all_char_vec_array)
                   all_char_num_vec_array = self.generate_file_and_reset_array(file_substring2, k, all_char_num_vec_array)
                   k += 1
                   
                else:
                    all_char_vec_array = np.array([])
                    all_char_num_vec_array = np.array([])
                    k += 1
                    
            count += 1


    def process_global_word_vectors(self, file, vec_file, divide_file_factor=2000):

        word_vector_file_substring = file + '_word_vectors_from_' + ntpath.basename(vec_file)
        words = []
        length_of_vectors = None

        with io.open(file, 'r', encoding="ISO-8859-1") as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                fields = line.split('\t')
                word = fields[0]
                words.append(word)
        f.close()

        word_vector_dictionary = {}

        with io.open(vec_file, 'r', encoding="ISO-8859-1") as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                word = line[0:line.index(' ')]
                rest = line[line.index(' ') + 1:]
                fields = rest.split(' ')
                vector = map(float, fields)
                length_of_vectors = len(vector)
                word_vector_dictionary[word] = vector
        f.close()

        vectors = np.array([])
        count = 0; k = 0
        for word in words:
            if word_vector_dictionary.get(word, None) is None:
                random_vector = np.random.rand(length_of_vectors, )
                word_vector_dictionary[word] = random_vector
            vectors = np.append(vectors, np.asarray(word_vector_dictionary[word]))

            if (count % divide_file_factor == 0 and count != 0) or count == len(words) - 1:
                if os.path.isfile(word_vector_file_substring + '_' + str(k)) is False:
                    vectors = self.generate_file_and_reset_array(word_vector_file_substring, k, vectors)
                    k += 1
                else:
                    vectors = np.array([])
                    k += 1
            count += 1
        return length_of_vectors


    def pad_data(self, array, dictionary, file_substring, divide_file_factor = 2000):
       i = 0; j = 0; count = 0; k = 0
       file_substring += '_padded'

       if len(array.shape) != 2:
            print 'error in shape of array: in padding() function of load_data module'
            return None

       zero = np.zeros(int(array.shape[1]))
       store = np.array([])

       for i in range(0, len(array)):
           temp = array[i]
           temp = temp.reshape(len(temp))
           store = np.append(store, temp)
           if (count % divide_file_factor == 0 and count != 0) or i == len(array)-1:
               if os.path.isfile(file_substring + '_' + str(k)) is False:
                   store = self.generate_file_and_reset_array(file_substring, k, store)
                   k += 1
               else:
                   store = np.array([])
                   k += 1

           count += 1

           if dictionary.get(i, None) is not None:
               value = int(dictionary.get(i))
               for j in range(0, value):
                   store = np.append(store, zero)
                   if (count % divide_file_factor == 0 and count != 0) or i == len(array)-1:
                       if os.path.isfile(file_substring + '_' + str(k)) is False:
                           store = self.generate_file_and_reset_array(file_substring, k, store)
                           k += 1
                       else:
                           store = np.array([])
                           k += 1

                   count += 1

