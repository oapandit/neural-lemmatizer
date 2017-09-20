encoding='unicode'
import os, io, cPickle as pickle, ntpath, gc, codecs
import preprocessing
import edit_tree
import numpy as np




class Load_data:

    def __init__(self, train_file, max_word_length = 25, divide_file_factor = 2000):
        self.train_file = train_file
        self.max_word_length = max_word_length
        self.divide_file_factor = divide_file_factor

    def read_parts_from_file(self, file_name_substring, reshape_size):
        filelist = []
        for root, dirs, files in os.walk('./'):
            for filen in files:
                if file_name_substring in filen:
                    filelist.append(filen)
        f_number = 0
        filelist_in_order = []
        for f in filelist:
            fname = file_name_substring + "_" + str(f_number)
            filelist_in_order.append(fname)
            f_number += 1
        complete_array = np.array([])
        first_access = True
        for fname in filelist_in_order:
            f = open(fname, 'rb')
            complete_array_part = pickle.load(f)
            complete_array_part = complete_array_part.reshape(-1, reshape_size)
            if (first_access):
                complete_array = complete_array_part
                first_access = False
            else:
                complete_array = np.concatenate((complete_array, complete_array_part), axis=0)
        return complete_array

    def load_train_char_data(self):
        Preprocessing = preprocessing.Preprocessing(self.train_file, self.max_word_length)
        Preprocessing.build_char_dic()
        Preprocessing.generate_char_one_hot_vec_and_num_encoding_for_file(self.train_file, self.divide_file_factor)
        train_file_substring2 = str(self.train_file) + '_char_num_encoded'
        train_char_data = self.read_parts_from_file(train_file_substring2, self.max_word_length)
        return train_char_data

    def load_test_char_data(self, test_file):
        Preprocessing = preprocessing.Preprocessing(self.train_file, self.max_word_length)
        Preprocessing.generate_char_one_hot_vec_and_num_encoding_for_file(test_file, self.divide_file_factor)
        test_file_substring = test_file + "_char_num_encoded"
        test_char_data = self.read_parts_from_file(test_file_substring, self.max_word_length)
        return test_char_data

    def load_class_annotation_and_unique_edit_trees_from_train_data(self):
        edit_tree_creator =  edit_tree.Edit_tree_creator()
        no_of_classes = edit_tree_creator.generate_class_annotated_file_and_unique_edit_trees(self.train_file, self.divide_file_factor)
        path2 = self.train_file + '_unique_edit_trees'
        class_annotation_file_substring = self.train_file + '_class_annotated'
        train_data_class_annotation = self.read_parts_from_file(class_annotation_file_substring, no_of_classes)
        f = open(path2, 'rb')
        unique_edit_trees_from_train_data = pickle.load(f)
        f.close()
        return train_data_class_annotation, unique_edit_trees_from_train_data

    def load_applicable_trees_data(self, file):
        edit_tree_creator = edit_tree.Edit_tree_creator()
        no_of_classes = edit_tree_creator.applicable_edit_trees_generation(file, self.train_file + '_unique_edit_trees', self.divide_file_factor)
        applicable_trees_file_substring = file + "_applicable_edit_trees_using_" + ntpath.basename(self.train_file) + '_unique_edit_trees'
        applicable_trees_data = self.read_parts_from_file(applicable_trees_file_substring, no_of_classes)
        return applicable_trees_data

    def load_global_word_vectors(self, file, vec_file):
        Preprocessing = preprocessing.Preprocessing(self.train_file, self.max_word_length)
        length_of_vectors = Preprocessing.process_global_word_vectors(file, vec_file, self.divide_file_factor)
        word_vector_file_substring = file + '_word_vectors_from_' + ntpath.basename(vec_file)
        word_vectors = self.read_parts_from_file(word_vector_file_substring, length_of_vectors)
        return word_vectors

    def zero_padding_information(self, file, max_sentence_length_in_train_data = None):
        i = 0
        sentences_lengths = []
        with io.open(file, 'r', encoding="utf-8") as f:
            for line in f:
                if line == '\n':
                    sentences_lengths.append(i)
                    i = 0
                else:
                    i += 1
        f.close()

        max_sentence_length = max(sentences_lengths)

        #********* Newly Added **********#
        temp_sentences_lengths = []
        if max_sentence_length_in_train_data is not None:
            if max_sentence_length > max_sentence_length_in_train_data:
                for i in sentences_lengths:
                    if i <= max_sentence_length_in_train_data:
                        temp_sentences_lengths.append(i)
                    else:
                        while i > max_sentence_length_in_train_data:
                            i -= max_sentence_length_in_train_data
                            temp_sentences_lengths.append(max_sentence_length_in_train_data)
                        if i > 0:
                            temp_sentences_lengths.append(i)


        if len(temp_sentences_lengths) > 0:
            sentences_lengths = temp_sentences_lengths
        if max_sentence_length_in_train_data is not None:
            max_sentence_length = max_sentence_length_in_train_data

        indices = []
        temp_list = None
        for i in range(1, len(sentences_lengths)+1):
            temp_list = sentences_lengths[0:i]
            indices.append(sum(temp_list)-1)
        del temp_list

        append_values = [max_sentence_length-i for i in sentences_lengths]
        return dict(zip(indices, append_values)), int(max_sentence_length)


    def load_padded_data(self, array, file_substring):
        file_substring += '_padded'
        reshape_size = int(array.shape[1])
        padded_array = self.read_parts_from_file(file_substring, reshape_size)
        return padded_array


















