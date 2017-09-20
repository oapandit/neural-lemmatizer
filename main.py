from __future__ import print_function
import load_data, models, preprocessing, edit_tree
import os, io, pickle, re, ntpath
import numpy as np
import gc, sys

np.random.seed(1337)  # for reproducibility
## Reading the Parameters
with io.open(str(sys.argv[1]), 'r') as f1:
    input_lines = f1.readlines()
f1.close()


char_network_cell = str(input_lines[0].strip())
max_word_length = int(input_lines[1].strip())
char_feature_output = int(input_lines[2].strip())
global_vector_file = str(input_lines[3].strip())
tree_classify_network_cell = str(input_lines[4].strip())
hidden_size = int(input_lines[5].strip())
nb_epoch = int(input_lines[6].strip())
batch_size = int(input_lines[7].strip())
divide_file_factor = int(input_lines[8].strip())


del input_lines
input_lines = None


# Load Training Data

train_file = str(sys.argv[2])

ld = load_data.Load_data(train_file, max_word_length, divide_file_factor)   #give the training file name

train_char_data = ld.load_train_char_data()
print ('train_char_data.shape: ' + str(train_char_data.shape))

train_data_class_annotation, unique_edit_trees_from_train_data = ld.load_class_annotation_and_unique_edit_trees_from_train_data()
print ('train_data_class_annotation: ' + str(train_data_class_annotation.shape))
print ('size of unique_edit_trees_from_train_data: ' + str(len(unique_edit_trees_from_train_data)))

train_data_applicable_trees = ld.load_applicable_trees_data(train_file)
print ('train_data_applicable_trees: ' + str(train_data_applicable_trees.shape))

train_data_global_vectors = ld.load_global_word_vectors(train_file, global_vector_file)
print ('train_data_global_vectors.shape: ' + str(train_data_global_vectors.shape))

nb_tree_classes = len(unique_edit_trees_from_train_data)
word_vector_size = int(train_data_global_vectors.shape[1])
embedded_char_vector_length = len(preprocessing.Preprocessing(train_file).get_char_dic())

#***************************** Training Phase Start *****************************#

#********************** Building Char Model *************************#

# no_of_words = int(train_char_data.shape[0])
char_model = models.get_char_model(char_network_cell, max_word_length, embedded_char_vector_length, char_feature_output)
print('char model summary:')
print(char_model.summary())

char_model_file_name = train_file + '_char_level_' + char_network_cell + '.h5'
if os.path.isfile(os.path.join('./', char_model_file_name)):
    char_model.load_weights(os.path.join('./', char_model_file_name))
    print ('loaded ' + char_model_file_name + ' from disk')
else:
    char_model.save_weights(os.path.join('./', char_model_file_name))
    print (char_model_file_name + ' : saved weights in the disk')

train_char_vectors = char_model.predict([train_char_data])
print ('train_char_vectors.shape is: ' + str(train_char_vectors.shape))


#*************************** Padding *****************************#

padding_dictionary, max_sentence_length = ld.zero_padding_information(train_file)
print ('max_sentence_length: ', max_sentence_length)

preprocessing.Preprocessing(train_file).pad_data(train_char_vectors, padding_dictionary, train_file + '_char_vectors', divide_file_factor)
train_char_vectors = ld.load_padded_data(train_char_vectors, train_file + '_char_vectors')
print ('After Padding train_char_vectors.shape is : ' + str(train_char_vectors.shape))

preprocessing.Preprocessing(train_file).pad_data(train_data_global_vectors, padding_dictionary, train_file + '_global_vectors', divide_file_factor)
train_data_global_vectors = ld.load_padded_data(train_data_global_vectors, train_file + '_global_vectors')
print ('After Padding train_data_global_vectors.shape is : ' + str(train_data_global_vectors.shape))

preprocessing.Preprocessing(train_file).pad_data(train_data_class_annotation, padding_dictionary, train_file + '_class_annotation', divide_file_factor)
train_data_class_annotation = ld.load_padded_data(train_data_class_annotation, train_file + '_class_annotation')
print ('After Padding train_data_class_annotation.shape is : ' + str(train_data_class_annotation.shape))

preprocessing.Preprocessing(train_file).pad_data(train_data_applicable_trees, padding_dictionary, train_file + '_applicable_trees', divide_file_factor)
train_data_applicable_trees = ld.load_padded_data(train_data_applicable_trees, train_file + '_applicable_trees')
print ('After Padding train_data_applicable_trees.shape is : ' + str(train_data_applicable_trees.shape))

############################ After Padding ########################
train_char_vectors = train_char_vectors.astype('float32')
train_data_global_vectors = train_data_global_vectors.astype('float32')

#*************************** Reshaping *****************************#

train_char_vectors = train_char_vectors.reshape(-1, max_sentence_length, int(train_char_vectors.shape[1]))
train_data_global_vectors = train_data_global_vectors.reshape(-1, max_sentence_length, int(train_data_global_vectors.shape[1]))
train_data_class_annotation = train_data_class_annotation.reshape(-1, max_sentence_length, int(train_data_class_annotation.shape[1]))
train_data_applicable_trees = train_data_applicable_trees.reshape(-1, max_sentence_length, train_data_applicable_trees.shape[1])
#*****************************************************************#

tree_classify_model = models.get_tree_classify_model(tree_classify_network_cell, max_sentence_length, word_vector_size, 2*char_feature_output, hidden_size, nb_tree_classes)
print ('tree classify model summary: ')
print (tree_classify_model.summary())


tree_classify_model.compile(optimizer='adam', loss=['categorical_crossentropy'],metrics=['accuracy'])
tree_classify_model_file_name = train_file + '_tree_classify_' + tree_classify_network_cell + '.h5'
if os.path.isfile(os.path.join('./', tree_classify_model_file_name)):
    tree_classify_model.load_weights(os.path.join('./', tree_classify_model_file_name))
    print ('loaded ' + tree_classify_model_file_name + 'from disk')
else:
    tree_classify_model.fit([train_data_global_vectors, train_char_vectors, train_data_applicable_trees], [train_data_class_annotation], nb_epoch=nb_epoch, batch_size=batch_size, )
    print('training completed')
    tree_classify_model.save_weights(os.path.join('./', tree_classify_model_file_name))
    print(tree_classify_model_file_name + ' : saved weights in the disk')


#************************************ Preprocessing of Test File *********************************#
test_file = None
if len(sys.argv) > 3:
    test_file = str(sys.argv[3])
else:
    sys.exit(0)

test_char_data= ld.load_test_char_data(test_file)
test_data_applicable_trees = ld.load_applicable_trees_data(test_file)
test_data_global_vectors = ld.load_global_word_vectors(test_file, global_vector_file)
test_char_vectors = char_model.predict([test_char_data])

#*************************** Padding of Test Files *****************************#

padding_dictionary, _ = ld.zero_padding_information(test_file, max_sentence_length)

preprocessing.Preprocessing(train_file).pad_data(test_char_vectors, padding_dictionary, test_file + '_as_test_and_' + train_file + '_as_train_char_vectors', divide_file_factor)
test_char_vectors = ld.load_padded_data(test_char_vectors, test_file + '_as_test_and_' + train_file + '_as_train_char_vectors')
print ('After padding test char vectors shape ', str(test_char_vectors.shape))

preprocessing.Preprocessing(train_file).pad_data(test_data_global_vectors, padding_dictionary, test_file + '_as_test_and_' + train_file + '_as_train_global_vectors', divide_file_factor)
test_data_global_vectors = ld.load_padded_data(test_data_global_vectors, test_file + '_as_test_and_' + train_file + '_as_train_global_vectors')
print ('After padding test global vectors shape ', str(test_data_global_vectors.shape))

preprocessing.Preprocessing(train_file).pad_data(test_data_applicable_trees, padding_dictionary, test_file + '_as_test_and_' + train_file + '_as_train_applicable_trees', divide_file_factor)
test_data_applicable_trees = ld.load_padded_data(test_data_applicable_trees, test_file + '_as_test_and_' + train_file + '_as_train_applicable_trees')
print ('After padding test applicable edit trees shape ', str(test_data_applicable_trees.shape))

test_char_vectors = test_char_vectors.astype('float32')
test_data_global_vectors = test_data_global_vectors.astype('float32')

#*************************** Reshaping of Test Data Arrays *****************************#

test_char_vectors = test_char_vectors.reshape(-1, max_sentence_length, int(test_char_vectors.shape[1]))
test_data_global_vectors = test_data_global_vectors.reshape(-1, max_sentence_length, int(test_data_global_vectors.shape[1]))
test_data_applicable_trees = test_data_applicable_trees.reshape(-1, max_sentence_length, test_data_applicable_trees.shape[1])
#***************************************************************************************#
#******************************** Prediction *******************************************#
output = tree_classify_model.predict([test_data_global_vectors, test_char_vectors, test_data_applicable_trees])
output = output.reshape(int(output.shape[0])*int(output.shape[1]), -1)
test_data_global_vectors = test_data_global_vectors.reshape(int(test_data_global_vectors.shape[0])*int(test_data_global_vectors.shape[1]), -1)
test_data_applicable_trees = test_data_applicable_trees.reshape(int(test_data_applicable_trees.shape[0])*int(test_data_applicable_trees.shape[1]), -1)

flag = [True if test_data_global_vectors[i].any() else False for i in range(0, len(test_data_global_vectors))]
output_trees = []
for i in range(0, len(output)):
    if flag[i] == False:
        continue
    max_prob = -1.0; index = -1
    for j in range(0, len(output[i])):
        if test_data_applicable_trees[i][j] == 1:
            if max_prob < output[i][j]:
                max_prob = output[i][j]
                index = j
    output_trees.append(index)

i = 0; j = 0; evaluation_flag = 0
edit_tree_creator = edit_tree.Edit_tree_creator()
output_file = str(test_file) + '_output'
file_writer = io.open(output_file, 'w', encoding="utf-8")
with io.open(test_file, 'r', encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line == '':
            file_writer.write(u'\n')
            continue
        fields = line.split('\t')
        predicted_lemma = edit_tree_creator.apply(unique_edit_trees_from_train_data[output_trees[i]], fields[0])
        i += 1
        if len(fields) >= 2:
            evaluation_flag = 1
        if evaluation_flag == 1:
            if fields[1] == predicted_lemma:
                j += 1;
        file_writer.write(fields[0] + '\t' + predicted_lemma + '\n')
f.close()
file_writer.close()
if evaluation_flag == 1:
    print ('Accuracy = ', float(j)*100.0/i)























