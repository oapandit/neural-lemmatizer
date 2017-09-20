from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Embedding, merge, Reshape
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional, Flatten
from keras.layers.core import Masking

def get_char_model(model_name, max_word_length, embedded_char_vector_length, char_feature_output):
    if model_name == 'BLSTM':
        char_input = Input(shape=(max_word_length,), dtype='float32', name='char_input')
        char_input1 = Embedding(1000, embedded_char_vector_length, input_length=max_word_length)(char_input)
        char_input2 = Dropout(0.2)(char_input1)
        lstm_out_forward = LSTM(char_feature_output, dropout_W=0.2, dropout_U=0.2)(char_input2)
        lstm_out_backward = LSTM(char_feature_output, dropout_W=0.2, dropout_U=0.2, go_backwards=True)(char_input2)
        merged = merge([lstm_out_forward, lstm_out_backward], mode='concat', concat_axis=1)
        model = Model(input=[char_input], output=[merged])
        return model

    elif model_name == 'BGRU':
        char_input = Input(shape=(max_word_length,), dtype='float32', name='char_input')
        char_input1 = Embedding(1000, embedded_char_vector_length, input_length=max_word_length)(char_input)
        char_input2 = Dropout(0.2)(char_input1)
        gru_out_forward = GRU(char_feature_output, dropout_W=0.2, dropout_U=0.2)(char_input2)
        gru_out_backward = GRU(char_feature_output, dropout_W=0.2, dropout_U=0.2, go_backwards=True)(char_input2)
        merged = merge([gru_out_forward, gru_out_backward], mode='concat', concat_axis=1)
        model = Model(input=[char_input], output=[merged])
        return model

    elif model_name == 'SimpleRNN':
        char_input = Input(shape=(max_word_length,), dtype='float32', name='char_input')
        char_input1 = Embedding(1000, embedded_char_vector_length, input_length=max_word_length)(char_input)
        char_input2 = Dropout(0.2)(char_input1)
        SimpleRNN_out_forward = SimpleRNN(char_feature_output, dropout_W=0.2, dropout_U=0.2)(char_input2)
        SimpleRNN_out_backward = SimpleRNN(char_feature_output, dropout_W=0.2, dropout_U=0.2, go_backwards=True)(char_input2)
        merged = merge([SimpleRNN_out_forward, SimpleRNN_out_backward], mode='concat', concat_axis=1)
        model = Model(input=[char_input], output=[merged])
        return model

def get_tree_classify_model(model_name, word_context_length, word_vector_size, char_feature_output, hidden_size, nb_tree_classes):

    if model_name == 'BLSTM':
        word_input = Input(shape=(word_context_length, word_vector_size,), name='word_input')
        char_vector_input = Input(shape=(word_context_length, char_feature_output,), name='char_vector_input')
        merged = merge([word_input, char_vector_input], mode='concat', concat_axis=2)
        merged = Masking(mask_value=0.,)(merged)
        x = Bidirectional(LSTM(hidden_size, return_sequences=True, dropout_W=0.2, dropout_U=0.2))(merged)
        applicable_tree_input = Input(shape=(word_context_length, nb_tree_classes,),  name='applicable_tree_input')
        x = merge([x, applicable_tree_input], mode='concat', concat_axis=2)
        main_loss1 = TimeDistributed(Dense(nb_tree_classes, activation='softplus'))(x)
        main_loss = Activation('softmax')(main_loss1)
        model = Model(input=[word_input, char_vector_input, applicable_tree_input], output=[main_loss])
        return model

    elif model_name == 'BGRU':
        word_input = Input(shape=(word_context_length, word_vector_size,), name='word_input')
        char_vector_input = Input(shape=(word_context_length, char_feature_output,), name='char_vector_input')
        merged = merge([word_input, char_vector_input], mode='concat', concat_axis=2)
        merged = Masking(mask_value=0., )(merged)
        x = Bidirectional(GRU(hidden_size, return_sequences=True, dropout_W=0.2, dropout_U=0.2))(merged)
        applicable_tree_input = Input(shape=(word_context_length, nb_tree_classes,),  name='applicable_tree_input')
        x = merge([x, applicable_tree_input], mode='concat', concat_axis=2)
        main_loss1 = TimeDistributed(Dense(nb_tree_classes, activation='softplus'))(x)
        main_loss = Activation('softmax')(main_loss1)
        model = Model(input=[word_input, char_vector_input, applicable_tree_input], output=[main_loss])
        return model

    elif model_name == 'SimpleRNN':
        word_input = Input(shape=(word_context_length, word_vector_size,), name='word_input')
        char_vector_input = Input(shape=(word_context_length, char_feature_output,), name='char_vector_input')
        merged = merge([word_input, char_vector_input], mode='concat', concat_axis=2)
        merged = Masking(mask_value=0., )(merged)
        x = Bidirectional(SimpleRNN(hidden_size, return_sequences=True, dropout_W=0.2, dropout_U=0.2))(merged)
        applicable_tree_input = Input(shape=(word_context_length, nb_tree_classes,),  name='applicable_tree_input')
        x = merge([x, applicable_tree_input], mode='concat', concat_axis=2)
        main_loss1 = TimeDistributed(Dense(nb_tree_classes, activation='softplus'))(x)
        main_loss = Activation('softmax')(main_loss1)
        model = Model(input=[word_input, char_vector_input, applicable_tree_input], output=[main_loss])
        return model



