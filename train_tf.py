import configparser
from data_u import Corpus
from Utility import *
from VanillaLSTM_TF import VanillaLSTM_TF


def load_params(param_file='params.ini'):
    options = {}
    config = configparser.ConfigParser()
    config.read(param_file)
    options['data'] = config['general']['data']
    options['type'] = config['general']['type']
    options['data_dir'] = options['data'] + '/' + options['type']
    options['word_emb'] = int(config['general']['word_emb'])
    options['batch_size'] = int(config['general']['batch_size'])
    options['e_batch_size'] = int(config['general']['e_batch_size'])
    options['state_size'] = int(config['general']['state_size'])
    options['epochs'] = int(config['general']['epochs'])
    options['lr'] = float(config['general']['lr'])
    options['test_freq'] = int(config['general']['test_freq'])

    return options


if __name__ == '__main__':
    options = load_params()
    corpus = Corpus(options['data_dir'])

    '''train_data = select_data(corpus.train, options['batch_size'])
    train_len = select_data(corpus.train_len, options['batch_size'])
    train_label = select_data(corpus.train_label, options['batch_size'])

    val_data = select_data(corpus.valid, options['e_batch_size'])
    val_len = select_data(corpus.valid_len, options['e_batch_size'])
    val_label = select_data(corpus.valid_label, options['e_batch_size'])

    test_data = select_data(corpus.test, options['e_batch_size'])
    test_len = select_data(corpus.test_len, options['e_batch_size'])
    test_label = select_data(corpus.test_label, options['e_batch_size'])'''

    '''emb_matrix = torch.load(options['data'] + '/emb_matrix.pt', map_location=lambda storage, loc: storage)
    word_idx_list = torch.load(options['data'] + '/word_idx_list.pt', map_location=lambda storage, loc: storage)
    word_idx_list = word_idx_list.numpy()
    print(emb_matrix.shape, word_idx_list.shape)'''
    n_classes = np.max(np.unique(corpus.train_label)) + 1
    options['seq_len'] = np.max(np.concatenate([corpus.train_len, corpus.valid_len, corpus.test_len]))
    ntokens = len(corpus.dictionary)
    # print(len(train_data))
    print(n_classes, ntokens)
    options['vocab_size'] = ntokens
    options['n_classes'] = n_classes

    model = VanillaLSTM_TF(options)
    model.run_model( (corpus.train, corpus.train_label, corpus.train_len),
                     (corpus.test, corpus.test_label, corpus.test_len),
                     (corpus.valid, corpus.valid_label, corpus.valid_len))