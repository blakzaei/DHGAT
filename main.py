#-- IMPORT -------------------------------------------------------------------
import sys
from script import load_ds
from script import download_fasttext
from script import feature_extraction
from script import split_train_test
from script import graph_construction
from script import  dhgat
#------------------------------------------------------------------------------

#-- Initiliaze ----------------------------------------------------------------
valid_features = ['speaker', 'party_affiliation',
                  'subject', 'state_info', 'job_title', 'context']
#------------------------------------------------------------------------------


def main(amount_labeled, n_iterations, n_epochs, g_features):

    load_ds.load_and_prepare_LIAR()
    download_fasttext.download()
    feature_extraction.generate_embeddings_using_fasttext()
    split_train_test.split(train_percent=amount_labeled,
                           number_of_iterations=n_iterations)
    graph_construction.creat_graph(g_features)
    dhgat.DHGAT(g_features= g_features,
                num_iterations= n_iterations,
                num_epochs= n_epochs,
                train_percent= amount_labeled,
                l1=0.5,
                l2=0.5)

#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
if __name__ == "__main__":
    
    if len(sys.argv) != 5:
        print('Usage: python main.py <amount_labeled> , <n_iterations>, <n_epochs>, <g_features>')
        sys.exit(1)
    
    amount_labeled = float(sys.argv[1])
    n_iterations = int(sys.argv[2])
    n_epochs = int(sys.argv[3])
    g_features = [item.strip() for item in sys.argv[4].split("-")]

    for feature in g_features:
        if feature not in valid_features:
            print(f'{feature} is not a valid feature.'
                  f'valid features = {valid_features}')
            sys.exit(1)
    
    msg = f'''
            Start DHGAT on LIAR Dataset\n
            A Heterogenous graph is created using features {g_features}\n
            {amount_labeled} amount of samples are labeled\n
            number of epochs for tarining networks is {n_epochs}\n
            after {n_iterations} iteration, results will be evaluated.
        '''
    print(msg)
    
    main(amount_labeled, n_iterations, n_epochs, g_features)
###############################################################################



    



