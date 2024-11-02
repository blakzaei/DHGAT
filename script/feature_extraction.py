#-- Import -------------------------------------------------------------------------------------
import pandas as pd
import fasttext
import fasttext.util
import os
#-----------------------------------------------------------------------------------------------

#-- Initialize ---------------------------------------------------------------------------------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(base_dir, 'data')
df_file = os.path.join(data_dir, 'liar_df.csv')
fasttext_file = os.path.join(data_dir, 'fasttext_model/cc.en.300.bin')
result_file = os.path.join(data_dir, 'liar_df_fasttext_embeddings.csv')
#-----------------------------------------------------------------------------------------------

#-- Function to get embeddings from fast text for each row in df -------------------------------
def get_text_embedding(text, model):
    embedding = model.get_sentence_vector(text)
    return embedding
#-----------------------------------------------------------------------------------------------

#-- Function to Create embeddings for total rows in df -----------------------------------------
def generate_embeddings_using_fasttext():

    #-- log --
    print('Creating Text Embeddings using FastText Model ...')

    #-- load fast-text model --
    print('loading FastText Model ...')
    model_ft_300d = fasttext.load_model(fasttext_file)

    #-- load df --
    df = pd.read_csv(df_file)

    #-- run fast-text --
    print('Creating embeddings ...')
    df['statement'] = df['statement'].astype(str)
    df['txt_embedding'] = df['statement'].apply(lambda x: get_text_embedding(x, model_ft_300d))

    #-- save --
    print('Saving results ...')

    df.to_csv(result_file, sep=',', encoding='utf-8', index=False)

    print('create_FastText_embeddings: DONE :)\n')
#-----------------------------------------------------------------------------------------------