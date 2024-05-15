import pandas as pd

def load_csvfile(csvfile, mode):
    list_of_dataframes = []
    for filename in csvfile:
        f = pd.read_csv(filename)
        if mode=='train':
            new_f = f
        elif mode =='valid':
            new_f = f
        else:
            new_f = f.iloc[100:]
        list_of_dataframes.append(new_f)
    merged_df = pd.concat(list_of_dataframes)
    print(len(merged_df))
    return merged_df

def get_trainfile():
    filepath = []
    
    df = load_csvfile(filepath, 'train')
    return df 
    
def get_validfile():
    filepath = []    
    
    df = load_csvfile(filepath, 'valid')
    return df 

def get_testfile():    
    seen_filepath = []
                
    unseen_filepath = []
    
    df_seen, df_unseen = load_csvfile(seen_filepath, 'test'), load_csvfile(unseen_filepath, 'test')   
    return df_seen, df_unseen 
