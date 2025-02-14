import pickle

def save_pickle(obj, file_name):
    pkl_file = open(file_name,'wb')
    pickle.dump(obj, pkl_file)
    pkl_file.close()