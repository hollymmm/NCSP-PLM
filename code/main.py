import os

def get_indepentdet_pred():
    run = 'python independent_feature_deep.py' + ' -f ' + 'esm' + ' -m ' + 'att'
    os.system(run)
    run = 'python independent_feature_deep.py' + ' -f ' + 'unirep' + ' -m ' + 'att'
    os.system(run)
    run = 'python independent_feature_deep.py' + ' -f ' + 'ProSE' + ' -m ' + 'bilstm'
    os.system(run)
    run = 'python independent_feature_deep.py' + ' -f ' + 'half_prot_t5_xl_uniref50_protein_embs' + ' -m ' + 'mlp'
    os.system(run)
    run = 'python independent_feature_deep.py' + ' -f ' + 'proteinbertglobal' + ' -m ' + 'mlp'
    os.system(run)
def get_5_folds_pred():
    run = 'python 5_folds_feature_deep.py' + ' -f ' + 'esm' + ' -m ' + 'att'
    os.system(run)
    run = 'python 5_folds_feature_deep.py' + ' -f ' + 'unirep' + ' -m ' + 'att'
    os.system(run)
    run = 'python 5_folds_feature_deep.py' + ' -f ' + 'ProSE' + ' -m ' + 'bilstm'
    os.system(run)
    run = 'python 5_folds_feature_deep.py' + ' -f ' + 'half_prot_t5_xl_uniref50_protein_embs' + ' -m ' + 'mlp'
    os.system(run)
    run = 'python 5_folds_feature_deep.py' + ' -f ' + 'proteinbertglobal' + ' -m ' + 'mlp'
    os.system(run)
def independent_vote():
    get_indepentdet_pred()
    run = 'python independent_vote.py'+' -w'+' 1 '+'1 '+'1 '+'3 '+'1'
    os.system(run)
def five_folds_vote():
    get_5_folds_pred()
    run = 'python 5_folds_vote.py' + ' -w' + ' 1 ' + '1 ' + '1 ' + '3 ' + '1'
    os.system(run)
if __name__ == '__main__':
    independent_vote()
    five_folds_vote()






