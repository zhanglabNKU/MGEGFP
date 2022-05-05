import lightgbm as lgb
from sklearn.model_selection import train_test_split
from validation import evaluate_performance
import numpy as np
from sklearn.preprocessing import StandardScaler,minmax_scale
from sklearn.metrics import zero_one_loss

from preprocessing import load_gene_list, extract_sub_gene_feature


# input subset number
task = 'level1'


path1 = './data/networks/yeast/'
path2 = './data/annotations/yeast/'


# load the learned feature matrix
feature_matrix_all = np.load('z_all_best.npy')

# print(feature_matrix_all.shape)



gene_list_all = load_gene_list(path1, 'yeast_string_genes.txt')
gene_list_level1 = load_gene_list(path2, 'yeast_mips_level1_genes.txt')
gene_list_level2 = load_gene_list(path2, 'yeast_mips_level2_genes.txt')
gene_list_level3 = load_gene_list(path2, 'yeast_mips_level3_genes.txt')


feature_matrix_level1 = np.zeros((4443,feature_matrix_all.shape[1]))
feature_matrix_level1 = extract_sub_gene_feature(gene_list_level1, gene_list_all,
                 feature_matrix_level1, feature_matrix_all)

feature_matrix_level2 = np.zeros((4428,feature_matrix_all.shape[1]))
feature_matrix_level2 = extract_sub_gene_feature(gene_list_level2, gene_list_all,
                 feature_matrix_level2, feature_matrix_all)

feature_matrix_level3 = np.zeros((4061,feature_matrix_all.shape[1]))
feature_matrix_level3 = extract_sub_gene_feature(gene_list_level3, gene_list_all,
                 feature_matrix_level3, feature_matrix_all)



# print(feature_matrix_level1.shape)
# print(feature_matrix_level2.shape)
# print(feature_matrix_level3.shape)



# X = StandardScaler().fit_transform(X)
# X = minmax_scale(X, axis=0)



if task == 'level1':
    X = feature_matrix_level1
    Y = np.load('./data/label_yeast/label_matrix_level1.npy')
elif task == 'level2':
    X = feature_matrix_level2
    Y = np.load('./data/label_yeast/label_matrix_level2.npy')
elif task == 'level3':
    X = feature_matrix_level3
    Y = np.load('./data/label_yeast/label_matrix_level3.npy')

# print(X.shape)


params = {

    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss', 'auc'},
    # 'metric': {'l2', 'auc'},
    'n_estimators':550,
    'learning_rate': 0.05,
    'num_leaves': 16,
    'max_depth':7,
    # 'subsample':0.8,
    # 'colsample_bytree':0.8,

    'force_col_wise': True,
    'min_data_in_leaf':41,
    'max_bin':35,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 20,

    'verbose': -1

    }



maupr_all = []
Maupr_all = []
acc_all = []
f1_all = []
zero_one_loss_all= []



# def softmax_func(a):
#     return np.exp(a)/np.sum(np.exp(a))

for j in range(10):


    shuffle_indices = np.random.permutation(np.arange(len(Y)))
    emb_shuffled = X[shuffle_indices,:]
    anno_shuffled = Y[shuffle_indices]


    test_sample_percentage = 0.1
    test_sample_index = int(test_sample_percentage * float(len(Y)))
    print(test_sample_index)
    X_test,X_dev,X_train= emb_shuffled[:test_sample_index,:],emb_shuffled[test_sample_index:2*test_sample_index,:],emb_shuffled[2*test_sample_index:,:]#(6,3555,500),(6,888,500)
    y_test,y_dev,y_train = anno_shuffled[:test_sample_index,:],anno_shuffled[test_sample_index:2*test_sample_index,:],anno_shuffled[2*test_sample_index:,:]

    y_score = np.zeros_like(y_test)
    y_pred = np.zeros_like(y_test)



    for i in range(y_train.shape[1]):
        train_data = lgb.Dataset(X_train, label=y_train[:,i])
        validation_data = lgb.Dataset(X_dev, label=y_dev[:,i])
        clf = lgb.train(params, train_data, valid_sets = [validation_data])
        y_score_sub = clf.predict(X_test)

        y_pred_sub = (y_score_sub >= 0.5)*1

        y_score[:, i] = y_score_sub
        y_pred[:, i] = y_pred_sub


    result = evaluate_performance(y_test, y_score, y_pred)
    zero = zero_one_loss(y_test,y_pred)



    maupr_all.append(result['m-aupr'])
    Maupr_all.append(result['M-aupr'])
    acc_all.append(result['acc'])
    f1_all.append(result['F1'])
    zero_one_loss_all.append(zero)





print('acc:',np.mean(acc_all),np.std(acc_all))
print('f1:',np.mean(f1_all),np.std(f1_all))
print('m-aupr:',np.mean(maupr_all),np.std(maupr_all))
print('M-aupr:',np.mean(Maupr_all),np.std(Maupr_all))
print('subset zero_one loss:',np.mean(zero_one_loss_all),np.std(zero_one_loss_all))




##
