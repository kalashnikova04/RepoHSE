import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score

df_train = pd.read_csv('train.csv', index_col=0) 
df_test = pd.read_csv('test.csv', index_col=0) 

X_train, X_val, y_train, y_val = train_test_split(df_train[['text']], df_train['label'],
                                                    test_size=0.3, random_state=42)

X_test = df_test[['text']]
y_test = df_test['label']

text_features = ['text']

train_pool = Pool(
        X_train, 
        y_train, 
        text_features=text_features,
        feature_names=text_features
    )

valid_pool = Pool(
        X_val, 
        y_val, 
        text_features=text_features,
        feature_names=text_features
    )

def objective(trial):

    catboost_params = {
        "loss_function": 'MultiClass',
        "iterations": trial.suggest_int("iterations", 1000, 3000),
        "depth": trial.suggest_int("depth", 2, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 0, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        'task_type': 'GPU',
        'early_stopping_rounds': trial.suggest_int("early_stopping_rounds", 100, 1000),
        'eval_metric': 'TotalF1:average=Macro',
        'verbose': 300
    }

    text_processing = {
        "tokenizers" : [{
            'tokenizer_id': 'Sense',
            'separator_type': 'BySense',
            'lowercasing': 'True',
            'token_types':['Word', 'Number', 'SentenceBreak'],
            'sub_tokens_policy':'SeveralTokens'
        },
        {
            "tokenizer_id" : "Space",
            "separator_type" : "ByDelimiter",
            "lowercasing": "True",
            "delimiter" : " "
        }],
    
        "dictionaries" : [{
            "dictionary_id" : "BiGram",
            "max_dictionary_size" : "50000",
            "occurrence_lower_bound" : "100",
            "gram_order" : "2"
        }, {
            "dictionary_id" : "Word",
            "max_dictionary_size" : "50000",
            "occurrence_lower_bound" : "100",
            "gram_order" : "1"
        },{
            "dictionary_id" : "3-Gram",
            "max_dictionary_size" : "50000",
            "occurrence_lower_bound" : "100",
            "gram_order" : "3"
        }],
        "feature_processing" : {
            "default" : [{
                "dictionaries_names" : ["BiGram", "Word", "3-Gram"],
                "feature_calcers" : ["BoW"],
                "tokenizers_names" : ["Space"]
            }, {
                "dictionaries_names" : ["Word", "BiGram", "3-Gram"],
                "feature_calcers" : ["NaiveBayes"],
                "tokenizers_names" : ["Space"]
            }],
        }
    }


    model = CatBoostClassifier(**catboost_params, text_processing=text_processing) 
    model.fit(train_pool, eval_set=valid_pool)
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds, average='macro')

    return f1