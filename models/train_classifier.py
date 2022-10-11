"""
TRAIN CLASSIFIER
Disaster Resoponse Project
Udacity - Data Science Nanodegree
How to run this script (Example)
> python train_classifier.py ../data/DisasterResponse.db classifier.pkl
Arguments:
    1) SQLite db path (containing pre-processed data)
    2) pickle file name to save ML model
"""

import nltk
nltk.download('omw-1.4')
nltk.download(['punkt', 'wordnet','averaged_perceptron_tagger'])
import nltk
import re
import nltk
import numpy as np
import pandas as pd
import pickle
import os
import sys
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import fbeta_score, classification_report
from scipy.stats.mstats import gmean

def load_data(database_filepath):
    """
    Load Data Function
    
    Arguments:
        database_filepath -> path to SQLite db
    Output:
        X -> feature DataFrame
        Y -> label DataFrame
        category_names -> used for data visualization (app)
    """
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","") + "_table"
    df = pd.read_sql_table(table_name,engine)
    df.columns = df.columns.str.replace(r"'|,|[(|)]", "")
    
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    """
    Tokenize function
    
    Arguments:
        text -> list of text messages (english)
    Output:
        clean_tokens -> tokenized text, clean for ML modeling
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip().lower()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """
    def starting_verb(self, text):
        sent_list = nltk.sent_tokenize(text)
        for sentence in sent_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    """
    Build Model function
    
    This function output is a Scikit ML Pipeline that process text messages
    according to NLP best-practice and apply a classifier.
    """
    model = Pipeline([
        ('feature', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = {
    'clf__estimator__learning_rate': [0.5, 1.0],
    'clf__estimator__n_estimators': [10, 20]
    }
    cv = GridSearchCV(pipeline2, param_grid=parameters, scoring='f1_micro', n_jobs=-1, verbose=3)
    
    return cv

def multioutput_fscore(y_true,y_pred,beta=1):
    """
    MultiOutput Fscore
    
    This is a performance metric of my own creation.
    It is a sort of geometric mean of the fbeta_score, computed on each label.
    
    It is compatible with multi-label and multi-class problems.
    It features some peculiarities (geometric mean, 100% removal...) to exclude
    trivial solutions and deliberatly under-estimate a stangd fbeta_score average.
    The aim is avoiding issues when dealing with multi-class/multi-label imbalanced cases.
    
    It can be used as scorer for GridSearchCV:
        scorer = make_scorer(multioutput_fscore,beta=1)
        
    Arguments:
        y_true -> labels
        y_prod -> predictions
        beta -> beta value of fscore metric
    
    Output:
        f1score -> customized fscore
    """
    score_list = []
    if isinstance(y_pred, pd.DataFrame) == True:
        y_pred = y_pred.values
    if isinstance(y_true, pd.DataFrame) == True:
        y_true = y_true.values
    for column in range(0,y_true.shape[1]):
        score = fbeta_score(y_true[:,column],y_pred[:,column],beta,average='weighted')
        score_list.append(score)
    f1score_numpy = np.asarray(score_list)
    f1score_numpy = f1score_numpy[f1score_numpy<1]
    f1score = gmean(f1score_numpy)
    return  f1score

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate Model function
    
    This function applies a ML pipeline to a test set and prints out the model performance (accuracy and f1score)
    
    Arguments:
        pipeline -> A valid scikit ML Pipeline
        X_test -> Test features
        Y_test -> Test labels
        category_names -> label names (multi-output)
    """
    Y_pred = model.predict(X_test)
     
    multi_f1 = multioutput_fscore(Y_test,Y_pred, beta = 1)
    overall_accuracy = (Y_pred == Y_test).mean().mean()

    print('Average overall accuracy {0:.2f}%'.format(overall_accuracy*100))
    print('F1 score (custom definition) {0:.2f}%'.format(multi_f1*100))

    # Print the whole classification report.
    Y_pred = pd.DataFrame(Y_pred, columns = Y_test.columns)
    
    for column in Y_test.columns:
        print('Model Performance with Category: {}'.format(column))
        print(classification_report(Y_test[column],Y_pred[column]))

def save_model(model, model_filepath):
    """
    Save Model function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        model -> GridSearchCV or Scikit Pipelin object
        model_filepath -> destination path to save .pkl file
    
    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    """
    Train Classifier Main function
    
    This function applies the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle
    
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
