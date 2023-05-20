#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os
import re
from nltk.corpus import stopwords

def predict(plot):
    
    clf = joblib.load(os.path.dirname(__file__) + '/movie_genre_clf.pkl') 
    tfidf_vectorizer = joblib.load(os.path.dirname(__file__) + '/tfidf_vectorizer.pkl') 
    
    # function for text cleaning 
    def clean_text(text):
        # remove backslash-apostrophe 
        text = re.sub("\'", "", text) 
        # remove everything except alphabets 
        text = re.sub("[^a-zA-Z]"," ",text) 
        # remove whitespaces 
        text = ' '.join(text.split()) 
        # convert text to lowercase 
        text = text.lower() 

        return text
    
    stop_words = set(stopwords.words('english'))
    # function to remove stopwords
    def remove_stopwords(text):
        no_stopword_text = [w for w in text.split() if not w in stop_words]
        return ' '.join(no_stopword_text)
    
    categories = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
        'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
        'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']

    # Top 5 predicted categories 
    def top5(q):
        q = clean_text(q)
        q = remove_stopwords(q)
        q_vec = tfidf_vectorizer.transform([q])
        q_pred = clf.predict_proba(q_vec)
        res = pd.DataFrame(q_pred, columns=categories)
        final_df = res.T.sort_values(by=0, ascending=False).head(5)
        return final_df.index

    return top5(plot)

if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add an URL')
    else:
        
        plot = sys.argv[1]
        
        p1 = predict(plot)
        
        print('Movie genre classification: ', p1)
        