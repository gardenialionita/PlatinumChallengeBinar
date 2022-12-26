import re
import pickle
import pandas as pd
import os

from keras_preprocessing.sequence import pad_sequences

class TextProcessing():
    def __init__(self) -> None:
        with open(os.path.join("models", "vectorizer.pkl"), 'rb') as f:
            self.vectorizer = pickle.load(f)

        with open(os.path.join("models", "tfidf.pkl"), 'rb') as f:
            self.tfidf = pickle.load(f)

        with open(os.path.join("models", 'tokenizer.pkl'),'rb') as f:
            self.tokenizer = pickle.load(f)
        pass

    def remove_emoji_csv(self,line) -> str:
        return re.sub(r"\\x[A-Za-z0-9./]+", "", line)

    def remove_emoji(self,line) -> str:
        return line.encode('ascii', 'ignore').decode("utf-8")

    def remove_enter(self,line) -> str:
        # line = line.replace('\n',' ')
        # line = line.replace('\\n',' ')
        return re.sub('\\n','', line)

    def remove_punct(self,line) -> str:
        return re.sub(r'[^\w\s\d]',' ',line)

    def get_bow(self,text):
        text = text.lower()
        text = self.remove_emoji(text)
        text = self.remove_punct(text)
        text = self.remove_enter(text)
        clean_text = text
        new_df = pd.DataFrame([text],columns=['text'])
        target_predict = self.vectorizer.transform(new_df['text'])
        target_predict = self.tfidf.transform(target_predict)
        return clean_text,target_predict

    def get_tokenizer(self,text) -> list:
        text = text.lower()
        text = self.remove_emoji(text)
        text = self.remove_punct(text)
        text = self.remove_enter(text)
        clean_text = text
        text = self.tokenizer.texts_to_sequences([text])
        return clean_text, pad_sequences(text, maxlen=78)