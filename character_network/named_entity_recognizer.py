from utils import load_subtitles_dataset
import spacy
from nltk.tokenize import sent_tokenize
import pandas as pd
from ast import literal_eval
import os
import sys
import pathlib
folder_path = pathlib.Path().parent.resolve()
sys.path.append(os.path.join(folder_path, '../'))


class NamedEntityRecognizer():
    def __init__(self):
        self.nlp_model = self.load_ner_model()

    def load_ner_model(self):
        nlp = spacy.load("en_core_web_trf")
        return nlp

    def get_ners_inference(self, script):
        sentences = sent_tokenize(script)

        ner_output = []
        for sentence in sentences:
            doc = self.nlp_model(sentence)
            ners = set()
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    fullname = ent.text
                    firstname = fullname.split(' ')[0]
                    firstname = firstname.strip()
                    ners.add(firstname)
            ner_output.append(ners)

        return ner_output

    def get_ners(self, dataset_path, output_path=None):
        if output_path is not None and os.path.exists(output_path):
            df = pd.read_csv(output_path)
            df['ners'] = df['ners'].apply(lambda x: literal_eval(
                x) if isinstance(x, str) else x)  # convert string to list
            return df

        # load dataset
        df = load_subtitles_dataset(dataset_path)

        # run inference
        df['ners'] = df['script'].apply(self.get_ners_inference)

        if output_path is not None:
            df.to_csv(output_path, index=False)

        return df
