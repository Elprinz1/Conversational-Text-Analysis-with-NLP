from utils import load_subtitles_dataset
import nltk
import torch
from transformers import pipeline
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd
import os
import sys
import pathlib
folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path, '../'))
# Add the parent directory to the system path for module imports
# sys.path.insert(0, str(folder_path.parent))
nltk.download('punkt')
nltk.download('punkt_tab')


class ThemeClassifier():
    def __init__(self, theme_list):
        self.model = 'facebook/bart-large-mnli'
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.theme_list = theme_list
        self.theme_classifier = self.load_model(self.device)

    def load_model(self, device):
        theme_classifier = pipeline(
            'zero-shot-classification',
            model=self.model,
            device=device
        )
        return theme_classifier

    def get_theme_inference(self, script):
        # Tokenize the script into sentences
        script_sentences = sent_tokenize(script)

        # Split the sentences into batches
        batch_size = 20
        batches = []
        for index in range(0, len(script_sentences), batch_size):
            sent = ' '.join(script_sentences[index:index+batch_size])
            batches.append(sent)

        # Get the theme inference from the model
        theme_output = self.theme_classifier(
            batches[:2],
            self.theme_list,
            multi_label=True
        )

        # Wrangle the output and store in a dictionary
        theme_dict = {}
        for output in theme_output:
            for label, score in zip(output['labels'], output['scores']):
                if label not in theme_dict:
                    theme_dict[label] = [score]
                else:
                    theme_dict[label].append(score)
        themes = {key: np.mean(np.array(value))
                  for key, value in theme_dict.items()}

        return themes

    def get_themes(self, dataset_path, save_model_path=None):
        # Check if save path exists
        if save_model_path is not None and os.path.exists(save_model_path):
            df = pd.read_csv(save_model_path)

            return df

        # load dataset
        df = load_subtitles_dataset(dataset_path=dataset_path)

        df = df.head(2)

        # Run inference
        output_theme = df['script'].apply(self.get_theme_inference)

        themes_df = pd.DataFrame(output_theme.to_list())
        df[themes_df.columns] = themes_df

        # Save output
        if save_model_path is not None:
            df.to_csv(save_model_path, index=False)

        return df
