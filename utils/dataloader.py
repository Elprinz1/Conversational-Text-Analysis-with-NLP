import pandas as pd
from glob import glob


def load_subtitles_dataset(dataset_path):
    paths = glob(dataset_path+'/*.ass')
    paths.sort()

    scripts = []
    episodes = []

    for path in paths:
        with open(path, 'r') as f:
            lines = f.readlines()
            lines = lines[27:]
            lines = [','.join(line.split(',')[9:]) for line in lines]
        lines = [line.replace('\\N', ' ') for line in lines]
        script = ' '.join(lines)

        episode = int(path.split('-')[-1].split('.')[0].strip())

        scripts.append(script)
        episodes.append(episode)

    df = pd.DataFrame.from_dict({'episode': episodes, 'script': scripts})

    return df
