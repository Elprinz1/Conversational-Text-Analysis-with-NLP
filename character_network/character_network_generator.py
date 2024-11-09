import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pyvis.network as net


class CharacterNetworkGenerator():
    """
    This class is responsible for generating the character network based on the named entities recognized in the script.

    The character network is a graph where the nodes are the characters and the edges are the interactions between them.

    The interactions are calculated based on the proximity of the characters in the script.

    The closer the characters are in the script, the higher the weight of the edge between them.

    The character network is represented as a pandas DataFrame with the following columns:

    - source: the source character
    - target: the target character
    - value: the weight of the edge between the source and target characters
    """

    def __init__(self) -> None:
        pass

    def generate_character_network(self, df):
        window_size = 10
        entity_relations = []
        for row in df['ners']:
            previous_entities = []
            for sentence in row:
                previous_entities.append(list(sentence))
                previous_entities = previous_entities[-window_size:]

                # Flattern the list from 2D to 1D
                previous_entities_flatten = sum(previous_entities, [])

                # Generate all possible combinations of entities
                for entity in sentence:
                    for previous_entity in previous_entities_flatten:
                        if entity != previous_entity:
                            entity_relations.append(
                                ([entity, previous_entity]))

        # Create the df
        entity_relations_df = pd.DataFrame({'value': entity_relations})
        entity_relations_df['source'] = entity_relations_df['value'].apply(
            lambda x: x[0])
        entity_relations_df['target'] = entity_relations_df['value'].apply(
            lambda x: x[1])
        entity_relations_df = entity_relations_df.groupby(
            ['source', 'target']).count().reset_index()
        entity_relations_df = entity_relations_df.sort_values(
            'value', ascending=False)

        return entity_relations_df

    def draw_character_network(self, relations_df):
        relations_df = relations_df.sort_values('value', ascending=False)
        network_df = relations_df[:200].copy()

        G = nx.from_pandas_edgelist(
            network_df,
            source='source',
            target='target',
            edge_attr='value',
            create_using=nx.Graph())

        # Plot the network with pyvis
        nt = net.Network(height='750px', width='1000px', notebook=True,
                         bgcolor='#222222', font_color='white', cdn_resources='remote')
        node_degrees = dict(G.degree)

        nx.set_node_attributes(G, node_degrees, 'size')
        nt.from_nx(G)

        html = nt.generate_html()
        html = html.replace("'", "\"")

        output_html = f"""<iframe style="width: 100%; height: 600px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera;
    display-capture; encrypted-media;" sandbox="allow-modals allow-forms
    allow-scripts allow-same-origin allow-popups
    allow-top-navigation-by-user-activation allow-downloads" allowfullscreen=""
    allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""

        return output_html
