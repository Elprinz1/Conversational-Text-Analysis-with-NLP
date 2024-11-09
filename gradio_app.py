import gradio as gr
from theme_classifier import ThemeClassifier
import pandas as pd
from character_network.named_entity_recognizer import NamedEntityRecognizer
from character_network.character_network_generator import CharacterNetworkGenerator


def get_themes(theme_list_str, subtitles_path, save_path):
    theme_list = theme_list_str.split(',')
    theme_list = [theme.strip() for theme in theme_list]
    theme_classifier = ThemeClassifier(theme_list)

    df = theme_classifier.get_themes(subtitles_path, save_path)

    # Remove dialogue column from df
    theme_list = [theme for theme in theme_list if theme != 'dialogue']
    df = df[theme_list]

    df = df[theme_list].sum().reset_index()
    df.columns = ['Theme', 'Score']

    chart_obj = gr.BarPlot(
        df,
        x='Score',
        y='Theme',
        title='Series Themes',
        tooltip=['Theme', 'Score'],
        vertical=False,
        sort=['x', 'y'],
        height=250
    )

    return chart_obj


def get_character_network(subtitles_path, ner_path):

    ner = NamedEntityRecognizer()
    df = ner.get_ners(subtitles_path, output_path=ner_path)

    cng = CharacterNetworkGenerator()
    relations_df = cng.generate_character_network(df)
    html = cng.draw_character_network(relations_df)

    return html


def main():
    with gr.Blocks() as iface:

        # Theme Classification Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Theme Classification (Zero Shot Classifier)</h1>")
                with gr.Row():  # inner row for 2 columns to work
                    with gr.Column():
                        df = pd.DataFrame()
                        plot = gr.BarPlot()
                    with gr.Column():
                        theme_list = gr.Textbox(label="Themes")
                        subtitles_path = gr.Textbox(
                            label="Subtitles or Scripts Path")
                        save_path = gr.Textbox(label="Path to save blob")
                        get_themes_button = gr.Button("Get Themes")
                        get_themes_button.click(
                            get_themes, inputs=[theme_list, subtitles_path, save_path], outputs=[plot])

        # Character Network Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Network (NERs & Graphs</h1>")
                with gr.Row():  # inner row for 2 columns to work
                    with gr.Column():
                        network_html = gr.HTML()
                    with gr.Column():
                        subtitles_path = gr.Textbox(
                            label="Subtitles or Scripts Path")
                        ner_path = gr.Textbox(label="Path to save NER graph")
                        get_network_graph = gr.Button(
                            "Get Character Network Graph")
                        get_network_graph.click(get_character_network, inputs=[
                                                subtitles_path, ner_path], outputs=[network_html])

    iface.launch(share=True, debug=True)


if __name__ == "__main__":
    main()
