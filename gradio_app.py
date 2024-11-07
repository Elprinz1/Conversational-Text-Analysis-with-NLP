import gradio as gr
from theme_classifier import ThemeClassifier
import pandas as pd


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


def main():
    with gr.Blocks() as iface:
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Theme Classification (Zero Shot Classifier)</h1>")
                with gr.Row():  # inner row for 2 columns to work
                    with gr.Column():
                        df = pd.DataFrame()
                        plot = gr.BarPlot(df )
                    with gr.Column():
                        theme_list = gr.Textbox(label="Themes")
                        subtitles_path = gr.Textbox(
                            label="Subtitles or Scripts Path")
                        save_path = gr.Textbox(label="Path to save blob")
                        get_themes_button = gr.Button("Get Themes")
                        get_themes_button.click(
                            get_themes, inputs=[theme_list, subtitles_path, save_path], outputs=[plot])

    iface.launch(share=True, debug=True)


if __name__ == "__main__":
    main()
