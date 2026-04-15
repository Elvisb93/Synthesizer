import gradio as gr
import pandas as pd

def update_df():
    data = {"A": [1, 2, 3], "B": [4, 5, 6]}
    return pd.DataFrame(data)

def update_list():
    return [[1, 4], [2, 5], [3, 6], [4, 7]]

with gr.Blocks() as demo:
    df1 = gr.Dataframe(headers=["A", "B"], interactive=True, row_count=(1, "dynamic"))
    df2 = gr.Dataframe(headers=["A", "B"], interactive=True)
    
    btn1 = gr.Button("Update DF")
    btn2 = gr.Button("Update List")
    
    btn1.click(update_df, outputs=df1)
    btn2.click(update_list, outputs=df2)

demo.launch()
