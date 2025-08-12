import dash
import tkinter as tk
from tkinter import filedialog
from layouts.base_layout import create_base_layout
from layouts.tab_layouts import tab1_layout, tab2_layout
from utils.data_loader import list_csv_files, extract_participant_id
from layouts.tab_layouts import tab3_layout

# Folder selection using tkinter
root = tk.Tk()
root.withdraw()
folder1 = filedialog.askdirectory(title="Select Concat_File folder")
folder2 = filedialog.askdirectory(title="Select the merged_labels folder")

# File lists
csv_files_hr = list_csv_files(folder1, "heart")
csv_files_belt = list_csv_files(folder1, "belt")
merge_lables_hr = list_csv_files(folder2, "heart")
csv_files = sorted(csv_files_hr, key=extract_participant_id)

# Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.layout = create_base_layout()

# Tab content callback
from dash import Input, Output, html
@app.callback(Output('tabs-content', 'children'), [Input('url', 'pathname')])
def render_content(pathname):
    if pathname == '/tab1':
        return tab1_layout(csv_files, csv_files_belt)
    elif pathname == '/tab2':
        return tab2_layout(merge_lables_hr)
    elif pathname == '/tab3':
        return tab3_layout()
    elif pathname == '/tab4':
        return html.Div("Tab 4 content")
    elif pathname == '/tab5':
        return html.Div("Tab 5 content")
    else:
        return html.Div("Select a tab")

# Register callbacks
from callbacks.tab1_callbacks import register_tab1_callbacks
from callbacks.tab2_callbacks import register_tab2_callbacks
from callbacks.belt_callbacks import register_belt_callbacks
from callbacks.tab3_callbacks import register_tab3_callbacks

register_tab1_callbacks(app, folder1)
register_tab2_callbacks(app, folder2)
register_belt_callbacks(app, folder1)
register_tab3_callbacks(app, folder2)

if __name__ == '__main__':
    app.run(debug=True)
