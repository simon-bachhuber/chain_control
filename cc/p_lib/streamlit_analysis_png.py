""" `streamlit run streamlit_analysis.py`
"""

import base64

import numpy as np
import pandas as pd
import streamlit as st

prefix1 = "/home/simon/Documents/PYTHON/adv_chain_control/sweeps/sweep_02_11/"
prefix2 = prefix1 + "run_gridsearch_01_11_22/"
EXTENSION = ".png"

df = pd.read_pickle(prefix1 + "objective_2022-11-02_18-28-41.pkl")
default_options = ["v1", 2, 2, 0, 0, True, True, "random_steps"]

keys = [
    "data_id",
    "m_ss",
    "c_ss",
    "m_n_layers",
    "c_n_layers",
    "m_dropout",
    "c_add_disturbances",
    "tree_transform",
]
values = [
    ["v1", "v2", "v3"],
    [2, 5, 20, 50],
    [2, 5, 20, 50],
    [0, 2],
    [0, 2],
    [True, False],
    [True, False],
    ["random_steps", "random_transition_to_step"],
]

# Increase width of usable area
st.set_page_config(layout="wide")

# Remove whitespace from the top of the page and sidebar
st.markdown(
    """
        <style>
               .css-18e3th9 {
                    padding-top: 2.0rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)


def show_pdf(file_path, width=950, height=1250):
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
            pdf_display = f'<iframe src="data:application/pdf;base64,\
                {base64_pdf}" width="{width}" height="{height}" type=\
                    "application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception:
        st.text("File not found!")


def show_png(file_path, width=950, height=1250):
    try:
        st.image(file_path)
    except Exception:
        st.text("File not found!")


def where(df, key, value):
    return df[df[f"config/{key}"] == value]


def MSE_from_df(options):
    df_temp = df.copy()
    for key, value in zip(keys, options):
        try:
            df_temp = where(df_temp, key, value)
        except KeyError:
            print(f"Key {key} not found in {df.columns}")
    try:
        return df_temp["MSE"].tolist()[0]
    except Exception:
        return np.nan


def local_MSE_search(options):
    baseline = MSE_from_df(options)

    mse_up_down = []
    for key, value, value_range in zip(keys, options, values):

        i = value_range.index(value)
        upper_bound = len(value_range)
        i_up = min(i + 1, upper_bound - 1)
        i_down = max(i - 1, 0)

        up_down = []
        for i in [i_up, i_down]:
            new_value = value_range[i]
            j = keys.index(key)
            vary_options = options.copy()
            vary_options[j] = new_value

            MSE_vary = MSE_from_df(vary_options)

            up_down.append((new_value, MSE_vary - baseline))
        mse_up_down.append(up_down)
    return mse_up_down


def set_select_boxes(options, key_start, not_if_exists=False):
    options_out = []
    for option in options:
        key_start += 1
        key = f"select_box{key_start}"
        if not_if_exists:
            if key in st.session_state:
                options_out.append(st.session_state[key])
                continue
        st.session_state[key] = option
        options_out.append(option)
    return options_out


def build_file_str(mse_up_down, st_key_start):
    file = ""
    st_key = st_key_start
    options = []
    for i, (key, value, mses) in enumerate(zip(keys, values, mse_up_down)):
        st_key += 1
        v = st.selectbox(key, value, key=f"select_box{st_key}")
        st.text(
            "{}: {:.2f} | {}: {:.2f}".format(
                mses[0][0], mses[0][1], mses[1][0], mses[1][1]
            )
        )
        options.append(v)
        file += f"{key}={v},"
    file += EXTENSION
    return file, options


options1 = set_select_boxes(default_options, 0, True)
options2 = set_select_boxes(default_options, 100, True)


col1, col2, col3, col4 = st.columns([1.5, 6, 1.5, 6])

with col1:
    if st.button("Sync `Pull from Right`"):
        options1 = set_select_boxes(options2, 0)
    mse_up_down1 = local_MSE_search(options1)
    file1, options1 = build_file_str(mse_up_down1, 0)

with col2:
    show_png(prefix2 + file1)

with col3:
    if st.button("Sync `Pull from Left`"):
        options2 = set_select_boxes(options1, 100)
    mse_up_down2 = local_MSE_search(options2)
    file2, options2 = build_file_str(mse_up_down2, 100)

with col4:
    show_png(prefix2 + file2)
