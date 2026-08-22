import streamlit as st
import pandas as pd
import numpy as np

st.title("🎈 My new Streamlit app")

st.title("Hello World!")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

# a sample dataframe
df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
})

df # magic commands (just like jupyter notebooks or inline python!)

# now trying what it does when passed to write function
st.write(df)

# using numpy to generate a df
numpy_df = np.random.randn(10, 20)
st.dataframe(numpy_df)

# highlighting elemnents in the df
pandas_df = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20))
)
st.dataframe(pandas_df.style.highlight_max(axis=0))

# static table generation
st.table(pandas_df)

# DRAW A LINE CHART
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c']
)

st.line_chart(chart_data)

# PLOT A MAP
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon']
)

st.map(map_data)

# WIDGETS: It's really straightforward, think of widgets as variables

# When you've got the data or model into the state that you want to explore,
# you can add in widgets like st.slider(), st.button() or st.selectbox().
# It's really straightforward — treat widgets as variables:


x = st.slider('x') # 👈 this is a widget
st.write(x, 'squared is', x * x)

# Widgets can also be accessed by key,
# if you choose to specify a string to use as the unique key for the widget
st.text_input("Your name", key="name")

# accessing the value at any point with:
st.session_state.name

# USE CHECKBOXES TO SHOW/HIDE DATA
if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['a', 'b', 'c']
    )

    chart_data

# USE SELECTBOX FOR OPTIONS
# Use st.selectbox to choose from a series, You can write in the options you want, or pass through an array or dataframe column.

option = st.selectbox(
    "Which number do you like bext?",
    df['first column']
)

'You selected: ', option

# USING st.sidebar() TO CREATE A LEFT PANEL SIDEBAR.
# ANY ELEMENT PASSED TO st.sidebar() WILL BE RENDERED IN THE SIDEBAR.
# Adding a select box to the sidebar
add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

# Add a slider to the sidebar
add_slider = st.sidebar.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0)
)

# st.columns() is a layout primitive that lets you place elements side by side in a horizontal row.
# You can use it to create a multi-column layout, or to place elements side by side
left_column, right_column = st.columns(2)
# You can use a column like st.sidebar:
left_column.button('Press me!')

# Or even better, call Streamlit functions inside a 'with' block:
with right_column:
    chosen = st.radio(
        'Sortig hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin")
    )
    st.write(f"You are in {chosen} house!")

# When adding long running computations to an app, we can use st.progress() to display status in real time
import time

'Starting a long computation...'

# add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
    # Update the progress bar with each iteration.
    latest_iteration.text(f"Iteration {i+1}")
    bar.progress(i + 1)
    time.sleep(0.1)

'...and now we\'re done!'