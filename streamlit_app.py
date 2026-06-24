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