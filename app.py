import streamlit as st
import numpy as np
import pandas as pd
# from skin_lesion_detection.data import get_data


st.markdown("""# Skin Lesion Detection Neural Network
Upload your photo to identify the type of skin lesion that you have
##""")

st.markdown('''### Image dataset
''')
# df = get_data()
# df

if st.button('Upload photo'):
    # print is visible in server output, not in the page
    print('Photo uploading!')
    st.write('Photo uploaded!')


if st.button('Diagnose'):
    # print is visible in server output, not in the page
    print('Runnign diagnosis!')
    st.write('Runnign diagnosis!')
