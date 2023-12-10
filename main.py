from st_class import ImageProcessorApp
import streamlit as st

if __name__ == '__main__':
    st.set_page_config(layout='wide')
    app = ImageProcessorApp()
    app.run()
