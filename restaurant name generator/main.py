import streamlit as st
import Langchain_helper

st.title("Restaurant Name Generator")

cuisine = st.sidebar.selectbox("Pick a Cuisine", ("Indian", "Italian", "Mexican", "Chinese", "Korean", "Arabian"))

if cuisine:
     response = Langchain_helper.generate_restaurant_name_and_items(cuisine)
     st.header(response['restaurant_name'].strip())
     menu_items = response['menu_items'].strip().split(",")
     st.write("**Menu Items**")
     for item in menu_items:
         st.write("-", item)
