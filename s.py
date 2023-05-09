import streamlit as st

# Define a function to update the session state variable with the slider value
def update_slider_value():
    if 'slider_value' not in st.session_state:
        st.session_state.slider_value = st.slider("Select a value", 0, 100, step=1)
    else:
        previous_value = st.session_state.slider_value
        st.session_state.slider_value = st.slider("Select a value", 0, 100, step=1, value=previous_value)
        delta = st.session_state.slider_value - previous_value
        st.write("Delta: ", delta)

# Call the update_slider_value function
update_slider_value()