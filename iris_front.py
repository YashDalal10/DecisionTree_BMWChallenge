import streamlit as st
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load the iris dataset
iris = load_iris()

# Create a decision tree classifier
dtc = DecisionTreeClassifier()

# Train the decision tree classifier
dtc.fit(iris.data, iris.target)

# Define the Streamlit app
def app():
    # Set the app title
    st.title("Decision Tree Output")

    # Create a sidebar with the input form
    with st.sidebar:
        st.header("Input Form")
        sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.0)
        sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.0)
        petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=1.0)
        petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=0.5)

    # Create a button to classify the input
    if st.button("Classify"):
        # Create a dictionary with the input data
        input_data = {
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        }

        # Convert the input data to a numpy array
        input_array = [[input_data["sepal_length"], input_data["sepal_width"], input_data["petal_length"], input_data["petal_width"]]]

        # Make a prediction using the decision tree classifier
        prediction = dtc.predict(input_array)[0]

        # Display the prediction
        st.write(f"The predicted iris species is: {iris.target_names[prediction]}")

    # Display the decision tree plot
    st.header("Decision Tree Plot")
    plot_tree(dtc)
