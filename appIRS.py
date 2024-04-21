import streamlit as st
from PIL import Image
import numpy as np
import result
import model



# Define function to search by text query
def search_by_text(query):
    # Your logic to search images by text query
    # Return relevant images
    return []

# Define function to search by uploading image
def search_by_image(uploaded_image):
    # Your logic to process uploaded image and find similar images
    # Return similar images
    return []

# Main Streamlit app
def main():   



    st.title("Photograph Library")
    st.write("Search for photographs")

    # Sidebar
    search_option = st.sidebar.selectbox("Search option", ["Text query", "Upload image"])

    if search_option == "Text query":
        query = st.text_input("Enter search query:")
        text_embedding = model.input_text_embedding(query)
        resulting_imgs = result.search_similar_vectors(result.database,text_embedding,10)
        print(resulting_imgs)
        if st.button("Search"):
            results = search_by_text(query)
            # Display search results
            image_paths = []
            for result1 in resulting_imgs['Name']:
                image_paths.append(f'{result1}')
            print(image_paths)
            # for result1 in image_paths:
            #     st.image(result1, caption='Similar Image', use_column_width=True)
            cols = st.columns(3) # Adjust the number of columns as desired

            for index, path in enumerate(image_paths):
                cols[index % 3].image(path) 
    
    elif search_option == "Upload image":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        if uploaded_image is not None:
            img = Image.open(uploaded_image)
            st.write("This is the Input Image")
            col1, col2, col3 = st.columns(3)  
            with col1, col2, col3:
                col2.image(img, width=300)
            
            img_embedding = model.input_img_embedding(img)
            resulting_imgs = result.search_similar_vectors(result.database,img_embedding,10)
            # st.image(img, caption='Uploaded Image', use_column_width=True)
            if st.button("Search similar"):
                results = search_by_image(img)
                print(resulting_imgs)
                image_paths = []
                for result1 in resulting_imgs['Name']:
                    image_paths.append(f'{result1}')
                print(image_paths)
                # for result1 in image_paths:
                #     st.image(result1, caption='Similar Image', use_column_width=True)
                cols = st.columns(3) # Adjust the number of columns as desired

                for index, path in enumerate(image_paths):
                    cols[index % 3].image(path) 

if __name__ == "__main__":
    main()
