
import streamlit as st
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import google.generativeai as genai

# Gemini AI API-Key

api_key = "AIzaSyA-kfOR-aJ8dFSsdpH2Z6EhCC3wEJ80rHQ"

# Setting up image classification emvironment and loading the transformer with pipeline and assigned task
processor = AutoImageProcessor.from_pretrained("hareeshr/medicinal_plants_image_detection")
model = AutoModelForImageClassification.from_pretrained("hareeshr/medicinal_plants_image_detection")
pipeline = pipeline(task="image-classification", model=model, image_processor=processor)

def predict(image):

    # Passing image to the transformer via pipeline to predict the result
    predictions = pipeline(image)

    # Returning the top prediction
    return predictions[0]

def main():

    # Setting up page title
    st.title("Medicinal Plant Classification and Information")

    # Creating a form layout
    with st.form("my_form"):
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

        # Ensuring if the image is uploaded successfully or not.
        if uploaded_file is not None:

            # Display the uploaded image
            image = Image.open(uploaded_file)
            
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Making the predict button with boolean return type
        clicked = st.form_submit_button("Predict")

        # When the user clicks on 'PREDICT' button
        if clicked:
            try:
                result = predict(image)
                label = result['label']
                score = result['score'] * 100
                st.success(f"The predicted image is {label} with {score:.2f}% confidence.")
                
                # Generate information using GenerativeAI
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name="gemini-pro")

                # Crafting the most efficient prompt using prompt engineering technique
                prompt = f"Provide detailed information about the medicinal plant '{label}'. Please include its botanical name, common names, medicinal properties, traditional uses, active compounds, potential health benefits, and any known contraindications or side effects. Additionally, discuss who can benefit from its usage, such as individuals with specific health conditions or symptoms, and who should avoid it, such as pregnant or breastfeeding women, individuals with certain medical conditions, or those taking specific medications. Please provide evidence-based information and cite credible sources where applicable."
                
                # Adding a spinner to make sense while generating details
                with st.spinner(f"Generating information about {label}..."):
                    response = model.generate_content(prompt)
                    if response:
                        
                        # Writing response to the UI
                        st.write(response.text)
                    else:
                        st.error("Failed to generate information. Please try again later.")

            # Error handling                        
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
