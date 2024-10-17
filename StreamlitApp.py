import streamlit as st
from QAWithPDF.data_ingestion import load_data
from QAWithPDF.embedding import download_gemini_embedding
from QAWithPDF.model_api import load_model
from exception import customexception  # Import your custom exception

def main():
    st.set_page_config("QA with Documents")
    
    doc = st.file_uploader("Upload your document")
    
    st.header("QA with Documents (Information Retrieval)")
    
    user_question = st.text_input("Ask your question")
    
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            try:
                if doc is None:
                    st.error("Please upload a document.")
                    return

                document = load_data(doc)
                model = load_model()
                query_engine = download_gemini_embedding(model, document)
                
                response = query_engine.query(user_question)
                st.write(response.response)

            except customexception as e:
                st.error(str(e))  # Display custom exception message
            except Exception as e:
                st.error("An unexpected error occurred: {}".format(str(e)))  # Handle other exceptions

if __name__ == "__main__":
    main()
