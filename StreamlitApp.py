import os 
import json 
import traceback
import pandas as pd 
from dotenv import load_dotenv
# from src.mcqGenerator.logger import logger
from src.mcqGenerator.utils import read_file,get_table_data
import streamlit as st
from langchain_community.callbacks import get_openai_callback
from src.mcqGenerator.MCQGenerator import generate_evaluate_chain




#loading json file 
with open('Response.json', 'r') as file:
    RESPONSE_JSON = json.load(file)
print(json.dumps(RESPONSE_JSON, indent=2))

#creating a title for the web app


st.title("MCQ Generator Application with Langchain ")


#create a form using st.form 

with st.form("users_input"):
    #file upload
    uploaded_file=st.file_uploader("Upload  a PDF or txt file ")

    #input fields 
    mcq_count=st.number_input("Enter the number of MCQs you want to generate",min_value=1,max_value=10)

    #subject selection
    subject=st.text_input("Enter the subject for the MCQs",max_chars=20)

    #Quiz tone
    tone =st.text_input("Complexity level of question",max_chars=20,placeholder='simple')

    #ADd Button 
    button=st.form_submit_button("Generate MCQs")


    #Check if button is clicked and all fields have input 

    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("Loading......"):
           try:
               text=read_file(uploaded_file)
               #Count tokens and the cost of API Call 
               with get_openai_callback() as cb:
                   response=generate_evaluate_chain(
                       {"text":text,"number":mcq_count,"subject":subject,"tone":tone,"response_json":json.dumps(RESPONSE_JSON, indent=2)}
                   )
                #  st.write(response)  



           except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("Error generating MCQs")
           else :
             print(f"Total Tokens:{cb.total_tokens}")
             print(f"Prompt Tokens:{cb.prompt_tokens}")
             print(f"Completion Tokens:{cb.completion_tokens}")
             print(f"Total Cost:{cb.total_cost}")
             if isinstance(response,dict):
                  #Extract the quiz data from the respnse 
                    quiz=response.get("quiz",None)
                    if quiz is not None:
                        print("quiz",quiz)
                        #convert the quiz data to a pandas dataframe
                        quiz_table_data=get_table_data(quiz)
                        if quiz_table_data is not None:
                            df=pd.DataFrame(quiz_table_data)
                            df.index=df.index+1
                            st.table(df)
                            st.text_area(label="Review",value=response['review'])
                        else:
                            st.error("Error generating MCQs")  
             else:
                 st.write(response)                


               