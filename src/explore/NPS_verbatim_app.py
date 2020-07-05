import streamlit as st
import pandas as pd
import os, urllib
# import call_model_predictions
import torch
import os
import json

from pathlib import Path

# Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_web_file_content_as_string(url):
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

def get_local_file_content_as_string(path):
    with open(path, 'r') as file:
        content = file.read()

    return content

class AnnotationPage(object):
    def __init__(self):
        self.instructions = st.markdown("""
        # Annotate your data using Prodigy.ai

        To annotate your data with pre-defined categories, or discover the aspect terms on the go we use the tool Prodigy. Prodigy is a scriptable tool to enable rapid iteration on your data enrichment efforts, and really 
        quickly annotate the data we'll need to get your customized state-of-art NLP model.
        """)

    def file_uploader(self):
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'jsonl'])
        df = pd.DataFrame()
        
        st.subheader("Show Dataset")
        number = st.number_input("Number of rows to view", step=1, min_value=1, max_value=1000, value=10)
        st.dataframe(df.head(number))



class InstructionsPage(object):
    def __init__(self):
        # Initialize the parts of the instructions page
        self.part_1 = st.empty()
        self.image_cat = st.empty()
        self.part_2 = st.empty()
        self.image_term = st.empty()
        self.part_3 = st.empty()


    def show_instructions(self):
        self.part_1.markdown("""
        ## Basics

        The goal of this app is to allow you to categorize texts (e.g. NPS verbatims and reviews) and associate a sentiment score to each category. 

        The categorization of text can happen in two ways: 
        - predefined categories
        - explicitly mentioned terms

        The two methods of categorization are best explained in the following two examples. Let's take a look at a review for a restaurant left by a customer.
        """)

        self.image_cat.image('assets/example_categorization.png', use_column_width=True, caption="Example of categorizing texts")

        self.part_2.markdown(
        """     
        Now above, we can see a list of available categories to the machine learning model [food, service, price, ambience, anecdote/misc]. These have been predefined for the model, and could be changed if required. In this case we pick out two clear classifications of aspects that can be found in the example. Using many examples in an annotated dataset (see chapter 2) the model is able to detect how words and phrases should be linked to the predefined aspects (a.k.a. categories). 
        The next step for the model is to identify what dependency words are linked to that particular word or phrase. This allows us to set a sentiment score to the category. Here we have basically defined that to the business the word _waiter_ implicitly is directed at the service, and _music_ is implicitly directed at the ambience. This leads us to the next method of defining categories.
        """)

        self.image_term.image("assets/example_aspectterms.png", use_column_width=True, caption="Example of aspect term")

        self.part_3.markdown(
        """
        In this case we don't want or have the predefined categories. we want to discover explicitly mentioned aspect terms, such as _waiter_ and _music_. And go ahead and assign the sentiment to those specific terms, instead of to the broader aspect categories.

        ## Annotating datasets

        👈 **Please select _Annotate raw data_ in the _steps_ in the sidebar to start.**

        This model uses state-of-art techniques within Natural Language Processing (NLP). NLP is a type of machine learning that allows for the analysis of unstructured textual data. There are many applications for NLP, such as: opinion mining and text classification. 
        This tool combines both of the previously mentioned examples in one model, and in order to do this the model needs training examples. 

        Here you need to make a decision on what kind of model you are going to need:

        - Sentiment associated with pre-defined categories (Aspect-Category Sentiment Classification)
        - Sentiment associated with explicitly mentioned aspect terms (Aspect-Term Sentiment Classification)

        In both cases we need your raw data containing at least: raw text, id (e.g. respondent id, or reviewer), and source (can be a specific restaurant, customer journey, etc). Optionally you should also include any variables you want to be able to filter on.

        ### Aspect-Category Sentiment Classification

        The first step to training a model using your pre-defined     

        > Guideline: at least 50 examples per category/term. Ideally 50 examples per category/term & sentiment combination.

        ## Training a model

        👈 **Please select _Train Models_ in the _steps_ in the sidebar to start.**

        ## Trying a model

        👈 **Please select _Run Models_ in the _steps_ in the sidebar to start.**


        ### Questions? Comments?

        Please ask in the [Streamlit community](https://discuss.streamlit.io).
        """
        )
    
    def hide_instructions(self):
        self.part_1.empty()
        self.part_2.empty()
        self.part_3.empty()
        self.image_cat.empty()
        self.image_term.empty()


def main():
    # Render the readme as markdown
    readme_text = st.markdown(get_local_file_content_as_string("instructions.md"))
    instructions = InstructionsPage()

    # Once we have the dependencies, add a selector for the app mode on the sidebar
    st.sidebar.title("Select a step in the process")
    app_mode = st.sidebar.selectbox("Choose the step", [
        "Show instructions", "Model Overview", "Annotate raw data", "Train New Models", "Run Existing Models"
    ])

    instructions.show_instructions()

    if app_mode == "Show instructions":
        st.sidebar.success("To continue select another step")
    elif app_mode == "Annotate raw data":
        readme_text.empty()
        instructions.hide_instructions()
        annotate_data()
    elif app_mode == "Model Overview":
        readme_text.empty()
        instructions.hide_instructions()
        model_overview()
        pass
    elif app_mode == "Train New Models":
        readme_text.empty()
        instructions.hide_instructions()
        pass
    elif app_mode == "Run Existing Models":
        readme_text.empty()
        instructions.hide_instructions()
        run_models()
        pass

def model_overview():
    st.title("Model Overview")

    st.write("Aspect-Target Sentiment Classification (ATSC) is a subtask of Aspect-Based Sentiment Analysis (ABSA),  \
        which has many applications e.g. in e-commerce, where data and insights from reviews can be leveraged to create  \
        value for businesses and customers. Recently, deep transfer-learning methods have been applied successfully to a \
        myriad of Natural Language Processing (NLP) tasks, including ATSC. Building on top of the prominent BERT language \
        model, we approach ATSC using a two-step procedure: self-supervised domain-specific BERT language model \
        finetuning, followed by supervised task-specific finetuning. Our findings on how to best exploit domain-specific \
        language model finetuning enable us to produce new state-of-the-art performance on the SemEval 2014 Task 4 \
        restaurants dataset. In addition, to explore the real-world robustness of our models, we perform cross-domain \
        evaluation. We show that a cross-domain adapted BERT language model performs significantly better than strong \
        baseline models like vanilla BERT-base and XLNet-base. Finally, we conduct a case study to interpret model \
        prediction errors."
    )

    st.subheader("Bain Practice Area Language Models")
    st.write("We can build more advanced models by training")

def annotate_data():
    st.title("Annotate raw data")
    st.markdown("""
    The purpose of annotation is to train a machine learning model by supervising the learning process. You know the correct answers. Therefore annotating a dataset in this case with either predefined aspect categories:
    """)

    st.image("assets/example_categorization.png", use_column_width=True, caption="Example of aspect category")

    st.markdown("""
    or by explicitly mentioned aspect terms:
    """)

    st.image("assets/example_aspectterms.png", use_column_width=True, caption="Example of aspect term")

    st.markdown("""Here you have to make the decision between either of the two, since reusing your annotations for the other type of ABSA isn't possible. 
    The decision between the two is simple:
    
    * Do you want control regarding the categories that should be discovered from your data? Choose Aspect Category.
    * Do you want to discover what 
    """)



def run_models():
    st.header("Run models")

    p = Path('../../data/models/')
    subdirectories = [str(x).split('/')[4] for x in p.iterdir() if x.is_dir()]

    option = st.selectbox('Choose an existing model', subdirectories)
    st.write('You selected:', option)

    # def load_model(model_name):
    #     """Placeholder function"""
    #     df_training_data = pd.DataFrame()
    #     if model_name == 'NPS_Prism':
    #         df_training_data = pd.read_json('../data/raw/bain.nosync/annotations_gold_revised_v2.jsonl',lines=True)
    #     elif model_name == 'BERT_banking_insurance__banking':
    #         df_training_data = pd.read_json('../data/raw/bain.nosync/annotations_gold_revised_v2.jsonl',lines=True)
    #     elif model_name == 'BERT_insurance':
    #         df_training_data = pd.read_json('../data/raw/bain.nosync/insurance_annotation_data_filtered.jsonl',lines=True)

    #     st.write(df_training_data)

    #     path_to_model = "data/models/" + model_name

    #     with open(path_to_model + "/config.json") as f:
    #         config = json.load(f)

    #     config_class, model_class, tokenizer_class = call_model_predictions.MODEL_CLASSES[config["model_type"]]
    #     model = model_class.from_pretrained(path_to_model)
    #     tokenizer = tokenizer_class.from_pretrained(path_to_model)
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     model.to(device)

    #     return model, tokenizer
        
    st.text("Sample of training data used for this model")
    # model, tokenizer = load_model(option)

    st.header('Input new data to the model')
    st.subheader('Upload data')
    st.file_uploader(label='Upload a CSV-file containing your verbatims')
    st.subheader('OR')
    custom_verbatim = st.text_input('Write your custom verbatim here')

    # if st.button('Test verbatim'):
    #     # call_model_predictions.predict()


     
if __name__ == "__main__":
    main()