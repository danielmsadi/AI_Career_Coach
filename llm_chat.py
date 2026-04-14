# Import necessary packages
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import Model, ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
import gradio as gr

# credentials 
credentials = Credentials(
      url = "https://us-south.ml.cloud.ibm.com",)
                  

# Model and project settings
model_id = "meta-llama/llama-3-2-11b-vision-instruct"  
project_id = "skills-network"
params = TextChatParameters(
    temperature=0.1,
    max_tokens=512
)

# Initialize the model
model = ModelInference(
    model_id=model_id,
    credentials=credentials,
    project_id=project_id,
    params=params
)

# Function to generate a response from the model
def generate_response(prompt_txt):
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt_txt
            },
        ]
    }
]   
    generated_response = model.chat(messages=messages)
    generated_text = generated_response['choices'][0]['message']['content']

    return generated_text

#Gradio interface
chat_application = gr.Interface(
    fn=generate_response,
    flagging_mode="never",
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output"),
    title="Dan Chatbot",
    description="Ask any question and the chatbot will try to answer."
)

# Launch the app
chat_application.launch()
