import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(".env")

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is missing. Please set it in the .env file.")
genai.configure(api_key=google_api_key)

def make_prompt1(system_instructions, user_context, user_question, example_context1, example_question1, example_answer1, example_context2, example_question2, example_answer2):
    prompt = f"""System: {system_instructions}
    Example Interaction 1:
    Provided Context : {example_context1}
    User: {example_question1}
    Assistant: {example_answer1}
    Example Interaction 2:
    Provided Context : {example_context2}
    User: {example_question2}
    Assistant: {example_answer2}
    Current Interaction:
    Provided Context : {user_context}
    User: {user_question}
    Assistant:"""
    return prompt   

def make_prompt2(system_instructions, user_question, example_question1, example_answer1, example_question2 =None, example_answer2=None):
    if example_question2 and example_answer2:
        prompt = f"""System: {system_instructions}
        Example Interaction 1:
        User: {example_question1}
        Assistant: {example_answer1}
        Example Interaction 2:
        User: {example_question2}
        Assistant: {example_answer2}
        Current Interaction:
        User: {user_question}
        Assistant:"""
    else:
        prompt = f"""System: {system_instructions}
        Example Interaction:
        User: {example_question1}
        Assistant: {example_answer1}
        Current Interaction:
        User: {user_question}
        Assistant:"""
    return prompt  
 
def send_request(prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-8b')
        
        response = model.generate_content(prompt)
        answer = response.text
        
        answer = (
            answer.replace('Response:', '')
                  .replace('->item', '')
        )
    except Exception as e:
        print(f"Error generating content: {e}")
        answer = 'NA'
    return answer
