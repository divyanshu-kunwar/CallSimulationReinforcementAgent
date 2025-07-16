from langgraph.graph import END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel , Field
from typing import Union , Literal, Optional, List
from langchain_google_genai import ChatGoogleGenerativeAI
import time 

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=10,
    api_key="API_KEY",
)

class callAnalysis(BaseModel):
    sentiment : Literal["Positive" , "Neutral" , "Negative"]
    interest_level : Literal["Interested" , "Confused" , "Disinterested"]
    intro_clarity : Literal["Clear" , "Confused"]
    objection_type : List[Literal["Cost" , "Time" , "Eligibility" , "None"]]
    objection : Optional[str] = Field(description="If any objection Describe in english What was the objection ?")
    call_outcome : Literal["Success" , "Failure" , "Follow-up-needed"]
    ignore_any_agent_question : bool

class CallSchema(BaseModel):
    general_prompt_for_agent : str
    ai_conversation : list = []
    human_conversation : list = []
    agent_name : str = "निखिल"
    call_to : str = "अमित"
    receiver : str = "अमित"
    scheme_name : str
    schema_content : str
    conversation_end : bool = False
    reply_style : str = "Neutral"
    call_analysis : Optional[callAnalysis] = None
    scheme_content_changed : Optional[str] = None
    general_base_prompt_changed : Optional[str] = None

class ReplySchema(BaseModel):
    reply : Union[str , None] = Field(description="Your reply")

def CallAgentNode(state : CallSchema):
        sys_prompt = f"""
You are a friendly and engaging conversational AI assistant named {state.agent_name} that talks with farmers.

Your primary goal is to:
1.  **Initiate the conversation (First Response ONLY):** Your *very first response* to the user must be a simple, polite inquiry to confirm if you are speaking with the correct person. You **must** use the provided `user_name`. For example: "नमस्ते! क्या मैं {state.call_to} जी से बात कर रहा हूँ?" (Hello! Am I speaking with {state.call_to} ji?) After this exact sentence, you **must end your response immediately**. Do not include any other information or questions in this initial turn.

2.  **Handle Initial Confirmation & Availability:**
    * If the user confirms they are {state.call_to}, proceed to introduce the scheme.
    * If the user indicates it's *not* {state.call_to}, politely ask if {state.call_to} is available. For example: "क्या मैं {state.call_to} से बात कर सकता हूँ?" (Excuse me. Is {state.call_to} available right now?)
    * If {state.call_to} becomes available, or if the current speaker indicates they can tell about this to {state.call_to} instead, then proceed to introduce the scheme.

After this you need to take care of following rules:
{state.general_prompt_for_agent}

* To end the conversation you must set the reply with None.

Here is a brief document to help you out with the conversation about {state.scheme_name}:
----
{state.schema_content}
----
"""
        conversation_history = [
                SystemMessage(sys_prompt)
        ]
        
        conversation_history.extend(state.ai_conversation)

        callAgentLLM = llm.with_structured_output(ReplySchema)
        time.sleep(10)
        response = callAgentLLM.invoke(conversation_history)

        if(response.reply == None):
              return {"conversation_end" : True}
        else:
              print("AI : ", response.reply)
              ai_conversation = state.ai_conversation + [AIMessage(content=response.reply)]
              human_conversation = state.human_conversation + [HumanMessage(content=response.reply)]
              return {"ai_conversation" :  ai_conversation,
                      "human_conversation" : human_conversation}

def humanAgentNode(state : CallSchema):
        sys_prompt = f"""
You are a farmer named {state.receiver}. 
Your task is to talk to the person over the call.
You must respond in Hindi language, and you must not use any numerics; rather, respond with the Hindi word for the numeral.
Keep the conversation flowing naturally.
You can say words like haan haan or thik hai etc to show interest . Or you can say rehne do , mt batao dobara mt call krna , rkh , rkho etc to show disinterest.
You can also act sometimes as if you are not hearding the conversation by using words like samajh nahi aaya, kya bol rahe ho etc.
You can also ask about price or say you want free. 
You must keep everything in devanagri script no other script should be in the response.
You may also ignore some user question and respond that you are not sure.
NOTE : For this conversation you should respond in {state.reply_style} sense.
* To end the conversation you must set the reply with None.
Do not make conversation very long.
"""
        conversation_history = [
                SystemMessage(sys_prompt)
        ]

        conversation_history.extend(state.human_conversation)
        callAgentLLM = llm.with_structured_output(ReplySchema)
        response = callAgentLLM.invoke(conversation_history)

        print("Human : ", response.reply)
        if(response.reply == None):
                return {"conversation_end" : True}
        else:
                ai_conversation = state.ai_conversation + [HumanMessage(content=response.reply)]
                human_conversation = state.human_conversation + [AIMessage(content=response.reply)]
                return {"ai_conversation" :  ai_conversation,
                        "human_conversation" : human_conversation}

def isEndCall(state : CallSchema):
    if state.conversation_end:
        return ["Call_Analysis_Agent"]
    else:
        return ["Human_Receiver"]

def isEndCall2(state : CallSchema):
    if state.conversation_end:
        return ["Call_Analysis_Agent"]
    else:
        return ["Call_Agent"]

def callAnalysisAgent(state : CallSchema):
    sys_prompt = f"""
You are a call analyst you have to analyse the call between the farmer and the call agent.

You need to evaluate the conversation for following aspects. 
1. Farmer Sentiment	- Detect the emotion or tone in each response.
    Valid responses are Positive, Negative, or Neutral.
    some hint to guess Positive: “haan haan”, Neutral: “theek hai”, Negative: “mat batana” etc. The conversation could contain other similar words that may help you guess.
2. Farmer Interest Level - Are they interested, confused, or disinterested?	. 
    Valid responses are Interested, Confused, or Disinterested
    Some hint to guess Keywords like “batao”, “dobara call karna”, “nahi chahiye” etc.
3. Intro Clarity - Did the farmer understand the opening pitch?	- Farmer says “samajh nahi aaya”, asks “kya bol rahe ho?” etc.
    Valid responses are Clear, Confused,
4. Objections / Concerns - Farmer raises cost/time/eligibility issues “Kitne ka hai?”, “Mujhe free chahiye”, “Main eligible hoon kya?” etc.
    Valid response for objection_type are Cost, Time, Eligibility, None
    Optionally valid response for objection is any string only if objection_type is Cost, Time or Eligibility
5. Call Outcome - Was it a success, failure, or follow-up needed? - Success = interested, Failure = rejection, Follow-up = “call later”
    Valid responses are Success, Failure, Follow-up-needed
6. Does the farmer ignore any question asked by agent ?
    Valid responses are True or False
"""
    
    conversation = ""
    for msg in state.ai_conversation:
        if(type(msg)== AIMessage):
            conversation += "\nAgent : " + msg.content
        if(type(msg)==HumanMessage):
            conversation += "\nFarmer : " + msg.content

    conversation_history = [
        SystemMessage(sys_prompt),
        HumanMessage(conversation)
    ]

    callAnalysisLLM = llm.with_structured_output(callAnalysis)
    result = callAnalysisLLM.invoke(conversation_history)
    print("\n\n ------------------- Analysis Result ----------------- ")
    print(result)
    return {"call_analysis" : result}

class DocumentScheme(BaseModel):
    content : str = Field(description="The content of the scheme")

class instructionScheme(BaseModel):
    instructions : str = Field(description="The modified instruction for the agent")

def ReinforcementAgent(state : CallSchema):
    scheme_content_changed = None
    general_base_prompt_changed = None

    analysis = state.call_analysis
    
    # Intro Simplification
    if analysis.intro_clarity == "Confused":
        sys_prompt = f""" 
You are a writer for {state.scheme} . You have written the following doc previously :
---
{state.scheme_content}
---
Based on this document our agent tried to talk to the farmer but farmer seemed to be confused during the intro. 
Simplify the document in such a way that it becomes easier for farmers to catch up.
Do not miss any existing information from the document in the process rather just simplify the words without changing the context.
"""
        conversation_history = [HumanMessage(sys_prompt)]
        introSimplificationAgent = llm.with_structured_output(DocumentScheme)
        result = introSimplificationAgent.invoke(conversation_history)
        print("Modified content of doc : \n" , result.content)
        scheme_content_changed = result.content

    additional_instruction = ""
    # Tone Softening
    if analysis.sentiment == "Negative" or analysis.sentiment == "Neutral":
        # change the base prompt to make the tone soften 
        additional_instruction += f"""
but farmer responded with a {analysis.sentiment} sentiment. 
change instruction prompt so that AI respond with more polite and soft tone.
"""

    # FAQ Clarity
    if analysis.interest_level == "Confused" or analysis.call_outcome == "Follow-up-needed":
        additional_instruction += f"""It seems that a follow up is needed . 
Add remove or change instruction prompt so that AI respond in a more engaging and personalized style for the user. 
"""


    # CTA Reframing
    if analysis.ignore_any_agent_question == True:
        add_example_question_to_prompt = True
        # change the base prompt to contain some simple FAQs questions 
        additional_instruction += f"""
It seems that the questions asked to the user was not interesting or doesn't contain value to him.
Add some same example question to the prompt that should be asked so that farmer will be interested in talking to the AI based on the below document.
{state.schema_content}
"""
        
    conversation = ""
    for msg in state.ai_conversation:
        if(type(msg)== AIMessage):
            conversation += "\nAgent : " + msg.content
        if(type(msg)==HumanMessage):
            conversation += "\nFarmer : " + msg.content

    sys_prompt = f"""
You are a prompt writer for a calling agent. This is the instruction prompt you have given to the model:
---
{state.general_prompt_for_agent}
---
Based on these instruction our AI agent tried to talk to the farmer. 
Given below is the conversation between agent and farmer.
---
{conversation}
---
{additional_instruction}
Analyze what went wrong and then try to modify the instruction prompt given to the model.
Respond with only the instructions.
"""
    conversation_history = [HumanMessage(sys_prompt)]
    promptModificationLLM = llm.with_structured_output(instructionScheme)
    result = promptModificationLLM.invoke(conversation_history)
    print("Modified Instruction : \n" , result.instructions)
    general_base_prompt_changed = result.instructions

    return {
        "general_base_prompt_changed" : general_base_prompt_changed,
        "scheme_content_changed" : scheme_content_changed,
    }





