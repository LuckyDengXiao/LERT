
import random
from openai import OpenAI
import httpx
import json
client = OpenAI(
    # This is the default and can be omitted
    base_url = "Your API",
    api_key = "Your KEY"
    http_client=httpx.Client(
        base_url = "Your API",
        follow_redirects=True,
    ),
)

def getResponseFromMessage(message):
    #prompting=message
    while True:
        attempt_time=0
        try:
            output = client.chat.completions.create(
            model="gpt-4-turbo", 
            #prompt=prompting,
            max_tokens=2048,
            temperature=1,
            #messages=[{"role": "user", "content":prompting}]
            messages=message
            )
            return output.choices[0].message.content
        except:
            attempt_time=attempt_time+1
            print("the %dth try"%(attempt_time))
            continue

with open("/tf/LineVul-main/bigvul_generatedData.json","r",encoding="utf-8") as fin:
    data=json.load(fin)
data_out={}
for i in data:
    data_out[i]=[]
    for j in data[i]:
        context_dict={}
        context_dict["first_iter"]=j
        #messages=[{"role": "user", "content":prompting}]
        prompting="Here is a function written in C or C++. The code is: {given_function} . \
        You need to give a score to evaluate this function on its relation to the CWE-id: {cwe_id} , complexity, length, and diversity of variable naming.\
        Please give me only one number about the quality of this function among 0-10."\
        .format(given_function=j,cwe_id=i)
        messages=[{"role": "user", "content":prompting}]
        prompting="You are an AI model that provides feedback on the quality of functions."
        messages.append({"role": "system", "content":prompting})
        prompting="You are an AI model that evaluates whether a function exemplifies a certain CWE type."
        messages.append({"role": "system", "content":prompting})
        prompting="The response content from you limit to only one number"
        messages.append({"role": "system", "content":prompting})
        #print(messages)
        score=getResponseFromMessage(messages)
        print("success getData with cwe_id:%s, the responce is %s" %(i,score))
        context_dict["second_iter"]=score
        data_out[i].append(context_dict)
        
prompting="Here is a function written in C or C++,scored {score}. The code is: {given_function} . \
        Now I need you to refine this period of code to match the exemplifies a certain CWE type {cwe_id} \
        better than given code."\
        .format(score=self.score,given_function=j,cwe_id=i)
messages=[{"role": "user", "content":prompting}]
prompting="You are an AI model that provides the function after refine."
messages.append({"role": "system", "content":prompting})
prompting="The score I provided is to evaluates whether a function exemplifies a certain CWE type."
messages.append({"role": "system", "content":prompting})
prompting="The response content from you limit to only a programming function written in C/C++."
messages.append({"role": "system", "content":prompting})
