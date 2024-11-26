#first we import boto3 and json 
import boto3, json

#create a boto3 session - stores config state and allows you to create service clients
session = boto3.Session()

#create a Bedrock Runtime Client instance - used to send API calls to AI models in Bedrock
bedrock = session.client(service_name='bedrock-runtime')

#here's our prompt telling the model what we want it to do, we can change this later
system_prompts = [{"text": "You are an app that creates reading lists for book groups."}]

#define an empty message list - to be used to pass the messages to the model
message_list = []

#here’s the message that I want to send to the model, we can change this later if we want
initial_message = {
            "role": "user",
               "content": [{"text": "Create a list of five novels suitable for a book group who are interested in classic novels."}],
               }

#the message above is appended to the message_list
message_list.append(initial_message)

#make an API call to the Bedrock Converse API, we define the model to use, the message, and inference parameters to use as well
response = bedrock.converse(
            modelId="anthropic.claude-v2",
                messages=message_list,
                    system=system_prompts,
                        inferenceConfig={
                                    "maxTokens": 2048,
                                            "temperature": 0,
                                                    "topP": 1
                                                        },
                        )

#invoke converse with all the parameters we provided above and after that, print the result 
response_message = response['output']['message']
print(json.dumps(response_message, indent=4))
