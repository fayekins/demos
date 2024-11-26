import boto3, json

session = boto3.Session()
bedrock = session.client(service_name='bedrock-runtime')

system_prompts = [{"text": "You are an app that creates reading lists for book groups."}]
message_list = []

initial_message = {
            "role": "user",
               "content": [{"text": "Create a list of five novels suitable for a book group who are interested in classic novels."}],
               }
               
message_list.append(initial_message)
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

response_message = response['output']['message']
print(json.dumps(response_message, indent=4))
