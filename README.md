##TODO: put the parsing of the output of the generator agent into the review agent class. That way, the review agent can read the raw output of the generator agent AND do the parsing (which is what it should do as the agent anyway). Then also, the output of the generate_completion() function for the multi_agent class can be the raw output.

Also: use ThreadPoolExecutor to work on multiple samples concurrently.

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def call_lambda_inference_api(payload):
    url = "https://api.lambdalabs.com/inference"  # Replace with the actual endpoint
    headers = {
        "Authorization": "Bearer YOUR_API_KEY",
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()  # Raises an error for bad responses
    return response.json()

# List of payloads you want to send in parallel
payloads = [
    {"input": "data1"},
    {"input": "data2"},
    # add more payload dictionaries as needed
]

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(call_lambda_inference_api, payload): payload for payload in payloads}
    for future in as_completed(futures):
        payload = futures[future]
        try:
            result = future.result()
            print("Result for payload", payload, ":", result)
        except Exception as exc:
            print("Payload", payload, "generated an exception:", exc)
