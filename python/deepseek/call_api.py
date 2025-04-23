from openai import OpenAI

client = OpenAI(api_key="sk-1d20bf12e914445ca9faeda156f85027", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "what model are you"},
    ],
    stream=False
)

print(response.choices[0].message.content)