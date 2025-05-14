import gradio as gr
import openai
from dotenv import load_dotenv
import os

# Tải API Key từ file .env
load_dotenv()
print("API Key:", os.getenv("GROQ_API_KEY"))
# Khởi tạo client Open AI với base URL của Groq
client = openai.OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Hàm gọi API Groq bằng SDK Open AI
def chat_with_bot(message, history):
    try:
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "Bạn là một trợ lý AI hữu ích."},
                {"role": "user", "content": message}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Lỗi: {str(e)}"

# Tạo giao diện Gradio
demo = gr.ChatInterface(
    fn=chat_with_bot,
    title="Chat Bot LLM (Groq via Open AI SDK)",
    description="Hỏi bất kỳ câu hỏi nào, bot sẽ trả lời!",
    theme="soft",
    examples=["Xin chào, bạn khỏe không?", "Hôm nay là ngày bao nhiêu?"]
)

if __name__ == "__main__":
    demo.launch(share=True)