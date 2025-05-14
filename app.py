import gradio as gr
import openai
from dotenv import load_dotenv
import os
import re
import requests
from bs4 import BeautifulSoup
import pdfplumber

load_dotenv()

client = openai.OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


def get_article_content(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = " ".join([para.get_text() for para in paragraphs])
        return article_text
    except Exception as e:
        return f"Lỗi khi lấy nội dung bài viết: {str(e)}"

def extract_pdf_content(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            pdf_text = ""
            for page in pdf.pages:
                pdf_text += page.extract_text()
        return pdf_text
    except Exception as e:
        return f"Lỗi khi trích xuất nội dung từ PDF: {str(e)}"

def format_history_to_messages(history):
    messages = [{"role": "system", "content": "Bạn là một trợ lý AI hữu ích."}]
    for user, bot in history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": bot})
    return messages

def chat_with_bot(message, history, pdf_text=None):
    try:
        url_pattern = re.compile(r'https?://\S+')
        urls = url_pattern.findall(message)
        memory_context = []
        if urls and message.strip():
            url = urls[0]
            article = get_article_content(url)
            memory_context.append(["Nội dung bài viết", article])
            user_message = message
        elif urls and not message.strip():
            url = urls[0]
            article = get_article_content(url)
            memory_context.append(["Nội dung bài viết", article])
            user_message = f"Tóm tắt nội dung bài viết:\n\n{article}"
        elif pdf_text and message.strip(): 
            memory_context.append(["Nội dung PDF", pdf_text]) 
            user_message = message
        elif pdf_text and not message.strip():
            memory_context.append(["Nội dung PDF", pdf_text]) 
            user_message = f"Tóm tắt nội dung PDF:\n\n{pdf_text}"
        else:
            user_message = message

        messages = format_history_to_messages(memory_context + history)
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            max_tokens=1000,
            temperature=0.5
        )

        bot_reply = response.choices[0].message.content
        history.append([message, bot_reply])
        return history
    except Exception as e:
        history.append([message, f"Lỗi: {str(e)}"])
        return history

with gr.Blocks(css="""
* {
    font-family: 'Segoe UI', sans-serif;
}
#centered {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
}
.gr-chatbot {
    border: 1px solid #ccc;
    border-radius: 14px;
    padding: 10px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    width: 100%;
    max-width: 700px;
    background-color: #f8f8f8;
}
textarea {
    border: 1px solid #CCCCCC !important;
    border-radius: 10px !important;
    padding: 12px !important;
    font-size: 16px;
    width: 100% !important;
    min-height: 68.8px !important;
    max-height: 150px !important;
    resize: vertical !important;
    overflow-y: auto;
    background-color: white !important;
    color: black !important;
}
button {
    background-color: #CCCCCC !important;
    color: black !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    font-weight: bold;
}
button:hover {
    background-color: #CCCCCC !important;
}
.file-upload {
    margin-top: 10px;
}
#upload-btn, #send-btn {
    width: 150px;    
    height: 68.8px;    
    font-size: 16px;  
    border-radius: 10px;  
}
""", theme=gr.themes.Soft()) as demo:
    
    with gr.Column(elem_id="centered"):
        gr.Markdown("""
            # 🤖 **Chatbot Thông Minh**
            ### <i>Trò chuyện, tìm kiếm câu trả lời và giải quyết vấn đề thông minh</i>
        """)

        chatbot = gr.Chatbot(
            label="Trò chuyện với Bot",
            height=450,
            show_label=False           
        )

        with gr.Row():
            txt = gr.Textbox(
                show_label=False,
                placeholder="Nhập nội dung để trò chuyện...",
                container=False,
                lines=1,
                elem_id="custom-textbox"
            )
            visible = False
            upload_btn = gr.Button("📎 Upload file PDF", scale=0, visible=(not visible) , elem_id="upload-btn")

            send_btn = gr.Button("🚀 Gửi", scale=0, elem_id="send-btn")
            
        pdf_upload = gr.File(
        file_types=[".pdf"],
        type="filepath",
        visible=False
        )

        pdf_visible_state = gr.State(False)

        def toggle_pdf_upload(visible):
            return gr.update(visible=not visible), not visible


        def respond(message, history, pdf_file):
            pdf_text = None
            if pdf_file:
                pdf_text = extract_pdf_content(pdf_file.name)
            updated_history = chat_with_bot(message, history, pdf_text)
            return "", updated_history

        txt.submit(respond, [txt, chatbot, pdf_upload], [txt, chatbot])
        send_btn.click(respond, [txt, chatbot, pdf_upload], [txt, chatbot])
        upload_btn.click(toggle_pdf_upload, pdf_visible_state, [pdf_upload, pdf_visible_state])
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)