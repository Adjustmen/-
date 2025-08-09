import logging
from typing import Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """
    一个简单的知识库类，用于存储和检索文档
    """
    def __init__(self, embedding_model, embedding_model_name: str, chunk_size: int, chunk_overlap: int):
        self.embedding_model = embedding_model
        
        self.vector_store = FAISS.from_texts(
            texts=["初始化知识库"],
            embedding=self.embedding_model
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        self.knowledge_items = []

    def add_texts(self, texts: List[str]):
        """
        向知识库添加文本
        """
        chunks = self.text_splitter.split_text("\n".join(texts))
        if not chunks:
            return
        
        # 将新知识添加到FAISS向量库
        new_vector_store = FAISS.from_texts(chunks, self.embedding_model)
        self.vector_store.merge_from(new_vector_store)
        self.knowledge_items.extend(chunks)
        logger.info(f"添加了 {len(chunks)} 条知识块")

    def add_knowledge_from_file(self, file_path: str):
        """
        从文件加载知识
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        self.add_texts([text])
        
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """
        从知识库中检索最相关的文档
        """
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

class ChatBot:
    """
    智能对话机器人核心类
    """
    def __init__(self, embedding_model, model_name: str, device: str = 'auto', load_in_4bit: bool = False, local_files_only: bool = False):
       # ...existing code...
        self.embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model)
# ...existing code...
        self.model_name = model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        
        # 根据main.py的配置，加载embedding模型和知识库
        # 这里需要从config中获取embedding模型名称，为了简洁，我们写死
        embedding_model_name = "root/conversation/local_model"
        # ...existing code...
        self.knowledge_base = KnowledgeBase(
            embedding_model=self.embedding_model,           # 传递已初始化的 embedding_model
            embedding_model_name=embedding_model,           # 如果需要模型名也可以传
            chunk_size=500,
            chunk_overlap=50
        )
# ...existing code...
        logger.info("初始化对话模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device,
                local_files_only=local_files_only
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=local_files_only)
            if device != 'cpu' and torch.cuda.is_available():
                self.model.to(device)
        
        logger.info("对话模型加载完成.")

    def add_knowledge_from_file(self, file_path: str):
        self.knowledge_base.add_knowledge_from_file(file_path)

    def generate_response(self, user_input: str, history: List[str] = [], system_type: str = 'default') -> str:
        """
        生成回复
        """
        logger.info(f"用户输入: {user_input}")
        
        # 检索知识库
        retrieved_docs = self.knowledge_base.retrieve(user_input, k=2)
        context = " ".join(retrieved_docs)
# ...existing code...
# 修正前
# prompt = f"上下文: {context}\n历史对话: {' '.join(history)}\n用户: {user_input}\n机器人:"

# 修正后
        history_str = []
        for item in history:
            if isinstance(item, list):
                history_str.append(" ".join(str(x) for x in item))
            else:
                history_str.append(str(item))
        prompt = f"上下文: {context}\n历史对话: {' | '.join(history_str)}\n用户: {user_input}\n机器人:"
# ...existing code...
        # 构建输入提示
        #prompt = f"上下文: {context}\n历史对话: {' '.join(history)}\n用户: {user_input}\n机器人:"
        
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        
        # 生成回复
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 简单地去除输入prompt
        response = response.replace(prompt, "", 1).strip()
        # chatbot_system.py 伪代码
        logger.info(f"模型回复: {response}")
        return response
    def reply(self, user_input):
        answer = self.knowledge_base.retrieve(user_input)
        if answer:
            return answer
        else:
        # 如果知识库没有，调用生成模型
            model_reply = self.model_generate(user_input)
        if not model_reply.strip():
            return "很抱歉，我暂时无法回答您的问题。"
        return model_reply
        

def create_gradio_interface(chatbot: ChatBot):
    """
    创建Gradio Web界面
    """
    def chatbot_fn(message, history):
        # 这是一个简化版本，实际需要处理历史和系统类型
        response = chatbot.generate_response(message, history)
        return response

    demo = gr.ChatInterface(
        chatbot_fn,
        title="智能对话机器人系统",
        description="一个集成了大语言模型和知识库的智能对话机器人。",
        examples=["介绍一下你们的退换货政策。", "图书馆开放时间是多久？"]
    )
    return demo
