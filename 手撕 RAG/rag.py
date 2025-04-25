import numpy as np  # 导入NumPy库，用于处理向量和矩阵运算
from ollama import embeddings, chat, Message  # 导入ollama库中的嵌入、聊天和消息模块

# 创建一个知识库类，用于存储和检索文档
class KnowledgeBase:
    def __init__(self, filepath):
        """
        初始化知识库，读取文件内容并生成文档的嵌入向量。
        :param filepath: 知识库文件路径
        """
        with open(filepath, 'r', encoding='UTF-8') as f:
            content = f.read()  # 读取文件内容
        self.docs = self.split_content(content)  # 将内容分割成小块，便于处理
        self.embeds = self.encode(self.docs)  # 对每个小块生成嵌入向量，用于后续的相似度计算
        
    @staticmethod
    def split_content(content, max_length=50):
        """
        将内容分割成指定长度的小块。
        :param content: 文本内容
        :param max_length: 每块的最大长度
        :return: 分割后的文本块列表
        """
        chunks = []  # 用于存储分割后的文本块
        for i in range(0, len(content), max_length):  # 按照指定长度切分文本
            chunks.append(content[i: i+max_length])  # 每次取出max_length长度的文本
        return chunks  # 返回分割后的文本块列表
    
    def encode(self, docs):
        """
        对文档列表中的每个文本生成嵌入向量。
        :param docs: 文档列表
        :return: 嵌入向量的数组
        """
        encodes = []  # 用于存储每个文档的嵌入向量
        for text in docs:  # 遍历每个文档块
            response = embeddings(model='nomic-embed-text', prompt=text)  # 调用嵌入模型生成嵌入
            encodes.append(response['embedding'])  # 提取嵌入向量并添加到列表
        return np.array(encodes)  # 转换为NumPy数组并返回，便于后续计算
    
    @staticmethod
    def similarity(A, B):
        """
        计算两个向量的余弦相似度。
        :param A: 向量A
        :param B: 向量B
        :return: 余弦相似度
        """
        dot_product = np.dot(A, B)  # 计算两个向量的点积
        norm_A = np.linalg.norm(A)  # 计算向量A的模（长度）
        norm_B = np.linalg.norm(B)  # 计算向量B的模（长度）
        return dot_product / (norm_A * norm_B)  # 计算余弦相似度并返回
    
    def search(self, query):
        """
        在知识库中搜索与查询最相似的文档。
        :param query: 查询文本
        :return: 最相似的文档
        """
        max_similarity = -1  # 初始化最大相似度为-1
        max_similarity_index = -1  # 初始化最大相似度对应的索引为-1
        query_embedding = self.encode([query])[0]  # 对查询文本生成嵌入向量
        for i, doc_embedding in enumerate(self.embeds):  # 遍历知识库中的嵌入向量
            similarity = self.similarity(query_embedding, doc_embedding)  # 计算查询与文档的相似度
            if similarity > max_similarity:  # 如果当前相似度更高，则更新最大相似度和索引
                max_similarity = similarity
                max_similarity_index = i
        return self.docs[max_similarity_index]  # 返回最相似的文档
    
# 创建一个RAG类，用于结合知识库和聊天模型生成响应
class RAG:
    def __init__(self, model, knowledge_base):
        """
        初始化RAG模型，加载知识库。
        :param model: 使用的聊天模型
        :param knowledge_base: 知识库实例
        """
        self.model = model  # 存储聊天模型实例
        self.kb = knowledge_base  # 存储知识库实例
        self.template = "基于{}的知识库，回答以下问题:{}"  # 定义模板，用于生成提示
    
    def generate_response(self, query):
        """
        生成响应，使用知识库中的信息。
        :param query: 查询文本
        :return: 响应文本
        """
        content = self.kb.search(query)  # 在知识库中搜索最相关的内容
        prompt = f"基于{content}的知识库，回答以下问题：{query}"  # 使用模板生成提示
        response = chat(model=self.model, messages=[Message(role='system', content=prompt)])  # 调用聊天模型生成响应
        return response['message']  # 返回生成的响应
        
# 创建RAG实例，加载聊天模型和知识库
rag = RAG(model='qwen2.5:7b', knowledge_base=KnowledgeBase('爱因斯坦.txt'))  # 指定模型和知识库文件路径

# 循环接收用户输入并生成响应
while True:  # 无限循环，直到用户手动终止
    q = input("请输入问题：")  # 提示用户输入问题
    r = rag.generate_response(q)  # 调用RAG实例生成响应
    print(r['content'])  # 打印响应内容

