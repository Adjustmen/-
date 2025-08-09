#!/usr/bin/env python3
"""
intelligent_agent_training.py
智能对话Agent训练实现模块
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
from typing import Dict, List, Any, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import sqlite3
import pickle
import hashlib
from pathlib import Path

# 尝试导入可选依赖
try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️ transformers未安装，将使用简化的嵌入方法")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️ scikit-learn未安装，将使用numpy实现相似度计算")

@dataclass
class ConversationContext:
    """对话上下文数据类"""
    user_id: str
    session_id: str
    conversation_history: List[Dict[str, str]]
    user_profile: Dict[str, Any]
    current_intent: str
    confidence_score: float
    timestamp: datetime

class UserProfileManager:
    """用户画像管理器"""
    
    def __init__(self, db_path: str = "data/user_profiles.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__ + ".UserProfileManager")
        self._init_database()
        
    def _init_database(self):
        """初始化数据库"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 用户画像表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            profile_data TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # 对话历史表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            session_id TEXT,
            message TEXT,
            response TEXT,
            intent TEXT,
            satisfaction_score REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # 用户行为表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_behaviors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            behavior_type TEXT,
            behavior_data TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        conn.commit()
        conn.close()
        self.logger.info(f"数据库初始化完成: {self.db_path}")
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """获取用户画像"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT profile_data FROM user_profiles WHERE user_id = ?", (user_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        else:
            # 创建默认用户画像
            default_profile = {
                "preferences": {},
                "interaction_style": "formal",
                "common_intents": [],
                "satisfaction_history": [],
                "total_conversations": 0,
                "avg_session_length": 0,
                "preferred_response_length": "medium",
                "topics_of_interest": [],
                "language_preference": "zh-CN",
                "created_at": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat()
            }
            self.update_user_profile(user_id, default_profile)
            return default_profile
    
    def update_user_profile(self, user_id: str, profile: Dict[str, Any]):
        """更新用户画像"""
        profile["last_active"] = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT OR REPLACE INTO user_profiles (user_id, profile_data, last_updated)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        """, (user_id, json.dumps(profile, ensure_ascii=False)))
        
        conn.commit()
        conn.close()
        
        self.logger.debug(f"用户画像已更新: {user_id}")
    
    def record_conversation(self, user_id: str, session_id: str, 
                          message: str, response: str, intent: str, 
                          satisfaction_score: float = None):
        """记录对话"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO conversation_history 
        (user_id, session_id, message, response, intent, satisfaction_score)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, session_id, message, response, intent, satisfaction_score))
        
        conn.commit()
        conn.close()
        
        # 更新用户画像统计信息
        self._update_profile_stats(user_id, intent, satisfaction_score)
    
    def _update_profile_stats(self, user_id: str, intent: str, satisfaction_score: float):
        """更新用户画像统计信息"""
        profile = self.get_user_profile(user_id)
        
        # 更新对话次数
        profile["total_conversations"] += 1
        
        # 更新常见意图
        if intent not in profile["common_intents"]:
            profile["common_intents"].append(intent)
        
        # 更新满意度历史
        if satisfaction_score is not None:
            profile["satisfaction_history"].append(satisfaction_score)
            # 只保留最近20次的满意度评分
            profile["satisfaction_history"] = profile["satisfaction_history"][-20:]
        
        self.update_user_profile(user_id, profile)
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """获取用户统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取对话统计
        cursor.execute("""
        SELECT COUNT(*) as total_conversations,
               AVG(satisfaction_score) as avg_satisfaction,
               COUNT(DISTINCT session_id) as total_sessions
        FROM conversation_history 
        WHERE user_id = ? AND satisfaction_score IS NOT NULL
        """, (user_id,))
        
        stats = cursor.fetchone()
        conn.close()
        
        profile = self.get_user_profile(user_id)
        
        return {
            "user_id": user_id,
            "total_conversations": stats[0] if stats[0] else 0,
            "avg_satisfaction": round(stats[1], 2) if stats[1] else 0,
            "total_sessions": stats[2] if stats[2] else 0,
            "interaction_style": profile.get("interaction_style", "unknown"),
            "common_intents": profile.get("common_intents", [])[:5],
            "satisfaction_history": profile.get("satisfaction_history", [])[-10:],
            "created_at": profile.get("created_at"),
            "last_active": profile.get("last_active")
        }

class IntentClassifier:
    """意图识别器"""
    
    def __init__(self, model_path: str = None):
        self.logger = logging.getLogger(__name__ + ".IntentClassifier")
        
        # 基于关键词的意图模式
        self.intent_patterns = {
            "greeting": {
                "keywords": ["你好", "hello", "hi", "早上好", "晚上好", "您好", "嗨"],
                "weight": 1.0
            },
            "question": {
                "keywords": ["什么", "怎么", "为什么", "如何", "哪个", "哪里", "什么时候", "?", "？"],
                "weight": 1.0
            },
            "complaint": {
                "keywords": ["投诉", "不满", "问题", "故障", "错误", "bug", "坏了", "不行"],
                "weight": 1.2
            },
            "praise": {
                "keywords": ["好", "棒", "优秀", "满意", "谢谢", "感谢", "不错", "很好"],
                "weight": 1.0
            },
            "request": {
                "keywords": ["请", "帮助", "需要", "想要", "希望", "能否", "可以", "麻烦"],
                "weight": 1.0
            },
            "goodbye": {
                "keywords": ["再见", "拜拜", "goodbye", "bye", "88", "结束", "退出"],
                "weight": 1.0
            },
            "confirmation": {
                "keywords": ["是的", "对", "没错", "确认", "同意", "可以", "行"],
                "weight": 1.0
            },
            "denial": {
                "keywords": ["不是", "不对", "错了", "不同意", "拒绝", "不行", "不可以"],
                "weight": 1.0
            }
        }
        
        # 尝试加载深度学习模型
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                self.model = self._load_model(model_path)
                self.logger.info("意图识别模型加载成功")
            except Exception as e:
                self.logger.warning(f"模型加载失败，使用规则模式: {e}")
    
    def classify_intent(self, message: str) -> Tuple[str, float]:
        """分类意图"""
        if self.model:
            return self._classify_with_model(message)
        else:
            return self._classify_with_patterns(message)
    
    def _classify_with_patterns(self, message: str) -> Tuple[str, float]:
        """基于模式的意图分类"""
        message_lower = message.lower().strip()
        
        if not message_lower:
            return "general", 0.1
        
        intent_scores = {}
        
        for intent, config in self.intent_patterns.items():
            score = 0
            keywords = config["keywords"]
            weight = config["weight"]
            
            # 计算关键词匹配得分
            for keyword in keywords:
                if keyword in message_lower:
                    # 考虑关键词长度和位置
                    keyword_score = len(keyword) / len(message_lower)
                    if message_lower.startswith(keyword):
                        keyword_score *= 1.5  # 开头匹配权重更高
                    score += keyword_score
            
            intent_scores[intent] = score * weight
        
        # 如果没有匹配到任何意图
        if not intent_scores or max(intent_scores.values()) == 0:
            return "general", 0.3
        
        # 找到最高得分的意图
        best_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[best_intent]
        
        # 计算置信度
        total_score = sum(intent_scores.values())
        confidence = min(max_score / max(total_score, 1e-6), 1.0)
        
        # 确保置信度在合理范围内
        confidence = max(0.1, min(confidence, 0.95))
        
        return best_intent, confidence
    
    def _classify_with_model(self, message: str) -> Tuple[str, float]:
        """基于深度学习模型的意图分类"""
        # 这里应该实现基于深度学习的意图分类
        # 暂时返回规则分类结果
        return self._classify_with_patterns(message)
    
    def _load_model(self, model_path: str):
        """加载意图分类模型"""
        # 这里可以实现模型加载逻辑
        return None
    
    def train_model(self, training_data: List[Tuple[str, str]]):
        """训练意图分类模型"""
        # 这里可以实现模型训练逻辑
        self.logger.info("意图分类模型训练功能待实现")
        pass

class ReinforcementLearningAgent:
    """强化学习Agent"""
    
    def __init__(self, state_dim: int = 128, action_dim: int = 10, lr: float = 0.001):
        self.logger = logging.getLogger(__name__ + ".RLAgent")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        
        # 构建DQN网络
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        
        # 复制权重到目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99  # 折扣因子
        
        # 目标网络更新频率
        self.target_update_freq = 100
        self.update_count = 0
        
        # 行为映射
        self.actions = [
            "provide_detailed_answer",    # 提供详细回答
            "ask_clarification",          # 询问澄清
            "suggest_alternatives",       # 建议替代方案
            "escalate_to_human",         # 转人工服务
            "provide_quick_answer",      # 提供简短回答
            "show_empathy",              # 表示同理心
            "request_feedback",          # 请求反馈
            "end_conversation",          # 结束对话
            "continue_conversation",     # 继续对话
            "provide_examples"           # 提供示例
        ]
        
        self.logger.info(f"强化学习Agent初始化完成，状态维度: {state_dim}, 行动数量: {action_dim}")
    
    def _build_network(self) -> nn.Module:
        """构建神经网络"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
    
    def get_state_vector(self, context: ConversationContext) -> np.ndarray:
        """将对话上下文转换为状态向量"""
        state = np.zeros(self.state_dim, dtype=np.float32)
        
        try:
            # 基础特征
            history_length = len(context.conversation_history)
            state[0] = min(history_length, 20) / 20  # 对话长度特征
            state[1] = context.confidence_score  # 意图置信度
            
            # 用户满意度特征
            satisfaction_history = context.user_profile.get("satisfaction_history", [])
            if satisfaction_history:
                state[2] = np.mean(satisfaction_history[-5:]) / 5.0  # 最近满意度
                state[3] = len(satisfaction_history) / 100  # 反馈数量
            
            # 会话时长特征
            if context.conversation_history:
                time_diff = (datetime.now() - context.timestamp).total_seconds()
                state[4] = min(time_diff, 3600) / 3600  # 归一化到1小时
            
            # 意图类型编码
            intent_mapping = {
                "greeting": 0.1, "question": 0.2, "complaint": 0.3,
                "praise": 0.4, "request": 0.5, "goodbye": 0.6,
                "confirmation": 0.7, "denial": 0.8, "general": 0.9
            }
            state[5] = intent_mapping.get(context.current_intent, 0.9)
            
            # 用户交互特征
            profile = context.user_profile
            state[6] = profile.get("total_conversations", 0) / 1000  # 总对话次数
            
            # 交互风格编码
            style_mapping = {"formal": 0.3, "casual": 0.7, "mixed": 0.5}
            state[7] = style_mapping.get(profile.get("interaction_style"), 0.5)
            
            # 常见意图频率
            common_intents = profile.get("common_intents", [])
            if context.current_intent in common_intents:
                state[8] = (common_intents.index(context.current_intent) + 1) / len(common_intents)
            
            # 时间特征
            current_hour = datetime.now().hour
            state[9] = current_hour / 24  # 当前小时
            
            # 填充其余维度（可以添加更多特征）
            for i in range(10, min(20, self.state_dim)):
                state[i] = np.random.normal(0, 0.1)  # 噪声特征
                
        except Exception as e:
            self.logger.error(f"状态向量生成失败: {e}")
            # 返回默认状态向量
            state = np.random.random(self.state_dim) * 0.1
        
        return state
    
    def select_action(self, state: np.ndarray) -> str:
        """选择行动"""
        if np.random.random() <= self.epsilon:
            # 随机探索
            action_idx = np.random.randint(self.action_dim)
        else:
            # 利用当前策略
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax().item()
        
        return self.actions[action_idx]
    
    def store_experience(self, state: np.ndarray, action: str, reward: float, 
                        next_state: np.ndarray, done: bool):
        """存储经验"""
        try:
            action_idx = self.actions.index(action)
            self.memory.append((state, action_idx, reward, next_state, done))
        except ValueError:
            self.logger.warning(f"未知行动: {action}")
    
    def train(self, batch_size: int = 32):
        """训练网络"""
        if len(self.memory) < batch_size:
            return
        
        try:
            # 随机采样经验
            indices = np.random.choice(len(self.memory), batch_size, replace=False)
            batch = [self.memory[i] for i in indices]
            
            states = torch.FloatTensor([experience[0] for experience in batch])
            actions = torch.LongTensor([experience[1] for experience in batch])
            rewards = torch.FloatTensor([experience[2] for experience in batch])
            next_states = torch.FloatTensor([experience[3] for experience in batch])
            dones = torch.BoolTensor([experience[4] for experience in batch])
            
            # 计算Q值
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            next_q_values = self.target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
            # 计算损失并更新网络
            loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.optimizer.step()
            
            # 更新目标网络
            self.update_count += 1
            if self.update_count % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            # 衰减探索率
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            self.logger.debug(f"训练完成，损失: {loss.item():.4f}, 探索率: {self.epsilon:.4f}")
            
        except Exception as e:
            self.logger.error(f"训练过程出错: {e}")

class ContextualMemory:
    """上下文记忆管理"""
    
    def __init__(self, max_contexts: int = 1000):
        self.logger = logging.getLogger(__name__ + ".ContextualMemory")
        self.max_contexts = max_contexts
        self.memory_bank = {}
        self.embedding_cache = {}
        
        # 初始化嵌入模型（如果可用）
        self.embedding_model = None
        if HAS_TRANSFORMERS:
            try:
                # 这里可以加载实际的嵌入模型
                self.embedding_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", local_files_only=True)
                self.embedding_model = AutoModel.from_pretrained("bert-base-chinese", local_files_only=True)
            except Exception as e:
                self.logger.warning(f"嵌入模型加载失败: {e}")
        
        self.logger.info(f"上下文记忆初始化，最大容量: {max_contexts}")
        
    def store_context(self, context: ConversationContext):
        """存储上下文"""
        key = f"{context.user_id}_{context.session_id}"
        
        # 生成上下文嵌入
        embedding = self._get_context_embedding(context)
        
        self.memory_bank[key] = {
            "context": context,
            "embeddings": embedding,
            "timestamp": datetime.now(),
            "access_count": 1
        }
        
        # 清理过期记忆
        if len(self.memory_bank) > self.max_contexts:
            self._cleanup_old_contexts()
        
        self.logger.debug(f"上下文已存储: {key}")
    
    def retrieve_similar_contexts(self, current_context: ConversationContext, 
                                 top_k: int = 5) -> List[ConversationContext]:
        """检索相似上下文"""
        if not self.memory_bank:
            return []
        
        current_embedding = self._get_context_embedding(current_context)
        similarities = []
        
        for key, stored in self.memory_bank.items():
            # 跳过当前用户的当前会话
            if key.startswith(f"{current_context.user_id}_{current_context.session_id}"):
                continue
            
            # 计算相似度
            similarity = self._calculate_similarity(current_embedding, stored["embeddings"])
            similarities.append((similarity, stored["context"], key))
            
            # 更新访问次数
            stored["access_count"] += 1
        
        # 按相似度排序并返回top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        result_contexts = []
        for similarity, context, key in similarities[:top_k]:
            if similarity > 0.3:  # 相似度阈值
                result_contexts.append(context)
                self.logger.debug(f"找到相似上下文: {key}, 相似度: {similarity:.3f}")
        
        return result_contexts
    
    def _get_context_embedding(self, context: ConversationContext) -> np.ndarray:
        """获取上下文嵌入"""
        # 生成上下文的文本表示
        text_parts = []
        
        # 添加最近的对话内容
        for msg in context.conversation_history[-3:]:
            content = msg.get("content", "")
            if content:
                text_parts.append(content)
        
        # 添加意图信息
        text_parts.append(f"intent:{context.current_intent}")
        
        # 添加用户偏好
        profile = context.user_profile
        if profile.get("topics_of_interest"):
            text_parts.append(f"interests:{','.join(profile['topics_of_interest'][:3])}")
        
        full_text = " ".join(text_parts)
        
        # 生成嵌入
        if self.embedding_model and HAS_TRANSFORMERS:
            return self._get_transformer_embedding(full_text)
        else:
            return self._get_simple_embedding(full_text)
    
    def _get_transformer_embedding(self, text: str) -> np.ndarray:
        """使用Transformer模型生成嵌入"""
        # 这里可以实现真正的Transformer嵌入
        # 暂时返回简单嵌入
        return self._get_simple_embedding(text)
    
    def _get_simple_embedding(self, text: str) -> np.ndarray:
        """简单的嵌入方法"""
        # 使用字符哈希生成固定维度的向量
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # 转换为128维向量
        embedding = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)
        
        # 扩展到128维
        if len(embedding) < 128:
            embedding = np.tile(embedding, (128 // len(embedding) + 1))[:128]
        
        # 归一化
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def _calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """计算相似度"""
        if HAS_SKLEARN:
            return cosine_similarity([emb1], [emb2])[0][0]
        else:
            # 使用numpy实现余弦相似度
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
    
    def _cleanup_old_contexts(self):
        """清理过期上下文"""
        cutoff_time = datetime.now() - timedelta(days=7)
        keys_to_remove = []
        
        # 按访问次数和时间排序，移除最不重要的记忆
        memory_items = list(self.memory_bank.items())
        memory_items.sort(key=lambda x: (x[1]["access_count"], x[1]["timestamp"]))
        
        # 移除最旧的记忆直到达到容量限制
        remove_count = len(self.memory_bank) - self.max_contexts + 100  # 多删一些以避免频繁清理
        for i in range(min(remove_count, len(memory_items))):
            keys_to_remove.append(memory_items[i][0])
        
        # 移除过期记忆
        for key, stored in self.memory_bank.items():
            if stored["timestamp"] < cutoff_time:
                keys_to_remove.append(key)
        
        for key in set(keys_to_remove):
            if key in self.memory_bank:
                del self.memory_bank[key]
        
        if keys_to_remove:
            self.logger.info(f"清理了 {len(keys_to_remove)} 个过期上下文")

class IntelligentChatBot:
    """智能聊天机器人（增强版）"""
    
    def __init__(self, base_chatbot, config: Dict[str, Any]):
        self.base_chatbot = base_chatbot
        self.config = config
        self.logger = logging.getLogger(__name__ + ".IntelligentChatBot")
        
        # 初始化智能组件
        self.user_manager = UserProfileManager()
        self.intent_classifier = IntentClassifier()
        self.rl_agent = ReinforcementLearningAgent()
        self.memory = ContextualMemory()
        
        # 对话状态管理
        self.active_contexts = {}
        self.conversation_count = 0
        
        # 配置参数
        self.max_active_contexts = config.get('intelligence', {}).get('max_active_contexts', 100)
        self.context_timeout = config.get('dialogue', {}).get('session_timeout', 3600)
        
        self.logger.info("智能聊天机器人初始化完成")
    
    def chat(self, user_id: str, message: str, session_id: str = None) -> Dict[str, Any]:
        """智能对话主方法"""
        try:
            # 生成会话ID
            if not session_id:
                session_id = f"{user_id}_{int(datetime.now().timestamp())}"
            
            # 获取或创建对话上下文
            context = self._get_or_create_context(user_id, session_id, message)
            
            # 意图识别
            intent, confidence = self.intent_classifier.classify_intent(message)
            context.current_intent = intent
            context.confidence_score = confidence
            
            # 获取相似历史对话
            similar_contexts = self.memory.retrieve_similar_contexts(context)
            
            # 强化学习决策
            state = self.rl_agent.get_state_vector(context)
            action = self.rl_agent.select_action(state)
            
            # 生成回复
            response = self._generate_response(context, action, similar_contexts)
            
            # 记录对话
            context.conversation_history.append({
                "role": "user", 
                "content": message, 
                "timestamp": datetime.now().isoformat()
            })
            context.conversation_history.append({
                "role": "assistant", 
                "content": response, 
                "timestamp": datetime.now().isoformat()
            })
            
            # 存储上下文和记录对话
            self.memory.store_context(context)
            self.active_contexts[f"{user_id}_{session_id}"] = context
            self.user_manager.record_conversation(user_id, session_id, message, response, intent)
            
            # 清理过期上下文
            self._cleanup_expired_contexts()
            
            # 更新统计
            self.conversation_count += 1
            
            self.logger.debug(f"对话完成 - 用户: {user_id}, 意图: {intent}, 行动: {action}")
            
            return {
                "response": response,
                "intent": intent,
                "confidence": confidence,
                "action": action,
                "context_id": f"{user_id}_{session_id}",
                "similar_contexts_count": len(similar_contexts)
            }
            
        except Exception as e:
            self.logger.error(f"对话处理失败: {e}")
            return {
                "response": "抱歉，我遇到了一些技术问题，请稍后再试。",
                "intent": "error",
                "confidence": 0.0,
                "action": "error_handling",
                "context_id": f"{user_id}_{session_id}" if session_id else None,
                "error": str(e)
            }
    
    def provide_feedback(self, context_id: str, satisfaction_score: float):
        """提供反馈用于强化学习"""
        try:
            if context_id not in self.active_contexts:
                self.logger.warning(f"上下文不存在: {context_id}")
                return False
            
            context = self.active_contexts[context_id]
            
            # 计算奖励
            reward = self._calculate_reward(satisfaction_score, context)
            
            # 更新强化学习模型
            if len(context.conversation_history) >= 4:  # 至少有一轮完整对话
                # 获取状态和动作
                current_state = self.rl_agent.get_state_vector(context)
                
                # 简化处理：使用最后的动作
                last_action = "continue_conversation"  # 可以从上下文中获取真实的动作
                
                # 创建下一个状态（当前状态的副本）
                next_state = current_state.copy()
                
                # 存储经验
                self.rl_agent.store_experience(
                    current_state, last_action, reward, next_state, False
                )
                
                # 执行训练
                self.rl_agent.train(batch_size=16)
            
            # 更新用户画像
            profile = self.user_manager.get_user_profile(context.user_id)
            profile["satisfaction_history"].append(satisfaction_score)
            
            # 只保留最近的评分
            if len(profile["satisfaction_history"]) > 50:
                profile["satisfaction_history"] = profile["satisfaction_history"][-50:]
            
            # 更新交互风格
            if satisfaction_score >= 4:
                # 用户满意，可能喜欢当前的交互风格
                pass
            elif satisfaction_score <= 2:
                # 用户不满意，可能需要调整交互风格
                if profile.get("interaction_style") == "formal":
                    profile["interaction_style"] = "casual"
                else:
                    profile["interaction_style"] = "formal"
            
            self.user_manager.update_user_profile(context.user_id, profile)
            
            self.logger.info(f"反馈已记录: {context_id}, 满意度: {satisfaction_score}, 奖励: {reward}")
            return True
            
        except Exception as e:
            self.logger.error(f"反馈处理失败: {e}")
            return False
    
    def _get_or_create_context(self, user_id: str, session_id: str, message: str) -> ConversationContext:
        """获取或创建对话上下文"""
        context_key = f"{user_id}_{session_id}"
        
        if context_key in self.active_contexts:
            context = self.active_contexts[context_key]
            context.timestamp = datetime.now()  # 更新时间戳
            return context
        
        # 创建新的上下文
        profile = self.user_manager.get_user_profile(user_id)
        
        context = ConversationContext(
            user_id=user_id,
            session_id=session_id,
            conversation_history=[],
            user_profile=profile,
            current_intent="",
            confidence_score=0.0,
            timestamp=datetime.now()
        )
        
        return context
    
    def _generate_response(self, context: ConversationContext, action: str, 
                          similar_contexts: List[ConversationContext]) -> str:
        """生成智能回复"""
        try:
            # 构建增强的提示
            message = context.conversation_history[-1]["content"] if context.conversation_history else ""
            
            # 根据行动类型调整提示前缀
            prompt_prefix = self._get_action_prompt(action, context)
            
            # 添加用户偏好信息
            preference_context = self._build_preference_context(context.user_profile)
            
            # 添加相似对话参考
            reference_context = self._build_reference_context(similar_contexts)
            
            # 构建完整提示
            enhanced_prompt = f"{prompt_prefix}{preference_context}{reference_context}{message}"
            
            # 使用基础聊天机器人生成回复
            if hasattr(self.base_chatbot, 'chat'):
                base_response = self.base_chatbot.chat(enhanced_prompt, context.user_id)
            else:
                # 回退到简单回复
                base_response = self._generate_fallback_response(message, context)
            
            # 后处理回复
            final_response = self._postprocess_response(base_response, action, context)
            
            return final_response
            
        except Exception as e:
            self.logger.error(f"回复生成失败: {e}")
            return "抱歉，我在处理您的问题时遇到了困难，请稍后再试。"
    
    def _get_action_prompt(self, action: str, context: ConversationContext) -> str:
        """获取行动对应的提示前缀"""
        action_prompts = {
            "provide_detailed_answer": "请提供详细和全面的解答：",
            "ask_clarification": "为了更好地帮助您，我需要了解更多信息。请问：",
            "suggest_alternatives": "我建议您可以考虑以下几个选择：",
            "escalate_to_human": "这个问题可能需要专业人员处理，",
            "provide_quick_answer": "简单来说：",
            "show_empathy": "我理解您的感受，让我来帮助您解决这个问题：",
            "request_feedback": "希望我的回答对您有帮助，",
            "end_conversation": "感谢您的咨询，",
            "continue_conversation": "",
            "provide_examples": "让我为您提供一些具体的例子："
        }
        
        return action_prompts.get(action, "")
    
    def _build_preference_context(self, user_profile: Dict[str, Any]) -> str:
        """构建用户偏好上下文"""
        context_parts = []
        
        # 交互风格
        interaction_style = user_profile.get("interaction_style", "formal")
        if interaction_style == "casual":
            context_parts.append("使用轻松友好的语气。")
        elif interaction_style == "formal":
            context_parts.append("使用专业正式的语气。")
        
        # 偏好的回复长度
        preferred_length = user_profile.get("preferred_response_length", "medium")
        if preferred_length == "short":
            context_parts.append("回答要简洁明了。")
        elif preferred_length == "detailed":
            context_parts.append("回答要详细全面。")
        
        # 兴趣主题
        interests = user_profile.get("topics_of_interest", [])
        if interests:
            context_parts.append(f"用户对{', '.join(interests[:3])}等主题感兴趣。")
        
        return " ".join(context_parts) + " " if context_parts else ""
    
    def _build_reference_context(self, similar_contexts: List[ConversationContext]) -> str:
        """构建相似对话参考上下文"""
        if not similar_contexts:
            return ""
        
        # 简化处理：只提供参考提示
        return f"参考{len(similar_contexts)}个相似对话的处理方式。"
    
    def _generate_fallback_response(self, message: str, context: ConversationContext) -> str:
        """生成回退回复"""
        intent = context.current_intent
        
        fallback_responses = {
            "greeting": "您好！我是智能助手，很高兴为您服务。有什么我可以帮助您的吗？",
            "question": "这是一个很好的问题。让我为您查找相关信息...",
            "complaint": "非常抱歉给您带来不便。我会尽力帮助您解决这个问题。",
            "praise": "谢谢您的肯定！我会继续努力为您提供更好的服务。",
            "request": "我明白您的需求，让我来帮助您...",
            "goodbye": "再见！如果您还有其他问题，随时可以联系我。",
            "general": "我理解您的问题，让我为您提供一些建议..."
        }
        
        return fallback_responses.get(intent, "我会尽力帮助您解决问题。")
    
    def _postprocess_response(self, response: str, action: str, context: ConversationContext) -> str:
        """后处理回复"""
        try:
            # 根据用户交互风格调整语气
            if context.user_profile.get("interaction_style") == "casual":
                response = response.replace("您", "你").replace("您的", "你的")
            
            # 根据行动添加特定后缀
            action_suffixes = {
                "request_feedback": "\n\n请问这个回答对您有帮助吗？您可以给我打个分（1-5分）。",
                "suggest_alternatives": "\n\n还有其他问题我可以帮助您吗？",
                "ask_clarification": "\n\n请您提供更多详细信息，这样我可以给出更准确的答案。",
                "show_empathy": "\n\n如果您还有其他困扰，我很愿意继续帮助您。"
            }
            
            if action in action_suffixes:
                response += action_suffixes[action]
            
            # 长度控制
            preferred_length = context.user_profile.get("preferred_response_length", "medium")
            if preferred_length == "short" and len(response) > 200:
                # 简化长回复
                sentences = response.split("。")
                response = "。".join(sentences[:2]) + "。"
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"回复后处理失败: {e}")
            return response
    
    def _calculate_reward(self, satisfaction_score: float, context: ConversationContext) -> float:
        """计算强化学习奖励"""
        # 基础奖励：将1-5分转换为-1到1的范围
        base_reward = (satisfaction_score - 3) / 2
        
        # 对话效率奖励/惩罚
        conversation_length = len(context.conversation_history)
        if conversation_length <= 4:  # 快速解决问题
            efficiency_bonus = 0.2
        elif conversation_length > 10:  # 对话过长
            efficiency_penalty = -0.1 * (conversation_length - 10) / 10
            efficiency_bonus = max(efficiency_penalty, -0.3)
        else:
            efficiency_bonus = 0
        
        # 意图识别准确性奖励
        confidence_bonus = 0
        if context.confidence_score > 0.8:
            confidence_bonus = 0.1
        elif context.confidence_score < 0.3:
            confidence_bonus = -0.1
        
        # 用户忠诚度奖励
        loyalty_bonus = 0
        total_conversations = context.user_profile.get("total_conversations", 0)
        if total_conversations > 10:  # 老用户
            loyalty_bonus = 0.1
        
        # 计算最终奖励
        final_reward = base_reward + efficiency_bonus + confidence_bonus + loyalty_bonus
        
        # 限制奖励范围
        final_reward = max(-1.0, min(1.0, final_reward))
        
        return final_reward
    
    def _cleanup_expired_contexts(self):
        """清理过期的对话上下文"""
        current_time = datetime.now()
        expired_keys = []
        
        for key, context in self.active_contexts.items():
            time_diff = (current_time - context.timestamp).total_seconds()
            if time_diff > self.context_timeout:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.active_contexts[key]
        
        # 限制活跃上下文数量
        if len(self.active_contexts) > self.max_active_contexts:
            # 按时间戳排序，删除最旧的
            sorted_contexts = sorted(
                self.active_contexts.items(), 
                key=lambda x: x[1].timestamp
            )
            
            remove_count = len(self.active_contexts) - self.max_active_contexts
            for i in range(remove_count):
                key = sorted_contexts[i][0]
                del self.active_contexts[key]
        
        if expired_keys:
            self.logger.debug(f"清理了 {len(expired_keys)} 个过期上下文")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            "total_conversations": self.conversation_count,
            "active_contexts": len(self.active_contexts),
            "memory_bank_size": len(self.memory.memory_bank),
            "rl_agent_epsilon": self.rl_agent.epsilon,
            "rl_memory_size": len(self.rl_agent.memory),
            "system_uptime": datetime.now().isoformat()
        }
    
    def save_models(self, save_dir: str):
        """保存训练的模型"""
        try:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # 保存强化学习模型
            rl_model_path = save_path / "rl_model.pt"
            torch.save({
                'q_network_state_dict': self.rl_agent.q_network.state_dict(),
                'target_network_state_dict': self.rl_agent.target_network.state_dict(),
                'optimizer_state_dict': self.rl_agent.optimizer.state_dict(),
                'epsilon': self.rl_agent.epsilon,
                'update_count': self.rl_agent.update_count
            }, rl_model_path)
            
            # 保存记忆库
            memory_path = save_path / "memory_bank.pkl"
            with open(memory_path, "wb") as f:
                pickle.dump(self.memory.memory_bank, f)
            
            # 保存系统统计信息
            stats_path = save_path / "system_stats.json"
            with open(stats_path, "w", encoding='utf-8') as f:
                json.dump(self.get_system_stats(), f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"模型已保存到 {save_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"模型保存失败: {e}")
            return False
    
    def load_models(self, save_dir: str):
        """加载训练的模型"""
        try:
            save_path = Path(save_dir)
            
            # 加载强化学习模型
            rl_model_path = save_path / "rl_model.pt"
            if rl_model_path.exists():
                checkpoint = torch.load(rl_model_path, map_location='cpu')
                self.rl_agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                self.rl_agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
                self.rl_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.rl_agent.epsilon = checkpoint.get('epsilon', self.rl_agent.epsilon)
                self.rl_agent.update_count = checkpoint.get('update_count', 0)
                self.logger.info("强化学习模型加载成功")
            
            # 加载记忆库
            memory_path = save_path / "memory_bank.pkl"
            if memory_path.exists():
                with open(memory_path, "rb") as f:
                    self.memory.memory_bank = pickle.load(f)
                self.logger.info(f"记忆库加载成功，包含 {len(self.memory.memory_bank)} 个上下文")
            
            return True
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            return False

# 工具函数
def create_sample_training_data():
    """创建示例训练数据"""
    sample_data = {
        "conversations": [
            {
                "user_id": "user_001",
                "messages": [
                    {"role": "user", "content": "你好", "intent": "greeting"},
                    {"role": "assistant", "content": "您好！我是智能助手，很高兴为您服务。"},
                    {"role": "user", "content": "我想了解一下产品信息", "intent": "question"},
                    {"role": "assistant", "content": "好的，我来为您介绍我们的产品..."}
                ],
                "satisfaction_score": 4.5
            }
        ],
        "user_profiles": [
            {
                "user_id": "user_001",
                "preferences": {"style": "formal", "length": "medium"},
                "topics": ["product", "service"]
            }
        ]
    }
    
    # 保存示例数据
    data_dir = Path("data/training")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    with open(data_dir / "sample_data.json", "w", encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print("✅ 示例训练数据已创建")

if __name__ == "__main__":
    # 测试代码
    print("智能对话Agent训练模块测试")
    
    # 创建示例数据
    create_sample_training_data()
    
    # 测试各个组件
    print("\n测试用户画像管理器...")
    user_manager = UserProfileManager("data/test_profiles.db")
    profile = user_manager.get_user_profile("test_user")
    print(f"用户画像: {profile}")
    
    print("\n测试意图识别器...")
    intent_classifier = IntentClassifier()
    intent, confidence = intent_classifier.classify_intent("你好，我需要帮助")
    print(f"意图: {intent}, 置信度: {confidence:.2f}")
    
    print("\n测试强化学习Agent...")
    rl_agent = ReinforcementLearningAgent()
    
    # 创建测试上下文
    test_context = ConversationContext(
        user_id="test_user",
        session_id="session_001", 
        conversation_history=[],
        user_profile=profile,
        current_intent=intent,
        confidence_score=confidence,
        timestamp=datetime.now()
    )
    
    state = rl_agent.get_state_vector(test_context)
    action = rl_agent.select_action(state)
    print(f"状态维度: {len(state)}, 选择的行动: {action}")
    
    print("\n测试上下文记忆...")
    memory = ContextualMemory()
    memory.store_context(test_context)
    similar = memory.retrieve_similar_contexts(test_context)
    print(f"相似上下文数量: {len(similar)}")
    
    print("\n✅ 所有组件测试完成！")
