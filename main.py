#!/usr/bin/env python3
"""
智能对话机器人系统 - 增强版主程序
集成强化学习、用户画像、智能决策等功能
"""

import os
import sys
import argparse
import yaml
import logging
import numpy as np
from pathlib import Path
import signal
import time
import threading
from typing import Dict, Any
import gradio as gr

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from chatbot_system import ChatBot, create_gradio_interface
    from intelligent_agent_training import IntelligentChatBot, UserProfileManager
except ImportError:
    print("⚠️ 使用内联模块...")
    exec(open("chatbot_system.py").read())
    exec(open("intelligent_agent_training.py").read())

class EnhancedChatBotSystem:
    """增强版聊天机器人系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化基础聊天机器人
        self.base_chatbot = None
        self.intelligent_bot = None
        
        # 系统状态
        self.is_training_mode = config.get('training', {}).get('enabled', False)
        self.model_save_dir = config.get('training', {}).get('save_dir', 'models/trained')
        
        # 训练线程
        self.training_thread = None
        self.training_active = False
    
    def initialize(self):
        """初始化系统"""
        try:
            # 初始化基础聊天机器人
            self.logger.info("初始化基础聊天机器人...")
            self.base_chatbot = ChatBot(
                model_name=self.config['model']['chat_model'],
                device=self.config['model']['device'],
                load_in_4bit=self.config['model'].get('load_in_4bit', False),
                embedding_model=self.config['model']['embedding_model'],
                local_files_only=True
            )
            
            # 初始化智能聊天机器人
            self.logger.info("初始化智能Agent...")
            self.intelligent_bot = IntelligentChatBot(self.base_chatbot, self.config)
            
            # 尝试加载已训练的模型
            if os.path.exists(self.model_save_dir):
                self.intelligent_bot.load_models(self.model_save_dir)
            
            # 如果启用训练模式，启动后台训练
            if self.is_training_mode:
                self.start_training_thread()
            
            self.logger.info("系统初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}")
            return False
    
    def start_training_thread(self):
        """启动训练线程"""
        if not self.training_active:
            self.training_active = True
            self.training_thread = threading.Thread(
                target=self._continuous_training,
                daemon=True
            )
            self.training_thread.start()
            self.logger.info("后台训练线程已启动")
    
    def stop_training_thread(self):
        """停止训练线程"""
        self.training_active = False
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5)
        self.logger.info("后台训练线程已停止")
    
    def _continuous_training(self):
        """持续训练循环"""
        training_interval = self.config.get('training', {}).get('interval', 300)  # 5分钟
        save_interval = self.config.get('training', {}).get('save_interval', 3600)  # 1小时
        
        last_save_time = time.time()
        
        while self.training_active:
            try:
                # 执行训练步骤
                if self.intelligent_bot and hasattr(self.intelligent_bot.rl_agent, 'train'):
                    self.intelligent_bot.rl_agent.train(batch_size=16)
                
                # 定期保存模型
                current_time = time.time()
                if current_time - last_save_time > save_interval:
                    self.save_models()
                    last_save_time = current_time
                
                time.sleep(training_interval)
                
            except Exception as e:
                self.logger.error(f"训练过程中出现错误: {e}")
                time.sleep(60)  # 出错后等待1分钟再继续
    
    def save_models(self):
        """保存模型"""
        try:
            if self.intelligent_bot:
                self.intelligent_bot.save_models(self.model_save_dir)
                self.logger.info("模型保存成功")
        except Exception as e:
            self.logger.error(f"模型保存失败: {e}")
    
    def chat_with_intelligence(self, message: str, user_id: str = "default", 
                             session_id: str = None, history: list = None) -> tuple:
        """智能对话接口"""
        try:
            if not self.intelligent_bot:
                # 回退到基础聊天机器人
                response = self.base_chatbot.chat(message, user_id)
                return response, history + [[message, response]] if history else [[message, response]]
            
            # 使用智能机器人进行对话
            result = self.intelligent_bot.chat(user_id, message, session_id)
            response = result["response"]
            
            # 更新历史记录
            new_history = history + [[message, response]] if history else [[message, response]]
            
            # 记录对话统计信息
            self._log_conversation_stats(result)
            
            return response, new_history
            
        except Exception as e:
            self.logger.error(f"对话处理失败: {e}")
            fallback_response = "抱歉，我遇到了一些技术问题，请稍后再试。"
            new_history = history + [[message, fallback_response]] if history else [[message, fallback_response]]
            return fallback_response, new_history
    
    def provide_user_feedback(self, context_id: str, satisfaction_score: float):
        """用户反馈接口"""
        try:
            if self.intelligent_bot:
                self.intelligent_bot.provide_feedback(context_id, satisfaction_score)
                self.logger.info(f"用户反馈已记录: {context_id}, 满意度: {satisfaction_score}")
        except Exception as e:
            self.logger.error(f"反馈处理失败: {e}")
    
    def _log_conversation_stats(self, result: Dict[str, Any]):
        """记录对话统计信息"""
        self.logger.debug(f"对话统计 - 意图: {result.get('intent')}, "
                         f"置信度: {result.get('confidence'):.2f}, "
                         f"行动: {result.get('action')}")

def create_enhanced_gradio_interface(system: EnhancedChatBotSystem):
    """创建增强版Gradio界面"""
    
    def chat_interface(message, history, user_id, system_type):
        """聊天界面处理函数"""
        if not message.strip():
            return "", history
        
        # 生成会话ID
        session_id = f"{user_id}_{int(time.time())}"
        
        response, new_history = system.chat_with_intelligence(
            message, user_id, session_id, history
        )
        
        return "", new_history
    
    def feedback_interface(rating, context_info):
        """反馈界面处理函数"""
        if context_info and rating:
            try:
                # 解析上下文信息（简化处理）
                parts = context_info.split('_')
                if len(parts) >= 2:
                    context_id = context_info
                    system.provide_user_feedback(context_id, float(rating))
                    return "✅ 反馈已提交，谢谢您的评价！"
            except Exception as e:
                return f"❌ 反馈提交失败: {e}"
        return "请提供有效的评分"
    
    def get_user_stats(user_id):
        """获取用户统计信息"""
        try:
            if system.intelligent_bot:
                profile = system.intelligent_bot.user_manager.get_user_profile(user_id)
                stats = f"""
                📊 用户统计信息:
                - 总对话次数: {profile.get('total_conversations', 0)}
                - 交互风格: {profile.get('interaction_style', '未知')}
                - 平均满意度: {np.mean(profile.get('satisfaction_history', [3]))::.2f}/5.0
                - 常见意图: {', '.join(profile.get('common_intents', [])[:3])}
                """
                return stats
        except Exception as e:
            return f"获取统计信息失败: {e}"
        return "暂无统计信息"
    
    # 创建Gradio界面
    with gr.Blocks(title="智能对话机器人系统 (AI增强版)", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 智能对话机器人系统 (AI增强版)")
        gr.Markdown("集成了强化学习、用户画像、智能决策的对话系统")
        
        with gr.Row():
            with gr.Column(scale=3):
                # 主对话区域
                chatbot = gr.Chatbot(
                    label="对话窗口",
                    height=500,
                    show_copy_button=True
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="输入消息",
                        placeholder="请输入您的问题...",
                        scale=4
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("清空对话", variant="secondary")
                    user_id_input = gr.Textbox(
                        label="用户ID",
                        value="default_user",
                        scale=2
                    )
                    system_type = gr.Dropdown(
                        choices=list(system.config.get('system_types', {}).keys()),
                        value="default",
                        label="系统类型",
                        scale=2
                    )
            
            with gr.Column(scale=1):
                # 侧边栏功能
                gr.Markdown("### 🎯 智能功能")
                
                # 用户反馈
                with gr.Group():
                    gr.Markdown("#### 📝 用户反馈")
                    feedback_rating = gr.Slider(
                        minimum=1, maximum=5, value=3, step=1,
                        label="满意度评分"
                    )
                    context_input = gr.Textbox(
                        label="对话ID",
                        placeholder="从系统日志获取...",
                        visible=False
                    )
                    feedback_btn = gr.Button("提交反馈", variant="secondary")
                    feedback_output = gr.Textbox(label="反馈结果", interactive=False)
                
                # 用户统计
                with gr.Group():
                    gr.Markdown("#### 📊 用户统计")
                    stats_btn = gr.Button("查看统计", variant="secondary")
                    stats_output = gr.Markdown("点击查看用户统计信息")
                
                # 系统状态
                with gr.Group():
                    gr.Markdown("#### ⚙️ 系统状态")
                    if system.is_training_mode:
                        gr.Markdown("🟢 训练模式：开启")
                    else:
                        gr.Markdown("🔴 训练模式：关闭")
                    
                    save_model_btn = gr.Button("保存模型", variant="secondary")
                    save_output = gr.Textbox(label="保存结果", interactive=False)
        
        # 事件绑定
        msg_input.submit(
            chat_interface,
            inputs=[msg_input, chatbot, user_id_input, system_type],
            outputs=[msg_input, chatbot]
        )
        
        send_btn.click(
            chat_interface,
            inputs=[msg_input, chatbot, user_id_input, system_type],
            outputs=[msg_input, chatbot]
        )
        
        clear_btn.click(
            lambda: ([], ""),
            outputs=[chatbot, msg_input]
        )
        
        feedback_btn.click(
            feedback_interface,
            inputs=[feedback_rating, context_input],
            outputs=[feedback_output]
        )
        
        stats_btn.click(
            get_user_stats,
            inputs=[user_id_input],
            outputs=[stats_output]
        )
        
        save_model_btn.click(
            lambda: (system.save_models(), "✅ 模型已保存")[1],
            outputs=[save_output]
        )
        
        # 添加使用说明
        gr.Markdown("""
        ### 📖 使用说明
        
        **智能功能特性：**
        - 🧠 **强化学习**: 根据用户反馈不断改进回答质量
        - 👤 **用户画像**: 记住用户偏好，提供个性化服务  
        - 🎯 **意图识别**: 智能理解用户真实需求
        - 💭 **上下文记忆**: 参考历史对话提供更好的回答
        - 📊 **智能决策**: 根据情况选择最佳回答策略
        
        **操作提示：**
        1. 输入用户ID以获得个性化体验
        2. 选择合适的系统类型（通用助手/客服助手等）
        3. 对回答质量进行评分帮助系统学习
        4. 查看个人统计了解使用情况
        """)
    
    return demo

def load_enhanced_config(config_path: str) -> Dict[Any, Any]:
    """加载增强版配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ 配置文件加载成功: {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ 配置文件未找到: {config_path}")
        print("使用默认配置...")
        return get_enhanced_default_config()
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        print("使用默认配置...")
        return get_enhanced_default_config()

def get_enhanced_default_config() -> Dict[Any, Any]:
    """获取增强版默认配置"""
    return {
        'model': {
            'chat_model': '/root/conversation/local_dialoGPT',
            'embedding_model': '/root/conversation/local_model',
            'load_in_4bit': False,
            'device': 'auto',
            'generation': {
                'max_new_tokens': 512,
                'temperature': 0.7,
                'top_p': 0.9,
                'do_sample': True
            }
        },
        'knowledge': {
            'chunk_size': 500,
            'chunk_overlap': 50,
            'similarity_threshold': 0.3,
            'max_retrieved_docs': 3
        },
        'dialogue': {
            'max_history': 10,
            'session_timeout': 3600
        },
        'web': {
            'title': '智能对话机器人系统 (AI增强版)',
            'host': '0.0.0.0',
            'port': 7860,
            'share': False,
            'theme': 'soft'
        },
        'logging': {
            'level': 'INFO',
            'file': 'logs/chatbot.log'
        },
        'system_types': {
            'default': {'name': '通用助手'},
            'customer_service': {'name': '客服助手'},
            'campus_qa': {'name': '校园问答'},
            'intelligent': {'name': 'AI智能助手'}
        },
        'training': {
            'enabled': True,
            'interval': 300,  # 训练间隔（秒）
            'save_interval': 3600,  # 模型保存间隔（秒）
            'save_dir': 'models/trained',
            'batch_size': 32,
            'learning_rate': 0.001
        },
        'intelligence': {
            'use_reinforcement_learning': True,
            'use_user_profiling': True,
            'use_contextual_memory': True,
            'memory_size': 1000,
            'similarity_threshold': 0.7
        }
    }

def setup_enhanced_logging(config: Dict[Any, Any]):
    """设置增强版日志"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO').upper())
    log_file = log_config.get('file', 'logs/chatbot.log')
    
    # 确保日志目录存在
    Path(log_file).parent.mkdir(exist_ok=True)
    
    # 配置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # 文件处理器 - 普通日志
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # 文件处理器 - 训练日志
    training_handler = logging.FileHandler(log_file.replace('.log', '_training.log'))
    training_handler.setFormatter(formatter)
    training_handler.addFilter(lambda record: 'training' in record.getMessage().lower())
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(training_handler)
    root_logger.addHandler(console_handler)
    
    print(f"✅ 增强版日志系统初始化完成，日志文件: {log_file}")

def signal_handler(signum, frame, system):
    """增强版信号处理器"""
    print(f"\n🛑 收到信号 {signum}，正在优雅关闭...")
    
    # 停止训练线程
    if system:
        system.stop_training_thread()
        system.save_models()
    
    print("👋 系统已安全关闭")
    sys.exit(0)

def main():
    """增强版主函数"""
    import numpy as np  # 添加numpy导入
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="智能对话机器人系统 (AI增强版)")
    parser.add_argument("--config", default="/root/conversation/config.yaml", help="配置文件路径")
    parser.add_argument("--host", help="Web服务器主机地址")
    parser.add_argument("--port", type=int, help="Web服务器端口")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="设备类型")
    parser.add_argument("--model", help="聊天模型路径")
    parser.add_argument("--load-in-4bit", action="store_true", help="使用4bit量化")
    parser.add_argument("--knowledge-dir", default="data/knowledge", help="知识库目录")
    parser.add_argument("--create-sample", action="store_true", help="创建示例知识文件")
    parser.add_argument("--enable-training", action="store_true", help="启用训练模式")
    parser.add_argument("--disable-training", action="store_true", help="禁用训练模式")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    
    args = parser.parse_args()
    
    print("🚀 启动智能对话机器人系统 (AI增强版)")
    print("=" * 60)
    
    # 加载配置
    config = load_enhanced_config(args.config)
    
    # 命令行参数覆盖配置
    if args.host:
        config['web']['host'] = args.host
    if args.port:
        config['web']['port'] = args.port
    if args.device:
        config['model']['device'] = args.device
    if args.model:
        config['model']['chat_model'] = args.model
    if args.load_in_4bit:
        config['model']['load_in_4bit'] = True
    if args.enable_training:
        config['training']['enabled'] = True
    if args.disable_training:
        config['training']['enabled'] = False
    if args.debug:
        config['logging']['level'] = 'DEBUG'
    
    # 设置日志
    setup_enhanced_logging(config)
    logger = logging.getLogger(__name__)
    
    # 初始化系统
    system = None
    
    try:
        # 创建示例知识文件
        if args.create_sample:
            create_sample_knowledge()
        
        # 初始化增强版系统
        print("🧠 初始化AI增强系统...")
        system = EnhancedChatBotSystem(config)
        
        if not system.initialize():
            print("❌ 系统初始化失败")
            return 1
        
        # 设置信号处理
        signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, system))
        signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, system))
        
        # 加载知识库
        print("📚 加载知识库...")
        if hasattr(system.base_chatbot, 'knowledge_base'):
            load_knowledge_from_directory(system.base_chatbot, args.knowledge_dir)
        
        # 创建增强版Web界面
        print("🌐 创建AI增强Web界面...")
        demo = create_enhanced_gradio_interface(system)
        
        # 显示启动信息
        web_config = config['web']
        host = web_config.get('host', '0.0.0.0')
        port = web_config.get('port', 7860)
        
        print("=" * 60)
        print("✅ AI增强系统启动成功!")
        print(f"🌐 Web界面: http://{host}:{port}")
        if host == '0.0.0.0':
            print(f"🔗 本地访问: http://localhost:{port}")
        print(f"📊 模型: {config['model']['chat_model']}")
        print(f"🧠 智能功能: {'✅' if config.get('intelligence', {}).get('use_reinforcement_learning') else '❌'} 强化学习")
        print(f"👤 用户画像: {'✅' if config.get('intelligence', {}).get('use_user_profiling') else '❌'} 已启用")  
        print(f"💭 上下文记忆: {'✅' if config.get('intelligence', {}).get('use_contextual_memory') else '❌'} 已启用")
        print(f"🎯 训练模式: {'✅ 已启用' if config.get('training', {}).get('enabled') else '❌ 已禁用'}")
        print("=" * 60)
        
        # 启动Web服务
        logger.info("启动AI增强Web服务...")
        demo.launch(
            server_name=host,
            server_port=port,
            share=web_config.get('share', False),
            show_api=False,
            quiet=False
        )
        
    except KeyboardInterrupt:
        print("\n👋 用户中断，系统关闭")
    except Exception as e:
        print(f"❌ 系统启动失败: {e}")
        logger.exception("系统启动异常")
        return 1
    finally:
        # 清理资源
        if system:
            system.stop_training_thread()
            system.save_models()
    
    print("👋 AI增强系统已关闭")
    return 0

# 从原main.py复制的辅助函数
def load_knowledge_from_directory(chatbot, knowledge_dir: str):
    """从目录加载知识文件"""
    knowledge_path = Path(knowledge_dir)
    if not knowledge_path.exists():
        print(f"⚠️ 知识库目录不存在: {knowledge_dir}")
        return
    
    loaded_files = 0
    supported_extensions = {'.txt', '.md', '.json'}
    
    for file_path in knowledge_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                if hasattr(chatbot, 'add_knowledge_from_file'):
                    chatbot.add_knowledge_from_file(str(file_path))
                    loaded_files += 1
                    print(f"  📄 {file_path.name}")
            except Exception as e:
                print(f"  ❌ 加载失败 {file_path.name}: {e}")
    
    print(f"✅ 从 {knowledge_dir} 加载了 {loaded_files} 个知识文件")

def create_sample_knowledge():
    """创建示例知识文件（复用原函数）"""
    knowledge_dir = Path("data/knowledge")
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    
    sample_files = {
        "ai_enhanced_features.txt": """
AI增强功能说明：

强化学习系统：
- 根据用户反馈自动优化回答策略
- 支持多种奖励机制和学习算法
- 实时调整对话风格和内容深度

用户画像系统：  
- 记录用户偏好和交互历史
- 个性化回答风格和内容推荐
- 智能预测用户需求

上下文记忆：
- 长期记住用户对话历史
- 跨会话信息关联和引用
- 智能相似对话检索

意图识别：
- 自动识别用户真实意图
- 支持多层次意图分析
- 动态调整回答策略

训练模式：
- 后台持续学习用户反馈
- 定期保存和更新模型
- 支持增量学习和模型优化
        """
    }
    
    # 添加原有的示例文件（简化版）
    sample_files.update({
        "customer_service.txt": "客服相关信息...",
        "campus_info.txt": "校园信息...", 
        "faq.txt": "常见问题..."
    })
    
    created_count = 0
    for filename, content in sample_files.items():
        file_path = knowledge_dir / filename
        if not file_path.exists():
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content.strip())
            created_count += 1
    
    if created_count > 0:
        print(f"✅ 创建了 {created_count} 个示例知识文件")

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
