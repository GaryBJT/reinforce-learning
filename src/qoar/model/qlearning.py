import numpy as np
from collections import defaultdict

class QOARQLearning:
    def __init__(self, alpha=0.8, gamma=0.9, a=0.5, b=0.0, c=0.5,
                 buffer_size=100, batch_size=50, replay_interval=10):
        """ 
        参数说明:
            alpha: 学习率 (默认0.8)
            gamma: 折扣因子 (默认0.9)
            a,b,c: 链路质量权重，需满足a+b+c=1
            buffer_size: 经验回放缓存容量 (默认1000)
            batch_size: 批量回放样本数 (默认50)
            replay_interval: 回放触发间隔 (默认10次更新)
        """
        self.set_parameters(alpha, gamma, a, b, c)
        self.q_table = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.link_quality = defaultdict(lambda: defaultdict(float))
        self.experience_buffer = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_interval = replay_interval
        self.update_counter = 0

    def set_parameters(self, alpha, gamma, a, b, c):
        """动态参数配置"""
        if abs(a + b + c - 1.0) > 1e-6:
            raise ValueError("链路质量权重之和必须为1")
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.gamma = gamma

    # region 核心Q学习方法
    def get_q_value(self, current_node, dest_node, next_hop):
        """获取Q值"""
        return self.q_table[current_node][dest_node].get(next_hop, 0.0)

    def set_q_value(self, current_node, dest_node, next_hop, value):
        """带自动初始化的Q值设置"""
        self.q_table[current_node][dest_node][next_hop] = value

    def get_max_q(self, node, dest_node):
        """获取节点到目标的最大Q值"""
        return max(self.q_table[node][dest_node].values(), default=0.0)

    def get_best_next_hop(self, current_node, dest_node):
        """确定性路由决策"""
        q_values = self.q_table[current_node][dest_node]
        if not q_values:
            return ""
        return max(q_values.items(), key=lambda x: x[1])[0]
    # endregion

    # region 探索机制（通过update_lq实现）
    def update_lq(self, sf, ef, bf, current_node, next_hop):
        """
        主动链路探索机制
        功能：
        1. 更新当前链路质量
        2. 触发所有相关路径的Q值更新
        3. 实现网络状态探索
        """
        lq = self.a * sf + self.b * ef + self.c * bf
        self.link_quality[current_node][next_hop] = lq
        
        # 遍历所有受影响的目标节点
        for dest_node in list(self.q_table[current_node].keys()):
            if next_hop in self.q_table[current_node][dest_node]:
                # 计算新的Q值
                current_q = self.get_q_value(current_node, dest_node, next_hop)
                next_max_q = self.get_max_q(next_hop, dest_node)
                
                # 特殊处理直达情况
                if next_hop == dest_node:
                    new_q = (1 - self.alpha)*current_q + self.alpha*lq*1
                else:
                    new_q = (1 - self.alpha)*current_q + self.alpha*lq*(self.gamma*next_max_q)
                
                self.set_q_value(current_node, dest_node, next_hop, new_q)
        
        return lq
    # endregion

    # region 经验回放系统
    def store_experience(self, experience):
        """存储单条经验"""
        if len(self.experience_buffer) >= self.buffer_size:
            self.experience_buffer.pop(0)
        self.experience_buffer.append(experience)

    def experience_replay(self):
        """批量经验回放"""
        if len(self.experience_buffer) < self.batch_size:
            return
        
        batch = np.random.choice(self.experience_buffer, self.batch_size, replace=False)
        for exp in batch:
            self._q_update(
                exp['current_node'], exp['next_hop'], exp['dest_node'],
                exp['sf'], exp['ef'], exp['bf'], exp['reward']
            )

    def _q_update(self, current_node, next_hop, dest_node, sf, ef, bf, reward):
        """内部统一更新逻辑"""
        lq = self.a * sf + self.b * ef + self.c * bf
        current_q = self.get_q_value(current_node, dest_node, next_hop)
        
        # 计算TD目标
        if next_hop == dest_node:
            td_target = lq * 1
        else:
            next_max_q = self.get_max_q(next_hop, dest_node)
            td_target = lq * (reward + self.gamma * next_max_q)
        
        new_q = (1 - self.alpha) * current_q + self.alpha * td_target
        self.set_q_value(current_node, dest_node, next_hop, new_q)
        return new_q
    # endregion

    # region 外部调用接口
    def update_q_value(self, sf, ef, bf, current_node, next_hop, dest_node, reward):
        """数据包触发的学习更新"""
        # 存储经验
        self.store_experience({
            'current_node': str(current_node),
            'next_hop': str(next_hop),
            'dest_node': str(dest_node),
            'sf': float(sf),
            'ef': float(ef),
            'bf': float(bf),
            'reward': float(reward)
        })
        
        # 执行实时更新
        new_q = self._q_update(current_node, next_hop, dest_node, sf, ef, bf, reward)
        
        # 触发定期回放
        self.update_counter += 1
        if self.update_counter % self.replay_interval == 0:
            self.experience_replay()
            self.update_counter = 0
            
        return new_q
    # endregion

# 全局单例
qlearning = QOARQLearning()

# region C++调用接口
def update_q_value(sf, ef, bf, current_node, next_hop, dest_node, reward):
    return qlearning.update_q_value(float(sf), float(ef), float(bf),
                                   str(current_node), str(next_hop), 
                                   str(dest_node), float(reward))

def update_lq(sf, ef, bf, current_node, next_hop):
    return qlearning.update_lq(float(sf), float(ef), float(bf),
                             str(current_node), str(next_hop))

def get_best_next_hop(current_node, dest_node):
    return qlearning.get_best_next_hop(str(current_node), str(dest_node))

def set_qlearning_params(alpha, gamma, a, b, c):
    try:
        qlearning.set_parameters(float(alpha), float(gamma),
                                float(a), float(b), float(c))
        print(f"参数更新成功: α={alpha}, γ={gamma}, a={a}, b={b}, c={c}")
    except ValueError as e:
        print(f"参数错误: {str(e)}")
        qlearning.set_parameters(0.8, 0.9, 0.5, 0.0, 0.5)  # 恢复安全默认值
# endregion