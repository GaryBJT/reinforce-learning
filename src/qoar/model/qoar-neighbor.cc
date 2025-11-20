/*
 * Copyright (c) 2009 IITP RAS
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Based on
 *      NS-2 QOAR model developed by the CMU/MONARCH group and optimized and
 *      tuned by Samir Das and Mahesh Marina, University of Cincinnati;
 *
 *      QOAR-UU implementation by Erik Nordström of Uppsala University
 *      https://web.archive.org/web/20100527072022/http://core.it.uu.se/core/index.php/QOAR-UU
 *
 * Authors: Elena Buchatskaia <borovkovaes@iitp.ru>
 *          Pavel Boyko <boyko@iitp.ru>
 */

#include "qoar-neighbor.h"

#include "ns3/log.h"
#include "ns3/wifi-mac-header.h"

#include <algorithm>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("QoarNeighbors");

namespace qoar
{
Neighbors::Neighbors(Time delay)
    : m_ntimer(Timer::CANCEL_ON_DESTROY)
{
    m_ntimer.SetDelay(delay);
    m_ntimer.SetFunction(&Neighbors::Purge, this);
    m_txErrorCallback = MakeCallback(&Neighbors::ProcessTxError, this);
}

bool
Neighbors::IsNeighbor(Ipv4Address addr)
{
    Purge();
    for (std::vector<Neighbor>::const_iterator i = m_nb.begin(); i != m_nb.end(); ++i)
    {
        if (i->m_neighborAddress == addr)
        {
            return true;
        }
    }
    return false;
}

Time
Neighbors::GetExpireTime(Ipv4Address addr)
{
    Purge();
    for (std::vector<Neighbor>::const_iterator i = m_nb.begin(); i != m_nb.end(); ++i)
    {
        if (i->m_neighborAddress == addr)
        {
            return (i->m_expireTime - Simulator::Now());
        }
    }
    return Seconds(0);
}

double
Neighbors::GetSf(Ipv4Address addr)
{
    Purge();
    for (std::vector<Neighbor>::const_iterator i = m_nb.begin(); i != m_nb.end(); ++i)
    {
        if (i->m_neighborAddress == addr)
        {
            return i->m_sf;
        }
    }
    return -1;  // 表示未找到该邻居
}
void
Neighbors::UpdateSf(Ipv4Address addr, double currentDistance)
{
    const double Rd = 250.0; // 无人机最大传输范围
    const double alpha = 0.7; // 平滑因子
for (std::vector<Neighbor>::iterator i = m_nb.begin(); i != m_nb.end(); ++i)
    {
        if (i->m_neighborAddress == addr)
        {
            // 计算时间间隔
            Time currentTime = Simulator::Now();
            Time timeDiff = currentTime - i->m_prevUpdateTime;
            
            // 计算相对速度（距离变化率）
            double velocity = 0.0;
            if (timeDiff.GetSeconds() > 0)
            {
                velocity = (currentDistance - i->m_prevDistance) / timeDiff.GetSeconds();
            }
            // 计算Δd（需处理首次更新的情况）
            double delta_d = 0.0;
            if (i->m_prevDistance > 0) { // 非首次更新
                delta_d = fabs(currentDistance - i->m_prevDistance);
            }

            // 计算当前 SF 值
            double new_sf;
            if (delta_d >= Rd) {
                new_sf = 0.0;
            } else {
                new_sf = 1.0 - (delta_d / Rd);
            }

            // 应用指数移动平均
            i->m_sf = alpha * new_sf + (1 - alpha) * i->m_sf;

            // 限制 SF 范围 [0.0, 1.0]
            i->m_sf = std::clamp(i->m_sf, 0.0, 1.0);
            
            // 保存当前值作为下次计算的前值
            i->m_prevDistance = currentDistance;
            i->m_prevUpdateTime = currentTime;
            return;
        }
    }
 
}
// 判断节点是否远离
bool
Neighbors::IsMovingAway(Ipv4Address addr, double currentDistance)
{
    Purge();
    for (std::vector<Neighbor>::const_iterator i = m_nb.begin(); i != m_nb.end(); ++i)
    {
        if (i->m_neighborAddress == addr && i->m_prevDistance > 0)
        {
            return currentDistance > i->m_prevDistance;
        }
    }
    return false;  // 默认假设不是远离
}

// 计算相对速度（返回绝对值）
double
Neighbors::GetRelativeVelocity(Ipv4Address addr, double currentDistance)
{
    for (std::vector<Neighbor>::iterator i = m_nb.begin(); i != m_nb.end(); ++i)
    {
        if (i->m_neighborAddress == addr)
        {
            Time currentTime = Simulator::Now();
            Time timeDiff = currentTime - i->m_prevUpdateTime;
            
            if (timeDiff.GetSeconds() > 0 && i->m_prevDistance > 0)
            {
                // 计算速度的绝对值
                return std::abs((currentDistance - i->m_prevDistance) / timeDiff.GetSeconds());
            }
        }
    }
    return 0.1; // 返回一个默认的小速度值，避免除零错误
}

// 计算自适应Hello间隔
double
Neighbors::CalculateAdaptiveHelloInterval(double Rd, Ipv4Address currentNode)
{
    Purge();
    double minInterval = std::numeric_limits<double>::max();
    bool hasValidNeighbor = false;
    
    // 获取活跃邻居列表
    std::vector<Ipv4Address> activeNeighbors = GetActiveNeighbors(currentNode);
    
    for (std::vector<Ipv4Address>::const_iterator i = activeNeighbors.begin(); i != activeNeighbors.end(); ++i)
    {
        Ipv4Address neighbor = *i;
        
        // 获取当前距离
        double currentDistance = GetCurrentDistance(neighbor);
        
        if (currentDistance <= 0)
            continue;  // 无效距离
            
        // 计算相对速度
        double relativeVelocity = GetRelativeVelocity(neighbor, currentDistance);
        
        if (relativeVelocity < 0.1)
            relativeVelocity = 0.1;  // 避免除零错误
            
        hasValidNeighbor = true;
        
        double interval;
        if (IsMovingAway(neighbor, currentDistance))
        {
            // 远离情况：[Rd-当前距离]/相对速度
            interval = std::max(0.1, (Rd - currentDistance) / relativeVelocity);
        }
        else
        {
            // 靠近情况：[Rd+当前距离]/相对速度
            interval = (Rd + currentDistance) / relativeVelocity;
        }
        
        // 更新最小间隔
        if (interval > 0 && interval < minInterval)
        {
            minInterval = interval;
        }
    }
    
    // 如果没有有效邻居或计算出的间隔不合理，返回默认值
    if (!hasValidNeighbor || minInterval <= 0 || minInterval == std::numeric_limits<double>::max())
    {
        return 2.0;  // 默认2秒
    }
    
    // 应用系数0.8
    return 0.8 * minInterval;
}

// 获取当前距离
double
Neighbors::GetCurrentDistance(Ipv4Address addr)
{
    for (std::vector<Neighbor>::const_iterator i = m_nb.begin(); i != m_nb.end(); ++i)
    {
        if (i->m_neighborAddress == addr)
        {
            return i->m_prevDistance;
        }
    }
    return -1;  // 未找到邻居
}


double
Neighbors::GetEf(Ipv4Address addr)
{
    Purge();
    for (std::vector<Neighbor>::const_iterator i = m_nb.begin(); i != m_nb.end(); ++i)
    {
        if (i->m_neighborAddress == addr)
        {
            return i->m_ef;
        }
    }
    return -1;  // 表示未找到该邻居
}
void 
Neighbors::UpdateEf(Ipv4Address addr,double energy)
{
    for (auto& neighbor : m_nb) {
        if (neighbor.m_neighborAddress == addr) {
   double emax = 800.0; // 最大能量值
   neighbor.m_ef = energy / emax; // 计算能量因子
   return;}}
}
double 
Neighbors::GetBf(Ipv4Address addr)
{
    Purge();
    for (std::vector<Neighbor>::const_iterator i = m_nb.begin(); i != m_nb.end(); ++i)
    {
        if (i->m_neighborAddress == addr)
        {
            return i->m_bf;
        }
    }
    return -1;  // 表示未找到该邻居
}
void 
Neighbors::UpdateBf(Ipv4Address addr, double citr)
{
    for (std::vector<Neighbor>::iterator i = m_nb.begin(); i != m_nb.end(); ++i)
    {
        if (i->m_neighborAddress == addr)
        {
            // 应用加权指数移动平均
            const double alpha = 0.3; // 平滑因子（可根据需要调整）
            
            // 如果没有之前的BF值，用当前CITR初始化
            if (i->m_bf < 0) {
                i->m_bf = citr;
            } else {
                // 应用WEMA公式
                i->m_bf = alpha * citr + (1 - alpha) * i->m_bf;
            }
            
            // 确保BF在[0.0, 1.0]范围内
            i->m_bf = std::max(0.0, std::min(1.0, i->m_bf));
            
            return;
        }
    }
}
double 
Neighbors::Getdf(Ipv4Address addr)
{
    Purge();
    for (std::vector<Neighbor>::const_iterator i = m_nb.begin(); i != m_nb.end(); ++i)
    {
        if (i->m_neighborAddress == addr)
        {
            return i->m_df;
        }
    }
    return -1;  // 表示未找到该邻居
}
void 
Neighbors::Updatedf(Ipv4Address addr,double delay)
{
    for (auto& neighbor : m_nb) {
        if (neighbor.m_neighborAddress == addr) {

   neighbor.m_df = delay; 
   return;
    }
    }
}
//两个Q值，一个是rreq的，一个是rrep的
double
Neighbors::GetQ(Ipv4Address addr)
{
    for (std::vector<Neighbor>::iterator i = m_nb.begin(); i != m_nb.end(); ++i)
    {
        if (i->m_neighborAddress == addr)
        {
            return i->m_q;
        }
 }
 return -1;
}

double
Neighbors::GetQf(Ipv4Address addr)
{
    for (std::vector<Neighbor>::iterator i = m_nb.begin(); i != m_nb.end(); ++i)
    {
        if (i->m_neighborAddress == addr)
        {
            return i->m_qf;
        }
   }
   return -1;
}
void
Neighbors::SetQ(Ipv4Address addr,double q)
{
    for (std::vector<Neighbor>::iterator i = m_nb.begin(); i != m_nb.end(); ++i)
    {
        if (i->m_neighborAddress == addr)
        {
            i->m_q = q; // 更新 Q 值
            Neighbors::UpdateMaxQ();
            return;
        }
    }
}

void
Neighbors::SetQf(Ipv4Address addr,double q)
{
    for (std::vector<Neighbor>::iterator i = m_nb.begin(); i != m_nb.end(); ++i)
    {
        if (i->m_neighborAddress == addr)
        {
            i->m_qf = q; // 更新 Q 值
            Neighbors::UpdateMaxQf();
            return;
        }
    }
}

//现在用的
void 
Neighbors::SetLq(Ipv4Address addr,double lq)
{
    for (std::vector<Neighbor>::iterator i = m_nb.begin(); i != m_nb.end(); ++i)
    {
        if (i->m_neighborAddress == addr)
        {
            i->m_lq = lq; // 更新 Q 值
            return;
        }
    }
}
double
Neighbors::GetLq(Ipv4Address addr)
{
    for (std::vector<Neighbor>::iterator i = m_nb.begin(); i != m_nb.end(); ++i)
    {
        if (i->m_neighborAddress == addr)
        {
            return i->m_lq;
        }
    }
    return -1;
}

void
Neighbors::UpdateMaxQf()
{
    Purge(); // Clean up expired neighbors first
    
    double maxQf = -1.0; // Initialize with a value lower than valid Q values
    Ipv4Address maxQfHop; // Will store the address of neighbor with highest Q
    
    // Iterate through all neighbors
    for (std::vector<Neighbor>::const_iterator i = m_nb.begin(); i != m_nb.end(); ++i)
    {
        // If this neighbor has a higher Q value than current maximum
        if (i->m_q > maxQf)
        {
            maxQf = i->m_q; // Update max Q value
            maxQfHop = i->m_neighborAddress; // Store the neighbor address
        }
    }
    
    // Update member variables with the results
    m_maxqf = maxQf;
    m_maxqfhop = maxQfHop;
    return;
}
void
Neighbors::UpdateMaxQ()
{
    Purge(); // Clean up expired neighbors first
    
    double maxQ = -1.0; // Initialize with a value lower than valid Q values
    Ipv4Address maxQHop; // Will store the address of neighbor with highest Q
    
    // Iterate through all neighbors
    for (std::vector<Neighbor>::const_iterator i = m_nb.begin(); i != m_nb.end(); ++i)
    {
        // If this neighbor has a higher Q value than current maximum
        if (i->m_q > maxQ)
        {
            maxQ = i->m_q; // Update max Q value
            maxQHop = i->m_neighborAddress; // Store the neighbor address
        }
    }
    
    // Update member variables with the results
    m_maxq = maxQ;
    return;
}
void
Neighbors::Update(Ipv4Address addr, Time expire)
{
    for (std::vector<Neighbor>::iterator i = m_nb.begin(); i != m_nb.end(); ++i)
    {
        if (i->m_neighborAddress == addr)
        {
            i->m_expireTime = std::max(expire + Simulator::Now(), i->m_expireTime);
            if (i->m_hardwareAddress == Mac48Address())
            {
                i->m_hardwareAddress = LookupMacAddress(i->m_neighborAddress);
            }
            // // 调用 UpdateSf 更新 SF 值
            // UpdateSf(addr, currentDistance);
            return;
        }
    }

    NS_LOG_LOGIC("Open link to " << addr);
    Neighbor neighbor(addr, LookupMacAddress(addr), expire + Simulator::Now());
    m_nb.push_back(neighbor);
    Purge();
}

/**
 * \brief CloseNeighbor structure
 */
struct CloseNeighbor
{
    /**
     * Check if the entry is expired
     *
     * \param nb Neighbors::Neighbor entry
     * \return true if expired, false otherwise
     */
    bool operator()(const Neighbors::Neighbor& nb) const
    {
        return ((nb.m_expireTime < Simulator::Now()) || nb.close);
    }
};

void
Neighbors::Purge()
{
    if (m_nb.empty())
    {
        return;
    }

    CloseNeighbor pred;
    if (!m_handleLinkFailure.IsNull())
    {
        for (std::vector<Neighbor>::iterator j = m_nb.begin(); j != m_nb.end(); ++j)
        {
            if (pred(*j))
            {
                NS_LOG_LOGIC("Close link to " << j->m_neighborAddress);

                m_handleLinkFailure(j->m_neighborAddress);
            }
        }
    }
    m_nb.erase(std::remove_if(m_nb.begin(), m_nb.end(), pred), m_nb.end());
    m_ntimer.Cancel();
    m_ntimer.Schedule();
}

void
Neighbors::ScheduleTimer()
{
    m_ntimer.Cancel();
    m_ntimer.Schedule();
}

void
Neighbors::AddArpCache(Ptr<ArpCache> a)
{
    m_arp.push_back(a);
}

void
Neighbors::DelArpCache(Ptr<ArpCache> a)
{
    m_arp.erase(std::remove(m_arp.begin(), m_arp.end(), a), m_arp.end());
}

Mac48Address
Neighbors::LookupMacAddress(Ipv4Address addr)
{
    Mac48Address hwaddr;
    for (std::vector<Ptr<ArpCache>>::const_iterator i = m_arp.begin(); i != m_arp.end(); ++i)
    {
        ArpCache::Entry* entry = (*i)->Lookup(addr);
        if (entry != nullptr && (entry->IsAlive() || entry->IsPermanent()) && !entry->IsExpired())
        {
            hwaddr = Mac48Address::ConvertFrom(entry->GetMacAddress());
            break;
        }
    }
    return hwaddr;
}

void
Neighbors::ProcessTxError(const WifiMacHeader& hdr)
{
    Mac48Address addr = hdr.GetAddr1();

    for (std::vector<Neighbor>::iterator i = m_nb.begin(); i != m_nb.end(); ++i)
    {
        if (i->m_hardwareAddress == addr)
        {
            i->close = true;
        }
    }
    Purge();
}
bool 
Neighbors::IsActive(const Neighbor& nb)
{
    return (nb.m_expireTime > Simulator::Now()) && !nb.close;
}

std::vector<Ipv4Address>
Neighbors::GetActiveNeighbors(Ipv4Address current)
{
    std::vector<Ipv4Address> activeList;
    
    // 遍历所有邻居条目
    for (std::vector<Neighbor>::const_iterator i = m_nb.begin(); i != m_nb.end(); ++i)
    {
        // 检查是否满足活跃条件
        if (IsActive(*i))
        {
            activeList.push_back(i->m_neighborAddress);

        }
        else{
            std::ostringstream oss;
                oss << current;
            std::string cur = oss.str();
            oss << i->m_neighborAddress;
            std::string neb = oss.str();
            m_qLearning.updatelq(0.0,0.0,0.0,cur,neb);
        }
    }
    
    return activeList;
}
} // namespace qoar
} // namespace ns3
