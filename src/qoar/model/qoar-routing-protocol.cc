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
#define NS_LOG_APPEND_CONTEXT                                                                      \
    if (m_ipv4)                                                                                    \
    {                                                                                              \
        std::clog << "[mynode " << m_ipv4->GetObject<Node>()->GetId() << "] ";                       \
    }

#include "qoar-routing-protocol.h"

#include "ns3/adhoc-wifi-mac.h"
#include "ns3/boolean.h"
#include "ns3/inet-socket-address.h"
#include "ns3/log.h"
#include "ns3/pointer.h"
#include "ns3/random-variable-stream.h"
#include "ns3/string.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/udp-header.h"
#include "ns3/udp-l4-protocol.h"
#include "ns3/udp-socket-factory.h"
#include "ns3/wifi-mpdu.h"
#include "ns3/wifi-net-device.h"
// #include "ns3/mobility-module.h"
#include "ns3/mobility-model.h"  // 必须包含移动模型头文件
#include "citr-adhoc-wifi-mac.h" // 包含CITR MAC头文件

#include <algorithm>
#include <limits>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("QoarRoutingProtocol");

namespace qoar
{
NS_OBJECT_ENSURE_REGISTERED(RoutingProtocol);

/// UDP Port for QOAR control traffic
const uint32_t RoutingProtocol::Qoar_PORT = 654;

/**
 * \ingroup qoar
 * \brief Tag used by QOAR implementation
 */
class DeferredRouteOutputTag : public Tag
{
  public:
    /**
     * \brief Constructor
     * \param o the output interface
     */
    DeferredRouteOutputTag(int32_t o = -1)
        : Tag(),
          m_oif(o)
    {
    }

    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::qoar::DeferredRouteOutputTag")
                                .SetParent<Tag>()
                                .SetGroupName("Qoar")
                                .AddConstructor<DeferredRouteOutputTag>();
        return tid;
    }

    TypeId GetInstanceTypeId() const override
    {
        return GetTypeId();
    }

    /**
     * \brief Get the output interface
     * \return the output interface
     */
    int32_t GetInterface() const
    {
        return m_oif;
    }

    /**
     * \brief Set the output interface
     * \param oif the output interface
     */
    void SetInterface(int32_t oif)
    {
        m_oif = oif;
    }

    uint32_t GetSerializedSize() const override
    {
        return sizeof(int32_t);
    }

    void Serialize(TagBuffer i) const override
    {
        i.WriteU32(m_oif);
    }

    void Deserialize(TagBuffer i) override
    {
        m_oif = i.ReadU32();
    }

    void Print(std::ostream& os) const override
    {
        os << "DeferredRouteOutputTag: output interface = " << m_oif;
    }

  private:
    /// Positive if output device is fixed in RouteOutput
    int32_t m_oif;
};

NS_OBJECT_ENSURE_REGISTERED(DeferredRouteOutputTag);

//-----------------------------------------------------------------------------
RoutingProtocol::RoutingProtocol()
    : m_rreqRetries(2),
      m_ttlStart(1),
      m_ttlIncrement(2),
      m_ttlThreshold(7),
      m_timeoutBuffer(2),
      m_rreqRateLimit(10),
      m_rerrRateLimit(10),
      m_activeRouteTimeout(Seconds(3)),
      m_netDiameter(35),
      m_nodeTraversalTime(MilliSeconds(40)),
      m_netTraversalTime(Time((2 * m_netDiameter) * m_nodeTraversalTime)),
      m_pathDiscoveryTime(Time(2 * m_netTraversalTime)),
      m_myRouteTimeout(Time(2 * std::max(m_pathDiscoveryTime, m_activeRouteTimeout))),
      m_helloInterval(Seconds(1)),
      m_allowedHelloLoss(2),
      m_deletePeriod(Time(5 * std::max(m_activeRouteTimeout, m_helloInterval))),
      m_nextHopWait(m_nodeTraversalTime + MilliSeconds(10)),
      m_blackListTimeout(Time(m_rreqRetries * m_netTraversalTime)),
      m_maxQueueLen(64),
      m_maxQueueTime(Seconds(30)),
      m_destinationOnly(false),
      m_gratuitousReply(true),
      m_enableHello(false),
      m_routingTable(m_deletePeriod),
      m_queue(m_maxQueueLen, m_maxQueueTime),
      m_requestId(0),
      m_seqNo(0),
      m_rreqIdCache(m_pathDiscoveryTime),
      m_dpd(m_pathDiscoveryTime),
      m_nb(m_helloInterval),
      m_rreqCount(0),
      m_rerrCount(0),
      m_htimer(Timer::CANCEL_ON_DESTROY),
      m_rreqRateLimitTimer(Timer::CANCEL_ON_DESTROY),
      m_rerrRateLimitTimer(Timer::CANCEL_ON_DESTROY),
      m_lastBcastTime(Seconds(0)),
      m_energySource(nullptr),
      m_enableAdaptiveHello(true),
      m_minHelloInterval(0.25),
      m_maxHelloInterval(5.0),
      m_radioRange(250.0),
      m_nodeCount(100),        // 初始化节点数量
      m_maxSpeed(40)           // 初始化最大速度
{
    m_nb.SetCallback(MakeCallback(&RoutingProtocol::SendRerrWhenBreaksLinkToNextHop, this));
}

TypeId
RoutingProtocol::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::qoar::RoutingProtocol")
            .SetParent<Ipv4RoutingProtocol>()
            .SetGroupName("Qoar")
            .AddConstructor<RoutingProtocol>()
            .AddAttribute("HelloInterval",
                          "HELLO messages emission interval.",
                          TimeValue(Seconds(1)),
                          MakeTimeAccessor(&RoutingProtocol::m_helloInterval),
                          MakeTimeChecker())
            .AddAttribute("TtlStart",
                          "Initial TTL value for RREQ.",
                          UintegerValue(1),
                          MakeUintegerAccessor(&RoutingProtocol::m_ttlStart),
                          MakeUintegerChecker<uint16_t>())
            .AddAttribute("TtlIncrement",
                          "TTL increment for each attempt using the expanding ring search for RREQ "
                          "dissemination.",
                          UintegerValue(2),
                          MakeUintegerAccessor(&RoutingProtocol::m_ttlIncrement),
                          MakeUintegerChecker<uint16_t>())
            .AddAttribute("TtlThreshold",
                          "Maximum TTL value for expanding ring search, TTL = NetDiameter is used "
                          "beyond this value.",
                          UintegerValue(7),
                          MakeUintegerAccessor(&RoutingProtocol::m_ttlThreshold),
                          MakeUintegerChecker<uint16_t>())
            .AddAttribute("TimeoutBuffer",
                          "Provide a buffer for the timeout.",
                          UintegerValue(2),
                          MakeUintegerAccessor(&RoutingProtocol::m_timeoutBuffer),
                          MakeUintegerChecker<uint16_t>())
            .AddAttribute("RreqRetries",
                          "Maximum number of retransmissions of RREQ to discover a route",
                          UintegerValue(2),
                          MakeUintegerAccessor(&RoutingProtocol::m_rreqRetries),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("RreqRateLimit",
                          "Maximum number of RREQ per second.",
                          UintegerValue(10),
                          MakeUintegerAccessor(&RoutingProtocol::m_rreqRateLimit),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("RerrRateLimit",
                          "Maximum number of RERR per second.",
                          UintegerValue(10),
                          MakeUintegerAccessor(&RoutingProtocol::m_rerrRateLimit),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("NodeTraversalTime",
                          "Conservative estimate of the average one hop traversal time for packets "
                          "and should include "
                          "queuing delays, interrupt processing times and transfer times.",
                          TimeValue(MilliSeconds(40)),
                          MakeTimeAccessor(&RoutingProtocol::m_nodeTraversalTime),
                          MakeTimeChecker())
            .AddAttribute(
                "NextHopWait",
                "Period of our waiting for the neighbour's RREP_ACK = 10 ms + NodeTraversalTime",
                TimeValue(MilliSeconds(50)),
                MakeTimeAccessor(&RoutingProtocol::m_nextHopWait),
                MakeTimeChecker())
            .AddAttribute("ActiveRouteTimeout",
                          "Period of time during which the route is considered to be valid",
                          TimeValue(Seconds(3)),
                          MakeTimeAccessor(&RoutingProtocol::m_activeRouteTimeout),
                          MakeTimeChecker())
            .AddAttribute("MyRouteTimeout",
                          "Value of lifetime field in RREP generating by this node = 2 * "
                          "max(ActiveRouteTimeout, PathDiscoveryTime)",
                          TimeValue(Seconds(11.2)),
                          MakeTimeAccessor(&RoutingProtocol::m_myRouteTimeout),
                          MakeTimeChecker())
            .AddAttribute("BlackListTimeout",
                          "Time for which the node is put into the blacklist = RreqRetries * "
                          "NetTraversalTime",
                          TimeValue(Seconds(5.6)),
                          MakeTimeAccessor(&RoutingProtocol::m_blackListTimeout),
                          MakeTimeChecker())
            .AddAttribute("DeletePeriod",
                          "DeletePeriod is intended to provide an upper bound on the time for "
                          "which an upstream node A "
                          "can have a neighbor B as an active next hop for destination D, while B "
                          "has invalidated the route to D."
                          " = 5 * max (HelloInterval, ActiveRouteTimeout)",
                          TimeValue(Seconds(15)),
                          MakeTimeAccessor(&RoutingProtocol::m_deletePeriod),
                          MakeTimeChecker())
            .AddAttribute("NetDiameter",
                          "Net diameter measures the maximum possible number of hops between two "
                          "nodes in the network",
                          UintegerValue(35),
                          MakeUintegerAccessor(&RoutingProtocol::m_netDiameter),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute(
                "NetTraversalTime",
                "Estimate of the average net traversal time = 2 * NodeTraversalTime * NetDiameter",
                TimeValue(Seconds(2.8)),
                MakeTimeAccessor(&RoutingProtocol::m_netTraversalTime),
                MakeTimeChecker())
            .AddAttribute(
                "PathDiscoveryTime",
                "Estimate of maximum time needed to find route in network = 2 * NetTraversalTime",
                TimeValue(Seconds(5.6)),
                MakeTimeAccessor(&RoutingProtocol::m_pathDiscoveryTime),
                MakeTimeChecker())
            .AddAttribute("MaxQueueLen",
                          "Maximum number of packets that we allow a routing protocol to buffer.",
                          UintegerValue(64),
                          MakeUintegerAccessor(&RoutingProtocol::SetMaxQueueLen,
                                               &RoutingProtocol::GetMaxQueueLen),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("MaxQueueTime",
                          "Maximum time packets can be queued (in seconds)",
                          TimeValue(Seconds(30)),
                          MakeTimeAccessor(&RoutingProtocol::SetMaxQueueTime,
                                           &RoutingProtocol::GetMaxQueueTime),
                          MakeTimeChecker())
            .AddAttribute("AllowedHelloLoss",
                          "Number of hello messages which may be loss for valid link.",
                          UintegerValue(2),
                          MakeUintegerAccessor(&RoutingProtocol::m_allowedHelloLoss),
                          MakeUintegerChecker<uint16_t>())
            .AddAttribute("GratuitousReply",
                          "Indicates whether a gratuitous RREP should be unicast to the node "
                          "originated route discovery.",
                          BooleanValue(true),
                          MakeBooleanAccessor(&RoutingProtocol::SetGratuitousReplyFlag,
                                              &RoutingProtocol::GetGratuitousReplyFlag),
                          MakeBooleanChecker())
            .AddAttribute("DestinationOnly",
                          "Indicates only the destination may respond to this RREQ.",
                          BooleanValue(false),
                          MakeBooleanAccessor(&RoutingProtocol::SetDestinationOnlyFlag,
                                              &RoutingProtocol::GetDestinationOnlyFlag),
                          MakeBooleanChecker())
            .AddAttribute("EnableHello",
                          "Indicates whether a hello messages enable.",
                          BooleanValue(true),
                          MakeBooleanAccessor(&RoutingProtocol::SetHelloEnable,
                                              &RoutingProtocol::GetHelloEnable),
                          MakeBooleanChecker())
            .AddAttribute("EnableBroadcast",
                          "Indicates whether a broadcast data packets forwarding enable.",
                          BooleanValue(true),
                          MakeBooleanAccessor(&RoutingProtocol::SetBroadcastEnable,
                                              &RoutingProtocol::GetBroadcastEnable),
                          MakeBooleanChecker())
            .AddAttribute("UniformRv",
                          "Access to the underlying UniformRandomVariable",
                          StringValue("ns3::UniformRandomVariable"),
                          MakePointerAccessor(&RoutingProtocol::m_uniformRandomVariable),
                          MakePointerChecker<UniformRandomVariable>())
              .AddAttribute("Alpha",
                            "Q-Learning learning rate",
                            DoubleValue(0.5),
                            MakeDoubleAccessor(&RoutingProtocol::m_alpha),
                            MakeDoubleChecker<double>())
              .AddAttribute("Gamma",
                            "Q-Learning discount factor",
                            DoubleValue(0.9),
                            MakeDoubleAccessor(&RoutingProtocol::m_gamma),
                            MakeDoubleChecker<double>())
              .AddAttribute("A",
                            "Link quality weight parameter A",
                            DoubleValue(0.4),
                            MakeDoubleAccessor(&RoutingProtocol::m_a),
                            MakeDoubleChecker<double>())
              .AddAttribute("B",
                            "Link quality weight parameter B",
                            DoubleValue(0.2),
                            MakeDoubleAccessor(&RoutingProtocol::m_b),
                            MakeDoubleChecker<double>())
              .AddAttribute("C",
                            "Link quality weight parameter C",
                            DoubleValue(0.4),
                            MakeDoubleAccessor(&RoutingProtocol::m_c),
                            MakeDoubleChecker<double>())
            .AddAttribute("EnableAdaptiveHello",
                         "Enable adaptive hello interval based on relative velocity",
                         BooleanValue(true),
                         MakeBooleanAccessor(&RoutingProtocol::m_enableAdaptiveHello),
                         MakeBooleanChecker())
            .AddAttribute("MinHelloInterval",
                         "Minimum hello interval for adaptive mechanism (seconds)",
                         DoubleValue(0.25),
                         MakeDoubleAccessor(&RoutingProtocol::m_minHelloInterval),
                         MakeDoubleChecker<double>())
            .AddAttribute("MaxHelloInterval",
                         "Maximum hello interval for adaptive mechanism (seconds)",
                         DoubleValue(5.0),
                         MakeDoubleAccessor(&RoutingProtocol::m_maxHelloInterval),
                         MakeDoubleChecker<double>())
            .AddAttribute("RadioRange",
                         "Radio communication range (meters)",
                         DoubleValue(250.0),
                         MakeDoubleAccessor(&RoutingProtocol::m_radioRange),
                         MakeDoubleChecker<double>())
            .AddAttribute("NodeCount",
                        "Number of nodes in the network",
                        UintegerValue(100),
                        MakeUintegerAccessor(&RoutingProtocol::m_nodeCount),
                        MakeUintegerChecker<uint32_t>())
            .AddAttribute("MaxSpeed",
                        "Maximum node movement speed (m/s)",
                        UintegerValue(40),
                        MakeUintegerAccessor(&RoutingProtocol::m_maxSpeed),
                        MakeUintegerChecker<uint32_t>());          
    return tid;
}

void
RoutingProtocol::SetMaxQueueLen(uint32_t len)
{
    m_maxQueueLen = len;
    m_queue.SetMaxQueueLen(len);
}

void
RoutingProtocol::SetMaxQueueTime(Time t)
{
    m_maxQueueTime = t;
    m_queue.SetQueueTimeout(t);
}

RoutingProtocol::~RoutingProtocol()
{
}

void
RoutingProtocol::DoDispose()
{
    m_ipv4 = nullptr;
    for (std::map<Ptr<Socket>, Ipv4InterfaceAddress>::iterator iter = m_socketAddresses.begin();
         iter != m_socketAddresses.end();
         iter++)
    {
        iter->first->Close();
    }
    m_socketAddresses.clear();
    for (std::map<Ptr<Socket>, Ipv4InterfaceAddress>::iterator iter =
             m_socketSubnetBroadcastAddresses.begin();
         iter != m_socketSubnetBroadcastAddresses.end();
         iter++)
    {
        iter->first->Close();
    }
    m_socketSubnetBroadcastAddresses.clear();
    Ipv4RoutingProtocol::DoDispose();
}

void
RoutingProtocol::PrintRoutingTable(Ptr<OutputStreamWrapper> stream, Time::Unit unit) const
{
    *stream->GetStream() << "Node: " << m_ipv4->GetObject<Node>()->GetId()
                         << "; Time: " << Now().As(unit)
                         << ", Local time: " << m_ipv4->GetObject<Node>()->GetLocalTime().As(unit)
                         << ", QOAR Routing table" << std::endl;

    m_routingTable.Print(stream, unit);
    *stream->GetStream() << std::endl;
}

int64_t
RoutingProtocol::AssignStreams(int64_t stream)
{
    NS_LOG_FUNCTION(this << stream);
    m_uniformRandomVariable->SetStream(stream);
    return 1;
}

void RoutingProtocol::Start()
{
    NS_LOG_FUNCTION(this);
    if (m_enableHello)
    {
        m_nb.ScheduleTimer();
    }
    m_rreqRateLimitTimer.SetFunction(&RoutingProtocol::RreqRateLimitTimerExpire, this);
    m_rreqRateLimitTimer.Schedule(Seconds(1));

    m_rerrRateLimitTimer.SetFunction(&RoutingProtocol::RerrRateLimitTimerExpire, this);
    m_rerrRateLimitTimer.Schedule(Seconds(1));

    // 与 DQN::setParameters(alpha, gamma, int nodes, int speed) 签名匹配
    (void)m_qLearning.setParameters(
        m_alpha,
        m_gamma,
        m_a,
        m_b,
        m_c
    );
}


Ptr<Ipv4Route>
RoutingProtocol::RouteOutput(Ptr<Packet> p,
                             const Ipv4Header& header,
                             Ptr<NetDevice> oif,
                             Socket::SocketErrno& sockerr)
{
    NS_LOG_FUNCTION(this << header << (oif ? oif->GetIfIndex() : 0));

    if (!p)
    {
        NS_LOG_DEBUG("Packet is == 0");
        return LoopbackRoute(header, oif); // later
    }
    if (m_socketAddresses.empty())
    {
        sockerr = Socket::ERROR_NOROUTETOHOST;
        NS_LOG_LOGIC("No qoar interfaces");
        Ptr<Ipv4Route> route;
        return route;
    }

    sockerr = Socket::ERROR_NOTERROR;
    Ptr<Ipv4Route> route;
    Ipv4Address dst = header.GetDestination();

    // ======== 使用 DQN（m_qLearning）建议的下一跳 ========
    const std::string currentNode = GetNodeAddressString();

    // 与 DQN 接口匹配：仅传 currentNode
    auto[bestNextHopStr,band] = m_qLearning.getBestNextHop(currentNode);

    if (!bestNextHopStr.empty())
    {   
        Ipv4Address nextHopAddr(bestNextHopStr.c_str());

        // 只接受“邻居”作为可转发下一跳
        if (m_nb.IsNeighbor(nextHopAddr))
        {
            NS_LOG_DEBUG("DQN suggests neighbor: " << bestNextHopStr);

            // 为了拿到输出接口等路由元信息，这里查邻居的路由表项
            RoutingTableEntry rtToNext;
            if (m_routingTable.LookupValidRoute(nextHopAddr, rtToNext))
            {
                route = rtToNext.GetRoute();

                // 若上层强制了 oif 且不一致，则构造一条匹配 oif 的单跳路由
                if (oif && route->GetOutputDevice() != oif)
                {
                    for (auto j = m_socketAddresses.begin(); j != m_socketAddresses.end(); ++j)
                    {
                        Ipv4InterfaceAddress iface = j->second;
                        if (m_ipv4->GetInterfaceForAddress(iface.GetLocal()) ==
                            m_ipv4->GetInterfaceForDevice(oif))
                        {
                            Ptr<Ipv4Route> newRoute = Create<Ipv4Route>();
                            newRoute->SetDestination(dst);
                            newRoute->SetSource(iface.GetLocal());
                            newRoute->SetGateway(nextHopAddr);
                            newRoute->SetOutputDevice(oif);
                            route = newRoute;
                            break;
                        }
                    }
                }
                if (!route) {
                std::cerr << "[错误] route 是空指针，无法设置 Action！" << std::endl;
                }
                route->SetAction(band);
                // ---- 5 因子奖励：sf/ef/bf/lq/delay ----
                auto clamp01 = [](double v) { return std::max(0.0, std::min(1.0, v)); };

                // 从邻居管理器读取链路指标（若未归一化，将被 clamp01 保护）
                double sf = m_nb.GetSf(nextHopAddr);  // 距离/空间因子（越小越好）
                double ef = m_nb.GetEf(nextHopAddr);  // 能量因子（越大越好）
                double bf = m_nb.GetBf(nextHopAddr);  // 带宽/干扰因子（越大越好）
                double lq = m_nb.GetLq(nextHopAddr);  // 综合链路质量（越大越好）

                // 估计单跳时延（毫秒）：使用协议配置的 NodeTraversalTime 作为保守估计
                // const double delayMs = static_cast<double>(m_nodeTraversalTime.GetMilliSeconds());
                double delayMs = m_nb.Getdf(nextHopAddr);
                // 将 sf（距离）与 delay 做到[0,1]的单调归一化：更小更优 => 值更接近 1
                const double sf_norm    = 1.0 / (1.0 + (sf / std::max(1.0, m_radioRange)));
                const double delay_norm = 1.0 / (1.0 + (delayMs / 100.0)); // 100ms 量纲基准

                // 其余三项做 [0,1] 截断
                const double ef_norm = clamp01(ef);
                const double bf_norm = clamp01(bf);
                const double lq_norm = clamp01(lq);

                // 已有 a/b/c 三个权重；余下权重平分给 lq 与 delay
                const double rest   = std::max(0.0, 1.0 - (m_a + m_b + m_c));
                const double w_lq   = rest;
                // const double w_dly  = 0.5 * rest;

                // 奖励：五因子加权和；直达目的给予微小加成
                double reward = m_a * sf_norm + m_b * delay_norm + m_c * bf_norm
                              + w_lq * lq_norm;
                if (nextHopAddr == dst)
                {
                    reward += 0.05; // 直达奖励（保持尺度稳定）
                }
                reward = clamp01(reward);
                const std::string dest=GetAddressString(dst);
                // 触发 MAPPO/DQN 学习（按 DQN 的现有签名）
                const bool terminal = (nextHopAddr == dst);


                //2.4Ghz
                Ptr<NetDevice> Ndev = m_ipv4->GetNetDevice(1);
                Ptr<WifiNetDevice> NwifiDev = DynamicCast<WifiNetDevice>(Ndev);
                Ptr<CitrAdhocWifiMac> NcitrMac = DynamicCast<CitrAdhocWifiMac>(NwifiDev->GetMac());
                const int firstQueueLength=NcitrMac->GetCitr();

                //5Ghz
                Ptr<NetDevice> ACdev = m_ipv4->GetNetDevice(2);
                Ptr<WifiNetDevice> ACwifiDev = DynamicCast<WifiNetDevice>(ACdev);
                Ptr<CitrAdhocWifiMac> ACcitrMac = DynamicCast<CitrAdhocWifiMac>(ACwifiDev->GetMac());
                const int secondQueueLength=ACcitrMac->GetCitr();
                // std::cout<<"firstQueueLength"<<firstQueueLength<<"\n";
                // std::cout<<"secondQueueLength"<<secondQueueLength<<"\n";

                int score24=0;
                int score5=0;
                int newband=0;
                if(sf_norm>0.9){
                    score5+=1;
                }else{
                    score24+=1;
                }
                score24+=4*firstQueueLength;
                score5+=4*secondQueueLength;
                if(score24>score5){
                    newband=0;
                }else{
                    newband=1;
                }
                m_qLearning.updateQValue(sf_norm,delay_norm,bf_norm,currentNode, bestNextHopStr, dest,newband,reward, terminal); 
       
                // 维护路由生命期
                UpdateRouteLifeTime(dst, m_activeRouteTimeout);
                UpdateRouteLifeTime(nextHopAddr, m_activeRouteTimeout);
                
                // 若路由对象来自邻居项，记得将“目的地址”设置为真正的 dst
                if (route)
                {
                    route->SetDestination(dst);
                }
                return route;
            }
        }
        else
        {
        //    std::cout<<"MAPPO suggested " << bestNextHopStr
        //                   << " is not a neighbor, fallback to traditional routing \n";
        }
    }
    // ======== 回退：传统路由查找 ========
    RoutingTableEntry rt;
    if (m_routingTable.LookupValidRoute(dst, rt))
    {
        route = rt.GetRoute();
        NS_ASSERT(route);
        NS_LOG_DEBUG("Exist route to " << route->GetDestination()
                      << " from interface " << route->GetSource());

        if (oif && route->GetOutputDevice() != oif)
        {
            NS_LOG_DEBUG("Output device doesn't match. Dropped.");
            sockerr = Socket::ERROR_NOROUTETOHOST;
            return Ptr<Ipv4Route>();
        }
        UpdateRouteLifeTime(dst, m_activeRouteTimeout);
        UpdateRouteLifeTime(route->GetGateway(), m_activeRouteTimeout);
        return route;
    }

    // ======== 无有效路由：回环，DeferredRouteOutput 里触发 RREQ ========
    uint32_t iif = (oif ? m_ipv4->GetInterfaceForDevice(oif) : -1);
    DeferredRouteOutputTag tag(iif);
    NS_LOG_DEBUG("Valid Route not found");
    if (!p->PeekPacketTag(tag))
    {
        p->AddPacketTag(tag);
    }
    return LoopbackRoute(header, oif);
}


void
RoutingProtocol::DeferredRouteOutput(Ptr<const Packet> p,
                                     const Ipv4Header& header,
                                     UnicastForwardCallback ucb,
                                     ErrorCallback ecb)
{
    NS_LOG_FUNCTION(this << p << header);
    NS_ASSERT(p && p != Ptr<Packet>());

    QueueEntry newEntry(p, header, ucb, ecb);
    bool result = m_queue.Enqueue(newEntry);
    if (result)
    {
        NS_LOG_LOGIC("Add packet " << p->GetUid() << " to queue. Protocol "
                                   << (uint16_t)header.GetProtocol());
        RoutingTableEntry rt;
        bool result = m_routingTable.LookupRoute(header.GetDestination(), rt);
        if (!result || ((rt.GetFlag() != IN_SEARCH) && result))
        {
            NS_LOG_LOGIC("Send new RREQ for outbound packet to " << header.GetDestination());
            SendRequest(header.GetDestination());
        }
    }
}

bool
RoutingProtocol::RouteInput(Ptr<const Packet> p,
                            const Ipv4Header& header,
                            Ptr<const NetDevice> idev,
                            UnicastForwardCallback ucb,
                            MulticastForwardCallback mcb,
                            LocalDeliverCallback lcb,
                            ErrorCallback ecb)
{
    NS_LOG_FUNCTION(this << p->GetUid() << header.GetDestination() << idev->GetAddress());
    if (m_socketAddresses.empty())
    {
        NS_LOG_LOGIC("No qoar interfaces");
        return false;
    }
    NS_ASSERT(m_ipv4);
    NS_ASSERT(p);
    // Check if input device supports IP
    NS_ASSERT(m_ipv4->GetInterfaceForDevice(idev) >= 0);
    int32_t iif = m_ipv4->GetInterfaceForDevice(idev);

    Ipv4Address dst = header.GetDestination();
    Ipv4Address origin = header.GetSource();

    // Deferred route request
    if (idev == m_lo)
    {
        DeferredRouteOutputTag tag;
        if (p->PeekPacketTag(tag))
        {
            DeferredRouteOutput(p, header, ucb, ecb);
            return true;
        }
    }

    // Duplicate of own packet
    if (IsMyOwnAddress(origin))
    {
        return true;
    }

    // QOAR is not a multicast routing protocol
    if (dst.IsMulticast())
    {
        return false;
    }

    // Broadcast local delivery/forwarding
    for (std::map<Ptr<Socket>, Ipv4InterfaceAddress>::const_iterator j = m_socketAddresses.begin();
         j != m_socketAddresses.end();
         ++j)
    {
        Ipv4InterfaceAddress iface = j->second;
        if (m_ipv4->GetInterfaceForAddress(iface.GetLocal()) == iif)
        {
            if (dst == iface.GetBroadcast() || dst.IsBroadcast())
            {
                if (m_dpd.IsDuplicate(p, header))
                {
                    NS_LOG_DEBUG("Duplicated packet " << p->GetUid() << " from " << origin
                                                      << ". Drop.");
                    return true;
                }
                UpdateRouteLifeTime(origin, m_activeRouteTimeout);
                Ptr<Packet> packet = p->Copy();
                if (lcb.IsNull() == false)
                {
                    NS_LOG_LOGIC("Broadcast local delivery to " << iface.GetLocal());
                    lcb(p, header, iif);
                    // Fall through to additional processing
                }
                else
                {
                    NS_LOG_ERROR("Unable to deliver packet locally due to null callback "
                                 << p->GetUid() << " from " << origin);
                    ecb(p, header, Socket::ERROR_NOROUTETOHOST);
                }
                if (!m_enableBroadcast)
                {
                    return true;
                }
                if (header.GetProtocol() == UdpL4Protocol::PROT_NUMBER)
                {
                    UdpHeader udpHeader;
                    p->PeekHeader(udpHeader);
                    if (udpHeader.GetDestinationPort() == Qoar_PORT)
                    {
                        // QOAR packets sent in broadcast are already managed
                        return true;
                    }
                }
                if (header.GetTtl() > 1)
                {
                    NS_LOG_LOGIC("Forward broadcast. TTL " << (uint16_t)header.GetTtl());
                    RoutingTableEntry toBroadcast;
                    if (m_routingTable.LookupRoute(dst, toBroadcast))
                    {
                        Ptr<Ipv4Route> route = toBroadcast.GetRoute();
                        ucb(route, packet, header);
                    }
                    else
                    {
                        NS_LOG_DEBUG("No route to forward broadcast. Drop packet " << p->GetUid());
                    }
                }
                else
                {
                    NS_LOG_DEBUG("TTL exceeded. Drop packet " << p->GetUid());
                }
                return true;
            }
        }
    }

    // Unicast local delivery
    if (m_ipv4->IsDestinationAddress(dst, iif))
    {
        UpdateRouteLifeTime(origin, m_activeRouteTimeout);
        RoutingTableEntry toOrigin;
        if (m_routingTable.LookupValidRoute(origin, toOrigin))
        {
            UpdateRouteLifeTime(toOrigin.GetNextHop(), m_activeRouteTimeout);
            m_nb.Update(toOrigin.GetNextHop(), m_activeRouteTimeout);
        }
        if (lcb.IsNull() == false)
        {
            NS_LOG_LOGIC("Unicast local delivery to " << dst);
            lcb(p, header, iif);
        }
        else
        {
            NS_LOG_ERROR("Unable to deliver packet locally due to null callback "
                         << p->GetUid() << " from " << origin);
            ecb(p, header, Socket::ERROR_NOROUTETOHOST);
        }
        return true;
    }

    // Check if input device supports IP forwarding
    if (m_ipv4->IsForwarding(iif) == false)
    {
        NS_LOG_LOGIC("Forwarding disabled for this interface");
        ecb(p, header, Socket::ERROR_NOROUTETOHOST);
        return true;
    }

    // Forwarding
    return Forwarding(p, header, ucb, ecb);
}

bool
RoutingProtocol::Forwarding(Ptr<const Packet> p,
                            const Ipv4Header& header,
                            UnicastForwardCallback ucb,
                            ErrorCallback ecb)
{
    NS_LOG_FUNCTION(this);
    Ipv4Address dst = header.GetDestination();
    Ipv4Address origin = header.GetSource();

    m_routingTable.Purge();

    // ======== 使用 MAPPO（m_qLearning）建议的下一跳（DQN API: 单参） ========
    const std::string currentNode = GetNodeAddressString();
    auto[bestNextHopStr,band] = m_qLearning.getBestNextHop(currentNode); // 修正：单参

    if (!bestNextHopStr.empty())
    {
        Ipv4Address nextHopAddr(bestNextHopStr.c_str());

        // 只接受“邻居”作为可转发下一跳
        if (m_nb.IsNeighbor(nextHopAddr))
        {
            NS_LOG_DEBUG("MAPPO suggests neighbor: " << bestNextHopStr);

            // 为了拿到输出接口等路由元信息，这里查邻居的路由表项
            RoutingTableEntry rtToNext;
            if (m_routingTable.LookupValidRoute(nextHopAddr, rtToNext))
            {
                Ptr<Ipv4Route> route = rtToNext.GetRoute();
                const int act=band;
                route->SetAction(act);
                // ---- 5 因子奖励：sf/ef/bf/lq/delay ----
                auto clamp01 = [](double v) { return std::max(0.0, std::min(1.0, v)); };

                // 从邻居管理器读取链路指标（若未归一化，将被 clamp01 保护）
                double sf = m_nb.GetSf(nextHopAddr);  // 距离/空间因子（越小越好）
                double ef = m_nb.GetEf(nextHopAddr);  // 能量因子（越大越好）
                double bf = m_nb.GetBf(nextHopAddr);  // 带宽/干扰因子（越大越好）
                double lq = m_nb.GetLq(nextHopAddr);  // 综合链路质量（越大越好）

                // 估计单跳时延（毫秒）：使用保守估计 NodeTraversalTime
                // const double delayMs = static_cast<double>(m_nodeTraversalTime.GetMilliSeconds());
                double delayMs = m_nb.Getdf(nextHopAddr);
                // 将 sf（距离）与 delay 做到[0,1]的单调归一化：更小更优 => 值更接近 1
                const double sf_norm    = 1.0 / (1.0 + (sf / std::max(1.0, m_radioRange)));
                const double delay_norm = 1.0 / (1.0 + (delayMs / 100.0)); // 100ms 量纲基准

                // 其余三项做 [0,1] 截断
                const double ef_norm = clamp01(ef);
                const double bf_norm = clamp01(bf);
                const double lq_norm = clamp01(lq);

                // 已有 a/b/c 三个权重；余下权重平分给 lq 与 delay
                const double rest  = std::max(0.0, 1.0 - (m_a + m_b + m_c));
                const double w_lq  = rest;
                // const double w_dly = 0.5 * rest;

                // 奖励：五因子加权和；直达目的给予微小加成
                double reward = m_a * sf_norm + m_b * delay_norm + m_c * bf_norm
                              + w_lq * lq_norm;
                if (nextHopAddr == dst) {
                    reward += 0.05; // 直达奖励（保持尺度稳定）
                }
                reward = clamp01(reward);
                
                //2.4Ghz
                Ptr<NetDevice> Ndev = m_ipv4->GetNetDevice(1);
                Ptr<WifiNetDevice> NwifiDev = DynamicCast<WifiNetDevice>(Ndev);
                Ptr<CitrAdhocWifiMac> NcitrMac = DynamicCast<CitrAdhocWifiMac>(NwifiDev->GetMac());
                const int firstQueueLength=NcitrMac->GetCitr();

                //5Ghz
                Ptr<NetDevice> ACdev = m_ipv4->GetNetDevice(2);
                Ptr<WifiNetDevice> ACwifiDev = DynamicCast<WifiNetDevice>(ACdev);
                Ptr<CitrAdhocWifiMac> ACcitrMac = DynamicCast<CitrAdhocWifiMac>(ACwifiDev->GetMac());
                const int secondQueueLength=ACcitrMac->GetCitr();
                // std::cout<<"firstQueueLength"<<firstQueueLength<<"\n";
                // std::cout<<"secondQueueLength"<<secondQueueLength<<"\n";
                
                int score24=0;
                int score5=0;
                int newband=0;
                if(sf_norm>0.9){
                    score5+=1;
                }else{
                    score24+=1;
                }
                score24+=4*firstQueueLength;
                score5+=4*secondQueueLength;
                if(score24>score5){
                    newband=0;
                }else{
                    newband=1;
                }

                const std::string dest=GetAddressString(dst);
                // 触发 MAPPO/DQN 学习（按 DQN 的现有签名）
                const bool terminal = (nextHopAddr == dst);

                m_qLearning.updateQValue(sf_norm,delay_norm,bf_norm,currentNode, bestNextHopStr, dest,newband,reward, terminal); 


                
                // 维护路由生命期
                UpdateRouteLifeTime(origin, m_activeRouteTimeout);
                UpdateRouteLifeTime(dst, m_activeRouteTimeout);
                UpdateRouteLifeTime(nextHopAddr, m_activeRouteTimeout);

                // 将真正的目的地址设置到 route
                if (route) {
                    route->SetDestination(dst);
                }

                // 转发
                ucb(route, p, header);
                return true;
            }
        }
        else
        {
            // std::cout<<"MAPPO suggested " << bestNextHopStr
            //               << " is not a neighbor, fallback to traditional routing \n";
        }
    }

    // ======== 回退：传统路由查找 ========
    RoutingTableEntry toDst;
    if (m_routingTable.LookupRoute(dst, toDst))
    {
        if (toDst.GetFlag() == VALID)
        {
            Ptr<Ipv4Route> route = toDst.GetRoute();
            NS_LOG_LOGIC(route->GetSource() << " forwarding to " << dst << " from " << origin
                                            << " packet " << p->GetUid());

            UpdateRouteLifeTime(origin, m_activeRouteTimeout);
            UpdateRouteLifeTime(dst, m_activeRouteTimeout);
            UpdateRouteLifeTime(route->GetGateway(), m_activeRouteTimeout);

            RoutingTableEntry toOrigin;
            m_routingTable.LookupRoute(origin, toOrigin);
            UpdateRouteLifeTime(toOrigin.GetNextHop(), m_activeRouteTimeout);

            m_nb.Update(route->GetGateway(), m_activeRouteTimeout);
            m_nb.Update(toOrigin.GetNextHop(), m_activeRouteTimeout);

            ucb(route, p, header);
            return true;
        }
        else
        {
            if (toDst.GetValidSeqNo())
            {
                SendRerrWhenNoRouteToForward(dst, toDst.GetSeqNo(), origin);
                NS_LOG_DEBUG("Drop packet " << p->GetUid() << " because no route to forward it.");
                return false;
            }
        }
    }

    NS_LOG_LOGIC("route not found to " << dst << ". Send RERR message.");
    NS_LOG_DEBUG("Drop packet " << p->GetUid() << " because no route to forward it.");
    SendRerrWhenNoRouteToForward(dst, 0, origin);
    return false;
}

void
RoutingProtocol::SetIpv4(Ptr<Ipv4> ipv4)
{
    NS_ASSERT(ipv4);
    NS_ASSERT(!m_ipv4);

    m_ipv4 = ipv4;

    // Create lo route. It is asserted that the only one interface up for now is loopback
    NS_ASSERT(m_ipv4->GetNInterfaces() == 1 &&
              m_ipv4->GetAddress(0, 0).GetLocal() == Ipv4Address("127.0.0.1"));
    m_lo = m_ipv4->GetNetDevice(0);
    NS_ASSERT(m_lo);
    // Remember lo route
    RoutingTableEntry rt(
        /*dev=*/m_lo,
        /*dst=*/Ipv4Address::GetLoopback(),
        /*vSeqNo=*/true,
        /*seqNo=*/0,
        /*iface=*/Ipv4InterfaceAddress(Ipv4Address::GetLoopback(), Ipv4Mask("255.0.0.0")),
        /*hops=*/1,
        /*nextHop=*/Ipv4Address::GetLoopback(),
        /*lifetime=*/Simulator::GetMaximumSimulationTime());
    m_routingTable.AddRoute(rt);

    Simulator::ScheduleNow(&RoutingProtocol::Start, this);
}

void
RoutingProtocol::NotifyInterfaceUp(uint32_t i)
{
    NS_LOG_FUNCTION(this << m_ipv4->GetAddress(i, 0).GetLocal());
    Ptr<Ipv4L3Protocol> l3 = m_ipv4->GetObject<Ipv4L3Protocol>();

 // ------------------------- 新增代码开始 -------------------------
    // 获取节点对象
    Ptr<Node> node = m_ipv4->GetObject<Node>();
    // 绑定能量源（假设每个节点只有一个）
    if (!m_energySource) { // 避免重复绑定
        m_energySource = node->GetObject<BasicEnergySource>();
        if (!m_energySource) {

        // std::cout<<"Node " << node->GetId() << " has no BasicEnergySource installed!";
        } else {

        // std::cout<<"Node " << node->GetId() << " energy source initialized: "
        //                     << m_energySource->GetRemainingEnergy() << " J";
    
        }
    }
    // ------------------------- 新增代码结束 -------------------------

    if (l3->GetNAddresses(i) > 1)
    {
        NS_LOG_WARN("QOAR does not work with more then one address per each interface.");
    }
    Ipv4InterfaceAddress iface = l3->GetAddress(i, 0);
    if (iface.GetLocal() == Ipv4Address("127.0.0.1"))
    {
        return;
    }

    // Create a socket to listen only on this interface
    Ptr<Socket> socket = Socket::CreateSocket(GetObject<Node>(), UdpSocketFactory::GetTypeId());
    NS_ASSERT(socket);
    socket->SetRecvCallback(MakeCallback(&RoutingProtocol::RecvQoar, this));
    socket->BindToNetDevice(l3->GetNetDevice(i));
    socket->Bind(InetSocketAddress(iface.GetLocal(), Qoar_PORT));
    socket->SetAllowBroadcast(true);
    socket->SetIpRecvTtl(true);
    m_socketAddresses.insert(std::make_pair(socket, iface));

    // create also a subnet broadcast socket
    socket = Socket::CreateSocket(GetObject<Node>(), UdpSocketFactory::GetTypeId());
    NS_ASSERT(socket);
    socket->SetRecvCallback(MakeCallback(&RoutingProtocol::RecvQoar, this));
    socket->BindToNetDevice(l3->GetNetDevice(i));
    socket->Bind(InetSocketAddress(iface.GetBroadcast(), Qoar_PORT));
    socket->SetAllowBroadcast(true);
    socket->SetIpRecvTtl(true);
    m_socketSubnetBroadcastAddresses.insert(std::make_pair(socket, iface));

    // Add local broadcast record to the routing table
    Ptr<NetDevice> dev = m_ipv4->GetNetDevice(m_ipv4->GetInterfaceForAddress(iface.GetLocal()));
    RoutingTableEntry rt(/*dev=*/dev,
                         /*dst=*/iface.GetBroadcast(),
                         /*vSeqNo=*/true,
                         /*seqNo=*/0,
                         /*iface=*/iface,
                         /*hops=*/1,
                         /*nextHop=*/iface.GetBroadcast(),
                         /*lifetime=*/Simulator::GetMaximumSimulationTime());
    m_routingTable.AddRoute(rt);

    if (l3->GetInterface(i)->GetArpCache())
    {
        m_nb.AddArpCache(l3->GetInterface(i)->GetArpCache());
    }

    // Allow neighbor manager use this interface for layer 2 feedback if possible
    Ptr<WifiNetDevice> wifi = dev->GetObject<WifiNetDevice>();
    if (!wifi)
    {
        return;
    }
    Ptr<WifiMac> mac = wifi->GetMac();
    if (!mac)
    {
        return;
    }

    mac->TraceConnectWithoutContext("DroppedMpdu",
                                    MakeCallback(&RoutingProtocol::NotifyTxError, this));
}

void
RoutingProtocol::NotifyTxError(WifiMacDropReason reason, Ptr<const WifiMpdu> mpdu)
{
    m_nb.GetTxErrorCallback()(mpdu->GetHeader());
}

void
RoutingProtocol::NotifyInterfaceDown(uint32_t i)
{
    NS_LOG_FUNCTION(this << m_ipv4->GetAddress(i, 0).GetLocal());

    // Disable layer 2 link state monitoring (if possible)
    Ptr<Ipv4L3Protocol> l3 = m_ipv4->GetObject<Ipv4L3Protocol>();
    Ptr<NetDevice> dev = l3->GetNetDevice(i);
    Ptr<WifiNetDevice> wifi = dev->GetObject<WifiNetDevice>();
    if (wifi)
    {
        Ptr<WifiMac> mac = wifi->GetMac()->GetObject<AdhocWifiMac>();
        if (mac)
        {
            mac->TraceDisconnectWithoutContext("DroppedMpdu",
                                               MakeCallback(&RoutingProtocol::NotifyTxError, this));
            m_nb.DelArpCache(l3->GetInterface(i)->GetArpCache());
        }
    }

    // Close socket
    Ptr<Socket> socket = FindSocketWithInterfaceAddress(m_ipv4->GetAddress(i, 0));
    NS_ASSERT(socket);
    socket->Close();
    m_socketAddresses.erase(socket);

    // Close socket
    socket = FindSubnetBroadcastSocketWithInterfaceAddress(m_ipv4->GetAddress(i, 0));
    NS_ASSERT(socket);
    socket->Close();
    m_socketSubnetBroadcastAddresses.erase(socket);

    if (m_socketAddresses.empty())
    {
        NS_LOG_LOGIC("No qoar interfaces");
        m_htimer.Cancel();
        m_nb.Clear();
        m_routingTable.Clear();
        return;
    }
    m_routingTable.DeleteAllRoutesFromInterface(m_ipv4->GetAddress(i, 0));
}

void
RoutingProtocol::NotifyAddAddress(uint32_t i, Ipv4InterfaceAddress address)
{
    NS_LOG_FUNCTION(this << " interface " << i << " address " << address);
    Ptr<Ipv4L3Protocol> l3 = m_ipv4->GetObject<Ipv4L3Protocol>();
    if (!l3->IsUp(i))
    {
        return;
    }
    if (l3->GetNAddresses(i) == 1)
    {
        Ipv4InterfaceAddress iface = l3->GetAddress(i, 0);
        Ptr<Socket> socket = FindSocketWithInterfaceAddress(iface);
        if (!socket)
        {
            if (iface.GetLocal() == Ipv4Address("127.0.0.1"))
            {
                return;
            }
            // Create a socket to listen only on this interface
            Ptr<Socket> socket =
                Socket::CreateSocket(GetObject<Node>(), UdpSocketFactory::GetTypeId());
            NS_ASSERT(socket);
            socket->SetRecvCallback(MakeCallback(&RoutingProtocol::RecvQoar, this));
            socket->BindToNetDevice(l3->GetNetDevice(i));
            socket->Bind(InetSocketAddress(iface.GetLocal(), Qoar_PORT));
            socket->SetAllowBroadcast(true);
            m_socketAddresses.insert(std::make_pair(socket, iface));

            // create also a subnet directed broadcast socket
            socket = Socket::CreateSocket(GetObject<Node>(), UdpSocketFactory::GetTypeId());
            NS_ASSERT(socket);
            socket->SetRecvCallback(MakeCallback(&RoutingProtocol::RecvQoar, this));
            socket->BindToNetDevice(l3->GetNetDevice(i));
            socket->Bind(InetSocketAddress(iface.GetBroadcast(), Qoar_PORT));
            socket->SetAllowBroadcast(true);
            socket->SetIpRecvTtl(true);
            m_socketSubnetBroadcastAddresses.insert(std::make_pair(socket, iface));

            // Add local broadcast record to the routing table
            Ptr<NetDevice> dev =
                m_ipv4->GetNetDevice(m_ipv4->GetInterfaceForAddress(iface.GetLocal()));
            RoutingTableEntry rt(/*dev=*/dev,
                                 /*dst=*/iface.GetBroadcast(),
                                 /*vSeqNo=*/true,
                                 /*seqNo=*/0,
                                 /*iface=*/iface,
                                 /*hops=*/1,
                                 /*nextHop=*/iface.GetBroadcast(),
                                 /*lifetime=*/Simulator::GetMaximumSimulationTime());
            m_routingTable.AddRoute(rt);
        }
    }
    else
    {
        NS_LOG_LOGIC("QOAR does not work with more then one address per each interface. Ignore "
                     "added address");
    }
}

void
RoutingProtocol::NotifyRemoveAddress(uint32_t i, Ipv4InterfaceAddress address)
{
    NS_LOG_FUNCTION(this);
    Ptr<Socket> socket = FindSocketWithInterfaceAddress(address);
    if (socket)
    {
        m_routingTable.DeleteAllRoutesFromInterface(address);
        socket->Close();
        m_socketAddresses.erase(socket);

        Ptr<Socket> unicastSocket = FindSubnetBroadcastSocketWithInterfaceAddress(address);
        if (unicastSocket)
        {
            unicastSocket->Close();
            m_socketAddresses.erase(unicastSocket);
        }

        Ptr<Ipv4L3Protocol> l3 = m_ipv4->GetObject<Ipv4L3Protocol>();
        if (l3->GetNAddresses(i))
        {
            Ipv4InterfaceAddress iface = l3->GetAddress(i, 0);
            // Create a socket to listen only on this interface
            Ptr<Socket> socket =
                Socket::CreateSocket(GetObject<Node>(), UdpSocketFactory::GetTypeId());
            NS_ASSERT(socket);
            socket->SetRecvCallback(MakeCallback(&RoutingProtocol::RecvQoar, this));
            // Bind to any IP address so that broadcasts can be received
            socket->BindToNetDevice(l3->GetNetDevice(i));
            socket->Bind(InetSocketAddress(iface.GetLocal(), Qoar_PORT));
            socket->SetAllowBroadcast(true);
            socket->SetIpRecvTtl(true);
            m_socketAddresses.insert(std::make_pair(socket, iface));

            // create also a unicast socket
            socket = Socket::CreateSocket(GetObject<Node>(), UdpSocketFactory::GetTypeId());
            NS_ASSERT(socket);
            socket->SetRecvCallback(MakeCallback(&RoutingProtocol::RecvQoar, this));
            socket->BindToNetDevice(l3->GetNetDevice(i));
            socket->Bind(InetSocketAddress(iface.GetBroadcast(), Qoar_PORT));
            socket->SetAllowBroadcast(true);
            socket->SetIpRecvTtl(true);
            m_socketSubnetBroadcastAddresses.insert(std::make_pair(socket, iface));

            // Add local broadcast record to the routing table
            Ptr<NetDevice> dev =
                m_ipv4->GetNetDevice(m_ipv4->GetInterfaceForAddress(iface.GetLocal()));
            RoutingTableEntry rt(/*dev=*/dev,
                                 /*dst=*/iface.GetBroadcast(),
                                 /*vSeqNo=*/true,
                                 /*seqNo=*/0,
                                 /*iface=*/iface,
                                 /*hops=*/1,
                                 /*nextHop=*/iface.GetBroadcast(),
                                 /*lifetime=*/Simulator::GetMaximumSimulationTime());
            m_routingTable.AddRoute(rt);
        }
        if (m_socketAddresses.empty())
        {
            NS_LOG_LOGIC("No qoar interfaces");
            m_htimer.Cancel();
            m_nb.Clear();
            m_routingTable.Clear();
            return;
        }
    }
    else
    {
        NS_LOG_LOGIC("Remove address not participating in QOAR operation");
    }
}

bool
RoutingProtocol::IsMyOwnAddress(Ipv4Address src)
{
    NS_LOG_FUNCTION(this << src);
    for (std::map<Ptr<Socket>, Ipv4InterfaceAddress>::const_iterator j = m_socketAddresses.begin();
         j != m_socketAddresses.end();
         ++j)
    {
        Ipv4InterfaceAddress iface = j->second;
        if (src == iface.GetLocal())
        {
            return true;
        }
    }
    return false;
}

Ptr<Ipv4Route>
RoutingProtocol::LoopbackRoute(const Ipv4Header& hdr, Ptr<NetDevice> oif) const
{
    NS_LOG_FUNCTION(this << hdr);
    NS_ASSERT(m_lo);
    Ptr<Ipv4Route> rt = Create<Ipv4Route>();
    rt->SetDestination(hdr.GetDestination());
    //
    // Source address selection here is tricky.  The loopback route is
    // returned when QOAR does not have a route; this causes the packet
    // to be looped back and handled (cached) in RouteInput() method
    // while a route is found. However, connection-oriented protocols
    // like TCP need to create an endpoint four-tuple (src, src port,
    // dst, dst port) and create a pseudo-header for checksumming.  So,
    // QOAR needs to guess correctly what the eventual source address
    // will be.
    //
    // For single interface, single address nodes, this is not a problem.
    // When there are possibly multiple outgoing interfaces, the policy
    // implemented here is to pick the first available QOAR interface.
    // If RouteOutput() caller specified an outgoing interface, that
    // further constrains the selection of source address
    //
    std::map<Ptr<Socket>, Ipv4InterfaceAddress>::const_iterator j = m_socketAddresses.begin();
    if (oif)
    {
        // Iterate to find an address on the oif device
        for (j = m_socketAddresses.begin(); j != m_socketAddresses.end(); ++j)
        {
            Ipv4Address addr = j->second.GetLocal();
            int32_t interface = m_ipv4->GetInterfaceForAddress(addr);
            if (oif == m_ipv4->GetNetDevice(static_cast<uint32_t>(interface)))
            {
                rt->SetSource(addr);
                break;
            }
        }
    }
    else
    {
        rt->SetSource(j->second.GetLocal());
    }
    NS_ASSERT_MSG(rt->GetSource() != Ipv4Address(), "Valid QOAR source address not found");
    rt->SetGateway(Ipv4Address("127.0.0.1"));
    rt->SetOutputDevice(m_lo);
    return rt;
}

void
RoutingProtocol::PlotTrainingResults(std::string filename)
{
    m_qLearning.PlotTrainingCurves(filename);  
}

void
RoutingProtocol::SendRequest(Ipv4Address dst)
{
    NS_LOG_FUNCTION(this << dst);
    // A node SHOULD NOT originate more than RREQ_RATELIMIT RREQ messages per second.
    if (m_rreqCount == m_rreqRateLimit)
    {
        Simulator::Schedule(m_rreqRateLimitTimer.GetDelayLeft() + MicroSeconds(100),
                            &RoutingProtocol::SendRequest,
                            this,
                            dst);
        return;
    }
    else
    {
        m_rreqCount++;
    }
    // Create RREQ header
    RreqHeader rreqHeader;
    rreqHeader.SetDst(dst);

    //新增代码：获取节点坐标
    Ptr<MobilityModel> mobility = m_ipv4->GetObject<MobilityModel>();
    if (mobility == nullptr) {
        NS_LOG_ERROR("No MobilityModel installed on this node!");
        return; // 或处理错误
    }
    Vector3D pos = mobility->GetPosition(); // 直接获取三维坐标
    double x = pos.x;
    double y = pos.y;
    double z = pos.z;
    rreqHeader.SetLocation(x,y,z); // 注入坐标

    if (m_energySource) {
        double remainingEnergy = m_energySource->GetRemainingEnergy();
        rreqHeader.SetEnergy(remainingEnergy); // 将能量值注入RREQ头部
    }else{
    //    std::cout<<("  mei bang ding ");
    }
    rreqHeader.SetDelay(Simulator::Now());
    std::string ipStr = GetAddressString(dst);  
    int interfaceIndex=1;
    if (ipStr.rfind("10.1.", 0) == 0)  
        {
            interfaceIndex=1;
        }
        else if (ipStr.rfind("10.2.", 0) == 0)
        {
            interfaceIndex=2;
        }

    Ptr<NetDevice> dev = m_ipv4->GetNetDevice(interfaceIndex);
    Ptr<WifiNetDevice> wifiDev = DynamicCast<WifiNetDevice>(dev);
    Ptr<CitrAdhocWifiMac> citrMac = DynamicCast<CitrAdhocWifiMac>(wifiDev->GetMac());
    double citrValue =citrMac->GetCitr();;
    rreqHeader.SetCITR(citrValue);
    rreqHeader.SetMaxQ(1);


    RoutingTableEntry rt;
    // Using the Hop field in Routing Table to manage the expanding ring search
    uint16_t ttl = m_ttlStart;
    if (m_routingTable.LookupRoute(dst, rt))
    {
        if (rt.GetFlag() != IN_SEARCH)
        {
            ttl = std::min<uint16_t>(rt.GetHop() + m_ttlIncrement, m_netDiameter);
        }
        else
        {
            ttl = rt.GetHop() + m_ttlIncrement;
            if (ttl > m_ttlThreshold)
            {
                ttl = m_netDiameter;
            }
        }
        if (ttl == m_netDiameter)
        {
            rt.IncrementRreqCnt();
        }
        if (rt.GetValidSeqNo())
        {
            rreqHeader.SetDstSeqno(rt.GetSeqNo());
        }
        else
        {
            rreqHeader.SetUnknownSeqno(true);
        }
        rt.SetHop(ttl);
        rt.SetFlag(IN_SEARCH);
        rt.SetLifeTime(m_pathDiscoveryTime);
        m_routingTable.Update(rt);
    }
    else
    {
        rreqHeader.SetUnknownSeqno(true);
        Ptr<NetDevice> dev = nullptr;
        RoutingTableEntry newEntry(/*dev=*/dev,
                                   /*dst=*/dst,
                                   /*vSeqNo=*/false,
                                   /*seqNo=*/0,
                                   /*iface=*/Ipv4InterfaceAddress(),
                                   /*hops=*/ttl,
                                   /*nextHop=*/Ipv4Address(),
                                   /*lifetime=*/m_pathDiscoveryTime);
        // Check if TtlStart == NetDiameter
        if (ttl == m_netDiameter)
        {
            newEntry.IncrementRreqCnt();
        }
        newEntry.SetFlag(IN_SEARCH);
        m_routingTable.AddRoute(newEntry);
    }

    if (m_gratuitousReply)
    {
        rreqHeader.SetGratuitousRrep(true);
    }
    if (m_destinationOnly)
    {
        rreqHeader.SetDestinationOnly(true);
    }

    m_seqNo++;
    rreqHeader.SetOriginSeqno(m_seqNo);
    m_requestId++;
    rreqHeader.SetId(m_requestId);

    // Send RREQ as subnet directed broadcast from each interface used by qoar
    for (std::map<Ptr<Socket>, Ipv4InterfaceAddress>::const_iterator j = m_socketAddresses.begin();
         j != m_socketAddresses.end();
         ++j)
    {
        Ptr<Socket> socket = j->first;
        Ipv4InterfaceAddress iface = j->second;

        rreqHeader.SetOrigin(iface.GetLocal());
        m_rreqIdCache.IsDuplicate(iface.GetLocal(), m_requestId);

        Ptr<Packet> packet = Create<Packet>();
        SocketIpTtlTag tag;
        tag.SetTtl(ttl);
        packet->AddPacketTag(tag);
        packet->AddHeader(rreqHeader);
        TypeHeader tHeader(QoarTYPE_RREQ);
        packet->AddHeader(tHeader);
        // Send to all-hosts broadcast if on /32 addr, subnet-directed otherwise
        Ipv4Address destination;
        if (iface.GetMask() == Ipv4Mask::GetOnes())
        {
            destination = Ipv4Address("255.255.255.255");
        }
        else
        {
            destination = iface.GetBroadcast();
        }
        NS_LOG_DEBUG("Send RREQ with id " << rreqHeader.GetId() << " to socket");
        m_lastBcastTime = Simulator::Now();
        Simulator::Schedule(Time(MilliSeconds(m_uniformRandomVariable->GetInteger(0, 10))),
                            &RoutingProtocol::SendTo,
                            this,
                            socket,
                            packet,
                            destination);
    }
    ScheduleRreqRetry(dst);
}

void
RoutingProtocol::SendTo(Ptr<Socket> socket, Ptr<Packet> packet, Ipv4Address destination)
{
    socket->SendTo(packet, 0, InetSocketAddress(destination, Qoar_PORT));
}

void
RoutingProtocol::ScheduleRreqRetry(Ipv4Address dst)
{
    NS_LOG_FUNCTION(this << dst);
    if (m_addressReqTimer.find(dst) == m_addressReqTimer.end())
    {
        Timer timer(Timer::CANCEL_ON_DESTROY);
        m_addressReqTimer[dst] = timer;
    }
    m_addressReqTimer[dst].SetFunction(&RoutingProtocol::RouteRequestTimerExpire, this);
    m_addressReqTimer[dst].Cancel();
    m_addressReqTimer[dst].SetArguments(dst);
    RoutingTableEntry rt;
    m_routingTable.LookupRoute(dst, rt);
    Time retry;
    if (rt.GetHop() < m_netDiameter)
    {
        retry = 2 * m_nodeTraversalTime * (rt.GetHop() + m_timeoutBuffer);
    }
    else
    {
        //NS_ABORT_MSG_UNLESS(rt.GetRreqCnt() > 0, "Unexpected value for GetRreqCount ()");
        uint16_t backoffFactor = rt.GetRreqCnt() - 1;
        NS_LOG_LOGIC("Applying binary exponential backoff factor " << backoffFactor);
        retry = m_netTraversalTime * (1 << backoffFactor);
    }
    m_addressReqTimer[dst].Schedule(retry);
    NS_LOG_LOGIC("Scheduled RREQ retry in " << retry.As(Time::S));
}

void
RoutingProtocol::RecvQoar(Ptr<Socket> socket)
{
    NS_LOG_FUNCTION(this << socket);
    Address sourceAddress;
    Ptr<Packet> packet = socket->RecvFrom(sourceAddress);
    InetSocketAddress inetSourceAddr = InetSocketAddress::ConvertFrom(sourceAddress);
    Ipv4Address sender = inetSourceAddr.GetIpv4();
    Ipv4Address receiver;

    if (m_socketAddresses.find(socket) != m_socketAddresses.end())
    {
        receiver = m_socketAddresses[socket].GetLocal();
    }
    else if (m_socketSubnetBroadcastAddresses.find(socket) !=
             m_socketSubnetBroadcastAddresses.end())
    {
        receiver = m_socketSubnetBroadcastAddresses[socket].GetLocal();
    }
    else
    {
        NS_ASSERT_MSG(false, "Received a packet from an unknown socket");
    }
    NS_LOG_DEBUG("myQOAR node " << this << " received a QOAR packet from " << sender << " to "
                              << receiver);

    UpdateRouteToNeighbor(sender, receiver);
    TypeHeader tHeader(QoarTYPE_RREQ);
    packet->RemoveHeader(tHeader);
    if (!tHeader.IsValid())
    {
        NS_LOG_DEBUG("mQOAR message " << packet->GetUid() << " with unknown type received: "
                                     << tHeader.Get() << ". Drop");
        return; // drop
    }
    switch (tHeader.Get())
    {
    case QoarTYPE_RREQ: {
        RecvRequest(packet, receiver, sender);
        break;
    }
    case QoarTYPE_RREP: {
        RecvReply(packet, receiver, sender);
        break;
    }
    case QoarTYPE_RERR: {
        RecvError(packet, sender);
        break;
    }
    case QoarTYPE_RREP_ACK: {
        RecvReplyAck(sender);
        break;
    }
    }
}

bool
RoutingProtocol::UpdateRouteLifeTime(Ipv4Address addr, Time lifetime)
{
    NS_LOG_FUNCTION(this << addr << lifetime);
    RoutingTableEntry rt;
    if (m_routingTable.LookupRoute(addr, rt))
    {
        if (rt.GetFlag() == VALID)
        {
            NS_LOG_DEBUG("Updating VALID route");
            rt.SetRreqCnt(0);
            rt.SetLifeTime(std::max(lifetime, rt.GetLifeTime()));
            m_routingTable.Update(rt);
            return true;
        }
    }
    return false;
}

void
RoutingProtocol::UpdateRouteToNeighbor(Ipv4Address sender, Ipv4Address receiver)
{
    NS_LOG_FUNCTION(this << "sender " << sender << " receiver " << receiver);
    RoutingTableEntry toNeighbor;
    if (!m_routingTable.LookupRoute(sender, toNeighbor))
    {
        Ptr<NetDevice> dev = m_ipv4->GetNetDevice(m_ipv4->GetInterfaceForAddress(receiver));
        RoutingTableEntry newEntry(
            /*dev=*/dev,
            /*dst=*/sender,
            /*vSeqNo=*/false,
            /*seqNo=*/0,
            /*iface=*/m_ipv4->GetAddress(m_ipv4->GetInterfaceForAddress(receiver), 0),
            /*hops=*/1,
            /*nextHop=*/sender,
            /*lifetime=*/m_activeRouteTimeout);
        m_routingTable.AddRoute(newEntry);
    }
    else
    {
        Ptr<NetDevice> dev = m_ipv4->GetNetDevice(m_ipv4->GetInterfaceForAddress(receiver));
        if (toNeighbor.GetValidSeqNo() && (toNeighbor.GetHop() == 1) &&
            (toNeighbor.GetOutputDevice() == dev))
        {
            toNeighbor.SetLifeTime(std::max(m_activeRouteTimeout, toNeighbor.GetLifeTime()));
        }
        else
        {
            RoutingTableEntry newEntry(
                /*dev=*/dev,
                /*dst=*/sender,
                /*vSeqNo=*/false,
                /*seqNo=*/0,
                /*iface=*/m_ipv4->GetAddress(m_ipv4->GetInterfaceForAddress(receiver), 0),
                /*hops=*/1,
                /*nextHop=*/sender,
                /*lifetime=*/std::max(m_activeRouteTimeout, toNeighbor.GetLifeTime()));
            m_routingTable.Update(newEntry);
        }
    }
}

double 
RoutingProtocol::CalculateDistance(double srcX,double srcY,double srcZ,Ptr<Ipv4> m_ipv4,Ipv4Address src)
{
         double distance = 0.0;
          // 获取源节点坐标（来自RREQ报文）
          // 获取当前节点坐标（通过MobilityModel）
          Ptr<MobilityModel> mobility = m_ipv4->GetObject<MobilityModel>();
          if (mobility) 
          {
              Vector3D currentPos = mobility->GetPosition();
              
              // 计算距离
              double dx = currentPos.x - srcX;
              double dy = currentPos.y - srcY;
              double dz = currentPos.z - srcZ;
              distance = std::sqrt(dx*dx + dy*dy + dz*dz);
   
              // 更新邻居表的SF值
              m_nb.UpdateSf(src, distance); // src是RREQ发送节点地址
          }
          else 
          {
              NS_LOG_WARN("Node " << m_ipv4->GetAddress(1, 0).GetLocal() 
                           << " has no MobilityModel! Cannot calculate distance.");
          }
          return distance;
}

double 
RoutingProtocol::GetCitr(Ipv4InterfaceAddress targetInterface)
{
    NS_LOG_FUNCTION(this << targetInterface.GetLocal());

    // 1. 检查IP协议栈是否初始化
    if (!m_ipv4)
    {
        std::cout<<"IPv4协议栈未初始化";
        return -1.0;
    }

    // 2. 获取目标接口的索引
    int32_t interfaceIndex = m_ipv4->GetInterfaceForAddress(targetInterface.GetLocal());
    if (interfaceIndex == -1)
    {
        std::cout<<"接口地址 " << targetInterface.GetLocal() << " 不存在";
        return -1.0;
    }

    // 3. 获取网络设备
    Ptr<NetDevice> dev = m_ipv4->GetNetDevice(interfaceIndex);
    if (!dev)
    {
        std::cout<<"无法获取接口 " << interfaceIndex << " 的网络设备";
        return -1.0;
    }

    // 4. 转换为WiFi设备
    Ptr<WifiNetDevice> wifiDev = DynamicCast<WifiNetDevice>(dev);
    if (!wifiDev)
    {
        std::cout<<"接口 " << interfaceIndex << " 不是WiFi设备";
        return -1.0;
    }

    // 5. 获取MAC层
    Ptr<CitrAdhocWifiMac> citrMac = DynamicCast<CitrAdhocWifiMac>(wifiDev->GetMac());
    if (!citrMac)
    {
        std::cout<<"MAC层未使用CitrAdhocWifiMac类型";
        return -1.0;
    }

    // 6. 返回CITR值
    return citrMac->GetCitr();
  
 }  

void
RoutingProtocol::RecvRequest(Ptr<Packet> p, Ipv4Address receiver, Ipv4Address src)
{
    NS_LOG_FUNCTION(this);
    RreqHeader rreqHeader;
    p->RemoveHeader(rreqHeader);

    // 1) 黑名单检查
    RoutingTableEntry toPrev;
    if (m_routingTable.LookupRoute(src, toPrev))
    {
        if (toPrev.IsUnidirectional())
        {
            NS_LOG_DEBUG("Ignoring RREQ from node in blacklist");
            return;
        }
    }

    // 2) 重复 (Origin, RREQ_ID) 过滤
    const uint32_t id    = rreqHeader.GetId();
    Ipv4Address    origin = rreqHeader.GetOrigin();
    if (m_rreqIdCache.IsDuplicate(origin, id))
    {
        NS_LOG_DEBUG("Ignoring RREQ due to duplicate");
        return;
    }

    // 3) 递增跳数
    uint8_t hop = rreqHeader.GetHopCount() + 1;
    rreqHeader.SetHopCount(hop);

    // ---------- 五因子：sf/ef/bf/lq + delay ----------
    // 3.1 位置 -> 距离（sf）
    const double srcX = rreqHeader.GetLocationX();
    const double srcY = rreqHeader.GetLocationY();
    const double srcZ = rreqHeader.GetLocationZ();
    RoutingProtocol::CalculateDistance(srcX, srcY, srcZ, m_ipv4, src); // 内部应更新 m_nb 的 sf

    // 3.2 能量（ef）
    const double energy = rreqHeader.GetEnergy();
    m_nb.UpdateEf(src, energy);

    // 3.3 渠道/队列因子（bf：之前作为 CITR/队列占用）
    const double citr = rreqHeader.GetCITR();
    // std::cout<<"citr in rreq:"<<citr<<"\n";
    m_nb.UpdateBf(src, citr);

    // 3.4 链路质量（lq：保持之前的定义）
    const double lq = 0.5 * m_nb.GetSf(src) + 0.5 * m_nb.GetBf(src);
    m_nb.SetLq(src, lq);

    // 3.5 时延（delay）：RREQ 内若带了 delay（以 Time 存），取其毫秒；否则回退本地估计
    Time delayIn = Simulator::Now()-rreqHeader.GetDelay();              // 注意：这是 Time 类型
    double delayMsIn = delayIn.IsZero()
                       ? static_cast<double>(m_nodeTraversalTime.GetMilliSeconds())
                       : static_cast<double>(delayIn.GetMilliSeconds());

    // 把“本节点的单跳估计”写回 RREQ（SetDelay 需要 Time）
    // rreqHeader.SetDelay(delayIn);
    m_nb.Updatedf(src, delayMsIn);
    // ---------- 奖励计算（与 RouteOutput/Forwarding 保持一致） ----------
    auto clamp01 = [](double v) { return std::max(0.0, std::min(1.0, v)); };

    const double sf_raw   = m_nb.GetSf(src);
    const double sf_norm  = 1.0 / (1.0 + (sf_raw / std::max(1.0, m_radioRange)));
    const double ef_norm  = clamp01(m_nb.GetEf(src));
    const double bf_norm  = clamp01(m_nb.GetBf(src));
    const double lq_norm  = clamp01(lq);
    const double delay_norm = 1.0 / (1.0 + (m_nb.Getdf(src) / 100.0)); // 100ms 量纲可按需调整

    const double rest  = std::max(0.0, 1.0 - (m_a + m_b + m_c));
    const double w_lq  = rest;
    // const double w_dly = 0.5 * rest;

    double reward = m_a * sf_norm + m_b * delay_norm + m_c * bf_norm
                  + w_lq * lq_norm;

    // 若本节点就是目的，给予微小加成
    m_dst = rreqHeader.GetDst();
    if (IsMyOwnAddress(m_dst))
    {
        reward += 0.05;
    }
    reward = clamp01(reward);

    // ---------- MAPPO 侧交互（对齐 DQN 接口命名） ----------
    const std::string currentNode = GetNodeAddressString(); // 当前节点
    const std::string prevHopStr  = GetAddressString(src);  // 上一跳（邻居）
    const std::string destStr     = GetAddressString(m_dst);
    const bool terminalHere = IsMyOwnAddress(m_dst);
    //2.4Ghz
    Ptr<NetDevice> Ndev = m_ipv4->GetNetDevice(1);
    Ptr<WifiNetDevice> NwifiDev = DynamicCast<WifiNetDevice>(Ndev);
    Ptr<CitrAdhocWifiMac> NcitrMac = DynamicCast<CitrAdhocWifiMac>(NwifiDev->GetMac());
    const int firstQueueLength=NcitrMac->GetCitr();

    //5Ghz
    Ptr<NetDevice> ACdev = m_ipv4->GetNetDevice(2);
    Ptr<WifiNetDevice> ACwifiDev = DynamicCast<WifiNetDevice>(ACdev);
    Ptr<CitrAdhocWifiMac> ACcitrMac = DynamicCast<CitrAdhocWifiMac>(ACwifiDev->GetMac());
    const int secondQueueLength=ACcitrMac->GetCitr();
    // std::cout<<"firstQueueLength"<<firstQueueLength<<"\n";
    // std::cout<<"secondQueueLength"<<secondQueueLength<<"\n";
    
    int score24=0;
    int score5=0;
    int newband=0;
    if(sf_norm>0.9){
        score5+=1;
    }else{
        score24+=1;
    }
    score24+=4*firstQueueLength;
    score5+=4*secondQueueLength;
    if(score24>score5){
        newband=0;
    }else{
        newband=1;
    }

    m_qLearning.updateQValue(sf_norm,delay_norm,bf_norm,currentNode, prevHopStr, destStr,newband,reward, terminalHere);

 
    rreqHeader.SetMaxQ(m_nb.GetMaxQ());

    // ---------- 反向路由维护 ----------
    RoutingTableEntry toOrigin;
    if (!m_routingTable.LookupRoute(origin, toOrigin))
    {
        Ptr<NetDevice> dev = m_ipv4->GetNetDevice(m_ipv4->GetInterfaceForAddress(receiver));
        RoutingTableEntry newEntry(
            /*dev=*/dev,
            /*dst=*/origin,
            /*vSeqNo=*/true,
            /*seqNo=*/rreqHeader.GetOriginSeqno(),
            /*iface=*/m_ipv4->GetAddress(m_ipv4->GetInterfaceForAddress(receiver), 0),
            /*hops=*/hop,
            /*nextHop=*/src,
            /*lifetime=*/Time(2 * m_netTraversalTime - 2 * hop * m_nodeTraversalTime));
        m_routingTable.AddRoute(newEntry);
    }
    else
    {
        if (toOrigin.GetValidSeqNo())
        {
            if (int32_t(rreqHeader.GetOriginSeqno()) - int32_t(toOrigin.GetSeqNo()) > 0)
            {
                toOrigin.SetSeqNo(rreqHeader.GetOriginSeqno());
            }
        }
        else
        {
            toOrigin.SetSeqNo(rreqHeader.GetOriginSeqno());
        }
        toOrigin.SetValidSeqNo(true);
        toOrigin.SetNextHop(src);
        toOrigin.SetOutputDevice(m_ipv4->GetNetDevice(m_ipv4->GetInterfaceForAddress(receiver)));
        toOrigin.SetInterface(m_ipv4->GetAddress(m_ipv4->GetInterfaceForAddress(receiver), 0));
        toOrigin.SetHop(hop);
        toOrigin.SetLifeTime(std::max(Time(2 * m_netTraversalTime - 2 * hop * m_nodeTraversalTime),
                                      toOrigin.GetLifeTime()));
        m_routingTable.Update(toOrigin);
    }

    // 邻居单跳表项
    RoutingTableEntry toNeighbor;
    if (!m_routingTable.LookupRoute(src, toNeighbor))
    {
        NS_LOG_DEBUG("Neighbor:" << src << " not found in routing table. Creating an entry");
        Ptr<NetDevice> dev = m_ipv4->GetNetDevice(m_ipv4->GetInterfaceForAddress(receiver));
        RoutingTableEntry newEntry(dev,
                                   src,
                                   false,
                                   rreqHeader.GetOriginSeqno(),
                                   m_ipv4->GetAddress(m_ipv4->GetInterfaceForAddress(receiver), 0),
                                   1,
                                   src,
                                   m_activeRouteTimeout);
        m_routingTable.AddRoute(newEntry);
    }
    else
    {
        toNeighbor.SetLifeTime(m_activeRouteTimeout);
        toNeighbor.SetValidSeqNo(false);
        toNeighbor.SetSeqNo(rreqHeader.GetOriginSeqno());
        toNeighbor.SetFlag(VALID);
        toNeighbor.SetOutputDevice(m_ipv4->GetNetDevice(m_ipv4->GetInterfaceForAddress(receiver)));
        toNeighbor.SetInterface(m_ipv4->GetAddress(m_ipv4->GetInterfaceForAddress(receiver), 0));
        toNeighbor.SetHop(1);
        toNeighbor.SetNextHop(src);
        m_routingTable.Update(toNeighbor);
    }
    m_nb.Update(src, Time(m_allowedHelloLoss * m_helloInterval));

    NS_LOG_LOGIC(receiver << " receive RREQ with hop count "
                          << static_cast<uint32_t>(rreqHeader.GetHopCount()) << " ID "
                          << rreqHeader.GetId() << " to destination " << rreqHeader.GetDst());

    // 目的本机：直接回 RREP
    if (IsMyOwnAddress(rreqHeader.GetDst()))
    {
        m_routingTable.LookupRoute(origin, toOrigin);
        NS_LOG_DEBUG("Send reply since I am the destination");
        SendReply(rreqHeader, toOrigin);
        return;
    }

    // 中继条件：已有到目的的有效路由且不过度限定
    RoutingTableEntry toDst;
    Ipv4Address dst = rreqHeader.GetDst();
    if (m_routingTable.LookupRoute(dst, toDst))
    {
        if (toDst.GetNextHop() == src)
        {
            NS_LOG_DEBUG("Drop RREQ from " << src << ", dest next hop " << toDst.GetNextHop());
            return;
        }
        if ((rreqHeader.GetUnknownSeqno() ||
             (int32_t(toDst.GetSeqNo()) - int32_t(rreqHeader.GetDstSeqno()) >= 0)) &&
            toDst.GetValidSeqNo())
        {
            if (!rreqHeader.GetDestinationOnly() && toDst.GetFlag() == VALID)
            {
                m_routingTable.LookupRoute(origin, toOrigin);
                SendReplyByIntermediateNode(toDst, toOrigin, rreqHeader.GetGratuitousRrep());
                return;
            }
            rreqHeader.SetDstSeqno(toDst.GetSeqNo());
            rreqHeader.SetUnknownSeqno(false);
        }
    }

    // TTL 检查
    SocketIpTtlTag tag;
    p->RemovePacketTag(tag);
    if (tag.GetTtl() < 2)
    {
        NS_LOG_DEBUG("TTL exceeded. Drop RREQ origin " << src << " destination " << dst);
        return;
    }

    // 继续广播 RREQ（此时 rreqHeader 已带本节点估计的 delay）
    for (auto j = m_socketAddresses.begin(); j != m_socketAddresses.end(); ++j)
    {
        Ptr<Socket> socket = j->first;
        Ipv4InterfaceAddress iface = j->second;
        Ptr<Packet> packet = Create<Packet>();
        SocketIpTtlTag ttl;
        ttl.SetTtl(tag.GetTtl() - 1);
        packet->AddPacketTag(ttl);
        packet->AddHeader(rreqHeader);
        TypeHeader tHeader(QoarTYPE_RREQ);
        packet->AddHeader(tHeader);

        Ipv4Address destination =
            (iface.GetMask() == Ipv4Mask::GetOnes()) ? Ipv4Address("255.255.255.255")
                                                     : iface.GetBroadcast();
        m_lastBcastTime = Simulator::Now();
        Simulator::Schedule(Time(MilliSeconds(m_uniformRandomVariable->GetInteger(0, 10))),
                            &RoutingProtocol::SendTo,
                            this,
                            socket,
                            packet,
                            destination);
    }
}

void
RoutingProtocol::SendReply(const RreqHeader& rreqHeader, const RoutingTableEntry& toOrigin)
{
    NS_LOG_FUNCTION(this << toOrigin.GetDestination());
    // 序列号处理
    if (!rreqHeader.GetUnknownSeqno() && (rreqHeader.GetDstSeqno() == m_seqNo + 1))
    {
        m_seqNo++;
    }

    RrepHeader rrepHeader(/*prefixSize=*/0,
                          /*hopCount=*/0,
                          /*dst=*/rreqHeader.GetDst(),
                          /*dstSeqNo=*/m_seqNo,
                          /*origin=*/toOrigin.GetDestination(),
                          /*lifetime=*/m_myRouteTimeout);

    // 位置坐标
    Ptr<MobilityModel> mobility = m_ipv4->GetObject<MobilityModel>();
    if (mobility == nullptr)
    {
        NS_LOG_ERROR("No MobilityModel installed on this node!");
        return;
    }
    const Vector3D pos = mobility->GetPosition();
    rrepHeader.SetLocation(pos.x, pos.y, pos.z);

    // 能量
    Ptr<Node> node = m_ipv4->GetObject<Node>();
    if (!m_energySource)
    {
        m_energySource = node->GetObject<BasicEnergySource>();
    }
    if (m_energySource)
    {
        const double remainingEnergy = m_energySource->GetRemainingEnergy();
        rrepHeader.SetEnergy(remainingEnergy);
    }

    // CITR（队列长度衡量值）——按现有 GetCitr 接口获取
    const Ipv4InterfaceAddress targetInterface = toOrigin.GetInterface();
    const double citrValue = GetCitr(targetInterface);
    rrepHeader.SetCITR(citrValue);

    // 将“单跳时延（毫秒）”写入控制包：接口是 SetDelay(Time)
    // const Time myDelay = m_nodeTraversalTime; // 一跳遍历时间作为估计
    rrepHeader.SetDelay(Simulator::Now());

    // 写入 MaxQ：不要调用不存在的 GetNeighbors()；改用现有的接口
    // 如果 Neighbors 实现里有 GetMaxQ()，保留下面这一行：
    rrepHeader.SetMaxQ(m_nb.GetMaxQ());
    // 如果没有 GetMaxQ()，可以注释上一行，替换为固定值或从其他模块获取：
    // rrepHeader.SetMaxQ(1.0);

    // 发送 RREP
    Ptr<Packet> packet = Create<Packet>();
    SocketIpTtlTag tag;
    tag.SetTtl(toOrigin.GetHop());
    packet->AddPacketTag(tag);
    packet->AddHeader(rrepHeader);
    TypeHeader tHeader(QoarTYPE_RREP);
    packet->AddHeader(tHeader);
    Ptr<Socket> socket = FindSocketWithInterfaceAddress(toOrigin.GetInterface());
    NS_ASSERT(socket);
    socket->SendTo(packet, 0, InetSocketAddress(toOrigin.GetNextHop(), Qoar_PORT));
}

void
RoutingProtocol::SendReplyByIntermediateNode(RoutingTableEntry& toDst,
                                             RoutingTableEntry& toOrigin,
                                             bool gratRep)
{
    NS_LOG_FUNCTION(this);

    // === 构造主 RREP（中继节点向 RREQ 源头返回） ===
    RrepHeader rrepHeader(/*prefixSize=*/0,
                          /*hopCount=*/toDst.GetHop(),
                          /*dst=*/toDst.GetDestination(),
                          /*dstSeqNo=*/toDst.GetSeqNo(),
                          /*origin=*/toOrigin.GetDestination(),
                          /*lifetime=*/toDst.GetLifeTime());

    // 位置坐标
    Ptr<MobilityModel> mobility = m_ipv4->GetObject<MobilityModel>();
    if (mobility == nullptr)
    {
        NS_LOG_ERROR("No MobilityModel installed on this node!");
        return;
    }
    const Vector3D pos = mobility->GetPosition();
    rrepHeader.SetLocation(pos.x, pos.y, pos.z);

    // 节点能量（若能获取到能量源）
    if (m_energySource)
    {
        const double remainingEnergy = m_energySource->GetRemainingEnergy();
        rrepHeader.SetEnergy(remainingEnergy);
    }else{
        // std::cout<<(" rrep mei bang ding ");
    }

    // CITR（基于回程接口）
    const Ipv4InterfaceAddress targetInterface = toOrigin.GetInterface();
    const double citrValue = GetCitr(targetInterface);
    rrepHeader.SetCITR(citrValue);

    // 单跳时延：直接写入协议配置的一跳遍历时间（Time 类型）
    rrepHeader.SetDelay(Simulator::Now());

    // 近似的最大Q（如果没有 getMaxQvalue，则用邻居管理器的聚合值/上界）
    rrepHeader.SetMaxQ(m_nb.GetMaxQ());

    // 若下一跳为邻居（1跳），请求 RREP-ACK 以检测单向链路
    if (toDst.GetHop() == 1)
    {
        rrepHeader.SetAckRequired(true);
        RoutingTableEntry toNextHop;
        m_routingTable.LookupRoute(toOrigin.GetNextHop(), toNextHop);
        toNextHop.m_ackTimer.SetFunction(&RoutingProtocol::AckTimerExpire, this);
        toNextHop.m_ackTimer.SetArguments(toNextHop.GetDestination(), m_blackListTimeout);
        toNextHop.m_ackTimer.SetDelay(m_nextHopWait);
    }

    // 维护前驱
    toDst.InsertPrecursor(toOrigin.GetNextHop());
    toOrigin.InsertPrecursor(toDst.GetNextHop());
    m_routingTable.Update(toDst);
    m_routingTable.Update(toOrigin);

    // 发送主 RREP
    {
        Ptr<Packet> packet = Create<Packet>();
        SocketIpTtlTag tag;
        tag.SetTtl(toOrigin.GetHop());
        packet->AddPacketTag(tag);
        packet->AddHeader(rrepHeader);
        TypeHeader tHeader(QoarTYPE_RREP);
        packet->AddHeader(tHeader);

        Ptr<Socket> socket = FindSocketWithInterfaceAddress(toOrigin.GetInterface());
        NS_ASSERT(socket);
        socket->SendTo(packet, 0, InetSocketAddress(toOrigin.GetNextHop(), Qoar_PORT));
    }

    // === 可选：生成并发送 Gratuitous RREP（给真正的目的节点，建立反向路） ===
    if (gratRep)
    {
        RrepHeader gratRepHeader(/*prefixSize=*/0,
                                 /*hopCount=*/toOrigin.GetHop(),
                                 /*dst=*/toOrigin.GetDestination(),   // 让目的也拿到到源的路
                                 /*dstSeqNo=*/toOrigin.GetSeqNo(),
                                 /*origin=*/toDst.GetDestination(),
                                 /*lifetime=*/toOrigin.GetLifeTime());

        // 同步携带位置信息/能量/CITR/时延/MaxQ
        gratRepHeader.SetLocation(pos.x, pos.y, pos.z);
        if (m_energySource)
        {
            const double remainingEnergy = m_energySource->GetRemainingEnergy();
            gratRepHeader.SetEnergy(remainingEnergy);
        }else{
        // std::cout<<(" rrep mei bang ding ");
    }
        gratRepHeader.SetCITR(citrValue);
        gratRepHeader.SetDelay(m_nodeTraversalTime);
        gratRepHeader.SetMaxQ(m_nb.GetMaxQ());

        Ptr<Packet> packetToDst = Create<Packet>();
        SocketIpTtlTag gratTag;
        gratTag.SetTtl(toDst.GetHop());
        packetToDst->AddPacketTag(gratTag);
        packetToDst->AddHeader(gratRepHeader);
        TypeHeader type(QoarTYPE_RREP);
        packetToDst->AddHeader(type);

        Ptr<Socket> socket = FindSocketWithInterfaceAddress(toDst.GetInterface());
        NS_ASSERT(socket);
        NS_LOG_LOGIC("Send gratuitous RREP " << packetToDst->GetUid());
        socket->SendTo(packetToDst, 0, InetSocketAddress(toDst.GetNextHop(), Qoar_PORT));
    }
}

void
RoutingProtocol::SendReplyAck(Ipv4Address neighbor)
{
    NS_LOG_FUNCTION(this << " to " << neighbor);
    RrepAckHeader h;
    TypeHeader typeHeader(QoarTYPE_RREP_ACK);
    Ptr<Packet> packet = Create<Packet>();
    SocketIpTtlTag tag;
    tag.SetTtl(1);
    packet->AddPacketTag(tag);
    packet->AddHeader(h);
    packet->AddHeader(typeHeader);
    RoutingTableEntry toNeighbor;
    m_routingTable.LookupRoute(neighbor, toNeighbor);
    Ptr<Socket> socket = FindSocketWithInterfaceAddress(toNeighbor.GetInterface());
    NS_ASSERT(socket);
    socket->SendTo(packet, 0, InetSocketAddress(neighbor, Qoar_PORT));
}

void
RoutingProtocol::RecvReply(Ptr<Packet> p, Ipv4Address receiver, Ipv4Address sender)
{
    NS_LOG_FUNCTION(this << " src " << sender);
    RrepHeader rrepHeader;
    p->RemoveHeader(rrepHeader);
    Ipv4Address dst = rrepHeader.GetDst();
    NS_LOG_LOGIC("RREP destination " << dst << " RREP origin " << rrepHeader.GetOrigin());
    m_dst = dst;

    // ---- 累加 RREP 跳数 ----
    uint8_t hop = rrepHeader.GetHopCount() + 1;
    rrepHeader.SetHopCount(hop);

    // ---- Hello 情形（RREP 的 dst==origin）沿用原逻辑 ----
    if (dst == rrepHeader.GetOrigin())
    {
        ProcessHello(rrepHeader, receiver);
        return;
    }

    // ============================================================
    // 收集五因子：sf（空间/距离）、ef（能量）、bf（信道/带宽/CITR）、lq（链路质量）、delay（控制报文字段）
    // ============================================================

    // 1) 从 RREP 的位置坐标计算与当前节点的空间距离（并写回邻居管理器的 Sf）
    const double srcX = rrepHeader.GetLocationX();
    const double srcY = rrepHeader.GetLocationY();
    const double srcZ = rrepHeader.GetLocationZ();
    (void)RoutingProtocol::CalculateDistance(srcX, srcY, srcZ, m_ipv4, sender);

    // 2) 能量因子：对方剩余能量
    const double energy = rrepHeader.GetEnergy();
    m_nb.UpdateEf(sender, energy);

    // 3) 信道/带宽因子（这里用 CITR 代表）：越大越好
    const double citr = rrepHeader.GetCITR();
    m_nb.UpdateBf(sender, citr);

    // 4) 组合链路质量（示例：0.5*Sf + 0.5*Bf；保持与此前一致）
    const double lq = 0.5 * m_nb.GetSf(sender) + 0.5 * m_nb.GetBf(sender);
    m_nb.SetLq(sender, lq);

    // 5) 控制包内携带的时延（毫秒）——注意 GetDelay() 是 ns3::Time，需要先取毫秒数
    Time delayIn=Simulator::Now()-rrepHeader.GetDelay();
    const double delayMs = static_cast<double>(delayIn.GetMilliSeconds()); // :contentReference[oaicite:1]{index=1}
    m_nb.Updatedf(sender, delayMs);
    // ============================================================
    // 组装 MAPPO 奖励（五因子）：sf / ef / bf / lq / delay
    // 规范化策略：
    //  - sf：距离越小越好 -> 1/(1 + d/RadioRange)
    //  - ef/bf/lq：假定为 [0,1] 或做 clamp
    //  - delay：越小越好 -> 1/(1 + delay/基准)，基准取 100ms
    // 权重：m_a, m_b, m_c 分配给 sf/ef/bf，剩余 (1-a-b-c) 均分给 lq 与 delay
    // ============================================================
    auto clamp01 = [](double v) { return std::max(0.0, std::min(1.0, v)); };

    const double sf_raw = m_nb.GetSf(sender);
    const double ef_raw = m_nb.GetEf(sender);
    const double bf_raw = m_nb.GetBf(sender);
    const double lq_raw = m_nb.GetLq(sender);

    const double sf_norm    = 1.0 / (1.0 + (sf_raw / std::max(1.0, m_radioRange))); // 小优->接近1
    const double ef_norm    = clamp01(ef_raw);
    const double bf_norm    = clamp01(bf_raw);
    const double lq_norm    = clamp01(lq_raw);
    const double delay_norm = 1.0 / (1.0 + (m_nb.Getdf(sender) / 100.0)); // 100ms 量纲

    const double rest  = std::max(0.0, 1.0 - (m_a + m_b + m_c));
    const double w_lq  = rest;
    // const double w_dly = 0.5 * rest;

    double reward = m_a * sf_norm + m_b * delay_norm + m_c * bf_norm
                  + w_lq * lq_norm ;

    // 直达目的的小幅奖励（保持尺度稳定）
    if (sender == dst) { reward += 0.05; }
    reward = clamp01(reward);

    // 触发一次 MAPPO 更新（使用 DQN/MAPPO 的现有签名）
    {
    const std::string currentNodeStr = GetNodeAddressString();
    const std::string senderStr      = GetAddressString(sender);
    const bool done                  = (sender == dst);
    const std::string dest   = GetAddressString(dst);
    //2.4Ghz
    Ptr<NetDevice> Ndev = m_ipv4->GetNetDevice(1);
    Ptr<WifiNetDevice> NwifiDev = DynamicCast<WifiNetDevice>(Ndev);
    Ptr<CitrAdhocWifiMac> NcitrMac = DynamicCast<CitrAdhocWifiMac>(NwifiDev->GetMac());
    const int firstQueueLength=NcitrMac->GetCitr();

    //5Ghz
    Ptr<NetDevice> ACdev = m_ipv4->GetNetDevice(2);
    Ptr<WifiNetDevice> ACwifiDev = DynamicCast<WifiNetDevice>(ACdev);
    Ptr<CitrAdhocWifiMac> ACcitrMac = DynamicCast<CitrAdhocWifiMac>(ACwifiDev->GetMac());
    const int secondQueueLength=ACcitrMac->GetCitr();
    // std::cout<<"firstQueueLength: "<<firstQueueLength<<"\n";
    // std::cout<<"secondQueueLength: "<<secondQueueLength<<"\n";
    
    int score24=0;
    int score5=0;
    int newband=0;
    if(sf_norm>0.9){
        score5+=1;
    }else{
        score24+=1;
    }
    score24+=4*firstQueueLength;
    score5+=4*secondQueueLength;
    if(score24>score5){
        newband=0;
    }else{
        newband=1;
    }
    m_qLearning.updateQValue(sf_norm,delay_norm,bf_norm,currentNodeStr, senderStr,dest,newband,reward, done);
    }

    // ============================================================
    // 按原 AODV/QOAR 流程更新路由并继续转发
    // ============================================================
    Ptr<NetDevice> dev = m_ipv4->GetNetDevice(m_ipv4->GetInterfaceForAddress(receiver));
    RoutingTableEntry newEntry(
        /*dev=*/dev,
        /*dst=*/dst,
        /*vSeqNo=*/true,
        /*seqNo=*/rrepHeader.GetDstSeqno(),
        /*iface=*/m_ipv4->GetAddress(m_ipv4->GetInterfaceForAddress(receiver), 0),
        /*hops=*/hop,
        /*nextHop=*/sender,
        /*lifetime=*/rrepHeader.GetLifeTime(),
        /*qvalue*/ reward
    );

    RoutingTableEntry toDst;
    if (m_routingTable.LookupRoute(dst, toDst))
    {
        if (!toDst.GetValidSeqNo())
        {
            m_routingTable.Update(newEntry);
        }
        else if ((int32_t(rrepHeader.GetDstSeqno()) - int32_t(toDst.GetSeqNo())) > 0)
        {
            m_routingTable.Update(newEntry);
        }
        else
        {
            if ((rrepHeader.GetDstSeqno() == toDst.GetSeqNo()) && (toDst.GetFlag() != VALID))
            {
                m_routingTable.Update(newEntry);
            }
            else if ((rrepHeader.GetDstSeqno() == toDst.GetSeqNo()) &&
                     (newEntry.Getqvalue() < toDst.Getqvalue()))
            {
                m_routingTable.Update(newEntry);
            }
        }
    }
    else
    {
        NS_LOG_LOGIC("add new route");
        m_routingTable.AddRoute(newEntry);
    }

    // RREP-ACK
    if (rrepHeader.GetAckRequired())
    {
        SendReplyAck(sender);
        rrepHeader.SetAckRequired(false);
    }

    NS_LOG_LOGIC("receiver " << receiver << " origin " << rrepHeader.GetOrigin());
    if (IsMyOwnAddress(rrepHeader.GetOrigin()))
    {
        if (toDst.GetFlag() == IN_SEARCH)
        {
            m_routingTable.Update(newEntry);
            m_addressReqTimer[dst].Cancel();
            m_addressReqTimer.erase(dst);
        }
        m_routingTable.LookupRoute(dst, toDst);
        SendPacketFromQueue(dst, toDst.GetRoute());
        return;
    }

    RoutingTableEntry toOrigin;
    if (!m_routingTable.LookupRoute(rrepHeader.GetOrigin(), toOrigin) ||
        toOrigin.GetFlag() == IN_SEARCH)
    {
        return; // Impossible! drop.
    }
    toOrigin.SetLifeTime(std::max(m_activeRouteTimeout, toOrigin.GetLifeTime()));
    m_routingTable.Update(toOrigin);

    // 维护前驱列表
    if (m_routingTable.LookupValidRoute(rrepHeader.GetDst(), toDst))
    {
        toDst.InsertPrecursor(toOrigin.GetNextHop());
        m_routingTable.Update(toDst);

        RoutingTableEntry toNextHopToDst;
        m_routingTable.LookupRoute(toDst.GetNextHop(), toNextHopToDst);
        toNextHopToDst.InsertPrecursor(toOrigin.GetNextHop());
        m_routingTable.Update(toNextHopToDst);

        toOrigin.InsertPrecursor(toDst.GetNextHop());
        m_routingTable.Update(toOrigin);

        RoutingTableEntry toNextHopToOrigin;
        m_routingTable.LookupRoute(toOrigin.GetNextHop(), toNextHopToOrigin);
        toNextHopToOrigin.InsertPrecursor(toDst.GetNextHop());
        m_routingTable.Update(toNextHopToOrigin);
    }

    // TTL 继续转发
    SocketIpTtlTag tag;
    p->RemovePacketTag(tag);
    if (tag.GetTtl() < 2)
    {
        NS_LOG_DEBUG("TTL exceeded. Drop RREP destination " << dst
                       << " origin " << rrepHeader.GetOrigin());
        return;
    }

    // 下游继续看到的“最大Q”（保持之前的接口）
    rrepHeader.SetMaxQ(m_nb.GetMaxQ());

    Ptr<Packet> packet = Create<Packet>();
    SocketIpTtlTag ttl;
    ttl.SetTtl(tag.GetTtl() - 1);
    packet->AddPacketTag(ttl);
    packet->AddHeader(rrepHeader);
    TypeHeader tHeader(QoarTYPE_RREP);
    packet->AddHeader(tHeader);
    Ptr<Socket> socket = FindSocketWithInterfaceAddress(toOrigin.GetInterface());
    NS_ASSERT(socket);
    socket->SendTo(packet, 0, InetSocketAddress(toOrigin.GetNextHop(), Qoar_PORT));
}

void
RoutingProtocol::RecvReplyAck(Ipv4Address neighbor)
{
    NS_LOG_FUNCTION(this);
    RoutingTableEntry rt;
    if (m_routingTable.LookupRoute(neighbor, rt))
    {
        rt.m_ackTimer.Cancel();
        rt.SetFlag(VALID);
        m_routingTable.Update(rt);
    }
}

void
RoutingProtocol::ProcessHello(const RrepHeader& rrepHeader, Ipv4Address receiver)
{
    NS_LOG_FUNCTION(this << "from " << rrepHeader.GetDst());

    // 邻居（Hello 的发送者）
    Ipv4Address sender = rrepHeader.GetOrigin();

    // 1) 位置坐标 -> 距离（更新 Sf）
    const double srcX = rrepHeader.GetLocationX();
    const double srcY = rrepHeader.GetLocationY();
    const double srcZ = rrepHeader.GetLocationZ();
    RoutingProtocol::CalculateDistance(srcX, srcY, srcZ, m_ipv4, sender); // 内部应更新 m_nb 的 Sf

    // 2) 能量 / 信道带宽（或干扰）指标
    const double energy = rrepHeader.GetEnergy();
    m_nb.UpdateEf(sender, energy);
    const double citr   = rrepHeader.GetCITR();
    m_nb.UpdateBf(sender, citr);

    // 3) 链路质量（保持原来的定义）
    const double lq = 0.5 * m_nb.GetSf(sender) + 0.5 * m_nb.GetBf(sender);
    m_nb.SetLq(sender, lq);

    // 4) 正确读取时延（毫秒标量）：GetDelay() 返回 ns3::Time，需取毫秒值
    const uint32_t delayMsU32 = static_cast<uint32_t>((Simulator::Now()-rrepHeader.GetDelay()).GetMilliSeconds()); // ns3::Time -> ms
    const double   delayMs    = static_cast<double>(delayMsU32);                                  // 转 double 参与归一化
    m_nb.Updatedf(sender, delayMs);

    // ================= 奖励：五因子加权 =================
    auto clamp01 = [](double v) { return std::max(0.0, std::min(1.0, v)); };

    const double sf_raw = m_nb.GetSf(sender);   // 若存的是“距离”，用距离型单调归一化（小优）
    const double ef_raw = m_nb.GetEf(sender);
    const double bf_raw = m_nb.GetBf(sender);
    const double lq_raw = m_nb.GetLq(sender);

    // sf：距离越小越好 -> 1 / (1 + d/RadioRange)
    const double sf_norm    = 1.0 / (1.0 + (sf_raw / std::max(1.0, m_radioRange)));
    const double ef_norm    = clamp01(ef_raw);
    const double bf_norm    = clamp01(bf_raw);
    const double lq_norm    = clamp01(lq_raw);
    const double delay_norm = 1.0 / (1.0 + (m_nb.Getdf(sender) / 100.0)); // 100ms 量纲基准

    // a/b/c 给 sf/ef/bf；剩余均分给 lq 与 delay
    const double rest  = std::max(0.0, 1.0 - (m_a + m_b + m_c));
    const double w_lq  = rest;
    // const double w_dly = 0.5 * rest;

    double reward = m_a * sf_norm + m_b * delay_norm + m_c * bf_norm
                  + w_lq * lq_norm ;

    // 直达目的的小幅加成（保持尺度稳定）
    if (sender == m_dst) { reward += 0.05; }
    reward = clamp01(reward);

    // ================= MAPPO/DQN 更新（使用现有的 4 参接口） =================
    const std::string currentNode = GetNodeAddressString();
    const std::string senderStr   = GetAddressString(sender);
    const bool        done        = (sender == m_dst);
    const std::string dest   = GetAddressString(m_dst);
    //2.4Ghz
    Ptr<NetDevice> Ndev = m_ipv4->GetNetDevice(1);
    Ptr<WifiNetDevice> NwifiDev = DynamicCast<WifiNetDevice>(Ndev);
    Ptr<CitrAdhocWifiMac> NcitrMac = DynamicCast<CitrAdhocWifiMac>(NwifiDev->GetMac());
    const int firstQueueLength=NcitrMac->GetCitr();

    //5Ghz
    Ptr<NetDevice> ACdev = m_ipv4->GetNetDevice(2);
    Ptr<WifiNetDevice> ACwifiDev = DynamicCast<WifiNetDevice>(ACdev);
    Ptr<CitrAdhocWifiMac> ACcitrMac = DynamicCast<CitrAdhocWifiMac>(ACwifiDev->GetMac());
    const int secondQueueLength=ACcitrMac->GetCitr();
    // std::cout<<"firstQueueLength"<<firstQueueLength<<"\n";
    // std::cout<<"secondQueueLength"<<secondQueueLength<<"\n";
    
    int score24=0;
    int score5=0;
    int newband=0;
    if(sf_norm>0.9){
        score5+=1;
    }else{
        score24+=1;
    }
    score24+=4*firstQueueLength;
    score5+=4*secondQueueLength;
    if(score24>score5){
        newband=0;
    }else{
        newband=1;
    }
    (void)m_qLearning.updateQValue(sf_norm,delay_norm,bf_norm,currentNode, senderStr, dest,newband,reward, done);

    // ================= 路由表维护：确保对邻居有一条活动路 =================
    RoutingTableEntry toNeighbor;
    if (!m_routingTable.LookupRoute(rrepHeader.GetDst(), toNeighbor))
    {
        Ptr<NetDevice> dev = m_ipv4->GetNetDevice(m_ipv4->GetInterfaceForAddress(receiver));
        RoutingTableEntry newEntry(
            /*dev=*/dev,
            /*dst=*/rrepHeader.GetDst(),
            /*vSeqNo=*/true,
            /*seqNo=*/rrepHeader.GetDstSeqno(),
            /*iface=*/m_ipv4->GetAddress(m_ipv4->GetInterfaceForAddress(receiver), 0),
            /*hops=*/1,
            /*nextHop=*/rrepHeader.GetDst(),
            /*lifetime=*/rrepHeader.GetLifeTime(),
            /*qvalue=*/1.0
        );
        m_routingTable.AddRoute(newEntry);
    }
    else
    {
        toNeighbor.SetLifeTime(std::max(Time(m_allowedHelloLoss * m_helloInterval),
                                        toNeighbor.GetLifeTime()));
        toNeighbor.SetSeqNo(rrepHeader.GetDstSeqno());
        toNeighbor.SetValidSeqNo(true);
        toNeighbor.SetFlag(VALID);
        toNeighbor.SetOutputDevice(m_ipv4->GetNetDevice(m_ipv4->GetInterfaceForAddress(receiver)));
        toNeighbor.SetInterface(m_ipv4->GetAddress(m_ipv4->GetInterfaceForAddress(receiver), 0));
        toNeighbor.SetHop(1);
        toNeighbor.Setqvalue(1.0);
        toNeighbor.SetNextHop(rrepHeader.GetDst());
        m_routingTable.Update(toNeighbor);
    }

    if (m_enableHello)
    {
        m_nb.Update(rrepHeader.GetDst(), Time(m_allowedHelloLoss * m_helloInterval));
    }
}

void
RoutingProtocol::RecvError(Ptr<Packet> p, Ipv4Address src)
{
    NS_LOG_FUNCTION(this << " from " << src);
    RerrHeader rerrHeader;
    p->RemoveHeader(rerrHeader);
    std::map<Ipv4Address, uint32_t> dstWithNextHopSrc;
    std::map<Ipv4Address, uint32_t> unreachable;
    m_routingTable.GetListOfDestinationWithNextHop(src, dstWithNextHopSrc);
    std::pair<Ipv4Address, uint32_t> un;
    while (rerrHeader.RemoveUnDestination(un))
    {
        for (std::map<Ipv4Address, uint32_t>::const_iterator i = dstWithNextHopSrc.begin();
             i != dstWithNextHopSrc.end();
             ++i)
        {
            if (i->first == un.first)
            {
                unreachable.insert(un);
            }
        }
    }

    std::vector<Ipv4Address> precursors;
    for (std::map<Ipv4Address, uint32_t>::const_iterator i = unreachable.begin();
         i != unreachable.end();)
    {
        if (!rerrHeader.AddUnDestination(i->first, i->second))
        {
            TypeHeader typeHeader(QoarTYPE_RERR);
            Ptr<Packet> packet = Create<Packet>();
            SocketIpTtlTag tag;
            tag.SetTtl(1);
            packet->AddPacketTag(tag);
            packet->AddHeader(rerrHeader);
            packet->AddHeader(typeHeader);
            SendRerrMessage(packet, precursors);
            rerrHeader.Clear();
        }
        else
        {
            RoutingTableEntry toDst;
            m_routingTable.LookupRoute(i->first, toDst);
            toDst.GetPrecursors(precursors);
            ++i;
        }
    }
    if (rerrHeader.GetDestCount() != 0)
    {
        TypeHeader typeHeader(QoarTYPE_RERR);
        Ptr<Packet> packet = Create<Packet>();
        SocketIpTtlTag tag;
        tag.SetTtl(1);
        packet->AddPacketTag(tag);
        packet->AddHeader(rerrHeader);
        packet->AddHeader(typeHeader);
        SendRerrMessage(packet, precursors);
    }
    m_routingTable.InvalidateRoutesWithDst(unreachable);
}

void
RoutingProtocol::RouteRequestTimerExpire(Ipv4Address dst)
{
    NS_LOG_LOGIC(this);
    RoutingTableEntry toDst;
    if (m_routingTable.LookupValidRoute(dst, toDst))
    {
        SendPacketFromQueue(dst, toDst.GetRoute());
        NS_LOG_LOGIC("route to " << dst << " found");
        return;
    }
    /*
     *  If a route discovery has been attempted RreqRetries times at the maximum TTL without
     *  receiving any RREP, all data packets destined for the corresponding destination SHOULD be
     *  dropped from the buffer and a Destination Unreachable message SHOULD be delivered to the
     * application.
     */
    if (toDst.GetRreqCnt() == m_rreqRetries)
    {
        NS_LOG_LOGIC("route discovery to " << dst << " has been attempted RreqRetries ("
                                           << m_rreqRetries << ") times with ttl "
                                           << m_netDiameter);
        m_addressReqTimer.erase(dst);
        m_routingTable.DeleteRoute(dst);
        NS_LOG_DEBUG("Route not found. Drop all packets with dst " << dst);
        m_queue.DropPacketWithDst(dst);
        return;
    }

    if (toDst.GetFlag() == IN_SEARCH)
    {
        NS_LOG_LOGIC("Resend RREQ to " << dst << " previous ttl " << toDst.GetHop());
        SendRequest(dst);
    }
    else
    {
        NS_LOG_DEBUG("Route down. Stop search. Drop packet with destination " << dst);
        m_addressReqTimer.erase(dst);
        m_routingTable.DeleteRoute(dst);
        m_queue.DropPacketWithDst(dst);
    }
}

void
RoutingProtocol::HelloTimerExpire()
{
    NS_LOG_FUNCTION(this);
    Time offset = Time(Seconds(0));
    if (m_lastBcastTime > Time(Seconds(0)))
    {
        offset = Simulator::Now() - m_lastBcastTime;
        NS_LOG_DEBUG("Hello deferred due to last bcast at:" << m_lastBcastTime);
    }
    else
    {
        SendHello();
    }
    m_htimer.Cancel();
    // 计算自适应Hello间隔
    Time helloInterval;
        if (m_enableAdaptiveHello)
    {
        // 获取当前节点地址
        Ipv4Address currentNodeAddr = m_ipv4->GetAddress(1, 0).GetLocal(); // 假设使用第一个接口
        
        // 计算自适应间隔
        double adaptiveSeconds = m_nb.CalculateAdaptiveHelloInterval(m_radioRange, currentNodeAddr);
        
        // 限制在配置的范围内
        adaptiveSeconds = std::min(std::max(adaptiveSeconds, m_minHelloInterval), m_maxHelloInterval);
        
        NS_LOG_DEBUG("Calculated adaptive Hello interval: " << adaptiveSeconds << "s");
        
        helloInterval = Seconds(adaptiveSeconds);
    }
    else
    {
        helloInterval = m_helloInterval;
    }
    Time diff = m_helloInterval - offset;
    m_htimer.Schedule(std::max(Time(Seconds(0)), diff));
    m_lastBcastTime = Time(Seconds(0));
}

void
RoutingProtocol::RreqRateLimitTimerExpire()
{
    NS_LOG_FUNCTION(this);
    m_rreqCount = 0;
    m_rreqRateLimitTimer.Schedule(Seconds(1));
}

void
RoutingProtocol::RerrRateLimitTimerExpire()
{
    NS_LOG_FUNCTION(this);
    m_rerrCount = 0;
    m_rerrRateLimitTimer.Schedule(Seconds(1));
}

void
RoutingProtocol::AckTimerExpire(Ipv4Address neighbor, Time blacklistTimeout)
{
    NS_LOG_FUNCTION(this);
    m_routingTable.MarkLinkAsUnidirectional(neighbor, blacklistTimeout);
}

void
RoutingProtocol::SendHello()
{
    NS_LOG_FUNCTION(this);

    // Broadcast a RREP with TTL = 1 (Hello)
    for (std::map<Ptr<Socket>, Ipv4InterfaceAddress>::const_iterator j = m_socketAddresses.begin();
         j != m_socketAddresses.end();
         ++j)
    {
        Ptr<Socket> socket = j->first;
        Ipv4InterfaceAddress iface = j->second;

        // Lifetime = AllowedHelloLoss * HelloInterval  （均为 Time，可直接相乘）
        RrepHeader helloHeader(/*prefixSize=*/0,
                               /*hopCount=*/0,
                               /*dst=*/iface.GetLocal(),
                               /*dstSeqNo=*/m_seqNo,
                               /*origin=*/iface.GetLocal(),
                               /*lifetime=*/m_allowedHelloLoss * m_helloInterval);

        // ---- 位置坐标 ----
        Ptr<MobilityModel> mobility = m_ipv4->GetObject<MobilityModel>();
        if (mobility == nullptr)
        {
            NS_LOG_ERROR("No MobilityModel installed on this node!");
            return; // 或者改成 continue; 取决于希望其余接口是否继续发
        }
        Vector3D pos = mobility->GetPosition();
        helloHeader.SetLocation(pos.x, pos.y, pos.z);

        // ---- 节点能量 ----
        if (m_energySource)
        {
            double remainingEnergy = m_energySource->GetRemainingEnergy();
            helloHeader.SetEnergy(remainingEnergy);
        }else{
        // std::cout<<(" rrep mei bang ding ");
    }

        // ---- CITR（由接口测得）----
        double citrValue = GetCitr(iface);
        helloHeader.SetCITR(citrValue);

        // ---- MaxQ（邻居侧统计）----
        helloHeader.SetMaxQ(m_nb.GetMaxQ());

        // ---- 单跳时延：SetDelay 需要 ns3::Time；发送时再计算差值----
        helloHeader.SetDelay(Simulator::Now());

        // 组包并以 TTL=1 广播
        Ptr<Packet> packet = Create<Packet>();
        SocketIpTtlTag tag;
        tag.SetTtl(1);
        packet->AddPacketTag(tag);
        packet->AddHeader(helloHeader);
        TypeHeader tHeader(QoarTYPE_RREP);
        packet->AddHeader(tHeader);

        // /32 则全网广播；否则子网广播
        Ipv4Address destination = (iface.GetMask() == Ipv4Mask::GetOnes())
                                  ? Ipv4Address("255.255.255.255")
                                  : iface.GetBroadcast();

        Time jitter = MilliSeconds(m_uniformRandomVariable->GetInteger(0, 10));
        Simulator::Schedule(jitter, &RoutingProtocol::SendTo, this, socket, packet, destination);
    }
}

void
RoutingProtocol::SendPacketFromQueue(Ipv4Address dst, Ptr<Ipv4Route> route)
{
    NS_LOG_FUNCTION(this);
    QueueEntry queueEntry;
    while (m_queue.Dequeue(dst, queueEntry))
    {
        DeferredRouteOutputTag tag;
        Ptr<Packet> p = ConstCast<Packet>(queueEntry.GetPacket());
        if (p->RemovePacketTag(tag) && tag.GetInterface() != -1 &&
            tag.GetInterface() != m_ipv4->GetInterfaceForDevice(route->GetOutputDevice()))
        {
            NS_LOG_DEBUG("Output device doesn't match. Dropped.");
            return;
        }
        UnicastForwardCallback ucb = queueEntry.GetUnicastForwardCallback();
        Ipv4Header header = queueEntry.GetIpv4Header();
        header.SetSource(route->GetSource());
        header.SetTtl(header.GetTtl() +
                      1); // compensate extra TTL decrement by fake loopback routing
        ucb(route, p, header);
    }
}

void
RoutingProtocol::SendRerrWhenBreaksLinkToNextHop(Ipv4Address nextHop)
{
    NS_LOG_FUNCTION(this << nextHop);
    RerrHeader rerrHeader;
    std::vector<Ipv4Address> precursors;
    std::map<Ipv4Address, uint32_t> unreachable;

    RoutingTableEntry toNextHop;
    if (!m_routingTable.LookupRoute(nextHop, toNextHop))
    {
        return;
    }
    toNextHop.GetPrecursors(precursors);
    rerrHeader.AddUnDestination(nextHop, toNextHop.GetSeqNo());
    m_routingTable.GetListOfDestinationWithNextHop(nextHop, unreachable);
    for (std::map<Ipv4Address, uint32_t>::const_iterator i = unreachable.begin();
         i != unreachable.end();)
    {
        if (!rerrHeader.AddUnDestination(i->first, i->second))
        {
            NS_LOG_LOGIC("Send RERR message with maximum size.");
            TypeHeader typeHeader(QoarTYPE_RERR);
            Ptr<Packet> packet = Create<Packet>();
            SocketIpTtlTag tag;
            tag.SetTtl(1);
            packet->AddPacketTag(tag);
            packet->AddHeader(rerrHeader);
            packet->AddHeader(typeHeader);
            SendRerrMessage(packet, precursors);
            rerrHeader.Clear();
        }
        else
        {
            RoutingTableEntry toDst;
            m_routingTable.LookupRoute(i->first, toDst);
            toDst.GetPrecursors(precursors);
            ++i;
        }
    }
    if (rerrHeader.GetDestCount() != 0)
    {
        TypeHeader typeHeader(QoarTYPE_RERR);
        Ptr<Packet> packet = Create<Packet>();
        SocketIpTtlTag tag;
        tag.SetTtl(1);
        packet->AddPacketTag(tag);
        packet->AddHeader(rerrHeader);
        packet->AddHeader(typeHeader);
        SendRerrMessage(packet, precursors);
    }
    unreachable.insert(std::make_pair(nextHop, toNextHop.GetSeqNo()));
    m_routingTable.InvalidateRoutesWithDst(unreachable);
}

void
RoutingProtocol::SendRerrWhenNoRouteToForward(Ipv4Address dst,
                                              uint32_t dstSeqNo,
                                              Ipv4Address origin)
{
    NS_LOG_FUNCTION(this);
    // A node SHOULD NOT originate more than RERR_RATELIMIT RERR messages per second.
    if (m_rerrCount == m_rerrRateLimit)
    {
        // Just make sure that the RerrRateLimit timer is running and will expire
        NS_ASSERT(m_rerrRateLimitTimer.IsRunning());
        // discard the packet and return
        NS_LOG_LOGIC("RerrRateLimit reached at "
                     << Simulator::Now().As(Time::S) << " with timer delay left "
                     << m_rerrRateLimitTimer.GetDelayLeft().As(Time::S) << "; suppressing RERR");
        return;
    }
    RerrHeader rerrHeader;
    rerrHeader.AddUnDestination(dst, dstSeqNo);
    RoutingTableEntry toOrigin;
    Ptr<Packet> packet = Create<Packet>();
    SocketIpTtlTag tag;
    tag.SetTtl(1);
    packet->AddPacketTag(tag);
    packet->AddHeader(rerrHeader);
    packet->AddHeader(TypeHeader(QoarTYPE_RERR));
    if (m_routingTable.LookupValidRoute(origin, toOrigin))
    {
        Ptr<Socket> socket = FindSocketWithInterfaceAddress(toOrigin.GetInterface());
        NS_ASSERT(socket);
        NS_LOG_LOGIC("Unicast RERR to the source of the data transmission");
        socket->SendTo(packet, 0, InetSocketAddress(toOrigin.GetNextHop(), Qoar_PORT));
    }
    else
    {
        for (std::map<Ptr<Socket>, Ipv4InterfaceAddress>::const_iterator i =
                 m_socketAddresses.begin();
             i != m_socketAddresses.end();
             ++i)
        {
            Ptr<Socket> socket = i->first;
            Ipv4InterfaceAddress iface = i->second;
            NS_ASSERT(socket);
            NS_LOG_LOGIC("Broadcast RERR message from interface " << iface.GetLocal());
            // Send to all-hosts broadcast if on /32 addr, subnet-directed otherwise
            Ipv4Address destination;
            if (iface.GetMask() == Ipv4Mask::GetOnes())
            {
                destination = Ipv4Address("255.255.255.255");
            }
            else
            {
                destination = iface.GetBroadcast();
            }
            socket->SendTo(packet->Copy(), 0, InetSocketAddress(destination, Qoar_PORT));
        }
    }
}

void
RoutingProtocol::SendRerrMessage(Ptr<Packet> packet, std::vector<Ipv4Address> precursors)
{
    NS_LOG_FUNCTION(this);

    if (precursors.empty())
    {
        NS_LOG_LOGIC("No precursors");
        return;
    }
    // A node SHOULD NOT originate more than RERR_RATELIMIT RERR messages per second.
    if (m_rerrCount == m_rerrRateLimit)
    {
        // Just make sure that the RerrRateLimit timer is running and will expire
        NS_ASSERT(m_rerrRateLimitTimer.IsRunning());
        // discard the packet and return
        NS_LOG_LOGIC("RerrRateLimit reached at "
                     << Simulator::Now().As(Time::S) << " with timer delay left "
                     << m_rerrRateLimitTimer.GetDelayLeft().As(Time::S) << "; suppressing RERR");
        return;
    }
    // If there is only one precursor, RERR SHOULD be unicast toward that precursor
    if (precursors.size() == 1)
    {
        RoutingTableEntry toPrecursor;
        if (m_routingTable.LookupValidRoute(precursors.front(), toPrecursor))
        {
            Ptr<Socket> socket = FindSocketWithInterfaceAddress(toPrecursor.GetInterface());
            NS_ASSERT(socket);
            NS_LOG_LOGIC("one precursor => unicast RERR to "
                         << toPrecursor.GetDestination() << " from "
                         << toPrecursor.GetInterface().GetLocal());
            Simulator::Schedule(Time(MilliSeconds(m_uniformRandomVariable->GetInteger(0, 10))),
                                &RoutingProtocol::SendTo,
                                this,
                                socket,
                                packet,
                                precursors.front());
            m_rerrCount++;
        }
        return;
    }

    //  Should only transmit RERR on those interfaces which have precursor nodes for the broken
    //  route
    std::vector<Ipv4InterfaceAddress> ifaces;
    RoutingTableEntry toPrecursor;
    for (std::vector<Ipv4Address>::const_iterator i = precursors.begin(); i != precursors.end();
         ++i)
    {
        if (m_routingTable.LookupValidRoute(*i, toPrecursor) &&
            std::find(ifaces.begin(), ifaces.end(), toPrecursor.GetInterface()) == ifaces.end())
        {
            ifaces.push_back(toPrecursor.GetInterface());
        }
    }

    for (std::vector<Ipv4InterfaceAddress>::const_iterator i = ifaces.begin(); i != ifaces.end();
         ++i)
    {
        Ptr<Socket> socket = FindSocketWithInterfaceAddress(*i);
        NS_ASSERT(socket);
        NS_LOG_LOGIC("Broadcast RERR message from interface " << i->GetLocal());
        // std::cout << "Broadcast RERR message from interface " << i->GetLocal () << std::endl;
        // Send to all-hosts broadcast if on /32 addr, subnet-directed otherwise
        Ptr<Packet> p = packet->Copy();
        Ipv4Address destination;
        if (i->GetMask() == Ipv4Mask::GetOnes())
        {
            destination = Ipv4Address("255.255.255.255");
        }
        else
        {
            destination = i->GetBroadcast();
        }
        Simulator::Schedule(Time(MilliSeconds(m_uniformRandomVariable->GetInteger(0, 10))),
                            &RoutingProtocol::SendTo,
                            this,
                            socket,
                            p,
                            destination);
    }
}

Ptr<Socket>
RoutingProtocol::FindSocketWithInterfaceAddress(Ipv4InterfaceAddress addr) const
{
    NS_LOG_FUNCTION(this << addr);
    for (std::map<Ptr<Socket>, Ipv4InterfaceAddress>::const_iterator j = m_socketAddresses.begin();
         j != m_socketAddresses.end();
         ++j)
    {
        Ptr<Socket> socket = j->first;
        Ipv4InterfaceAddress iface = j->second;
        if (iface == addr)
        {
            return socket;
        }
    }
    Ptr<Socket> socket;
    return socket;
}

Ptr<Socket>
RoutingProtocol::FindSubnetBroadcastSocketWithInterfaceAddress(Ipv4InterfaceAddress addr) const
{
    NS_LOG_FUNCTION(this << addr);
    for (std::map<Ptr<Socket>, Ipv4InterfaceAddress>::const_iterator j =
             m_socketSubnetBroadcastAddresses.begin();
         j != m_socketSubnetBroadcastAddresses.end();
         ++j)
    {
        Ptr<Socket> socket = j->first;
        Ipv4InterfaceAddress iface = j->second;
        if (iface == addr)
        {
            return socket;
        }
    }
    Ptr<Socket> socket;
    return socket;
}

void
RoutingProtocol::DoInitialize()
{
    NS_LOG_FUNCTION(this);
    uint32_t startTime;
    if (m_enableHello)
    {
        m_htimer.SetFunction(&RoutingProtocol::HelloTimerExpire, this);
        startTime = m_uniformRandomVariable->GetInteger(0, 100);
        NS_LOG_DEBUG("Starting at time " << startTime << "ms");
        m_htimer.Schedule(MilliSeconds(startTime));
    }
    Ipv4RoutingProtocol::DoInitialize();
}

std::string
RoutingProtocol::GetNodeAddressString()
{
  // 获取节点的主IP地址并转换为字符串
  std::ostringstream oss;
  oss << m_ipv4->GetAddress(1, 0).GetLocal();//获取2.4GhzIP地址
  return oss.str();
}

std::string
RoutingProtocol::GetAddressString(Ipv4Address addr)
{
  // 将Ipv4Address转换为字符串
  std::ostringstream oss;
  oss << addr;
  return oss.str();
}



} // namespace qoar
} // namespace ns3
