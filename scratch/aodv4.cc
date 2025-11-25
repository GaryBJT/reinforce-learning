// 多段流量+多条流
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/config-store-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/qoar-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/energy-module.h"
#include "ns3/netanim-module.h"
#include "ns3/qoar-qlearning.h"
#include "ns3/citr-adhoc-wifi-mac.h"
// #include "ns3/aodv-module.h"

#include <Python.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("QoarSimulation");

// -----------------------------------------------------------------------------
// 工具与结构
// -----------------------------------------------------------------------------

struct IfPair
{
  Ptr<Ipv4> ipv4;
  uint32_t  if24;
  uint32_t  if5;
};

static inline std::string
IpToString (const Ipv4Address &addr)
{
  std::ostringstream oss;
  oss << addr;
  return oss.str ();
}

static inline bool
HasPrefix (const std::string &s, const std::string &p)
{
  return s.rfind (p, 0) == 0;
}

static inline uint64_t
KeyPair (uint32_t me, uint32_t nh)
{
  return (static_cast<uint64_t>(me) << 32) | nh;
}

// -----------------------------------------------------------------------------
// RlSteeringRouting：包装下层（QoAR）路由，按跳选择 2.4G/5G
// -----------------------------------------------------------------------------

class RlSteeringRouting : public Ipv4RoutingProtocol
{
public:
  static TypeId GetTypeId ()
  {
    static TypeId tid = TypeId ("ns3::RlSteeringRouting")
                          .SetParent<Ipv4RoutingProtocol> ();
    return tid;
  }

  RlSteeringRouting () {}
  ~RlSteeringRouting () override {}
  //初始化路由封装器
  void Configure (
    Ptr<Ipv4RoutingProtocol> lower,
    const std::vector<IfPair> &pairs,
    const std::vector<Ipv4Address> &addr24,
    const std::vector<Ipv4Address> &addr5,
    Ptr<FlowMonitor> fm,
    Ptr<Ipv4FlowClassifier> clf,
    uint16_t basePort,
    uint32_t nNodes)
  {
    m_lower    = lower;
    m_pairs    = pairs;
    m_addr24   = addr24;
    m_addr5    = addr5;
    m_fm       = fm;
    m_clf      = clf;
    m_basePort = basePort;
    m_nNodes   = nNodes;

    for (uint32_t i = 0; i < nNodes; ++i)
      {
        m_ip2idx[IpToString (addr24[i])] = i;
        m_ip2idx[IpToString (addr5[i])]  = i;
      }
  }

  // Ipv4RoutingProtocol -------------------------------------------------------
  void SetIpv4 (Ptr<Ipv4> ipv4) override
  {
    m_ipv4 = ipv4;
    
  }
  //fa bao
  Ptr<Ipv4Route>
  RouteOutput (Ptr<Packet> p,
               const Ipv4Header &header,
               Ptr<NetDevice> oif,
               Socket::SocketErrno &err) override
  {
    if (!m_lower)
      {
        return nullptr;
      }

    Ptr<Ipv4Route> route = m_lower->RouteOutput (p, header, oif, err);
    if (!route)
      {
        return nullptr;
      }

    Ipv4Address gw = route->GetGateway ();
    if (gw == Ipv4Address ())
      {
        gw = header.GetDestination ();
      }
    
    // std::cout<<gw<<"fabao"<<std::endl;

    uint32_t nh = IndexOfIp (gw);
    if (nh == UINT32_MAX)
      {
        return route;
      }
    int action=route->GetAction();
    // if(action==-1){
    //   std::cout<<"shi bai "<<std::endl;
    // }else{
    //   std::cout<<"action "<<action<<std::endl;
    // }
    uint32_t me = m_ipv4->GetObject<Node> ()->GetId ();
    // std::vector<double> obs = CollectObsFor (me);
    // int action = PyChooseBand (obs); // 0=2.4, 1=5

    ApplyBandToRoute (route, me, nh, action, header.GetDestination ());
    return route;
  }
  //shou bao zhuan fa 
  bool
  RouteInput (Ptr<const Packet> packet,
              const Ipv4Header &header,
              Ptr<const NetDevice> idev,
              UnicastForwardCallback ucb,
              MulticastForwardCallback mcb,
              LocalDeliverCallback lcb,
              ErrorCallback ecb) override
  {
    if (!m_lower)
      {
        return false;
      }

    struct Binder
    {
      static void Do (
        Ptr<RlSteeringRouting> self,
        UnicastForwardCallback   orig,
        Ptr<Ipv4Route>           route,
        Ptr<const Packet>        pkt,
        const Ipv4Header        &hdr)
      {
        Ipv4Address gw = route->GetGateway ();
        if (gw == Ipv4Address ())
          {
            gw = hdr.GetDestination ();
          }

        uint32_t nh = self->IndexOfIp (gw);
        if (nh != UINT32_MAX)
          {
            uint32_t me = self->m_ipv4->GetObject<Node> ()->GetId ();
            int action=route->GetAction();
            self->ApplyBandToRoute (route, me, nh, action, hdr.GetDestination ());
          }

        orig (route, pkt, hdr);
      }
    };

    UnicastForwardCallback steerUcb =
      MakeBoundCallback (&Binder::Do, Ptr<RlSteeringRouting> (this), ucb);

    return m_lower->RouteInput (packet, header, idev, steerUcb, mcb, lcb, ecb);
  }

  void NotifyInterfaceUp (uint32_t i) override
  {
    if (m_lower)
      {
        m_lower->NotifyInterfaceUp (i);
      }
  }

  void NotifyInterfaceDown (uint32_t i) override
  {
    if (m_lower)
      {
        m_lower->NotifyInterfaceDown (i);
      }
  }

  void NotifyAddAddress (uint32_t i, Ipv4InterfaceAddress addr) override
  {
    if (m_lower)
      {
        m_lower->NotifyAddAddress (i, addr);
      }
  }

  void NotifyRemoveAddress (uint32_t i, Ipv4InterfaceAddress addr) override
  {
    if (m_lower)
      {
        m_lower->NotifyRemoveAddress (i, addr);
      }
  }

  void PrintRoutingTable (Ptr<OutputStreamWrapper> stream,
                          Time::Unit              unit) const override
  {
    if (m_lower)
      {
        m_lower->PrintRoutingTable (stream, unit);
      }
  }

private:

  // 切换日志 ---------------------------------------------------------------

  void MaybeLogSwitch (uint32_t       me,
                       uint32_t       nh,
                       int            newBand,
                       const Ipv4Address &dst,
                       const Ipv4Address &gwForThisBand)
  {
    uint64_t key = KeyPair (me, nh);

    auto it = m_lastBand.find (key);
    if (it == m_lastBand.end ())
      {
        // 首次建立 steer 状态，不打印“切换”，仅记录当前 band
        m_lastBand[key] = newBand;
        return;
      }

    int prevBand = it->second;
    if (prevBand == newBand)
      {
        // 未发生切换，不打印
        return;
      }

    // 发生切换：记录并打印
    m_lastBand[key] = newBand;

    double now = Simulator::Now ().GetSeconds ();

    std::ostringstream oss;
    oss.setf (std::ios::fixed);
    oss << std::setprecision (3) << now << "s "
        << "[STEER] node=" << me
        << " nh=" << nh
        << " dst=" << IpToString (dst)
        << " band=" << (newBand ? "5G" : "2.4G")
        << " (prev=" << (prevBand ? "5G" : "2.4G") << ")"
        << " gw=" << IpToString (gwForThisBand);

    NS_LOG_UNCOND (oss.str ());
  }

  // 将“选制式”落到 Ipv4Route 上 ------------------------------------------

  void ApplyBandToRoute (Ptr<Ipv4Route> route,
                         uint32_t       me,
                         uint32_t       nh,
                         int            action,
                         Ipv4Address    dst)
  {
    const IfPair &ifp = m_pairs[me];

    // 先确定这跳将要使用的网关地址（对应制式），用于日志
    Ipv4Address gwAddr = (action == 0) ? m_addr24[nh] : m_addr5[nh];

    // 切换日志：只在 band 改变时打印（含当前时间）
    // MaybeLogSwitch (me, nh, action, dst, gwAddr);

    if (action == 0)
      {
        route->SetOutputDevice (m_ipv4->GetNetDevice (ifp.if24));
        route->SetSource (m_ipv4->GetAddress (ifp.if24, 0).GetLocal ());
        route->SetGateway (m_addr24[nh]); // 邻居 2.4G 地址
      }
    else
      {
        route->SetOutputDevice (m_ipv4->GetNetDevice (ifp.if5));
        route->SetSource (m_ipv4->GetAddress (ifp.if5, 0).GetLocal ());
        route->SetGateway (m_addr5[nh]);  // 邻居 5G 地址
      }
  }

  // 观测收集（保留 FlowMonitor 口径） --------------------------------------

  double NodeSpeed (uint32_t idx) const
  {
    Ptr<Node>          n   = m_pairs[idx].ipv4->GetObject<Node> ();
    Ptr<MobilityModel> mob = n->GetObject<MobilityModel> ();
    return mob ? mob->GetVelocity ().GetLength () : 0.0;
  }

  uint32_t IndexOfIp (const Ipv4Address &a) const
  {
    auto it = m_ip2idx.find (IpToString (a));
    return (it == m_ip2idx.end ()) ? UINT32_MAX : it->second;
  }

private:
  Ptr<Ipv4>                      m_ipv4;
  Ptr<Ipv4RoutingProtocol>       m_lower;
  std::vector<IfPair>            m_pairs;
  std::vector<Ipv4Address>       m_addr24;
  std::vector<Ipv4Address>       m_addr5;
  std::unordered_map<std::string, uint32_t> m_ip2idx;

  Ptr<FlowMonitor>               m_fm;
  Ptr<Ipv4FlowClassifier>        m_clf;
  std::map<FlowId, FlowMonitor::FlowStats> m_prev;
  Time                           m_last;
  uint16_t                       m_basePort {9000};
  uint32_t                       m_nNodes {0};

  // 最近一次对 (me, nh) 的制式选择（用于判定“是否发生切换”）
  std::unordered_map<uint64_t, int> m_lastBand;

  PyObject                      *m_py {nullptr};
};

// -----------------------------------------------------------------------------
// 原有统计与工具（保持不变）
// -----------------------------------------------------------------------------

static void
PrintSimulationTime ()
{
  std::cout << "当前仿真时间: "
            << Simulator::Now ().GetSeconds ()
            << " 秒\n";

  Simulator::Schedule (Seconds (20.0), &PrintSimulationTime);
}
static EnergySourceContainer sources;
// 打印能量的函数
void PrintRemainingEnergy() {

  for (uint32_t i = 0; i < sources.GetN(); ++i) {
    Ptr<BasicEnergySource> src = DynamicCast<BasicEnergySource>(sources.Get(i));
    double energy = src->GetRemainingEnergy();
    std::cout << "Time: " << Simulator::Now().GetSeconds()
              << "s, Node[" << i << "] Remaining energy: "
              << energy << " J" << std::endl;
  }

  // 每 20 秒再次调用
  Simulator::Schedule(Seconds(20.0), &PrintRemainingEnergy);
}

static void
PrintRoutingOverheadStats (Ptr<FlowMonitor>        flowMonitor,
                           Ptr<Ipv4FlowClassifier> classifier,
                           uint16_t                basePort,
                           uint32_t                nNodes)
{
  uint32_t totalRoutingPackets = 0;
  uint32_t totalRoutingBytes   = 0;
  uint32_t totalAppPackets     = 0;
  uint32_t totalAppBytes       = 0;

  auto stats = flowMonitor->GetFlowStats ();

  uint16_t minAppPort = basePort;
  uint16_t maxAppPort = basePort + static_cast<uint16_t> (nNodes) - 1;

  for (auto it = stats.begin (); it != stats.end (); ++it)
    {
      Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow (it->first);
      if (t.protocol != 17)
        {
          continue; // 只看 UDP
        }

      bool isApp = (t.destinationPort >= minAppPort && t.destinationPort <= maxAppPort);

      if (isApp)
        {
          totalAppPackets += it->second.txPackets;
          totalAppBytes   += it->second.txBytes;
        }
      else
        {
          totalRoutingPackets += it->second.txPackets;
          totalRoutingBytes   += it->second.txBytes;
        }
    }

  std::cout << "\n===== QOAR路由开销统计（全网） =====\n";
  std::cout << "应用端口范围: [" << minAppPort << ", " << maxAppPort << "]\n";
  std::cout << "路由控制包数量: " << totalRoutingPackets << "\n";
  std::cout << "路由控制字节数: " << totalRoutingBytes << " 字节\n";
  std::cout << "应用层包数量: " << totalAppPackets << "\n";
  std::cout << "应用层字节数: " << totalAppBytes << " 字节\n";

  if (totalAppPackets > 0)
    {
      double npo = static_cast<double> (totalRoutingPackets) / totalAppPackets;
      double nbo = static_cast<double> (totalRoutingBytes)   / totalAppBytes;
      std::cout << "归一化包开销: " << npo << " 控制包/数据包\n";
      std::cout << "归一化字节开销: " << nbo << " 控制字节/数据字节\n";
    }

  double routingRatio = 0.0;
  if (totalRoutingPackets + totalAppPackets > 0)
    {
      routingRatio = static_cast<double> (totalRoutingPackets)
                     / (totalRoutingPackets + totalAppPackets)
                     * 100.0;
    }

  std::cout << "路由开销比例: " << routingRatio << "%\n";
  std::cout << "=================================\n";
}

static void
PrintPhyInfo (const std::string &tag, Ptr<NetDevice> dev)
{
  Ptr<WifiNetDevice> w   = dev->GetObject<WifiNetDevice> ();
  Ptr<WifiPhy>       phy = w->GetPhy ();

  UintegerValue ch;
  UintegerValue wid;

  phy->GetAttribute ("ChannelNumber", ch);
  phy->GetAttribute ("ChannelWidth",  wid);

  NS_LOG_UNCOND ("0.00s [" << tag << "] Ch=" << ch.Get ()
                << " Width=" << wid.Get () << "MHz");
}

// -----------------------------------------------------------------------------
// 主程序（其余逻辑保持不变）
// -----------------------------------------------------------------------------

int
main (int argc, char *argv[])
{
  // 参数 ---------------------------------------------------------------
  uint32_t nNodes   = 30;
  double   simTime  = 200.0;
  double   txPower  = 16.0;
  double   rxPower  = -80.0;
  double   maxSpeed = 40.0;

  double alpha = 0.01;
  double gamma = 0.9;
  double a     = 0.4;
  double b     = 0.2;
  double c     = 0.4;

  int seed = 11111;

  double   switchPeriod = 30.0;
  double   t0           = 5.0;
  uint16_t basePort     = 9000;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("nNodes",       "无人机节点数量",         nNodes);
  cmd.AddValue ("simTime",      "仿真时间（秒）",        simTime);
  cmd.AddValue ("Alpha",        "学习率",               alpha);
  cmd.AddValue ("Gamma",        "折扣因子",             gamma);
  cmd.AddValue ("A",            "Q值更新参数A",         a);
  cmd.AddValue ("B",            "Q值更新参数B",         b);
  cmd.AddValue ("C",            "Q值更新参数C",         c);
  cmd.AddValue ("seed",         "种子",                 seed);
  cmd.AddValue ("speed",        "速度",                 maxSpeed);
  cmd.AddValue ("basePort",     "应用端口起始",         basePort);
  cmd.AddValue ("switchPeriod", "应用分段窗口（秒）",    switchPeriod);
  cmd.AddValue ("t0",           "业务分段起始（秒）",    t0);
  cmd.Parse (argc, argv);

  SeedManager::SetSeed (seed);
  // c = 1 - a;

  Config::SetDefault ("ns3::qoar::RoutingProtocol::Alpha",     DoubleValue (alpha));
  Config::SetDefault ("ns3::qoar::RoutingProtocol::Gamma",     DoubleValue (gamma));
  Config::SetDefault ("ns3::qoar::RoutingProtocol::A",         DoubleValue (a));
  Config::SetDefault ("ns3::qoar::RoutingProtocol::B",         DoubleValue (b));
  Config::SetDefault ("ns3::qoar::RoutingProtocol::C",         DoubleValue (c));
  Config::SetDefault ("ns3::qoar::RoutingProtocol::MaxSpeed",  UintegerValue (maxSpeed));
  Config::SetDefault ("ns3::qoar::RoutingProtocol::NodeCount", UintegerValue (nNodes));

  LogComponentEnable ("QoarSimulation", LOG_LEVEL_INFO);

  // 节点 + 移动 -------------------------------------------------------
  NodeContainer nodes;
  nodes.Create (nNodes);

  Ptr<RandomBoxPositionAllocator> positionAlloc = CreateObject<RandomBoxPositionAllocator> ();

  auto URV = [] (double mn, double mx) {
    Ptr<UniformRandomVariable> u = CreateObject<UniformRandomVariable> ();
    u->SetAttribute ("Min", DoubleValue (mn));
    u->SetAttribute ("Max", DoubleValue (mx));
    return u;
  };

  positionAlloc->SetX (URV (0.0, 2000.0));
  positionAlloc->SetY (URV (0.0, 2000.0));
  positionAlloc->SetZ (URV (20.0, 120.0));

  std::stringstream ssSpeed;
  ssSpeed << "ns3::UniformRandomVariable[Min=0.0|Max=" << maxSpeed << "]";

  MobilityHelper mobility;
  mobility.SetPositionAllocator (positionAlloc);
  mobility.SetMobilityModel (
    "ns3::RandomWaypointMobilityModel",
    "Speed", StringValue (ssSpeed.str ()),
    "Pause", StringValue ("ns3::ConstantRandomVariable[Constant=0.0]"),
    "PositionAllocator", PointerValue (positionAlloc));
  mobility.Install (nodes);

  // 双频 Wi-Fi ---------------------------------------------------------
  WifiHelper wifiN;
  wifiN.SetStandard (WIFI_STANDARD_80211n);
  wifiN.SetRemoteStationManager ("ns3::MinstrelHtWifiManager");

  YansWifiChannelHelper chN = YansWifiChannelHelper::Default ();
  YansWifiPhyHelper     phyN;
  phyN.SetChannel (chN.Create ());
  phyN.Set ("ChannelSettings", StringValue ("{6, 20, BAND_2_4GHZ, 0}"));
  phyN.Set ("TxPowerStart", DoubleValue (txPower));
  phyN.Set ("TxPowerEnd",   DoubleValue (txPower));
  phyN.Set ("TxPowerLevels", UintegerValue (1));
  phyN.Set ("RxSensitivity", DoubleValue (rxPower));
  phyN.SetErrorRateModel ("ns3::NistErrorRateModel");

  WifiMacHelper macN;
  // 若无 CITR，请改为：macN.SetType("ns3::AdhocWifiMac");
  macN.SetType ("ns3::CitrAdhocWifiMac",
                "UpdateInterval", TimeValue (Seconds (1.0)));

  NetDeviceContainer devN = wifiN.Install (phyN, macN, nodes);

  WifiHelper wifiAC;
  wifiAC.SetStandard (WIFI_STANDARD_80211ac);
  wifiAC.SetRemoteStationManager ("ns3::MinstrelHtWifiManager");

  YansWifiChannelHelper chAC = YansWifiChannelHelper::Default ();
  YansWifiPhyHelper     phyAC;
  phyAC.SetChannel (chAC.Create ());
  phyAC.Set ("ChannelSettings", StringValue ("{58, 80, BAND_5GHZ, 0}"));
  phyAC.Set ("TxPowerStart", DoubleValue (txPower));
  phyAC.Set ("TxPowerEnd",   DoubleValue (txPower));
  phyAC.Set ("TxPowerLevels", UintegerValue (1));
  phyAC.Set ("RxSensitivity", DoubleValue (rxPower));
  phyAC.SetErrorRateModel ("ns3::NistErrorRateModel");

  WifiMacHelper macAC;
  // 若无 CITR，请改为：macAC.SetType("ns3::AdhocWifiMac");
  macAC.SetType ("ns3::CitrAdhocWifiMac",
                 "UpdateInterval", TimeValue (Seconds (1.0)));

  NetDeviceContainer devAC = wifiAC.Install (phyAC, macAC, nodes);

  // 协议栈（QoAR） -----------------------------------------------------
  InternetStackHelper internet;
  QoarHelper          qoar;

  internet.SetRoutingHelper (qoar);
  internet.Install (nodes);

  // IP 地址 ------------------------------------------------------------
  Ipv4AddressHelper ipv4;

  ipv4.SetBase ("10.1.0.0", "255.255.0.0");
  Ipv4InterfaceContainer ifN = ipv4.Assign (devN);

  ipv4.SetBase ("10.2.0.0", "255.255.0.0");
  Ipv4InterfaceContainer ifAC = ipv4.Assign (devAC);

  
  //绑定能量
  BasicEnergySourceHelper energyHelper;
  energyHelper.Set("BasicEnergySourceInitialEnergyJ", DoubleValue(800));
  sources = energyHelper.Install(nodes);
    /* device energy model */
  WifiRadioEnergyModelHelper radioEnergyHelper;
  // configure radio energy model
  radioEnergyHelper.Set("TxCurrentA", DoubleValue(5));
  // install device model
  DeviceEnergyModelContainer deviceNModels = radioEnergyHelper.Install(devN, sources);
  DeviceEnergyModelContainer deviceACModels = radioEnergyHelper.Install(devAC, sources);


  // 记录接口并保持 Up（不再 Up/Down） ---------------------------------
  std::vector<IfPair>       pairs;
  std::vector<Ipv4Address>  addrN (nNodes);
  std::vector<Ipv4Address>  addrAC (nNodes);

  pairs.reserve (nodes.GetN ());

  for (uint32_t i = 0; i < nodes.GetN (); ++i)
    {
      Ptr<Ipv4> ip = nodes.Get (i)->GetObject<Ipv4> ();

      uint32_t iN  = ip->GetInterfaceForDevice (devN.Get (i));
      uint32_t iAC = ip->GetInterfaceForDevice (devAC.Get (i));

      ip->SetUp (iN);
      ip->SetUp (iAC);

      pairs.push_back ({ip, iN, iAC});

      addrN[i]  = ifN.GetAddress (i);
      addrAC[i] = ifAC.GetAddress (i);

      PrintPhyInfo ("Node" + std::to_string (i) + "-2.4G", devN.Get (i));
      PrintPhyInfo ("Node" + std::to_string (i) + "-5G",   devAC.Get (i));
    }

  // Echo 应用（原分段） -------------------------------------------------
  for (uint32_t i = 0; i < nNodes; ++i)
    {
      UdpEchoServerHelper server (basePort + i);
      ApplicationContainer s = server.Install (nodes.Get (i));
      s.Start (Seconds (1.0));
      s.Stop  (Seconds (simTime));
    }

  Ptr<UniformRandomVariable> jitter = CreateObject<UniformRandomVariable> ();
  jitter->SetAttribute ("Min", DoubleValue (0.0));
  jitter->SetAttribute ("Max", DoubleValue (0.5));

  // auto InstallEchoClientWindow =
  //   [&] (Ptr<Node>     src,
  //        Ipv4Address   dst,
  //        uint16_t      port,
  //        double        wStart,
  //        double        wEnd) -> ApplicationContainer
  // {
  //   if (wEnd <= wStart || wStart >= simTime)
  //     {
  //       return ApplicationContainer ();
  //     }

  //   double s = std::max (wStart + jitter->GetValue (), 1.5);
  //   double e = std::min (wEnd, simTime - 1.0);

  //   if (e <= s)
  //     {
  //       return ApplicationContainer ();
  //     }

  //   UdpEchoClientHelper cli (dst, port);
  //   cli.SetAttribute ("MaxPackets", UintegerValue (10000000));
  //   cli.SetAttribute ("Interval",   TimeValue (Seconds (0.1)));
  //   cli.SetAttribute ("PacketSize", UintegerValue (500));

  //   ApplicationContainer c = cli.Install (src);
  //   c.Start (Seconds (s));
  //   c.Stop  (Seconds (e));
  //   return c;
  // };

  std::vector<ApplicationContainer> clients;
  clients.reserve (nNodes * (nNodes - 1) * 2);

  
  //每段时间切换频段发送流量？？

  // const double P   = switchPeriod;
  // const double w1s = t0;
  // const double w1e = t0 + P;     // 2.4G 段
  // const double w2s = t0 + P;
  // const double w2e = t0 + 2 * P; // 5G 段
  // const double w3s = t0 + 2 * P;
  // const double w3e = simTime;    // 2.4G 段

  // for (uint32_t s = 0; s < nNodes; ++s)
  //   {
  //     for (uint32_t d = 0; d < nNodes; ++d)
  //       {
  //         if (s == d)
  //           {
  //             continue;
  //           }

  //           uint16_t port = basePort + d;

  //           clients.push_back (
  //             InstallEchoClientWindow (nodes.Get (s), ifN.GetAddress (d), port, w1s, w1e));

  //           clients.push_back (
  //             InstallEchoClientWindow (nodes.Get (s), ifAC.GetAddress (d), port, w2s, w2e));

  //           clients.push_back (
  //             InstallEchoClientWindow (nodes.Get (s), ifN.GetAddress (d), port, w3s, w3e));
  //       }
  //   }

  for (uint32_t src = 0; src < nNodes; ++src) {
    for (uint32_t dst = 0; dst < nNodes; ++dst) {
      if (src == dst) continue;
      uint16_t dstPort = basePort + dst;
      UdpEchoClientHelper echoClient(ifN.GetAddress(dst), dstPort);
      echoClient.SetAttribute("MaxPackets", UintegerValue(10000000));
      echoClient.SetAttribute("Interval", TimeValue(Seconds(0.1)));
      echoClient.SetAttribute("PacketSize", UintegerValue(500));

      ApplicationContainer app = echoClient.Install(nodes.Get(src));
      double st = 2.0 + jitter->GetValue(); // 分散启动
      app.Start(Seconds(st));
      app.Stop(Seconds(simTime - 1));
      clients.push_back(app);
    }
  }

  // FlowMonitor --------------------------------------------------------
  FlowMonitorHelper fmHelper;
  Ptr<FlowMonitor>   fm  = fmHelper.InstallAll ();
  Ptr<Ipv4FlowClassifier> classifier =DynamicCast<Ipv4FlowClassifier> (fmHelper.GetClassifier ());

  // 用 RL 包装 QoAR：替换顶层路由为 RlSteeringRouting（内部委托 QoAR） ----
  for (uint32_t i = 0; i < nNodes; ++i)
    {
      Ptr<Node> node = nodes.Get (i);
      Ptr<Ipv4>  ip   = node->GetObject<Ipv4> ();

      Ptr<Ipv4RoutingProtocol> lower = ip->GetRoutingProtocol (); // QoAR

      Ptr<RlSteeringRouting> rl = CreateObject<RlSteeringRouting> ();
      rl->SetIpv4 (ip);
      rl->Configure (lower, pairs, addrN, addrAC, fm, classifier, basePort, nNodes);

      ip->SetRoutingProtocol (rl);
    }

  // 运行 ---------------------------------------------------------------
  Simulator::Schedule (Seconds (10.0), &PrintSimulationTime);
  // Simulator::Schedule (Seconds (10.0), &PrintRemainingEnergy);
  Simulator::Stop (Seconds (simTime));
  Simulator::Run ();

  // 结果 ---------------------------------------------------------------
  fm->CheckForLostPackets ();

  std::map<FlowId, FlowMonitor::FlowStats> stats = fm->GetFlowStats ();

  double totalTxPackets = 0.0;
  double totalRxPackets = 0.0;
  double totalDelay     = 0.0;
  double totalJitter    = 0.0;

  uint16_t minAppPort = basePort;
  uint16_t maxAppPort = basePort + static_cast<uint16_t> (nNodes) - 1;

  for (auto it = stats.begin (); it != stats.end (); ++it)
    {
      Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow (it->first);
      if (t.protocol != 17)
        {
          continue; // UDP
        }

      if (t.destinationPort < minAppPort || t.destinationPort > maxAppPort)
        {
          continue;
        }

      totalTxPackets += it->second.txPackets;
      totalRxPackets += it->second.rxPackets;
      totalDelay     += it->second.delaySum.GetSeconds ();
      totalJitter    += it->second.jitterSum.GetSeconds ();
    }
  fm->SerializeToXmlFile("test.xml", true, true);
  if (totalTxPackets > 0 && totalRxPackets > 0)
    {
      double PDR      = (totalRxPackets / totalTxPackets) * 100.0;
      double avgDelay = totalDelay  / totalRxPackets;
      double avgJitter= totalJitter / totalRxPackets;

      std::cout << "\n===== QOAR仿真结果（全网聚合） =====\n";
      std::cout << "alpha=" << alpha << " gamma=" << gamma
                << " a=" << a << " b=" << b << " c=" << c
                << " nNodes=" << nNodes
                << " flows=" << (nNodes * (nNodes - 1))
                << "\n";
      std::cout << "PDR: " << PDR << " %\n";
      std::cout << "avgDelay: " << avgDelay << " 秒\n";
      std::cout << "avgJitter: " << avgJitter << " 秒\n";
      std::cout << "=================================\n";
    }
  else
    {
      std::cout << "\n===== 仿真结果 =====\n";
      std::cout << "未收集到足够的统计数据\n";
      std::cout << "=================================\n";
    }

  PrintRoutingOverheadStats (fm, classifier, basePort, nNodes);

  Simulator::Destroy ();
  return 0;
}

