/*
 * QOAR协议无人机网络仿真程序（NS3.38兼容版）
 * 描述：使用QOAR路由协议模拟3D空间中的无人机通信网络
 * 适用于NS3.38版本
 */

// aodv比较
#include "ns3/aodv-module.h"
#include "ns3/applications-module.h"
#include "ns3/citr-adhoc-wifi-mac.h" // 包含CITR MAC头文件
#include "ns3/config-store-module.h"
#include "ns3/core-module.h"
#include "ns3/energy-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/netanim-module.h"
#include "ns3/network-module.h"
#include "ns3/qoar-module.h"
#include "ns3/qoar-qlearning.h"
#include "ns3/wifi-module.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("QoarSimulation");

// 定期打印
void
PrintSimulationTime()
{
    std::cout << "当前仿真时间: " << Simulator::Now().GetSeconds() << " 秒" << std::endl;
    Simulator::Schedule(Seconds(20.0), &PrintSimulationTime);
}

// 路由开销统计（按端口区间识别应用流量）
void
PrintRoutingOverheadStats(Ptr<FlowMonitor> flowMonitor,
                          Ptr<Ipv4FlowClassifier> classifier,
                          uint16_t basePort,
                          uint32_t nNodes)
{
    uint32_t totalRoutingPackets = 0;
    uint32_t totalRoutingBytes = 0;
    uint32_t totalAppPackets = 0;
    uint32_t totalAppBytes = 0;

    auto stats = flowMonitor->GetFlowStats();
    uint16_t minAppPort = basePort;
    uint16_t maxAppPort = basePort + static_cast<uint16_t>(nNodes) - 1;

    for (auto it = stats.begin(); it != stats.end(); ++it)
    {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(it->first);
        if (t.protocol != 17)
            continue; // 只看UDP

        // bool isApp = (t.destinationPort >= minAppPort && t.destinationPort <= maxAppPort);
        bool isApp = t.destinationPort == maxAppPort;
        if (isApp)
        {
            totalAppPackets += it->second.txPackets;
            totalAppBytes += it->second.txBytes;
        }
        else
        {
            totalRoutingPackets += it->second.txPackets;
            totalRoutingBytes += it->second.txBytes;
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
        double normalizedPacketOverhead = (double)totalRoutingPackets / totalAppPackets;
        double normalizedByteOverhead = (double)totalRoutingBytes / totalAppBytes;
        std::cout << "归一化包开销: " << normalizedPacketOverhead << " 控制包/数据包\n";
        std::cout << "归一化字节开销: " << normalizedByteOverhead << " 控制字节/数据字节\n";
    }

    double routingRatio = 0.0;
    if (totalRoutingPackets + totalAppPackets > 0)
    {
        routingRatio =
            (double)totalRoutingPackets / (totalRoutingPackets + totalAppPackets) * 100.0;
    }
    std::cout << "路由开销比例: " << routingRatio << "%\n";
    std::cout << "=================================\n";
}

int
main(int argc, char* argv[])
{
    // ==================== 仿真参数设置 ====================
    uint32_t nNodes = 30;   // 无人机节点数量
    double simTime = 200.0; // 仿真时间（秒）
    double txPower = 16.0;  // 传输功率（dBm）
    double rxPower = -80.0; // 接收灵敏度（dBm）
    double maxSpeed = 40.0; // 最大速度（米/秒）  30
    double alpha = 0.01, gamma = 0.9, a = 0.5, b = 0.0, c = 0.5;
    int seed = 11111;

    // 应用端口起始（每个节点的EchoServer端口 = basePort + 节点索引）
    uint16_t basePort = 9000;

    CommandLine cmd(__FILE__);
    cmd.AddValue("nNodes", "无人机节点数量", nNodes);
    cmd.AddValue("simTime", "仿真时间（秒）", simTime);
    cmd.AddValue("Alpha", "学习率", alpha);
    cmd.AddValue("Gamma", "折扣因子", gamma);
    cmd.AddValue("A", "Q值更新参数A", a);
    cmd.AddValue("B", "Q值更新参数B", b);
    cmd.AddValue("C", "Q值更新参数C", c);
    cmd.AddValue("seed", "种子", seed);
    cmd.AddValue("speed", "速度", maxSpeed);
    cmd.AddValue("basePort", "应用端口起始", basePort);
    cmd.Parse(argc, argv);

    SeedManager::SetSeed(seed);

    Config::SetDefault("ns3::qoar::RoutingProtocol::Alpha", DoubleValue(alpha));
    Config::SetDefault("ns3::qoar::RoutingProtocol::Gamma", DoubleValue(gamma));
    Config::SetDefault("ns3::qoar::RoutingProtocol::A", DoubleValue(a));
    Config::SetDefault("ns3::qoar::RoutingProtocol::B", DoubleValue(b));
    Config::SetDefault("ns3::qoar::RoutingProtocol::C", DoubleValue(c));
    Config::SetDefault("ns3::qoar::RoutingProtocol::MaxSpeed", UintegerValue(maxSpeed));
    Config::SetDefault("ns3::qoar::RoutingProtocol::NodeCount", UintegerValue(nNodes));

    LogComponentEnable("QoarSimulation", LOG_LEVEL_INFO);

    // ==================== 创建节点 ====================
    NS_LOG_INFO("创建" << nNodes << "个无人机节点...");
    NodeContainer nodes;
    nodes.Create(nNodes);
    NS_LOG_INFO("完成创建节点数：" << nodes.GetN());

    // ==================== 配置3D移动 ====================
    NS_LOG_INFO("配置3D随机路点移动模型...");
    Ptr<RandomBoxPositionAllocator> positionAlloc =
        CreateObject<RandomBoxPositionAllocator>(); // 角度
    Ptr<UniformRandomVariable> xPos = CreateObject<UniformRandomVariable>();
    xPos->SetAttribute("Min", DoubleValue(0.0));
    xPos->SetAttribute("Max", DoubleValue(2000.0));
    positionAlloc->SetX(xPos);

    Ptr<UniformRandomVariable> yPos = CreateObject<UniformRandomVariable>();
    yPos->SetAttribute("Min", DoubleValue(0.0));
    yPos->SetAttribute("Max", DoubleValue(2000.0));
    positionAlloc->SetY(yPos);

    Ptr<UniformRandomVariable> zPos = CreateObject<UniformRandomVariable>();
    zPos->SetAttribute("Min", DoubleValue(20.0)); // 50-70
    zPos->SetAttribute("Max", DoubleValue(120.0));
    positionAlloc->SetZ(zPos);

    std::stringstream ssSpeed;
    ssSpeed << "ns3::UniformRandomVariable[Min=0.0|Max=" << maxSpeed << "]";
    std::stringstream ssPause;
    ssPause << "ns3::ConstantRandomVariable[Constant=0.0]";

    MobilityHelper mobilityAdhoc;
    mobilityAdhoc.SetPositionAllocator(positionAlloc);
    mobilityAdhoc.SetMobilityModel("ns3::RandomWaypointMobilityModel",
                                   "Speed",
                                   StringValue(ssSpeed.str()),
                                   "Pause",
                                   StringValue(ssPause.str()),
                                   "PositionAllocator",
                                   PointerValue(positionAlloc));
    mobilityAdhoc.Install(nodes);

    // ==================== 配置Wi-Fi ====================
    NS_LOG_INFO("配置IEEE 802.11b Wi-Fi网络...");
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211n);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode",
                                 StringValue("DsssRate2Mbps"),
                                 "ControlMode",
                                 StringValue("DsssRate1Mbps"));

    YansWifiPhyHelper wifiPhy;
    YansWifiChannelHelper wifiChannel;
    wifiChannel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    wifiChannel.AddPropagationLoss("ns3::FriisPropagationLossModel");
    wifiPhy.SetErrorRateModel("ns3::NistErrorRateModel");
    wifiPhy.SetChannel(wifiChannel.Create());
    wifiPhy.Set("TxPowerStart", DoubleValue(txPower));
    wifiPhy.Set("TxPowerEnd", DoubleValue(txPower));
    wifiPhy.Set("TxPowerLevels", UintegerValue(1));
    wifiPhy.Set("TxGain", DoubleValue(1));
    wifiPhy.Set("RxGain", DoubleValue(1));
    wifiPhy.Set("RxSensitivity", DoubleValue(rxPower));
    // WifiHelper wifiN;
    // wifiN.SetStandard (WIFI_STANDARD_80211n);
    // wifiN.SetRemoteStationManager ("ns3::MinstrelHtWifiManager");

    // YansWifiChannelHelper chN = YansWifiChannelHelper::Default ();
    // YansWifiPhyHelper     phyN;
    // phyN.SetChannel (chN.Create ());
    // phyN.Set ("ChannelSettings", StringValue ("{6, 20, BAND_2_4GHZ, 0}"));
    // phyN.Set ("TxPowerStart", DoubleValue (txPower));
    // phyN.Set ("TxPowerEnd",   DoubleValue (txPower));
    // phyN.Set ("TxPowerLevels", UintegerValue (1));
    // phyN.Set ("RxSensitivity", DoubleValue (rxPower));
    // phyN.SetErrorRateModel ("ns3::NistErrorRateModel");

    // WifiMacHelper macN;
    // // 若无 CITR，请改为：macN.SetType("ns3::AdhocWifiMac");
    // macN.SetType ("ns3::CitrAdhocWifiMac",
    //               "UpdateInterval", TimeValue (Seconds (1.0)));

    // NetDeviceContainer devN = wifiN.Install (phyN, macN, nodes);

    WifiMacHelper wifiMac;
    // wifiMac.SetType("ns3::CitrAdhocWifiMac", "UpdateInterval", TimeValue(Seconds(1.0)));
    wifiMac.SetType("ns3::AdhocWifiMac");
    NetDeviceContainer devices = wifi.Install(wifiPhy, wifiMac, nodes);

    // ==================== 安装协议栈（QOAR） ====================
    NS_LOG_INFO("安装Internet协议栈...");
    // InternetStackHelper internet;
    // QoarHelper qoar;
    // internet.SetRoutingHelper(qoar);
    // internet.Install(nodes);
    AodvHelper aodv;
    InternetStackHelper internet;
    internet.SetRoutingHelper(aodv); // 使用 AODV 作为路由协议
    internet.Install(nodes);

    // ==================== 分配IP地址 ====================
    NS_LOG_INFO("分配IP地址...");
    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = ipv4.Assign(devices);

    // ==================== 设置应用（每节点既收又发） ====================
    NS_LOG_INFO("配置UDP Echo：每节点既是Server也是Client（全互联）...");
    Ptr<UniformRandomVariable> jitter = CreateObject<UniformRandomVariable>();
    jitter->SetAttribute("Min", DoubleValue(0.0));
    jitter->SetAttribute("Max", DoubleValue(0.5)); // 避免同一时刻爆发

    // 定义目的节点为最后一个节点
    uint32_t dst = nNodes - 1;
    Ipv4Address dstAddr = interfaces.GetAddress(dst);
    uint16_t dstPort = basePort + dst;
    
    // 安装服务端
    UdpEchoServerHelper echoServer(dstPort);
    ApplicationContainer serverApps = echoServer.Install(nodes.Get(dst));
    serverApps.Start(Seconds(1.0));
    serverApps.Stop(Seconds(simTime));

    // 4. 安装多个客户端应用，指向同一目的节点
    uint32_t nSources = 1;
    std::vector<uint32_t> candidates;
    for (uint32_t i = 0; i < nNodes - 1; ++i)
    {
        candidates.push_back(i);
    }
    std::random_shuffle(candidates.begin(), candidates.end());
    std::vector<uint32_t> sources(candidates.begin(),
                                  candidates.begin() + std::min(nSources, nNodes - 1));
    // 5. 安装客户端应用
    std::vector<ApplicationContainer> clients;
    clients.reserve(nSources);

    for (uint32_t src : sources)
    {
        UdpEchoClientHelper echoClient(dstAddr, dstPort);
        echoClient.SetAttribute("MaxPackets", UintegerValue(10000000));
        echoClient.SetAttribute("Interval", TimeValue(Seconds(0.1)));
        echoClient.SetAttribute("PacketSize", UintegerValue(500));

        ApplicationContainer app = echoClient.Install(nodes.Get(src));
        double st = 2.0 + jitter->GetValue(); // 分散启动时间
        app.Start(Seconds(st));
        app.Stop(Seconds(simTime - 1));
        clients.push_back(app);

        std::cout << "Node " << src << " -> Node " << dst << " started at " << st << "s"
                  << std::endl;
    }

    // 为每个节点安装一个 EchoServer，端口 = basePort + 节点索引
    // std::vector<ApplicationContainer> serverApps(nNodes);
    // for (uint32_t i = 0; i < nNodes; ++i) {
    //   UdpEchoServerHelper echoServer(basePort + i);
    //   serverApps[i] = echoServer.Install(nodes.Get(i));
    //   serverApps[i].Start(Seconds(1.0));
    //   serverApps[i].Stop(Seconds(simTime));
    // }

    // 为每个节点安装 EchoClient，指向“所有其他节点”的服务器
    // std::vector<ApplicationContainer> clientApps;
    // clientApps.reserve(nNodes * (nNodes - 1));
    // for (uint32_t src = 0; src < nNodes; ++src) {
    //   for (uint32_t dst = 0; dst < nNodes; ++dst) {
    //     if (src == dst) continue;
    //     uint16_t dstPort = basePort + dst;
    //     UdpEchoClientHelper echoClient(interfaces.GetAddress(dst), dstPort);
    //     echoClient.SetAttribute("MaxPackets", UintegerValue(10000000));
    //     echoClient.SetAttribute("Interval", TimeValue(Seconds(0.1)));
    //     echoClient.SetAttribute("PacketSize", UintegerValue(500));

    //     ApplicationContainer app = echoClient.Install(nodes.Get(src));
    //     double st = 2.0 + jitter->GetValue(); // 分散启动
    //     app.Start(Seconds(st));
    //     app.Stop(Seconds(simTime - 1));
    //     clientApps.push_back(app);
    //   }
    // }

    // ==================== 监控 ====================
    NS_LOG_INFO("配置流量监控...");
    FlowMonitorHelper flowHelper;
    Ptr<FlowMonitor> flowMonitor = flowHelper.InstallAll();

    Simulator::Schedule(Seconds(10.0), &PrintSimulationTime);

    // ==================== 运行仿真 ====================
    NS_LOG_INFO("开始运行仿真，仿真时间：" << simTime << "秒");
    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    // ==================== 结果统计（全网聚合） ====================
    NS_LOG_INFO("收集仿真结果...");
    flowMonitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(flowHelper.GetClassifier());
    std::map<FlowId, FlowMonitor::FlowStats> stats = flowMonitor->GetFlowStats();

    double totalTxPackets = 0;
    double totalRxPackets = 0;
    double totalDelay = 0;
    double totalJitter = 0;

    uint16_t minAppPort = basePort;
    uint16_t maxAppPort = basePort + static_cast<uint16_t>(nNodes) - 1;

    for (auto it = stats.begin(); it != stats.end(); ++it)
    {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(it->first);
        if (t.protocol != 17)
            continue; // 只看UDP
        // 只统计 Echo 应用流（目标端口在区间内）
        // if (t.destinationPort < minAppPort || t.destinationPort > maxAppPort)
        //     continue;
        if (t.destinationPort != dstPort)
        {
            continue;
        }
        totalTxPackets += it->second.txPackets;
        totalRxPackets += it->second.rxPackets;
        totalDelay += it->second.delaySum.GetSeconds();
        totalJitter += it->second.jitterSum.GetSeconds();

        // 可选：打印单流信息（量大时请注释掉）
        // std::cout << "流 " << it->first << " (" << t.sourceAddress << " -> " <<
        // t.destinationAddress
        //           << ":" << t.destinationPort << ")\n";
    }

    if (totalTxPackets > 0 && totalRxPackets > 0)
    {
        double PDR = (totalRxPackets / totalTxPackets) * 100.0;
        double avgDelay = totalDelay / totalRxPackets;
        double avgJitter = totalJitter / totalRxPackets;

        std::cout << "\n===== QOAR仿真结果（全网聚合） =====\n";
        std::cout << "alpha=" << alpha << " gamma=" << gamma << " a=" << a << " b=" << b
                  << " c=" << c << " nNodes=" << nNodes << " flows=" << (nNodes * (nNodes - 1))
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

    // 打印路由开销（按端口区间区分应用/路由）
    PrintRoutingOverheadStats(flowMonitor, classifier, basePort, nNodes);

    Simulator::Destroy();
    NS_LOG_INFO("仿真结束");
    return 0;
}
