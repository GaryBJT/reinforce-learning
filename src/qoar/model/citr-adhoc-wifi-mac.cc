/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

#include "citr-adhoc-wifi-mac.h"

#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/pointer.h"          // PointerValue
#include "ns3/txop.h"             // Txop
#include "ns3/wifi-mac.h"
#include "ns3/wifi-mac-queue.h"
#include "ns3/wifi-mpdu.h"

#include <algorithm>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("CitrAdhocWifiMac");
NS_OBJECT_ENSURE_REGISTERED (CitrAdhocWifiMac);

TypeId
CitrAdhocWifiMac::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::CitrAdhocWifiMac")
    .SetParent<AdhocWifiMac> ()
    .SetGroupName ("Wifi")
    .AddConstructor<CitrAdhocWifiMac> ()
    .AddAttribute ("UpdateInterval",
                   "Update interval for queue-idleness metric (CITR).",
                   TimeValue (Seconds (0.5)),
                   MakeTimeAccessor (&CitrAdhocWifiMac::m_updateInterval),
                   MakeTimeChecker ())
    .AddAttribute ("QueueCapacity",
                   "Logical capacity to normalize queue length (packets).",
                   UintegerValue (100),
                   MakeUintegerAccessor (&CitrAdhocWifiMac::m_queueCapacity),
                   MakeUintegerChecker<uint32_t> (1))
    .AddAttribute ("Smoothing",
                   "Exponential smoothing factor in (0,1], 1=no smoothing.",
                   DoubleValue (0.6),
                   MakeDoubleAccessor (&CitrAdhocWifiMac::m_smoothing),
                   MakeDoubleChecker<double> (1e-6, 1.0))
    // 关键修复：AddTraceSource 需要“回调签名字符串”（第4个参数）
    .AddTraceSource ("Citr",
                     "Queue idleness (1 - normalized length), updated periodically.",
                     MakeTraceSourceAccessor (&CitrAdhocWifiMac::m_citrTrace),
                     "ns3::CitrAdhocWifiMac::CitrTracedCallback")
    ;
  return tid;
}

CitrAdhocWifiMac::CitrAdhocWifiMac ()
  : m_queueLen (0),
    m_citr (1.0),
    m_inited (false),
    m_hasQueueSignals (false)
{
  NS_LOG_FUNCTION (this);
}

CitrAdhocWifiMac::~CitrAdhocWifiMac ()
{
  NS_LOG_FUNCTION (this);
  m_updateEvent.Cancel ();
}

void
CitrAdhocWifiMac::DoInitialize ()
{
  AdhocWifiMac::DoInitialize ();
  StartCitrMeasurement ();
}

void
CitrAdhocWifiMac::StartCitrMeasurement ()
{
  NS_LOG_FUNCTION (this);
  if (m_inited)
    {
      return;
    }
  m_inited = true;

  // 1) 取默认 Txop（非 QoS 接入）
  m_txop = GetTxop ();

  // 2) 通过 Attribute "Queue" 取到 WifiMacQueue
  m_queue = nullptr;
  if (m_txop)
    {
      PointerValue pv;
      bool ok = m_txop->GetAttributeFailSafe ("Queue", pv);
      if (!ok)
        {
          // 兼容极少数版本：无 FailSafe 则直接 GetAttribute（如果可用）
          m_txop->GetAttribute ("Queue", pv);
        }
      Ptr<Object> obj = pv.Get<Object> ();
      if (obj)
        {
          m_queue = obj->GetObject<WifiMacQueue> ();
        }
    }

  // 3) 连接队列 trace（若成功拿到队列）
  m_hasQueueSignals = false;
  if (m_queue)
    {
      m_hasQueueSignals = true;
      m_queue->TraceConnectWithoutContext ("Enqueue",
        MakeCallback (&CitrAdhocWifiMac::OnEnqueue, this));
      m_queue->TraceConnectWithoutContext ("Dequeue",
        MakeCallback (&CitrAdhocWifiMac::OnDequeue, this));
      m_queue->TraceConnectWithoutContext ("Drop",
        MakeCallback (&CitrAdhocWifiMac::OnDrop, this));
      m_queueLen = ReadQueueLen ();
    }
  else
    {
      NS_LOG_WARN ("[CitrAdhocWifiMac] Could not obtain WifiMacQueue via Attribute 'Queue'; "
                   "falling back to polling-only mode (no enqueue/dequeue traces).");
    }

  // 4) 启动周期更新（保证 > 0）
  if (m_updateInterval.IsStrictlyPositive ())
    {
      m_updateEvent = Simulator::Schedule (m_updateInterval,
                                           &CitrAdhocWifiMac::UpdateCitr, this);
    }
}

bool
CitrAdhocWifiMac::AttachQueueTraces ()
{
  // 已在 StartCitrMeasurement() 中处理；保留旧调用路径
  return (m_queue != nullptr);
}

uint32_t
CitrAdhocWifiMac::ReadQueueLen () const
{
  if (m_queue)
    {
      // ns-3.37：以 MPDU 计数
      return m_queue->GetNPackets ();
    }
  return m_queueLen;
}

void
CitrAdhocWifiMac::OnEnqueue (Ptr<const WifiMpdu> /*mpdu*/)
{
  if (m_queue) { m_queueLen = m_queue->GetNPackets (); }
  else         { ++m_queueLen; }
}

void
CitrAdhocWifiMac::OnDequeue (Ptr<const WifiMpdu> /*mpdu*/)
{
  if (m_queue) { m_queueLen = m_queue->GetNPackets (); }
  else         { if (m_queueLen > 0) --m_queueLen; }
}

void
CitrAdhocWifiMac::OnDrop (Ptr<const WifiMpdu> /*mpdu*/)
{
  if (m_queue) { m_queueLen = m_queue->GetNPackets (); }
}

void
CitrAdhocWifiMac::UpdateCitr ()
{
  NS_LOG_FUNCTION (this);

  const uint32_t qLenNow = ReadQueueLen ();
  const double   cap     = static_cast<double> (m_queueCapacity > 0 ? m_queueCapacity : 1);

  const double normLen = std::min (1.0, qLenNow / cap);
  const double idleNow = 1.0 - normLen;

  const double alpha = m_smoothing; // (0,1], 1=不平滑
  m_citr = alpha * idleNow + (1.0 - alpha) * m_citr;

  // 对外 Trace 回调
  m_citrTrace (m_citr);

  NS_LOG_DEBUG ("[CITR-Queue] qLen=" << qLenNow
                << " idleNow=" << idleNow
                << " smoothed=" << m_citr);

  if (m_updateInterval.IsStrictlyPositive ())
    {
      m_updateEvent = Simulator::Schedule (m_updateInterval,
                                           &CitrAdhocWifiMac::UpdateCitr, this);
    }
}

double
CitrAdhocWifiMac::GetCitr () const
{
  return m_citr;
}

double
CitrAdhocWifiMac::GetCitr (uint32_t nodeId) const
{
  NS_LOG_DEBUG ("Node " << nodeId << " queue-idleness: " << m_citr);
  return m_citr;
}

} // namespace ns3

