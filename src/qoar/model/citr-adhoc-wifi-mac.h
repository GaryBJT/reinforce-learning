/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
#ifndef CITR_ADHOC_WIFI_MAC_H
#define CITR_ADHOC_WIFI_MAC_H

#include "ns3/adhoc-wifi-mac.h"
#include "ns3/nstime.h"
#include "ns3/event-id.h"
#include "ns3/traced-callback.h"
#include "ns3/type-id.h"

namespace ns3 {

class WifiMacQueue;
class WifiMpdu;
class Txop;  // ns-3.37: 默认接入使用 Txop

class CitrAdhocWifiMac : public AdhocWifiMac
{
public:
  static TypeId GetTypeId (void);
  CitrAdhocWifiMac ();
  ~CitrAdhocWifiMac () override;

  void DoInitialize () override;

  double GetCitr () const;
  double GetCitr (uint32_t nodeId) const;

  // 为 AddTraceSource 的“回调签名字符串”提供一个明确的 typedef
  typedef TracedCallback<double> CitrTracedCallback;

protected:
  void StartCitrMeasurement ();
  bool AttachQueueTraces ();
  uint32_t ReadQueueLen () const;

  // ns-3.37 队列事件：单位是 MPDU
  void OnEnqueue (Ptr<const WifiMpdu>);
  void OnDequeue (Ptr<const WifiMpdu>);
  void OnDrop    (Ptr<const WifiMpdu>);

  void UpdateCitr ();

private:
  // Attributes
  Time     m_updateInterval;   // 更新周期
  uint32_t m_queueCapacity;    // 队列容量（用于归一化）
  double   m_smoothing;        // 指数平滑系数 (0,1]，1=不平滑

  // Internal
  EventId  m_updateEvent;
  Ptr<WifiMacQueue> m_queue;   // 通过 Attribute "Queue" 获取
  Ptr<Txop>         m_txop;

  uint32_t m_queueLen;
  double   m_citr;
  bool     m_inited;
  bool     m_hasQueueSignals;

  TracedCallback<double> m_citrTrace; // 对外 trace
};

} // namespace ns3
#endif // CITR_ADHOC_WIFI_MAC_H

