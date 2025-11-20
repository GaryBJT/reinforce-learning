/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
#include "qoar-packet.h"

#include "ns3/address-utils.h"
#include "ns3/assert.h"
#include "ns3/packet.h"

namespace ns3 {
namespace qoar {

// ---- 浮点与 uint64 的按位转换（用于网络序列化） ----
static inline uint64_t DoubleToUint64(double v) {
  union { double d; uint64_t u; } x; x.d = v; return x.u;
}
static inline double Uint64ToDouble(uint64_t u) {
  union { uint64_t u; double d; } x; x.u = u; return x.d;
}

/* ---------------- TypeHeader ---------------- */

NS_OBJECT_ENSURE_REGISTERED(TypeHeader);

TypeId
TypeHeader::GetTypeId()
{
  static TypeId tid = TypeId("ns3::qoar::TypeHeader")
      .SetParent<Header>()
      .SetGroupName("Qoar")
      .AddConstructor<TypeHeader>();
  return tid;
}

TypeHeader::TypeHeader(MessageType t)
  : m_type(t),
    m_valid(true)
{}

TypeId
TypeHeader::GetInstanceTypeId() const
{
  return GetTypeId();
}

uint32_t
TypeHeader::GetSerializedSize() const
{
  return 1;
}

void
TypeHeader::Serialize(Buffer::Iterator i) const
{
  i.WriteU8(static_cast<uint8_t>(m_type));
}

uint32_t
TypeHeader::Deserialize(Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  const uint8_t t = i.ReadU8();
  m_valid = true;
  switch (t) {
    case QoarTYPE_RREQ:
    case QoarTYPE_RREP:
    case QoarTYPE_RERR:
    case QoarTYPE_RREP_ACK:
      m_type = static_cast<MessageType>(t);
      break;
    default:
      m_valid = false;
      break;
  }
  const uint32_t dist = i.GetDistanceFrom(start);
  NS_ASSERT(dist == GetSerializedSize());
  return dist;
}

void
TypeHeader::Print(std::ostream& os) const
{
  switch (m_type) {
    case QoarTYPE_RREQ:     os << "RREQ"; break;
    case QoarTYPE_RREP:     os << "RREP"; break;
    case QoarTYPE_RERR:     os << "RERR"; break;
    case QoarTYPE_RREP_ACK: os << "RREP_ACK"; break;
    default:                os << "UNKNOWN_TYPE"; break;
  }
}

bool
TypeHeader::operator==(const TypeHeader& o) const
{
  return (m_type == o.m_type && m_valid == o.m_valid);
}

std::ostream&
operator<<(std::ostream& os, const TypeHeader& h)
{
  h.Print(os);
  return os;
}

/* ---------------- RREQ ---------------- */

NS_OBJECT_ENSURE_REGISTERED(RreqHeader);

TypeId
RreqHeader::GetTypeId()
{
  static TypeId tid = TypeId("ns3::qoar::RreqHeader")
      .SetParent<Header>()
      .SetGroupName("Qoar")
      .AddConstructor<RreqHeader>();
  return tid;
}

RreqHeader::RreqHeader(uint8_t flags,
                       uint8_t reserved,
                       uint8_t hopCount,
                       uint32_t requestID,
                       Ipv4Address dst,
                       uint32_t dstSeqNo,
                       Ipv4Address origin,
                       uint32_t originSeqNo,
                       double m_energy,
                       double m_locationX,
                       double m_locationY,
                       double m_locationZ,
                       double m_citr,
                       double m_maxq,
                       uint32_t m_delay)
  : m_flags(flags),
    m_reserved(reserved),
    m_hopCount(hopCount),
    m_requestID(requestID),
    m_dst(dst),
    m_dstSeqNo(dstSeqNo),
    m_origin(origin),
    m_originSeqNo(originSeqNo),
    m_energy(m_energy),
    m_locationX(m_locationX),
    m_locationY(m_locationY),
    m_locationZ(m_locationZ),
    m_citr(m_citr),
    m_max_q(m_maxq),   // 注意：形参名是 m_maxq，成员是 m_max_q
    m_delay(m_delay)   // 以毫秒存储
{}

TypeId
RreqHeader::GetInstanceTypeId() const
{
  return GetTypeId();
}

/**
 * AODV 基础字段 23 字节 + 6 个 double(8B) + delay(4B)
 */
uint32_t
RreqHeader::GetSerializedSize() const
{
  return 23 + 6 * sizeof(double) + 4;
}

void
RreqHeader::Serialize(Buffer::Iterator i) const
{
  i.WriteU8(m_flags);
  i.WriteU8(m_reserved);
  i.WriteU8(m_hopCount);
  i.WriteHtonU32(m_requestID);
  WriteTo(i, m_dst);
  i.WriteHtonU32(m_dstSeqNo);
  WriteTo(i, m_origin);
  i.WriteHtonU32(m_originSeqNo);

  i.WriteHtonU64(DoubleToUint64(m_energy));
  i.WriteHtonU64(DoubleToUint64(m_locationX));
  i.WriteHtonU64(DoubleToUint64(m_locationY));
  i.WriteHtonU64(DoubleToUint64(m_locationZ));
  i.WriteHtonU64(DoubleToUint64(m_citr));
  i.WriteHtonU64(DoubleToUint64(m_max_q));

  i.WriteHtonU32(m_delay);
}

uint32_t
RreqHeader::Deserialize(Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  m_flags     = i.ReadU8();
  m_reserved  = i.ReadU8();
  m_hopCount  = i.ReadU8();
  m_requestID = i.ReadNtohU32();
  ReadFrom(i, m_dst);
  m_dstSeqNo  = i.ReadNtohU32();
  ReadFrom(i, m_origin);
  m_originSeqNo = i.ReadNtohU32();

  m_energy    = Uint64ToDouble(i.ReadNtohU64());
  m_locationX = Uint64ToDouble(i.ReadNtohU64());
  m_locationY = Uint64ToDouble(i.ReadNtohU64());
  m_locationZ = Uint64ToDouble(i.ReadNtohU64());
  m_citr      = Uint64ToDouble(i.ReadNtohU64());
  m_max_q     = Uint64ToDouble(i.ReadNtohU64());

  m_delay     = i.ReadNtohU32();

  const uint32_t dist = i.GetDistanceFrom(start);
  NS_ASSERT(dist == GetSerializedSize());
  return dist;
}

void
RreqHeader::Print(std::ostream& os) const
{
  os << "RREQ ID " << m_requestID
     << " destination: ipv4 " << m_dst << " sequence number " << m_dstSeqNo
     << " source: ipv4 " << m_origin << " sequence number " << m_originSeqNo
     << " flags: Gratuitous RREP " << GetGratuitousRrep()
     << " Destination only " << GetDestinationOnly()
     << " Unknown sequence number " << GetUnknownSeqno();
}

std::ostream&
operator<<(std::ostream& os, const RreqHeader& h)
{
  h.Print(os);
  return os;
}

void
RreqHeader::SetGratuitousRrep(bool f)
{
  if (f) m_flags |= (1 << 5);
  else   m_flags &= ~(1 << 5);
}

bool
RreqHeader::GetGratuitousRrep() const
{
  return (m_flags & (1 << 5));
}

void
RreqHeader::SetDestinationOnly(bool f)
{
  if (f) m_flags |= (1 << 4);
  else   m_flags &= ~(1 << 4);
}

bool
RreqHeader::GetDestinationOnly() const
{
  return (m_flags & (1 << 4));
}

void
RreqHeader::SetUnknownSeqno(bool f)
{
  if (f) m_flags |= (1 << 3);
  else   m_flags &= ~(1 << 3);
}

bool
RreqHeader::GetUnknownSeqno() const
{
  return (m_flags & (1 << 3));
}

bool
RreqHeader::operator==(const RreqHeader& o) const
{
  return (m_flags == o.m_flags &&
          m_reserved == o.m_reserved &&
          m_hopCount == o.m_hopCount &&
          m_requestID == o.m_requestID &&
          m_dst == o.m_dst &&
          m_dstSeqNo == o.m_dstSeqNo &&
          m_origin == o.m_origin &&
          m_originSeqNo == o.m_originSeqNo);
}

/* ---------------- RREP ---------------- */

NS_OBJECT_ENSURE_REGISTERED(RrepHeader);

TypeId
RrepHeader::GetTypeId()
{
  static TypeId tid = TypeId("ns3::qoar::RrepHeader")
      .SetParent<Header>()
      .SetGroupName("Qoar")
      .AddConstructor<RrepHeader>();
  return tid;
}

RrepHeader::RrepHeader(uint8_t prefixSize,
                       uint8_t hopCount,
                       Ipv4Address dst,
                       uint32_t dstSeqNo,
                       Ipv4Address origin,
                       Time lifetime,
                       double m_energy,
                       double m_locationX,
                       double m_locationY,
                       double m_locationZ,
                       double m_citr,
                       double m_maxq,
                       uint32_t m_delay)
  : m_flags(0),
    m_prefixSize(prefixSize),
    m_hopCount(hopCount),
    m_dst(dst),
    m_dstSeqNo(dstSeqNo),
    m_origin(origin),
    m_energy(m_energy),
    m_locationX(m_locationX),
    m_locationY(m_locationY),
    m_locationZ(m_locationZ),
    m_citr(m_citr),
    m_max_q(m_maxq),     // 形参 m_maxq -> 成员 m_max_q
    m_delay(m_delay)
{
  m_lifeTime = static_cast<uint32_t>(lifetime.GetMilliSeconds());
}

TypeId
RrepHeader::GetInstanceTypeId() const
{
  return GetTypeId();
}

/**
 * AODV 基础字段 19 字节 + 6 个 double(8B) + delay(4B)
 */
uint32_t
RrepHeader::GetSerializedSize() const
{
  return 19 + 6 * sizeof(double) + 4;
}

void
RrepHeader::Serialize(Buffer::Iterator i) const
{
  i.WriteU8(m_flags);
  i.WriteU8(m_prefixSize);
  i.WriteU8(m_hopCount);
  WriteTo(i, m_dst);
  i.WriteHtonU32(m_dstSeqNo);
  WriteTo(i, m_origin);
  i.WriteHtonU32(m_lifeTime);

  i.WriteHtonU64(DoubleToUint64(m_energy));
  i.WriteHtonU64(DoubleToUint64(m_locationX));
  i.WriteHtonU64(DoubleToUint64(m_locationY));
  i.WriteHtonU64(DoubleToUint64(m_locationZ));
  i.WriteHtonU64(DoubleToUint64(m_citr));
  i.WriteHtonU64(DoubleToUint64(m_max_q));

  i.WriteHtonU32(m_delay);
}

uint32_t
RrepHeader::Deserialize(Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  m_flags      = i.ReadU8();
  m_prefixSize = i.ReadU8();
  m_hopCount   = i.ReadU8();
  ReadFrom(i, m_dst);
  m_dstSeqNo   = i.ReadNtohU32();
  ReadFrom(i, m_origin);
  m_lifeTime   = i.ReadNtohU32();

  m_energy    = Uint64ToDouble(i.ReadNtohU64());
  m_locationX = Uint64ToDouble(i.ReadNtohU64());
  m_locationY = Uint64ToDouble(i.ReadNtohU64());
  m_locationZ = Uint64ToDouble(i.ReadNtohU64());
  m_citr      = Uint64ToDouble(i.ReadNtohU64());
  m_max_q     = Uint64ToDouble(i.ReadNtohU64());

  m_delay     = i.ReadNtohU32();

  const uint32_t dist = i.GetDistanceFrom(start);
  NS_ASSERT(dist == GetSerializedSize());
  return dist;
}

void
RrepHeader::Print(std::ostream& os) const
{
  os << "destination: ipv4 " << m_dst
     << " sequence number " << m_dstSeqNo;
  if (m_prefixSize != 0) {
    os << " prefix size " << m_prefixSize;
  }
  os << " source ipv4 " << m_origin
     << " lifetime " << m_lifeTime
     << " acknowledgment required flag " << GetAckRequired();
}

void
RrepHeader::SetLifeTime(Time t)
{
  m_lifeTime = t.GetMilliSeconds();
}

Time
RrepHeader::GetLifeTime() const
{
  return MilliSeconds(m_lifeTime);
}

void
RrepHeader::SetAckRequired(bool f)
{
  if (f) m_flags |= (1 << 6);
  else   m_flags &= ~(1 << 6);
}

bool
RrepHeader::GetAckRequired() const
{
  return (m_flags & (1 << 6));
}

void
RrepHeader::SetPrefixSize(uint8_t sz)
{
  m_prefixSize = sz;
}

uint8_t
RrepHeader::GetPrefixSize() const
{
  return m_prefixSize;
}

/**
 * 注意：你头文件中 SetHello 的参数顺序是
 * (..., max_q, citr, delay) —— 这里严格按该顺序实现
 */
void
RrepHeader::SetHello(Ipv4Address src,
                     uint32_t srcSeqNo,
                     Time lifetime,
                     double energy,
                     double locationX,
                     double locationY,
                     double locationZ,
                     double max_q,
                     double citr,
                     Time delay)
{
  m_flags      = 0;
  m_prefixSize = 0;
  m_hopCount   = 0;
  m_dst        = src;
  m_dstSeqNo   = srcSeqNo;
  m_origin     = src;
  m_lifeTime   = lifetime.GetMilliSeconds();

  m_energy     = energy;
  m_locationX  = locationX;
  m_locationY  = locationY;
  m_locationZ  = locationZ;
  m_max_q      = max_q;              // 先 max_q
  m_citr       = citr;               // 再 citr
  m_delay      = delay.GetMilliSeconds();
}

std::ostream&
operator<<(std::ostream& os, const RrepHeader& h)
{
  h.Print(os);
  return os;
}

bool
RrepHeader::operator==(const RrepHeader& o) const
{
  return (m_flags == o.m_flags &&
          m_prefixSize == o.m_prefixSize &&
          m_hopCount == o.m_hopCount &&
          m_dst == o.m_dst &&
          m_dstSeqNo == o.m_dstSeqNo &&
          m_origin == o.m_origin &&
          m_lifeTime == o.m_lifeTime);
}

/* ---------------- RrepAckHeader ---------------- */

NS_OBJECT_ENSURE_REGISTERED(RrepAckHeader);

TypeId
RrepAckHeader::GetTypeId()
{
  static TypeId tid = TypeId("ns3::qoar::RrepAckHeader")
      .SetParent<Header>()
      .SetGroupName("Qoar")
      .AddConstructor<RrepAckHeader>();
  return tid;
}

RrepAckHeader::RrepAckHeader()
  : m_reserved(0)
{}

TypeId
RrepAckHeader::GetInstanceTypeId() const
{
  return GetTypeId();
}

uint32_t
RrepAckHeader::GetSerializedSize() const
{
  return 1;
}

void
RrepAckHeader::Serialize(Buffer::Iterator i) const
{
  i.WriteU8(m_reserved);
}

uint32_t
RrepAckHeader::Deserialize(Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  m_reserved = i.ReadU8();
  const uint32_t dist = i.GetDistanceFrom(start);
  NS_ASSERT(dist == GetSerializedSize());
  return dist;
}

void
RrepAckHeader::Print(std::ostream& os) const
{
}

bool
RrepAckHeader::operator==(const RrepAckHeader& o) const
{
  return m_reserved == o.m_reserved;
}

std::ostream&
operator<<(std::ostream& os, const RrepAckHeader& h)
{
  h.Print(os);
  return os;
}

/* ---------------- RERR ---------------- */

NS_OBJECT_ENSURE_REGISTERED(RerrHeader);

TypeId
RerrHeader::GetTypeId()
{
  static TypeId tid = TypeId("ns3::qoar::RerrHeader")
      .SetParent<Header>()
      .SetGroupName("Qoar")
      .AddConstructor<RerrHeader>();
  return tid;
}

RerrHeader::RerrHeader()
  : m_flag(0),
    m_reserved(0)
{}

TypeId
RerrHeader::GetInstanceTypeId() const
{
  return GetTypeId();
}

uint32_t
RerrHeader::GetSerializedSize() const
{
  return (3 + 8 * GetDestCount());
}

void
RerrHeader::Serialize(Buffer::Iterator i) const
{
  i.WriteU8(m_flag);
  i.WriteU8(m_reserved);
  i.WriteU8(GetDestCount());
  for (const auto& kv : m_unreachableDstSeqNo) {
    WriteTo(i, kv.first);
    i.WriteHtonU32(kv.second);
  }
}

uint32_t
RerrHeader::Deserialize(Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  m_flag     = i.ReadU8();
  m_reserved = i.ReadU8();
  const uint8_t destCount = i.ReadU8();

  m_unreachableDstSeqNo.clear();
  Ipv4Address addr; uint32_t seq;
  for (uint8_t k = 0; k < destCount; ++k) {
    ReadFrom(i, addr);
    seq = i.ReadNtohU32();
    m_unreachableDstSeqNo.insert(std::make_pair(addr, seq));
  }

  const uint32_t dist = i.GetDistanceFrom(start);
  NS_ASSERT(dist == GetSerializedSize());
  return dist;
}

void
RerrHeader::Print(std::ostream& os) const
{
  os << "Unreachable destination (ipv4 address, seq. number):";
  for (const auto& kv : m_unreachableDstSeqNo) {
    os << kv.first << ", " << kv.second;
  }
  os << "No delete flag " << GetNoDelete();
}

void
RerrHeader::SetNoDelete(bool f)
{
  if (f) m_flag |= (1 << 0);
  else   m_flag &= ~(1 << 0);
}

bool
RerrHeader::GetNoDelete() const
{
  return (m_flag & (1 << 0));
}

bool
RerrHeader::AddUnDestination(Ipv4Address dst, uint32_t seqNo)
{
  if (m_unreachableDstSeqNo.find(dst) != m_unreachableDstSeqNo.end()) {
    return true;
  }
  NS_ASSERT(GetDestCount() < 255);
  m_unreachableDstSeqNo.insert(std::make_pair(dst, seqNo));
  return true;
}

bool
RerrHeader::RemoveUnDestination(std::pair<Ipv4Address, uint32_t>& un)
{
  if (m_unreachableDstSeqNo.empty()) {
    return false;
  }
  auto it = m_unreachableDstSeqNo.begin();
  un = *it;
  m_unreachableDstSeqNo.erase(it);
  return true;
}

void
RerrHeader::Clear()
{
  m_unreachableDstSeqNo.clear();
  m_flag = 0;
  m_reserved = 0;
}

std::ostream&
operator<<(std::ostream& os, const RerrHeader& h)
{
  h.Print(os);
  return os;
}

} // namespace qoar
} // namespace ns3

