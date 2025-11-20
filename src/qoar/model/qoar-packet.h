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
#ifndef QoarPACKET_H
#define QoarPACKET_H

#include "ns3/enum.h"
#include "ns3/header.h"
#include "ns3/ipv4-address.h"
#include "ns3/nstime.h"

#include <iostream>
#include <map>

namespace ns3
{
namespace qoar
{

enum MessageType
{
    QoarTYPE_RREQ = 1,
    QoarTYPE_RREP = 2,
    QoarTYPE_RERR = 3,
    QoarTYPE_RREP_ACK = 4
};

class TypeHeader : public Header
{
  public:
    TypeHeader(MessageType t = QoarTYPE_RREQ);
    static TypeId GetTypeId();
    TypeId GetInstanceTypeId() const override;
    uint32_t GetSerializedSize() const override;
    void Serialize(Buffer::Iterator start) const override;
    uint32_t Deserialize(Buffer::Iterator start) override;
    void Print(std::ostream& os) const override;

    MessageType Get() const { return m_type; }
    bool IsValid() const { return m_valid; }
    bool operator==(const TypeHeader& o) const;

  private:
    MessageType m_type;
    bool m_valid;
};

std::ostream& operator<<(std::ostream& os, const TypeHeader& h);

/* ---------------- RREQ ---------------- */

class RreqHeader : public Header
{
  public:
    RreqHeader(uint8_t flags = 0,
               uint8_t reserved = 0,
               uint8_t hopCount = 0,
               uint32_t requestID = 0,
               Ipv4Address dst = Ipv4Address(),
               uint32_t dstSeqNo = 0,
               Ipv4Address origin = Ipv4Address(),
               uint32_t originSeqNo = 0,
               double m_energy = 0.0,
               double m_locationX = 0.0,
               double m_locationY = 0.0,
               double m_locationZ = 0.0,
               double m_citr = 0.0,
               double m_maxq = 0.0,
               uint32_t m_delay = 0);

    static TypeId GetTypeId();
    TypeId GetInstanceTypeId() const override;
    uint32_t GetSerializedSize() const override;
    void Serialize(Buffer::Iterator start) const override;
    uint32_t Deserialize(Buffer::Iterator start) override;
    void Print(std::ostream& os) const override;

    // 扩展字段：能量/位置/CITR/MaxQ/时延
    void SetEnergy(double energy) { m_energy = energy; }
    double GetEnergy() const { return m_energy; }
    void SetLocation(double x, double y, double z) { m_locationX = x; m_locationY = y; m_locationZ = z; }
    double GetLocationX() const { return m_locationX; }
    double GetLocationY() const { return m_locationY; }
    double GetLocationZ() const { return m_locationZ; }
    void SetCITR(double citr) { m_citr = citr; }
    double GetCITR() const { return m_citr; }
    void SetMaxQ(double max_q) { m_max_q = max_q; }
    double GetMaxQ() const { return m_max_q; }

    // 新增：时延（毫秒）
    void SetDelay(Time t) { m_delay = static_cast<uint32_t>(t.GetMilliSeconds()); }
    Time GetDelay() const { return MilliSeconds(m_delay); }
    uint32_t GetDelayMs() const { return m_delay; }
    void SetDelayMs(uint32_t ms) { m_delay = ms; }

    // 标准字段
    void SetHopCount(uint8_t count) { m_hopCount = count; }
    uint8_t GetHopCount() const { return m_hopCount; }
    void SetId(uint32_t id) { m_requestID = id; }
    uint32_t GetId() const { return m_requestID; }
    void SetDst(Ipv4Address a) { m_dst = a; }
    Ipv4Address GetDst() const { return m_dst; }
    void SetDstSeqno(uint32_t s) { m_dstSeqNo = s; }
    uint32_t GetDstSeqno() const { return m_dstSeqNo; }
    void SetOrigin(Ipv4Address a) { m_origin = a; }
    Ipv4Address GetOrigin() const { return m_origin; }
    void SetOriginSeqno(uint32_t s) { m_originSeqNo = s; }
    uint32_t GetOriginSeqno() const { return m_originSeqNo; }

    // Flags
    void SetGratuitousRrep(bool f);
    bool GetGratuitousRrep() const;
    void SetDestinationOnly(bool f);
    bool GetDestinationOnly() const;
    void SetUnknownSeqno(bool f);
    bool GetUnknownSeqno() const;

    bool operator==(const RreqHeader& o) const;

  private:
    uint8_t m_flags;
    uint8_t m_reserved;
    uint8_t m_hopCount;
    uint32_t m_requestID;
    Ipv4Address m_dst;
    uint32_t m_dstSeqNo;
    Ipv4Address m_origin;
    uint32_t m_originSeqNo;

    // 扩展域
    double m_energy;
    double m_locationX;
    double m_locationY;
    double m_locationZ;
    double m_citr;
    double m_max_q;
    uint32_t m_delay; // 毫秒
};

std::ostream& operator<<(std::ostream& os, const RreqHeader&);

/* ---------------- RREP ---------------- */

class RrepHeader : public Header
{
  public:
    RrepHeader(uint8_t prefixSize = 0,
               uint8_t hopCount = 0,
               Ipv4Address dst = Ipv4Address(),
               uint32_t dstSeqNo = 0,
               Ipv4Address origin = Ipv4Address(),
               Time lifetime = MilliSeconds(0),
               double m_energy = 0.0,
               double m_locationX = 0.0,
               double m_locationY = 0.0,
               double m_locationZ = 0.0,
               double m_citr = 0.0,
               double m_maxq = 0.0,
               uint32_t m_delay = 0);

    static TypeId GetTypeId();
    TypeId GetInstanceTypeId() const override;
    uint32_t GetSerializedSize() const override;
    void Serialize(Buffer::Iterator start) const override;
    uint32_t Deserialize(Buffer::Iterator start) override;
    void Print(std::ostream& os) const override;

    // 扩展字段：能量/位置/CITR/MaxQ/时延
    void SetEnergy(double energy) { m_energy = energy; }
    double GetEnergy() const { return m_energy; }
    void SetLocation(double x, double y, double z) { m_locationX = x; m_locationY = y; m_locationZ = z; }
    double GetLocationX() const { return m_locationX; }
    double GetLocationY() const { return m_locationY; }
    double GetLocationZ() const { return m_locationZ; }
    void SetCITR(double citr) { m_citr = citr; }
    double GetCITR() const { return m_citr; }
    void SetMaxQ(double max_q) { m_max_q = max_q; }
    double GetMaxQ() const { return m_max_q; }

    // 新增：时延（毫秒）
    void SetDelay(Time t) { m_delay = static_cast<uint32_t>(t.GetMilliSeconds()); }
    Time GetDelay() const { return MilliSeconds(m_delay); }
    uint32_t GetDelayMs() const { return m_delay; }
    void SetDelayMs(uint32_t ms) { m_delay = ms; }

    // 标准字段
    void SetHopCount(uint8_t count) { m_hopCount = count; }
    uint8_t GetHopCount() const { return m_hopCount; }
    void SetDst(Ipv4Address a) { m_dst = a; }
    Ipv4Address GetDst() const { return m_dst; }
    void SetDstSeqno(uint32_t s) { m_dstSeqNo = s; }
    uint32_t GetDstSeqno() const { return m_dstSeqNo; }
    void SetOrigin(Ipv4Address a) { m_origin = a; }
    Ipv4Address GetOrigin() const { return m_origin; }

    void SetLifeTime(Time t);
    Time GetLifeTime() const;

    // Flags
    void SetAckRequired(bool f);
    bool GetAckRequired() const;
    void SetPrefixSize(uint8_t sz);
    uint8_t GetPrefixSize() const;

    /**
     * 将 RREP 配置成 Hello 报文（扩展：新增 delay）
     * 原版签名为 (src, srcSeqNo, lifetime)，此处在末尾新增 Time delay
     */
    void SetHello(Ipv4Address src, uint32_t srcSeqNo, Time lifetime,
                  double energy, double locationX, double locationY, double locationZ,
                  double max_q, double citr, Time delay);

    bool operator==(const RrepHeader& o) const;

  private:
    uint8_t m_flags;
    uint8_t m_prefixSize;
    uint8_t m_hopCount;
    Ipv4Address m_dst;
    uint32_t m_dstSeqNo;
    Ipv4Address m_origin;
    uint32_t m_lifeTime; // 毫秒
    double m_energy;
    double m_locationX;
    double m_locationY;
    double m_locationZ;
    double m_citr;
    double m_max_q;
    uint32_t m_delay;    // 毫秒（新增）
};

std::ostream& operator<<(std::ostream& os, const RrepHeader&);

/* ---------------- RREP-ACK ---------------- */

class RrepAckHeader : public Header
{
  public:
    RrepAckHeader();
    static TypeId GetTypeId();
    TypeId GetInstanceTypeId() const override;
    uint32_t GetSerializedSize() const override;
    void Serialize(Buffer::Iterator start) const override;
    uint32_t Deserialize(Buffer::Iterator start) override;
    void Print(std::ostream& os) const override;
    bool operator==(const RrepAckHeader& o) const;

  private:
    uint8_t m_reserved;
};

std::ostream& operator<<(std::ostream& os, const RrepAckHeader&);

/* ---------------- RERR ---------------- */

class RerrHeader : public Header
{
  public:
    RerrHeader();
    static TypeId GetTypeId();
    TypeId GetInstanceTypeId() const override;
    uint32_t GetSerializedSize() const override;
    void Serialize(Buffer::Iterator i) const override;
    uint32_t Deserialize(Buffer::Iterator start) override;
    void Print(std::ostream& os) const override;

    void SetNoDelete(bool f);
    bool GetNoDelete() const;

    bool AddUnDestination(Ipv4Address dst, uint32_t seqNo);
    bool RemoveUnDestination(std::pair<Ipv4Address, uint32_t>& un);
    void Clear();

    uint8_t GetDestCount() const
    {
        return (uint8_t)m_unreachableDstSeqNo.size();
    }

    bool operator==(const RerrHeader& o) const;

  private:
    uint8_t m_flag;
    uint8_t m_reserved;
    std::map<Ipv4Address, uint32_t> m_unreachableDstSeqNo;
};

std::ostream& operator<<(std::ostream& os, const RerrHeader&);

} // namespace qoar
} // namespace ns3

#endif /* QoarPACKET_H */

