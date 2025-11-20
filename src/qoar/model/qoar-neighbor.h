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

#ifndef QoarNEIGHBOR_H
#define QoarNEIGHBOR_H

#include "ns3/arp-cache.h"
#include "ns3/callback.h"
#include "ns3/ipv4-address.h"
#include "ns3/simulator.h"
#include "ns3/timer.h"
#include "qoar-dqn.h" // 包含Q学习接口头文件

#include <vector>
#include <cmath>

namespace ns3
{

class WifiMacHeader;

namespace qoar
{

class RoutingProtocol;

/**
 * \ingroup qoar
 * \brief maintain list of active neighbors
 */
class Neighbors
{
  public:
    /**
     * constructor
     * \param delay the delay time for purging the list of neighbors
     */
    Neighbors(Time delay);

    /// Neighbor description
    struct Neighbor
    {
        /// Neighbor IPv4 address
        Ipv4Address m_neighborAddress;
        /// Neighbor MAC address
        Mac48Address m_hardwareAddress;
        /// Neighbor expire time
        Time m_expireTime;
        /// Neighbor close indicator
        bool close;
        /// 前一时隙的距离记录
        double m_prevDistance;
        /// Neighbor SF stability factor
        double m_ef;    // 能量因子
        double m_sf; // 稳定性因子
        double m_bf; // 带宽因子
        double m_q;
        double m_qf;
        double m_lq;
        double m_df;
         /// 前一次更新的时间戳
        Time m_prevUpdateTime;
        /**
         * \brief Neighbor structure constructor
         *
         * \param ip Ipv4Address entry
         * \param mac Mac48Address entry
         * \param t Time expire time
         */
        Neighbor(Ipv4Address ip, Mac48Address mac, Time t)
            : m_neighborAddress(ip),
              m_hardwareAddress(mac),
              m_expireTime(t),
              close(false),
              m_prevDistance(0.0),  // 初始前次距离为0
              m_prevUpdateTime(Simulator::Now()),  // 初始化时间戳
              m_ef(0.0), // 能量因子初始值
              m_sf(0.0), // 稳定性因子初始值
              m_bf(0.0), // 带宽因子初始值
              m_q(0.0),
              m_qf(0.0),
              m_df(0.0)
        {
        }
    };

    /**
     * Return expire time for neighbor node with address addr, if exists, else return 0.
     * \param addr the IP address of the neighbor node
     * \returns the expire time for the neighbor node
     */
    Time GetExpireTime(Ipv4Address addr);
    /**
     * Check that node with address addr is neighbor
     * \param addr the IP address to check
     * \returns true if the node with IP address is a neighbor
     */
    bool IsNeighbor(Ipv4Address addr);
    /**
     * Update expire time for entry with address addr, if it exists, else add new entry
     * \param addr the IP address to check
     * \param expire the expire time for the address
     */
    void Update(Ipv4Address addr, Time expire);
     // void Update(Ipv4Address addr, Time expire, double currentDistance);
/**
 * Get SF stability factor for neighbor with address addr
 * \param addr the IP address of the neighbor node
 * \returns the SF value for the neighbor node, -1 if not found
 */


    /// Remove all expired entries
    void Purge();
    /// Schedule m_ntimer.
    void ScheduleTimer();

    /// Remove all entries
    void Clear()
    {
        m_nb.clear();
    }

    /**
     * Add ARP cache to be used to allow layer 2 notifications processing
     * \param a pointer to the ARP cache to add
     */
    void AddArpCache(Ptr<ArpCache> a);
    /**
     * Don't use given ARP cache any more (interface is down)
     * \param a pointer to the ARP cache to delete
     */
    void DelArpCache(Ptr<ArpCache> a);

    /**
     * Get callback to ProcessTxError
     * \returns the callback function
     */
    Callback<void, const WifiMacHeader&> GetTxErrorCallback() const
    {
        return m_txErrorCallback;
    }

    /**
     * Set link failure callback
     * \param cb the callback function
     */
    void SetCallback(Callback<void, Ipv4Address> cb)
    {
        m_handleLinkFailure = cb;
    }

    /**
     * Get link failure callback
     * \returns the link failure callback
     */
    Callback<void, Ipv4Address> GetCallback() const
    {
        return m_handleLinkFailure;
    }

   /**
     * Calculate relative velocity with neighbor
     * \param addr neighbor address
     * \returns relative velocity (positive means approaching, negative means receding)
     */
    double GetRelativeVelocity(Ipv4Address addr, double currentDistance);
    double CalculateAdaptiveHelloInterval(double Rd, Ipv4Address currentNode);
    bool IsMovingAway(Ipv4Address addr, double currentDistance);
    /**
     * Get current distance to neighbor
     * \param addr neighbor address
     * \returns current distance
     */
    double GetCurrentDistance(Ipv4Address addr);

   double GetSf(Ipv4Address addr);
   double GetBf(Ipv4Address addr);
   double GetEf(Ipv4Address addr);
   double Getdf(Ipv4Address addr);
   double GetQ(Ipv4Address addr);
   double GetQf(Ipv4Address addr);
   double GetMaxQ() const { return m_maxq; };
   double GetMaxQf() const { return m_maxqf; };
   double GetLq(Ipv4Address addr);
   Ipv4Address GetDst() const { return m_destination; };
   void UpdateSf(Ipv4Address addr, double currentDistance);
   void UpdateBf(Ipv4Address addr,double bf);
   void UpdateEf(Ipv4Address addr,double energy);
   void Updatedf(Ipv4Address addr,double energy);
   void UpdateMaxQ();
   void UpdateMaxQf();
   void SetQ(Ipv4Address addr,double q);
   void SetQf(Ipv4Address addr,double q);
   void SetDst(Ipv4Address addr){ m_destination = addr;};
   void SetLq(Ipv4Address addr,double lq);
   std::vector<Ipv4Address> GetActiveNeighbors(Ipv4Address current);
  private:
    /// link failure callback
    Callback<void, Ipv4Address> m_handleLinkFailure;
    /// TX error callback
    Callback<void, const WifiMacHeader&> m_txErrorCallback;
    /// Timer for neighbor's list. Schedule Purge().
    Timer m_ntimer;
    /// vector of entries
    std::vector<Neighbor> m_nb;
    /// list of ARP cached to be used for layer 2 notifications processing
    std::vector<Ptr<ArpCache>> m_arp;
    double m_sf;
    double m_bf;
    double m_ef;
    double m_df;
    double m_q;
    double m_qf;
    double m_maxq;
    double m_maxqf;
    Ipv4Address m_destination;
    Ipv4Address m_maxqfhop;
    ns3::qoar::DQN m_qLearning; // Q-Learning接口
    bool IsActive(const Neighbor& nb);
    
    /**
     * Find MAC address by IP using list of ARP caches
     *
     * \param addr the IP address to lookup
     * \returns the MAC address for the IP address
     */
    Mac48Address LookupMacAddress(Ipv4Address addr);
    /**
     * Process layer 2 TX error notification
     * \param hdr header of the packet
     */
    void ProcessTxError(const WifiMacHeader& hdr);
};

} // namespace qoar
} // namespace ns3

#endif /* QoarNEIGHBOR_H */
