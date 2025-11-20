// qoar-dqn.h  (MAPPO 后端适配版；保留原类名与方法签名)

#ifndef QOAR_DQN_H
#define QOAR_DQN_H

#include <string>
#include <Python.h>
#include <iostream>

namespace ns3 {
namespace qoar {

/**
 * \brief MAPPO 接口类（对外保持原 DQN 类名与方法签名，以最小改动替换后端）
 *
 * 约定的 Python 侧函数（模块名：qoar_mappo）：
 *   - update_q_value(currentNode: str, nextHop: str, reward: float, done: bool) -> bool
 *   - set_mappo_params(alpha: float, gamma: float, nodes: int, speed: int) -> bool
 *   - get_best_next_hop(currentNode: str) -> str
 *   - get_q_value(currentNode: str, nextHop: str) -> float
 *
 * 说明：
 * - 以上签名保持与现有 C++ 侧调用一致，只是把算法从 DQN 换成了 MAPPO。
 * - MAPPO 背后通常采用 PPO-Clip 目标、GAE、集中式 Critic（CTDE），但这些都在 Python 内部实现。:contentReference[oaicite:1]{index=1}
 */
class DQN {
  public:
    /**
     * \brief 构造函数：初始化 Python 解释器并加载 MAPPO 模块
     */
    DQN() {
      // 若 Python 解释器尚未初始化，则初始化
      if (!Py_IsInitialized()) {
        Py_Initialize();
        initialized = true;
        std::cout << "Python解释器已初始化" << std::endl;
      }
      
      // 确保 sys.path 包含当前目录
      PyRun_SimpleString("import sys");
      PyRun_SimpleString("sys.path.append('.')");

      // === 导入 MAPPO 模块（替代原 qoar_dqn） ===
      pModule = PyImport_ImportModule("qoar_dqn");
      // pModule = PyImport_ImportModule("qoar_lstm");
      if (!pModule) {
        std::cerr << "无法导入MAPPO模块 'qoar_dqn'" << std::endl;
        PyErr_Print();
        return;
      }

      // 绑定 update_q_value
      pUpdateFunc = PyObject_GetAttrString(pModule, "update_q_value");
      if (!pUpdateFunc || !PyCallable_Check(pUpdateFunc)) {
        std::cerr << "无法找到函数 'update_q_value'" << std::endl;
        PyErr_Print();
        return;
      }

      // 绑定 set_mappo_params（替代 set_dqn_params）
      pSetParamsFunc = PyObject_GetAttrString(pModule, "set_mappo_params");
      if (!pSetParamsFunc || !PyCallable_Check(pSetParamsFunc)) {
        std::cerr << "无法找到函数 'set_mappo_params'" << std::endl;
        PyErr_Print();
        return;
      }

      // 绑定 get_best_next_hop
      pGetBestFunc = PyObject_GetAttrString(pModule, "get_best_next_hop");
      if (!pGetBestFunc || !PyCallable_Check(pGetBestFunc)) {
        std::cerr << "无法找到函数 'get_best_next_hop'" << std::endl;
        PyErr_Print();
        return;
      }
      // 绑定 UpdateLq
      pUpdateLq = PyObject_GetAttrString(pModule, "update_lq");
      if (!pUpdateLq || !PyCallable_Check(pUpdateLq)) {
        std::cerr << "无法找到函数 'UpdateLq'" << std::endl;
        PyErr_Print();
        return;
      }
      // 绑定 UpdateLq
      pPlotTrainingCurves = PyObject_GetAttrString(pModule, "plot_training_curves");
      if (!pPlotTrainingCurves || !PyCallable_Check(pPlotTrainingCurves)) {
        std::cerr << "无法找到函数 'pPlotTrainingCurves'" << std::endl;
        PyErr_Print();
        return;
      }
    }

    /**
     * \brief 析构：释放 Python 对象（不在此处终止解释器）
     */
    ~DQN() {
      Py_XDECREF(pUpdateFunc);
      Py_XDECREF(pSetParamsFunc);
      Py_XDECREF(pGetBestFunc);
      Py_XDECREF(pUpdateLq);
      Py_XDECREF(pPlotTrainingCurves);
      Py_XDECREF(pModule);
      // 解释器终止请用静态 finalizePython()（若需要全局结束）
    }

    /**
     * \brief 更新（MAPPO）价值/策略：与原 DQN 接口保持一致
     * \param currentNode 当前节点
     * \param nextHop     下一跳
     * \param reward      即时奖励
     * \param done        是否终止
     * \return Python 返回的布尔值（是否成功）
     */
    //xiugai
    bool updateQValue(double sf,
                      double ef,
                      double bf,
                      const std::string& currentNode,
                      const std::string& nextHop,
                      const std::string& destNode,
                      int band,
                      double reward,
                      bool done = false) {
      bool success = false;

      PyGILState_STATE gstate = PyGILState_Ensure();

      PyObject* pArgs = PyTuple_New(9);
      PyTuple_SetItem(pArgs, 0, PyFloat_FromDouble(sf));
      PyTuple_SetItem(pArgs, 1, PyFloat_FromDouble(ef));
      PyTuple_SetItem(pArgs, 2, PyFloat_FromDouble(bf));
      PyTuple_SetItem(pArgs, 3, PyUnicode_FromString(currentNode.c_str()));
      PyTuple_SetItem(pArgs, 4, PyUnicode_FromString(nextHop.c_str()));
      PyTuple_SetItem(pArgs, 5, PyUnicode_FromString(destNode.c_str()));
      PyTuple_SetItem(pArgs, 6, PyLong_FromLong(band));
      PyTuple_SetItem(pArgs, 7, PyFloat_FromDouble(reward));
      PyTuple_SetItem(pArgs, 8, PyBool_FromLong(done ? 1 : 0));

      PyObject* pValue = PyObject_CallObject(pUpdateFunc, pArgs);
      Py_DECREF(pArgs);

      if (pValue) {
        success = PyObject_IsTrue(pValue);
        Py_DECREF(pValue);
      } else {
        PyErr_Print();
        std::cerr << "调用 'update_q_value' 失败" << std::endl;
      }

      PyGILState_Release(gstate);
      return success;
    }

    /**
     * \brief 设置 MAPPO 参数（接口保持不变，内部调用 set_mappo_params）
     * \param alpha 学习率（用于优化器/策略更新步长）
     * \param gamma 折扣因子
     * \param a 
     * \param b 
     * \param c 
     */
    bool setParameters(double alpha, double gamma, double a, double b,double c) {
      bool success = false;

      PyGILState_STATE gstate = PyGILState_Ensure();

      PyObject* pArgs = PyTuple_New(5);
      PyTuple_SetItem(pArgs, 0, PyFloat_FromDouble(alpha));
      PyTuple_SetItem(pArgs, 1, PyFloat_FromDouble(gamma));
      PyTuple_SetItem(pArgs, 2, PyFloat_FromDouble(a));
      PyTuple_SetItem(pArgs, 3, PyFloat_FromDouble(b));
      PyTuple_SetItem(pArgs, 4, PyFloat_FromDouble(c));
      PyObject* pValue = PyObject_CallObject(pSetParamsFunc, pArgs);
      Py_DECREF(pArgs);

      if (pValue) {
        success = PyObject_IsTrue(pValue);
        Py_DECREF(pValue);
      } else {
        PyErr_Print();
        std::cerr << "调用 'set_mappo_params' 失败" << std::endl;
      }

      PyGILState_Release(gstate);
      return success;
    }

    /**
     * \brief 获取策略给定当前节点的最佳下一跳（MAPPO Actor 输出），接口保持不变
     */
    // 返回类型改为 std::pair<std::string, int>，存储 (节点名称, band)
    std::pair<std::string, int> getBestNextHop(const std::string& currentNode) {
    // 初始化返回值：默认节点为空字符串，band 为 -1（表示无效）
    std::pair<std::string, int> result("", 0);

    // 获取 GIL（全局解释器锁）
    PyGILState_STATE gstate = PyGILState_Ensure();

    // 构造参数：仅传入 currentNode（字符串类型）
    PyObject* pArgs = PyTuple_New(1);
    PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(currentNode.c_str()));

    // 调用 Python 函数 pGetBestFunc
    PyObject* pValue = PyObject_CallObject(pGetBestFunc, pArgs);
    Py_DECREF(pArgs);  // 释放参数对象

    if (pValue) {
        // 检查返回值是否为长度为 2 的元组（(节点名称, band)）
        if (PyTuple_Check(pValue) && PyTuple_Size(pValue) == 2) {
            // 解析第一个元素：节点名称（字符串）
            PyObject* pNode = PyTuple_GetItem(pValue, 0);
            if (PyUnicode_Check(pNode)) {
                const char* cstr = PyUnicode_AsUTF8(pNode);
                if (cstr) {
                    result.first = cstr;  // 赋值节点名称
                }
            }

            // 解析第二个元素：band（整数）
            PyObject* pBand = PyTuple_GetItem(pValue, 1);
            if (PyLong_Check(pBand)) {
                result.second = PyLong_AsLong(pBand);  // 赋值 band（转换为 int）
            }
        } else {
            std::cerr << "Python 函数 'get_best_next_hop' 返回值格式错误，应为 (str, int) 元组" << std::endl;
        }
        Py_DECREF(pValue);  // 释放返回值对象
    } else {
        // 函数调用失败，打印错误信息
        PyErr_Print();
        std::cerr << "调用 'get_best_next_hop' 失败" << std::endl;
    }

    // 释放 GIL
    PyGILState_Release(gstate);

    return result;
  }

    /**
     * \brief 获取“Q值/偏好分”的近似指标（由 MAPPO 模块提供：如 V(s) 或 A(s,a) 的代理值）
     *        接口保持不变：输入 (currentNode, nextHop) 返回 float
     */
    double getQValue(const std::string& currentNode, const std::string& nextHop) {
      double result = 0.0;

      PyGILState_STATE gstate = PyGILState_Ensure();

      PyObject* pQValueFunc = PyObject_GetAttrString(pModule, "get_q_value");
      if (!pQValueFunc || !PyCallable_Check(pQValueFunc)) {
        std::cerr << "无法找到函数 'get_q_value'" << std::endl;
        PyErr_Print();
        PyGILState_Release(gstate);
        return result;
      }

      PyObject* pArgs = PyTuple_New(2);
      PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(currentNode.c_str()));
      PyTuple_SetItem(pArgs, 1, PyUnicode_FromString(nextHop.c_str()));

      PyObject* pValue = PyObject_CallObject(pQValueFunc, pArgs);
      Py_DECREF(pArgs);
      Py_DECREF(pQValueFunc);

      if (pValue) {
        result = PyFloat_AsDouble(pValue);
        Py_DECREF(pValue);
      } else {
        PyErr_Print();
        std::cerr << "调用 'get_q_value' 失败" << std::endl;
      }

      PyGILState_Release(gstate);
      return result;
    }

    double updatelq(double sf, double ef, double bf,
                  const std::string& currentNode,
                  const std::string& nextHop) {
    if (!pUpdateLq) return 0.0;
   PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* args = PyTuple_New(5);
    PyTuple_SetItem(args, 0, PyFloat_FromDouble(sf));
    PyTuple_SetItem(args, 1, PyFloat_FromDouble(ef));
    PyTuple_SetItem(args, 2, PyFloat_FromDouble(bf));
    PyTuple_SetItem(args, 3, PyUnicode_FromString(currentNode.c_str()));
    PyTuple_SetItem(args, 4, PyUnicode_FromString(nextHop.c_str()));
    PyObject* ret = PyObject_CallObject(pUpdateLq, args);
    Py_DECREF(args);
    if (!ret) { std::cerr << "[Py] update_lq 调用失败\n"; PyErr_Print(); return 0.0; }
    double v = PyFloat_Check(ret) ? PyFloat_AsDouble(ret) : 0.0;
    Py_DECREF(ret);
    PyGILState_Release(gstate);
    return v;
  }
    // int randomAction() {
    // // 默认返回-1表示失败
    // int action = -1;

    // // 获取GIL（确保线程安全）
    // PyGILState_STATE gstate = PyGILState_Ensure();

    // // 该函数无参数，构造空元组
    // PyObject* pArgs = PyTuple_New(0);

    // // 调用Python函数
    // PyObject* pValue = PyObject_CallObject(pGetRandom, pArgs);
    // Py_DECREF(pArgs);  // 释放参数对象

    // if (pValue) {
    //     // 检查返回值是否为整数
    //     if (PyLong_Check(pValue)) {
    //         action = PyLong_AsLong(pValue);  // 转换为C++的int
    //     } else {
    //         std::cerr << "random_action: Python函数返回非整数类型" << std::endl;
    //     }
    //     Py_DECREF(pValue);  // 释放返回值对象
    // } else {
    //     // 调用失败，打印Python错误信息
    //     PyErr_Print();
    //     std::cerr << "调用 'get_random_action' 失败" << std::endl;
    // }

    // 释放GIL
  //   PyGILState_Release(gstate);

  //   return action;
  //  }

    void PlotTrainingCurves(const std::string& filename){
      if (!pPlotTrainingCurves) return;
      PyGILState_STATE gstate = PyGILState_Ensure();
      PyObject* args = PyTuple_New(1);
      PyTuple_SetItem(args, 0, PyUnicode_FromString(filename.c_str()));
      PyObject* ret = PyObject_CallObject(pPlotTrainingCurves, args);
      Py_DECREF(args);
      if (!ret) { std::cerr << "[Py] plot_training_curves 调用失败\n"; PyErr_Print(); }
      else { Py_DECREF(ret); }
      PyGILState_Release(gstate);
    }
    /**
     * \brief （可选）进程退出时终止 Python 解释器
     */
    static void finalizePython() {
      if (Py_IsInitialized()) {
        Py_Finalize();
        std::cout << "Python解释器已终止" << std::endl;
      }
    }

  private:
    PyObject* pModule        = nullptr; ///< Python 模块对象（qoar_mappo）
    PyObject* pUpdateFunc    = nullptr; ///< update_q_value
    PyObject* pSetParamsFunc = nullptr; ///< set_mappo_params
    PyObject* pGetBestFunc   = nullptr; ///< get_best_next_hop
    PyObject* pUpdateLq     = nullptr; ////updatelq
    PyObject* pPlotTrainingCurves = nullptr; ///< plot_training_curves
    bool initialized         = false;   ///< 是否由本对象初始化 Python
};

} // namespace qoar
} // namespace ns3

#endif /* QOAR_DQN_H */

