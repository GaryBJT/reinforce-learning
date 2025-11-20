#pragma once
#ifndef NS3_QOAR_QLEARNING_H_
#define NS3_QOAR_QLEARNING_H_

#include <Python.h>
#include <string>
#include <iostream>

namespace ns3 { namespace qoar {

// 仅供本头内部使用：避免与别处符号撞名
namespace detail {
static const char kPySearchDir[] =
    "/home/aoteman/ns-allinone-3.37/ns-3.37/src/qoar/python";
} // namespace detail

// RAII 的 GIL 守卫：每次跨入 Python C-API 必须持有 GIL
// 见官方文档 PyGILState_Ensure/Release 要求。:contentReference[oaicite:3]{index=3}
struct GilGuard {
  PyGILState_STATE s;
  GilGuard()  { s = PyGILState_Ensure(); }
  ~GilGuard() { PyGILState_Release(s);  }
};

/**
 * Q-Learning 接口（兼容 MAPPO 后端）
 * - 仅此头暴露完整定义；其它头（neighbor.h / rtable.h）不再包含它。
 * - 通过 GIL 守卫 + 判空 + PyErr_Print()，最大限度避免崩溃。
 */
class QLearning {
public:
  QLearning() {
    if (!Py_IsInitialized()) {
      Py_Initialize();
      std::cout << "[Py] Initialized\n";
      // Python 3.7+ 中 GIL 已在 Py_Initialize() 内初始化，
      // PyEval_InitThreads() 自 3.9 起已弃用且无效果。:contentReference[oaicite:4]{index=4}
    }

    GilGuard gil;

    // 注入 Python 搜索路径：绝对路径优先，其次当前目录
    PyRun_SimpleString("import sys, os");
    {
      std::string cmd = std::string("sys.path.insert(0, '") + detail::kPySearchDir + "')";
      PyRun_SimpleString(cmd.c_str());
      PyRun_SimpleString("sys.path.insert(0, '.')");
    }

    // 双模块回退：优先 qoar_mappo，失败回退 qoar_qlearning
    pModule = PyImport_ImportModule("qoar_mappo");
    if (!pModule) {
      PyErr_Clear();
      pModule = PyImport_ImportModule("qoar_qlearning");
    }
    if (!pModule) {
      std::cerr << "[Py] 导入模块失败：qoar_mappo/qoar_qlearning 都不可用\n";
      PyErr_Print();
      return;
    }
    if (PyObject* f = PyObject_GetAttrString(pModule, "__file__")) {
      std::cerr << "[Py] loaded: " << PyUnicode_AsUTF8(f) << "\n";
      Py_DECREF(f);
    }

    auto bind = [&](PyObject*& slot, const char* name, bool optional=false) {
      slot = PyObject_GetAttrString(pModule, name);
      if (!slot || !PyCallable_Check(slot)) {
        if (!optional) {
          std::cerr << "[Py] 缺少可调用函数: " << name << "\n";
          PyErr_Print();
        }
        Py_XDECREF(slot);
        slot = nullptr;
      }
    };

    // 预绑定全部函数
    bind(pUpdateQ,       "update_q_value");
    bind(pSetParams,     "set_qlearning_params");
    bind(pGetBest,       "get_best_next_hop");
    bind(pUpdateLq,      "update_lq");
    bind(pGetQOpt,       "get_q_value", /*optional*/true);
  }

  ~QLearning() {
    Py_XDECREF(pGetQOpt);
    Py_XDECREF(pUpdateLq);
    Py_XDECREF(pGetBest);
    Py_XDECREF(pSetParams);
    Py_XDECREF(pUpdateQ);
    Py_XDECREF(pModule);
  }

  void setParameters(double alpha, double gamma, double a, double b, double c) {
    if (!pSetParams) return;
    GilGuard gil;
    PyObject* args = PyTuple_New(5);
    // 注意：PyTuple_SetItem 会“窃取引用” —— 不要对同一对象再 DECREF。:contentReference[oaicite:5]{index=5}
    PyTuple_SetItem(args, 0, PyFloat_FromDouble(alpha));
    PyTuple_SetItem(args, 1, PyFloat_FromDouble(gamma));
    PyTuple_SetItem(args, 2, PyFloat_FromDouble(a));
    PyTuple_SetItem(args, 3, PyFloat_FromDouble(b));
    PyTuple_SetItem(args, 4, PyFloat_FromDouble(c));
    PyObject* ret = PyObject_CallObject(pSetParams, args);
    Py_DECREF(args);
    if (!ret) { std::cerr << "[Py] set_qlearning_params 调用失败\n"; PyErr_Print(); }
    else Py_DECREF(ret);
  }

  double updateQvalue(double sf, double ef, double bf,
                      const std::string& currentNode,
                      const std::string& nextHop,
                      const std::string& destNode,
                      double reward) {
    if (!pUpdateQ) return 0.0;
    GilGuard gil;
    PyObject* args = PyTuple_New(7);
    PyTuple_SetItem(args, 0, PyFloat_FromDouble(sf));
    PyTuple_SetItem(args, 1, PyFloat_FromDouble(ef));
    PyTuple_SetItem(args, 2, PyFloat_FromDouble(bf));
    PyTuple_SetItem(args, 3, PyUnicode_FromString(currentNode.c_str()));
    PyTuple_SetItem(args, 4, PyUnicode_FromString(nextHop.c_str()));
    PyTuple_SetItem(args, 5, PyUnicode_FromString(destNode.c_str()));
    PyTuple_SetItem(args, 6, PyFloat_FromDouble(reward));
    PyObject* ret = PyObject_CallObject(pUpdateQ, args);
    Py_DECREF(args);
    if (!ret) {
      std::cerr << "[Py] update_q_value 调用失败\n"; PyErr_Print(); return 0.0;
    }
    double v = PyFloat_Check(ret) ? PyFloat_AsDouble(ret) : 0.0;
    Py_DECREF(ret);
    return v;
  }

  std::string getBestNextHop(const std::string& currentNode,
                             const std::string& destNode) {
    std::string out;
    if (!pGetBest) return out;
    GilGuard gil;
    PyObject* args = PyTuple_New(2);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(currentNode.c_str()));
    PyTuple_SetItem(args, 1, PyUnicode_FromString(destNode.c_str()));
    PyObject* ret = PyObject_CallObject(pGetBest, args);
    Py_DECREF(args);
    if (!ret) { std::cerr << "[Py] get_best_next_hop 调用失败\n"; PyErr_Print(); return out; }
    if (const char* s = PyUnicode_AsUTF8(ret)) out.assign(s);
    Py_DECREF(ret);
    return out;
  }

  double updatelq(double sf, double ef, double bf,
                  const std::string& currentNode,
                  const std::string& nextHop) {
    if (!pUpdateLq) return 0.0;
    GilGuard gil;
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
    return v;
  }

  // 近似“maxQ”：best action 的偏好分/值（若 Python 提供 get_q_value）
  double getMaxQvalue(const std::string& currentNode,
                      const std::string& destNode) {
    if (!pGetBest) return 0.0;
    GilGuard gil;

    PyObject* argsBest = PyTuple_New(2);
    PyTuple_SetItem(argsBest, 0, PyUnicode_FromString(currentNode.c_str()));
    PyTuple_SetItem(argsBest, 1, PyUnicode_FromString(destNode.c_str()));
    PyObject* best = PyObject_CallObject(pGetBest, argsBest);
    Py_DECREF(argsBest);
    if (!best) { std::cerr << "[Py] get_best_next_hop 失败(getMaxQvalue)\n"; PyErr_Print(); return 0.0; }
    std::string bestNext;
    if (const char* s = PyUnicode_AsUTF8(best)) bestNext.assign(s);
    Py_DECREF(best);
    if (bestNext.empty() || !pGetQOpt) return 0.0;

    PyObject* args = PyTuple_New(2);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(currentNode.c_str()));
    PyTuple_SetItem(args, 1, PyUnicode_FromString(bestNext.c_str()));
    PyObject* val = PyObject_CallObject(pGetQOpt, args);
    Py_DECREF(args);
    if (!val) { std::cerr << "[Py] get_q_value 失败(getMaxQvalue)\n"; PyErr_Print(); return 0.0; }
    double v = PyFloat_Check(val) ? PyFloat_AsDouble(val) : 0.0;
    Py_DECREF(val);
    return v;
  }

private:
  PyObject* pModule   = nullptr;
  PyObject* pUpdateQ  = nullptr;
  PyObject* pSetParams= nullptr;
  PyObject* pGetBest  = nullptr;
  PyObject* pUpdateLq = nullptr;
  PyObject* pGetQOpt  = nullptr; // optional
};

}} // namespace ns3::qoar

#endif // NS3_QOAR_QLEARNING_H_
