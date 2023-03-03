/**
 * Copyright 2022 The VQLite Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <iostream>
#include <string>
#include <vector>

#include "scann/scann_ops/cc/vqindex_api.h"

using namespace std;

#ifdef __cplusplus
extern "C" {
#endif

static PyObject* _vqlite_init(PyObject* self, PyObject* args)
{
    char* index_dir_in = NULL;
    int index_dir_len_in = 0;
    int32_t dim = 128;
    int32_t storage_type = STORAGE_MEMORY;
    int64_t brute_threshold = 0;
    if (!PyArg_ParseTuple(args, "s#iiL", &index_dir_in, &index_dir_len_in, &dim, &storage_type, &brute_threshold)) {
        Py_RETURN_NONE;
    }

    string index_dir(index_dir_in, index_dir_len_in);

    index_config_t config_i;
    config_i.dim_ = dim;
    config_i.index_type_ = INDEX_TYPE_SCANN;
    config_i.storage_type_ = (storage_type_t)storage_type;
    config_i.partitioning_train_sample_rate_ = 0.2;
    config_i.hash_train_sample_rate_ = 0.1;
    config_i.brute_threshold_ = brute_threshold;

    void* vql_handler = NULL;
    Py_BEGIN_ALLOW_THREADS;
    vql_handler = vqindex_init(index_dir.c_str(), config_i);
    Py_END_ALLOW_THREADS

    intptr_t scann_t = (intptr_t)vql_handler;

    PyObject* res = Py_BuildValue("L", scann_t);
    return res;
}

static PyObject* _vqlite_search(PyObject* self, PyObject* args)
{
    PyObject* query_list;
    intptr_t vql_handler_i = 0;
    int topk = 30, reorder_topk = 128, nprobe = 128;
    if (!PyArg_ParseTuple(args, "LO!iii", &vql_handler_i, &PyList_Type, &query_list, &topk,
            &reorder_topk, &nprobe)) {
        Py_RETURN_NONE;
    }

    int32_t dim = 0, npoints = 0;

    std::vector<float> query_vec;
    Py_ssize_t list_size = PyList_Size(query_list);
    npoints = list_size;
    for (Py_ssize_t i = 0; i < list_size; i++) {
        PyObject* sublist = PyList_GetItem(query_list, i);
        if (!PyList_Check(sublist)) {
            PyErr_SetString(PyExc_TypeError, "List must contain lists");
            Py_RETURN_NONE;
        }
        Py_ssize_t sublist_size = PyList_Size(sublist);
        dim = sublist_size;

        for (Py_ssize_t j = 0; j < sublist_size; j++) {
            float value = PyFloat_AsDouble(PyList_GetItem(sublist, j));
            query_vec.push_back(value);
            if (PyErr_Occurred())
                Py_RETURN_NONE;
        }
    }

    // cout << "query list num: " << query_vec.size()/128 << endl;

    void* vql_handler = (void*)(vql_handler_i);
    std::vector<result_search_t> res;
    res.resize(npoints * topk);
    params_search_t params;
    params.nprobe_ = nprobe;
    params.reorder_topk_ = reorder_topk;
    params.topk_ = topk;
    int ret_state = 0;
    Py_BEGIN_ALLOW_THREADS;
    ret_state
        = vqindex_search(vql_handler, query_vec.data(), query_vec.size(), res.data(), params);
    Py_END_ALLOW_THREADS;

    if (ret_state != RET_CODE_OK) {
        Py_RETURN_NONE;
    }

    // cout << "result: " << ret_state << endl;

    PyObject* ret_list = PyList_New(0);
    for (int i = 0; i < npoints; i++) {
        PyObject* ret_sublist = PyList_New(0);
        for (int j = 0; j < topk; j++) {
            PyObject* item_obj = PyDict_New();
            PyObject* idx = Py_BuildValue("I", res[i * topk + j].idx_);
            PyDict_SetItemString(item_obj, "idx", idx);
            Py_XDECREF(idx);
            PyObject* vid = Py_BuildValue("L", res[i * topk + j].vid_);
            PyDict_SetItemString(item_obj, "vid", vid);
            Py_XDECREF(vid);
            PyObject* score = Py_BuildValue("f", res[i * topk + j].score_);
            PyDict_SetItemString(item_obj, "score", score);
            Py_XDECREF(score);

            PyList_Append(ret_sublist, item_obj);
            Py_XDECREF(item_obj);
            // cout << res[i][j].first << ": " << res[i][j].second << endl;
        }
        PyList_Append(ret_list, ret_sublist);
        Py_XDECREF(ret_sublist);
    }

    PyObject* ret_res = Py_BuildValue("(O)", ret_list);
    Py_XDECREF(ret_list);

    return ret_res;
}

static PyObject* _vqlite_add(PyObject* self, PyObject* args)
{
    PyObject *datasets_list, *vids_list;
    intptr_t vql_handler_i = 0;
    if (!PyArg_ParseTuple(
            args, "LO!O!", &vql_handler_i, &PyList_Type, &datasets_list, &PyList_Type, &vids_list)) {
        std::cout << "Parse Params Fail." << std::endl;
        Py_RETURN_FALSE;
    }

    std::vector<float> datasets_vec;
    Py_ssize_t list_size = PyList_Size(datasets_list);
    for (Py_ssize_t i = 0; i < list_size; i++) {
        PyObject* sublist = PyList_GetItem(datasets_list, i);
        if (!PyList_Check(sublist)) {
            PyErr_SetString(PyExc_TypeError, "List must contain lists");
            Py_RETURN_FALSE;
        }
        Py_ssize_t sublist_size = PyList_Size(sublist);

        for (Py_ssize_t j = 0; j < sublist_size; j++) {
            float value = PyFloat_AsDouble(PyList_GetItem(sublist, j));
            datasets_vec.push_back(value);
            if (PyErr_Occurred())
                Py_RETURN_FALSE;
        }
    }

    std::vector<int64_t> vids_vec;
    Py_ssize_t vids_list_size = PyList_Size(vids_list);
    for (Py_ssize_t i = 0; i < vids_list_size; i++) {
        int64_t vid = PyLong_AsLongLong(PyList_GetItem(vids_list, i));
        if (PyErr_Occurred())
            Py_RETURN_FALSE;
        vids_vec.push_back(vid);
    }

    if (list_size != vids_vec.size()) {
        std::cout << "list_size=" << list_size << "; vids_vec.size()=" << vids_vec.size()
                  << std::endl;
    }

    std::cout << "list_size=" << list_size << "; vids_vec.size()=" << vids_vec.size()
                  << std::endl;
    void* vql_handler = (void*)(vql_handler_i);

    ret_code_t ret_a = RET_CODE_OK;
    Py_BEGIN_ALLOW_THREADS;
    ret_a = vqindex_add(vql_handler, datasets_vec.data(), datasets_vec.size(), vids_vec.data());
    Py_END_ALLOW_THREADS;

    if (ret_a != RET_CODE_OK) {
        Py_RETURN_FALSE;
    }

    Py_RETURN_TRUE;
}

static PyObject* _vqlite_train(PyObject* self, PyObject* args)
{
    intptr_t vql_handler_i = 0;
    int train_type_i = 0, nthreads = 0, nlist=0;
    if (!PyArg_ParseTuple(args, "Liii", &vql_handler_i, &train_type_i, &nlist, &nthreads)) {
        Py_RETURN_FALSE;
    }

    if (nthreads < 0) {
        nthreads = 0;
    }
    train_type_t train_type = TRAIN_TYPE_DEFAULT;
    if (train_type_i >= (int)TRAIN_TYPE_DEFAULT && train_type_i <= (int)TRAIN_TYPE_ADD) {
        train_type = (train_type_t) train_type_i;
    }

    void* vql_handler = (void*)(vql_handler_i);
    ret_code_t ret_t = RET_CODE_OK;
    Py_BEGIN_ALLOW_THREADS;
    ret_t = vqindex_train(vql_handler, train_type, nlist, nthreads);
    Py_END_ALLOW_THREADS;

    if (ret_t != RET_CODE_OK) {
        Py_RETURN_FALSE;
    }
    Py_RETURN_TRUE;
}

static PyObject* _vqlite_dump(PyObject* self, PyObject* args)
{
    intptr_t vql_handler_i = 0;
    if (!PyArg_ParseTuple(args, "L", &vql_handler_i)) {
        Py_RETURN_FALSE;
    }

    void* vql_handler = (void*)(vql_handler_i);
    ret_code_t ret_d = RET_CODE_OK;
    Py_BEGIN_ALLOW_THREADS;
    ret_d = vqindex_dump(vql_handler);
    Py_END_ALLOW_THREADS;

    if (ret_d != RET_CODE_OK) {
        Py_RETURN_FALSE;
    }
    Py_RETURN_TRUE;
}

static PyObject* _vqlite_release(PyObject* self, PyObject* args)
{
    intptr_t vql_handler_i = 0;
    if (!PyArg_ParseTuple(args, "L", &vql_handler_i)) {
        Py_RETURN_NONE;
    }

    void* vql_handler = (void*)(vql_handler_i);
    vqindex_release(vql_handler);

    Py_RETURN_NONE;
}

static PyObject* _vqlite_stats(PyObject* self, PyObject* args)
{
    intptr_t vql_handler_i = 0;
    if (!PyArg_ParseTuple(args, "L", &vql_handler_i)) {
        Py_RETURN_NONE;
    }

    void* vql_handler = (void*)(vql_handler_i);
    index_stats_t stats_ret = vqindex_stats(vql_handler);

    PyObject* res_dict = PyDict_New();
    PyObject *obj_t = NULL, *res = NULL;

    obj_t = Py_BuildValue("L", stats_ret.datasets_size_);
    PyDict_SetItemString(res_dict, "datasets_size", obj_t);
    Py_XDECREF(obj_t);
    obj_t = Py_BuildValue("L", stats_ret.vid_size_);
    PyDict_SetItemString(res_dict, "vid_size", obj_t);
    Py_XDECREF(obj_t);
    obj_t = Py_BuildValue("L", stats_ret.index_size_);
    PyDict_SetItemString(res_dict, "index_size", obj_t);
    Py_XDECREF(obj_t);
    obj_t = Py_BuildValue("i", stats_ret.index_nlist_);
    PyDict_SetItemString(res_dict, "index_nlist", obj_t);
    Py_XDECREF(obj_t);
    obj_t = Py_BuildValue("i", stats_ret.dim_);
    PyDict_SetItemString(res_dict, "dim", obj_t);
    Py_XDECREF(obj_t);
    obj_t = Py_BuildValue("L", stats_ret.brute_threshold_);
    PyDict_SetItemString(res_dict, "brute_threshold", obj_t);
    Py_XDECREF(obj_t);

    obj_t = Py_BuildValue("i", stats_ret.current_status_);
    PyDict_SetItemString(res_dict, "current_status", obj_t);
    Py_XDECREF(obj_t);

    if (stats_ret.is_brute_) {
        PyDict_SetItemString(res_dict, "is_brute", Py_True);
    } else {
        PyDict_SetItemString(res_dict, "is_brute", Py_False);
    }

    res = Py_BuildValue("(O)", res_dict);
    Py_XDECREF(res_dict);

    return res;
}

static PyMethodDef all_methods[] = {
    { "init", _vqlite_init, METH_VARARGS, "init(index_dir, config)" },
    { "search", _vqlite_search, METH_VARARGS,
        "search(handler,querylist,topk,reorder_topk,nprobe)" },
    { "add", _vqlite_add, METH_VARARGS, "add(handler,datasets,vids)" },
    { "release", _vqlite_release, METH_VARARGS, "release(handler)" },
    { "train", _vqlite_train, METH_VARARGS, "train(handler, train_type, nthreads)" },
    { "dump", _vqlite_dump, METH_VARARGS, "dump(handler)" },
    { "stats", _vqlite_stats, METH_VARARGS, "stats(handler)" },
    { NULL, NULL, 0, NULL },
};

#if PY_MAJOR_VERSION >= 3
#define MOD_ERROR_VAL NULL
#define MOD_SUCCESS_VAL(val) val
#define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
#define MOD_DEF(ob, name, doc, methods)                                                            \
    static struct PyModuleDef moduledef = {                                                        \
        PyModuleDef_HEAD_INIT,                                                                     \
        name,                                                                                      \
        doc,                                                                                       \
        -1,                                                                                        \
        methods,                                                                                   \
    };                                                                                             \
    ob = PyModule_Create(&moduledef);
#else
#define MOD_ERROR_VAL
#define MOD_SUCCESS_VAL(val)
#define MOD_INIT(name) void init##name(void)
#define MOD_DEF(ob, name, doc, methods) ob = Py_InitModule3(name, methods, doc);
#endif

MOD_INIT(vqindex_py)
{
    PyObject* m;
    MOD_DEF(m, "vqindex_py", NULL, all_methods)
    if (m == NULL)
        return MOD_ERROR_VAL;

    return MOD_SUCCESS_VAL(m);
}

#ifdef __cplusplus
}
#endif
