ctypedef int npy_intp
ctypedef signed char        npy_int8
ctypedef signed short       npy_int16
ctypedef signed int         npy_int32
ctypedef signed long long   npy_int64
ctypedef unsigned char        npy_uint8
ctypedef unsigned short       npy_uint16
ctypedef unsigned int         npy_uint32
ctypedef unsigned long long   npy_uint64
ctypedef float        npy_float32
ctypedef double       npy_float64
ctypedef npy_intp       intp_t
ctypedef npy_int8       int8_t
ctypedef npy_int16      int16_t
ctypedef npy_int32      int32_t
ctypedef npy_int64      int64_t
ctypedef npy_uint8       uint8_t
ctypedef npy_uint16      uint16_t
ctypedef npy_uint32      uint32_t
ctypedef npy_uint64      uint64_t
ctypedef npy_float32    float32_t
ctypedef npy_float64    float64_t

ctypedef void (*PyUFuncGenericFunction) (char **, npy_intp *, npy_intp *, void *)


cdef extern from "":
    ctypedef class numpy.ndarray [clr "NumpyDotNet::ndarray"]:
        pass

    ctypedef class numpy.dtype [clr "NumpyDotNet::dtype"]:
        pass


cdef extern from "npy_descriptor.h":
    ctypedef struct NpyArray_Descr:
        pass

cdef extern from "npy_ufunc_object.h":
    ctypedef struct NpyUFuncObject:
        pass

    ctypedef void (*NpyUFuncGenericFunction) (char **, npy_intp *,
                                              npy_intp *, void *)

    NpyUFuncObject *NpyUFunc_FromFuncAndDataAndSignature(NpyUFuncGenericFunction *func, void **data,
                                     char *types, int ntypes,
                                     int nin, int nout, int identity,
                                     char *name, char *doc,
                                     int check_return, char *signature)

cdef extern from "npy_api.h":
    NpyArray_Descr *NpyArray_DescrFromType(int typenum)

cdef extern from "npy_ironpython.h":
    object Npy_INTERFACE_ufunc "Npy_INTERFACE_OBJECT" (NpyUFuncObject*)
    object Npy_INTERFACE_descr "Npy_INTERFACE_OBJECT" (NpyArray_Descr*)

cdef inline object PyUFunc_FromFuncAndData(PyUFuncGenericFunction* func, void** data,
        char* types, int ntypes, int nin, int nout,
        int identity, char* name, char* doc, int c):
   return Npy_INTERFACE_ufunc(NpyUFunc_FromFuncAndDataAndSignature(func, data, types, ntypes, nin, nout, identity, name, doc, c, NULL))

cdef inline object PyArray_ZEROS(int ndim, intp_t *shape, int typenum, int fortran):
    shape_list = []
    cdef int i
    for i in range(ndim):
        shape_list.append(shape[i])
    import numpy
    return numpy.zeros(shape_list, Npy_INTERFACE_descr(NpyArray_DescrFromType(typenum)), 'F' if fortran else 'C')

cdef inline void* PyArray_DATA(ndarray n):
    raise NotImplementedError

cdef inline intp_t* PyArray_DIMS(ndarray n):
    raise NotImplementedError
