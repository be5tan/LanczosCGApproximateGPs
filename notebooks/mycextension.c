#define PY_SSIZE_T_CLEAN // Reccomended before including Python.h
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <math.h>
#include <cblas.h>

static PyObject *cgActions(PyObject *self, PyObject *args) {
    // Parse the arguments
    PyArrayObject *pyMatrix;
    PyArrayObject *pyResponse;
    int maxIter;
    PyArg_ParseTuple(args, "OOi", &pyMatrix, &pyResponse, &maxIter);
    if (PyErr_Occurred()) {
        return NULL;
    }
    if(!PyArray_Check(pyMatrix) 
       || PyArray_TYPE(pyMatrix) != NPY_DOUBLE
       || !PyArray_IS_C_CONTIGUOUS(pyMatrix)
    ){
        PyErr_SetString(PyExc_TypeError,
            "matrixArg must be C-contiguous numpy array of type double.");
        return NULL;
    }
    if(!PyArray_Check(pyResponse) 
       || PyArray_TYPE(pyResponse) != NPY_DOUBLE
       || !PyArray_IS_C_CONTIGUOUS(pyResponse)
    ){
        PyErr_SetString(PyExc_TypeError,
            "responseArg must be C-contiguous numpy array of type double.");
        return NULL;
    }

    int64_t size = PyArray_SIZE(pyResponse);
    double *matrix = PyArray_DATA(pyMatrix);
    double *response = PyArray_DATA(pyResponse);

    // Initialize the collection of actions
    // Create PyArray_Dims object
    npy_intp dims[2] = {size, 1};  // Dimensions: size rows, size columns
    PyArray_Dims *dimensions = (PyArray_Dims *)malloc(sizeof(PyArray_Dims));
    dimensions->len = 2;
    dimensions->ptr = dims;
    // PyArray_Dims *newshape = PyArray_IntpFromArray(dims, 2);
    PyObject *cgActionMatrix = PyArray_Newshape((PyArrayObject *)pyResponse, dimensions, NPY_CORDER);

    // cblas computations

    double *gradient = (double *)malloc(size * sizeof(double));
    cblas_dcopy(size, response, 1, gradient, 1);
    cblas_dscal(size, -1.0, gradient, 1);

    double sqrGradientNorm = cblas_ddot(size, gradient, 1, gradient, 1);

    double *matrixGradientProduct = (double *)malloc(size * sizeof(double));
    cblas_dgemv(CblasRowMajor, CblasNoTrans, size, size, 1.0, matrix, size,
                gradient, 1, 0.0, matrixGradientProduct, 1);



    // Print the result vector y
    printf("Result vector y:\n");
    for (int i = 0; i < size; ++i) {
        printf("%f\n", matrixGradientProduct[i]);
    }


    // Old stuff


    // Initialize solution at zero
    int nd = 1;  // Number of dimensions
    npy_intp solutionDim[1] = {size};  // Dimensions: size rows, size columns
    PyArray_Descr *dtype = PyArray_DescrFromType(NPY_FLOAT64);  // Double type
    int fortran = 0;  // Row-major order
    PyArrayObject *solution = (PyArrayObject *)PyArray_Zeros(nd, solutionDim, dtype, fortran); 

    // PyObject *result = PyArray_Concatenate((PyObject *)PyTuple_Pack(2, matrix, reshaped_array), 1);


    // int nd = 2;  // Number of dimensions
    // npy_intp dims[2] = {size, size};  // Dimensions: size rows, size columns
    // PyArray_Descr *dtype = PyArray_DescrFromType(NPY_FLOAT64);  // Double type
    // int fortran = 0;  // Row-major order

    // PyArrayObject *cgActionMatrix;
    // cgActionMatrix = (PyArrayObject *)PyArray_Zeros(nd, dims, dtype, fortran);

    // return cgActionMatrix;
    return PyArray_Return(solution);
}

static PyMethodDef mycextension_methods[] = {
    {"cgActions", cgActions, METH_VARARGS, "Collects conjugate gradient actions"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef mycextension = {
    PyModuleDef_HEAD_INIT,
    "name",   /* name of module */
    NULL,     /* module documentation, may be NULL */
    -1,
    mycextension_methods
};

PyMODINIT_FUNC
PyInit_mycextension(void) {
    printf("Initializing mycextension module...\n");
    PyObject *module = PyModule_Create(&mycextension);
    import_array();
    return module;
}
