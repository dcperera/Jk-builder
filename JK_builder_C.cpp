#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#include <iostream>
#include <omp.h>

namespace py = pybind11;

std::vector<py::array> form_JK_conventional(py::array_t<double> I, py::array_t<double> D)
{
    py::buffer_info I_info = I.request();
    py::buffer_info D_info = D.request();
    
    if(I_info.ndim != 4) throw std::runtime_error("I is not a rank-4 tensor");
    if(D_info.ndim != 2) throw std::runtime_error("D is not a matrix");

    size_t dim = D_info.shape[0];
    size_t dim2 = dim*dim;
    size_t dim3 = dim*dim2;
    
    const double * I_data = static_cast<double *>(I_info.ptr);
    const double * D_data = static_cast<double *>(D_info.ptr);
    
    std::vector<double> J_data(dim * dim);
    std::vector<double> K_data(dim * dim);

    #pragma omp parallel for num_threads(4) schedule(dynamic)
    for(size_t p = 0; p < dim; p++)
    {
        for(size_t q = 0; q <= p; q++)
        {
            double Jvalue = 0.;
            double Kvalue = 0.;
            for(size_t r = 0; r < dim; r++)
            {
                for(size_t s = 0; s < r; s++)
                {
                    Jvalue += 2.*I_data[p * dim3 + q * dim2 + r * dim + s] * D_data[r * dim + s];
                }
                Jvalue += I_data[p * dim3 + q * dim2 + r * dim + r] * D_data[r * dim + r];
                for(size_t i = 0; i < dim; i++)
                {
                    Kvalue += I_data[p * dim3 + r * dim2 + q * dim + i] * D_data[r * dim + i];
                }
            }
            J_data[p * dim + q] = Jvalue;
            J_data[q * dim + p] = Jvalue;
            K_data[p * dim + q]	= Kvalue;
            K_data[q * dim + p]	= Kvalue;            
        }
    }
    
    py::buffer_info Jbuf =
    {
        J_data.data(),
        sizeof(double),
        py::format_descriptor<double>::format(),
        2,
        {dim, dim},
        {dim * sizeof(double), sizeof(double)}
    };
    
    py::buffer_info Kbuf =
    {
        K_data.data(),
        sizeof(double),
        py::format_descriptor<double>::format(),
        2,
        {dim, dim},
        {dim * sizeof(double), sizeof(double)}
    };
    
    py::array J(Jbuf);
    py::array K(Kbuf);
    return {J, K};
}

std::vector<py::array> form_JK_df(py::array_t<double> Ig, py::array_t<double> D,  py::array_t<double> C)
{
    py::buffer_info Ig_info = Ils.request();
    py::buffer_info D_info = D.request();
    py::buffer_info C_info = C.request();
    
    if(I_info.ndim != 4) throw std::runtime_error("I is not a rank-4 tensor");
    if(D_info.ndim != 2) throw std::runtime_error("D is not a matrix");

    size_t dimD = D_info.shape[0];
    size_t dimIg = Ig_info.shape[0];
    
    const double * Ig_data = static_cast<double *>(Ig_info.ptr);
    const double * D_data = static_cast<double *>(D_info.ptr);
    const double * C_data = static_cast<double *>(C_info.ptr);

    std::vector<double> kaiP_data(dimIg);
    std::vector<double> zeta_data(dimIg * dimD * dimD);
    
    std::vector<double> J_data(dim * dim);
    std::vector<double> K_data(dim * dim);

    #pragma omp parallel for num_threads(4) schedule(dynamic)
    for(size_t p = 0; p < dimIg; p++)
    {
        double value = 0.;
        for(size_t l = 0; l < dimD; l++)
        {
            for(size_t s = 0; s < dimD; s++)
            {
                value += Ig_data[p * dimIg * dimD + l * dimD + s] * D_data[l * dimD + s];
            }
        }
        kaiP_data[p] += value;
    }
    #pragma omp parallel for num_threads(4) schedule(dynamic)
    for(size_t l = 0; l < dimD; l++)
    {
        for(size_t s = 0; s <= l; s++)
        {
            double value = 0.;
            for(size_t p = 0; p < dimIg; p++)
            {
                value += Ig_data[p * dimIg * dimD + l * dimD + s] * kaiP_data[p];
            }
            J_data[l * dimD + s] = value;
            J_data[s * dimD + l] = value;
        }
    }

    #pragma omp parallel for num_threads(4) schedule(dynamic)
    for(size_t p = 0; p < dimIg; p++)
    {
        for(size_t l = 0; l < dimD; l++)
        {
            for(size_t m = 0; m < dimD; m++)
            {
                double value = 0.;
                for(size_t s = 0; s < dimD; s++)
                {
                    value += Ig_data[p * dimIg * dimD + l * dimD + s] * C_data[m * dimD + s];
                }
                zeta[p * dimIg * dimD + l * dimD + m] = value;
            }
        }
    }
    #pragma omp parallel for num_threads(4) schedule(dynamic)
    for(size_t l = 0; l < dimD; l++)
    {
        for(size_t s = 0; s <= l; s++)
        {
            double value = 0.;
            for(size_t p = 0; p < dimIg; p++)
            {
                for(size_t r = 0; r < dimD; r++)
                {
                    value += zeta[p * dimIg * dimD + l * dimD + r] * zeta[p * dimIg * dimD + s * dimD + r];
                }
            }
            K_data[l * dimD + s] = value;
            K_data[s * dimD + l] = value;
        }
    }

    py::buffer_info Jbuf =
    {
        J_data.data(),
        sizeof(double),
        py::format_descriptor<double>::format(),
        2,
        {dim, dim},
        {dim * sizeof(double), sizeof(double)}
    };
    
    py::buffer_info Kbuf =
    {
        K_data.data(),
        sizeof(double),
        py::format_descriptor<double>::format(),
        2,
        {dim, dim},
        {dim * sizeof(double), sizeof(double)}
    };
    
    py::array J(Jbuf);
    py::array K(Kbuf);
    return {J, K};
}

PYBIND11_PLUGIN(JK_builder_C)
{
    py::module m("JK_builder_C", "Computes J and K");
    m.def("form_JK_conventional", &form_JK_conventional, "Computes J and K using conventional method");
    m.def("form_JK_df", &form_JK_df, "Computes J and K using density fitting");
    return m.ptr();
}
