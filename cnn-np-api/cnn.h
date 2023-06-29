//
//  cnn.h
//  cnn-np-api
//
//  Created by Brian Kelley on 6/29/23.
//

#ifndef cnn_h
#define cnn_h

//
//  main.cpp
//  cnn-np-api
//
//  Created by Brian Kelley on 5/28/23.
//  shamelessly stolen from https://victorzhou.com/blog/intro-to-cnns-part-2/
//   to try to get a conv neural network in C++ for embedded systems
//

#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xbuilder.hpp>
#include <fstream>


struct Conv {
    int num_filters = 8;
    xt::xarray<double> filters;
    xt::xarray<double> last_input;
    Conv () {
        filters = xt::random::randn({num_filters, 3, 3}, 0., 1.)/9.;
    }
    
    Conv(int num_filters) : num_filters(num_filters) {
        filters = xt::random::randn({num_filters, 3, 3}, 0., 1.)/9.;
    }
    
    Conv(xt::xarray<double> &filters) : num_filters((int)filters.shape(0)), filters(filters) {
    }
    
    xt::xarray<double> forward(const xt::xarray<double> &input) {
        int h = static_cast<int>(input.shape(0)) - 2;
        int w = static_cast<int>(input.shape(1)) - 2;
        last_input = input;

        xt::xarray<double> output = xt::zeros<double>({
            h, w, num_filters});
        for(int i=0; i<h; ++i) {
            for(int j=0; j<w; ++j) {
                auto im = xt::view(input, xt::range(i,i+3), xt::range(j,j+3));
                auto filtered = xt::sum(im * filters, {1,2});
                for(int f=0;f<num_filters;++f)
                    output(i,j,f) = filtered(f);
            }
        }
        return output;
    }
    
    void backprop(const xt::xarray<double> &d_l_d_out, double learn_rate) {
        xt::xarray<double> d_l_d_filters = xt::zeros<double>(filters.shape());
        int h = static_cast<int>(last_input.shape(0)) - 2;
        int w = static_cast<int>(last_input.shape(1)) - 2;
        for(int i=0; i<h; ++i) {
            for(int j=0; j<w; ++j) {
                for(int f=0; f<num_filters; ++f) {
                    auto im = xt::view(last_input, xt::range(i,i+3), xt::range(j,j+3));
                    d_l_d_filters(f) = d_l_d_filters(f) + (d_l_d_out(i,j,f) * im)[0]; // is this right???
                }
            }
        }
        filters -= learn_rate * d_l_d_filters;
    }
        
};

struct MaxPool {
    xt::xarray<double> last_input;
    xt::xarray<double> forward(const xt::xarray<double> &input) {
        last_input = input;
        int h = (int)input.shape(0);
        int w = (int)input.shape(1);
        int num_filters = (int)input.shape(2);
        
        xt::xarray<double>output = xt::zeros<double>({h/2, w/2, num_filters});
        for(int i=0; i<h/2; ++i) {
            for(int j=0; j<w/2; ++j) {
                auto im = xt::view(input, xt::range(i*2,(i*2)+2), xt::range(j*2,(j*2)+2), xt::all());
                auto m = xt::amax(im);
                for(int f=0;f<num_filters;++f)
                    output(i,j,f) = m[f];
            }
        }
        return output;
    }
    
    xt::xarray<double> backprop(const xt::xarray<double> &d_l_d_out, double /*learn_rate*/) {
        int h = (int)last_input.shape(0);
        int w = (int)last_input.shape(1);
        xt::xarray<double> d_l_d_input = xt::zeros<double>(last_input.shape());
        for(int i=0; i<h/2; ++i) {
            for(int j=0; j<w/2; ++j) {
                auto im = xt::view(last_input, xt::range(i,i+2), xt::range(j,j+2), xt::all());
                auto amax = xt::amax(im, {0,1})[0];
                for(int i2=0; i2<im.shape(0); ++i2) {
                    for(int j2=0; j2<im.shape(1); ++j2) {
                        for(int f2=0; f2<im.shape(2); ++f2) {
                            if(im(i2,j2,f2) == amax) {
                                d_l_d_input(i*2+i2, j*2+j2, f2) = d_l_d_out(i,j,f2);
                            }
                        }
                    }
                }
            }
        }
        return d_l_d_input;
    }
};

template<class T>
void shape(std::string s, const T&t) {
    int i=0;
    std::cout << s << " ";
    while(t.shape(i) < 1000000) {
        if (i) std::cout << ", ";
        std::cout << t.shape(i);
        ++i;
    }
    std::cout << std::endl;
}

struct SoftMax {
    xt::xarray<double> weights;
    xt::xarray<double> biases;
    xt::xarray<double> last_input;
    xt::xarray<double> last_totals;
    std::vector<double> last_input_shape;
    SoftMax(int input_len, int nodes) {
        weights = xt::random::randn({input_len, nodes}, 0., 1.)/input_len;
        //weights = xt::ones<double>({input_len, nodes})/input_len;
        biases = xt::zeros<double>({nodes});
    }
    
    xt::xarray<double> forward(const xt::xarray<double> &input) {
        last_input_shape.resize(input.shape().size());
        std::copy(input.shape().begin(), input.shape().end(), last_input_shape.begin());
        xt::xarray<double> flattened_input = xt::flatten(input);
        last_input = flattened_input;
       
        //auto dot = 0.0;
        auto dot = xt::linalg::dot(flattened_input, weights);
        auto totals = dot + biases;
        last_totals = totals;
        auto exponents = xt::exp(totals);
        //std::cout << "exponents: " << exponents << std::endl;
        return exponents/xt::sum(exponents, {0});
    }
    
    xt::xarray<double> backprop(const xt::xarray<double> &d_l_d_out, double learn_rate) {
        // find the non zero gradient
        int i=0;
        for(i=0; i< d_l_d_out.shape(0); ++i) {
            if(d_l_d_out(i) != 0) break;
        }
        assert(i!=d_l_d_out.shape(0));
        
        auto gradient = d_l_d_out(i);
        // e^totals
        auto t_exp = xt::exp(last_totals);
        //  sum of all e^totals
        double S = xt::sum(t_exp)[0];
        // Gradient of out[i] against all totals
        double S2 = S*S;
        double t_i = t_exp(i);
        xt::xarray<double> d_out_d_t = -t_i * t_exp/(S2);
        d_out_d_t(i) = t_i * (S-t_i)/S2;
        
        // Gradients of totals against weights/biases/input
        auto d_t_d_w = last_input;
        auto d_t_d_b = 1.;
        auto d_t_d_inputs = weights;
        // Gradients of loss against weights/biases/input
        auto d_l_d_t = gradient * d_out_d_t;
        auto a = xt::view(d_t_d_w, xt::all(), xt::newaxis());
        auto b = xt::transpose(xt::view(d_l_d_t, xt::all(), xt::newaxis()));
        auto d_l_d_w = xt::linalg::dot(a, b);
        auto d_l_d_b = d_l_d_t * d_t_d_b;
        auto d_l_d_inputs = xt::linalg::dot(d_t_d_inputs, d_l_d_t);
    
        // update weights/biases
        weights -= learn_rate * d_l_d_w;
        biases -= learn_rate * d_l_d_b;
        return d_l_d_inputs.reshape(last_input_shape);
    }
};



#endif /* cnn_h */
