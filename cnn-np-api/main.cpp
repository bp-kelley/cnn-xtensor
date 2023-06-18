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
                output(i,j) = xt::sum(im * filters, {1,2})[0];
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
            for(int j=0/2; j<w; ++j) {
                auto im = xt::view(input, xt::range(i,i+2), xt::range(j,j+2));
                output(i,j) = xt::amax(im, {1,2})[0];
            }
        }
        return output;
    }
    
    xt::xarray<double> backprop(const xt::xarray<double> &d_l_d_out, double /*learn_rate*/) {
        int h = (int)last_input.shape(0);
        int w = (int)last_input.shape(1);
        xt::xarray<double> d_l_d_input = xt::zeros<double>(last_input.shape());
        for(int i=0; i<h; ++i) {
            for(int j=0; j<w; ++j) {
                auto im = xt::view(last_input, xt::range(i,i+2), xt::range(j,j+2));
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
        biases = xt::zeros<double>({nodes});
    }
    
    xt::xarray<double> forward(const xt::xarray<double> &input) {
        last_input_shape.resize(input.shape().size());
        std::copy(input.shape().begin(), input.shape().end(), last_input_shape.begin());
        xt::xarray<double> flattened_input = xt::flatten(input);
        shape("input", input);
        last_input = flattened_input;
        shape("last_input", last_input);
        auto dot = 0.0;
        //xt::linalg::dot(input, weights); //- don't need blas for this!!!
        for(int i=0; i<flattened_input.shape(0); ++i ) {
            dot += flattened_input(i) * weights(i);
        }
        auto totals = dot + biases;
        last_totals = totals;
        auto exponents = xt::exp(totals);
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


void test() {
    xt::xarray<double> image {
        {0,0,0},
        {0,3,0},
        {0,0,0}
    };
    xt::xarray<double> image2 {
        {1,1,1},
        {1,3,1},
        {1,1,1}
    };
    {
        xt::xarray<double> filters = {
            {
                {1,1,1},
                {1,1,1},
                {1,1,1}
            },
        };
        
        auto conv = Conv(filters);
        auto output = conv.forward(image);
        assert (output(0) == 3);
        output = conv.forward(image2);
        assert(output(0) == 11);
    }
    {
        xt::xarray<double> filters = {
            {
                {0,0,0},
                {0,1,0},
                {0,0,0}
            },
        };
        
        auto conv = Conv(filters);
        auto output = conv.forward(image);
        assert (output(0) == 3);
        output = conv.forward(image2);
        assert(output(0) == 3);
    }
}

void train() {
    Conv conv(8);
    MaxPool pool;
    SoftMax softmax(13 * 13 * 8, 10);
    
    xt::xarray<double> image = {{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,30,97,144,236,206,230,144,148,254,254,240,0,0,0 },
        { 0,0,0,0,0,0,0,0,0,0,0,0,0,50,230,254,253,253,253,253,253,253,253,253,191,0,0,0 },
        { 0,0,0,0,0,0,0,0,0,0,0,0,0,188,253,254,253,171,197,154,154,154,154,154,87,0,0,0 },
        { 0,0,0,0,0,0,0,0,0,0,0,11,157,251,253,242,77,2,5,0,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,0,0,0,0,0,0,122,253,253,253,111,0,0,0,0,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,0,0,0,0,0,119,225,253,253,166,24,0,0,0,0,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,0,0,0,1,108,240,253,253,253,147,123,93,12,1,0,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,0,0,18,159,253,253,253,253,253,253,254,253,253,149,16,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,0,0,57,249,253,253,253,253,230,176,221,253,253,253,169,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,0,0,100,253,253,253,200,124,23,0,19,62,235,253,198,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,0,0,0,107,111,24,0,0,0,0,0,0,197,255,254,70,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,30,196,253,253,117,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,147,246,253,207,29,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,3,100,33,0,0,0,0,0,0,0,66,217,253,253,150,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,130,253,197,5,0,0,0,0,0,25,130,253,253,200,107,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,206,253,232,8,0,0,0,0,28,160,254,253,253,59,0,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,101,253,253,233,160,35,16,122,210,253,254,250,199,6,0,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,13,158,253,253,253,253,253,253,253,253,229,154,0,0,0,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,0,46,157,250,253,253,253,253,253,220,29,0,0,0,0,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,0,0,0,41,191,230,249,143,61,19,0,0,0,0,0,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 }};
    int label = 1;
    auto out = softmax.forward(pool.forward(conv.forward(image)));
    auto loss = -log(out[label]);
    auto acc = xt::argmax(out)[0] == label ? 1 : 0;
    
    //1 if(np.argmax(out) == label) else 0
    double lr = 0.005;
    xt::xarray<double> gradient = xt::zeros<double>({10});
    gradient(label) = -1./out(label);
       
       
       //Backprop
    std::cout << "loss " << loss << " acc " << acc << " " << gradient << std::endl;
    xt::xarray<double> g1 = softmax.backprop(gradient, lr);
    xt::xarray<double> g2 = pool.backprop(g1, lr);
    conv.backprop(g2, lr);
    
}
int main(int argc, const char * argv[]) {
    // insert code here...
    test();
    train();
    return 0;

}
