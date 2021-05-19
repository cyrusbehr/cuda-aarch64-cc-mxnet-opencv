/*!
*  Copyright (c) 2016 by Contributors
* \file op.h
* \brief definition of all the operators
* \author Chuntao Hong, Xin Li
*/

#ifndef MXNET_CPP_OP_H_
#define MXNET_CPP_OP_H_

#include <string>
#include <vector>
#include "mxnet-cpp/base.h"
#include "mxnet-cpp/shape.h"
#include "mxnet-cpp/op_util.h"
#include "mxnet-cpp/operator.h"
#include "dmlc/optional.h"
#include "nnvm/tuple.h"

namespace mxnet {
namespace cpp {

/*!
 * \brief Batch normalization.
 *
 *        This operator is DEPRECATED. Perform BatchNorm on the input.
 *
 *        Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as
 *        well as offset ``beta``.
 *
 *        Assume the input has more than one dimension and we normalize along axis 1.
 *        We first compute the mean and variance along this axis:
 *
 *        .. math::
 *
 *        data\_mean[i] = mean(data[:,i,:,...]) \\
 *        data\_var[i] = var(data[:,i,:,...])
 *
 *        Then compute the normalized output, which has the same shape as input, as
 *
 *        .. math::
 *
 *        out[:,i,:,...] = \frac{data[:,i,:,...] -
 *
 *        Both *mean* and *var* returns a scalar by treating the input as a vector.
 *
 *        Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
 *        have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both
 *        ``data_var`` as well, which are needed for the backward pass.
 *
 *        Besides the inputs and the outputs, this operator accepts two auxiliary
 *        states, ``moving_mean`` and ``moving_var``, which are *k*-length
 *        vectors. They are global statistics for the whole dataset, which are updated
 *        by::
 *
 *        moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
 *        moving_var = moving_var * momentum + data_var * (1 - momentum)
 *
 *        If ``use_global_stats`` is set to be true, then ``moving_mean`` and
 *        ``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute
 *        the output. It is often used during inference.
 *
 *        Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is
 *        then set ``gamma`` to 1 and its gradient to 0.
 *
 *        There's no sparse support for this operator, and it will exhibit problematic
 *        sparse tensors.
 *
 *
 *
 *        Defined in ../src/operator/batch_norm_v1.cc:L95
 * \param symbol_name name of the resulting symbol
 * \param data Input data to batch normalization
 * \param gamma gamma array
 * \param beta beta array
 * \param eps Epsilon to prevent div 0
 * \param momentum Momentum for moving average
 * \param fix_gamma Fix gamma while training
 * \param use_global_stats Whether use global moving statistics instead of local
 * \param output_mean_var Output All,normal mean and var
 * \return new symbol
 */
inline Symbol BatchNorm_v1(const std::string& symbol_name,
                           Symbol data,
                           Symbol gamma,
                           Symbol beta,
                           mx_float eps = 0.00100000005,
                           mx_float momentum = 0.899999976,
                           bool fix_gamma = true,
                           bool use_global_stats = false,
                           bool output_mean_var = false) {
  return Operator("BatchNorm_v1")
           .SetParam("eps", eps)
           .SetParam("momentum", momentum)
           .SetParam("fix_gamma", fix_gamma)
           .SetParam("use_global_stats", use_global_stats)
           .SetParam("output_mean_var", output_mean_var)
           .SetInput("data", data)
           .SetInput("gamma", gamma)
           .SetInput("beta", beta)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Check if all the float numbers in the array are finite (used for AMP)
 *
 *
 *        Defined in ../src/operator/contrib/all_finite.cc:L101
 * \param symbol_name name of the resulting symbol
 * \param data Array
 * \param init_output Initialize output to 1.
 * \return new symbol
 */
inline Symbol all_finite(const std::string& symbol_name,
                         Symbol data,
                         bool init_output = true) {
  return Operator("all_finite")
           .SetParam("init_output", init_output)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Check if all the float numbers in all the arrays are finite (used for AMP)
 *
 *
 *        Defined in ../src/operator/contrib/all_finite.cc:L133
 * \param symbol_name name of the resulting symbol
 * \param data Arrays
 * \param num_arrays Number of arrays.
 * \param init_output Initialize output to 1.
 * \return new symbol
 */
inline Symbol multi_all_finite(const std::string& symbol_name,
                               const std::vector<Symbol>& data,
                               int num_arrays = 1,
                               bool init_output = true) {
  return Operator("multi_all_finite")
           .SetParam("num_arrays", num_arrays)
           .SetParam("init_output", init_output)
(data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Computes the Khatri-Rao product of the input matrices.
 *
 *        Given a collection of :math:`n` input matrices,
 *
 *        .. math::
 *        A_1 \in \mathbb{R}^{M_1 \times M}, \ldots, A_n \in \mathbb{R}^{M_n \times N},
 *
 *        the (column-wise) Khatri-Rao product is defined as the matrix,
 *
 *        .. math::
 *        X = A_1 \otimes \cdots \otimes A_n \in \mathbb{R}^{(M_1 \cdots M_n) \times N},
 *
 *        where the :math:`k` th column is equal to the column-wise outer product
 *        :math:`{A_1}_k \otimes \cdots \otimes {A_n}_k` where :math:`{A_i}_k` is the kth
 *        column of the ith matrix.
 *
 *        Example::
 *
 *        >>> A = mx.nd.array([[1, -1],
 *        >>>                  [2, -3]])
 *        >>> B = mx.nd.array([[1, 4],
 *        >>>                  [2, 5],
 *        >>>                  [3, 6]])
 *        >>> C = mx.nd.khatri_rao(A, B)
 *        >>> print(C.asnumpy())
 *        [[  1.  -4.]
 *        [  2.  -5.]
 *        [  3.  -6.]
 *        [  2. -12.]
 *        [  4. -15.]
 *        [  6. -18.]]
 *
 *
 *
 *        Defined in ../src/operator/contrib/krprod.cc:L108
 * \param symbol_name name of the resulting symbol
 * \param args Positional input matrices
 * \return new symbol
 */
inline Symbol khatri_rao(const std::string& symbol_name,
                         const std::vector<Symbol>& args) {
  return Operator("khatri_rao")
(args)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Compute the LARS coefficients of multiple weights and grads from their sums of
 *
 *
 *        Defined in ../src/operator/contrib/multi_lars.cc:L37
 * \param symbol_name name of the resulting symbol
 * \param lrs Learning rates to scale by LARS coefficient
 * \param weights_sum_sq sum of square of weights arrays
 * \param grads_sum_sq sum of square of gradients arrays
 * \param wds weight decays
 * \param eta LARS eta
 * \param eps LARS eps
 * \param rescale_grad Gradient rescaling factor
 * \return new symbol
 */
inline Symbol multi_lars(const std::string& symbol_name,
                         Symbol lrs,
                         Symbol weights_sum_sq,
                         Symbol grads_sum_sq,
                         Symbol wds,
                         mx_float eta,
                         mx_float eps,
                         mx_float rescale_grad = 1) {
  return Operator("multi_lars")
           .SetParam("eta", eta)
           .SetParam("eps", eps)
           .SetParam("rescale_grad", rescale_grad)
           .SetInput("lrs", lrs)
           .SetInput("weights_sum_sq", weights_sum_sq)
           .SetInput("grads_sum_sq", grads_sum_sq)
           .SetInput("wds", wds)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Compute the sums of squares of multiple arrays
 *
 *
 *        Defined in ../src/operator/contrib/multi_sum_sq.cc:L36
 * \param symbol_name name of the resulting symbol
 * \param data Arrays
 * \param num_arrays number of input arrays.
 * \return new symbol
 */
inline Symbol multi_sum_sq(const std::string& symbol_name,
                           const std::vector<Symbol>& data,
                           int num_arrays) {
  return Operator("multi_sum_sq")
           .SetParam("num_arrays", num_arrays)
(data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Update function for Stochastic Gradient Descent (SDG) optimizer.
 *
 *        It updates the weights using::
 *
 *        weight = weight - learning_rate * (gradient + wd * weight)
 *
 *
 *
 *        Defined in ../src/operator/contrib/preloaded_multi_sgd.cc:L42
 * \param symbol_name name of the resulting symbol
 * \param data Weights, gradients, learning rates and weight decays
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param num_weights Number of updated weights.
 * \return new symbol
 */
inline Symbol preloaded_multi_sgd_update(const std::string& symbol_name,
                                         const std::vector<Symbol>& data,
                                         mx_float rescale_grad = 1,
                                         mx_float clip_gradient = -1,
                                         int num_weights = 1) {
  return Operator("preloaded_multi_sgd_update")
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("num_weights", num_weights)
(data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Momentum update function for Stochastic Gradient Descent (SGD) optimizer.
 *
 *        Momentum update has better convergence rates on neural networks. Mathematically
 *        like below:
 *
 *        .. math::
 *
 *        v_1 = \alpha * \nabla J(W_0)\\
 *        v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\
 *        W_t = W_{t-1} + v_t
 *
 *        It updates the weights using::
 *
 *        v = momentum * v - learning_rate * gradient
 *        weight += v
 *
 *        Where the parameter ``momentum`` is the decay rate of momentum estimates at
 *
 *
 *
 *        Defined in ../src/operator/contrib/preloaded_multi_sgd.cc:L91
 * \param symbol_name name of the resulting symbol
 * \param data Weights, gradients, momentum, learning rates and weight decays
 * \param momentum The decay rate of momentum estimates at each epoch.
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param num_weights Number of updated weights.
 * \return new symbol
 */
inline Symbol preloaded_multi_sgd_mom_update(const std::string& symbol_name,
                                             const std::vector<Symbol>& data,
                                             mx_float momentum = 0,
                                             mx_float rescale_grad = 1,
                                             mx_float clip_gradient = -1,
                                             int num_weights = 1) {
  return Operator("preloaded_multi_sgd_mom_update")
           .SetParam("momentum", momentum)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("num_weights", num_weights)
(data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Update function for multi-precision Stochastic Gradient Descent (SDG) optimizer.
 *
 *        It updates the weights using::
 *
 *        weight = weight - learning_rate * (gradient + wd * weight)
 *
 *
 *
 *        Defined in ../src/operator/contrib/preloaded_multi_sgd.cc:L140
 * \param symbol_name name of the resulting symbol
 * \param data Weights, gradients, learning rates and weight decays
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param num_weights Number of updated weights.
 * \return new symbol
 */
inline Symbol preloaded_multi_mp_sgd_update(const std::string& symbol_name,
                                            const std::vector<Symbol>& data,
                                            mx_float rescale_grad = 1,
                                            mx_float clip_gradient = -1,
                                            int num_weights = 1) {
  return Operator("preloaded_multi_mp_sgd_update")
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("num_weights", num_weights)
(data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Momentum update function for multi-precision Stochastic Gradient Descent (SGD)
 *
 *        Momentum update has better convergence rates on neural networks. Mathematically
 *        like below:
 *
 *        .. math::
 *
 *        v_1 = \alpha * \nabla J(W_0)\\
 *        v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\
 *        W_t = W_{t-1} + v_t
 *
 *        It updates the weights using::
 *
 *        v = momentum * v - learning_rate * gradient
 *        weight += v
 *
 *        Where the parameter ``momentum`` is the decay rate of momentum estimates at
 *
 *
 *
 *        Defined in ../src/operator/contrib/preloaded_multi_sgd.cc:L200
 * \param symbol_name name of the resulting symbol
 * \param data Weights, gradients, momentums, learning rates and weight decays
 * \param momentum The decay rate of momentum estimates at each epoch.
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param num_weights Number of updated weights.
 * \return new symbol
 */
inline Symbol preloaded_multi_mp_sgd_mom_update(const std::string& symbol_name,
                                                const std::vector<Symbol>& data,
                                                mx_float momentum = 0,
                                                mx_float rescale_grad = 1,
                                                mx_float clip_gradient = -1,
                                                int num_weights = 1) {
  return Operator("preloaded_multi_mp_sgd_mom_update")
           .SetParam("momentum", momentum)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("num_weights", num_weights)
(data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Set to zero multiple arrays
 *
 *
 *        Defined in ../src/operator/contrib/reset_arrays.cc:L36
 * \param symbol_name name of the resulting symbol
 * \param data Arrays
 * \param num_arrays number of input arrays.
 * \return new symbol
 */
inline Symbol reset_arrays(const std::string& symbol_name,
                           const std::vector<Symbol>& data,
                           int num_arrays) {
  return Operator("reset_arrays")
           .SetParam("num_arrays", num_arrays)
(data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Apply a custom operator implemented in a frontend language (like Python).
 *
 *        Custom operators should override required methods like `forward` and `backward`.
 *        The custom operator must be registered before it can be used.
 *        Please check the tutorial here:
 *
 *
 *
 *        Defined in ../src/operator/custom/custom.cc:L547
 * \param symbol_name name of the resulting symbol
 * \param data Input data for the custom operator.
 * \param op_type Name of the custom operator. This is the name that is passed to
 * \return new symbol
 */
inline Symbol Custom(const std::string& symbol_name,
                     const std::vector<Symbol>& data,
                     const std::string& op_type) {
  return Operator("Custom")
(data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Apply a sparse regularization to the output a sigmoid activation function.
 * \param symbol_name name of the resulting symbol
 * \param data Input data.
 * \param sparseness_target The sparseness target
 * \param penalty The tradeoff parameter for the sparseness penalty
 * \param momentum The momentum for running average
 * \return new symbol
 */
inline Symbol IdentityAttachKLSparseReg(const std::string& symbol_name,
                                        Symbol data,
                                        mx_float sparseness_target = 0.100000001,
                                        mx_float penalty = 0.00100000005,
                                        mx_float momentum = 0.899999976) {
  return Operator("IdentityAttachKLSparseReg")
           .SetParam("sparseness_target", sparseness_target)
           .SetParam("penalty", penalty)
           .SetParam("momentum", momentum)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \brief Activation function to be applied.
 */
enum class LeakyReLUActType {
  kElu = 0,
  kGelu = 1,
  kLeaky = 2,
  kPrelu = 3,
  kRrelu = 4,
  kSelu = 5
};

/*!
 * \brief Applies Leaky rectified linear unit activation element-wise to the input.
 *
 *        Leaky ReLUs attempt to fix the "dying ReLU" problem by allowing a small `slope`
 *        when the input is negative and has a slope of one when input is positive.
 *
 *        The following modified ReLU Activation functions are supported:
 *
 *        - *elu*: Exponential Linear Unit. `y = x > 0 ? x : slope * (exp(x)-1)`
 *        - *selu*: Scaled Exponential Linear Unit. `y = lambda * (x > 0 ? x : alpha *
 *        *lambda = 1.0507009873554804934193349852946* and *alpha =
 *        - *leaky*: Leaky ReLU. `y = x > 0 ? x : slope * x`
 *        - *prelu*: Parametric ReLU. This is same as *leaky* except that `slope` is
 *        - *rrelu*: Randomized ReLU. same as *leaky* but the `slope` is uniformly and
 *        *[lower_bound, upper_bound)* for training, while fixed to be
 *        *(lower_bound+upper_bound)/2* for inference.
 *
 *
 *
 *        Defined in ../src/operator/leaky_relu.cc:L163
 * \param symbol_name name of the resulting symbol
 * \param data Input data to activation function.
 * \param gamma Input data to activation function.
 * \param act_type Activation function to be applied.
 * \param slope Init slope for the activation. (For leaky and elu only)
 * \param lower_bound Lower bound of random slope. (For rrelu only)
 * \param upper_bound Upper bound of random slope. (For rrelu only)
 * \return new symbol
 */
inline Symbol LeakyReLU(const std::string& symbol_name,
                        Symbol data,
                        Symbol gamma,
                        LeakyReLUActType act_type = LeakyReLUActType::kLeaky,
                        mx_float slope = 0.25,
                        mx_float lower_bound = 0.125,
                        mx_float upper_bound = 0.333999991) {
  static const char *LeakyReLUActTypeValues[] = {
    "elu",
    "gelu",
    "leaky",
    "prelu",
    "rrelu",
    "selu"
  };
  return Operator("LeakyReLU")
           .SetParam("act_type", LeakyReLUActTypeValues[int(act_type)])
           .SetParam("slope", slope)
           .SetParam("lower_bound", lower_bound)
           .SetParam("upper_bound", upper_bound)
           .SetInput("data", data)
           .SetInput("gamma", gamma)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Calculate cross entropy of softmax output and one-hot label.
 *
 *        - This operator computes the cross entropy in two steps:
 *        - Applies softmax function on the input array.
 *        - Computes and returns the cross entropy loss between the softmax output and
 *
 *        - The softmax function and cross entropy loss is given by:
 *
 *        - Softmax Function:
 *
 *        .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}
 *
 *        - Cross Entropy Function:
 *
 *        .. math:: \text{CE(label, output)} = - \sum_i \text{label}_i
 *
 *        Example::
 *
 *        x = [[1, 2, 3],
 *        [11, 7, 5]]
 *
 *        label = [2, 0]
 *
 *        softmax(x) = [[0.09003057, 0.24472848, 0.66524094],
 *        [0.97962922, 0.01794253, 0.00242826]]
 *
 *        softmax_cross_entropy(data, label) = - log(0.66524084) - log(0.97962922) =
 *
 *
 *
 *        Defined in ../src/operator/loss_binary_op.cc:L59
 * \param symbol_name name of the resulting symbol
 * \param data Input data
 * \param label Input label
 * \return new symbol
 */
inline Symbol softmax_cross_entropy(const std::string& symbol_name,
                                    Symbol data,
                                    Symbol label) {
  return Operator("softmax_cross_entropy")
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*! \brief Activation function to be applied.
 */
enum class ActivationActType {
  kRelu = 0,
  kSigmoid = 1,
  kSoftrelu = 2,
  kSoftsign = 3,
  kTanh = 4
};

/*!
 * \brief Applies an activation function element-wise to the input.
 *
 *        The following activation functions are supported:
 *
 *        - `relu`: Rectified Linear Unit, :math:`y = max(x, 0)`
 *        - `sigmoid`: :math:`y = \frac{1}{1 + exp(-x)}`
 *        - `tanh`: Hyperbolic tangent, :math:`y = \frac{exp(x) - exp(-x)}{exp(x) +
 *        - `softrelu`: Soft ReLU, or SoftPlus, :math:`y = log(1 + exp(x))`
 *        - `softsign`: :math:`y = \frac{x}{1 + abs(x)}`
 *
 *
 *
 *        Defined in ../src/operator/nn/activation.cc:L165
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \param act_type Activation function to be applied.
 * \return new symbol
 */
inline Symbol Activation(const std::string& symbol_name,
                         Symbol data,
                         ActivationActType act_type) {
  static const char *ActivationActTypeValues[] = {
    "relu",
    "sigmoid",
    "softrelu",
    "softsign",
    "tanh"
  };
  return Operator("Activation")
           .SetParam("act_type", ActivationActTypeValues[int(act_type)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Batch normalization.
 *
 *        Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as
 *        well as offset ``beta``.
 *
 *        Assume the input has more than one dimension and we normalize along axis 1.
 *        We first compute the mean and variance along this axis:
 *
 *        .. math::
 *
 *        data\_mean[i] = mean(data[:,i,:,...]) \\
 *        data\_var[i] = var(data[:,i,:,...])
 *
 *        Then compute the normalized output, which has the same shape as input, as
 *
 *        .. math::
 *
 *        out[:,i,:,...] = \frac{data[:,i,:,...] -
 *
 *        Both *mean* and *var* returns a scalar by treating the input as a vector.
 *
 *        Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
 *        have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both
 *        the inverse of ``data_var``, which are needed for the backward pass. Note that
 *        two outputs are blocked.
 *
 *        Besides the inputs and the outputs, this operator accepts two auxiliary
 *        states, ``moving_mean`` and ``moving_var``, which are *k*-length
 *        vectors. They are global statistics for the whole dataset, which are updated
 *        by::
 *
 *        moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
 *        moving_var = moving_var * momentum + data_var * (1 - momentum)
 *
 *        If ``use_global_stats`` is set to be true, then ``moving_mean`` and
 *        ``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute
 *        the output. It is often used during inference.
 *
 *        The parameter ``axis`` specifies which axis of the input shape denotes
 *        the 'channel' (separately normalized groups).  The default is 1.  Specifying -1
 *        axis to be the last item in the input shape.
 *
 *        Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is
 *        then set ``gamma`` to 1 and its gradient to 0.
 *
 *        .. Note::
 *        When ``fix_gamma`` is set to True, no sparse support is provided. If
 *        the sparse tensors will fallback.
 *
 *
 *
 *        Defined in ../src/operator/nn/batch_norm.cc:L607
 * \param symbol_name name of the resulting symbol
 * \param data Input data to batch normalization
 * \param gamma gamma array
 * \param beta beta array
 * \param moving_mean running mean of input
 * \param moving_var running variance of input
 * \param eps Epsilon to prevent div 0. Must be no less than CUDNN_BN_MIN_EPSILON defined
 * \param momentum Momentum for moving average
 * \param fix_gamma Fix gamma while training
 * \param use_global_stats Whether use global moving statistics instead of local
 * \param output_mean_var Output the mean and inverse std
 * \param axis Specify which shape axis the channel is specified
 * \param cudnn_off Do not select CUDNN operator, if available
 * \param min_calib_range The minimum scalar value in the form of float32 obtained
 *        through calibration. If present, it will be used to by quantized batch norm op
 * \param max_calib_range The maximum scalar value in the form of float32 obtained
 *        through calibration. If present, it will be used to by quantized batch norm op
 * \return new symbol
 */
inline Symbol BatchNorm(const std::string& symbol_name,
                        Symbol data,
                        Symbol gamma,
                        Symbol beta,
                        Symbol moving_mean,
                        Symbol moving_var,
                        double eps = 0.0010000000474974513,
                        mx_float momentum = 0.899999976,
                        bool fix_gamma = true,
                        bool use_global_stats = false,
                        bool output_mean_var = false,
                        int axis = 1,
                        bool cudnn_off = false,
                        dmlc::optional<float> min_calib_range = dmlc::optional<float>(),
                        dmlc::optional<float> max_calib_range = dmlc::optional<float>()) {
  return Operator("BatchNorm")
           .SetParam("eps", eps)
           .SetParam("momentum", momentum)
           .SetParam("fix_gamma", fix_gamma)
           .SetParam("use_global_stats", use_global_stats)
           .SetParam("output_mean_var", output_mean_var)
           .SetParam("axis", axis)
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("min_calib_range", min_calib_range)
           .SetParam("max_calib_range", max_calib_range)
           .SetInput("data", data)
           .SetInput("gamma", gamma)
           .SetInput("beta", beta)
           .SetInput("moving_mean", moving_mean)
           .SetInput("moving_var", moving_var)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Joins input arrays along a given axis.
 *
 *        .. note:: `Concat` is deprecated. Use `concat` instead.
 *
 *        The dimensions of the input arrays should be the same except the axis along
 *        which they will be concatenated.
 *        The dimension of the output array along the concatenated axis will be equal
 *        to the sum of the corresponding dimensions of the input arrays.
 *
 *        The storage type of ``concat`` output depends on storage types of inputs
 *
 *        - concat(csr, csr, ..., csr, dim=0) = csr
 *        - otherwise, ``concat`` generates output with default storage
 *
 *        Example::
 *
 *        x = [[1,1],[2,2]]
 *        y = [[3,3],[4,4],[5,5]]
 *        z = [[6,6], [7,7],[8,8]]
 *
 *        concat(x,y,z,dim=0) = [[ 1.,  1.],
 *        [ 2.,  2.],
 *        [ 3.,  3.],
 *        [ 4.,  4.],
 *        [ 5.,  5.],
 *        [ 6.,  6.],
 *        [ 7.,  7.],
 *        [ 8.,  8.]]
 *
 *        Note that you cannot concat x,y,z along dimension 1 since dimension
 *        0 is not the same for all the input arrays.
 *
 *        concat(y,z,dim=1) = [[ 3.,  3.,  6.,  6.],
 *        [ 4.,  4.,  7.,  7.],
 *        [ 5.,  5.,  8.,  8.]]
 *
 *
 *
 *        Defined in ../src/operator/nn/concat.cc:L385
 * \param symbol_name name of the resulting symbol
 * \param data List of arrays to concatenate
 * \param num_args Number of inputs to be concated.
 * \param dim the dimension to be concated.
 * \return new symbol
 */
inline Symbol Concat(const std::string& symbol_name,
                     const std::vector<Symbol>& data,
                     int num_args,
                     int dim = 1) {
  return Operator("Concat")
           .SetParam("num_args", num_args)
           .SetParam("dim", dim)
(data)
           .CreateSymbol(symbol_name);
}

/*! \brief Whether to pick convolution algo by running performance test.
 */
enum class ConvolutionCudnnTune {
  kNone = 0,
  kFastest = 1,
  kLimited_workspace = 2,
  kOff = 3
};

/*! \brief Set layout for input, output and weight. Empty for
 *        default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.NHWC and NDHWC are
 */
enum class ConvolutionLayout {
  kNone = 0,
  kNCDHW = 1,
  kNCHW = 2,
  kNCW = 3,
  kNDHWC = 4,
  kNHWC = 5
};

/*!
 * \brief Compute *N*-D convolution on *(N+2)*-D input.
 *
 *        In the 2-D convolution, given input data with shape *(batch_size,
 *        channel, height, width)*, the output is computed by
 *
 *        .. math::
 *
 *        out[n,i,:,:] = bias[i] + \sum_{j=0}^{channel} data[n,j,:,:] \star
 *        weight[i,j,:,:]
 *
 *        where :math:`\star` is the 2-D cross-correlation operator.
 *
 *        For general 2-D convolution, the shapes are
 *
 *        - **data**: *(batch_size, channel, height, width)*
 *        - **weight**: *(num_filter, channel, kernel[0], kernel[1])*
 *        - **bias**: *(num_filter,)*
 *        - **out**: *(batch_size, num_filter, out_height, out_width)*.
 *
 *        Define::
 *
 *        f(x,k,p,s,d) = floor((x+2*p-d*(k-1)-1)/s)+1
 *
 *        then we have::
 *
 *        out_height=f(height, kernel[0], pad[0], stride[0], dilate[0])
 *        out_width=f(width, kernel[1], pad[1], stride[1], dilate[1])
 *
 *        If ``no_bias`` is set to be true, then the ``bias`` term is ignored.
 *
 *        The default data ``layout`` is *NCHW*, namely *(batch_size, channel, height,
 *        width)*. We can choose other layouts such as *NWC*.
 *
 *        If ``num_group`` is larger than 1, denoted by *g*, then split the input ``data``
 *        evenly into *g* parts along the channel axis, and also evenly split ``weight``
 *        along the first dimension. Next compute the convolution on the *i*-th part of
 *        the data with the *i*-th weight part. The output is obtained by concatenating
 *        the *g* results.
 *
 *        1-D convolution does not have *height* dimension but only *width* in space.
 *
 *        - **data**: *(batch_size, channel, width)*
 *        - **weight**: *(num_filter, channel, kernel[0])*
 *        - **bias**: *(num_filter,)*
 *        - **out**: *(batch_size, num_filter, out_width)*.
 *
 *        3-D convolution adds an additional *depth* dimension besides *height* and
 *        *width*. The shapes are
 *
 *        - **data**: *(batch_size, channel, depth, height, width)*
 *        - **weight**: *(num_filter, channel, kernel[0], kernel[1], kernel[2])*
 *        - **bias**: *(num_filter,)*
 *        - **out**: *(batch_size, num_filter, out_depth, out_height, out_width)*.
 *
 *        Both ``weight`` and ``bias`` are learnable parameters.
 *
 *        There are other options to tune the performance.
 *
 *        - **cudnn_tune**: enable this option leads to higher startup time but may give
 *        faster speed. Options are
 *
 *        - **off**: no tuning
 *        - **limited_workspace**:run test and pick the fastest algorithm that doesn't
 *        exceed workspace limit.
 *        - **fastest**: pick the fastest algorithm and ignore workspace limit.
 *        - **None** (default): the behavior is determined by environment variable
 *        ``MXNET_CUDNN_AUTOTUNE_DEFAULT``. 0 for off, 1 for limited workspace
 *        (default), 2 for fastest.
 *
 *        - **workspace**: A large number leads to more (GPU) memory usage but may improve
 *        the performance.
 *
 *
 *
 *        Defined in ../src/operator/nn/convolution.cc:L476
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the ConvolutionOp.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param kernel Convolution kernel size: (w,), (h, w) or (d, h, w)
 * \param num_filter Convolution filter(channel) number
 * \param stride Convolution stride: (w,), (h, w) or (d, h, w). Defaults to 1 for each
 * \param dilate Convolution dilate: (w,), (h, w) or (d, h, w). Defaults to 1 for each
 * \param pad Zero pad for convolution: (w,), (h, w) or (d, h, w). Defaults to no padding.
 * \param num_group Number of group partitions.
 * \param workspace Maximum temporary workspace allowed (MB) in convolution.This
 *        parameter has two usages. When CUDNN is not used, it determines the effective
 *        batch size of the convolution kernel. When CUDNN is used, it controls the
 *        maximum temporary storage used for tuning the best CUDNN kernel when
 * \param no_bias Whether to disable bias parameter.
 * \param cudnn_tune Whether to pick convolution algo by running performance test.
 * \param cudnn_off Turn off cudnn for this layer.
 * \param layout Set layout for input, output and weight. Empty for
 *        default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.NHWC and NDHWC are
 * \return new symbol
 */
inline Symbol Convolution(const std::string& symbol_name,
                          Symbol data,
                          Symbol weight,
                          Symbol bias,
                          Shape kernel,
                          uint32_t num_filter,
                          Shape stride = Shape(),
                          Shape dilate = Shape(),
                          Shape pad = Shape(),
                          uint32_t num_group = 1,
                          uint64_t workspace = 1024,
                          bool no_bias = false,
                          ConvolutionCudnnTune cudnn_tune = ConvolutionCudnnTune::kNone,
                          bool cudnn_off = false,
                          ConvolutionLayout layout = ConvolutionLayout::kNone) {
  static const char *ConvolutionCudnnTuneValues[] = {
    "None",
    "fastest",
    "limited_workspace",
    "off"
  };
  static const char *ConvolutionLayoutValues[] = {
    "None",
    "NCDHW",
    "NCHW",
    "NCW",
    "NDHWC",
    "NHWC"
  };
  return Operator("Convolution")
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("dilate", dilate)
           .SetParam("pad", pad)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
           .SetParam("cudnn_tune", ConvolutionCudnnTuneValues[int(cudnn_tune)])
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("layout", ConvolutionLayoutValues[int(layout)])
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol(symbol_name);
}

/*! \brief Set the label that is reserved for blank label.If "first", 0-th label is
 *        reserved, and label values for tokens in the vocabulary are between ``1`` and
 *        ``alphabet_size-1``, and the padding mask is ``-1``. If "last", last label
 *        value ``alphabet_size-1`` is reserved for blank label instead, and label values
 *        for tokens in the vocabulary are between ``0`` and ``alphabet_size-2``, and the
 */
enum class CTCLossBlankLabel {
  kFirst = 0,
  kLast = 1
};

/*!
 * \brief Connectionist Temporal Classification Loss.
 *
 *        .. note:: The existing alias ``contrib_CTCLoss`` is deprecated.
 *
 *        The shapes of the inputs and outputs:
 *
 *        - **data**: `(sequence_length, batch_size, alphabet_size)`
 *        - **label**: `(batch_size, label_sequence_length)`
 *        - **out**: `(batch_size)`
 *
 *        The `data` tensor consists of sequences of activation vectors (without applying
 *        with i-th channel in the last dimension corresponding to i-th label
 *        for i between 0 and alphabet_size-1 (i.e always 0-indexed).
 *        Alphabet size should include one additional value reserved for blank label.
 *        When `blank_label` is ``"first"``, the ``0``-th channel is be reserved for
 *        activation of blank label, or otherwise if it is "last",
 *        reserved for blank label.
 *
 *        ``label`` is an index matrix of integers. When `blank_label` is ``"first"``,
 *        the value 0 is then reserved for blank label, and should not be passed in this
 *        when `blank_label` is ``"last"``, the value `(alphabet_size-1)` is reserved for
 *
 *        If a sequence of labels is shorter than *label_sequence_length*, use the special
 *        padding value at the end of the sequence to conform it to the correct
 *        length. The padding value is `0` when `blank_label` is ``"first"``, and `-1`
 *
 *        For example, suppose the vocabulary is `[a, b, c]`, and in one batch we have
 *        'ba', 'cbb', and 'abac'. When `blank_label` is ``"first"``, we can index the
 *        `{'a': 1, 'b': 2, 'c': 3}`, and we reserve the 0-th channel for blank label in
 *        The resulting `label` tensor should be padded to be::
 *
 *        [[2, 1, 0, 0], [3, 2, 2, 0], [1, 2, 1, 3]]
 *
 *        When `blank_label` is ``"last"``, we can index the labels as
 *        `{'a': 0, 'b': 1, 'c': 2}`, and we reserve the channel index 3 for blank label
 *        The resulting `label` tensor should be padded to be::
 *
 *        [[1, 0, -1, -1], [2, 1, 1, -1], [0, 1, 0, 2]]
 *
 *        ``out`` is a list of CTC loss values, one per example in the batch.
 *
 *        See *Connectionist Temporal Classification: Labelling Unsegmented
 *        Sequence Data with Recurrent Neural Networks*, A. Graves *et al*. for more
 *        information on the definition and the algorithm.
 *
 *
 *
 *        Defined in ../src/operator/nn/ctc_loss.cc:L100
 * \param symbol_name name of the resulting symbol
 * \param data Input ndarray
 * \param label Ground-truth labels for the loss.
 * \param data_lengths Lengths of data for each of the samples. Only required when
 * \param label_lengths Lengths of labels for each of the samples. Only required when
 * \param use_data_lengths Whether the data lenghts are decided by `data_lengths`. If
 * \param use_label_lengths Whether the label lenghts are decided by `label_lengths`, or
 *        derived from `padding_mask`. If false, the lengths are derived from the first
 *        occurrence of the value of `padding_mask`. The value of `padding_mask` is ``0``
 *        when first CTC label is reserved for blank, and ``-1`` when last label is
 * \param blank_label Set the label that is reserved for blank label.If "first", 0-th
 *        label is reserved, and label values for tokens in the vocabulary are between
 *        ``1`` and ``alphabet_size-1``, and the padding mask is ``-1``. If "last", last
 *        label value ``alphabet_size-1`` is reserved for blank label instead, and label
 *        values for tokens in the vocabulary are between ``0`` and ``alphabet_size-2``,
 * \return new symbol
 */
inline Symbol CTCLoss(const std::string& symbol_name,
                      Symbol data,
                      Symbol label,
                      Symbol data_lengths,
                      Symbol label_lengths,
                      bool use_data_lengths = false,
                      bool use_label_lengths = false,
                      CTCLossBlankLabel blank_label = CTCLossBlankLabel::kFirst) {
  static const char *CTCLossBlankLabelValues[] = {
    "first",
    "last"
  };
  return Operator("CTCLoss")
           .SetParam("use_data_lengths", use_data_lengths)
           .SetParam("use_label_lengths", use_label_lengths)
           .SetParam("blank_label", CTCLossBlankLabelValues[int(blank_label)])
           .SetInput("data", data)
           .SetInput("label", label)
           .SetInput("data_lengths", data_lengths)
           .SetInput("label_lengths", label_lengths)
           .CreateSymbol(symbol_name);
}

/*! \brief Whether to pick convolution algorithm by running performance test.
 */
enum class DeconvolutionCudnnTune {
  kNone = 0,
  kFastest = 1,
  kLimited_workspace = 2,
  kOff = 3
};

/*! \brief Set layout for input, output and weight. Empty for default layout, NCW for 1d,
 */
enum class DeconvolutionLayout {
  kNone = 0,
  kNCDHW = 1,
  kNCHW = 2,
  kNCW = 3,
  kNDHWC = 4,
  kNHWC = 5
};

/*!
 * \brief Computes 1D or 2D transposed convolution (aka fractionally strided convolution)
 *        of the input tensor. This operation can be seen as the gradient of Convolution
 *        operation with respect to its input. Convolution usually reduces the size of
 *        the input. Transposed convolution works the other way, going from a smaller
 * \param symbol_name name of the resulting symbol
 * \param data Input tensor to the deconvolution operation.
 * \param weight Weights representing the kernel.
 * \param bias Bias added to the result after the deconvolution operation.
 * \param kernel Deconvolution kernel size: (w,), (h, w) or (d, h, w). This is same as
 * \param num_filter Number of output filters.
 * \param stride The stride used for the corresponding convolution: (w,), (h, w) or (d,
 * \param dilate Dilation factor for each dimension of the input: (w,), (h, w) or (d, h,
 * \param pad The amount of implicit zero padding added during convolution for each
 *        dimension of the input: (w,), (h, w) or (d, h, w). ``(kernel-1)/2`` is usually
 *        a good choice. If `target_shape` is set, `pad` will be ignored and a padding
 * \param adj Adjustment for output shape: (w,), (h, w) or (d, h, w). If `target_shape`
 * \param target_shape Shape of the output tensor: (w,), (h, w) or (d, h, w).
 * \param num_group Number of groups partition.
 * \param workspace Maximum temporary workspace allowed (MB) in deconvolution.This
 *        parameter has two usages. When CUDNN is not used, it determines the effective
 *        batch size of the deconvolution kernel. When CUDNN is used, it controls the
 *        maximum temporary storage used for tuning the best CUDNN kernel when
 * \param no_bias Whether to disable bias parameter.
 * \param cudnn_tune Whether to pick convolution algorithm by running performance test.
 * \param cudnn_off Turn off cudnn for this layer.
 * \param layout Set layout for input, output and weight. Empty for default layout, NCW
 * \return new symbol
 */
inline Symbol Deconvolution(const std::string& symbol_name,
                            Symbol data,
                            Symbol weight,
                            Symbol bias,
                            Shape kernel,
                            uint32_t num_filter,
                            Shape stride = Shape(),
                            Shape dilate = Shape(),
                            Shape pad = Shape(),
                            Shape adj = Shape(),
                            Shape target_shape = Shape(),
                            uint32_t num_group = 1,
                            uint64_t workspace = 512,
                            bool no_bias = true,
                            DeconvolutionCudnnTune cudnn_tune = DeconvolutionCudnnTune::kNone,
                            bool cudnn_off = false,
                            DeconvolutionLayout layout = DeconvolutionLayout::kNone) {
  static const char *DeconvolutionCudnnTuneValues[] = {
    "None",
    "fastest",
    "limited_workspace",
    "off"
  };
  static const char *DeconvolutionLayoutValues[] = {
    "None",
    "NCDHW",
    "NCHW",
    "NCW",
    "NDHWC",
    "NHWC"
  };
  return Operator("Deconvolution")
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("dilate", dilate)
           .SetParam("pad", pad)
           .SetParam("adj", adj)
           .SetParam("target_shape", target_shape)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
           .SetParam("cudnn_tune", DeconvolutionCudnnTuneValues[int(cudnn_tune)])
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("layout", DeconvolutionLayoutValues[int(layout)])
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol(symbol_name);
}

/*! \brief Whether to only turn on dropout during training or to also turn on for
 */
enum class DropoutMode {
  kAlways = 0,
  kTraining = 1
};

/*!
 * \brief Applies dropout operation to input array.
 *
 *        - During training, each element of the input is set to zero with probability p.
 *        The whole array is rescaled by :math:`1/(1-p)` to keep the expected
 *        sum of the input unchanged.
 *
 *        - During testing, this operator does not change the input if mode is 'training'.
 *        If mode is 'always', the same computaion as during training will be applied.
 *
 *        Example::
 *
 *        random.seed(998)
 *        input_array = array([[3., 0.5,  -0.5,  2., 7.],
 *        [2., -0.4,   7.,  3., 0.2]])
 *        a = symbol.Variable('a')
 *        dropout = symbol.Dropout(a, p = 0.2)
 *        executor = dropout.simple_bind(a = input_array.shape)
 *
 *        ## If training
 *        executor.forward(is_train = True, a = input_array)
 *        executor.outputs
 *        [[ 3.75   0.625 -0.     2.5    8.75 ]
 *        [ 2.5   -0.5    8.75   3.75   0.   ]]
 *
 *        ## If testing
 *        executor.forward(is_train = False, a = input_array)
 *        executor.outputs
 *        [[ 3.     0.5   -0.5    2.     7.   ]
 *        [ 2.    -0.4    7.     3.     0.2  ]]
 *
 *
 *        Defined in ../src/operator/nn/dropout.cc:L96
 * \param symbol_name name of the resulting symbol
 * \param data Input array to which dropout will be applied.
 * \param p Fraction of the input that gets dropped out during training time.
 * \param mode Whether to only turn on dropout during training or to also turn on for
 * \param axes Axes for variational dropout kernel.
 * \param cudnn_off Whether to turn off cudnn in dropout operator. This option is ignored
 * \return new symbol
 */
inline Symbol Dropout(const std::string& symbol_name,
                      Symbol data,
                      mx_float p = 0.5,
                      DropoutMode mode = DropoutMode::kTraining,
                      Shape axes = Shape(),
                      dmlc::optional<bool> cudnn_off = dmlc::optional<bool>(0)) {
  static const char *DropoutModeValues[] = {
    "always",
    "training"
  };
  return Operator("Dropout")
           .SetParam("p", p)
           .SetParam("mode", DropoutModeValues[int(mode)])
           .SetParam("axes", axes)
           .SetParam("cudnn_off", cudnn_off)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Applies a linear transformation: :math:`Y = XW^T + b`.
 *
 *        If ``flatten`` is set to be true, then the shapes are:
 *
 *        - **data**: `(batch_size, x1, x2, ..., xn)`
 *        - **weight**: `(num_hidden, x1 * x2 * ... * xn)`
 *        - **bias**: `(num_hidden,)`
 *        - **out**: `(batch_size, num_hidden)`
 *
 *        If ``flatten`` is set to be false, then the shapes are:
 *
 *        - **data**: `(x1, x2, ..., xn, input_dim)`
 *        - **weight**: `(num_hidden, input_dim)`
 *        - **bias**: `(num_hidden,)`
 *        - **out**: `(x1, x2, ..., xn, num_hidden)`
 *
 *        The learnable parameters include both ``weight`` and ``bias``.
 *
 *        If ``no_bias`` is set to be true, then the ``bias`` term is ignored.
 *
 *        .. Note::
 *
 *        The sparse support for FullyConnected is limited to forward evaluation with
 *        weight and bias, where the length of `weight.indices` and `bias.indices` must
 *        to `num_hidden`. This could be useful for model inference with `row_sparse`
 *        trained with importance sampling or noise contrastive estimation.
 *
 *        To compute linear transformation with 'csr' sparse data, sparse.dot is
 *        of sparse.FullyConnected.
 *
 *
 *
 *        Defined in ../src/operator/nn/fully_connected.cc:L287
 * \param symbol_name name of the resulting symbol
 * \param data Input data.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param num_hidden Number of hidden nodes of the output.
 * \param no_bias Whether to disable bias parameter.
 * \param flatten Whether to collapse all but the first axis of the input data tensor.
 * \return new symbol
 */
inline Symbol FullyConnected(const std::string& symbol_name,
                             Symbol data,
                             Symbol weight,
                             Symbol bias,
                             int num_hidden,
                             bool no_bias = false,
                             bool flatten = true) {
  return Operator("FullyConnected")
           .SetParam("num_hidden", num_hidden)
           .SetParam("no_bias", no_bias)
           .SetParam("flatten", flatten)
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Group normalization.
 *
 *        The input channels are separated into ``num_groups`` groups, each containing
 *        The mean and standard-deviation are calculated separately over the each group.
 *
 *        .. math::
 *
 *        data = data.reshape((N, num_groups, C // num_groups, ...))
 *        out = \frac{data - mean(data, axis)}{\sqrt{var(data, axis) + \epsilon}} * gamma
 *
 *        Both ``gamma`` and ``beta`` are learnable parameters.
 *
 *
 *
 *        Defined in ../src/operator/nn/group_norm.cc:L77
 * \param symbol_name name of the resulting symbol
 * \param data Input data
 * \param gamma gamma array
 * \param beta beta array
 * \param num_groups Total number of groups.
 * \param eps An `epsilon` parameter to prevent division by 0.
 * \param output_mean_var Output the mean and std calculated along the given axis.
 * \return new symbol
 */
inline Symbol GroupNorm(const std::string& symbol_name,
                        Symbol data,
                        Symbol gamma,
                        Symbol beta,
                        int num_groups = 1,
                        mx_float eps = 9.99999975e-06,
                        bool output_mean_var = false) {
  return Operator("GroupNorm")
           .SetParam("num_groups", num_groups)
           .SetParam("eps", eps)
           .SetParam("output_mean_var", output_mean_var)
           .SetInput("data", data)
           .SetInput("gamma", gamma)
           .SetInput("beta", beta)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Extract sliding blocks from input array.
 *
 *        This operator is used in vanilla convolution implementation to transform the
 *        blocks on image to column matrix, then the convolution operation can be computed
 *        by matrix multiplication between column and convolution weight. Due to the close
 *        relation between im2col and convolution, the concept of **kernel**, **stride**,
 *        **dilate** and **pad** in this operator are inherited from convolution
 *
 *        Given the input data of shape :math:`(N, C, *)`, where :math:`N` is the batch
 *        :math:`C` is the channel size, and :math:`*` is the arbitrary spatial dimension,
 *        the output column array is always with shape :math:`(N, C \times
 *        where :math:`C \times \prod(\text{kernel})` is the block size, and :math:`W` is
 *        block number which is the spatial size of the convolution output with same
 *        Only 1-D, 2-D and 3-D of spatial dimension is supported in this operator.
 *
 *
 *
 *        Defined in ../src/operator/nn/im2col.cc:L100
 * \param symbol_name name of the resulting symbol
 * \param data Input array to extract sliding blocks.
 * \param kernel Sliding kernel size: (w,), (h, w) or (d, h, w).
 * \param stride The stride between adjacent sliding blocks in spatial dimension: (w,),
 * \param dilate The spacing between adjacent kernel points: (w,), (h, w) or (d, h, w).
 * \param pad The zero-value padding size on both sides of spatial dimension: (w,), (h,
 * \return new symbol
 */
inline Symbol im2col(const std::string& symbol_name,
                     Symbol data,
                     Shape kernel,
                     Shape stride = Shape(),
                     Shape dilate = Shape(),
                     Shape pad = Shape()) {
  return Operator("im2col")
           .SetParam("kernel", kernel)
           .SetParam("stride", stride)
           .SetParam("dilate", dilate)
           .SetParam("pad", pad)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Combining the output column matrix of im2col back to image array.
 *
 *        Like :class:`~mxnet.ndarray.im2col`, this operator is also used in the vanilla
 *        implementation. Despite the name, col2im is not the reverse operation of
 *        may be overlaps between neighbouring sliding blocks, the column elements cannot
 *        put back into image. Instead, they are accumulated (i.e., summed) in the input
 *        just like the gradient computation, so col2im is the gradient of im2col and
 *
 *        Using the notation in im2col, given an input column array of shape
 *        :math:`(N, C \times  \prod(\text{kernel}), W)`, this operator accumulates the
 *        into output array of shape :math:`(N, C, \text{output_size}[0],
 *        Only 1-D, 2-D and 3-D of spatial dimension is supported in this operator.
 *
 *
 *
 *        Defined in ../src/operator/nn/im2col.cc:L182
 * \param symbol_name name of the resulting symbol
 * \param data Input array to combine sliding blocks.
 * \param output_size The spatial dimension of image array: (w,), (h, w) or (d, h, w).
 * \param kernel Sliding kernel size: (w,), (h, w) or (d, h, w).
 * \param stride The stride between adjacent sliding blocks in spatial dimension: (w,),
 * \param dilate The spacing between adjacent kernel points: (w,), (h, w) or (d, h, w).
 * \param pad The zero-value padding size on both sides of spatial dimension: (w,), (h,
 * \return new symbol
 */
inline Symbol col2im(const std::string& symbol_name,
                     Symbol data,
                     Shape output_size,
                     Shape kernel,
                     Shape stride = Shape(),
                     Shape dilate = Shape(),
                     Shape pad = Shape()) {
  return Operator("col2im")
           .SetParam("output_size", output_size)
           .SetParam("kernel", kernel)
           .SetParam("stride", stride)
           .SetParam("dilate", dilate)
           .SetParam("pad", pad)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Layer normalization.
 *
 *        Normalizes the channels of the input tensor by mean and variance, and applies a
 *        well as offset ``beta``.
 *
 *        Assume the input has more than one dimension and we normalize along axis 1.
 *        We first compute the mean and variance along this axis and then
 *        compute the normalized output, which has the same shape as input, as following:
 *
 *        .. math::
 *
 *        out = \frac{data - mean(data, axis)}{\sqrt{var(data, axis) + \epsilon}} * gamma
 *
 *        Both ``gamma`` and ``beta`` are learnable parameters.
 *
 *        Unlike BatchNorm and InstanceNorm,  the *mean* and *var* are computed along the
 *
 *        Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
 *        have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both
 *        ``data_std``. Note that no gradient will be passed through these two outputs.
 *
 *        The parameter ``axis`` specifies which axis of the input shape denotes
 *        the 'channel' (separately normalized groups).  The default is -1, which sets
 *        axis to be the last item in the input shape.
 *
 *
 *
 *        Defined in ../src/operator/nn/layer_norm.cc:L159
 * \param symbol_name name of the resulting symbol
 * \param data Input data to layer normalization
 * \param gamma gamma array
 * \param beta beta array
 * \param axis The axis to perform layer normalization. Usually, this should be be axis
 * \param eps An `epsilon` parameter to prevent division by 0.
 * \param output_mean_var Output the mean and std calculated along the given axis.
 * \return new symbol
 */
inline Symbol LayerNorm(const std::string& symbol_name,
                        Symbol data,
                        Symbol gamma,
                        Symbol beta,
                        int axis = -1,
                        mx_float eps = 9.99999975e-06,
                        bool output_mean_var = false) {
  return Operator("LayerNorm")
           .SetParam("axis", axis)
           .SetParam("eps", eps)
           .SetParam("output_mean_var", output_mean_var)
           .SetInput("data", data)
           .SetInput("gamma", gamma)
           .SetInput("beta", beta)
           .CreateSymbol(symbol_name);
}

/*! \brief DType of the output in case this can't be inferred. Defaults to the same as
 */
enum class Log_softmaxDtype {
  kNone = 0,
  kFloat16 = 1,
  kFloat32 = 2,
  kFloat64 = 3
};

/*!
 * \brief Computes the log softmax of the input.
 *        This is equivalent to computing softmax followed by log.
 *
 *        Examples::
 *
 *        >>> x = mx.nd.array([1, 2, .1])
 *        >>> mx.nd.log_softmax(x).asnumpy()
 *        array([-1.41702998, -0.41702995, -2.31702995], dtype=float32)
 *
 *        >>> x = mx.nd.array( [[1, 2, .1],[.1, 2, 1]] )
 *        >>> mx.nd.log_softmax(x, axis=0).asnumpy()
 *        array([[-0.34115392, -0.69314718, -1.24115396],
 *        [-1.24115396, -0.69314718, -0.34115392]], dtype=float32)
 *
 *
 *
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \param axis The axis along which to compute softmax.
 * \param temperature Temperature parameter in softmax
 * \param dtype DType of the output in case this can't be inferred. Defaults to the same
 * \param use_length Whether to use the length input as a mask over the data input.
 * \return new symbol
 */
inline Symbol log_softmax(const std::string& symbol_name,
                          Symbol data,
                          int axis = -1,
                          dmlc::optional<double> temperature = dmlc::optional<double>(),
                          Log_softmaxDtype dtype = Log_softmaxDtype::kNone,
                          dmlc::optional<bool> use_length = dmlc::optional<bool>(0)) {
  static const char *Log_softmaxDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("log_softmax")
           .SetParam("axis", axis)
           .SetParam("temperature", temperature)
           .SetParam("dtype", Log_softmaxDtypeValues[int(dtype)])
           .SetParam("use_length", use_length)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Applies local response normalization to the input.
 *
 *        The local response normalization layer performs "lateral inhibition" by
 *        over local input regions.
 *
 *        If :math:`a_{x,y}^{i}` is the activity of a neuron computed by applying kernel
 *        :math:`(x, y)` and then applying the ReLU nonlinearity, the response-normalized
 *        activity :math:`b_{x,y}^{i}` is given by the expression:
 *
 *        .. math::
 *        b_{x,y}^{i} = \frac{a_{x,y}^{i}}{\Bigg({k + \frac{\alpha}{n} \sum_{j=max(0,
 *
 *        where the sum runs over :math:`n` "adjacent" kernel maps at the same spatial
 *        number of kernels in the layer.
 *
 *
 *
 *        Defined in ../src/operator/nn/lrn.cc:L158
 * \param symbol_name name of the resulting symbol
 * \param data Input data to LRN
 * \param nsize normalization window width in elements.
 * \param alpha The variance scaling parameter :math:`lpha` in the LRN expression.
 * \param beta The power parameter :math:`eta` in the LRN expression.
 * \param knorm The parameter :math:`k` in the LRN expression.
 * \return new symbol
 */
inline Symbol LRN(const std::string& symbol_name,
                  Symbol data,
                  uint32_t nsize,
                  mx_float alpha = 9.99999975e-05,
                  mx_float beta = 0.75,
                  mx_float knorm = 2) {
  return Operator("LRN")
           .SetParam("nsize", nsize)
           .SetParam("alpha", alpha)
           .SetParam("beta", beta)
           .SetParam("knorm", knorm)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief
 *        Calculate the mean and variance of `data`.
 *
 *        The mean and variance are calculated by aggregating the contents of data across
 *        If x is 1-D and axes = [0] this is just the mean and variance of a vector.
 *
 *        Example:
 *
 *        x = [[1, 2, 3], [4, 5, 6]]
 *        mean, var = moments(data=x, axes=[0])
 *        mean = [2.5, 3.5, 4.5]
 *        var = [2.25, 2.25, 2.25]
 *        mean, var = moments(data=x, axes=[1])
 *        mean = [2.0, 5.0]
 *        var = [0.66666667, 0.66666667]
 *        mean, var = moments(data=x, axis=[0, 1])
 *        mean = [3.5]
 *        var = [2.9166667]
 *
 *
 *
 *        Defined in ../src/operator/nn/moments.cc:L54
 * \param symbol_name name of the resulting symbol
 * \param data Input ndarray
 * \param axes Array of ints. Axes along which to compute mean and variance.
 * \param keepdims produce moments with the same dimensionality as the input.
 * \return new symbol
 */
inline Symbol moments(const std::string& symbol_name,
                      Symbol data,
                      dmlc::optional<Shape> axes = dmlc::optional<Shape>(),
                      bool keepdims = false) {
  return Operator("moments")
           .SetParam("axes", axes)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \brief Pooling type to be applied.
 */
enum class PoolingPoolType {
  kAvg = 0,
  kLp = 1,
  kMax = 2,
  kSum = 3
};

/*! \brief Pooling convention to be applied.
 */
enum class PoolingPoolingConvention {
  kFull = 0,
  kSame = 1,
  kValid = 2
};

/*! \brief Set layout for input and output. Empty for
 *        default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.
 */
enum class PoolingLayout {
  kNone = 0,
  kNCDHW = 1,
  kNCHW = 2,
  kNCW = 3,
  kNDHWC = 4,
  kNHWC = 5,
  kNWC = 6
};

/*!
 * \brief Performs pooling on the input.
 *
 *        The shapes for 1-D pooling are
 *
 *        - **data** and **out**: *(batch_size, channel, width)* (NCW layout) or
 *        *(batch_size, width, channel)* (NWC layout),
 *
 *        The shapes for 2-D pooling are
 *
 *        - **data** and **out**: *(batch_size, channel, height, width)* (NCHW layout) or
 *        *(batch_size, height, width, channel)* (NHWC layout),
 *
 *        out_height = f(height, kernel[0], pad[0], stride[0])
 *        out_width = f(width, kernel[1], pad[1], stride[1])
 *
 *        The definition of *f* depends on ``pooling_convention``, which has two options:
 *
 *        - **valid** (default)::
 *
 *        f(x, k, p, s) = floor((x+2*p-k)/s)+1
 *
 *        - **full**, which is compatible with Caffe::
 *
 *        f(x, k, p, s) = ceil((x+2*p-k)/s)+1
 *
 *        When ``global_pool`` is set to be true, then global pooling is performed. It
 *        ``kernel=(height, width)`` and set the appropiate padding to 0.
 *
 *        Three pooling options are supported by ``pool_type``:
 *
 *        - **avg**: average pooling
 *        - **max**: max pooling
 *        - **sum**: sum pooling
 *        - **lp**: Lp pooling
 *
 *        For 3-D pooling, an additional *depth* dimension is added before
 *        *height*. Namely the input data and output will have shape *(batch_size,
 *        height, width)* (NCDHW layout) or *(batch_size, depth, height, width, channel)*
 *
 *        Notes on Lp pooling:
 *
 *        Lp pooling was first introduced by this paper:
 *        L-1 pooling is simply sum pooling, while L-inf pooling is simply max pooling.
 *        We can see that Lp pooling stands between those two, in practice the most
 *
 *        For each window ``X``, the mathematical expression for Lp pooling is:
 *
 *        :math:`f(X) = \sqrt[p]{\sum_{x}^{X} x^p}`
 *
 *
 *
 *        Defined in ../src/operator/nn/pooling.cc:L417
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the pooling operator.
 * \param kernel Pooling kernel size: (y, x) or (d, y, x)
 * \param pool_type Pooling type to be applied.
 * \param global_pool Ignore kernel size, do global pooling based on current input
 * \param cudnn_off Turn off cudnn pooling and use MXNet pooling operator.
 * \param pooling_convention Pooling convention to be applied.
 * \param stride Stride: for pooling (y, x) or (d, y, x). Defaults to 1 for each
 * \param pad Pad for pooling: (y, x) or (d, y, x). Defaults to no padding.
 * \param p_value Value of p for Lp pooling, can be 1 or 2, required for Lp Pooling.
 * \param count_include_pad Only used for AvgPool, specify whether to count padding
 *        elements for averagecalculation. For example, with a 5*5 kernel on a 3*3 corner
 *        of a image,the sum of the 9 valid elements will be divided by 25 if this is set
 * \param layout Set layout for input and output. Empty for
 *        default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.
 * \return new symbol
 */
inline Symbol Pooling(const std::string& symbol_name,
                      Symbol data,
                      Shape kernel = Shape(),
                      PoolingPoolType pool_type = PoolingPoolType::kMax,
                      bool global_pool = false,
                      bool cudnn_off = false,
                      PoolingPoolingConvention pooling_convention = PoolingPoolingConvention::kValid,
                      Shape stride = Shape(),
                      Shape pad = Shape(),
                      dmlc::optional<int> p_value = dmlc::optional<int>(),
                      dmlc::optional<bool> count_include_pad = dmlc::optional<bool>(),
                      PoolingLayout layout = PoolingLayout::kNone) {
  static const char *PoolingPoolTypeValues[] = {
    "avg",
    "lp",
    "max",
    "sum"
  };
  static const char *PoolingPoolingConventionValues[] = {
    "full",
    "same",
    "valid"
  };
  static const char *PoolingLayoutValues[] = {
    "None",
    "NCDHW",
    "NCHW",
    "NCW",
    "NDHWC",
    "NHWC",
    "NWC"
  };
  return Operator("Pooling")
           .SetParam("kernel", kernel)
           .SetParam("pool_type", PoolingPoolTypeValues[int(pool_type)])
           .SetParam("global_pool", global_pool)
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("pooling_convention", PoolingPoolingConventionValues[int(pooling_convention)])
           .SetParam("stride", stride)
           .SetParam("pad", pad)
           .SetParam("p_value", p_value)
           .SetParam("count_include_pad", count_include_pad)
           .SetParam("layout", PoolingLayoutValues[int(layout)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \brief DType of the output in case this can't be inferred. Defaults to the same as
 */
enum class SoftmaxDtype {
  kNone = 0,
  kFloat16 = 1,
  kFloat32 = 2,
  kFloat64 = 3
};

/*!
 * \brief Applies the softmax function.
 *
 *        The resulting array contains elements in the range (0,1) and the elements along
 *
 *        .. math::
 *        softmax(\mathbf{z/t})_j = \frac{e^{z_j/t}}{\sum_{k=1}^K e^{z_k/t}}
 *
 *        for :math:`j = 1, ..., K`
 *
 *        t is the temperature parameter in softmax function. By default, t equals 1.0
 *
 *        Example::
 *
 *        x = [[ 1.  1.  1.]
 *        [ 1.  1.  1.]]
 *
 *        softmax(x,axis=0) = [[ 0.5  0.5  0.5]
 *        [ 0.5  0.5  0.5]]
 *
 *        softmax(x,axis=1) = [[ 0.33333334,  0.33333334,  0.33333334],
 *        [ 0.33333334,  0.33333334,  0.33333334]]
 *
 *
 *
 *        Defined in ../src/operator/nn/softmax.cc:L136
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \param length The length array.
 * \param axis The axis along which to compute softmax.
 * \param temperature Temperature parameter in softmax
 * \param dtype DType of the output in case this can't be inferred. Defaults to the same
 * \param use_length Whether to use the length input as a mask over the data input.
 * \return new symbol
 */
inline Symbol softmax(const std::string& symbol_name,
                      Symbol data,
                      Symbol length,
                      int axis = -1,
                      dmlc::optional<double> temperature = dmlc::optional<double>(),
                      SoftmaxDtype dtype = SoftmaxDtype::kNone,
                      dmlc::optional<bool> use_length = dmlc::optional<bool>(0)) {
  static const char *SoftmaxDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("softmax")
           .SetParam("axis", axis)
           .SetParam("temperature", temperature)
           .SetParam("dtype", SoftmaxDtypeValues[int(dtype)])
           .SetParam("use_length", use_length)
           .SetInput("data", data)
           .SetInput("length", length)
           .CreateSymbol(symbol_name);
}

/*! \brief Specifies how to compute the softmax. If set to ``instance``, it computes
 *        softmax for each instance. If set to ``channel``, It computes cross channel
 */
enum class SoftmaxActivationMode {
  kChannel = 0,
  kInstance = 1
};

/*!
 * \brief Applies softmax activation to input. This is intended for internal layers.
 *
 *        .. note::
 *
 *        This operator has been deprecated, please use `softmax`.
 *
 *        If `mode` = ``instance``, this operator will compute a softmax for each
 *        This is the default mode.
 *
 *        If `mode` = ``channel``, this operator will compute a k-class softmax at each
 *        of each instance, where `k` = ``num_channel``. This mode can only be used when
 *        has at least 3 dimensions.
 *        This can be used for `fully convolutional network`, `image segmentation`, etc.
 *
 *        Example::
 *
 *        >>> input_array = mx.nd.array([[3., 0.5, -0.5, 2., 7.],
 *        >>>                            [2., -.4, 7.,   3., 0.2]])
 *        >>> softmax_act = mx.nd.SoftmaxActivation(input_array)
 *        >>> print softmax_act.asnumpy()
 *        [[  1.78322066e-02   1.46375655e-03   5.38485940e-04   6.56010211e-03
 *        [  6.56221947e-03   5.95310994e-04   9.73919690e-01   1.78379621e-02
 *
 *
 *
 *        Defined in ../src/operator/nn/softmax_activation.cc:L59
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \param mode Specifies how to compute the softmax. If set to ``instance``, it computes
 *        softmax for each instance. If set to ``channel``, It computes cross channel
 * \return new symbol
 */
inline Symbol SoftmaxActivation(const std::string& symbol_name,
                                Symbol data,
                                SoftmaxActivationMode mode = SoftmaxActivationMode::kInstance) {
  static const char *SoftmaxActivationModeValues[] = {
    "channel",
    "instance"
  };
  return Operator("SoftmaxActivation")
           .SetParam("mode", SoftmaxActivationModeValues[int(mode)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \brief DType of the output in case this can't be inferred. Defaults to the same as
 */
enum class SoftminDtype {
  kNone = 0,
  kFloat16 = 1,
  kFloat32 = 2,
  kFloat64 = 3
};

/*!
 * \brief Applies the softmin function.
 *
 *        The resulting array contains elements in the range (0,1) and the elements along
 *        up to 1.
 *
 *        .. math::
 *        softmin(\mathbf{z/t})_j = \frac{e^{-z_j/t}}{\sum_{k=1}^K e^{-z_k/t}}
 *
 *        for :math:`j = 1, ..., K`
 *
 *        t is the temperature parameter in softmax function. By default, t equals 1.0
 *
 *        Example::
 *
 *        x = [[ 1.  2.  3.]
 *        [ 3.  2.  1.]]
 *
 *        softmin(x,axis=0) = [[ 0.88079703,  0.5,  0.11920292],
 *        [ 0.11920292,  0.5,  0.88079703]]
 *
 *        softmin(x,axis=1) = [[ 0.66524094,  0.24472848,  0.09003057],
 *        [ 0.09003057,  0.24472848,  0.66524094]]
 *
 *
 *
 *        Defined in ../src/operator/nn/softmin.cc:L57
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \param axis The axis along which to compute softmax.
 * \param temperature Temperature parameter in softmax
 * \param dtype DType of the output in case this can't be inferred. Defaults to the same
 * \param use_length Whether to use the length input as a mask over the data input.
 * \return new symbol
 */
inline Symbol softmin(const std::string& symbol_name,
                      Symbol data,
                      int axis = -1,
                      dmlc::optional<double> temperature = dmlc::optional<double>(),
                      SoftminDtype dtype = SoftminDtype::kNone,
                      dmlc::optional<bool> use_length = dmlc::optional<bool>(0)) {
  static const char *SoftminDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("softmin")
           .SetParam("axis", axis)
           .SetParam("temperature", temperature)
           .SetParam("dtype", SoftminDtypeValues[int(dtype)])
           .SetParam("use_length", use_length)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \brief upsampling method
 */
enum class UpSamplingSampleType {
  kBilinear = 0,
  kNearest = 1
};

/*! \brief How to handle multiple input. concat means concatenate upsampled images along
 *        the channel dimension. sum means add all images together, only available for
 */
enum class UpSamplingMultiInputMode {
  kConcat = 0,
  kSum = 1
};

/*!
 * \brief Upsamples the given input data.
 *
 *        Two algorithms (``sample_type``) are available for upsampling:
 *
 *        - Nearest Neighbor
 *        - Bilinear
 *
 *        **Nearest Neighbor Upsampling**
 *
 *        Input data is expected to be NCHW.
 *
 *        Example::
 *
 *        x = [[[[1. 1. 1.]
 *        [1. 1. 1.]
 *        [1. 1. 1.]]]]
 *
 *        UpSampling(x, scale=2, sample_type='nearest') = [[[[1. 1. 1. 1. 1. 1.]
 *        [1. 1. 1. 1. 1. 1.]
 *        [1. 1. 1. 1. 1. 1.]
 *        [1. 1. 1. 1. 1. 1.]
 *        [1. 1. 1. 1. 1. 1.]
 *        [1. 1. 1. 1. 1. 1.]]]]
 *
 *        **Bilinear Upsampling**
 *
 *        Uses `deconvolution` algorithm under the hood. You need provide both input data
 *
 *        Input data is expected to be NCHW.
 *
 *        `num_filter` is expected to be same as the number of channels.
 *
 *        Example::
 *
 *        x = [[[[1. 1. 1.]
 *        [1. 1. 1.]
 *        [1. 1. 1.]]]]
 *
 *        w = [[[[1. 1. 1. 1.]
 *        [1. 1. 1. 1.]
 *        [1. 1. 1. 1.]
 *        [1. 1. 1. 1.]]]]
 *
 *        UpSampling(x, w, scale=2, sample_type='bilinear', num_filter=1) = [[[[1. 2. 2.
 *        [2. 4. 4. 4. 4. 2.]
 *        [2. 4. 4. 4. 4. 2.]
 *        [2. 4. 4. 4. 4. 2.]
 *        [2. 4. 4. 4. 4. 2.]
 *        [1. 2. 2. 2. 2. 1.]]]]
 *
 *
 *        Defined in ../src/operator/nn/upsampling.cc:L173
 * \param symbol_name name of the resulting symbol
 * \param data Array of tensors to upsample. For bilinear upsampling, there should be 2
 * \param scale Up sampling scale
 * \param sample_type upsampling method
 * \param num_args Number of inputs to be upsampled. For nearest neighbor upsampling,
 *        this can be 1-N; the size of output will be(scale*h_0,scale*w_0) and all other
 *        inputs will be upsampled to thesame size. For bilinear upsampling this must be
 * \param num_filter Input filter. Only used by bilinear sample_type.Since bilinear
 * \param multi_input_mode How to handle multiple input. concat means concatenate
 *        upsampled images along the channel dimension. sum means add all images
 * \param workspace Tmp workspace for deconvolution (MB)
 * \return new symbol
 */
inline Symbol UpSampling(const std::string& symbol_name,
                         const std::vector<Symbol>& data,
                         int scale,
                         UpSamplingSampleType sample_type,
                         int num_args,
                         int num_filter = 0,
                         UpSamplingMultiInputMode multi_input_mode = UpSamplingMultiInputMode::kConcat,
                         uint64_t workspace = 512) {
  static const char *UpSamplingSampleTypeValues[] = {
    "bilinear",
    "nearest"
  };
  static const char *UpSamplingMultiInputModeValues[] = {
    "concat",
    "sum"
  };
  return Operator("UpSampling")
           .SetParam("scale", scale)
           .SetParam("sample_type", UpSamplingSampleTypeValues[int(sample_type)])
           .SetParam("num_args", num_args)
           .SetParam("num_filter", num_filter)
           .SetParam("multi_input_mode", UpSamplingMultiInputModeValues[int(multi_input_mode)])
           .SetParam("workspace", workspace)
(data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Update function for SignSGD optimizer.
 *
 *        .. math::
 *
 *        g_t = \nabla J(W_{t-1})\\
 *        W_t = W_{t-1} - \eta_t \text{sign}(g_t)
 *
 *        It updates the weights using::
 *
 *        weight = weight - learning_rate * sign(gradient)
 *
 *        .. note::
 *        - sparse ndarray not supported for this optimizer yet.
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L63
 * \param symbol_name name of the resulting symbol
 * \param weight Weight
 * \param grad Gradient
 * \param lr Learning rate
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \return new symbol
 */
inline Symbol signsgd_update(const std::string& symbol_name,
                             Symbol weight,
                             Symbol grad,
                             mx_float lr,
                             mx_float wd = 0,
                             mx_float rescale_grad = 1,
                             mx_float clip_gradient = -1) {
  return Operator("signsgd_update")
           .SetParam("lr", lr)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief SIGN momentUM (Signum) optimizer.
 *
 *        .. math::
 *
 *        g_t = \nabla J(W_{t-1})\\
 *        m_t = \beta m_{t-1} + (1 - \beta) g_t\\
 *        W_t = W_{t-1} - \eta_t \text{sign}(m_t)
 *
 *        It updates the weights using::
 *        state = momentum * state + (1-momentum) * gradient
 *        weight = weight - learning_rate * sign(state)
 *
 *        Where the parameter ``momentum`` is the decay rate of momentum estimates at
 *
 *        .. note::
 *        - sparse ndarray not supported for this optimizer yet.
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L92
 * \param symbol_name name of the resulting symbol
 * \param weight Weight
 * \param grad Gradient
 * \param mom Momentum
 * \param lr Learning rate
 * \param momentum The decay rate of momentum estimates at each epoch.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param wd_lh The amount of weight decay that does not go into gradient/momentum
 * \return new symbol
 */
inline Symbol signum_update(const std::string& symbol_name,
                            Symbol weight,
                            Symbol grad,
                            Symbol mom,
                            mx_float lr,
                            mx_float momentum = 0,
                            mx_float wd = 0,
                            mx_float rescale_grad = 1,
                            mx_float clip_gradient = -1,
                            mx_float wd_lh = 0) {
  return Operator("signum_update")
           .SetParam("lr", lr)
           .SetParam("momentum", momentum)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("wd_lh", wd_lh)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("mom", mom)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Update function for Stochastic Gradient Descent (SDG) optimizer.
 *
 *        It updates the weights using::
 *
 *        weight = weight - learning_rate * (gradient + wd * weight)
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L329
 * \param symbol_name name of the resulting symbol
 * \param data Weights
 * \param lrs Learning rates.
 * \param wds Weight decay augments the objective function with a regularization term
 *        that penalizes large weights. The penalty scales with the square of the
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param num_weights Number of updated weights.
 * \return new symbol
 */
inline Symbol multi_sgd_update(const std::string& symbol_name,
                               const std::vector<Symbol>& data,
                               nnvm::Tuple<mx_float> lrs,
                               nnvm::Tuple<mx_float> wds,
                               mx_float rescale_grad = 1,
                               mx_float clip_gradient = -1,
                               int num_weights = 1) {
  return Operator("multi_sgd_update")
           .SetParam("lrs", lrs)
           .SetParam("wds", wds)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("num_weights", num_weights)
(data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Momentum update function for Stochastic Gradient Descent (SGD) optimizer.
 *
 *        Momentum update has better convergence rates on neural networks. Mathematically
 *        like below:
 *
 *        .. math::
 *
 *        v_1 = \alpha * \nabla J(W_0)\\
 *        v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\
 *        W_t = W_{t-1} + v_t
 *
 *        It updates the weights using::
 *
 *        v = momentum * v - learning_rate * gradient
 *        weight += v
 *
 *        Where the parameter ``momentum`` is the decay rate of momentum estimates at
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L374
 * \param symbol_name name of the resulting symbol
 * \param data Weights, gradients and momentum
 * \param lrs Learning rates.
 * \param wds Weight decay augments the objective function with a regularization term
 *        that penalizes large weights. The penalty scales with the square of the
 * \param momentum The decay rate of momentum estimates at each epoch.
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param num_weights Number of updated weights.
 * \return new symbol
 */
inline Symbol multi_sgd_mom_update(const std::string& symbol_name,
                                   const std::vector<Symbol>& data,
                                   nnvm::Tuple<mx_float> lrs,
                                   nnvm::Tuple<mx_float> wds,
                                   mx_float momentum = 0,
                                   mx_float rescale_grad = 1,
                                   mx_float clip_gradient = -1,
                                   int num_weights = 1) {
  return Operator("multi_sgd_mom_update")
           .SetParam("lrs", lrs)
           .SetParam("wds", wds)
           .SetParam("momentum", momentum)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("num_weights", num_weights)
(data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Update function for multi-precision Stochastic Gradient Descent (SDG) optimizer.
 *
 *        It updates the weights using::
 *
 *        weight = weight - learning_rate * (gradient + wd * weight)
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L417
 * \param symbol_name name of the resulting symbol
 * \param data Weights
 * \param lrs Learning rates.
 * \param wds Weight decay augments the objective function with a regularization term
 *        that penalizes large weights. The penalty scales with the square of the
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param num_weights Number of updated weights.
 * \return new symbol
 */
inline Symbol multi_mp_sgd_update(const std::string& symbol_name,
                                  const std::vector<Symbol>& data,
                                  nnvm::Tuple<mx_float> lrs,
                                  nnvm::Tuple<mx_float> wds,
                                  mx_float rescale_grad = 1,
                                  mx_float clip_gradient = -1,
                                  int num_weights = 1) {
  return Operator("multi_mp_sgd_update")
           .SetParam("lrs", lrs)
           .SetParam("wds", wds)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("num_weights", num_weights)
(data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Momentum update function for multi-precision Stochastic Gradient Descent (SGD)
 *
 *        Momentum update has better convergence rates on neural networks. Mathematically
 *        like below:
 *
 *        .. math::
 *
 *        v_1 = \alpha * \nabla J(W_0)\\
 *        v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\
 *        W_t = W_{t-1} + v_t
 *
 *        It updates the weights using::
 *
 *        v = momentum * v - learning_rate * gradient
 *        weight += v
 *
 *        Where the parameter ``momentum`` is the decay rate of momentum estimates at
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L472
 * \param symbol_name name of the resulting symbol
 * \param data Weights
 * \param lrs Learning rates.
 * \param wds Weight decay augments the objective function with a regularization term
 *        that penalizes large weights. The penalty scales with the square of the
 * \param momentum The decay rate of momentum estimates at each epoch.
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param num_weights Number of updated weights.
 * \return new symbol
 */
inline Symbol multi_mp_sgd_mom_update(const std::string& symbol_name,
                                      const std::vector<Symbol>& data,
                                      nnvm::Tuple<mx_float> lrs,
                                      nnvm::Tuple<mx_float> wds,
                                      mx_float momentum = 0,
                                      mx_float rescale_grad = 1,
                                      mx_float clip_gradient = -1,
                                      int num_weights = 1) {
  return Operator("multi_mp_sgd_mom_update")
           .SetParam("lrs", lrs)
           .SetParam("wds", wds)
           .SetParam("momentum", momentum)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("num_weights", num_weights)
(data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Update function for Stochastic Gradient Descent (SGD) optimizer.
 *
 *        It updates the weights using::
 *
 *        weight = weight - learning_rate * (gradient + wd * weight)
 *
 *        However, if gradient is of ``row_sparse`` storage type and ``lazy_update`` is
 *        only the row slices whose indices appear in grad.indices are updated::
 *
 *        for row in gradient.indices:
 *        weight[row] = weight[row] - learning_rate * (gradient[row] + wd * weight[row])
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L524
 * \param symbol_name name of the resulting symbol
 * \param weight Weight
 * \param grad Gradient
 * \param lr Learning rate
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param lazy_update If true, lazy updates are applied if gradient's stype is row_sparse.
 * \return new symbol
 */
inline Symbol sgd_update(const std::string& symbol_name,
                         Symbol weight,
                         Symbol grad,
                         mx_float lr,
                         mx_float wd = 0,
                         mx_float rescale_grad = 1,
                         mx_float clip_gradient = -1,
                         bool lazy_update = true) {
  return Operator("sgd_update")
           .SetParam("lr", lr)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("lazy_update", lazy_update)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Momentum update function for Stochastic Gradient Descent (SGD) optimizer.
 *
 *        Momentum update has better convergence rates on neural networks. Mathematically
 *        like below:
 *
 *        .. math::
 *
 *        v_1 = \alpha * \nabla J(W_0)\\
 *        v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\
 *        W_t = W_{t-1} + v_t
 *
 *        It updates the weights using::
 *
 *        v = momentum * v - learning_rate * gradient
 *        weight += v
 *
 *        Where the parameter ``momentum`` is the decay rate of momentum estimates at
 *
 *        However, if grad's storage type is ``row_sparse``, ``lazy_update`` is True and
 *        type is the same as momentum's storage type,
 *        only the row slices whose indices appear in grad.indices are updated (for both
 *
 *        for row in gradient.indices:
 *        v[row] = momentum[row] * v[row] - learning_rate * gradient[row]
 *        weight[row] += v[row]
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L565
 * \param symbol_name name of the resulting symbol
 * \param weight Weight
 * \param grad Gradient
 * \param mom Momentum
 * \param lr Learning rate
 * \param momentum The decay rate of momentum estimates at each epoch.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param lazy_update If true, lazy updates are applied if gradient's stype is row_sparse
 * \return new symbol
 */
inline Symbol sgd_mom_update(const std::string& symbol_name,
                             Symbol weight,
                             Symbol grad,
                             Symbol mom,
                             mx_float lr,
                             mx_float momentum = 0,
                             mx_float wd = 0,
                             mx_float rescale_grad = 1,
                             mx_float clip_gradient = -1,
                             bool lazy_update = true) {
  return Operator("sgd_mom_update")
           .SetParam("lr", lr)
           .SetParam("momentum", momentum)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("lazy_update", lazy_update)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("mom", mom)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Updater function for multi-precision sgd optimizer
 * \param symbol_name name of the resulting symbol
 * \param weight Weight
 * \param grad gradient
 * \param weight32 Weight32
 * \param lr Learning rate
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param lazy_update If true, lazy updates are applied if gradient's stype is row_sparse.
 * \return new symbol
 */
inline Symbol mp_sgd_update(const std::string& symbol_name,
                            Symbol weight,
                            Symbol grad,
                            Symbol weight32,
                            mx_float lr,
                            mx_float wd = 0,
                            mx_float rescale_grad = 1,
                            mx_float clip_gradient = -1,
                            bool lazy_update = true) {
  return Operator("mp_sgd_update")
           .SetParam("lr", lr)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("lazy_update", lazy_update)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("weight32", weight32)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Updater function for multi-precision sgd optimizer
 * \param symbol_name name of the resulting symbol
 * \param weight Weight
 * \param grad Gradient
 * \param mom Momentum
 * \param weight32 Weight32
 * \param lr Learning rate
 * \param momentum The decay rate of momentum estimates at each epoch.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param lazy_update If true, lazy updates are applied if gradient's stype is row_sparse
 * \return new symbol
 */
inline Symbol mp_sgd_mom_update(const std::string& symbol_name,
                                Symbol weight,
                                Symbol grad,
                                Symbol mom,
                                Symbol weight32,
                                mx_float lr,
                                mx_float momentum = 0,
                                mx_float wd = 0,
                                mx_float rescale_grad = 1,
                                mx_float clip_gradient = -1,
                                bool lazy_update = true) {
  return Operator("mp_sgd_mom_update")
           .SetParam("lr", lr)
           .SetParam("momentum", momentum)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("lazy_update", lazy_update)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("mom", mom)
           .SetInput("weight32", weight32)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief The FTML optimizer described in
 *        *FTML - Follow the Moving Leader in Deep Learning*,
 *        available at http://proceedings.mlr.press/v70/zheng17a/zheng17a.pdf.
 *
 *        .. math::
 *
 *        g_t = \nabla J(W_{t-1})\\
 *        v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\
 *        d_t = \frac{ 1 - \beta_1^t }{ \eta_t } (\sqrt{ \frac{ v_t }{ 1 - \beta_2^t } }
 *        \sigma_t = d_t - \beta_1 d_{t-1}
 *        z_t = \beta_1 z_{ t-1 } + (1 - \beta_1^t) g_t - \sigma_t W_{t-1}
 *        W_t = - \frac{ z_t }{ d_t }
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L640
 * \param symbol_name name of the resulting symbol
 * \param weight Weight
 * \param grad Gradient
 * \param d Internal state ``d_t``
 * \param v Internal state ``v_t``
 * \param z Internal state ``z_t``
 * \param lr Learning rate.
 * \param t Number of update.
 * \param beta1 Generally close to 0.5.
 * \param beta2 Generally close to 1.
 * \param epsilon Epsilon to prevent div 0.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_grad Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \return new symbol
 */
inline Symbol ftml_update(const std::string& symbol_name,
                          Symbol weight,
                          Symbol grad,
                          Symbol d,
                          Symbol v,
                          Symbol z,
                          mx_float lr,
                          int t,
                          mx_float beta1 = 0.600000024,
                          mx_float beta2 = 0.999000013,
                          double epsilon = 9.9999999392252903e-09,
                          mx_float wd = 0,
                          mx_float rescale_grad = 1,
                          mx_float clip_grad = -1) {
  return Operator("ftml_update")
           .SetParam("lr", lr)
           .SetParam("t", t)
           .SetParam("beta1", beta1)
           .SetParam("beta2", beta2)
           .SetParam("epsilon", epsilon)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_grad", clip_grad)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("d", d)
           .SetInput("v", v)
           .SetInput("z", z)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Update function for Adam optimizer. Adam is seen as a generalization
 *        of AdaGrad.
 *
 *        Adam update consists of the following steps, where g represents gradient and m,
 *        are 1st and 2nd order moment estimates (mean and variance).
 *
 *        .. math::
 *
 *        g_t = \nabla J(W_{t-1})\\
 *        m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\\
 *        v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\
 *        W_t = W_{t-1} - \alpha \frac{ m_t }{ \sqrt{ v_t } + \epsilon }
 *
 *        It updates the weights using::
 *
 *        m = beta1*m + (1-beta1)*grad
 *        v = beta2*v + (1-beta2)*(grad**2)
 *        w += - learning_rate * m / (sqrt(v) + epsilon)
 *
 *        However, if grad's storage type is ``row_sparse``, ``lazy_update`` is True and
 *        type of weight is the same as those of m and v,
 *        only the row slices whose indices appear in grad.indices are updated (for w, m
 *
 *        for row in grad.indices:
 *        m[row] = beta1*m[row] + (1-beta1)*grad[row]
 *        v[row] = beta2*v[row] + (1-beta2)*(grad[row]**2)
 *        w[row] += - learning_rate * m[row] / (sqrt(v[row]) + epsilon)
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L688
 * \param symbol_name name of the resulting symbol
 * \param weight Weight
 * \param grad Gradient
 * \param mean Moving mean
 * \param var Moving variance
 * \param lr Learning rate
 * \param beta1 The decay rate for the 1st moment estimates.
 * \param beta2 The decay rate for the 2nd moment estimates.
 * \param epsilon A small constant for numerical stability.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param lazy_update If true, lazy updates are applied if gradient's stype is row_sparse
 * \return new symbol
 */
inline Symbol adam_update(const std::string& symbol_name,
                          Symbol weight,
                          Symbol grad,
                          Symbol mean,
                          Symbol var,
                          mx_float lr,
                          mx_float beta1 = 0.899999976,
                          mx_float beta2 = 0.999000013,
                          mx_float epsilon = 9.99999994e-09,
                          mx_float wd = 0,
                          mx_float rescale_grad = 1,
                          mx_float clip_gradient = -1,
                          bool lazy_update = true) {
  return Operator("adam_update")
           .SetParam("lr", lr)
           .SetParam("beta1", beta1)
           .SetParam("beta2", beta2)
           .SetParam("epsilon", epsilon)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("lazy_update", lazy_update)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("mean", mean)
           .SetInput("var", var)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Update function for Nesterov Accelerated Gradient( NAG) optimizer.
 *        It updates the weights using the following formula,
 *
 *        .. math::
 *        v_t = \gamma v_{t-1} + \eta * \nabla J(W_{t-1} - \gamma v_{t-1})\\
 *        W_t = W_{t-1} - v_t
 *
 *        Where
 *        :math:`\eta` is the learning rate of the optimizer
 *        :math:`\gamma` is the decay rate of the momentum estimate
 *        :math:`\v_t` is the update vector at time step `t`
 *        :math:`\W_t` is the weight vector at time step `t`
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L726
 * \param symbol_name name of the resulting symbol
 * \param weight Weight
 * \param grad Gradient
 * \param mom Momentum
 * \param lr Learning rate
 * \param momentum The decay rate of momentum estimates at each epoch.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \return new symbol
 */
inline Symbol nag_mom_update(const std::string& symbol_name,
                             Symbol weight,
                             Symbol grad,
                             Symbol mom,
                             mx_float lr,
                             mx_float momentum = 0,
                             mx_float wd = 0,
                             mx_float rescale_grad = 1,
                             mx_float clip_gradient = -1) {
  return Operator("nag_mom_update")
           .SetParam("lr", lr)
           .SetParam("momentum", momentum)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("mom", mom)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Update function for multi-precision Nesterov Accelerated Gradient( NAG)
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L745
 * \param symbol_name name of the resulting symbol
 * \param weight Weight
 * \param grad Gradient
 * \param mom Momentum
 * \param weight32 Weight32
 * \param lr Learning rate
 * \param momentum The decay rate of momentum estimates at each epoch.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \return new symbol
 */
inline Symbol mp_nag_mom_update(const std::string& symbol_name,
                                Symbol weight,
                                Symbol grad,
                                Symbol mom,
                                Symbol weight32,
                                mx_float lr,
                                mx_float momentum = 0,
                                mx_float wd = 0,
                                mx_float rescale_grad = 1,
                                mx_float clip_gradient = -1) {
  return Operator("mp_nag_mom_update")
           .SetParam("lr", lr)
           .SetParam("momentum", momentum)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("mom", mom)
           .SetInput("weight32", weight32)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Update function for `RMSProp` optimizer.
 *
 *        `RMSprop` is a variant of stochastic gradient descent where the gradients are
 *        divided by a cache which grows with the sum of squares of recent gradients?
 *
 *        `RMSProp` is similar to `AdaGrad`, a popular variant of `SGD` which adaptively
 *        tunes the learning rate of each parameter. `AdaGrad` lowers the learning rate
 *        each parameter monotonically over the course of training.
 *        While this is analytically motivated for convex optimizations, it may not be
 *        for non-convex problems. `RMSProp` deals with this heuristically by allowing the
 *        learning rates to rebound as the denominator decays over time.
 *
 *        Define the Root Mean Square (RMS) error criterion of the gradient as
 *        :math:`RMS[g]_t = \sqrt{E[g^2]_t + \epsilon}`, where :math:`g` represents
 *        gradient and :math:`E[g^2]_t` is the decaying average over past squared
 *
 *        The :math:`E[g^2]_t` is given by:
 *
 *        .. math::
 *        E[g^2]_t = \gamma * E[g^2]_{t-1} + (1-\gamma) * g_t^2
 *
 *        The update step is
 *
 *        .. math::
 *        \theta_{t+1} = \theta_t - \frac{\eta}{RMS[g]_t} g_t
 *
 *        The RMSProp code follows the version in
 *        http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
 *        Tieleman & Hinton, 2012.
 *
 *        Hinton suggests the momentum term :math:`\gamma` to be 0.9 and the learning rate
 *        :math:`\eta` to be 0.001.
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L797
 * \param symbol_name name of the resulting symbol
 * \param weight Weight
 * \param grad Gradient
 * \param n n
 * \param lr Learning rate
 * \param gamma1 The decay rate of momentum estimates.
 * \param epsilon A small constant for numerical stability.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param clip_weights Clip weights to the range of [-clip_weights, clip_weights] If
 *        clip_weights <= 0, weight clipping is turned off. weights = max(min(weights,
 * \return new symbol
 */
inline Symbol rmsprop_update(const std::string& symbol_name,
                             Symbol weight,
                             Symbol grad,
                             Symbol n,
                             mx_float lr,
                             mx_float gamma1 = 0.949999988,
                             mx_float epsilon = 9.99999994e-09,
                             mx_float wd = 0,
                             mx_float rescale_grad = 1,
                             mx_float clip_gradient = -1,
                             mx_float clip_weights = -1) {
  return Operator("rmsprop_update")
           .SetParam("lr", lr)
           .SetParam("gamma1", gamma1)
           .SetParam("epsilon", epsilon)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("clip_weights", clip_weights)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("n", n)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Update function for RMSPropAlex optimizer.
 *
 *        `RMSPropAlex` is non-centered version of `RMSProp`.
 *
 *        Define :math:`E[g^2]_t` is the decaying average over past squared gradient and
 *        :math:`E[g]_t` is the decaying average over past gradient.
 *
 *        .. math::
 *        E[g^2]_t = \gamma_1 * E[g^2]_{t-1} + (1 - \gamma_1) * g_t^2\\
 *        E[g]_t = \gamma_1 * E[g]_{t-1} + (1 - \gamma_1) * g_t\\
 *        \Delta_t = \gamma_2 * \Delta_{t-1} - \frac{\eta}{\sqrt{E[g^2]_t - E[g]_t^2 +
 *
 *        The update step is
 *
 *        .. math::
 *        \theta_{t+1} = \theta_t + \Delta_t
 *
 *        The RMSPropAlex code follows the version in
 *        http://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45) by Alex Graves, 2013.
 *
 *        Graves suggests the momentum term :math:`\gamma_1` to be 0.95, :math:`\gamma_2`
 *        to be 0.9 and the learning rate :math:`\eta` to be 0.0001.
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L836
 * \param symbol_name name of the resulting symbol
 * \param weight Weight
 * \param grad Gradient
 * \param n n
 * \param g g
 * \param delta delta
 * \param lr Learning rate
 * \param gamma1 Decay rate.
 * \param gamma2 Decay rate.
 * \param epsilon A small constant for numerical stability.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param clip_weights Clip weights to the range of [-clip_weights, clip_weights] If
 *        clip_weights <= 0, weight clipping is turned off. weights = max(min(weights,
 * \return new symbol
 */
inline Symbol rmspropalex_update(const std::string& symbol_name,
                                 Symbol weight,
                                 Symbol grad,
                                 Symbol n,
                                 Symbol g,
                                 Symbol delta,
                                 mx_float lr,
                                 mx_float gamma1 = 0.949999988,
                                 mx_float gamma2 = 0.899999976,
                                 mx_float epsilon = 9.99999994e-09,
                                 mx_float wd = 0,
                                 mx_float rescale_grad = 1,
                                 mx_float clip_gradient = -1,
                                 mx_float clip_weights = -1) {
  return Operator("rmspropalex_update")
           .SetParam("lr", lr)
           .SetParam("gamma1", gamma1)
           .SetParam("gamma2", gamma2)
           .SetParam("epsilon", epsilon)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("clip_weights", clip_weights)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("n", n)
           .SetInput("g", g)
           .SetInput("delta", delta)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Update function for Ftrl optimizer.
 *        Referenced from *Ad Click Prediction: a View from the Trenches*, available at
 *        http://dl.acm.org/citation.cfm?id=2488200.
 *
 *        It updates the weights using::
 *
 *        rescaled_grad = clip(grad * rescale_grad, clip_gradient)
 *        z += rescaled_grad - (sqrt(n + rescaled_grad**2) - sqrt(n)) * weight /
 *        n += rescaled_grad**2
 *        w = (sign(z) * lamda1 - z) / ((beta + sqrt(n)) / learning_rate + wd) * (abs(z)
 *
 *        If w, z and n are all of ``row_sparse`` storage type,
 *        only the row slices whose indices appear in grad.indices are updated (for w, z
 *
 *        for row in grad.indices:
 *        rescaled_grad[row] = clip(grad[row] * rescale_grad, clip_gradient)
 *        z[row] += rescaled_grad[row] - (sqrt(n[row] + rescaled_grad[row]**2) -
 *        n[row] += rescaled_grad[row]**2
 *        w[row] = (sign(z[row]) * lamda1 - z[row]) / ((beta + sqrt(n[row])) /
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L876
 * \param symbol_name name of the resulting symbol
 * \param weight Weight
 * \param grad Gradient
 * \param z z
 * \param n Square of grad
 * \param lr Learning rate
 * \param lamda1 The L1 regularization coefficient.
 * \param beta Per-Coordinate Learning Rate beta.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \return new symbol
 */
inline Symbol ftrl_update(const std::string& symbol_name,
                          Symbol weight,
                          Symbol grad,
                          Symbol z,
                          Symbol n,
                          mx_float lr,
                          mx_float lamda1 = 0.00999999978,
                          mx_float beta = 1,
                          mx_float wd = 0,
                          mx_float rescale_grad = 1,
                          mx_float clip_gradient = -1) {
  return Operator("ftrl_update")
           .SetParam("lr", lr)
           .SetParam("lamda1", lamda1)
           .SetParam("beta", beta)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("z", z)
           .SetInput("n", n)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Phase I of lamb update it performs the following operations and returns g:.
 *
 *        Link to paper: https://arxiv.org/pdf/1904.00962.pdf
 *
 *        .. math::
 *        \begin{gather*}
 *        grad = grad * rescale_grad
 *        if (grad < -clip_gradient)
 *        then
 *        grad = -clip_gradient
 *        if (grad > clip_gradient)
 *        then
 *        grad = clip_gradient
 *
 *        mean = beta1 * mean + (1 - beta1) * grad;
 *        variance = beta2 * variance + (1. - beta2) * grad ^ 2;
 *
 *        if (bias_correction)
 *        then
 *        mean_hat = mean / (1. - beta1^t);
 *        var_hat = var / (1 - beta2^t);
 *        g = mean_hat / (var_hat^(1/2) + epsilon) + wd * weight;
 *        else
 *        g = mean / (var_data^(1/2) + epsilon) + wd * weight;
 *        \end{gather*}
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L953
 * \param symbol_name name of the resulting symbol
 * \param weight Weight
 * \param grad Gradient
 * \param mean Moving mean
 * \param var Moving variance
 * \param t Index update count.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param beta1 The decay rate for the 1st moment estimates.
 * \param beta2 The decay rate for the 2nd moment estimates.
 * \param epsilon A small constant for numerical stability.
 * \param bias_correction Whether to use bias correction.
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \return new symbol
 */
inline Symbol lamb_update_phase1(const std::string& symbol_name,
                                 Symbol weight,
                                 Symbol grad,
                                 Symbol mean,
                                 Symbol var,
                                 int t,
                                 mx_float wd,
                                 mx_float beta1 = 0.899999976,
                                 mx_float beta2 = 0.999000013,
                                 mx_float epsilon = 9.99999997e-07,
                                 bool bias_correction = true,
                                 mx_float rescale_grad = 1,
                                 mx_float clip_gradient = -1) {
  return Operator("lamb_update_phase1")
           .SetParam("t", t)
           .SetParam("wd", wd)
           .SetParam("beta1", beta1)
           .SetParam("beta2", beta2)
           .SetParam("epsilon", epsilon)
           .SetParam("bias_correction", bias_correction)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("mean", mean)
           .SetInput("var", var)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Phase II of lamb update it performs the following operations and updates grad.
 *
 *        Link to paper: https://arxiv.org/pdf/1904.00962.pdf
 *
 *        .. math::
 *        \begin{gather*}
 *        if (lower_bound >= 0)
 *        then
 *        r1 = max(r1, lower_bound)
 *        if (upper_bound >= 0)
 *        then
 *        r1 = max(r1, upper_bound)
 *
 *        if (r1 == 0 or r2 == 0)
 *        then
 *        lr = lr
 *        else
 *        lr = lr * (r1/r2)
 *        weight = weight - lr * g
 *        \end{gather*}
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L992
 * \param symbol_name name of the resulting symbol
 * \param weight Weight
 * \param g Output of lamb_update_phase 1
 * \param r1 r1
 * \param r2 r2
 * \param lr Learning rate
 * \param lower_bound Lower limit of norm of weight. If lower_bound <= 0, Lower limit is
 * \param upper_bound Upper limit of norm of weight. If upper_bound <= 0, Upper limit is
 * \return new symbol
 */
inline Symbol lamb_update_phase2(const std::string& symbol_name,
                                 Symbol weight,
                                 Symbol g,
                                 Symbol r1,
                                 Symbol r2,
                                 mx_float lr,
                                 mx_float lower_bound = -1,
                                 mx_float upper_bound = -1) {
  return Operator("lamb_update_phase2")
           .SetParam("lr", lr)
           .SetParam("lower_bound", lower_bound)
           .SetParam("upper_bound", upper_bound)
           .SetInput("weight", weight)
           .SetInput("g", g)
           .SetInput("r1", r1)
           .SetInput("r2", r2)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Mixed Precision version of Phase I of lamb update
 *        it performs the following operations and returns g:.
 *
 *        Link to paper: https://arxiv.org/pdf/1904.00962.pdf
 *
 *        .. math::
 *        \begin{gather*}
 *        grad32 = grad(float16) * rescale_grad
 *        if (grad < -clip_gradient)
 *        then
 *        grad = -clip_gradient
 *        if (grad > clip_gradient)
 *        then
 *        grad = clip_gradient
 *
 *        mean = beta1 * mean + (1 - beta1) * grad;
 *        variance = beta2 * variance + (1. - beta2) * grad ^ 2;
 *
 *        if (bias_correction)
 *        then
 *        mean_hat = mean / (1. - beta1^t);
 *        var_hat = var / (1 - beta2^t);
 *        g = mean_hat / (var_hat^(1/2) + epsilon) + wd * weight32;
 *        else
 *        g = mean / (var_data^(1/2) + epsilon) + wd * weight32;
 *        \end{gather*}
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L1033
 * \param symbol_name name of the resulting symbol
 * \param weight Weight
 * \param grad Gradient
 * \param mean Moving mean
 * \param var Moving variance
 * \param weight32 Weight32
 * \param t Index update count.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param beta1 The decay rate for the 1st moment estimates.
 * \param beta2 The decay rate for the 2nd moment estimates.
 * \param epsilon A small constant for numerical stability.
 * \param bias_correction Whether to use bias correction.
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \return new symbol
 */
inline Symbol mp_lamb_update_phase1(const std::string& symbol_name,
                                    Symbol weight,
                                    Symbol grad,
                                    Symbol mean,
                                    Symbol var,
                                    Symbol weight32,
                                    int t,
                                    mx_float wd,
                                    mx_float beta1 = 0.899999976,
                                    mx_float beta2 = 0.999000013,
                                    mx_float epsilon = 9.99999997e-07,
                                    bool bias_correction = true,
                                    mx_float rescale_grad = 1,
                                    mx_float clip_gradient = -1) {
  return Operator("mp_lamb_update_phase1")
           .SetParam("t", t)
           .SetParam("wd", wd)
           .SetParam("beta1", beta1)
           .SetParam("beta2", beta2)
           .SetParam("epsilon", epsilon)
           .SetParam("bias_correction", bias_correction)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("mean", mean)
           .SetInput("var", var)
           .SetInput("weight32", weight32)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Mixed Precision version Phase II of lamb update
 *        it performs the following operations and updates grad.
 *
 *        Link to paper: https://arxiv.org/pdf/1904.00962.pdf
 *
 *        .. math::
 *        \begin{gather*}
 *        if (lower_bound >= 0)
 *        then
 *        r1 = max(r1, lower_bound)
 *        if (upper_bound >= 0)
 *        then
 *        r1 = max(r1, upper_bound)
 *
 *        if (r1 == 0 or r2 == 0)
 *        then
 *        lr = lr
 *        else
 *        lr = lr * (r1/r2)
 *        weight32 = weight32 - lr * g
 *        weight(float16) = weight32
 *        \end{gather*}
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L1075
 * \param symbol_name name of the resulting symbol
 * \param weight Weight
 * \param g Output of mp_lamb_update_phase 1
 * \param r1 r1
 * \param r2 r2
 * \param weight32 Weight32
 * \param lr Learning rate
 * \param lower_bound Lower limit of norm of weight. If lower_bound <= 0, Lower limit is
 * \param upper_bound Upper limit of norm of weight. If upper_bound <= 0, Upper limit is
 * \return new symbol
 */
inline Symbol mp_lamb_update_phase2(const std::string& symbol_name,
                                    Symbol weight,
                                    Symbol g,
                                    Symbol r1,
                                    Symbol r2,
                                    Symbol weight32,
                                    mx_float lr,
                                    mx_float lower_bound = -1,
                                    mx_float upper_bound = -1) {
  return Operator("mp_lamb_update_phase2")
           .SetParam("lr", lr)
           .SetParam("lower_bound", lower_bound)
           .SetParam("upper_bound", upper_bound)
           .SetInput("weight", weight)
           .SetInput("g", g)
           .SetInput("r1", r1)
           .SetInput("r2", r2)
           .SetInput("weight32", weight32)
           .CreateSymbol(symbol_name);
}

/*! \brief Padding type to use. "constant" pads with `constant_value` "edge" pads using
 *        the edge values of the input array "reflect" pads by reflecting values with
 */
enum class PadMode {
  kConstant = 0,
  kEdge = 1,
  kReflect = 2
};

/*!
 * \brief Pads an input array with a constant or edge values of the array.
 *
 *        .. note:: `Pad` is deprecated. Use `pad` instead.
 *
 *        .. note:: Current implementation only supports 4D and 5D input arrays with
 *        only on axes 1, 2 and 3. Expects axes 4 and 5 in `pad_width` to be zero.
 *
 *        This operation pads an input array with either a `constant_value` or edge values
 *        along each axis of the input array. The amount of padding is specified by
 *
 *        `pad_width` is a tuple of integer padding widths for each axis of the format
 *        ``(before_1, after_1, ... , before_N, after_N)``. The `pad_width` should be of
 *        where ``N`` is the number of dimensions of the array.
 *
 *        For dimension ``N`` of the input array, ``before_N`` and ``after_N`` indicates
 *        to add before and after the elements of the array along dimension ``N``.
 *        The widths of the higher two dimensions ``before_1``, ``after_1``, ``before_2``,
 *        ``after_2`` must be 0.
 *
 *        Example::
 *
 *        x = [[[[  1.   2.   3.]
 *        [  4.   5.   6.]]
 *
 *        [[  7.   8.   9.]
 *        [ 10.  11.  12.]]]
 *
 *
 *        [[[ 11.  12.  13.]
 *        [ 14.  15.  16.]]
 *
 *        [[ 17.  18.  19.]
 *        [ 20.  21.  22.]]]]
 *
 *        pad(x,mode="edge", pad_width=(0,0,0,0,1,1,1,1)) =
 *
 *        [[[[  1.   1.   2.   3.   3.]
 *        [  1.   1.   2.   3.   3.]
 *        [  4.   4.   5.   6.   6.]
 *        [  4.   4.   5.   6.   6.]]
 *
 *        [[  7.   7.   8.   9.   9.]
 *        [  7.   7.   8.   9.   9.]
 *        [ 10.  10.  11.  12.  12.]
 *        [ 10.  10.  11.  12.  12.]]]
 *
 *
 *        [[[ 11.  11.  12.  13.  13.]
 *        [ 11.  11.  12.  13.  13.]
 *        [ 14.  14.  15.  16.  16.]
 *        [ 14.  14.  15.  16.  16.]]
 *
 *        [[ 17.  17.  18.  19.  19.]
 *        [ 17.  17.  18.  19.  19.]
 *        [ 20.  20.  21.  22.  22.]
 *        [ 20.  20.  21.  22.  22.]]]]
 *
 *        pad(x, mode="constant", constant_value=0, pad_width=(0,0,0,0,1,1,1,1)) =
 *
 *        [[[[  0.   0.   0.   0.   0.]
 *        [  0.   1.   2.   3.   0.]
 *        [  0.   4.   5.   6.   0.]
 *        [  0.   0.   0.   0.   0.]]
 *
 *        [[  0.   0.   0.   0.   0.]
 *        [  0.   7.   8.   9.   0.]
 *        [  0.  10.  11.  12.   0.]
 *        [  0.   0.   0.   0.   0.]]]
 *
 *
 *        [[[  0.   0.   0.   0.   0.]
 *        [  0.  11.  12.  13.   0.]
 *        [  0.  14.  15.  16.   0.]
 *        [  0.   0.   0.   0.   0.]]
 *
 *        [[  0.   0.   0.   0.   0.]
 *        [  0.  17.  18.  19.   0.]
 *        [  0.  20.  21.  22.   0.]
 *        [  0.   0.   0.   0.   0.]]]]
 *
 *
 *
 *
 *        Defined in ../src/operator/pad.cc:L766
 * \param symbol_name name of the resulting symbol
 * \param data An n-dimensional input array.
 * \param mode Padding type to use. "constant" pads with `constant_value` "edge" pads
 *        using the edge values of the input array "reflect" pads by reflecting values
 * \param pad_width Widths of the padding regions applied to the edges of each axis. It
 *        is a tuple of integer padding widths for each axis of the format ``(before_1,
 *        after_1, ... , before_N, after_N)``. It should be of length ``2*N`` where ``N``
 *        is the number of dimensions of the array.This is equivalent to pad_width in
 * \param constant_value The value used for padding when `mode` is "constant".
 * \return new symbol
 */
inline Symbol Pad(const std::string& symbol_name,
                  Symbol data,
                  PadMode mode,
                  Shape pad_width,
                  double constant_value = 0) {
  static const char *PadModeValues[] = {
    "constant",
    "edge",
    "reflect"
  };
  return Operator("Pad")
           .SetParam("mode", PadModeValues[int(mode)])
           .SetParam("pad_width", pad_width)
           .SetParam("constant_value", constant_value)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Adds arguments element-wise.
 *
 *        The storage type of ``elemwise_add`` output depends on storage types of inputs
 *
 *        - elemwise_add(row_sparse, row_sparse) = row_sparse
 *        - elemwise_add(csr, csr) = csr
 *        - elemwise_add(default, csr) = default
 *        - elemwise_add(csr, default) = default
 *        - elemwise_add(default, rsp) = default
 *        - elemwise_add(rsp, default) = default
 *        - otherwise, ``elemwise_add`` generates output with default storage
 *
 *
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol elemwise_add(const std::string& symbol_name,
                           Symbol lhs,
                           Symbol rhs) {
  return Operator("elemwise_add")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Multiplies arguments element-wise.
 *
 *        The storage type of ``elemwise_mul`` output depends on storage types of inputs
 *
 *        - elemwise_mul(default, default) = default
 *        - elemwise_mul(row_sparse, row_sparse) = row_sparse
 *        - elemwise_mul(default, row_sparse) = row_sparse
 *        - elemwise_mul(row_sparse, default) = row_sparse
 *        - elemwise_mul(csr, csr) = csr
 *        - otherwise, ``elemwise_mul`` generates output with default storage
 *
 *
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol elemwise_mul(const std::string& symbol_name,
                           Symbol lhs,
                           Symbol rhs) {
  return Operator("elemwise_mul")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Flattens the input array into a 2-D array by collapsing the higher dimensions.
 *        .. note:: `Flatten` is deprecated. Use `flatten` instead.
 *        For an input array with shape ``(d1, d2, ..., dk)``, `flatten` operation
 *        the input array into an output array of shape ``(d1, d2*...*dk)``.
 *        Note that the behavior of this function is different from numpy.ndarray.flatten,
 *        which behaves similar to mxnet.ndarray.reshape((-1,)).
 *        Example::
 *        x = [[
 *        [1,2,3],
 *        [4,5,6],
 *        [7,8,9]
 *        ],
 *        [    [1,2,3],
 *        [4,5,6],
 *        [7,8,9]
 *        ]],
 *        flatten(x) = [[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
 *        [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]]
 *
 *
 *        Defined in ../src/operator/tensor/matrix_op.cc:L250
 * \param symbol_name name of the resulting symbol
 * \param data Input array.
 * \return new symbol
 */
inline Symbol Flatten(const std::string& symbol_name,
                      Symbol data) {
  return Operator("Flatten")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \brief Data type of weight.
 */
enum class EmbeddingDtype {
  kBfloat16 = 0,
  kFloat16 = 1,
  kFloat32 = 2,
  kFloat64 = 3,
  kInt32 = 4,
  kInt64 = 5,
  kInt8 = 6,
  kUint8 = 7
};

/*!
 * \brief Maps integer indices to vector representations (embeddings).
 *
 *        This operator maps words to real-valued vectors in a high-dimensional space,
 *        called word embeddings. These embeddings can capture semantic and syntactic
 *        For example, it has been noted that in the learned embedding spaces, similar
 *        to be close to each other and dissimilar words far apart.
 *
 *        For an input array of shape (d1, ..., dK),
 *        the shape of an output array is (d1, ..., dK, output_dim).
 *        All the input values should be integers in the range [0, input_dim).
 *
 *        If the input_dim is ip0 and output_dim is op0, then shape of the embedding
 *        (ip0, op0).
 *
 *        When "sparse_grad" is False, if any index mentioned is too large, it is
 *        addresses the last vector in an embedding matrix.
 *        When "sparse_grad" is True, an error will be raised if invalid indices are
 *
 *        Examples::
 *
 *        input_dim = 4
 *        output_dim = 5
 *
 *        // Each row in weight matrix y represents a word. So, y = (w0,w1,w2,w3)
 *        y = [[  0.,   1.,   2.,   3.,   4.],
 *        [  5.,   6.,   7.,   8.,   9.],
 *        [ 10.,  11.,  12.,  13.,  14.],
 *        [ 15.,  16.,  17.,  18.,  19.]]
 *
 *        // Input array x represents n-grams(2-gram). So, x = [(w1,w3), (w0,w2)]
 *        x = [[ 1.,  3.],
 *        [ 0.,  2.]]
 *
 *        // Mapped input x to its vector representation y.
 *        Embedding(x, y, 4, 5) = [[[  5.,   6.,   7.,   8.,   9.],
 *        [ 15.,  16.,  17.,  18.,  19.]],
 *
 *        [[  0.,   1.,   2.,   3.,   4.],
 *        [ 10.,  11.,  12.,  13.,  14.]]]
 *
 *
 *        The storage type of weight can be either row_sparse or default.
 *
 *        .. Note::
 *
 *        If "sparse_grad" is set to True, the storage type of gradient w.r.t weights
 *        "row_sparse". Only a subset of optimizers support sparse gradients, including
 *        and Adam. Note that by default lazy updates is turned on, which may perform
 *        from standard updates. For more details, please check the Optimization API at:
 *        https://mxnet.incubator.apache.org/api/python/optimization/optimization.html
 *
 *
 *
 *        Defined in ../src/operator/tensor/indexing_op.cc:L598
 * \param symbol_name name of the resulting symbol
 * \param data The input array to the embedding operator.
 * \param weight The embedding weight matrix.
 * \param input_dim Vocabulary size of the input indices.
 * \param output_dim Dimension of the embedding vectors.
 * \param dtype Data type of weight.
 * \param sparse_grad Compute row sparse gradient in the backward calculation. If set to
 * \return new symbol
 */
inline Symbol Embedding(const std::string& symbol_name,
                        Symbol data,
                        Symbol weight,
                        int input_dim,
                        int output_dim,
                        EmbeddingDtype dtype = EmbeddingDtype::kFloat32,
                        bool sparse_grad = false) {
  static const char *EmbeddingDtypeValues[] = {
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "int32",
    "int64",
    "int8",
    "uint8"
  };
  return Operator("Embedding")
           .SetParam("input_dim", input_dim)
           .SetParam("output_dim", output_dim)
           .SetParam("dtype", EmbeddingDtypeValues[int(dtype)])
           .SetParam("sparse_grad", sparse_grad)
           .SetInput("data", data)
           .SetInput("weight", weight)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Computes and optimizes for squared loss during backward propagation.
 *        Just outputs ``data`` during forward propagation.
 *
 *        If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i`
 *        then the squared loss estimated over :math:`n` samples is defined as
 *
 *        :math:`\text{SquaredLoss}(\textbf{Y}, \hat{\textbf{Y}} ) = \frac{1}{n}
 *
 *        .. note::
 *        Use the LinearRegressionOutput as the final output layer of a net.
 *
 *        The storage type of ``label`` can be ``default`` or ``csr``
 *
 *        - LinearRegressionOutput(default, default) = default
 *        - LinearRegressionOutput(default, csr) = default
 *
 *        By default, gradients of this loss function are scaled by factor `1/m`, where m
 *        The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.
 *
 *
 *
 *        Defined in ../src/operator/regression_output.cc:L92
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the function.
 * \param label Input label to the function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol LinearRegressionOutput(const std::string& symbol_name,
                                     Symbol data,
                                     Symbol label,
                                     mx_float grad_scale = 1) {
  return Operator("LinearRegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Computes mean absolute error of the input.
 *
 *        MAE is a risk metric corresponding to the expected value of the absolute error.
 *
 *        If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i`
 *        then the mean absolute error (MAE) estimated over :math:`n` samples is defined
 *
 *        :math:`\text{MAE}(\textbf{Y}, \hat{\textbf{Y}} ) = \frac{1}{n} \sum_{i=0}^{n-1}
 *
 *        .. note::
 *        Use the MAERegressionOutput as the final output layer of a net.
 *
 *        The storage type of ``label`` can be ``default`` or ``csr``
 *
 *        - MAERegressionOutput(default, default) = default
 *        - MAERegressionOutput(default, csr) = default
 *
 *        By default, gradients of this loss function are scaled by factor `1/m`, where m
 *        The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.
 *
 *
 *
 *        Defined in ../src/operator/regression_output.cc:L120
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the function.
 * \param label Input label to the function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol MAERegressionOutput(const std::string& symbol_name,
                                  Symbol data,
                                  Symbol label,
                                  mx_float grad_scale = 1) {
  return Operator("MAERegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Applies a logistic function to the input.
 *
 *        The logistic function, also known as the sigmoid function, is computed as
 *        :math:`\frac{1}{1+exp(-\textbf{x})}`.
 *
 *        Commonly, the sigmoid is used to squash the real-valued output of a linear model
 *        :math:`wTx+b` into the [0,1] range so that it can be interpreted as a
 *        It is suitable for binary classification or probability prediction tasks.
 *
 *        .. note::
 *        Use the LogisticRegressionOutput as the final output layer of a net.
 *
 *        The storage type of ``label`` can be ``default`` or ``csr``
 *
 *        - LogisticRegressionOutput(default, default) = default
 *        - LogisticRegressionOutput(default, csr) = default
 *
 *        The loss function used is the Binary Cross Entropy Loss:
 *
 *        :math:`-{(y\log(p) + (1 - y)\log(1 - p))}`
 *
 *        Where `y` is the ground truth probability of positive outcome for a given
 *        example, and `p` the probability predicted by the model. By default, gradients
 *        of this loss function are scaled by factor `1/m`, where m is the number of
 *        The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.
 *
 *
 *
 *        Defined in ../src/operator/regression_output.cc:L152
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the function.
 * \param label Input label to the function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol LogisticRegressionOutput(const std::string& symbol_name,
                                       Symbol data,
                                       Symbol label,
                                       mx_float grad_scale = 1) {
  return Operator("LogisticRegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*! \brief the type of RNN to compute
 */
enum class RNNMode {
  kGru = 0,
  kLstm = 1,
  kRnn_relu = 2,
  kRnn_tanh = 3
};

/*!
 * \brief Applies recurrent layers to input data. Currently, vanilla RNN, LSTM and GRU are
 *        implemented, with both multi-layer and bidirectional support.
 *
 *        When the input data is of type float32 and the environment variables
 *        and MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION are set to 1, this operator will
 *        pseudo-float16 precision (float32 math with float16 I/O) precision in order to
 *        Tensor Cores on suitable NVIDIA GPUs. This can sometimes give significant
 *
 *        **Vanilla RNN**
 *
 *        Applies a single-gate recurrent layer to input X. Two kinds of activation
 *        ReLU and Tanh.
 *
 *        With ReLU activation function:
 *
 *        .. math::
 *        h_t = relu(W_{ih} * x_t + b_{ih}  +  W_{hh} * h_{(t-1)} + b_{hh})
 *
 *        With Tanh activtion function:
 *
 *        .. math::
 *        h_t = \tanh(W_{ih} * x_t + b_{ih}  +  W_{hh} * h_{(t-1)} + b_{hh})
 *
 *        Reference paper: Finding structure in time - Elman, 1988.
 *        https://crl.ucsd.edu/~elman/Papers/fsit.pdf
 *
 *        **LSTM**
 *
 *        Long Short-Term Memory - Hochreiter, 1997.
 *
 *        .. math::
 *        \begin{array}{ll}
 *        i_t = \mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
 *        f_t = \mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
 *        g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hc} h_{(t-1)} + b_{hg}) \\
 *        o_t = \mathrm{sigmoid}(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
 *        c_t = f_t * c_{(t-1)} + i_t * g_t \\
 *        h_t = o_t * \tanh(c_t)
 *        \end{array}
 *
 *        With the projection size being set, LSTM could use the projection feature to
 *        size and give some speedups without significant damage to the accuracy.
 *
 *        Long Short-Term Memory Based Recurrent Neural Network Architectures for Large
 *        Recognition - Sak et al. 2014. https://arxiv.org/abs/1402.1128
 *
 *        .. math::
 *        \begin{array}{ll}
 *        i_t = \mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{ri} r_{(t-1)} + b_{ri}) \\
 *        f_t = \mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{rf} r_{(t-1)} + b_{rf}) \\
 *        g_t = \tanh(W_{ig} x_t + b_{ig} + W_{rc} r_{(t-1)} + b_{rg}) \\
 *        o_t = \mathrm{sigmoid}(W_{io} x_t + b_{o} + W_{ro} r_{(t-1)} + b_{ro}) \\
 *        c_t = f_t * c_{(t-1)} + i_t * g_t \\
 *        h_t = o_t * \tanh(c_t)
 *        r_t = W_{hr} h_t
 *        \end{array}
 *
 *        **GRU**
 *
 *        Gated Recurrent Unit - Cho et al. 2014. http://arxiv.org/abs/1406.1078
 *
 *        The definition of GRU here is slightly different from paper but compatible with
 *
 *        .. math::
 *        \begin{array}{ll}
 *        r_t = \mathrm{sigmoid}(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
 *        z_t = \mathrm{sigmoid}(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
 *        n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
 *        h_t = (1 - z_t) * n_t + z_t * h_{(t-1)} \\
 *        \end{array}
 *
 *
 *        Defined in ../src/operator/rnn.cc:L369
 * \param symbol_name name of the resulting symbol
 * \param data Input data to RNN
 * \param parameters Vector of all RNN trainable parameters concatenated
 * \param state initial hidden state of the RNN
 * \param state_cell initial cell state for LSTM networks (only for LSTM)
 * \param sequence_length Vector of valid sequence lengths for each element in batch.
 * \param state_size size of the state for each layer
 * \param num_layers number of stacked layers
 * \param mode the type of RNN to compute
 * \param bidirectional whether to use bidirectional recurrent layers
 * \param p drop rate of the dropout on the outputs of each RNN layer, except the last
 * \param state_outputs Whether to have the states as symbol outputs.
 * \param projection_size size of project size
 * \param lstm_state_clip_min Minimum clip value of LSTM states. This option must be used
 * \param lstm_state_clip_max Maximum clip value of LSTM states. This option must be used
 * \param lstm_state_clip_nan Whether to stop NaN from propagating in state by clipping
 * \param use_sequence_length If set to true, this layer takes in an extra input
 * \return new symbol
 */
inline Symbol RNN(const std::string& symbol_name,
                  Symbol data,
                  Symbol parameters,
                  Symbol state,
                  Symbol state_cell,
                  Symbol sequence_length,
                  uint32_t state_size,
                  uint32_t num_layers,
                  RNNMode mode,
                  bool bidirectional = false,
                  mx_float p = 0,
                  bool state_outputs = false,
                  dmlc::optional<int> projection_size = dmlc::optional<int>(),
                  dmlc::optional<double> lstm_state_clip_min = dmlc::optional<double>(),
                  dmlc::optional<double> lstm_state_clip_max = dmlc::optional<double>(),
                  bool lstm_state_clip_nan = false,
                  bool use_sequence_length = false) {
  static const char *RNNModeValues[] = {
    "gru",
    "lstm",
    "rnn_relu",
    "rnn_tanh"
  };
  return Operator("RNN")
           .SetParam("state_size", state_size)
           .SetParam("num_layers", num_layers)
           .SetParam("mode", RNNModeValues[int(mode)])
           .SetParam("bidirectional", bidirectional)
           .SetParam("p", p)
           .SetParam("state_outputs", state_outputs)
           .SetParam("projection_size", projection_size)
           .SetParam("lstm_state_clip_min", lstm_state_clip_min)
           .SetParam("lstm_state_clip_max", lstm_state_clip_max)
           .SetParam("lstm_state_clip_nan", lstm_state_clip_nan)
           .SetParam("use_sequence_length", use_sequence_length)
           .SetInput("data", data)
           .SetInput("parameters", parameters)
           .SetInput("state", state)
           .SetInput("state_cell", state_cell)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Performs region of interest(ROI) pooling on the input array.
 *
 *        ROI pooling is a variant of a max pooling layer, in which the output size is
 *        region of interest is a parameter. Its purpose is to perform max pooling on the
 *        of non-uniform sizes to obtain fixed-size feature maps. ROI pooling is a
 *        layer mostly used in training a `Fast R-CNN` network for object detection.
 *
 *        This operator takes a 4D feature map as an input array and region proposals as
 *        then it pools over sub-regions of input and produces a fixed-sized output array
 *        regardless of the ROI size.
 *
 *        To crop the feature map accordingly, you can resize the bounding box coordinates
 *        by changing the parameters `rois` and `spatial_scale`.
 *
 *        The cropped feature maps are pooled by standard max pooling operation to a
 *        indicated by a `pooled_size` parameter. batch_size will change to the number of
 *        bounding boxes after `ROIPooling`.
 *
 *        The size of each region of interest doesn't have to be perfectly divisible by
 *        the number of pooling sections(`pooled_size`).
 *
 *        Example::
 *
 *        x = [[[[  0.,   1.,   2.,   3.,   4.,   5.],
 *        [  6.,   7.,   8.,   9.,  10.,  11.],
 *        [ 12.,  13.,  14.,  15.,  16.,  17.],
 *        [ 18.,  19.,  20.,  21.,  22.,  23.],
 *        [ 24.,  25.,  26.,  27.,  28.,  29.],
 *        [ 30.,  31.,  32.,  33.,  34.,  35.],
 *        [ 36.,  37.,  38.,  39.,  40.,  41.],
 *        [ 42.,  43.,  44.,  45.,  46.,  47.]]]]
 *
 *        // region of interest i.e. bounding box coordinates.
 *        y = [[0,0,0,4,4]]
 *
 *        // returns array of shape (2,2) according to the given roi with max pooling.
 *        ROIPooling(x, y, (2,2), 1.0) = [[[[ 14.,  16.],
 *        [ 26.,  28.]]]]
 *
 *        // region of interest is changed due to the change in `spacial_scale` parameter.
 *        ROIPooling(x, y, (2,2), 0.7) = [[[[  7.,   9.],
 *        [ 19.,  21.]]]]
 *
 *
 *
 *        Defined in ../src/operator/roi_pooling.cc:L225
 * \param symbol_name name of the resulting symbol
 * \param data The input array to the pooling operator,  a 4D Feature maps
 * \param rois Bounding box coordinates, a 2D array of [[batch_index, x1, y1, x2, y2]],
 *        where (x1, y1) and (x2, y2) are top left and bottom right corners of designated
 *        region of interest. `batch_index` indicates the index of corresponding image in
 * \param pooled_size ROI pooling output shape (h,w)
 * \param spatial_scale Ratio of input feature map height (or w) to raw image height (or
 * \return new symbol
 */
inline Symbol ROIPooling(const std::string& symbol_name,
                         Symbol data,
                         Symbol rois,
                         Shape pooled_size,
                         mx_float spatial_scale) {
  return Operator("ROIPooling")
           .SetParam("pooled_size", pooled_size)
           .SetParam("spatial_scale", spatial_scale)
           .SetInput("data", data)
           .SetInput("rois", rois)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Sets all elements outside the sequence to a constant value.
 *
 *        This function takes an n-dimensional input array of the form
 *        [max_sequence_length, batch_size, other_feature_dims] and returns an array of
 *
 *        Parameter `sequence_length` is used to handle variable-length sequences.
 *        should be an input array of positive ints of dimension [batch_size].
 *        To use this parameter, set `use_sequence_length` to `True`,
 *        otherwise each example in the batch is assumed to have the max sequence length
 *        this operator works as the `identity` operator.
 *
 *        Example::
 *
 *        x = [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]]]
 *
 *        // Batch 1
 *        B1 = [[  1.,   2.,   3.],
 *        [  7.,   8.,   9.],
 *        [ 13.,  14.,  15.]]
 *
 *        // Batch 2
 *        B2 = [[  4.,   5.,   6.],
 *        [ 10.,  11.,  12.],
 *        [ 16.,  17.,  18.]]
 *
 *        // works as identity operator when sequence_length parameter is not used
 *        SequenceMask(x) = [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]]]
 *
 *        // sequence_length [1,1] means 1 of each batch will be kept
 *        // and other rows are masked with default mask value = 0
 *        SequenceMask(x, sequence_length=[1,1], use_sequence_length=True) =
 *        [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  0.,   0.,   0.],
 *        [  0.,   0.,   0.]],
 *
 *        [[  0.,   0.,   0.],
 *        [  0.,   0.,   0.]]]
 *
 *        // sequence_length [2,3] means 2 of batch B1 and 3 of batch B2 will be kept
 *        // and other rows are masked with value = 1
 *        SequenceMask(x, sequence_length=[2,3], use_sequence_length=True, value=1) =
 *        [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  7.,   8.,   9.],
 *        [  10.,  11.,  12.]],
 *
 *        [[   1.,   1.,   1.],
 *        [  16.,  17.,  18.]]]
 *
 *
 *
 *        Defined in ../src/operator/sequence_mask.cc:L186
 * \param symbol_name name of the resulting symbol
 * \param data n-dimensional input array of the form [max_sequence_length, batch_size,
 * \param sequence_length vector of sequence lengths of the form [batch_size]
 * \param use_sequence_length If set to true, this layer takes in an extra input
 * \param value The value to be used as a mask.
 * \param axis The sequence axis. Only values of 0 and 1 are currently supported.
 * \return new symbol
 */
inline Symbol SequenceMask(const std::string& symbol_name,
                           Symbol data,
                           Symbol sequence_length,
                           bool use_sequence_length = false,
                           mx_float value = 0,
                           int axis = 0) {
  return Operator("SequenceMask")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetParam("value", value)
           .SetParam("axis", axis)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Splits an array along a particular axis into multiple sub-arrays.
 *
 *        .. note:: ``SliceChannel`` is deprecated. Use ``split`` instead.
 *
 *        **Note** that `num_outputs` should evenly divide the length of the axis
 *        along which to split the array.
 *
 *        Example::
 *
 *        x  = [[[ 1.]
 *        [ 2.]]
 *        [[ 3.]
 *        [ 4.]]
 *        [[ 5.]
 *        [ 6.]]]
 *        x.shape = (3, 2, 1)
 *
 *        y = split(x, axis=1, num_outputs=2) // a list of 2 arrays with shape (3, 1, 1)
 *        y = [[[ 1.]]
 *        [[ 3.]]
 *        [[ 5.]]]
 *
 *        [[[ 2.]]
 *        [[ 4.]]
 *        [[ 6.]]]
 *
 *        y[0].shape = (3, 1, 1)
 *
 *        z = split(x, axis=0, num_outputs=3) // a list of 3 arrays with shape (1, 2, 1)
 *        z = [[[ 1.]
 *        [ 2.]]]
 *
 *        [[[ 3.]
 *        [ 4.]]]
 *
 *        [[[ 5.]
 *        [ 6.]]]
 *
 *        z[0].shape = (1, 2, 1)
 *
 *        `squeeze_axis=1` removes the axis with length 1 from the shapes of the output
 *        **Note** that setting `squeeze_axis` to ``1`` removes axis with length 1 only
 *        along the `axis` which it is split.
 *        Also `squeeze_axis` can be set to true only if ``input.shape[axis] ==
 *
 *        Example::
 *
 *        z = split(x, axis=0, num_outputs=3, squeeze_axis=1) // a list of 3 arrays with
 *        z = [[ 1.]
 *        [ 2.]]
 *
 *        [[ 3.]
 *        [ 4.]]
 *
 *        [[ 5.]
 *        [ 6.]]
 *        z[0].shape = (2 ,1 )
 *
 *
 *
 *        Defined in ../src/operator/slice_channel.cc:L107
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param num_outputs Number of splits. Note that this should evenly divide the length of
 * \param axis Axis along which to split.
 * \param squeeze_axis If true, Removes the axis with length 1 from the shapes of the
 *        output arrays. **Note** that setting `squeeze_axis` to ``true`` removes axis
 *        with length 1 only along the `axis` which it is split. Also `squeeze_axis` can
 * \return new symbol
 */
inline Symbol SliceChannel(const std::string& symbol_name,
                           Symbol data,
                           int num_outputs,
                           int axis = 1,
                           bool squeeze_axis = false) {
  return Operator("SliceChannel")
           .SetParam("num_outputs", num_outputs)
           .SetParam("axis", axis)
           .SetParam("squeeze_axis", squeeze_axis)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \brief Normalizes the gradient.
 */
enum class SoftmaxOutputNormalization {
  kBatch = 0,
  kNull = 1,
  kValid = 2
};

/*!
 * \brief Computes the gradient of cross entropy loss with respect to softmax output.
 *
 *        - This operator computes the gradient in two steps.
 *        The cross entropy loss does not actually need to be computed.
 *
 *        - Applies softmax function on the input array.
 *        - Computes and returns the gradient of cross entropy loss w.r.t. the softmax
 *
 *        - The softmax function, cross entropy loss and gradient is given by:
 *
 *        - Softmax Function:
 *
 *        .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}
 *
 *        - Cross Entropy Function:
 *
 *        .. math:: \text{CE(label, output)} = - \sum_i \text{label}_i
 *
 *        - The gradient of cross entropy loss w.r.t softmax output:
 *
 *        .. math:: \text{gradient} = \text{output} - \text{label}
 *
 *        - During forward propagation, the softmax function is computed for each
 *
 *        For general *N*-D input arrays with shape :math:`(d_1, d_2, ..., d_n)`. The
 *        :math:`s=d_1 \cdot d_2 \cdot \cdot \cdot d_n`. We can use the parameters
 *        and `multi_output` to specify the way to compute softmax:
 *
 *        - By default, `preserve_shape` is ``false``. This operator will reshape the
 *        into a 2-D array with shape :math:`(d_1, \frac{s}{d_1})` and then compute the
 *        each row in the reshaped array, and afterwards reshape it back to the original
 *        :math:`(d_1, d_2, ..., d_n)`.
 *        - If `preserve_shape` is ``true``, the softmax function will be computed along
 *        the last axis (`axis` = ``-1``).
 *        - If `multi_output` is ``true``, the softmax function will be computed along
 *        the second axis (`axis` = ``1``).
 *
 *        - During backward propagation, the gradient of cross-entropy loss w.r.t softmax
 *        The provided label can be a one-hot label array or a probability label array.
 *
 *        - If the parameter `use_ignore` is ``true``, `ignore_label` can specify input
 *        with a particular label to be ignored during backward propagation. **This has
 *        softmax `output` has same shape as `label`**.
 *
 *        Example::
 *
 *        data = [[1,2,3,4],[2,2,2,2],[3,3,3,3],[4,4,4,4]]
 *        label = [1,0,2,3]
 *        ignore_label = 1
 *        SoftmaxOutput(data=data, label = label,\
 *        multi_output=true, use_ignore=true,\
 *        ignore_label=ignore_label)
 *        ## forward softmax output
 *        [[ 0.0320586   0.08714432  0.23688284  0.64391428]
 *        [ 0.25        0.25        0.25        0.25      ]
 *        [ 0.25        0.25        0.25        0.25      ]
 *        [ 0.25        0.25        0.25        0.25      ]]
 *        ## backward gradient output
 *        [[ 0.    0.    0.    0.  ]
 *        [-0.75  0.25  0.25  0.25]
 *        [ 0.25  0.25 -0.75  0.25]
 *        [ 0.25  0.25  0.25 -0.75]]
 *        ## notice that the first row is all 0 because label[0] is 1, which is equal to
 *
 *        - The parameter `grad_scale` can be used to rescale the gradient, which is
 *        give each loss function different weights.
 *
 *        - This operator also supports various ways to normalize the gradient by
 *        The `normalization` is applied if softmax output has different shape than the
 *        The `normalization` mode can be set to the followings:
 *
 *        - ``'null'``: do nothing.
 *        - ``'batch'``: divide the gradient by the batch size.
 *        - ``'valid'``: divide the gradient by the number of instances which are not
 *
 *
 *
 *        Defined in ../src/operator/softmax_output.cc:L243
 * \param symbol_name name of the resulting symbol
 * \param data Input array.
 * \param label Ground truth label.
 * \param grad_scale Scales the gradient by a float factor.
 * \param ignore_label The instances whose `labels` == `ignore_label` will be ignored
 * \param multi_output If set to ``true``, the softmax function will be computed along
 *        axis ``1``. This is applied when the shape of input array differs from the
 * \param use_ignore If set to ``true``, the `ignore_label` value will not contribute to
 * \param preserve_shape If set to ``true``, the softmax function will be computed along
 * \param normalization Normalizes the gradient.
 * \param out_grad Multiplies gradient with output gradient element-wise.
 * \param smooth_alpha Constant for computing a label smoothed version of
 *        cross-entropyfor the backwards pass.  This constant gets subtracted from
 *        theone-hot encoding of the gold label and distributed uniformly toall other
 * \return new symbol
 */
inline Symbol SoftmaxOutput(const std::string& symbol_name,
                            Symbol data,
                            Symbol label,
                            mx_float grad_scale = 1,
                            mx_float ignore_label = -1,
                            bool multi_output = false,
                            bool use_ignore = false,
                            bool preserve_shape = false,
                            SoftmaxOutputNormalization normalization = SoftmaxOutputNormalization::kNull,
                            bool out_grad = false,
                            mx_float smooth_alpha = 0) {
  static const char *SoftmaxOutputNormalizationValues[] = {
    "batch",
    "null",
    "valid"
  };
  return Operator("SoftmaxOutput")
           .SetParam("grad_scale", grad_scale)
           .SetParam("ignore_label", ignore_label)
           .SetParam("multi_output", multi_output)
           .SetParam("use_ignore", use_ignore)
           .SetParam("preserve_shape", preserve_shape)
           .SetParam("normalization", SoftmaxOutputNormalizationValues[int(normalization)])
           .SetParam("out_grad", out_grad)
           .SetParam("smooth_alpha", smooth_alpha)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Interchanges two axes of an array.
 *
 *        Examples::
 *
 *        x = [[1, 2, 3]])
 *        swapaxes(x, 0, 1) = [[ 1],
 *        [ 2],
 *        [ 3]]
 *
 *        x = [[[ 0, 1],
 *        [ 2, 3]],
 *        [[ 4, 5],
 *        [ 6, 7]]]  // (2,2,2) array
 *
 *        swapaxes(x, 0, 2) = [[[ 0, 4],
 *        [ 2, 6]],
 *        [[ 1, 5],
 *        [ 3, 7]]]
 *
 *
 *        Defined in ../src/operator/swapaxis.cc:L70
 * \param symbol_name name of the resulting symbol
 * \param data Input array.
 * \param dim1 the first axis to be swapped.
 * \param dim2 the second axis to be swapped.
 * \return new symbol
 */
inline Symbol SwapAxis(const std::string& symbol_name,
                       Symbol data,
                       int dim1 = 0,
                       int dim2 = 0) {
  return Operator("SwapAxis")
           .SetParam("dim1", dim1)
           .SetParam("dim2", dim2)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \brief Output data type.
 */
enum class Amp_castDtype {
  kBfloat16 = 0,
  kFloat16 = 1,
  kFloat32 = 2,
  kFloat64 = 3,
  kInt32 = 4,
  kInt64 = 5,
  kInt8 = 6,
  kUint8 = 7
};

/*!
 * \brief Cast function between low precision float/FP32 used by AMP.
 *
 *        It casts only between low precision float/FP32 and does not do anything for
 *
 *
 *        Defined in ../src/operator/tensor/amp_cast.cc:L121
 * \param symbol_name name of the resulting symbol
 * \param data The input.
 * \param dtype Output data type.
 * \return new symbol
 */
inline Symbol amp_cast(const std::string& symbol_name,
                       Symbol data,
                       Amp_castDtype dtype) {
  static const char *Amp_castDtypeValues[] = {
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "int32",
    "int64",
    "int8",
    "uint8"
  };
  return Operator("amp_cast")
           .SetParam("dtype", Amp_castDtypeValues[int(dtype)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Cast function used by AMP, that casts its inputs to the common widest type.
 *
 *        It casts only between low precision float/FP32 and does not do anything for
 *
 *
 *
 *        Defined in ../src/operator/tensor/amp_cast.cc:L165
 * \param symbol_name name of the resulting symbol
 * \param data Weights
 * \param num_outputs Number of input/output pairs to be casted to the widest type.
 * \param cast_narrow Whether to cast to the narrowest type
 * \return new symbol
 */
inline Symbol amp_multicast(const std::string& symbol_name,
                            const std::vector<Symbol>& data,
                            int num_outputs,
                            bool cast_narrow = false) {
  return Operator("amp_multicast")
           .SetParam("num_outputs", num_outputs)
           .SetParam("cast_narrow", cast_narrow)
(data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Computes the max of array elements over given axes.
 *
 *        Defined in ../src/operator/tensor/./broadcast_reduce_op.h:L32
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis The axis or axes along which to perform the reduction.
 *
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *
 *        If `axis` is int, a reduction is performed on a particular axis.
 *
 *        If `axis` is a tuple of ints, a reduction is performed on all the axes
 *        specified in the tuple.
 *
 *        If `exclude` is true, reduction will be performed on the axes that are
 *        NOT in axis instead.
 *
 *        Negative values means indexing from right to left.
 * \param keepdims If this is set to `True`, the reduced axes are left in the result as
 * \param exclude Whether to perform reduction on axis that are NOT in axis instead.
 * \return new symbol
 */
inline Symbol max(const std::string& symbol_name,
                  Symbol data,
                  dmlc::optional<Shape> axis = dmlc::optional<Shape>(),
                  bool keepdims = false,
                  bool exclude = false) {
  return Operator("max")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("exclude", exclude)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Computes the min of array elements over given axes.
 *
 *        Defined in ../src/operator/tensor/./broadcast_reduce_op.h:L47
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis The axis or axes along which to perform the reduction.
 *
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *
 *        If `axis` is int, a reduction is performed on a particular axis.
 *
 *        If `axis` is a tuple of ints, a reduction is performed on all the axes
 *        specified in the tuple.
 *
 *        If `exclude` is true, reduction will be performed on the axes that are
 *        NOT in axis instead.
 *
 *        Negative values means indexing from right to left.
 * \param keepdims If this is set to `True`, the reduced axes are left in the result as
 * \param exclude Whether to perform reduction on axis that are NOT in axis instead.
 * \return new symbol
 */
inline Symbol min(const std::string& symbol_name,
                  Symbol data,
                  dmlc::optional<Shape> axis = dmlc::optional<Shape>(),
                  bool keepdims = false,
                  bool exclude = false) {
  return Operator("min")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("exclude", exclude)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \brief The data type of the output.
 */
enum class NormOutDtype {
  kNone = 0,
  kFloat16 = 1,
  kFloat32 = 2,
  kFloat64 = 3,
  kInt32 = 4,
  kInt64 = 5,
  kInt8 = 6
};

/*!
 * \brief Computes the norm on an NDArray.
 *
 *        This operator computes the norm on an NDArray with the specified axis, depending
 *        on the value of the ord parameter. By default, it computes the L2 norm on the
 *        array. Currently only ord=2 supports sparse ndarrays.
 *
 *        Examples::
 *
 *        x = [[[1, 2],
 *        [3, 4]],
 *        [[2, 2],
 *        [5, 6]]]
 *
 *        norm(x, ord=2, axis=1) = [[3.1622777 4.472136 ]
 *        [5.3851647 6.3245554]]
 *
 *        norm(x, ord=1, axis=1) = [[4., 6.],
 *        [7., 8.]]
 *
 *        rsp = x.cast_storage('row_sparse')
 *
 *        norm(rsp) = [5.47722578]
 *
 *        csr = x.cast_storage('csr')
 *
 *        norm(csr) = [5.47722578]
 *
 *
 *
 *        Defined in ../src/operator/tensor/broadcast_reduce_norm_value.cc:L89
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param ord Order of the norm. Currently ord=1 and ord=2 is supported.
 * \param axis The axis or axes along which to perform the reduction.
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *        If `axis` is int, a reduction is performed on a particular axis.
 *        If `axis` is a 2-tuple, it specifies the axes that hold 2-D matrices,
 *        and the matrix norms of these matrices are computed.
 * \param out_dtype The data type of the output.
 * \param keepdims If this is set to `True`, the reduced axis is left in the result as
 * \return new symbol
 */
inline Symbol norm(const std::string& symbol_name,
                   Symbol data,
                   int ord = 2,
                   dmlc::optional<Shape> axis = dmlc::optional<Shape>(),
                   NormOutDtype out_dtype = NormOutDtype::kNone,
                   bool keepdims = false) {
  static const char *NormOutDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64",
    "int32",
    "int64",
    "int8"
  };
  return Operator("norm")
           .SetParam("ord", ord)
           .SetParam("axis", axis)
           .SetParam("out_dtype", NormOutDtypeValues[int(out_dtype)])
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns indices of the maximum values along an axis.
 *
 *        In the case of multiple occurrences of maximum values, the indices
 *        are returned.
 *
 *        Examples::
 *
 *        x = [[ 0.,  1.,  2.],
 *        [ 3.,  4.,  5.]]
 *
 *        // argmax along axis 0
 *        argmax(x, axis=0) = [ 1.,  1.,  1.]
 *
 *        // argmax along axis 1
 *        argmax(x, axis=1) = [ 2.,  2.]
 *
 *        // argmax along axis 1 keeping same dims as an input array
 *        argmax(x, axis=1, keepdims=True) = [[ 2.],
 *        [ 2.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/broadcast_reduce_op_index.cc:L52
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis The axis along which to perform the reduction. Negative values means
 *        indexing from right to left. ``Requires axis to be set as int, because global
 * \param keepdims If this is set to `True`, the reduced axis is left in the result as
 * \return new symbol
 */
inline Symbol argmax(const std::string& symbol_name,
                     Symbol data,
                     dmlc::optional<int> axis = dmlc::optional<int>(),
                     bool keepdims = false) {
  return Operator("argmax")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns indices of the minimum values along an axis.
 *
 *        In the case of multiple occurrences of minimum values, the indices
 *        are returned.
 *
 *        Examples::
 *
 *        x = [[ 0.,  1.,  2.],
 *        [ 3.,  4.,  5.]]
 *
 *        // argmin along axis 0
 *        argmin(x, axis=0) = [ 0.,  0.,  0.]
 *
 *        // argmin along axis 1
 *        argmin(x, axis=1) = [ 0.,  0.]
 *
 *        // argmin along axis 1 keeping same dims as an input array
 *        argmin(x, axis=1, keepdims=True) = [[ 0.],
 *        [ 0.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/broadcast_reduce_op_index.cc:L77
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis The axis along which to perform the reduction. Negative values means
 *        indexing from right to left. ``Requires axis to be set as int, because global
 * \param keepdims If this is set to `True`, the reduced axis is left in the result as
 * \return new symbol
 */
inline Symbol argmin(const std::string& symbol_name,
                     Symbol data,
                     dmlc::optional<int> axis = dmlc::optional<int>(),
                     bool keepdims = false) {
  return Operator("argmin")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns argmax indices of each channel from the input array.
 *
 *        The result will be an NDArray of shape (num_channel,).
 *
 *        In case of multiple occurrences of the maximum values, the indices
 *        are returned.
 *
 *        Examples::
 *
 *        x = [[ 0.,  1.,  2.],
 *        [ 3.,  4.,  5.]]
 *
 *        argmax_channel(x) = [ 2.,  2.]
 *
 *
 *
 *        Defined in ../src/operator/tensor/broadcast_reduce_op_index.cc:L97
 * \param symbol_name name of the resulting symbol
 * \param data The input array
 * \return new symbol
 */
inline Symbol argmax_channel(const std::string& symbol_name,
                             Symbol data) {
  return Operator("argmax_channel")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \brief Specify how out-of-bound indices behave. Default is "clip". "clip" means clip
 *        to the range. So, if all indices mentioned are too large, they are replaced by
 *        the index that addresses the last element along an axis.  "wrap" means to wrap
 */
enum class PickMode {
  kClip = 0,
  kWrap = 1
};

/*!
 * \brief Picks elements from an input array according to the input indices along the
 *
 *        Given an input array of shape ``(d0, d1)`` and indices of shape ``(i0,)``, the
 *        an output array of shape ``(i0,)`` with::
 *
 *        output[i] = input[i, indices[i]]
 *
 *        By default, if any index mentioned is too large, it is replaced by the index
 *        the last element along an axis (the `clip` mode).
 *
 *        This function supports n-dimensional input and (n-1)-dimensional indices arrays.
 *
 *        Examples::
 *
 *        x = [[ 1.,  2.],
 *        [ 3.,  4.],
 *        [ 5.,  6.]]
 *
 *        // picks elements with specified indices along axis 0
 *        pick(x, y=[0,1], 0) = [ 1.,  4.]
 *
 *        // picks elements with specified indices along axis 1
 *        pick(x, y=[0,1,0], 1) = [ 1.,  4.,  5.]
 *
 *        // picks elements with specified indices along axis 1 using 'wrap' mode
 *        // to place indicies that would normally be out of bounds
 *        pick(x, y=[2,-1,-2], 1, mode='wrap') = [ 1.,  4.,  5.]
 *
 *        y = [[ 1.],
 *        [ 0.],
 *        [ 2.]]
 *
 *        // picks elements with specified indices along axis 1 and dims are maintained
 *        pick(x, y, 1, keepdims=True) = [[ 2.],
 *        [ 3.],
 *        [ 6.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/broadcast_reduce_op_index.cc:L151
 * \param symbol_name name of the resulting symbol
 * \param data The input array
 * \param index The index array
 * \param axis int or None. The axis to picking the elements. Negative values means
 *        indexing from right to left. If is `None`, the elements in the index w.r.t the
 * \param keepdims If true, the axis where we pick the elements is left in the result as
 * \param mode Specify how out-of-bound indices behave. Default is "clip". "clip" means
 *        clip to the range. So, if all indices mentioned are too large, they are
 *        replaced by the index that addresses the last element along an axis.  "wrap"
 * \return new symbol
 */
inline Symbol pick(const std::string& symbol_name,
                   Symbol data,
                   Symbol index,
                   dmlc::optional<int> axis = dmlc::optional<int>(-1),
                   bool keepdims = false,
                   PickMode mode = PickMode::kClip) {
  static const char *PickModeValues[] = {
    "clip",
    "wrap"
  };
  return Operator("pick")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("mode", PickModeValues[int(mode)])
           .SetInput("data", data)
           .SetInput("index", index)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Broadcasts the input array over particular axes.
 *
 *        Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
 *        `(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.
 *
 *        `broadcast_axes` is an alias to the function `broadcast_axis`.
 *
 *        Example::
 *
 *        // given x of shape (1,2,1)
 *        x = [[[ 1.],
 *        [ 2.]]]
 *
 *        // broadcast x on on axis 2
 *        broadcast_axis(x, axis=2, size=3) = [[[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]]]
 *        // broadcast x on on axes 0 and 2
 *        broadcast_axis(x, axis=(0,2), size=(2,3)) = [[[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]],
 *        [[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]]]
 *
 *
 *        Defined in ../src/operator/tensor/broadcast_reduce_op_value.cc:L93
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis The axes to perform the broadcasting.
 * \param size Target sizes of the broadcasting axes.
 * \return new symbol
 */
inline Symbol broadcast_axis(const std::string& symbol_name,
                             Symbol data,
                             Shape axis = Shape(),
                             Shape size = Shape()) {
  return Operator("broadcast_axis")
           .SetParam("axis", axis)
           .SetParam("size", size)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Broadcasts the input array to a new shape.
 *
 *        Broadcasting is a mechanism that allows NDArrays to perform arithmetic
 *        with arrays of different shapes efficiently without creating multiple copies of
 *        Also see, `Broadcasting
 *        <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ for more
 *
 *        Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
 *        `(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.
 *
 *        For example::
 *
 *        broadcast_to([[1,2,3]], shape=(2,3)) = [[ 1.,  2.,  3.],
 *        [ 1.,  2.,  3.]])
 *
 *        The dimension which you do not want to change can also be kept as `0` which
 *        So with `shape=(2,0)`, we will obtain the same result as in the above example.
 *
 *
 *
 *        Defined in ../src/operator/tensor/broadcast_reduce_op_value.cc:L117
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param shape The shape of the desired array. We can set the dim to zero if it's same
 *        as the original. E.g `A = broadcast_to(B, shape=(10, 0, 0))` has the same
 * \return new symbol
 */
inline Symbol broadcast_to(const std::string& symbol_name,
                           Symbol data,
                           Shape shape = Shape()) {
  return Operator("broadcast_to")
           .SetParam("shape", shape)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Broadcasts lhs to have the same shape as rhs.
 *
 *        Broadcasting is a mechanism that allows NDArrays to perform arithmetic
 *        with arrays of different shapes efficiently without creating multiple copies of
 *        Also see, `Broadcasting
 *        <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ for more
 *
 *        Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
 *        `(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.
 *
 *        For example::
 *
 *        broadcast_like([[1,2,3]], [[5,6,7],[7,8,9]]) = [[ 1.,  2.,  3.],
 *        [ 1.,  2.,  3.]])
 *
 *        broadcast_like([9], [1,2,3,4,5], lhs_axes=(0,), rhs_axes=(-1,)) = [9,9,9,9,9]
 *
 *
 *
 *        Defined in ../src/operator/tensor/broadcast_reduce_op_value.cc:L179
 * \param symbol_name name of the resulting symbol
 * \param lhs First input.
 * \param rhs Second input.
 * \param lhs_axes Axes to perform broadcast on in the first input array
 * \param rhs_axes Axes to copy from the second input array
 * \return new symbol
 */
inline Symbol broadcast_like(const std::string& symbol_name,
                             Symbol lhs,
                             Symbol rhs,
                             dmlc::optional<Shape> lhs_axes = dmlc::optional<Shape>(),
                             dmlc::optional<Shape> rhs_axes = dmlc::optional<Shape>()) {
  return Operator("broadcast_like")
           .SetParam("lhs_axes", lhs_axes)
           .SetParam("rhs_axes", rhs_axes)
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Computes the product of array elements over given axes.
 *
 *        Defined in ../src/operator/tensor/./broadcast_reduce_op.h:L31
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis The axis or axes along which to perform the reduction.
 *
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *
 *        If `axis` is int, a reduction is performed on a particular axis.
 *
 *        If `axis` is a tuple of ints, a reduction is performed on all the axes
 *        specified in the tuple.
 *
 *        If `exclude` is true, reduction will be performed on the axes that are
 *        NOT in axis instead.
 *
 *        Negative values means indexing from right to left.
 * \param keepdims If this is set to `True`, the reduced axes are left in the result as
 * \param exclude Whether to perform reduction on axis that are NOT in axis instead.
 * \return new symbol
 */
inline Symbol prod(const std::string& symbol_name,
                   Symbol data,
                   dmlc::optional<Shape> axis = dmlc::optional<Shape>(),
                   bool keepdims = false,
                   bool exclude = false) {
  return Operator("prod")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("exclude", exclude)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Computes the product of array elements over given axes treating Not a Numbers
 *
 *
 *
 *        Defined in ../src/operator/tensor/broadcast_reduce_prod_value.cc:L47
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis The axis or axes along which to perform the reduction.
 *
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *
 *        If `axis` is int, a reduction is performed on a particular axis.
 *
 *        If `axis` is a tuple of ints, a reduction is performed on all the axes
 *        specified in the tuple.
 *
 *        If `exclude` is true, reduction will be performed on the axes that are
 *        NOT in axis instead.
 *
 *        Negative values means indexing from right to left.
 * \param keepdims If this is set to `True`, the reduced axes are left in the result as
 * \param exclude Whether to perform reduction on axis that are NOT in axis instead.
 * \return new symbol
 */
inline Symbol nanprod(const std::string& symbol_name,
                      Symbol data,
                      dmlc::optional<Shape> axis = dmlc::optional<Shape>(),
                      bool keepdims = false,
                      bool exclude = false) {
  return Operator("nanprod")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("exclude", exclude)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Computes the sum of array elements over given axes.
 *
 *        .. Note::
 *
 *        `sum` and `sum_axis` are equivalent.
 *        For ndarray of csr storage type summation along axis 0 and axis 1 is supported.
 *        Setting keepdims or exclude to True will cause a fallback to dense operator.
 *
 *        Example::
 *
 *        data = [[[1, 2], [2, 3], [1, 3]],
 *        [[1, 4], [4, 3], [5, 2]],
 *        [[7, 1], [7, 2], [7, 3]]]
 *
 *        sum(data, axis=1)
 *        [[  4.   8.]
 *        [ 10.   9.]
 *        [ 21.   6.]]
 *
 *        sum(data, axis=[1,2])
 *        [ 12.  19.  27.]
 *
 *        data = [[1, 2, 0],
 *        [3, 0, 1],
 *        [4, 1, 0]]
 *
 *        csr = cast_storage(data, 'csr')
 *
 *        sum(csr, axis=0)
 *        [ 8.  3.  1.]
 *
 *        sum(csr, axis=1)
 *        [ 3.  4.  5.]
 *
 *
 *
 *        Defined in ../src/operator/tensor/broadcast_reduce_sum_value.cc:L67
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis The axis or axes along which to perform the reduction.
 *
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *
 *        If `axis` is int, a reduction is performed on a particular axis.
 *
 *        If `axis` is a tuple of ints, a reduction is performed on all the axes
 *        specified in the tuple.
 *
 *        If `exclude` is true, reduction will be performed on the axes that are
 *        NOT in axis instead.
 *
 *        Negative values means indexing from right to left.
 * \param keepdims If this is set to `True`, the reduced axes are left in the result as
 * \param exclude Whether to perform reduction on axis that are NOT in axis instead.
 * \return new symbol
 */
inline Symbol sum(const std::string& symbol_name,
                  Symbol data,
                  dmlc::optional<Shape> axis = dmlc::optional<Shape>(),
                  bool keepdims = false,
                  bool exclude = false) {
  return Operator("sum")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("exclude", exclude)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Computes the mean of array elements over given axes.
 *
 *        Defined in ../src/operator/tensor/./broadcast_reduce_op.h:L84
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis The axis or axes along which to perform the reduction.
 *
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *
 *        If `axis` is int, a reduction is performed on a particular axis.
 *
 *        If `axis` is a tuple of ints, a reduction is performed on all the axes
 *        specified in the tuple.
 *
 *        If `exclude` is true, reduction will be performed on the axes that are
 *        NOT in axis instead.
 *
 *        Negative values means indexing from right to left.
 * \param keepdims If this is set to `True`, the reduced axes are left in the result as
 * \param exclude Whether to perform reduction on axis that are NOT in axis instead.
 * \return new symbol
 */
inline Symbol mean(const std::string& symbol_name,
                   Symbol data,
                   dmlc::optional<Shape> axis = dmlc::optional<Shape>(),
                   bool keepdims = false,
                   bool exclude = false) {
  return Operator("mean")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("exclude", exclude)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Computes the sum of array elements over given axes treating Not a Numbers
 *
 *
 *
 *        Defined in ../src/operator/tensor/broadcast_reduce_sum_value.cc:L102
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \param axis The axis or axes along which to perform the reduction.
 *
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *
 *        If `axis` is int, a reduction is performed on a particular axis.
 *
 *        If `axis` is a tuple of ints, a reduction is performed on all the axes
 *        specified in the tuple.
 *
 *        If `exclude` is true, reduction will be performed on the axes that are
 *        NOT in axis instead.
 *
 *        Negative values means indexing from right to left.
 * \param keepdims If this is set to `True`, the reduced axes are left in the result as
 * \param exclude Whether to perform reduction on axis that are NOT in axis instead.
 * \return new symbol
 */
inline Symbol nansum(const std::string& symbol_name,
                     Symbol data,
                     dmlc::optional<Shape> axis = dmlc::optional<Shape>(),
                     bool keepdims = false,
                     bool exclude = false) {
  return Operator("nansum")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("exclude", exclude)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \brief Output storage type.
 */
enum class Cast_storageStype {
  kCsr = 0,
  kDefault = 1,
  kRow_sparse = 2
};

/*!
 * \brief Casts tensor storage type to the new type.
 *
 *        When an NDArray with default storage type is cast to csr or row_sparse storage,
 *        the result is compact, which means:
 *
 *        - for csr, zero values will not be retained
 *        - for row_sparse, row slices of all zeros will not be retained
 *
 *        The storage type of ``cast_storage`` output depends on stype parameter:
 *
 *        - cast_storage(csr, 'default') = default
 *        - cast_storage(row_sparse, 'default') = default
 *        - cast_storage(default, 'csr') = csr
 *        - cast_storage(default, 'row_sparse') = row_sparse
 *        - cast_storage(csr, 'csr') = csr
 *        - cast_storage(row_sparse, 'row_sparse') = row_sparse
 *
 *        Example::
 *
 *        dense = [[ 0.,  1.,  0.],
 *        [ 2.,  0.,  3.],
 *        [ 0.,  0.,  0.],
 *        [ 0.,  0.,  0.]]
 *
 *        # cast to row_sparse storage type
 *        rsp = cast_storage(dense, 'row_sparse')
 *        rsp.indices = [0, 1]
 *        rsp.values = [[ 0.,  1.,  0.],
 *        [ 2.,  0.,  3.]]
 *
 *        # cast to csr storage type
 *        csr = cast_storage(dense, 'csr')
 *        csr.indices = [1, 0, 2]
 *        csr.values = [ 1.,  2.,  3.]
 *        csr.indptr = [0, 1, 3, 3, 3]
 *
 *
 *
 *        Defined in ../src/operator/tensor/cast_storage.cc:L71
 * \param symbol_name name of the resulting symbol
 * \param data The input.
 * \param stype Output storage type.
 * \return new symbol
 */
inline Symbol cast_storage(const std::string& symbol_name,
                           Symbol data,
                           Cast_storageStype stype) {
  static const char *Cast_storageStypeValues[] = {
    "csr",
    "default",
    "row_sparse"
  };
  return Operator("cast_storage")
           .SetParam("stype", Cast_storageStypeValues[int(stype)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Return the elements, either from x or y, depending on the condition.
 *
 *        Given three ndarrays, condition, x, and y, return an ndarray with the elements
 *        depending on the elements from condition are true or false. x and y must have
 *        If condition has the same shape as x, each element in the output array is from
 *        corresponding element in the condition is true, and from y if false.
 *
 *        If condition does not have the same shape as x, it must be a 1D array whose
 *        the same as x's first dimension size. Each row of the output array is from x's
 *        if the corresponding element from condition is true, and from y's row if false.
 *
 *        Note that all non-zero values are interpreted as ``True`` in condition.
 *
 *        Examples::
 *
 *        x = [[1, 2], [3, 4]]
 *        y = [[5, 6], [7, 8]]
 *        cond = [[0, 1], [-1, 0]]
 *
 *        where(cond, x, y) = [[5, 2], [3, 8]]
 *
 *        csr_cond = cast_storage(cond, 'csr')
 *
 *        where(csr_cond, x, y) = [[5, 2], [3, 8]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/control_flow_op.cc:L57
 * \param symbol_name name of the resulting symbol
 * \param condition condition array
 * \param x
 * \param y
 * \return new symbol
 */
inline Symbol where(const std::string& symbol_name,
                    Symbol condition,
                    Symbol x,
                    Symbol y) {
  return Operator("where")
           .SetInput("condition", condition)
           .SetInput("x", x)
           .SetInput("y", y)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Extracts a diagonal or constructs a diagonal array.
 *
 *        ``diag``'s behavior depends on the input array dimensions:
 *
 *        - 1-D arrays: constructs a 2-D array with the input as its diagonal, all other
 *        - N-D arrays: extracts the diagonals of the sub-arrays with axes specified by
 *        The output shape would be decided by removing the axes numbered ``axis1`` and
 *        input shape and appending to the result a new axis with the size of the
 *
 *        For example, when the input shape is `(2, 3, 4, 5)`, ``axis1`` and ``axis2``
 *        respectively and ``k`` is 0, the resulting shape would be `(3, 5, 2)`.
 *
 *        Examples::
 *
 *        x = [[1, 2, 3],
 *        [4, 5, 6]]
 *
 *        diag(x) = [1, 5]
 *
 *        diag(x, k=1) = [2, 6]
 *
 *        diag(x, k=-1) = [4]
 *
 *        x = [1, 2, 3]
 *
 *        diag(x) = [[1, 0, 0],
 *        [0, 2, 0],
 *        [0, 0, 3]]
 *
 *        diag(x, k=1) = [[0, 1, 0],
 *        [0, 0, 2],
 *        [0, 0, 0]]
 *
 *        diag(x, k=-1) = [[0, 0, 0],
 *        [1, 0, 0],
 *        [0, 2, 0]]
 *
 *        x = [[[1, 2],
 *        [3, 4]],
 *
 *        [[5, 6],
 *        [7, 8]]]
 *
 *        diag(x) = [[1, 7],
 *        [2, 8]]
 *
 *        diag(x, k=1) = [[3],
 *        [4]]
 *
 *        diag(x, axis1=-2, axis2=-1) = [[1, 4],
 *        [5, 8]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/diag_op.cc:L87
 * \param symbol_name name of the resulting symbol
 * \param data Input ndarray
 * \param k Diagonal in question. The default is 0. Use k>0 for diagonals above the main
 *        diagonal, and k<0 for diagonals below the main diagonal. If input has shape (S0
 * \param axis1 The first axis of the sub-arrays of interest. Ignored when the input is a
 * \param axis2 The second axis of the sub-arrays of interest. Ignored when the input is
 * \return new symbol
 */
inline Symbol diag(const std::string& symbol_name,
                   Symbol data,
                   int k = 0,
                   int axis1 = 0,
                   int axis2 = 1) {
  return Operator("diag")
           .SetParam("k", k)
           .SetParam("axis1", axis1)
           .SetParam("axis2", axis2)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \brief The desired storage type of the forward output given by user, if thecombination
 *        of input storage types and this hint does not matchany implemented ones, the
 *        dot operator will perform fallback operationand still produce an output of the
 */
enum class DotForwardStype {
  kNone = 0,
  kCsr = 1,
  kDefault = 2,
  kRow_sparse = 3
};

/*!
 * \brief Dot product of two arrays.
 *
 *        ``dot``'s behavior depends on the input array dimensions:
 *
 *        - 1-D arrays: inner product of vectors
 *        - 2-D arrays: matrix multiplication
 *        - N-D arrays: a sum product over the last axis of the first input and the first
 *        axis of the second input
 *
 *        For example, given 3-D ``x`` with shape `(n,m,k)` and ``y`` with shape
 *        result array will have shape `(n,m,r,s)`. It is computed by::
 *
 *        dot(x,y)[i,j,a,b] = sum(x[i,j,:]*y[:,a,b])
 *
 *        Example::
 *
 *        x = reshape([0,1,2,3,4,5,6,7], shape=(2,2,2))
 *        y = reshape([7,6,5,4,3,2,1,0], shape=(2,2,2))
 *        dot(x,y)[0,0,1,1] = 0
 *        sum(x[0,0,:]*y[:,1,1]) = 0
 *
 *        The storage type of ``dot`` output depends on storage types of inputs,
 *        forward_stype option for output storage type. Implemented sparse operations
 *
 *        - dot(default, default, transpose_a=True/False, transpose_b=True/False) =
 *        - dot(csr, default, transpose_a=True) = default
 *        - dot(csr, default, transpose_a=True) = row_sparse
 *        - dot(csr, default) = default
 *        - dot(csr, row_sparse) = default
 *        - dot(default, csr) = csr (CPU only)
 *        - dot(default, csr, forward_stype='default') = default
 *        - dot(default, csr, transpose_b=True, forward_stype='default') = default
 *
 *        If the combination of input storage types and forward_stype does not match any
 *        above patterns, ``dot`` will fallback and generate output with default storage.
 *
 *        .. Note::
 *
 *        If the storage type of the lhs is "csr", the storage type of gradient w.r.t rhs
 *        "row_sparse". Only a subset of optimizers support sparse gradients, including
 *        and Adam. Note that by default lazy updates is turned on, which may perform
 *        from standard updates. For more details, please check the Optimization API at:
 *        https://mxnet.incubator.apache.org/api/python/optimization/optimization.html
 *
 *
 *
 *        Defined in ../src/operator/tensor/dot.cc:L77
 * \param symbol_name name of the resulting symbol
 * \param lhs The first input
 * \param rhs The second input
 * \param transpose_a If true then transpose the first input before dot.
 * \param transpose_b If true then transpose the second input before dot.
 * \param forward_stype The desired storage type of the forward output given by user, if
 *        thecombination of input storage types and this hint does not matchany
 *        implemented ones, the dot operator will perform fallback operationand still
 * \return new symbol
 */
inline Symbol dot(const std::string& symbol_name,
                  Symbol lhs,
                  Symbol rhs,
                  bool transpose_a = false,
                  bool transpose_b = false,
                  DotForwardStype forward_stype = DotForwardStype::kNone) {
  static const char *DotForwardStypeValues[] = {
    "None",
    "csr",
    "default",
    "row_sparse"
  };
  return Operator("dot")
           .SetParam("transpose_a", transpose_a)
           .SetParam("transpose_b", transpose_b)
           .SetParam("forward_stype", DotForwardStypeValues[int(forward_stype)])
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*! \brief The desired storage type of the forward output given by user, if thecombination
 *        of input storage types and this hint does not matchany implemented ones, the
 *        dot operator will perform fallback operationand still produce an output of the
 */
enum class Batch_dotForwardStype {
  kNone = 0,
  kCsr = 1,
  kDefault = 2,
  kRow_sparse = 3
};

/*!
 * \brief Batchwise dot product.
 *
 *        ``batch_dot`` is used to compute dot product of ``x`` and ``y`` when ``x`` and
 *        ``y`` are data in batch, namely N-D (N >= 3) arrays in shape of `(B0, ..., B_i,
 *
 *        For example, given ``x`` with shape `(B_0, ..., B_i, N, M)` and ``y`` with shape
 *        `(B_0, ..., B_i, M, K)`, the result array will have shape `(B_0, ..., B_i, N,
 *        which is computed by::
 *
 *        batch_dot(x,y)[b_0, ..., b_i, :, :] = dot(x[b_0, ..., b_i, :, :], y[b_0, ...,
 *
 *
 *
 *        Defined in ../src/operator/tensor/dot.cc:L127
 * \param symbol_name name of the resulting symbol
 * \param lhs The first input
 * \param rhs The second input
 * \param transpose_a If true then transpose the first input before dot.
 * \param transpose_b If true then transpose the second input before dot.
 * \param forward_stype The desired storage type of the forward output given by user, if
 *        thecombination of input storage types and this hint does not matchany
 *        implemented ones, the dot operator will perform fallback operationand still
 * \return new symbol
 */
inline Symbol batch_dot(const std::string& symbol_name,
                        Symbol lhs,
                        Symbol rhs,
                        bool transpose_a = false,
                        bool transpose_b = false,
                        Batch_dotForwardStype forward_stype = Batch_dotForwardStype::kNone) {
  static const char *Batch_dotForwardStypeValues[] = {
    "None",
    "csr",
    "default",
    "row_sparse"
  };
  return Operator("batch_dot")
           .SetParam("transpose_a", transpose_a)
           .SetParam("transpose_b", transpose_b)
           .SetParam("forward_stype", Batch_dotForwardStypeValues[int(forward_stype)])
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise sum of the input arrays with broadcasting.
 *
 *        `broadcast_plus` is an alias to the function `broadcast_add`.
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_add(x, y) = [[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]]
 *
 *        broadcast_plus(x, y) = [[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]]
 *
 *        Supported sparse operations:
 *
 *        broadcast_add(csr, dense(1D)) = dense
 *        broadcast_add(dense(1D), csr) = dense
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L58
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_add(const std::string& symbol_name,
                            Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_add")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise difference of the input arrays with broadcasting.
 *
 *        `broadcast_minus` is an alias to the function `broadcast_sub`.
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_sub(x, y) = [[ 1.,  1.,  1.],
 *        [ 0.,  0.,  0.]]
 *
 *        broadcast_minus(x, y) = [[ 1.,  1.,  1.],
 *        [ 0.,  0.,  0.]]
 *
 *        Supported sparse operations:
 *
 *        broadcast_sub/minus(csr, dense(1D)) = dense
 *        broadcast_sub/minus(dense(1D), csr) = dense
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L106
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_sub(const std::string& symbol_name,
                            Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_sub")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise product of the input arrays with broadcasting.
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_mul(x, y) = [[ 0.,  0.,  0.],
 *        [ 1.,  1.,  1.]]
 *
 *        Supported sparse operations:
 *
 *        broadcast_mul(csr, dense(1D)) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L146
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_mul(const std::string& symbol_name,
                            Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_mul")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise division of the input arrays with broadcasting.
 *
 *        Example::
 *
 *        x = [[ 6.,  6.,  6.],
 *        [ 6.,  6.,  6.]]
 *
 *        y = [[ 2.],
 *        [ 3.]]
 *
 *        broadcast_div(x, y) = [[ 3.,  3.,  3.],
 *        [ 2.,  2.,  2.]]
 *
 *        Supported sparse operations:
 *
 *        broadcast_div(csr, dense(1D)) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L187
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_div(const std::string& symbol_name,
                            Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_div")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise modulo of the input arrays with broadcasting.
 *
 *        Example::
 *
 *        x = [[ 8.,  8.,  8.],
 *        [ 8.,  8.,  8.]]
 *
 *        y = [[ 2.],
 *        [ 3.]]
 *
 *        broadcast_mod(x, y) = [[ 0.,  0.,  0.],
 *        [ 2.,  2.,  2.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L222
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_mod(const std::string& symbol_name,
                            Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_mod")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns result of first array elements raised to powers from second array,
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_power(x, y) = [[ 2.,  2.,  2.],
 *        [ 4.,  4.,  4.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L45
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_power(const std::string& symbol_name,
                              Symbol lhs,
                              Symbol rhs) {
  return Operator("broadcast_power")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise maximum of the input arrays with broadcasting.
 *
 *        This function compares two input arrays and returns a new array having the
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_maximum(x, y) = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L81
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_maximum(const std::string& symbol_name,
                                Symbol lhs,
                                Symbol rhs) {
  return Operator("broadcast_maximum")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise minimum of the input arrays with broadcasting.
 *
 *        This function compares two input arrays and returns a new array having the
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_maximum(x, y) = [[ 0.,  0.,  0.],
 *        [ 1.,  1.,  1.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L117
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_minimum(const std::string& symbol_name,
                                Symbol lhs,
                                Symbol rhs) {
  return Operator("broadcast_minimum")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns the hypotenuse of a right angled triangle, given its "legs"
 *        with broadcasting.
 *
 *        It is equivalent to doing :math:`sqrt(x_1^2 + x_2^2)`.
 *
 *        Example::
 *
 *        x = [[ 3.,  3.,  3.]]
 *
 *        y = [[ 4.],
 *        [ 4.]]
 *
 *        broadcast_hypot(x, y) = [[ 5.,  5.,  5.],
 *        [ 5.,  5.,  5.]]
 *
 *        z = [[ 0.],
 *        [ 4.]]
 *
 *        broadcast_hypot(x, z) = [[ 3.,  3.,  3.],
 *        [ 5.,  5.,  5.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L158
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_hypot(const std::string& symbol_name,
                              Symbol lhs,
                              Symbol rhs) {
  return Operator("broadcast_hypot")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns the result of element-wise **equal to** (==) comparison operation with
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_equal(x, y) = [[ 0.,  0.,  0.],
 *        [ 1.,  1.,  1.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L46
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_equal(const std::string& symbol_name,
                              Symbol lhs,
                              Symbol rhs) {
  return Operator("broadcast_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns the result of element-wise **not equal to** (!=) comparison operation
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_not_equal(x, y) = [[ 1.,  1.,  1.],
 *        [ 0.,  0.,  0.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L64
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_not_equal(const std::string& symbol_name,
                                  Symbol lhs,
                                  Symbol rhs) {
  return Operator("broadcast_not_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns the result of element-wise **greater than** (>) comparison operation
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_greater(x, y) = [[ 1.,  1.,  1.],
 *        [ 0.,  0.,  0.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L82
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_greater(const std::string& symbol_name,
                                Symbol lhs,
                                Symbol rhs) {
  return Operator("broadcast_greater")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns the result of element-wise **greater than or equal to** (>=) comparison
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_greater_equal(x, y) = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L100
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_greater_equal(const std::string& symbol_name,
                                      Symbol lhs,
                                      Symbol rhs) {
  return Operator("broadcast_greater_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns the result of element-wise **lesser than** (<) comparison operation
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_lesser(x, y) = [[ 0.,  0.,  0.],
 *        [ 0.,  0.,  0.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L118
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_lesser(const std::string& symbol_name,
                               Symbol lhs,
                               Symbol rhs) {
  return Operator("broadcast_lesser")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns the result of element-wise **lesser than or equal to** (<=) comparison
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_lesser_equal(x, y) = [[ 0.,  0.,  0.],
 *        [ 1.,  1.,  1.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L136
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_lesser_equal(const std::string& symbol_name,
                                     Symbol lhs,
                                     Symbol rhs) {
  return Operator("broadcast_lesser_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns the result of element-wise **logical and** with broadcasting.
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_logical_and(x, y) = [[ 0.,  0.,  0.],
 *        [ 1.,  1.,  1.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L154
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_logical_and(const std::string& symbol_name,
                                    Symbol lhs,
                                    Symbol rhs) {
  return Operator("broadcast_logical_and")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns the result of element-wise **logical or** with broadcasting.
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  0.],
 *        [ 1.,  1.,  0.]]
 *
 *        y = [[ 1.],
 *        [ 0.]]
 *
 *        broadcast_logical_or(x, y) = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  0.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L172
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_logical_or(const std::string& symbol_name,
                                   Symbol lhs,
                                   Symbol rhs) {
  return Operator("broadcast_logical_or")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns the result of element-wise **logical xor** with broadcasting.
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  0.],
 *        [ 1.,  1.,  0.]]
 *
 *        y = [[ 1.],
 *        [ 0.]]
 *
 *        broadcast_logical_xor(x, y) = [[ 0.,  0.,  1.],
 *        [ 1.,  1.,  0.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L190
 * \param symbol_name name of the resulting symbol
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_logical_xor(const std::string& symbol_name,
                                    Symbol lhs,
                                    Symbol rhs) {
  return Operator("broadcast_logical_xor")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Subtracts arguments element-wise.
 *
 *        The storage type of ``elemwise_sub`` output depends on storage types of inputs
 *
 *        - elemwise_sub(row_sparse, row_sparse) = row_sparse
 *        - elemwise_sub(csr, csr) = csr
 *        - elemwise_sub(default, csr) = default
 *        - elemwise_sub(csr, default) = default
 *        - elemwise_sub(default, rsp) = default
 *        - elemwise_sub(rsp, default) = default
 *        - otherwise, ``elemwise_sub`` generates output with default storage
 *
 *
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol elemwise_sub(const std::string& symbol_name,
                           Symbol lhs,
                           Symbol rhs) {
  return Operator("elemwise_sub")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Divides arguments element-wise.
 *
 *        The storage type of ``elemwise_div`` output is always dense
 *
 *
 * \param symbol_name name of the resulting symbol
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol elemwise_div(const std::string& symbol_name,
                           Symbol lhs,
                           Symbol rhs) {
  return Operator("elemwise_div")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Calculate Smooth L1 Loss(lhs, scalar) by summing
 *
 *        .. math::
 *
 *        f(x) =
 *        \begin{cases}
 *        (\sigma x)^2/2,& \text{if }x < 1/\sigma^2\\
 *        |x|-0.5/\sigma^2,& \text{otherwise}
 *        \end{cases}
 *
 *        where :math:`x` is an element of the tensor *lhs* and :math:`\sigma` is the
 *
 *        Example::
 *
 *        smooth_l1([1, 2, 3, 4]) = [0.5, 1.5, 2.5, 3.5]
 *        smooth_l1([1, 2, 3, 4], scalar=1) = [0.5, 1.5, 2.5, 3.5]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_scalar_op_extended.cc:L109
 * \param symbol_name name of the resulting symbol
 * \param data source input
 * \param scalar scalar input
 * \return new symbol
 */
inline Symbol smooth_l1(const std::string& symbol_name,
                        Symbol data,
                        mx_float scalar) {
  return Operator("smooth_l1")
           .SetParam("scalar", scalar)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Adds all input arguments element-wise.
 *
 *        .. math::
 *        add\_n(a_1, a_2, ..., a_n) = a_1 + a_2 + ... + a_n
 *
 *        ``add_n`` is potentially more efficient than calling ``add`` by `n` times.
 *
 *        The storage type of ``add_n`` output depends on storage types of inputs
 *
 *        - add_n(row_sparse, row_sparse, ..) = row_sparse
 *        - add_n(default, csr, default) = default
 *        - add_n(any input combinations longer than 4 (>4) with at least one default
 *        - otherwise, ``add_n`` falls all inputs back to default storage and generates
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_sum.cc:L156
 * \param symbol_name name of the resulting symbol
 * \param args Positional input arguments
 * \return new symbol
 */
inline Symbol add_n(const std::string& symbol_name,
                    const std::vector<Symbol>& args) {
  return Operator("add_n")
(args)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Computes rectified linear activation.
 *
 *        .. math::
 *        max(features, 0)
 *
 *        The storage type of ``relu`` output depends upon the input storage type:
 *
 *        - relu(default) = default
 *        - relu(row_sparse) = row_sparse
 *        - relu(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L85
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol relu(const std::string& symbol_name,
                   Symbol data) {
  return Operator("relu")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Computes sigmoid of x element-wise.
 *
 *        .. math::
 *        y = 1 / (1 + exp(-x))
 *
 *        The storage type of ``sigmoid`` output is always dense
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L119
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sigmoid(const std::string& symbol_name,
                      Symbol data) {
  return Operator("sigmoid")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Computes hard sigmoid of x element-wise.
 *
 *        .. math::
 *        y = max(0, min(1, alpha * x + beta))
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L161
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \param alpha Slope of hard sigmoid
 * \param beta Bias of hard sigmoid.
 * \return new symbol
 */
inline Symbol hard_sigmoid(const std::string& symbol_name,
                           Symbol data,
                           mx_float alpha = 0.200000003,
                           mx_float beta = 0.5) {
  return Operator("hard_sigmoid")
           .SetParam("alpha", alpha)
           .SetParam("beta", beta)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Computes softsign of x element-wise.
 *
 *        .. math::
 *        y = x / (1 + abs(x))
 *
 *        The storage type of ``softsign`` output is always dense
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L191
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol softsign(const std::string& symbol_name,
                       Symbol data) {
  return Operator("softsign")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Stops gradient computation.
 *
 *        Stops the accumulated gradient of the inputs from flowing through this operator
 *        in the backward direction. In other words, this operator prevents the
 *        of its inputs to be taken into account for computing gradients.
 *
 *        Example::
 *
 *        v1 = [1, 2]
 *        v2 = [0, 1]
 *        a = Variable('a')
 *        b = Variable('b')
 *        b_stop_grad = stop_gradient(3 * b)
 *        loss = MakeLoss(b_stop_grad + a)
 *
 *        executor = loss.simple_bind(ctx=cpu(), a=(1,2), b=(1,2))
 *        executor.forward(is_train=True, a=v1, b=v2)
 *        executor.outputs
 *        [ 1.  5.]
 *
 *        executor.backward()
 *        executor.grad_arrays
 *        [ 0.  0.]
 *        [ 1.  1.]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L325
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol BlockGrad(const std::string& symbol_name,
                        Symbol data) {
  return Operator("BlockGrad")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Make your own loss function in network construction.
 *
 *        This operator accepts a customized loss function symbol as a terminal loss and
 *        the symbol should be an operator with no backward dependency.
 *        The output of this function is the gradient of loss with respect to the input
 *
 *        For example, if you are a making a cross entropy loss function. Assume ``out``
 *        predicted output and ``label`` is the true label, then the cross entropy can be
 *
 *        cross_entropy = label * log(out) + (1 - label) * log(1 - out)
 *        loss = make_loss(cross_entropy)
 *
 *        We will need to use ``make_loss`` when we are creating our own loss function or
 *        combine multiple loss functions. Also we may want to stop some variables'
 *        from backpropagation. See more detail in ``BlockGrad`` or ``stop_gradient``.
 *
 *        The storage type of ``make_loss`` output depends upon the input storage type:
 *
 *        - make_loss(default) = default
 *        - make_loss(row_sparse) = row_sparse
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L358
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol make_loss(const std::string& symbol_name,
                        Symbol data) {
  return Operator("make_loss")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Reshape some or all dimensions of `lhs` to have the same shape as some or all
 *
 *        Returns a **view** of the `lhs` array with a new shape without altering any
 *
 *        Example::
 *
 *        x = [1, 2, 3, 4, 5, 6]
 *        y = [[0, -4], [3, 2], [2, 2]]
 *        reshape_like(x, y) = [[1, 2], [3, 4], [5, 6]]
 *
 *        More precise control over how dimensions are inherited is achieved by
 *        slices over the `lhs` and `rhs` array dimensions. Only the sliced `lhs`
 *        are reshaped to the `rhs` sliced dimensions, with the non-sliced `lhs`
 *
 *        Examples::
 *
 *        - lhs shape = (30,7), rhs shape = (15,2,4), lhs_begin=0, lhs_end=1,
 *        - lhs shape = (3, 5), rhs shape = (1,15,4), lhs_begin=0, lhs_end=2,
 *
 *        Negative indices are supported, and `None` can be used for either `lhs_end` or
 *
 *        Example::
 *
 *        - lhs shape = (30, 12), rhs shape = (4, 2, 2, 3), lhs_begin=-1, lhs_end=None,
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L511
 * \param symbol_name name of the resulting symbol
 * \param lhs First input.
 * \param rhs Second input.
 * \param lhs_begin Defaults to 0. The beginning index along which the lhs dimensions are
 * \param lhs_end Defaults to None. The ending index along which the lhs dimensions are
 * \param rhs_begin Defaults to 0. The beginning index along which the rhs dimensions are
 * \param rhs_end Defaults to None. The ending index along which the rhs dimensions are
 * \return new symbol
 */
inline Symbol reshape_like(const std::string& symbol_name,
                           Symbol lhs,
                           Symbol rhs,
                           dmlc::optional<int> lhs_begin = dmlc::optional<int>(),
                           dmlc::optional<int> lhs_end = dmlc::optional<int>(),
                           dmlc::optional<int> rhs_begin = dmlc::optional<int>(),
                           dmlc::optional<int> rhs_end = dmlc::optional<int>()) {
  return Operator("reshape_like")
           .SetParam("lhs_begin", lhs_begin)
           .SetParam("lhs_end", lhs_end)
           .SetParam("rhs_begin", rhs_begin)
           .SetParam("rhs_end", rhs_end)
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns a 1D int64 array containing the shape of data.
 *
 *        Example::
 *
 *        shape_array([[1,2,3,4], [5,6,7,8]]) = [2,4]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L573
 * \param symbol_name name of the resulting symbol
 * \param data Input Array.
 * \return new symbol
 */
inline Symbol shape_array(const std::string& symbol_name,
                          Symbol data) {
  return Operator("shape_array")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns a 1D int64 array containing the size of data.
 *
 *        Example::
 *
 *        size_array([[1,2,3,4], [5,6,7,8]]) = [8]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L624
 * \param symbol_name name of the resulting symbol
 * \param data Input Array.
 * \return new symbol
 */
inline Symbol size_array(const std::string& symbol_name,
                         Symbol data) {
  return Operator("size_array")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \brief Output data type.
 */
enum class CastDtype {
  kBfloat16 = 0,
  kBool = 1,
  kFloat16 = 2,
  kFloat32 = 3,
  kFloat64 = 4,
  kInt32 = 5,
  kInt64 = 6,
  kInt8 = 7,
  kUint8 = 8
};

/*!
 * \brief Casts all elements of the input to a new type.
 *
 *        .. note:: ``Cast`` is deprecated. Use ``cast`` instead.
 *
 *        Example::
 *
 *        cast([0.9, 1.3], dtype='int32') = [0, 1]
 *        cast([1e20, 11.1], dtype='float16') = [inf, 11.09375]
 *        cast([300, 11.1, 10.9, -1, -3], dtype='uint8') = [44, 11, 10, 255, 253]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L664
 * \param symbol_name name of the resulting symbol
 * \param data The input.
 * \param dtype Output data type.
 * \return new symbol
 */
inline Symbol Cast(const std::string& symbol_name,
                   Symbol data,
                   CastDtype dtype) {
  static const char *CastDtypeValues[] = {
    "bfloat16",
    "bool",
    "float16",
    "float32",
    "float64",
    "int32",
    "int64",
    "int8",
    "uint8"
  };
  return Operator("Cast")
           .SetParam("dtype", CastDtypeValues[int(dtype)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Numerical negative of the argument, element-wise.
 *
 *        The storage type of ``negative`` output depends upon the input storage type:
 *
 *        - negative(default) = default
 *        - negative(row_sparse) = row_sparse
 *        - negative(csr) = csr
 *
 *
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol negative(const std::string& symbol_name,
                       Symbol data) {
  return Operator("negative")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise absolute value of the input.
 *
 *        Example::
 *
 *        abs([-2, 0, 3]) = [2, 0, 3]
 *
 *        The storage type of ``abs`` output depends upon the input storage type:
 *
 *        - abs(default) = default
 *        - abs(row_sparse) = row_sparse
 *        - abs(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L720
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol abs(const std::string& symbol_name,
                  Symbol data) {
  return Operator("abs")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise sign of the input.
 *
 *        Example::
 *
 *        sign([-2, 0, 3]) = [-1, 0, 1]
 *
 *        The storage type of ``sign`` output depends upon the input storage type:
 *
 *        - sign(default) = default
 *        - sign(row_sparse) = row_sparse
 *        - sign(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L758
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sign(const std::string& symbol_name,
                   Symbol data) {
  return Operator("sign")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise rounded value to the nearest integer of the input.
 *
 *        Example::
 *
 *        round([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  2., -2.,  2.,  2.]
 *
 *        The storage type of ``round`` output depends upon the input storage type:
 *
 *        - round(default) = default
 *        - round(row_sparse) = row_sparse
 *        - round(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L777
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol round(const std::string& symbol_name,
                    Symbol data) {
  return Operator("round")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise rounded value to the nearest integer of the input.
 *
 *        .. note::
 *        - For input ``n.5`` ``rint`` returns ``n`` while ``round`` returns ``n+1``.
 *        - For input ``-n.5`` both ``rint`` and ``round`` returns ``-n-1``.
 *
 *        Example::
 *
 *        rint([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  1., -2.,  2.,  2.]
 *
 *        The storage type of ``rint`` output depends upon the input storage type:
 *
 *        - rint(default) = default
 *        - rint(row_sparse) = row_sparse
 *        - rint(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L798
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol rint(const std::string& symbol_name,
                   Symbol data) {
  return Operator("rint")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise ceiling of the input.
 *
 *        The ceil of the scalar x is the smallest integer i, such that i >= x.
 *
 *        Example::
 *
 *        ceil([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  2.,  2.,  3.]
 *
 *        The storage type of ``ceil`` output depends upon the input storage type:
 *
 *        - ceil(default) = default
 *        - ceil(row_sparse) = row_sparse
 *        - ceil(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L817
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol ceil(const std::string& symbol_name,
                   Symbol data) {
  return Operator("ceil")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise floor of the input.
 *
 *        The floor of the scalar x is the largest integer i, such that i <= x.
 *
 *        Example::
 *
 *        floor([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-3., -2.,  1.,  1.,  2.]
 *
 *        The storage type of ``floor`` output depends upon the input storage type:
 *
 *        - floor(default) = default
 *        - floor(row_sparse) = row_sparse
 *        - floor(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L836
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol floor(const std::string& symbol_name,
                    Symbol data) {
  return Operator("floor")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Return the element-wise truncated value of the input.
 *
 *        The truncated value of the scalar x is the nearest integer i which is closer to
 *        zero than x is. In short, the fractional part of the signed number x is
 *
 *        Example::
 *
 *        trunc([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  1.,  1.,  2.]
 *
 *        The storage type of ``trunc`` output depends upon the input storage type:
 *
 *        - trunc(default) = default
 *        - trunc(row_sparse) = row_sparse
 *        - trunc(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L856
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol trunc(const std::string& symbol_name,
                    Symbol data) {
  return Operator("trunc")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise rounded value to the nearest \
 *        integer towards zero of the input.
 *
 *        Example::
 *
 *        fix([-2.1, -1.9, 1.9, 2.1]) = [-2., -1.,  1., 2.]
 *
 *        The storage type of ``fix`` output depends upon the input storage type:
 *
 *        - fix(default) = default
 *        - fix(row_sparse) = row_sparse
 *        - fix(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L874
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol fix(const std::string& symbol_name,
                  Symbol data) {
  return Operator("fix")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise gauss error function of the input.
 *
 *        Example::
 *
 *        erf([0, -1., 10.]) = [0., -0.8427, 1.]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L886
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol erf(const std::string& symbol_name,
                  Symbol data) {
  return Operator("erf")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise inverse gauss error function of the input.
 *
 *        Example::
 *
 *        erfinv([0, 0.5., -1.]) = [0., 0.4769, -inf]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L908
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol erfinv(const std::string& symbol_name,
                     Symbol data) {
  return Operator("erfinv")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns the gamma function (extension of the factorial function \
 *        to the reals), computed element-wise on the input array.
 *
 *        The storage type of ``gamma`` output is always dense
 *
 *
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol gamma(const std::string& symbol_name,
                    Symbol data) {
  return Operator("gamma")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise log of the absolute value of the gamma function \
 *        of the input.
 *
 *        The storage type of ``gammaln`` output is always dense
 *
 *
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol gammaln(const std::string& symbol_name,
                      Symbol data) {
  return Operator("gammaln")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns the result of logical NOT (!) function
 *
 *        Example:
 *        logical_not([-2., 0., 1.]) = [0., 1., 0.]
 *
 *
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol logical_not(const std::string& symbol_name,
                          Symbol data) {
  return Operator("logical_not")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise exponential value of the input.
 *
 *        .. math::
 *        exp(x) = e^x \approx 2.718^x
 *
 *        Example::
 *
 *        exp([0, 1, 2]) = [1., 2.71828175, 7.38905621]
 *
 *        The storage type of ``exp`` output is always dense
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_logexp.cc:L64
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol exp(const std::string& symbol_name,
                  Symbol data) {
  return Operator("exp")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise Natural logarithmic value of the input.
 *
 *        The natural logarithm is logarithm in base *e*, so that ``log(exp(x)) = x``
 *
 *        The storage type of ``log`` output is always dense
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_logexp.cc:L77
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol log(const std::string& symbol_name,
                  Symbol data) {
  return Operator("log")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise Base-10 logarithmic value of the input.
 *
 *        ``10**log10(x) = x``
 *
 *        The storage type of ``log10`` output is always dense
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_logexp.cc:L94
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol log10(const std::string& symbol_name,
                    Symbol data) {
  return Operator("log10")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise Base-2 logarithmic value of the input.
 *
 *        ``2**log2(x) = x``
 *
 *        The storage type of ``log2`` output is always dense
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_logexp.cc:L106
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol log2(const std::string& symbol_name,
                   Symbol data) {
  return Operator("log2")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise ``log(1 + x)`` value of the input.
 *
 *        This function is more accurate than ``log(1 + x)``  for small ``x`` so that
 *        :math:`1+x\approx 1`
 *
 *        The storage type of ``log1p`` output depends upon the input storage type:
 *
 *        - log1p(default) = default
 *        - log1p(row_sparse) = row_sparse
 *        - log1p(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_logexp.cc:L199
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol log1p(const std::string& symbol_name,
                    Symbol data) {
  return Operator("log1p")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns ``exp(x) - 1`` computed element-wise on the input.
 *
 *        This function provides greater precision than ``exp(x) - 1`` for small values
 *
 *        The storage type of ``expm1`` output depends upon the input storage type:
 *
 *        - expm1(default) = default
 *        - expm1(row_sparse) = row_sparse
 *        - expm1(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_logexp.cc:L244
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol expm1(const std::string& symbol_name,
                    Symbol data) {
  return Operator("expm1")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns the reciprocal of the argument, element-wise.
 *
 *        Calculates 1/x.
 *
 *        Example::
 *
 *        reciprocal([-2, 1, 3, 1.6, 0.2]) = [-0.5, 1.0, 0.33333334, 0.625, 5.0]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_pow.cc:L43
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol reciprocal(const std::string& symbol_name,
                         Symbol data) {
  return Operator("reciprocal")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise squared value of the input.
 *
 *        .. math::
 *        square(x) = x^2
 *
 *        Example::
 *
 *        square([2, 3, 4]) = [4, 9, 16]
 *
 *        The storage type of ``square`` output depends upon the input storage type:
 *
 *        - square(default) = default
 *        - square(row_sparse) = row_sparse
 *        - square(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_pow.cc:L119
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol square(const std::string& symbol_name,
                     Symbol data) {
  return Operator("square")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise square-root value of the input.
 *
 *        .. math::
 *        \textrm{sqrt}(x) = \sqrt{x}
 *
 *        Example::
 *
 *        sqrt([4, 9, 16]) = [2, 3, 4]
 *
 *        The storage type of ``sqrt`` output depends upon the input storage type:
 *
 *        - sqrt(default) = default
 *        - sqrt(row_sparse) = row_sparse
 *        - sqrt(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_pow.cc:L170
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sqrt(const std::string& symbol_name,
                   Symbol data) {
  return Operator("sqrt")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise inverse square-root value of the input.
 *
 *        .. math::
 *        rsqrt(x) = 1/\sqrt{x}
 *
 *        Example::
 *
 *        rsqrt([4,9,16]) = [0.5, 0.33333334, 0.25]
 *
 *        The storage type of ``rsqrt`` output is always dense
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_pow.cc:L221
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol rsqrt(const std::string& symbol_name,
                    Symbol data) {
  return Operator("rsqrt")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise cube-root value of the input.
 *
 *        .. math::
 *        cbrt(x) = \sqrt[3]{x}
 *
 *        Example::
 *
 *        cbrt([1, 8, -125]) = [1, 2, -5]
 *
 *        The storage type of ``cbrt`` output depends upon the input storage type:
 *
 *        - cbrt(default) = default
 *        - cbrt(row_sparse) = row_sparse
 *        - cbrt(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_pow.cc:L270
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol cbrt(const std::string& symbol_name,
                   Symbol data) {
  return Operator("cbrt")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise inverse cube-root value of the input.
 *
 *        .. math::
 *        rcbrt(x) = 1/\sqrt[3]{x}
 *
 *        Example::
 *
 *        rcbrt([1,8,-125]) = [1.0, 0.5, -0.2]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_pow.cc:L323
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol rcbrt(const std::string& symbol_name,
                    Symbol data) {
  return Operator("rcbrt")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Computes the element-wise sine of the input array.
 *
 *        The input should be in radians (:math:`2\pi` rad equals 360 degrees).
 *
 *        .. math::
 *        sin([0, \pi/4, \pi/2]) = [0, 0.707, 1]
 *
 *        The storage type of ``sin`` output depends upon the input storage type:
 *
 *        - sin(default) = default
 *        - sin(row_sparse) = row_sparse
 *        - sin(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_trig.cc:L47
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sin(const std::string& symbol_name,
                  Symbol data) {
  return Operator("sin")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Computes the element-wise cosine of the input array.
 *
 *        The input should be in radians (:math:`2\pi` rad equals 360 degrees).
 *
 *        .. math::
 *        cos([0, \pi/4, \pi/2]) = [1, 0.707, 0]
 *
 *        The storage type of ``cos`` output is always dense
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_trig.cc:L90
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol cos(const std::string& symbol_name,
                  Symbol data) {
  return Operator("cos")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Computes the element-wise tangent of the input array.
 *
 *        The input should be in radians (:math:`2\pi` rad equals 360 degrees).
 *
 *        .. math::
 *        tan([0, \pi/4, \pi/2]) = [0, 1, -inf]
 *
 *        The storage type of ``tan`` output depends upon the input storage type:
 *
 *        - tan(default) = default
 *        - tan(row_sparse) = row_sparse
 *        - tan(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_trig.cc:L140
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol tan(const std::string& symbol_name,
                  Symbol data) {
  return Operator("tan")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise inverse sine of the input array.
 *
 *        The input should be in the range `[-1, 1]`.
 *        The output is in the closed interval of [:math:`-\pi/2`, :math:`\pi/2`].
 *
 *        .. math::
 *        arcsin([-1, -.707, 0, .707, 1]) = [-\pi/2, -\pi/4, 0, \pi/4, \pi/2]
 *
 *        The storage type of ``arcsin`` output depends upon the input storage type:
 *
 *        - arcsin(default) = default
 *        - arcsin(row_sparse) = row_sparse
 *        - arcsin(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_trig.cc:L187
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arcsin(const std::string& symbol_name,
                     Symbol data) {
  return Operator("arcsin")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise inverse cosine of the input array.
 *
 *        The input should be in range `[-1, 1]`.
 *        The output is in the closed interval :math:`[0, \pi]`
 *
 *        .. math::
 *        arccos([-1, -.707, 0, .707, 1]) = [\pi, 3\pi/4, \pi/2, \pi/4, 0]
 *
 *        The storage type of ``arccos`` output is always dense
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_trig.cc:L233
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arccos(const std::string& symbol_name,
                     Symbol data) {
  return Operator("arccos")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns element-wise inverse tangent of the input array.
 *
 *        The output is in the closed interval :math:`[-\pi/2, \pi/2]`
 *
 *        .. math::
 *        arctan([-1, 0, 1]) = [-\pi/4, 0, \pi/4]
 *
 *        The storage type of ``arctan`` output depends upon the input storage type:
 *
 *        - arctan(default) = default
 *        - arctan(row_sparse) = row_sparse
 *        - arctan(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_trig.cc:L282
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arctan(const std::string& symbol_name,
                     Symbol data) {
  return Operator("arctan")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Converts each element of the input array from radians to degrees.
 *
 *        .. math::
 *        degrees([0, \pi/2, \pi, 3\pi/2, 2\pi]) = [0, 90, 180, 270, 360]
 *
 *        The storage type of ``degrees`` output depends upon the input storage type:
 *
 *        - degrees(default) = default
 *        - degrees(row_sparse) = row_sparse
 *        - degrees(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_trig.cc:L332
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol degrees(const std::string& symbol_name,
                      Symbol data) {
  return Operator("degrees")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Converts each element of the input array from degrees to radians.
 *
 *        .. math::
 *        radians([0, 90, 180, 270, 360]) = [0, \pi/2, \pi, 3\pi/2, 2\pi]
 *
 *        The storage type of ``radians`` output depends upon the input storage type:
 *
 *        - radians(default) = default
 *        - radians(row_sparse) = row_sparse
 *        - radians(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_trig.cc:L351
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol radians(const std::string& symbol_name,
                      Symbol data) {
  return Operator("radians")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns the hyperbolic sine of the input array, computed element-wise.
 *
 *        .. math::
 *        sinh(x) = 0.5\times(exp(x) - exp(-x))
 *
 *        The storage type of ``sinh`` output depends upon the input storage type:
 *
 *        - sinh(default) = default
 *        - sinh(row_sparse) = row_sparse
 *        - sinh(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_trig.cc:L371
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sinh(const std::string& symbol_name,
                   Symbol data) {
  return Operator("sinh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns the hyperbolic cosine  of the input array, computed element-wise.
 *
 *        .. math::
 *        cosh(x) = 0.5\times(exp(x) + exp(-x))
 *
 *        The storage type of ``cosh`` output is always dense
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_trig.cc:L409
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol cosh(const std::string& symbol_name,
                   Symbol data) {
  return Operator("cosh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns the hyperbolic tangent of the input array, computed element-wise.
 *
 *        .. math::
 *        tanh(x) = sinh(x) / cosh(x)
 *
 *        The storage type of ``tanh`` output depends upon the input storage type:
 *
 *        - tanh(default) = default
 *        - tanh(row_sparse) = row_sparse
 *        - tanh(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_trig.cc:L451
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol tanh(const std::string& symbol_name,
                   Symbol data) {
  return Operator("tanh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns the element-wise inverse hyperbolic sine of the input array, \
 *        computed element-wise.
 *
 *        The storage type of ``arcsinh`` output depends upon the input storage type:
 *
 *        - arcsinh(default) = default
 *        - arcsinh(row_sparse) = row_sparse
 *        - arcsinh(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_trig.cc:L494
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arcsinh(const std::string& symbol_name,
                      Symbol data) {
  return Operator("arcsinh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns the element-wise inverse hyperbolic cosine of the input array, \
 *        computed element-wise.
 *
 *        The storage type of ``arccosh`` output is always dense
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_trig.cc:L535
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arccosh(const std::string& symbol_name,
                      Symbol data) {
  return Operator("arccosh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns the element-wise inverse hyperbolic tangent of the input array, \
 *        computed element-wise.
 *
 *        The storage type of ``arctanh`` output depends upon the input storage type:
 *
 *        - arctanh(default) = default
 *        - arctanh(row_sparse) = row_sparse
 *        - arctanh(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_trig.cc:L579
 * \param symbol_name name of the resulting symbol
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arctanh(const std::string& symbol_name,
                      Symbol data) {
  return Operator("arctanh")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \brief Specify how out-of-bound indices bahave. Default is "clip". "clip" means clip
 *        to the range. So, if all indices mentioned are too large, they are replaced by
 *        the index that addresses the last element along an axis. "wrap" means to wrap
 */
enum class TakeMode {
  kClip = 0,
  kRaise = 1,
  kWrap = 2
};

/*!
 * \brief Takes elements from an input array along the given axis.
 *
 *        This function slices the input array along a particular axis with the provided
 *
 *        Given data tensor of rank r >= 1, and indices tensor of rank q, gather entries
 *        dimension of data (by default outer-most one as axis=0) indexed by indices, and
 *        in an output tensor of rank q + (r - 1).
 *
 *        Examples::
 *
 *        x = [4.  5.  6.]
 *
 *        // Trivial case, take the second element along the first axis.
 *
 *        take(x, [1]) = [ 5. ]
 *
 *        // The other trivial case, axis=-1, take the third element along the first axis
 *
 *        take(x, [3], axis=-1, mode='clip') = [ 6. ]
 *
 *        x = [[ 1.,  2.],
 *        [ 3.,  4.],
 *        [ 5.,  6.]]
 *
 *        // In this case we will get rows 0 and 1, then 1 and 2. Along axis 0
 *
 *        take(x, [[0,1],[1,2]]) = [[[ 1.,  2.],
 *        [ 3.,  4.]],
 *
 *        [[ 3.,  4.],
 *        [ 5.,  6.]]]
 *
 *        // In this case we will get rows 0 and 1, then 1 and 2 (calculated by wrapping
 *        // Along axis 1
 *
 *        take(x, [[0, 3], [-1, -2]], axis=1, mode='wrap') = [[[ 1.  2.]
 *        [ 2.  1.]]
 *
 *        [[ 3.  4.]
 *        [ 4.  3.]]
 *
 *        [[ 5.  6.]
 *        [ 6.  5.]]]
 *
 *        The storage type of ``take`` output depends upon the input storage type:
 *
 *        - take(default, default) = default
 *        - take(csr, default, axis=0) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/indexing_op.cc:L777
 * \param symbol_name name of the resulting symbol
 * \param a The input array.
 * \param indices The indices of the values to be extracted.
 * \param axis The axis of input array to be taken.For input tensor of rank r, it could
 * \param mode Specify how out-of-bound indices bahave. Default is "clip". "clip" means
 *        clip to the range. So, if all indices mentioned are too large, they are
 *        replaced by the index that addresses the last element along an axis. "wrap"
 * \return new symbol
 */
inline Symbol take(const std::string& symbol_name,
                   Symbol a,
                   Symbol indices,
                   int axis = 0,
                   TakeMode mode = TakeMode::kClip) {
  static const char *TakeModeValues[] = {
    "clip",
    "raise",
    "wrap"
  };
  return Operator("take")
           .SetParam("axis", axis)
           .SetParam("mode", TakeModeValues[int(mode)])
           .SetInput("a", a)
           .SetInput("indices", indices)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Takes elements from a data batch.
 *
 *        .. note::
 *        `batch_take` is deprecated. Use `pick` instead.
 *
 *        Given an input array of shape ``(d0, d1)`` and indices of shape ``(i0,)``, the
 *        an output array of shape ``(i0,)`` with::
 *
 *        output[i] = input[i, indices[i]]
 *
 *        Examples::
 *
 *        x = [[ 1.,  2.],
 *        [ 3.,  4.],
 *        [ 5.,  6.]]
 *
 *        // takes elements with specified indices
 *        batch_take(x, [0,1,0]) = [ 1.  4.  5.]
 *
 *
 *
 *        Defined in ../src/operator/tensor/indexing_op.cc:L836
 * \param symbol_name name of the resulting symbol
 * \param a The input array
 * \param indices The index array
 * \return new symbol
 */
inline Symbol batch_take(const std::string& symbol_name,
                         Symbol a,
                         Symbol indices) {
  return Operator("batch_take")
           .SetInput("a", a)
           .SetInput("indices", indices)
           .CreateSymbol(symbol_name);
}

/*! \brief DType of the output
 */
enum class One_hotDtype {
  kBfloat16 = 0,
  kFloat16 = 1,
  kFloat32 = 2,
  kFloat64 = 3,
  kInt32 = 4,
  kInt64 = 5,
  kInt8 = 6,
  kUint8 = 7
};

/*!
 * \brief Returns a one-hot array.
 *
 *        The locations represented by `indices` take value `on_value`, while all
 *        other locations take value `off_value`.
 *
 *        `one_hot` operation with `indices` of shape ``(i0, i1)`` and `depth`  of ``d``
 *        in an output array of shape ``(i0, i1, d)`` with::
 *
 *        output[i,j,:] = off_value
 *        output[i,j,indices[i,j]] = on_value
 *
 *        Examples::
 *
 *        one_hot([1,0,2,0], 3) = [[ 0.  1.  0.]
 *        [ 1.  0.  0.]
 *        [ 0.  0.  1.]
 *        [ 1.  0.  0.]]
 *
 *        one_hot([1,0,2,0], 3, on_value=8, off_value=1,
 *        dtype='int32') = [[1 8 1]
 *        [8 1 1]
 *        [1 1 8]
 *        [8 1 1]]
 *
 *        one_hot([[1,0],[1,0],[2,0]], 3) = [[[ 0.  1.  0.]
 *        [ 1.  0.  0.]]
 *
 *        [[ 0.  1.  0.]
 *        [ 1.  0.  0.]]
 *
 *        [[ 0.  0.  1.]
 *        [ 1.  0.  0.]]]
 *
 *
 *        Defined in ../src/operator/tensor/indexing_op.cc:L883
 * \param symbol_name name of the resulting symbol
 * \param indices array of locations where to set on_value
 * \param depth Depth of the one hot dimension.
 * \param on_value The value assigned to the locations represented by indices.
 * \param off_value The value assigned to the locations not represented by indices.
 * \param dtype DType of the output
 * \return new symbol
 */
inline Symbol one_hot(const std::string& symbol_name,
                      Symbol indices,
                      int depth,
                      double on_value = 1,
                      double off_value = 0,
                      One_hotDtype dtype = One_hotDtype::kFloat32) {
  static const char *One_hotDtypeValues[] = {
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "int32",
    "int64",
    "int8",
    "uint8"
  };
  return Operator("one_hot")
           .SetParam("depth", depth)
           .SetParam("on_value", on_value)
           .SetParam("off_value", off_value)
           .SetParam("dtype", One_hotDtypeValues[int(dtype)])
           .SetInput("indices", indices)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Gather elements or slices from `data` and store to a tensor whose
 *        shape is defined by `indices`.
 *
 *        Given `data` with shape `(X_0, X_1, ..., X_{N-1})` and indices with shape
 *        `(M, Y_0, ..., Y_{K-1})`, the output will have shape `(Y_0, ..., Y_{K-1}, X_M,
 *        where `M <= N`. If `M == N`, output shape will simply be `(Y_0, ..., Y_{K-1})`.
 *
 *        The elements in output is defined as follows::
 *
 *        output[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}] = data[indices[0, y_0, ...,
 *        ...,
 *        indices[M-1, y_0, ..., y_{K-1}],
 *        x_M, ..., x_{N-1}]
 *
 *        Examples::
 *
 *        data = [[0, 1], [2, 3]]
 *        indices = [[1, 1, 0], [0, 1, 0]]
 *        gather_nd(data, indices) = [2, 3, 0]
 *
 *        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
 *        indices = [[0, 1], [1, 0]]
 *        gather_nd(data, indices) = [[3, 4], [5, 6]]
 *
 *
 * \param symbol_name name of the resulting symbol
 * \param data data
 * \param indices indices
 * \return new symbol
 */
inline Symbol gather_nd(const std::string& symbol_name,
                        Symbol data,
                        Symbol indices) {
  return Operator("gather_nd")
           .SetInput("data", data)
           .SetInput("indices", indices)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Scatters data into a new tensor according to indices.
 *
 *        Given `data` with shape `(Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1})` and indices
 *        `(M, Y_0, ..., Y_{K-1})`, the output will have shape `(X_0, X_1, ..., X_{N-1})`,
 *        where `M <= N`. If `M == N`, data shape should simply be `(Y_0, ..., Y_{K-1})`.
 *
 *        The elements in output is defined as follows::
 *
 *        output[indices[0, y_0, ..., y_{K-1}],
 *        ...,
 *        indices[M-1, y_0, ..., y_{K-1}],
 *        x_M, ..., x_{N-1}] = data[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}]
 *
 *        all other entries in output are 0.
 *
 *        .. warning::
 *
 *        If the indices have duplicates, the result will be non-deterministic and
 *        the gradient of `scatter_nd` will not be correct!!
 *
 *
 *        Examples::
 *
 *        data = [2, 3, 0]
 *        indices = [[1, 1, 0], [0, 1, 0]]
 *        shape = (2, 2)
 *        scatter_nd(data, indices, shape) = [[0, 0], [2, 3]]
 *
 *        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
 *        indices = [[0, 1], [1, 1]]
 *        shape = (2, 2, 2, 2)
 *        scatter_nd(data, indices, shape) = [[[[0, 0],
 *        [0, 0]],
 *
 *        [[1, 2],
 *        [3, 4]]],
 *
 *        [[[0, 0],
 *        [0, 0]],
 *
 *        [[5, 6],
 *        [7, 8]]]]
 *
 *
 * \param symbol_name name of the resulting symbol
 * \param data data
 * \param indices indices
 * \param shape Shape of output.
 * \return new symbol
 */
inline Symbol scatter_nd(const std::string& symbol_name,
                         Symbol data,
                         Symbol indices,
                         Shape shape) {
  return Operator("scatter_nd")
           .SetParam("shape", shape)
           .SetInput("data", data)
           .SetInput("indices", indices)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Return an array of zeros with the same shape, type and storage type
 *        as the input array.
 *
 *        The storage type of ``zeros_like`` output depends on the storage type of the
 *
 *        - zeros_like(row_sparse) = row_sparse
 *        - zeros_like(csr) = csr
 *        - zeros_like(default) = default
 *
 *        Examples::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        zeros_like(x) = [[ 0.,  0.,  0.],
 *        [ 0.,  0.,  0.]]
 *
 *
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol zeros_like(const std::string& symbol_name,
                         Symbol data) {
  return Operator("zeros_like")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Return an array of ones with the same shape and type
 *        as the input array.
 *
 *        Examples::
 *
 *        x = [[ 0.,  0.,  0.],
 *        [ 0.,  0.,  0.]]
 *
 *        ones_like(x) = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *
 * \param symbol_name name of the resulting symbol
 * \param data The input
 * \return new symbol
 */
inline Symbol ones_like(const std::string& symbol_name,
                        Symbol data) {
  return Operator("ones_like")
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Reshapes the input array.
 *        .. note:: ``Reshape`` is deprecated, use ``reshape``
 *        Given an array and a shape, this function returns a copy of the array in the
 *        The shape is a tuple of integers such as (2,3,4). The size of the new shape
 *        Example::
 *        reshape([1,2,3,4], shape=(2,2)) = [[1,2], [3,4]]
 *        Some dimensions of the shape can take special values from the set {0, -1, -2,
 *        - ``0``  copy this dimension from the input to the output shape.
 *        Example::
 *        - input shape = (2,3,4), shape = (4,0,2), output shape = (4,3,2)
 *        - input shape = (2,3,4), shape = (2,0,0), output shape = (2,3,4)
 *        - ``-1`` infers the dimension of the output shape by using the remainder of the
 *        keeping the size of the new array same as that of the input array.
 *        At most one dimension of shape can be -1.
 *        Example::
 *        - input shape = (2,3,4), shape = (6,1,-1), output shape = (6,1,4)
 *        - input shape = (2,3,4), shape = (3,-1,8), output shape = (3,1,8)
 *        - input shape = (2,3,4), shape=(-1,), output shape = (24,)
 *        - ``-2`` copy all/remainder of the input dimensions to the output shape.
 *        Example::
 *        - input shape = (2,3,4), shape = (-2,), output shape = (2,3,4)
 *        - input shape = (2,3,4), shape = (2,-2), output shape = (2,3,4)
 *        - input shape = (2,3,4), shape = (-2,1,1), output shape = (2,3,4,1,1)
 *        - ``-3`` use the product of two consecutive dimensions of the input shape as
 *        Example::
 *        - input shape = (2,3,4), shape = (-3,4), output shape = (6,4)
 *        - input shape = (2,3,4,5), shape = (-3,-3), output shape = (6,20)
 *        - input shape = (2,3,4), shape = (0,-3), output shape = (2,12)
 *        - input shape = (2,3,4), shape = (-3,-2), output shape = (6,4)
 *        - ``-4`` split one dimension of the input into two dimensions passed subsequent
 *        Example::
 *        - input shape = (2,3,4), shape = (-4,1,2,-2), output shape =(1,2,3,4)
 *        - input shape = (2,3,4), shape = (2,-4,-1,3,-2), output shape = (2,1,3,4)
 *        If the argument `reverse` is set to 1, then the special values are inferred
 *        Example::
 *        - without reverse=1, for input shape = (10,5,4), shape = (-1,0), output shape
 *        - with reverse=1, output shape will be (50,4).
 *
 *
 *        Defined in ../src/operator/tensor/matrix_op.cc:L175
 * \param symbol_name name of the resulting symbol
 * \param data Input data to reshape.
 * \param shape The target shape
 * \param reverse If true then the special values are inferred from right to left
 * \param target_shape (Deprecated! Use ``shape`` instead.) Target new shape. One and
 * \param keep_highest (Deprecated! Use ``shape`` instead.) Whether keep the highest dim
 *        unchanged.If set to true, then the first dim in target_shape is ignored,and
 * \return new symbol
 */
inline Symbol Reshape(const std::string& symbol_name,
                      Symbol data,
                      Shape shape = Shape(),
                      bool reverse = false,
                      Shape target_shape = Shape(),
                      bool keep_highest = false) {
  return Operator("Reshape")
           .SetParam("shape", shape)
           .SetParam("reverse", reverse)
           .SetParam("target_shape", target_shape)
           .SetParam("keep_highest", keep_highest)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Permutes the dimensions of an array.
 *        Examples::
 *        x = [[ 1, 2],
 *        [ 3, 4]]
 *        transpose(x) = [[ 1.,  3.],
 *        [ 2.,  4.]]
 *        x = [[[ 1.,  2.],
 *        [ 3.,  4.]],
 *        [[ 5.,  6.],
 *        [ 7.,  8.]]]
 *        transpose(x) = [[[ 1.,  5.],
 *        [ 3.,  7.]],
 *        [[ 2.,  6.],
 *        [ 4.,  8.]]]
 *        transpose(x, axes=(1,0,2)) = [[[ 1.,  2.],
 *        [ 5.,  6.]],
 *        [[ 3.,  4.],
 *        [ 7.,  8.]]]
 *
 *
 *        Defined in ../src/operator/tensor/matrix_op.cc:L328
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param axes Target axis order. By default the axes will be inverted.
 * \return new symbol
 */
inline Symbol transpose(const std::string& symbol_name,
                        Symbol data,
                        Shape axes = Shape()) {
  return Operator("transpose")
           .SetParam("axes", axes)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Inserts a new axis of size 1 into the array shape
 *        For example, given ``x`` with shape ``(2,3,4)``, then ``expand_dims(x, axis=1)``
 *        will return a new array with shape ``(2,1,3,4)``.
 *
 *
 *        Defined in ../src/operator/tensor/matrix_op.cc:L395
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param axis Position where new axis is to be inserted. Suppose that the input
 *        `NDArray`'s dimension is `ndim`, the range of the inserted axis is `[-ndim,
 * \return new symbol
 */
inline Symbol expand_dims(const std::string& symbol_name,
                          Symbol data,
                          int axis) {
  return Operator("expand_dims")
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Slices a region of the array.
 *        .. note:: ``crop`` is deprecated. Use ``slice`` instead.
 *        This function returns a sliced array between the indices given
 *        by `begin` and `end` with the corresponding `step`.
 *        For an input array of ``shape=(d_0, d_1, ..., d_n-1)``,
 *        slice operation with ``begin=(b_0, b_1...b_m-1)``,
 *        ``end=(e_0, e_1, ..., e_m-1)``, and ``step=(s_0, s_1, ..., s_m-1)``,
 *        where m <= n, results in an array with the shape
 *        ``(|e_0-b_0|/|s_0|, ..., |e_m-1-b_m-1|/|s_m-1|, d_m, ..., d_n-1)``.
 *        The resulting array's *k*-th dimension contains elements
 *        from the *k*-th dimension of the input array starting
 *        from index ``b_k`` (inclusive) with step ``s_k``
 *        until reaching ``e_k`` (exclusive).
 *        If the *k*-th elements are `None` in the sequence of `begin`, `end`,
 *        and `step`, the following rule will be used to set default values.
 *        If `s_k` is `None`, set `s_k=1`. If `s_k > 0`, set `b_k=0`, `e_k=d_k`;
 *        else, set `b_k=d_k-1`, `e_k=-1`.
 *        The storage type of ``slice`` output depends on storage types of inputs
 *        - slice(csr) = csr
 *        - otherwise, ``slice`` generates output with default storage
 *        .. note:: When input data storage type is csr, it only supports
 *        step=(), or step=(None,), or step=(1,) to generate a csr output.
 *        For other step parameter values, it falls back to slicing
 *        a dense tensor.
 *        Example::
 *        x = [[  1.,   2.,   3.,   4.],
 *        [  5.,   6.,   7.,   8.],
 *        [  9.,  10.,  11.,  12.]]
 *        slice(x, begin=(0,1), end=(2,4)) = [[ 2.,  3.,  4.],
 *        [ 6.,  7.,  8.]]
 *        slice(x, begin=(None, 0), end=(None, 3), step=(-1, 2)) = [[9., 11.],
 *        [5.,  7.],
 *        [1.,  3.]]
 *
 *
 *        Defined in ../src/operator/tensor/matrix_op.cc:L482
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param begin starting indices for the slice operation, supports negative indices.
 * \param end ending indices for the slice operation, supports negative indices.
 * \param step step for the slice operation, supports negative values.
 * \return new symbol
 */
inline Symbol slice(const std::string& symbol_name,
                    Symbol data,
                    Shape begin,
                    Shape end,
                    Shape step = Shape()) {
  return Operator("slice")
           .SetParam("begin", begin)
           .SetParam("end", end)
           .SetParam("step", step)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Slices along a given axis.
 *        Returns an array slice along a given `axis` starting from the `begin` index
 *        to the `end` index.
 *        Examples::
 *        x = [[  1.,   2.,   3.,   4.],
 *        [  5.,   6.,   7.,   8.],
 *        [  9.,  10.,  11.,  12.]]
 *        slice_axis(x, axis=0, begin=1, end=3) = [[  5.,   6.,   7.,   8.],
 *        [  9.,  10.,  11.,  12.]]
 *        slice_axis(x, axis=1, begin=0, end=2) = [[  1.,   2.],
 *        [  5.,   6.],
 *        [  9.,  10.]]
 *        slice_axis(x, axis=1, begin=-3, end=-1) = [[  2.,   3.],
 *        [  6.,   7.],
 *        [ 10.,  11.]]
 *
 *
 *        Defined in ../src/operator/tensor/matrix_op.cc:L571
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param axis Axis along which to be sliced, supports negative indexes.
 * \param begin The beginning index along the axis to be sliced,  supports negative
 * \param end The ending index along the axis to be sliced,  supports negative indexes.
 * \return new symbol
 */
inline Symbol slice_axis(const std::string& symbol_name,
                         Symbol data,
                         int axis,
                         int begin,
                         dmlc::optional<int> end) {
  return Operator("slice_axis")
           .SetParam("axis", axis)
           .SetParam("begin", begin)
           .SetParam("end", end)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Slices a region of the array like the shape of another array.
 *        This function is similar to ``slice``, however, the `begin` are always `0`s
 *        and `end` of specific axes are inferred from the second input `shape_like`.
 *        Given the second `shape_like` input of ``shape=(d_0, d_1, ..., d_n-1)``,
 *        a ``slice_like`` operator with default empty `axes`, it performs the
 *        following operation:
 *        `` out = slice(input, begin=(0, 0, ..., 0), end=(d_0, d_1, ..., d_n-1))``.
 *        When `axes` is not empty, it is used to speficy which axes are being sliced.
 *        Given a 4-d input data, ``slice_like`` operator with ``axes=(0, 2, -1)``
 *        will perform the following operation:
 *        `` out = slice(input, begin=(0, 0, 0, 0), end=(d_0, None, d_2, d_3))``.
 *        Note that it is allowed to have first and second input with different
 *        however, you have to make sure the `axes` are specified and not exceeding the
 *        dimension limits.
 *        For example, given `input_1` with ``shape=(2,3,4,5)`` and `input_2` with
 *        ``shape=(1,2,3)``, it is not allowed to use:
 *        `` out = slice_like(a, b)`` because ndim of `input_1` is 4, and ndim of
 *        is 3.
 *        The following is allowed in this situation:
 *        `` out = slice_like(a, b, axes=(0, 2))``
 *        Example::
 *        x = [[  1.,   2.,   3.,   4.],
 *        [  5.,   6.,   7.,   8.],
 *        [  9.,  10.,  11.,  12.]]
 *        y = [[  0.,   0.,   0.],
 *        [  0.,   0.,   0.]]
 *        slice_like(x, y) = [[ 1.,  2.,  3.]
 *        [ 5.,  6.,  7.]]
 *        slice_like(x, y, axes=(0, 1)) = [[ 1.,  2.,  3.]
 *        [ 5.,  6.,  7.]]
 *        slice_like(x, y, axes=(0)) = [[ 1.,  2.,  3.,  4.]
 *        [ 5.,  6.,  7.,  8.]]
 *        slice_like(x, y, axes=(-1)) = [[  1.,   2.,   3.]
 *        [  5.,   6.,   7.]
 *        [  9.,  10.,  11.]]
 *
 *
 *        Defined in ../src/operator/tensor/matrix_op.cc:L625
 * \param symbol_name name of the resulting symbol
 * \param data Source input
 * \param shape_like Shape like input
 * \param axes List of axes on which input data will be sliced according to the
 *        corresponding size of the second input. By default will slice on all axes.
 * \return new symbol
 */
inline Symbol slice_like(const std::string& symbol_name,
                         Symbol data,
                         Symbol shape_like,
                         Shape axes = Shape()) {
  return Operator("slice_like")
           .SetParam("axes", axes)
           .SetInput("data", data)
           .SetInput("shape_like", shape_like)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Clips (limits) the values in an array.
 *        Given an interval, values outside the interval are clipped to the interval
 *        Clipping ``x`` between `a_min` and `a_max` would be::
 *        .. math::
 *        clip(x, a_min, a_max) = \max(\min(x, a_max), a_min))
 *        Example::
 *        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
 *        clip(x,1,8) = [ 1.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  8.]
 *        The storage type of ``clip`` output depends on storage types of inputs and the
 *        parameter values:
 *        - clip(default) = default
 *        - clip(row_sparse, a_min <= 0, a_max >= 0) = row_sparse
 *        - clip(csr, a_min <= 0, a_max >= 0) = csr
 *        - clip(row_sparse, a_min < 0, a_max < 0) = default
 *        - clip(row_sparse, a_min > 0, a_max > 0) = default
 *        - clip(csr, a_min < 0, a_max < 0) = csr
 *        - clip(csr, a_min > 0, a_max > 0) = csr
 *
 *
 *        Defined in ../src/operator/tensor/matrix_op.cc:L677
 * \param symbol_name name of the resulting symbol
 * \param data Input array.
 * \param a_min Minimum value
 * \param a_max Maximum value
 * \return new symbol
 */
inline Symbol clip(const std::string& symbol_name,
                   Symbol data,
                   mx_float a_min,
                   mx_float a_max) {
  return Operator("clip")
           .SetParam("a_min", a_min)
           .SetParam("a_max", a_max)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Repeats elements of an array.
 *        By default, ``repeat`` flattens the input array into 1-D and then repeats the
 *        elements::
 *        x = [[ 1, 2],
 *        [ 3, 4]]
 *        repeat(x, repeats=2) = [ 1.,  1.,  2.,  2.,  3.,  3.,  4.,  4.]
 *        The parameter ``axis`` specifies the axis along which to perform repeat::
 *        repeat(x, repeats=2, axis=1) = [[ 1.,  1.,  2.,  2.],
 *        [ 3.,  3.,  4.,  4.]]
 *        repeat(x, repeats=2, axis=0) = [[ 1.,  2.],
 *        [ 1.,  2.],
 *        [ 3.,  4.],
 *        [ 3.,  4.]]
 *        repeat(x, repeats=2, axis=-1) = [[ 1.,  1.,  2.,  2.],
 *        [ 3.,  3.,  4.,  4.]]
 *
 *
 *        Defined in ../src/operator/tensor/matrix_op.cc:L744
 * \param symbol_name name of the resulting symbol
 * \param data Input data array
 * \param repeats The number of repetitions for each element.
 * \param axis The axis along which to repeat values. The negative numbers are
 *        interpreted counting from the backward. By default, use the flattened input
 * \return new symbol
 */
inline Symbol repeat(const std::string& symbol_name,
                     Symbol data,
                     int repeats,
                     dmlc::optional<int> axis = dmlc::optional<int>()) {
  return Operator("repeat")
           .SetParam("repeats", repeats)
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Repeats the whole array multiple times.
 *        If ``reps`` has length *d*, and input array has dimension of *n*. There are
 *        three cases:
 *        - **n=d**. Repeat *i*-th dimension of the input by ``reps[i]`` times::
 *        x = [[1, 2],
 *        [3, 4]]
 *        tile(x, reps=(2,3)) = [[ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.],
 *        [ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.]]
 *        - **n>d**. ``reps`` is promoted to length *n* by pre-pending 1's to it. Thus for
 *        an input shape ``(2,3)``, ``repos=(2,)`` is treated as ``(1,2)``::
 *        tile(x, reps=(2,)) = [[ 1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.]]
 *        - **n<d**. The input is promoted to be d-dimensional by prepending new axes. So
 *        shape ``(2,2)`` array is promoted to ``(1,2,2)`` for 3-D replication::
 *        tile(x, reps=(2,2,3)) = [[[ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.],
 *        [ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.]],
 *        [[ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.],
 *        [ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.]]]
 *
 *
 *        Defined in ../src/operator/tensor/matrix_op.cc:L796
 * \param symbol_name name of the resulting symbol
 * \param data Input data array
 * \param reps The number of times for repeating the tensor a. Each dim size of reps must
 *        be a positive integer. If reps has length d, the result will have dimension of
 *        max(d, a.ndim); If a.ndim < d, a is promoted to be d-dimensional by prepending
 * \return new symbol
 */
inline Symbol tile(const std::string& symbol_name,
                   Symbol data,
                   Shape reps) {
  return Operator("tile")
           .SetParam("reps", reps)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Reverses the order of elements along given axis while preserving array shape.
 *        Note: reverse and flip are equivalent. We use reverse in the following examples.
 *        Examples::
 *        x = [[ 0.,  1.,  2.,  3.,  4.],
 *        [ 5.,  6.,  7.,  8.,  9.]]
 *        reverse(x, axis=0) = [[ 5.,  6.,  7.,  8.,  9.],
 *        [ 0.,  1.,  2.,  3.,  4.]]
 *        reverse(x, axis=1) = [[ 4.,  3.,  2.,  1.,  0.],
 *        [ 9.,  8.,  7.,  6.,  5.]]
 *
 *
 *        Defined in ../src/operator/tensor/matrix_op.cc:L832
 * \param symbol_name name of the resulting symbol
 * \param data Input data array
 * \param axis The axis which to reverse elements.
 * \return new symbol
 */
inline Symbol reverse(const std::string& symbol_name,
                      Symbol data,
                      Shape axis) {
  return Operator("reverse")
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Join a sequence of arrays along a new axis.
 *        The axis parameter specifies the index of the new axis in the dimensions of the
 *        result. For example, if axis=0 it will be the first dimension and if axis=-1 it
 *        will be the last dimension.
 *        Examples::
 *        x = [1, 2]
 *        y = [3, 4]
 *        stack(x, y) = [[1, 2],
 *        [3, 4]]
 *        stack(x, y, axis=1) = [[1, 3],
 *        [2, 4]]
 *
 * \param symbol_name name of the resulting symbol
 * \param data List of arrays to stack
 * \param num_args Number of inputs to be stacked.
 * \param axis The axis in the result array along which the input arrays are stacked.
 * \return new symbol
 */
inline Symbol stack(const std::string& symbol_name,
                    const std::vector<Symbol>& data,
                    int num_args,
                    int axis = 0) {
  return Operator("stack")
           .SetParam("num_args", num_args)
           .SetParam("axis", axis)
(data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Remove single-dimensional entries from the shape of an array.
 *        Same behavior of defining the output tensor shape as numpy.squeeze for the most
 *        See the following note for exception.
 *        Examples::
 *        data = [[[0], [1], [2]]]
 *        squeeze(data) = [0, 1, 2]
 *        squeeze(data, axis=0) = [[0], [1], [2]]
 *        squeeze(data, axis=2) = [[0, 1, 2]]
 *        squeeze(data, axis=(0, 2)) = [0, 1, 2]
 *        .. Note::
 *        The output of this operator will keep at least one dimension not removed. For
 *        squeeze([[[4]]]) = [4], while in numpy.squeeze, the output will become a scalar.
 *
 * \param symbol_name name of the resulting symbol
 * \param data data to squeeze
 * \param axis Selects a subset of the single-dimensional entries in the shape. If an
 * \return new symbol
 */
inline Symbol squeeze(const std::string& symbol_name,
                      Symbol data,
                      dmlc::optional<Shape> axis = dmlc::optional<Shape>()) {
  return Operator("squeeze")
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Rearranges(permutes) data from depth into blocks of spatial data.
 *        Similar to ONNX DepthToSpace operator:
 *        https://github.com/onnx/onnx/blob/master/docs/Operators.md#DepthToSpace.
 *        The output is a new tensor where the values from depth dimension are moved in
 *        to height and width dimension. The reverse of this operation is
 *        .. math::
 *        \begin{gather*}
 *        x \prime = reshape(x, [N, block\_size, block\_size, C / (block\_size ^ 2), H *
 *        x \prime \prime = transpose(x \prime, [0, 3, 4, 1, 5, 2]) \\
 *        y = reshape(x \prime \prime, [N, C / (block\_size ^ 2), H * block\_size, W *
 *        \end{gather*}
 *        where :math:`x` is an input tensor with default layout as :math:`[N, C, H, W]`:
 *        and :math:`y` is the output tensor of layout :math:`[N, C / (block\_size ^ 2),
 *        Example::
 *        x = [[[[0, 1, 2],
 *        [3, 4, 5]],
 *        [[6, 7, 8],
 *        [9, 10, 11]],
 *        [[12, 13, 14],
 *        [15, 16, 17]],
 *        [[18, 19, 20],
 *        [21, 22, 23]]]]
 *        depth_to_space(x, 2) = [[[[0, 6, 1, 7, 2, 8],
 *        [12, 18, 13, 19, 14, 20],
 *        [3, 9, 4, 10, 5, 11],
 *        [15, 21, 16, 22, 17, 23]]]]
 *
 *
 *        Defined in ../src/operator/tensor/matrix_op.cc:L972
 * \param symbol_name name of the resulting symbol
 * \param data Input ndarray
 * \param block_size Blocks of [block_size. block_size] are moved
 * \return new symbol
 */
inline Symbol depth_to_space(const std::string& symbol_name,
                             Symbol data,
                             int block_size) {
  return Operator("depth_to_space")
           .SetParam("block_size", block_size)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Rearranges(permutes) blocks of spatial data into depth.
 *        Similar to ONNX SpaceToDepth operator:
 *        https://github.com/onnx/onnx/blob/master/docs/Operators.md#SpaceToDepth
 *        The output is a new tensor where the values from height and width dimension are
 *        moved to the depth dimension. The reverse of this operation is
 *        .. math::
 *        \begin{gather*}
 *        x \prime = reshape(x, [N, C, H / block\_size, block\_size, W / block\_size,
 *        x \prime \prime = transpose(x \prime, [0, 3, 5, 1, 2, 4]) \\
 *        y = reshape(x \prime \prime, [N, C * (block\_size ^ 2), H / block\_size, W /
 *        \end{gather*}
 *        where :math:`x` is an input tensor with default layout as :math:`[N, C, H, W]`:
 *        and :math:`y` is the output tensor of layout :math:`[N, C * (block\_size ^ 2),
 *        Example::
 *        x = [[[[0, 6, 1, 7, 2, 8],
 *        [12, 18, 13, 19, 14, 20],
 *        [3, 9, 4, 10, 5, 11],
 *        [15, 21, 16, 22, 17, 23]]]]
 *        space_to_depth(x, 2) = [[[[0, 1, 2],
 *        [3, 4, 5]],
 *        [[6, 7, 8],
 *        [9, 10, 11]],
 *        [[12, 13, 14],
 *        [15, 16, 17]],
 *        [[18, 19, 20],
 *        [21, 22, 23]]]]
 *
 *
 *        Defined in ../src/operator/tensor/matrix_op.cc:L1019
 * \param symbol_name name of the resulting symbol
 * \param data Input ndarray
 * \param block_size Blocks of [block_size. block_size] are moved
 * \return new symbol
 */
inline Symbol space_to_depth(const std::string& symbol_name,
                             Symbol data,
                             int block_size) {
  return Operator("space_to_depth")
           .SetParam("block_size", block_size)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \brief The return type.
 *        "value" means to return the top k values, "indices" means to return the indices
 *        of the top k values, "mask" means to return a mask array containing 0 and 1. 1
 *        means the top k values. "both" means to return a list of both values and
 */
enum class TopkRetTyp {
  kBoth = 0,
  kIndices = 1,
  kMask = 2,
  kValue = 3
};

/*! \brief DType of the output indices when ret_typ is "indices" or "both". An error will
 */
enum class TopkDtype {
  kFloat16 = 0,
  kFloat32 = 1,
  kFloat64 = 2,
  kInt32 = 3,
  kInt64 = 4,
  kUint8 = 5
};

/*!
 * \brief Returns the indices of the top *k* elements in an input array along the given
 *        axis (by default).
 *        If ret_type is set to 'value' returns the value of top *k* elements (instead of
 *        In case of ret_type = 'both', both value and index would be returned.
 *        The returned elements will be sorted.
 *
 *        Examples::
 *
 *        x = [[ 0.3,  0.2,  0.4],
 *        [ 0.1,  0.3,  0.2]]
 *
 *        // returns an index of the largest element on last axis
 *        topk(x) = [[ 2.],
 *        [ 1.]]
 *
 *        // returns the value of top-2 largest elements on last axis
 *        topk(x, ret_typ='value', k=2) = [[ 0.4,  0.3],
 *        [ 0.3,  0.2]]
 *
 *        // returns the value of top-2 smallest elements on last axis
 *        topk(x, ret_typ='value', k=2, is_ascend=1) = [[ 0.2 ,  0.3],
 *        [ 0.1 ,  0.2]]
 *
 *        // returns the value of top-2 largest elements on axis 0
 *        topk(x, axis=0, ret_typ='value', k=2) = [[ 0.3,  0.3,  0.4],
 *        [ 0.1,  0.2,  0.2]]
 *
 *        // flattens and then returns list of both values and indices
 *        topk(x, ret_typ='both', k=2) = [[[ 0.4,  0.3], [ 0.3,  0.2]] ,  [[ 2.,  0.], [
 *
 *
 *
 *        Defined in ../src/operator/tensor/ordering_op.cc:L68
 * \param symbol_name name of the resulting symbol
 * \param data The input array
 * \param axis Axis along which to choose the top k indices. If not given, the flattened
 * \param k Number of top elements to select, should be always smaller than or equal to
 * \param ret_typ The return type.
 *        "value" means to return the top k values, "indices" means to return the indices
 *        of the top k values, "mask" means to return a mask array containing 0 and 1. 1
 *        means the top k values. "both" means to return a list of both values and
 * \param is_ascend Whether to choose k largest or k smallest elements. Top K largest
 * \param dtype DType of the output indices when ret_typ is "indices" or "both". An error
 * \return new symbol
 */
inline Symbol topk(const std::string& symbol_name,
                   Symbol data,
                   dmlc::optional<int> axis = dmlc::optional<int>(-1),
                   int k = 1,
                   TopkRetTyp ret_typ = TopkRetTyp::kIndices,
                   bool is_ascend = false,
                   TopkDtype dtype = TopkDtype::kFloat32) {
  static const char *TopkRetTypValues[] = {
    "both",
    "indices",
    "mask",
    "value"
  };
  static const char *TopkDtypeValues[] = {
    "float16",
    "float32",
    "float64",
    "int32",
    "int64",
    "uint8"
  };
  return Operator("topk")
           .SetParam("axis", axis)
           .SetParam("k", k)
           .SetParam("ret_typ", TopkRetTypValues[int(ret_typ)])
           .SetParam("is_ascend", is_ascend)
           .SetParam("dtype", TopkDtypeValues[int(dtype)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Returns a sorted copy of an input array along the given axis.
 *
 *        Examples::
 *
 *        x = [[ 1, 4],
 *        [ 3, 1]]
 *
 *        // sorts along the last axis
 *        sort(x) = [[ 1.,  4.],
 *        [ 1.,  3.]]
 *
 *        // flattens and then sorts
 *        sort(x, axis=None) = [ 1.,  1.,  3.,  4.]
 *
 *        // sorts along the first axis
 *        sort(x, axis=0) = [[ 1.,  1.],
 *        [ 3.,  4.]]
 *
 *        // in a descend order
 *        sort(x, is_ascend=0) = [[ 4.,  1.],
 *        [ 3.,  1.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/ordering_op.cc:L133
 * \param symbol_name name of the resulting symbol
 * \param data The input array
 * \param axis Axis along which to choose sort the input tensor. If not given, the
 * \param is_ascend Whether to sort in ascending or descending order.
 * \return new symbol
 */
inline Symbol sort(const std::string& symbol_name,
                   Symbol data,
                   dmlc::optional<int> axis = dmlc::optional<int>(-1),
                   bool is_ascend = true) {
  return Operator("sort")
           .SetParam("axis", axis)
           .SetParam("is_ascend", is_ascend)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \brief DType of the output indices. It is only valid when ret_typ is "indices" or
 *        "both". An error will be raised if the selected data type cannot precisely
 */
enum class ArgsortDtype {
  kFloat16 = 0,
  kFloat32 = 1,
  kFloat64 = 2,
  kInt32 = 3,
  kInt64 = 4,
  kUint8 = 5
};

/*!
 * \brief Returns the indices that would sort an input array along the given axis.
 *
 *        This function performs sorting along the given axis and returns an array of
 *        as an input array that index data in sorted order.
 *
 *        Examples::
 *
 *        x = [[ 0.3,  0.2,  0.4],
 *        [ 0.1,  0.3,  0.2]]
 *
 *        // sort along axis -1
 *        argsort(x) = [[ 1.,  0.,  2.],
 *        [ 0.,  2.,  1.]]
 *
 *        // sort along axis 0
 *        argsort(x, axis=0) = [[ 1.,  0.,  1.]
 *        [ 0.,  1.,  0.]]
 *
 *        // flatten and then sort
 *        argsort(x, axis=None) = [ 3.,  1.,  5.,  0.,  4.,  2.]
 *
 *
 *        Defined in ../src/operator/tensor/ordering_op.cc:L185
 * \param symbol_name name of the resulting symbol
 * \param data The input array
 * \param axis Axis along which to sort the input tensor. If not given, the flattened
 * \param is_ascend Whether to sort in ascending or descending order.
 * \param dtype DType of the output indices. It is only valid when ret_typ is "indices"
 *        or "both". An error will be raised if the selected data type cannot precisely
 * \return new symbol
 */
inline Symbol argsort(const std::string& symbol_name,
                      Symbol data,
                      dmlc::optional<int> axis = dmlc::optional<int>(-1),
                      bool is_ascend = true,
                      ArgsortDtype dtype = ArgsortDtype::kFloat32) {
  static const char *ArgsortDtypeValues[] = {
    "float16",
    "float32",
    "float64",
    "int32",
    "int64",
    "uint8"
  };
  return Operator("argsort")
           .SetParam("axis", axis)
           .SetParam("is_ascend", is_ascend)
           .SetParam("dtype", ArgsortDtypeValues[int(dtype)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Applies bilinear sampling to input feature map.
 *
 *        Bilinear Sampling is the key of  [NIPS2015] \"Spatial Transformer Networks\".
 *        except that the operator has the backward pass.
 *
 *        Given :math:`data` and :math:`grid`, then the output is computed by
 *
 *        .. math::
 *        x_{src} = grid[batch, 0, y_{dst}, x_{dst}] \\
 *        y_{src} = grid[batch, 1, y_{dst}, x_{dst}] \\
 *        output[batch, channel, y_{dst}, x_{dst}] = G(data[batch, channel, y_{src},
 *
 *        :math:`x_{dst}`, :math:`y_{dst}` enumerate all spatial locations in
 *        The out-boundary points will be padded with zeros.The shape of the output will
 *
 *        The operator assumes that :math:`data` has 'NCHW' layout and :math:`grid` has
 *
 *        BilinearSampler often cooperates with GridGenerator which generates sampling
 *        GridGenerator supports two kinds of transformation: ``affine`` and ``warp``.
 *        If users want to design a CustomOp to manipulate :math:`grid`, please firstly
 *
 *        Example 1::
 *
 *        ## Zoom out data two times
 *        data = array([[[[1, 4, 3, 6],
 *        [1, 8, 8, 9],
 *        [0, 4, 1, 5],
 *        [1, 0, 1, 3]]]])
 *
 *        affine_matrix = array([[2, 0, 0],
 *        [0, 2, 0]])
 *
 *        affine_matrix = reshape(affine_matrix, shape=(1, 6))
 *
 *        grid = GridGenerator(data=affine_matrix, transform_type='affine',
 *
 *        out = BilinearSampler(data, grid)
 *
 *        out
 *        [[[[ 0,   0,     0,   0],
 *        [ 0,   3.5,   6.5, 0],
 *        [ 0,   1.25,  2.5, 0],
 *        [ 0,   0,     0,   0]]]
 *
 *
 *        Example 2::
 *
 *        ## shift data horizontally by -1 pixel
 *
 *        data = array([[[[1, 4, 3, 6],
 *        [1, 8, 8, 9],
 *        [0, 4, 1, 5],
 *        [1, 0, 1, 3]]]])
 *
 *        warp_maxtrix = array([[[[1, 1, 1, 1],
 *        [1, 1, 1, 1],
 *        [1, 1, 1, 1],
 *        [1, 1, 1, 1]],
 *        [[0, 0, 0, 0],
 *        [0, 0, 0, 0],
 *        [0, 0, 0, 0],
 *        [0, 0, 0, 0]]]])
 *
 *        grid = GridGenerator(data=warp_matrix, transform_type='warp')
 *        out = BilinearSampler(data, grid)
 *
 *        out
 *        [[[[ 4,  3,  6,  0],
 *        [ 8,  8,  9,  0],
 *        [ 4,  1,  5,  0],
 *        [ 0,  1,  3,  0]]]
 *
 *
 *        Defined in ../src/operator/bilinear_sampler.cc:L256
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the BilinearsamplerOp.
 * \param grid Input grid to the BilinearsamplerOp.grid has two channels: x_src, y_src
 * \param cudnn_off whether to turn cudnn off
 * \return new symbol
 */
inline Symbol BilinearSampler(const std::string& symbol_name,
                              Symbol data,
                              Symbol grid,
                              dmlc::optional<bool> cudnn_off = dmlc::optional<bool>()) {
  return Operator("BilinearSampler")
           .SetParam("cudnn_off", cudnn_off)
           .SetInput("data", data)
           .SetInput("grid", grid)
           .CreateSymbol(symbol_name);
}

/*! \brief Whether to pick convolution algo by running performance test.
 *        Leads to higher startup time but may give faster speed. Options are:
 *        'off': no tuning
 *        'limited_workspace': run test and pick the fastest algorithm that doesn't
 *        'fastest': pick the fastest algorithm and ignore workspace limit.
 *        If set to None (default), behavior is determined by environment
 *        variable MXNET_CUDNN_AUTOTUNE_DEFAULT: 0 for off,
 *        1 for limited workspace (default), 2 for fastest.
 */
enum class Convolution_v1CudnnTune {
  kNone = 0,
  kFastest = 1,
  kLimited_workspace = 2,
  kOff = 3
};

/*! \brief Set layout for input, output and weight. Empty for
 *        default layout: NCHW for 2d and NCDHW for 3d.
 */
enum class Convolution_v1Layout {
  kNone = 0,
  kNCDHW = 1,
  kNCHW = 2,
  kNDHWC = 3,
  kNHWC = 4
};

/*!
 * \brief This operator is DEPRECATED. Apply convolution to input then add a bias.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the ConvolutionV1Op.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param kernel convolution kernel size: (h, w) or (d, h, w)
 * \param num_filter convolution filter(channel) number
 * \param stride convolution stride: (h, w) or (d, h, w)
 * \param dilate convolution dilate: (h, w) or (d, h, w)
 * \param pad pad for convolution: (h, w) or (d, h, w)
 * \param num_group Number of group partitions. Equivalent to slicing input into num_group
 *        partitions, apply convolution on each, then concatenate the results
 * \param workspace Maximum temporary workspace allowed for convolution (MB).This
 *        parameter determines the effective batch size of the convolution kernel, which
 *        may be smaller than the given batch size. Also, the workspace will be
 * \param no_bias Whether to disable bias parameter.
 * \param cudnn_tune Whether to pick convolution algo by running performance test.
 *        Leads to higher startup time but may give faster speed. Options are:
 *        'off': no tuning
 *        'limited_workspace': run test and pick the fastest algorithm that doesn't
 *        'fastest': pick the fastest algorithm and ignore workspace limit.
 *        If set to None (default), behavior is determined by environment
 *        variable MXNET_CUDNN_AUTOTUNE_DEFAULT: 0 for off,
 *        1 for limited workspace (default), 2 for fastest.
 * \param cudnn_off Turn off cudnn for this layer.
 * \param layout Set layout for input, output and weight. Empty for
 *        default layout: NCHW for 2d and NCDHW for 3d.
 * \return new symbol
 */
inline Symbol Convolution_v1(const std::string& symbol_name,
                             Symbol data,
                             Symbol weight,
                             Symbol bias,
                             Shape kernel,
                             uint32_t num_filter,
                             Shape stride = Shape(),
                             Shape dilate = Shape(),
                             Shape pad = Shape(),
                             uint32_t num_group = 1,
                             uint64_t workspace = 1024,
                             bool no_bias = false,
                             Convolution_v1CudnnTune cudnn_tune = Convolution_v1CudnnTune::kNone,
                             bool cudnn_off = false,
                             Convolution_v1Layout layout = Convolution_v1Layout::kNone) {
  static const char *Convolution_v1CudnnTuneValues[] = {
    "None",
    "fastest",
    "limited_workspace",
    "off"
  };
  static const char *Convolution_v1LayoutValues[] = {
    "None",
    "NCDHW",
    "NCHW",
    "NDHWC",
    "NHWC"
  };
  return Operator("Convolution_v1")
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("dilate", dilate)
           .SetParam("pad", pad)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
           .SetParam("cudnn_tune", Convolution_v1CudnnTuneValues[int(cudnn_tune)])
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("layout", Convolution_v1LayoutValues[int(layout)])
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Applies correlation to inputs.
 *
 *        The correlation layer performs multiplicative patch comparisons between two
 *
 *        Given two multi-channel feature maps :math:`f_{1}, f_{2}`, with :math:`w`,
 *        the correlation layer lets the network compare each patch from :math:`f_{1}`
 *
 *        For now we consider only a single comparison of two patches. The 'correlation'
 *        :math:`x_{2}` in the second map is then defined as:
 *
 *        .. math::
 *
 *        c(x_{1}, x_{2}) = \sum_{o \in [-k,k] \times [-k,k]} <f_{1}(x_{1} + o),
 *
 *        for a square patch of size :math:`K:=2k+1`.
 *
 *        Note that the equation above is identical to one step of a convolution in
 *        neural networks, but instead of convolving data with a filter, it convolves
 *        data. For this reason, it has no training weights.
 *
 *        Computing :math:`c(x_{1}, x_{2})` involves :math:`c * K^{2}` multiplications.
 *
 *        Given a maximum displacement :math:`d`, for each location :math:`x_{1}` it
 *        computes correlations :math:`c(x_{1}, x_{2})` only in a neighborhood of size
 *        by limiting the range of :math:`x_{2}`. We use strides :math:`s_{1}, s_{2}`, to
 *        quantize :math:`x_{1}` globally and to quantize :math:`x_{2}` within the
 *        centered around :math:`x_{1}`.
 *
 *        The final output is defined by the following expression:
 *
 *        .. math::
 *        out[n, q, i, j] = c(x_{i, j}, x_{q})
 *
 *        where :math:`i` and :math:`j` enumerate spatial locations in :math:`f_{1}`, and
 *
 *
 *        Defined in ../src/operator/correlation.cc:L198
 * \param symbol_name name of the resulting symbol
 * \param data1 Input data1 to the correlation.
 * \param data2 Input data2 to the correlation.
 * \param kernel_size kernel size for Correlation must be an odd number
 * \param max_displacement Max displacement of Correlation
 * \param stride1 stride1 quantize data1 globally
 * \param stride2 stride2 quantize data2 within the neighborhood centered around data1
 * \param pad_size pad for Correlation
 * \param is_multiply operation type is either multiplication or subduction
 * \return new symbol
 */
inline Symbol Correlation(const std::string& symbol_name,
                          Symbol data1,
                          Symbol data2,
                          uint32_t kernel_size = 1,
                          uint32_t max_displacement = 1,
                          uint32_t stride1 = 1,
                          uint32_t stride2 = 1,
                          uint32_t pad_size = 0,
                          bool is_multiply = true) {
  return Operator("Correlation")
           .SetParam("kernel_size", kernel_size)
           .SetParam("max_displacement", max_displacement)
           .SetParam("stride1", stride1)
           .SetParam("stride2", stride2)
           .SetParam("pad_size", pad_size)
           .SetParam("is_multiply", is_multiply)
           .SetInput("data1", data1)
           .SetInput("data2", data2)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief
 *
 *        .. note:: `Crop` is deprecated. Use `slice` instead.
 *
 *        Crop the 2nd and 3rd dim of input data, with the corresponding size of h_w or
 *        with width and height of the second input symbol, i.e., with one input, we need
 *        specify the crop height and width, otherwise the second input symbol's size
 *
 *
 *        Defined in ../src/operator/crop.cc:L50
 * \param symbol_name name of the resulting symbol
 * \param data Tensor or List of Tensors, the second input will be used as crop_like
 * \param num_args Number of inputs for crop, if equals one, then we will use the h_wfor
 *        crop height and width, else if equals two, then we will use the heightand width
 * \param offset crop offset coordinate: (y, x)
 * \param h_w crop height and width: (h, w)
 * \param center_crop If set to true, then it will use be the center_crop,or it will crop
 * \return new symbol
 */
inline Symbol Crop(const std::string& symbol_name,
                   const std::vector<Symbol>& data,
                   int num_args,
                   Shape offset = Shape(0,0),
                   Shape h_w = Shape(0,0),
                   bool center_crop = false) {
  return Operator("Crop")
           .SetParam("num_args", num_args)
           .SetParam("offset", offset)
           .SetParam("h_w", h_w)
           .SetParam("center_crop", center_crop)
(data)
           .CreateSymbol(symbol_name);
}

/*! \brief The type of transformation. For `affine`, input data should be an affine matrix
 *        of size (batch, 6). For `warp`, input data should be an optical flow of size
 */
enum class GridGeneratorTransformType {
  kAffine = 0,
  kWarp = 1
};

/*!
 * \brief Generates 2D sampling grid for bilinear sampling.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the function.
 * \param transform_type The type of transformation. For `affine`, input data should be
 *        an affine matrix of size (batch, 6). For `warp`, input data should be an
 * \param target_shape Specifies the output shape (H, W). This is required if
 *        transformation type is `affine`. If transformation type is `warp`, this
 * \return new symbol
 */
inline Symbol GridGenerator(const std::string& symbol_name,
                            Symbol data,
                            GridGeneratorTransformType transform_type,
                            Shape target_shape = Shape(0,0)) {
  static const char *GridGeneratorTransformTypeValues[] = {
    "affine",
    "warp"
  };
  return Operator("GridGenerator")
           .SetParam("transform_type", GridGeneratorTransformTypeValues[int(transform_type)])
           .SetParam("target_shape", target_shape)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Applies instance normalization to the n-dimensional input array.
 *
 *        This operator takes an n-dimensional input array where (n>2) and normalizes
 *        the input using the following formula:
 *
 *        .. math::
 *
 *        out = \frac{x - mean[data]}{ \sqrt{Var[data]} + \epsilon} * gamma + beta
 *
 *        This layer is similar to batch normalization layer (`BatchNorm`)
 *        with two differences: first, the normalization is
 *        carried out per example (instance), not over a batch. Second, the
 *        same normalization is applied both at test and train time. This
 *        operation is also known as `contrast normalization`.
 *
 *        If the input data is of shape [batch, channel, spacial_dim1, spacial_dim2, ...],
 *        `gamma` and `beta` parameters must be vectors of shape [channel].
 *
 *        This implementation is based on this paper [1]_
 *
 *        .. [1] Instance Normalization: The Missing Ingredient for Fast Stylization,
 *        D. Ulyanov, A. Vedaldi, V. Lempitsky, 2016 (arXiv:1607.08022v2).
 *
 *        Examples::
 *
 *        // Input of shape (2,1,2)
 *        x = [[[ 1.1,  2.2]],
 *        [[ 3.3,  4.4]]]
 *
 *        // gamma parameter of length 1
 *        gamma = [1.5]
 *
 *        // beta parameter of length 1
 *        beta = [0.5]
 *
 *        // Instance normalization is calculated with the above formula
 *        InstanceNorm(x,gamma,beta) = [[[-0.997527  ,  1.99752665]],
 *        [[-0.99752653,  1.99752724]]]
 *
 *
 *
 *        Defined in ../src/operator/instance_norm.cc:L95
 * \param symbol_name name of the resulting symbol
 * \param data An n-dimensional input array (n > 2) of the form [batch, channel,
 * \param gamma A vector of length 'channel', which multiplies the normalized input.
 * \param beta A vector of length 'channel', which is added to the product of the
 * \param eps An `epsilon` parameter to prevent division by 0.
 * \return new symbol
 */
inline Symbol InstanceNorm(const std::string& symbol_name,
                           Symbol data,
                           Symbol gamma,
                           Symbol beta,
                           mx_float eps = 0.00100000005) {
  return Operator("InstanceNorm")
           .SetParam("eps", eps)
           .SetInput("data", data)
           .SetInput("gamma", gamma)
           .SetInput("beta", beta)
           .CreateSymbol(symbol_name);
}

/*! \brief Specify the dimension along which to compute L2 norm.
 */
enum class L2NormalizationMode {
  kChannel = 0,
  kInstance = 1,
  kSpatial = 2
};

/*!
 * \brief Normalize the input array using the L2 norm.
 *
 *        For 1-D NDArray, it computes::
 *
 *        out = data / sqrt(sum(data ** 2) + eps)
 *
 *        For N-D NDArray, if the input array has shape (N, N, ..., N),
 *
 *        with ``mode`` = ``instance``, it normalizes each instance in the
 *        array by its L2 norm.::
 *
 *        for i in 0...N
 *        out[i,:,:,...,:] = data[i,:,:,...,:] / sqrt(sum(data[i,:,:,...,:] ** 2) + eps)
 *
 *        with ``mode`` = ``channel``, it normalizes each channel in the array by its L2
 *
 *        for i in 0...N
 *        out[:,i,:,...,:] = data[:,i,:,...,:] / sqrt(sum(data[:,i,:,...,:] ** 2) + eps)
 *
 *        with ``mode`` = ``spatial``, it normalizes the cross channel norm for each
 *        in the array by its L2 norm.::
 *
 *        for dim in 2...N
 *        for i in 0...N
 *        out[.....,i,...] = take(out, indices=i, axis=dim) / sqrt(sum(take(out,
 *        -dim-
 *
 *        Example::
 *
 *        x = [[[1,2],
 *        [3,4]],
 *        [[2,2],
 *        [5,6]]]
 *
 *        L2Normalization(x, mode='instance')
 *        =[[[ 0.18257418  0.36514837]
 *        [ 0.54772252  0.73029673]]
 *        [[ 0.24077171  0.24077171]
 *        [ 0.60192931  0.72231513]]]
 *
 *        L2Normalization(x, mode='channel')
 *        =[[[ 0.31622776  0.44721359]
 *        [ 0.94868326  0.89442718]]
 *        [[ 0.37139067  0.31622776]
 *        [ 0.92847669  0.94868326]]]
 *
 *        L2Normalization(x, mode='spatial')
 *        =[[[ 0.44721359  0.89442718]
 *        [ 0.60000002  0.80000001]]
 *        [[ 0.70710677  0.70710677]
 *        [ 0.6401844   0.76822126]]]
 *
 *
 *
 *        Defined in ../src/operator/l2_normalization.cc:L196
 * \param symbol_name name of the resulting symbol
 * \param data Input array to normalize.
 * \param eps A small constant for numerical stability.
 * \param mode Specify the dimension along which to compute L2 norm.
 * \return new symbol
 */
inline Symbol L2Normalization(const std::string& symbol_name,
                              Symbol data,
                              mx_float eps = 1.00000001e-10,
                              L2NormalizationMode mode = L2NormalizationMode::kInstance) {
  static const char *L2NormalizationModeValues[] = {
    "channel",
    "instance",
    "spatial"
  };
  return Operator("L2Normalization")
           .SetParam("eps", eps)
           .SetParam("mode", L2NormalizationModeValues[int(mode)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \brief If this is set to null, the output gradient will not be normalized. If this is
 *        set to batch, the output gradient will be divided by the batch size. If this is
 *        set to valid, the output gradient will be divided by the number of valid input
 */
enum class MakeLossNormalization {
  kBatch = 0,
  kNull = 1,
  kValid = 2
};

/*!
 * \brief Make your own loss function in network construction.
 *
 *        This operator accepts a customized loss function symbol as a terminal loss and
 *        the symbol should be an operator with no backward dependency.
 *        The output of this function is the gradient of loss with respect to the input
 *
 *        For example, if you are a making a cross entropy loss function. Assume ``out``
 *        predicted output and ``label`` is the true label, then the cross entropy can be
 *
 *        cross_entropy = label * log(out) + (1 - label) * log(1 - out)
 *        loss = MakeLoss(cross_entropy)
 *
 *        We will need to use ``MakeLoss`` when we are creating our own loss function or
 *        combine multiple loss functions. Also we may want to stop some variables'
 *        from backpropagation. See more detail in ``BlockGrad`` or ``stop_gradient``.
 *
 *        In addition, we can give a scale to the loss by setting ``grad_scale``,
 *        so that the gradient of the loss will be rescaled in the backpropagation.
 *
 *        .. note:: This operator should be used as a Symbol instead of NDArray.
 *
 *
 *
 *        Defined in ../src/operator/make_loss.cc:L71
 * \param symbol_name name of the resulting symbol
 * \param data Input array.
 * \param grad_scale Gradient scale as a supplement to unary and binary operators
 * \param valid_thresh clip each element in the array to 0 when it is less than
 * \param normalization If this is set to null, the output gradient will not be
 *        normalized. If this is set to batch, the output gradient will be divided by the
 *        batch size. If this is set to valid, the output gradient will be divided by the
 * \return new symbol
 */
inline Symbol MakeLoss(const std::string& symbol_name,
                       Symbol data,
                       mx_float grad_scale = 1,
                       mx_float valid_thresh = 0,
                       MakeLossNormalization normalization = MakeLossNormalization::kNull) {
  static const char *MakeLossNormalizationValues[] = {
    "batch",
    "null",
    "valid"
  };
  return Operator("MakeLoss")
           .SetParam("grad_scale", grad_scale)
           .SetParam("valid_thresh", valid_thresh)
           .SetParam("normalization", MakeLossNormalizationValues[int(normalization)])
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*! \brief Pooling type to be applied.
 */
enum class Pooling_v1PoolType {
  kAvg = 0,
  kMax = 1,
  kSum = 2
};

/*! \brief Pooling convention to be applied.
 */
enum class Pooling_v1PoolingConvention {
  kFull = 0,
  kValid = 1
};

/*!
 * \brief This operator is DEPRECATED.
 *        Perform pooling on the input.
 *
 *        The shapes for 2-D pooling is
 *
 *        - **data**: *(batch_size, channel, height, width)*
 *        - **out**: *(batch_size, num_filter, out_height, out_width)*, with::
 *
 *        out_height = f(height, kernel[0], pad[0], stride[0])
 *        out_width = f(width, kernel[1], pad[1], stride[1])
 *
 *        The definition of *f* depends on ``pooling_convention``, which has two options:
 *
 *        - **valid** (default)::
 *
 *        f(x, k, p, s) = floor((x+2*p-k)/s)+1
 *
 *        - **full**, which is compatible with Caffe::
 *
 *        f(x, k, p, s) = ceil((x+2*p-k)/s)+1
 *
 *        But ``global_pool`` is set to be true, then do a global pooling, namely reset
 *        ``kernel=(height, width)``.
 *
 *        Three pooling options are supported by ``pool_type``:
 *
 *        - **avg**: average pooling
 *        - **max**: max pooling
 *        - **sum**: sum pooling
 *
 *        1-D pooling is special case of 2-D pooling with *weight=1* and
 *        *kernel[1]=1*.
 *
 *        For 3-D pooling, an additional *depth* dimension is added before
 *        *height*. Namely the input data will have shape *(batch_size, channel, depth,
 *        height, width)*.
 *
 *
 *
 *        Defined in ../src/operator/pooling_v1.cc:L104
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the pooling operator.
 * \param kernel pooling kernel size: (y, x) or (d, y, x)
 * \param pool_type Pooling type to be applied.
 * \param global_pool Ignore kernel size, do global pooling based on current input
 * \param pooling_convention Pooling convention to be applied.
 * \param stride stride: for pooling (y, x) or (d, y, x)
 * \param pad pad for pooling: (y, x) or (d, y, x)
 * \return new symbol
 */
inline Symbol Pooling_v1(const std::string& symbol_name,
                         Symbol data,
                         Shape kernel = Shape(),
                         Pooling_v1PoolType pool_type = Pooling_v1PoolType::kMax,
                         bool global_pool = false,
                         Pooling_v1PoolingConvention pooling_convention = Pooling_v1PoolingConvention::kValid,
                         Shape stride = Shape(),
                         Shape pad = Shape()) {
  static const char *Pooling_v1PoolTypeValues[] = {
    "avg",
    "max",
    "sum"
  };
  static const char *Pooling_v1PoolingConventionValues[] = {
    "full",
    "valid"
  };
  return Operator("Pooling_v1")
           .SetParam("kernel", kernel)
           .SetParam("pool_type", Pooling_v1PoolTypeValues[int(pool_type)])
           .SetParam("global_pool", global_pool)
           .SetParam("pooling_convention", Pooling_v1PoolingConventionValues[int(pooling_convention)])
           .SetParam("stride", stride)
           .SetParam("pad", pad)
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Takes the last element of a sequence.
 *
 *        This function takes an n-dimensional input array of the form
 *        [max_sequence_length, batch_size, other_feature_dims] and returns a
 *        of the form [batch_size, other_feature_dims].
 *
 *        Parameter `sequence_length` is used to handle variable-length sequences.
 *        an input array of positive ints of dimension [batch_size]. To use this
 *        set `use_sequence_length` to `True`, otherwise each example in the batch is
 *        to have the max sequence length.
 *
 *        .. note:: Alternatively, you can also use `take` operator.
 *
 *        Example::
 *
 *        x = [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.],
 *        [  7.,   8.,   9.]],
 *
 *        [[ 10.,   11.,   12.],
 *        [ 13.,   14.,   15.],
 *        [ 16.,   17.,   18.]],
 *
 *        [[  19.,   20.,   21.],
 *        [  22.,   23.,   24.],
 *        [  25.,   26.,   27.]]]
 *
 *        // returns last sequence when sequence_length parameter is not used
 *        SequenceLast(x) = [[  19.,   20.,   21.],
 *        [  22.,   23.,   24.],
 *        [  25.,   26.,   27.]]
 *
 *        // sequence_length is used
 *        SequenceLast(x, sequence_length=[1,1,1], use_sequence_length=True) =
 *        [[  1.,   2.,   3.],
 *        [  4.,   5.,   6.],
 *        [  7.,   8.,   9.]]
 *
 *        // sequence_length is used
 *        SequenceLast(x, sequence_length=[1,2,3], use_sequence_length=True) =
 *        [[  1.,    2.,   3.],
 *        [  13.,  14.,  15.],
 *        [  25.,  26.,  27.]]
 *
 *
 *
 *        Defined in ../src/operator/sequence_last.cc:L106
 * \param symbol_name name of the resulting symbol
 * \param data n-dimensional input array of the form [max_sequence_length, batch_size,
 * \param sequence_length vector of sequence lengths of the form [batch_size]
 * \param use_sequence_length If set to true, this layer takes in an extra input
 * \param axis The sequence axis. Only values of 0 and 1 are currently supported.
 * \return new symbol
 */
inline Symbol SequenceLast(const std::string& symbol_name,
                           Symbol data,
                           Symbol sequence_length,
                           bool use_sequence_length = false,
                           int axis = 0) {
  return Operator("SequenceLast")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetParam("axis", axis)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Reverses the elements of each sequence.
 *
 *        This function takes an n-dimensional input array of the form
 *        and returns an array of the same shape.
 *
 *        Parameter `sequence_length` is used to handle variable-length sequences.
 *        `sequence_length` should be an input array of positive ints of dimension
 *        To use this parameter, set `use_sequence_length` to `True`,
 *        otherwise each example in the batch is assumed to have the max sequence length.
 *
 *        Example::
 *
 *        x = [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]]]
 *
 *        // Batch 1
 *        B1 = [[  1.,   2.,   3.],
 *        [  7.,   8.,   9.],
 *        [ 13.,  14.,  15.]]
 *
 *        // Batch 2
 *        B2 = [[  4.,   5.,   6.],
 *        [ 10.,  11.,  12.],
 *        [ 16.,  17.,  18.]]
 *
 *        // returns reverse sequence when sequence_length parameter is not used
 *        SequenceReverse(x) = [[[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]],
 *
 *        [[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]]]
 *
 *        // sequence_length [2,2] means 2 rows of
 *        // both batch B1 and B2 will be reversed.
 *        SequenceReverse(x, sequence_length=[2,2], use_sequence_length=True) =
 *        [[[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]]]
 *
 *        // sequence_length [2,3] means 2 of batch B2 and 3 of batch B3
 *        // will be reversed.
 *        SequenceReverse(x, sequence_length=[2,3], use_sequence_length=True) =
 *        [[[  7.,   8.,   9.],
 *        [ 16.,  17.,  18.]],
 *
 *        [[  1.,   2.,   3.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[ 13.,  14,   15.],
 *        [  4.,   5.,   6.]]]
 *
 *
 *
 *        Defined in ../src/operator/sequence_reverse.cc:L122
 * \param symbol_name name of the resulting symbol
 * \param data n-dimensional input array of the form [max_sequence_length, batch_size,
 * \param sequence_length vector of sequence lengths of the form [batch_size]
 * \param use_sequence_length If set to true, this layer takes in an extra input
 * \param axis The sequence axis. Only 0 is currently supported.
 * \return new symbol
 */
inline Symbol SequenceReverse(const std::string& symbol_name,
                              Symbol data,
                              Symbol sequence_length,
                              bool use_sequence_length = false,
                              int axis = 0) {
  return Operator("SequenceReverse")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetParam("axis", axis)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol(symbol_name);
}

/*! \brief transformation type
 */
enum class SpatialTransformerTransformType {
  kAffine = 0
};

/*! \brief sampling type
 */
enum class SpatialTransformerSamplerType {
  kBilinear = 0
};

/*!
 * \brief Applies a spatial transformer to input feature map.
 * \param symbol_name name of the resulting symbol
 * \param data Input data to the SpatialTransformerOp.
 * \param loc localisation net, the output dim should be 6 when transform_type is affine.
 * \param transform_type transformation type
 * \param sampler_type sampling type
 * \param target_shape output shape(h, w) of spatial transformer: (y, x)
 * \param cudnn_off whether to turn cudnn off
 * \return new symbol
 */
inline Symbol SpatialTransformer(const std::string& symbol_name,
                                 Symbol data,
                                 Symbol loc,
                                 SpatialTransformerTransformType transform_type,
                                 SpatialTransformerSamplerType sampler_type,
                                 Shape target_shape = Shape(0,0),
                                 dmlc::optional<bool> cudnn_off = dmlc::optional<bool>()) {
  static const char *SpatialTransformerTransformTypeValues[] = {
    "affine"
  };
  static const char *SpatialTransformerSamplerTypeValues[] = {
    "bilinear"
  };
  return Operator("SpatialTransformer")
           .SetParam("transform_type", SpatialTransformerTransformTypeValues[int(transform_type)])
           .SetParam("sampler_type", SpatialTransformerSamplerTypeValues[int(sampler_type)])
           .SetParam("target_shape", target_shape)
           .SetParam("cudnn_off", cudnn_off)
           .SetInput("data", data)
           .SetInput("loc", loc)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Computes support vector machine based transformation of the input.
 *
 *        This tutorial demonstrates using SVM as output layer for classification instead
 *        https://github.com/dmlc/mxnet/tree/master/example/svm_mnist.
 *
 *
 * \param symbol_name name of the resulting symbol
 * \param data Input data for SVM transformation.
 * \param label Class label for the input data.
 * \param margin The loss function penalizes outputs that lie outside this margin.
 * \param regularization_coefficient Regularization parameter for the SVM. This balances
 * \param use_linear Whether to use L1-SVM objective. L2-SVM objective is used by default.
 * \return new symbol
 */
inline Symbol SVMOutput(const std::string& symbol_name,
                        Symbol data,
                        Symbol label,
                        mx_float margin = 1,
                        mx_float regularization_coefficient = 1,
                        bool use_linear = false) {
  return Operator("SVMOutput")
           .SetParam("margin", margin)
           .SetParam("regularization_coefficient", regularization_coefficient)
           .SetParam("use_linear", use_linear)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Fill one element of each line(row for python, column for R/Julia) in lhs
 *        according to index indicated by rhs and values indicated by mhs. This function
 * \param symbol_name name of the resulting symbol
 * \param lhs Left operand to the function.
 * \param mhs Middle operand to the function.
 * \param rhs Right operand to the function.
 * \return new symbol
 */
inline Symbol fill_element_0index(const std::string& symbol_name,
                                  Symbol lhs,
                                  Symbol mhs,
                                  Symbol rhs) {
  return Operator("fill_element_0index")
           .SetInput("lhs", lhs)
           .SetInput("mhs", mhs)
           .SetInput("rhs", rhs)
           .CreateSymbol(symbol_name);
}

/*!
 * \brief Batch normalization.
 *
 *        This operator is DEPRECATED. Perform BatchNorm on the input.
 *
 *        Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as
 *        well as offset ``beta``.
 *
 *        Assume the input has more than one dimension and we normalize along axis 1.
 *        We first compute the mean and variance along this axis:
 *
 *        .. math::
 *
 *        data\_mean[i] = mean(data[:,i,:,...]) \\
 *        data\_var[i] = var(data[:,i,:,...])
 *
 *        Then compute the normalized output, which has the same shape as input, as
 *
 *        .. math::
 *
 *        out[:,i,:,...] = \frac{data[:,i,:,...] -
 *
 *        Both *mean* and *var* returns a scalar by treating the input as a vector.
 *
 *        Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
 *        have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both
 *        ``data_var`` as well, which are needed for the backward pass.
 *
 *        Besides the inputs and the outputs, this operator accepts two auxiliary
 *        states, ``moving_mean`` and ``moving_var``, which are *k*-length
 *        vectors. They are global statistics for the whole dataset, which are updated
 *        by::
 *
 *        moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
 *        moving_var = moving_var * momentum + data_var * (1 - momentum)
 *
 *        If ``use_global_stats`` is set to be true, then ``moving_mean`` and
 *        ``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute
 *        the output. It is often used during inference.
 *
 *        Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is
 *        then set ``gamma`` to 1 and its gradient to 0.
 *
 *        There's no sparse support for this operator, and it will exhibit problematic
 *        sparse tensors.
 *
 *
 *
 *        Defined in ../src/operator/batch_norm_v1.cc:L95
 * \param data Input data to batch normalization
 * \param gamma gamma array
 * \param beta beta array
 * \param eps Epsilon to prevent div 0
 * \param momentum Momentum for moving average
 * \param fix_gamma Fix gamma while training
 * \param use_global_stats Whether use global moving statistics instead of local
 * \param output_mean_var Output All,normal mean and var
 * \return new symbol
 */
inline Symbol BatchNorm_v1(Symbol data,
                           Symbol gamma,
                           Symbol beta,
                           mx_float eps = 0.00100000005,
                           mx_float momentum = 0.899999976,
                           bool fix_gamma = true,
                           bool use_global_stats = false,
                           bool output_mean_var = false) {
  return Operator("BatchNorm_v1")
           .SetParam("eps", eps)
           .SetParam("momentum", momentum)
           .SetParam("fix_gamma", fix_gamma)
           .SetParam("use_global_stats", use_global_stats)
           .SetParam("output_mean_var", output_mean_var)
           .SetInput("data", data)
           .SetInput("gamma", gamma)
           .SetInput("beta", beta)
           .CreateSymbol();
}

/*!
 * \brief Check if all the float numbers in the array are finite (used for AMP)
 *
 *
 *        Defined in ../src/operator/contrib/all_finite.cc:L101
 * \param data Array
 * \param init_output Initialize output to 1.
 * \return new symbol
 */
inline Symbol all_finite(Symbol data,
                         bool init_output = true) {
  return Operator("all_finite")
           .SetParam("init_output", init_output)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Check if all the float numbers in all the arrays are finite (used for AMP)
 *
 *
 *        Defined in ../src/operator/contrib/all_finite.cc:L133
 * \param data Arrays
 * \param num_arrays Number of arrays.
 * \param init_output Initialize output to 1.
 * \return new symbol
 */
inline Symbol multi_all_finite(const std::vector<Symbol>& data,
                               int num_arrays = 1,
                               bool init_output = true) {
  return Operator("multi_all_finite")
           .SetParam("num_arrays", num_arrays)
           .SetParam("init_output", init_output)
(data)
           .CreateSymbol();
}

/*!
 * \brief Computes the Khatri-Rao product of the input matrices.
 *
 *        Given a collection of :math:`n` input matrices,
 *
 *        .. math::
 *        A_1 \in \mathbb{R}^{M_1 \times M}, \ldots, A_n \in \mathbb{R}^{M_n \times N},
 *
 *        the (column-wise) Khatri-Rao product is defined as the matrix,
 *
 *        .. math::
 *        X = A_1 \otimes \cdots \otimes A_n \in \mathbb{R}^{(M_1 \cdots M_n) \times N},
 *
 *        where the :math:`k` th column is equal to the column-wise outer product
 *        :math:`{A_1}_k \otimes \cdots \otimes {A_n}_k` where :math:`{A_i}_k` is the kth
 *        column of the ith matrix.
 *
 *        Example::
 *
 *        >>> A = mx.nd.array([[1, -1],
 *        >>>                  [2, -3]])
 *        >>> B = mx.nd.array([[1, 4],
 *        >>>                  [2, 5],
 *        >>>                  [3, 6]])
 *        >>> C = mx.nd.khatri_rao(A, B)
 *        >>> print(C.asnumpy())
 *        [[  1.  -4.]
 *        [  2.  -5.]
 *        [  3.  -6.]
 *        [  2. -12.]
 *        [  4. -15.]
 *        [  6. -18.]]
 *
 *
 *
 *        Defined in ../src/operator/contrib/krprod.cc:L108
 * \param args Positional input matrices
 * \return new symbol
 */
inline Symbol khatri_rao(const std::vector<Symbol>& args) {
  return Operator("khatri_rao")
(args)
           .CreateSymbol();
}

/*!
 * \brief Compute the LARS coefficients of multiple weights and grads from their sums of
 *
 *
 *        Defined in ../src/operator/contrib/multi_lars.cc:L37
 * \param lrs Learning rates to scale by LARS coefficient
 * \param weights_sum_sq sum of square of weights arrays
 * \param grads_sum_sq sum of square of gradients arrays
 * \param wds weight decays
 * \param eta LARS eta
 * \param eps LARS eps
 * \param rescale_grad Gradient rescaling factor
 * \return new symbol
 */
inline Symbol multi_lars(Symbol lrs,
                         Symbol weights_sum_sq,
                         Symbol grads_sum_sq,
                         Symbol wds,
                         mx_float eta,
                         mx_float eps,
                         mx_float rescale_grad = 1) {
  return Operator("multi_lars")
           .SetParam("eta", eta)
           .SetParam("eps", eps)
           .SetParam("rescale_grad", rescale_grad)
           .SetInput("lrs", lrs)
           .SetInput("weights_sum_sq", weights_sum_sq)
           .SetInput("grads_sum_sq", grads_sum_sq)
           .SetInput("wds", wds)
           .CreateSymbol();
}

/*!
 * \brief Compute the sums of squares of multiple arrays
 *
 *
 *        Defined in ../src/operator/contrib/multi_sum_sq.cc:L36
 * \param data Arrays
 * \param num_arrays number of input arrays.
 * \return new symbol
 */
inline Symbol multi_sum_sq(const std::vector<Symbol>& data,
                           int num_arrays) {
  return Operator("multi_sum_sq")
           .SetParam("num_arrays", num_arrays)
(data)
           .CreateSymbol();
}

/*!
 * \brief Update function for Stochastic Gradient Descent (SDG) optimizer.
 *
 *        It updates the weights using::
 *
 *        weight = weight - learning_rate * (gradient + wd * weight)
 *
 *
 *
 *        Defined in ../src/operator/contrib/preloaded_multi_sgd.cc:L42
 * \param data Weights, gradients, learning rates and weight decays
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param num_weights Number of updated weights.
 * \return new symbol
 */
inline Symbol preloaded_multi_sgd_update(const std::vector<Symbol>& data,
                                         mx_float rescale_grad = 1,
                                         mx_float clip_gradient = -1,
                                         int num_weights = 1) {
  return Operator("preloaded_multi_sgd_update")
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("num_weights", num_weights)
(data)
           .CreateSymbol();
}

/*!
 * \brief Momentum update function for Stochastic Gradient Descent (SGD) optimizer.
 *
 *        Momentum update has better convergence rates on neural networks. Mathematically
 *        like below:
 *
 *        .. math::
 *
 *        v_1 = \alpha * \nabla J(W_0)\\
 *        v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\
 *        W_t = W_{t-1} + v_t
 *
 *        It updates the weights using::
 *
 *        v = momentum * v - learning_rate * gradient
 *        weight += v
 *
 *        Where the parameter ``momentum`` is the decay rate of momentum estimates at
 *
 *
 *
 *        Defined in ../src/operator/contrib/preloaded_multi_sgd.cc:L91
 * \param data Weights, gradients, momentum, learning rates and weight decays
 * \param momentum The decay rate of momentum estimates at each epoch.
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param num_weights Number of updated weights.
 * \return new symbol
 */
inline Symbol preloaded_multi_sgd_mom_update(const std::vector<Symbol>& data,
                                             mx_float momentum = 0,
                                             mx_float rescale_grad = 1,
                                             mx_float clip_gradient = -1,
                                             int num_weights = 1) {
  return Operator("preloaded_multi_sgd_mom_update")
           .SetParam("momentum", momentum)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("num_weights", num_weights)
(data)
           .CreateSymbol();
}

/*!
 * \brief Update function for multi-precision Stochastic Gradient Descent (SDG) optimizer.
 *
 *        It updates the weights using::
 *
 *        weight = weight - learning_rate * (gradient + wd * weight)
 *
 *
 *
 *        Defined in ../src/operator/contrib/preloaded_multi_sgd.cc:L140
 * \param data Weights, gradients, learning rates and weight decays
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param num_weights Number of updated weights.
 * \return new symbol
 */
inline Symbol preloaded_multi_mp_sgd_update(const std::vector<Symbol>& data,
                                            mx_float rescale_grad = 1,
                                            mx_float clip_gradient = -1,
                                            int num_weights = 1) {
  return Operator("preloaded_multi_mp_sgd_update")
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("num_weights", num_weights)
(data)
           .CreateSymbol();
}

/*!
 * \brief Momentum update function for multi-precision Stochastic Gradient Descent (SGD)
 *
 *        Momentum update has better convergence rates on neural networks. Mathematically
 *        like below:
 *
 *        .. math::
 *
 *        v_1 = \alpha * \nabla J(W_0)\\
 *        v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\
 *        W_t = W_{t-1} + v_t
 *
 *        It updates the weights using::
 *
 *        v = momentum * v - learning_rate * gradient
 *        weight += v
 *
 *        Where the parameter ``momentum`` is the decay rate of momentum estimates at
 *
 *
 *
 *        Defined in ../src/operator/contrib/preloaded_multi_sgd.cc:L200
 * \param data Weights, gradients, momentums, learning rates and weight decays
 * \param momentum The decay rate of momentum estimates at each epoch.
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param num_weights Number of updated weights.
 * \return new symbol
 */
inline Symbol preloaded_multi_mp_sgd_mom_update(const std::vector<Symbol>& data,
                                                mx_float momentum = 0,
                                                mx_float rescale_grad = 1,
                                                mx_float clip_gradient = -1,
                                                int num_weights = 1) {
  return Operator("preloaded_multi_mp_sgd_mom_update")
           .SetParam("momentum", momentum)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("num_weights", num_weights)
(data)
           .CreateSymbol();
}

/*!
 * \brief Set to zero multiple arrays
 *
 *
 *        Defined in ../src/operator/contrib/reset_arrays.cc:L36
 * \param data Arrays
 * \param num_arrays number of input arrays.
 * \return new symbol
 */
inline Symbol reset_arrays(const std::vector<Symbol>& data,
                           int num_arrays) {
  return Operator("reset_arrays")
           .SetParam("num_arrays", num_arrays)
(data)
           .CreateSymbol();
}

/*!
 * \brief Apply a custom operator implemented in a frontend language (like Python).
 *
 *        Custom operators should override required methods like `forward` and `backward`.
 *        The custom operator must be registered before it can be used.
 *        Please check the tutorial here:
 *
 *
 *
 *        Defined in ../src/operator/custom/custom.cc:L547
 * \param data Input data for the custom operator.
 * \param op_type Name of the custom operator. This is the name that is passed to
 * \return new symbol
 */
inline Symbol Custom(const std::vector<Symbol>& data,
                     const std::string& op_type) {
  return Operator("Custom")
(data)
           .CreateSymbol();
}

/*!
 * \brief Apply a sparse regularization to the output a sigmoid activation function.
 * \param data Input data.
 * \param sparseness_target The sparseness target
 * \param penalty The tradeoff parameter for the sparseness penalty
 * \param momentum The momentum for running average
 * \return new symbol
 */
inline Symbol IdentityAttachKLSparseReg(Symbol data,
                                        mx_float sparseness_target = 0.100000001,
                                        mx_float penalty = 0.00100000005,
                                        mx_float momentum = 0.899999976) {
  return Operator("IdentityAttachKLSparseReg")
           .SetParam("sparseness_target", sparseness_target)
           .SetParam("penalty", penalty)
           .SetParam("momentum", momentum)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Applies Leaky rectified linear unit activation element-wise to the input.
 *
 *        Leaky ReLUs attempt to fix the "dying ReLU" problem by allowing a small `slope`
 *        when the input is negative and has a slope of one when input is positive.
 *
 *        The following modified ReLU Activation functions are supported:
 *
 *        - *elu*: Exponential Linear Unit. `y = x > 0 ? x : slope * (exp(x)-1)`
 *        - *selu*: Scaled Exponential Linear Unit. `y = lambda * (x > 0 ? x : alpha *
 *        *lambda = 1.0507009873554804934193349852946* and *alpha =
 *        - *leaky*: Leaky ReLU. `y = x > 0 ? x : slope * x`
 *        - *prelu*: Parametric ReLU. This is same as *leaky* except that `slope` is
 *        - *rrelu*: Randomized ReLU. same as *leaky* but the `slope` is uniformly and
 *        *[lower_bound, upper_bound)* for training, while fixed to be
 *        *(lower_bound+upper_bound)/2* for inference.
 *
 *
 *
 *        Defined in ../src/operator/leaky_relu.cc:L163
 * \param data Input data to activation function.
 * \param gamma Input data to activation function.
 * \param act_type Activation function to be applied.
 * \param slope Init slope for the activation. (For leaky and elu only)
 * \param lower_bound Lower bound of random slope. (For rrelu only)
 * \param upper_bound Upper bound of random slope. (For rrelu only)
 * \return new symbol
 */
inline Symbol LeakyReLU(Symbol data,
                        Symbol gamma,
                        LeakyReLUActType act_type = LeakyReLUActType::kLeaky,
                        mx_float slope = 0.25,
                        mx_float lower_bound = 0.125,
                        mx_float upper_bound = 0.333999991) {
  static const char *LeakyReLUActTypeValues[] = {
    "elu",
    "gelu",
    "leaky",
    "prelu",
    "rrelu",
    "selu"
  };
  return Operator("LeakyReLU")
           .SetParam("act_type", LeakyReLUActTypeValues[int(act_type)])
           .SetParam("slope", slope)
           .SetParam("lower_bound", lower_bound)
           .SetParam("upper_bound", upper_bound)
           .SetInput("data", data)
           .SetInput("gamma", gamma)
           .CreateSymbol();
}

/*!
 * \brief Calculate cross entropy of softmax output and one-hot label.
 *
 *        - This operator computes the cross entropy in two steps:
 *        - Applies softmax function on the input array.
 *        - Computes and returns the cross entropy loss between the softmax output and
 *
 *        - The softmax function and cross entropy loss is given by:
 *
 *        - Softmax Function:
 *
 *        .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}
 *
 *        - Cross Entropy Function:
 *
 *        .. math:: \text{CE(label, output)} = - \sum_i \text{label}_i
 *
 *        Example::
 *
 *        x = [[1, 2, 3],
 *        [11, 7, 5]]
 *
 *        label = [2, 0]
 *
 *        softmax(x) = [[0.09003057, 0.24472848, 0.66524094],
 *        [0.97962922, 0.01794253, 0.00242826]]
 *
 *        softmax_cross_entropy(data, label) = - log(0.66524084) - log(0.97962922) =
 *
 *
 *
 *        Defined in ../src/operator/loss_binary_op.cc:L59
 * \param data Input data
 * \param label Input label
 * \return new symbol
 */
inline Symbol softmax_cross_entropy(Symbol data,
                                    Symbol label) {
  return Operator("softmax_cross_entropy")
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \brief Applies an activation function element-wise to the input.
 *
 *        The following activation functions are supported:
 *
 *        - `relu`: Rectified Linear Unit, :math:`y = max(x, 0)`
 *        - `sigmoid`: :math:`y = \frac{1}{1 + exp(-x)}`
 *        - `tanh`: Hyperbolic tangent, :math:`y = \frac{exp(x) - exp(-x)}{exp(x) +
 *        - `softrelu`: Soft ReLU, or SoftPlus, :math:`y = log(1 + exp(x))`
 *        - `softsign`: :math:`y = \frac{x}{1 + abs(x)}`
 *
 *
 *
 *        Defined in ../src/operator/nn/activation.cc:L165
 * \param data The input array.
 * \param act_type Activation function to be applied.
 * \return new symbol
 */
inline Symbol Activation(Symbol data,
                         ActivationActType act_type) {
  static const char *ActivationActTypeValues[] = {
    "relu",
    "sigmoid",
    "softrelu",
    "softsign",
    "tanh"
  };
  return Operator("Activation")
           .SetParam("act_type", ActivationActTypeValues[int(act_type)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Batch normalization.
 *
 *        Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as
 *        well as offset ``beta``.
 *
 *        Assume the input has more than one dimension and we normalize along axis 1.
 *        We first compute the mean and variance along this axis:
 *
 *        .. math::
 *
 *        data\_mean[i] = mean(data[:,i,:,...]) \\
 *        data\_var[i] = var(data[:,i,:,...])
 *
 *        Then compute the normalized output, which has the same shape as input, as
 *
 *        .. math::
 *
 *        out[:,i,:,...] = \frac{data[:,i,:,...] -
 *
 *        Both *mean* and *var* returns a scalar by treating the input as a vector.
 *
 *        Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
 *        have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both
 *        the inverse of ``data_var``, which are needed for the backward pass. Note that
 *        two outputs are blocked.
 *
 *        Besides the inputs and the outputs, this operator accepts two auxiliary
 *        states, ``moving_mean`` and ``moving_var``, which are *k*-length
 *        vectors. They are global statistics for the whole dataset, which are updated
 *        by::
 *
 *        moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
 *        moving_var = moving_var * momentum + data_var * (1 - momentum)
 *
 *        If ``use_global_stats`` is set to be true, then ``moving_mean`` and
 *        ``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute
 *        the output. It is often used during inference.
 *
 *        The parameter ``axis`` specifies which axis of the input shape denotes
 *        the 'channel' (separately normalized groups).  The default is 1.  Specifying -1
 *        axis to be the last item in the input shape.
 *
 *        Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is
 *        then set ``gamma`` to 1 and its gradient to 0.
 *
 *        .. Note::
 *        When ``fix_gamma`` is set to True, no sparse support is provided. If
 *        the sparse tensors will fallback.
 *
 *
 *
 *        Defined in ../src/operator/nn/batch_norm.cc:L607
 * \param data Input data to batch normalization
 * \param gamma gamma array
 * \param beta beta array
 * \param moving_mean running mean of input
 * \param moving_var running variance of input
 * \param eps Epsilon to prevent div 0. Must be no less than CUDNN_BN_MIN_EPSILON defined
 * \param momentum Momentum for moving average
 * \param fix_gamma Fix gamma while training
 * \param use_global_stats Whether use global moving statistics instead of local
 * \param output_mean_var Output the mean and inverse std
 * \param axis Specify which shape axis the channel is specified
 * \param cudnn_off Do not select CUDNN operator, if available
 * \param min_calib_range The minimum scalar value in the form of float32 obtained
 *        through calibration. If present, it will be used to by quantized batch norm op
 * \param max_calib_range The maximum scalar value in the form of float32 obtained
 *        through calibration. If present, it will be used to by quantized batch norm op
 * \return new symbol
 */
inline Symbol BatchNorm(Symbol data,
                        Symbol gamma,
                        Symbol beta,
                        Symbol moving_mean,
                        Symbol moving_var,
                        double eps = 0.0010000000474974513,
                        mx_float momentum = 0.899999976,
                        bool fix_gamma = true,
                        bool use_global_stats = false,
                        bool output_mean_var = false,
                        int axis = 1,
                        bool cudnn_off = false,
                        dmlc::optional<float> min_calib_range = dmlc::optional<float>(),
                        dmlc::optional<float> max_calib_range = dmlc::optional<float>()) {
  return Operator("BatchNorm")
           .SetParam("eps", eps)
           .SetParam("momentum", momentum)
           .SetParam("fix_gamma", fix_gamma)
           .SetParam("use_global_stats", use_global_stats)
           .SetParam("output_mean_var", output_mean_var)
           .SetParam("axis", axis)
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("min_calib_range", min_calib_range)
           .SetParam("max_calib_range", max_calib_range)
           .SetInput("data", data)
           .SetInput("gamma", gamma)
           .SetInput("beta", beta)
           .SetInput("moving_mean", moving_mean)
           .SetInput("moving_var", moving_var)
           .CreateSymbol();
}

/*!
 * \brief Joins input arrays along a given axis.
 *
 *        .. note:: `Concat` is deprecated. Use `concat` instead.
 *
 *        The dimensions of the input arrays should be the same except the axis along
 *        which they will be concatenated.
 *        The dimension of the output array along the concatenated axis will be equal
 *        to the sum of the corresponding dimensions of the input arrays.
 *
 *        The storage type of ``concat`` output depends on storage types of inputs
 *
 *        - concat(csr, csr, ..., csr, dim=0) = csr
 *        - otherwise, ``concat`` generates output with default storage
 *
 *        Example::
 *
 *        x = [[1,1],[2,2]]
 *        y = [[3,3],[4,4],[5,5]]
 *        z = [[6,6], [7,7],[8,8]]
 *
 *        concat(x,y,z,dim=0) = [[ 1.,  1.],
 *        [ 2.,  2.],
 *        [ 3.,  3.],
 *        [ 4.,  4.],
 *        [ 5.,  5.],
 *        [ 6.,  6.],
 *        [ 7.,  7.],
 *        [ 8.,  8.]]
 *
 *        Note that you cannot concat x,y,z along dimension 1 since dimension
 *        0 is not the same for all the input arrays.
 *
 *        concat(y,z,dim=1) = [[ 3.,  3.,  6.,  6.],
 *        [ 4.,  4.,  7.,  7.],
 *        [ 5.,  5.,  8.,  8.]]
 *
 *
 *
 *        Defined in ../src/operator/nn/concat.cc:L385
 * \param data List of arrays to concatenate
 * \param num_args Number of inputs to be concated.
 * \param dim the dimension to be concated.
 * \return new symbol
 */
inline Symbol Concat(const std::vector<Symbol>& data,
                     int num_args,
                     int dim = 1) {
  return Operator("Concat")
           .SetParam("num_args", num_args)
           .SetParam("dim", dim)
(data)
           .CreateSymbol();
}

/*!
 * \brief Compute *N*-D convolution on *(N+2)*-D input.
 *
 *        In the 2-D convolution, given input data with shape *(batch_size,
 *        channel, height, width)*, the output is computed by
 *
 *        .. math::
 *
 *        out[n,i,:,:] = bias[i] + \sum_{j=0}^{channel} data[n,j,:,:] \star
 *        weight[i,j,:,:]
 *
 *        where :math:`\star` is the 2-D cross-correlation operator.
 *
 *        For general 2-D convolution, the shapes are
 *
 *        - **data**: *(batch_size, channel, height, width)*
 *        - **weight**: *(num_filter, channel, kernel[0], kernel[1])*
 *        - **bias**: *(num_filter,)*
 *        - **out**: *(batch_size, num_filter, out_height, out_width)*.
 *
 *        Define::
 *
 *        f(x,k,p,s,d) = floor((x+2*p-d*(k-1)-1)/s)+1
 *
 *        then we have::
 *
 *        out_height=f(height, kernel[0], pad[0], stride[0], dilate[0])
 *        out_width=f(width, kernel[1], pad[1], stride[1], dilate[1])
 *
 *        If ``no_bias`` is set to be true, then the ``bias`` term is ignored.
 *
 *        The default data ``layout`` is *NCHW*, namely *(batch_size, channel, height,
 *        width)*. We can choose other layouts such as *NWC*.
 *
 *        If ``num_group`` is larger than 1, denoted by *g*, then split the input ``data``
 *        evenly into *g* parts along the channel axis, and also evenly split ``weight``
 *        along the first dimension. Next compute the convolution on the *i*-th part of
 *        the data with the *i*-th weight part. The output is obtained by concatenating
 *        the *g* results.
 *
 *        1-D convolution does not have *height* dimension but only *width* in space.
 *
 *        - **data**: *(batch_size, channel, width)*
 *        - **weight**: *(num_filter, channel, kernel[0])*
 *        - **bias**: *(num_filter,)*
 *        - **out**: *(batch_size, num_filter, out_width)*.
 *
 *        3-D convolution adds an additional *depth* dimension besides *height* and
 *        *width*. The shapes are
 *
 *        - **data**: *(batch_size, channel, depth, height, width)*
 *        - **weight**: *(num_filter, channel, kernel[0], kernel[1], kernel[2])*
 *        - **bias**: *(num_filter,)*
 *        - **out**: *(batch_size, num_filter, out_depth, out_height, out_width)*.
 *
 *        Both ``weight`` and ``bias`` are learnable parameters.
 *
 *        There are other options to tune the performance.
 *
 *        - **cudnn_tune**: enable this option leads to higher startup time but may give
 *        faster speed. Options are
 *
 *        - **off**: no tuning
 *        - **limited_workspace**:run test and pick the fastest algorithm that doesn't
 *        exceed workspace limit.
 *        - **fastest**: pick the fastest algorithm and ignore workspace limit.
 *        - **None** (default): the behavior is determined by environment variable
 *        ``MXNET_CUDNN_AUTOTUNE_DEFAULT``. 0 for off, 1 for limited workspace
 *        (default), 2 for fastest.
 *
 *        - **workspace**: A large number leads to more (GPU) memory usage but may improve
 *        the performance.
 *
 *
 *
 *        Defined in ../src/operator/nn/convolution.cc:L476
 * \param data Input data to the ConvolutionOp.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param kernel Convolution kernel size: (w,), (h, w) or (d, h, w)
 * \param num_filter Convolution filter(channel) number
 * \param stride Convolution stride: (w,), (h, w) or (d, h, w). Defaults to 1 for each
 * \param dilate Convolution dilate: (w,), (h, w) or (d, h, w). Defaults to 1 for each
 * \param pad Zero pad for convolution: (w,), (h, w) or (d, h, w). Defaults to no padding.
 * \param num_group Number of group partitions.
 * \param workspace Maximum temporary workspace allowed (MB) in convolution.This
 *        parameter has two usages. When CUDNN is not used, it determines the effective
 *        batch size of the convolution kernel. When CUDNN is used, it controls the
 *        maximum temporary storage used for tuning the best CUDNN kernel when
 * \param no_bias Whether to disable bias parameter.
 * \param cudnn_tune Whether to pick convolution algo by running performance test.
 * \param cudnn_off Turn off cudnn for this layer.
 * \param layout Set layout for input, output and weight. Empty for
 *        default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.NHWC and NDHWC are
 * \return new symbol
 */
inline Symbol Convolution(Symbol data,
                          Symbol weight,
                          Symbol bias,
                          Shape kernel,
                          uint32_t num_filter,
                          Shape stride = Shape(),
                          Shape dilate = Shape(),
                          Shape pad = Shape(),
                          uint32_t num_group = 1,
                          uint64_t workspace = 1024,
                          bool no_bias = false,
                          ConvolutionCudnnTune cudnn_tune = ConvolutionCudnnTune::kNone,
                          bool cudnn_off = false,
                          ConvolutionLayout layout = ConvolutionLayout::kNone) {
  static const char *ConvolutionCudnnTuneValues[] = {
    "None",
    "fastest",
    "limited_workspace",
    "off"
  };
  static const char *ConvolutionLayoutValues[] = {
    "None",
    "NCDHW",
    "NCHW",
    "NCW",
    "NDHWC",
    "NHWC"
  };
  return Operator("Convolution")
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("dilate", dilate)
           .SetParam("pad", pad)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
           .SetParam("cudnn_tune", ConvolutionCudnnTuneValues[int(cudnn_tune)])
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("layout", ConvolutionLayoutValues[int(layout)])
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol();
}

/*!
 * \brief Connectionist Temporal Classification Loss.
 *
 *        .. note:: The existing alias ``contrib_CTCLoss`` is deprecated.
 *
 *        The shapes of the inputs and outputs:
 *
 *        - **data**: `(sequence_length, batch_size, alphabet_size)`
 *        - **label**: `(batch_size, label_sequence_length)`
 *        - **out**: `(batch_size)`
 *
 *        The `data` tensor consists of sequences of activation vectors (without applying
 *        with i-th channel in the last dimension corresponding to i-th label
 *        for i between 0 and alphabet_size-1 (i.e always 0-indexed).
 *        Alphabet size should include one additional value reserved for blank label.
 *        When `blank_label` is ``"first"``, the ``0``-th channel is be reserved for
 *        activation of blank label, or otherwise if it is "last",
 *        reserved for blank label.
 *
 *        ``label`` is an index matrix of integers. When `blank_label` is ``"first"``,
 *        the value 0 is then reserved for blank label, and should not be passed in this
 *        when `blank_label` is ``"last"``, the value `(alphabet_size-1)` is reserved for
 *
 *        If a sequence of labels is shorter than *label_sequence_length*, use the special
 *        padding value at the end of the sequence to conform it to the correct
 *        length. The padding value is `0` when `blank_label` is ``"first"``, and `-1`
 *
 *        For example, suppose the vocabulary is `[a, b, c]`, and in one batch we have
 *        'ba', 'cbb', and 'abac'. When `blank_label` is ``"first"``, we can index the
 *        `{'a': 1, 'b': 2, 'c': 3}`, and we reserve the 0-th channel for blank label in
 *        The resulting `label` tensor should be padded to be::
 *
 *        [[2, 1, 0, 0], [3, 2, 2, 0], [1, 2, 1, 3]]
 *
 *        When `blank_label` is ``"last"``, we can index the labels as
 *        `{'a': 0, 'b': 1, 'c': 2}`, and we reserve the channel index 3 for blank label
 *        The resulting `label` tensor should be padded to be::
 *
 *        [[1, 0, -1, -1], [2, 1, 1, -1], [0, 1, 0, 2]]
 *
 *        ``out`` is a list of CTC loss values, one per example in the batch.
 *
 *        See *Connectionist Temporal Classification: Labelling Unsegmented
 *        Sequence Data with Recurrent Neural Networks*, A. Graves *et al*. for more
 *        information on the definition and the algorithm.
 *
 *
 *
 *        Defined in ../src/operator/nn/ctc_loss.cc:L100
 * \param data Input ndarray
 * \param label Ground-truth labels for the loss.
 * \param data_lengths Lengths of data for each of the samples. Only required when
 * \param label_lengths Lengths of labels for each of the samples. Only required when
 * \param use_data_lengths Whether the data lenghts are decided by `data_lengths`. If
 * \param use_label_lengths Whether the label lenghts are decided by `label_lengths`, or
 *        derived from `padding_mask`. If false, the lengths are derived from the first
 *        occurrence of the value of `padding_mask`. The value of `padding_mask` is ``0``
 *        when first CTC label is reserved for blank, and ``-1`` when last label is
 * \param blank_label Set the label that is reserved for blank label.If "first", 0-th
 *        label is reserved, and label values for tokens in the vocabulary are between
 *        ``1`` and ``alphabet_size-1``, and the padding mask is ``-1``. If "last", last
 *        label value ``alphabet_size-1`` is reserved for blank label instead, and label
 *        values for tokens in the vocabulary are between ``0`` and ``alphabet_size-2``,
 * \return new symbol
 */
inline Symbol CTCLoss(Symbol data,
                      Symbol label,
                      Symbol data_lengths,
                      Symbol label_lengths,
                      bool use_data_lengths = false,
                      bool use_label_lengths = false,
                      CTCLossBlankLabel blank_label = CTCLossBlankLabel::kFirst) {
  static const char *CTCLossBlankLabelValues[] = {
    "first",
    "last"
  };
  return Operator("CTCLoss")
           .SetParam("use_data_lengths", use_data_lengths)
           .SetParam("use_label_lengths", use_label_lengths)
           .SetParam("blank_label", CTCLossBlankLabelValues[int(blank_label)])
           .SetInput("data", data)
           .SetInput("label", label)
           .SetInput("data_lengths", data_lengths)
           .SetInput("label_lengths", label_lengths)
           .CreateSymbol();
}

/*!
 * \brief Computes 1D or 2D transposed convolution (aka fractionally strided convolution)
 *        of the input tensor. This operation can be seen as the gradient of Convolution
 *        operation with respect to its input. Convolution usually reduces the size of
 *        the input. Transposed convolution works the other way, going from a smaller
 * \param data Input tensor to the deconvolution operation.
 * \param weight Weights representing the kernel.
 * \param bias Bias added to the result after the deconvolution operation.
 * \param kernel Deconvolution kernel size: (w,), (h, w) or (d, h, w). This is same as
 * \param num_filter Number of output filters.
 * \param stride The stride used for the corresponding convolution: (w,), (h, w) or (d,
 * \param dilate Dilation factor for each dimension of the input: (w,), (h, w) or (d, h,
 * \param pad The amount of implicit zero padding added during convolution for each
 *        dimension of the input: (w,), (h, w) or (d, h, w). ``(kernel-1)/2`` is usually
 *        a good choice. If `target_shape` is set, `pad` will be ignored and a padding
 * \param adj Adjustment for output shape: (w,), (h, w) or (d, h, w). If `target_shape`
 * \param target_shape Shape of the output tensor: (w,), (h, w) or (d, h, w).
 * \param num_group Number of groups partition.
 * \param workspace Maximum temporary workspace allowed (MB) in deconvolution.This
 *        parameter has two usages. When CUDNN is not used, it determines the effective
 *        batch size of the deconvolution kernel. When CUDNN is used, it controls the
 *        maximum temporary storage used for tuning the best CUDNN kernel when
 * \param no_bias Whether to disable bias parameter.
 * \param cudnn_tune Whether to pick convolution algorithm by running performance test.
 * \param cudnn_off Turn off cudnn for this layer.
 * \param layout Set layout for input, output and weight. Empty for default layout, NCW
 * \return new symbol
 */
inline Symbol Deconvolution(Symbol data,
                            Symbol weight,
                            Symbol bias,
                            Shape kernel,
                            uint32_t num_filter,
                            Shape stride = Shape(),
                            Shape dilate = Shape(),
                            Shape pad = Shape(),
                            Shape adj = Shape(),
                            Shape target_shape = Shape(),
                            uint32_t num_group = 1,
                            uint64_t workspace = 512,
                            bool no_bias = true,
                            DeconvolutionCudnnTune cudnn_tune = DeconvolutionCudnnTune::kNone,
                            bool cudnn_off = false,
                            DeconvolutionLayout layout = DeconvolutionLayout::kNone) {
  static const char *DeconvolutionCudnnTuneValues[] = {
    "None",
    "fastest",
    "limited_workspace",
    "off"
  };
  static const char *DeconvolutionLayoutValues[] = {
    "None",
    "NCDHW",
    "NCHW",
    "NCW",
    "NDHWC",
    "NHWC"
  };
  return Operator("Deconvolution")
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("dilate", dilate)
           .SetParam("pad", pad)
           .SetParam("adj", adj)
           .SetParam("target_shape", target_shape)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
           .SetParam("cudnn_tune", DeconvolutionCudnnTuneValues[int(cudnn_tune)])
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("layout", DeconvolutionLayoutValues[int(layout)])
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol();
}

/*!
 * \brief Applies dropout operation to input array.
 *
 *        - During training, each element of the input is set to zero with probability p.
 *        The whole array is rescaled by :math:`1/(1-p)` to keep the expected
 *        sum of the input unchanged.
 *
 *        - During testing, this operator does not change the input if mode is 'training'.
 *        If mode is 'always', the same computaion as during training will be applied.
 *
 *        Example::
 *
 *        random.seed(998)
 *        input_array = array([[3., 0.5,  -0.5,  2., 7.],
 *        [2., -0.4,   7.,  3., 0.2]])
 *        a = symbol.Variable('a')
 *        dropout = symbol.Dropout(a, p = 0.2)
 *        executor = dropout.simple_bind(a = input_array.shape)
 *
 *        ## If training
 *        executor.forward(is_train = True, a = input_array)
 *        executor.outputs
 *        [[ 3.75   0.625 -0.     2.5    8.75 ]
 *        [ 2.5   -0.5    8.75   3.75   0.   ]]
 *
 *        ## If testing
 *        executor.forward(is_train = False, a = input_array)
 *        executor.outputs
 *        [[ 3.     0.5   -0.5    2.     7.   ]
 *        [ 2.    -0.4    7.     3.     0.2  ]]
 *
 *
 *        Defined in ../src/operator/nn/dropout.cc:L96
 * \param data Input array to which dropout will be applied.
 * \param p Fraction of the input that gets dropped out during training time.
 * \param mode Whether to only turn on dropout during training or to also turn on for
 * \param axes Axes for variational dropout kernel.
 * \param cudnn_off Whether to turn off cudnn in dropout operator. This option is ignored
 * \return new symbol
 */
inline Symbol Dropout(Symbol data,
                      mx_float p = 0.5,
                      DropoutMode mode = DropoutMode::kTraining,
                      Shape axes = Shape(),
                      dmlc::optional<bool> cudnn_off = dmlc::optional<bool>(0)) {
  static const char *DropoutModeValues[] = {
    "always",
    "training"
  };
  return Operator("Dropout")
           .SetParam("p", p)
           .SetParam("mode", DropoutModeValues[int(mode)])
           .SetParam("axes", axes)
           .SetParam("cudnn_off", cudnn_off)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Applies a linear transformation: :math:`Y = XW^T + b`.
 *
 *        If ``flatten`` is set to be true, then the shapes are:
 *
 *        - **data**: `(batch_size, x1, x2, ..., xn)`
 *        - **weight**: `(num_hidden, x1 * x2 * ... * xn)`
 *        - **bias**: `(num_hidden,)`
 *        - **out**: `(batch_size, num_hidden)`
 *
 *        If ``flatten`` is set to be false, then the shapes are:
 *
 *        - **data**: `(x1, x2, ..., xn, input_dim)`
 *        - **weight**: `(num_hidden, input_dim)`
 *        - **bias**: `(num_hidden,)`
 *        - **out**: `(x1, x2, ..., xn, num_hidden)`
 *
 *        The learnable parameters include both ``weight`` and ``bias``.
 *
 *        If ``no_bias`` is set to be true, then the ``bias`` term is ignored.
 *
 *        .. Note::
 *
 *        The sparse support for FullyConnected is limited to forward evaluation with
 *        weight and bias, where the length of `weight.indices` and `bias.indices` must
 *        to `num_hidden`. This could be useful for model inference with `row_sparse`
 *        trained with importance sampling or noise contrastive estimation.
 *
 *        To compute linear transformation with 'csr' sparse data, sparse.dot is
 *        of sparse.FullyConnected.
 *
 *
 *
 *        Defined in ../src/operator/nn/fully_connected.cc:L287
 * \param data Input data.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param num_hidden Number of hidden nodes of the output.
 * \param no_bias Whether to disable bias parameter.
 * \param flatten Whether to collapse all but the first axis of the input data tensor.
 * \return new symbol
 */
inline Symbol FullyConnected(Symbol data,
                             Symbol weight,
                             Symbol bias,
                             int num_hidden,
                             bool no_bias = false,
                             bool flatten = true) {
  return Operator("FullyConnected")
           .SetParam("num_hidden", num_hidden)
           .SetParam("no_bias", no_bias)
           .SetParam("flatten", flatten)
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol();
}

/*!
 * \brief Group normalization.
 *
 *        The input channels are separated into ``num_groups`` groups, each containing
 *        The mean and standard-deviation are calculated separately over the each group.
 *
 *        .. math::
 *
 *        data = data.reshape((N, num_groups, C // num_groups, ...))
 *        out = \frac{data - mean(data, axis)}{\sqrt{var(data, axis) + \epsilon}} * gamma
 *
 *        Both ``gamma`` and ``beta`` are learnable parameters.
 *
 *
 *
 *        Defined in ../src/operator/nn/group_norm.cc:L77
 * \param data Input data
 * \param gamma gamma array
 * \param beta beta array
 * \param num_groups Total number of groups.
 * \param eps An `epsilon` parameter to prevent division by 0.
 * \param output_mean_var Output the mean and std calculated along the given axis.
 * \return new symbol
 */
inline Symbol GroupNorm(Symbol data,
                        Symbol gamma,
                        Symbol beta,
                        int num_groups = 1,
                        mx_float eps = 9.99999975e-06,
                        bool output_mean_var = false) {
  return Operator("GroupNorm")
           .SetParam("num_groups", num_groups)
           .SetParam("eps", eps)
           .SetParam("output_mean_var", output_mean_var)
           .SetInput("data", data)
           .SetInput("gamma", gamma)
           .SetInput("beta", beta)
           .CreateSymbol();
}

/*!
 * \brief Extract sliding blocks from input array.
 *
 *        This operator is used in vanilla convolution implementation to transform the
 *        blocks on image to column matrix, then the convolution operation can be computed
 *        by matrix multiplication between column and convolution weight. Due to the close
 *        relation between im2col and convolution, the concept of **kernel**, **stride**,
 *        **dilate** and **pad** in this operator are inherited from convolution
 *
 *        Given the input data of shape :math:`(N, C, *)`, where :math:`N` is the batch
 *        :math:`C` is the channel size, and :math:`*` is the arbitrary spatial dimension,
 *        the output column array is always with shape :math:`(N, C \times
 *        where :math:`C \times \prod(\text{kernel})` is the block size, and :math:`W` is
 *        block number which is the spatial size of the convolution output with same
 *        Only 1-D, 2-D and 3-D of spatial dimension is supported in this operator.
 *
 *
 *
 *        Defined in ../src/operator/nn/im2col.cc:L100
 * \param data Input array to extract sliding blocks.
 * \param kernel Sliding kernel size: (w,), (h, w) or (d, h, w).
 * \param stride The stride between adjacent sliding blocks in spatial dimension: (w,),
 * \param dilate The spacing between adjacent kernel points: (w,), (h, w) or (d, h, w).
 * \param pad The zero-value padding size on both sides of spatial dimension: (w,), (h,
 * \return new symbol
 */
inline Symbol im2col(Symbol data,
                     Shape kernel,
                     Shape stride = Shape(),
                     Shape dilate = Shape(),
                     Shape pad = Shape()) {
  return Operator("im2col")
           .SetParam("kernel", kernel)
           .SetParam("stride", stride)
           .SetParam("dilate", dilate)
           .SetParam("pad", pad)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Combining the output column matrix of im2col back to image array.
 *
 *        Like :class:`~mxnet.ndarray.im2col`, this operator is also used in the vanilla
 *        implementation. Despite the name, col2im is not the reverse operation of
 *        may be overlaps between neighbouring sliding blocks, the column elements cannot
 *        put back into image. Instead, they are accumulated (i.e., summed) in the input
 *        just like the gradient computation, so col2im is the gradient of im2col and
 *
 *        Using the notation in im2col, given an input column array of shape
 *        :math:`(N, C \times  \prod(\text{kernel}), W)`, this operator accumulates the
 *        into output array of shape :math:`(N, C, \text{output_size}[0],
 *        Only 1-D, 2-D and 3-D of spatial dimension is supported in this operator.
 *
 *
 *
 *        Defined in ../src/operator/nn/im2col.cc:L182
 * \param data Input array to combine sliding blocks.
 * \param output_size The spatial dimension of image array: (w,), (h, w) or (d, h, w).
 * \param kernel Sliding kernel size: (w,), (h, w) or (d, h, w).
 * \param stride The stride between adjacent sliding blocks in spatial dimension: (w,),
 * \param dilate The spacing between adjacent kernel points: (w,), (h, w) or (d, h, w).
 * \param pad The zero-value padding size on both sides of spatial dimension: (w,), (h,
 * \return new symbol
 */
inline Symbol col2im(Symbol data,
                     Shape output_size,
                     Shape kernel,
                     Shape stride = Shape(),
                     Shape dilate = Shape(),
                     Shape pad = Shape()) {
  return Operator("col2im")
           .SetParam("output_size", output_size)
           .SetParam("kernel", kernel)
           .SetParam("stride", stride)
           .SetParam("dilate", dilate)
           .SetParam("pad", pad)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Layer normalization.
 *
 *        Normalizes the channels of the input tensor by mean and variance, and applies a
 *        well as offset ``beta``.
 *
 *        Assume the input has more than one dimension and we normalize along axis 1.
 *        We first compute the mean and variance along this axis and then
 *        compute the normalized output, which has the same shape as input, as following:
 *
 *        .. math::
 *
 *        out = \frac{data - mean(data, axis)}{\sqrt{var(data, axis) + \epsilon}} * gamma
 *
 *        Both ``gamma`` and ``beta`` are learnable parameters.
 *
 *        Unlike BatchNorm and InstanceNorm,  the *mean* and *var* are computed along the
 *
 *        Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
 *        have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both
 *        ``data_std``. Note that no gradient will be passed through these two outputs.
 *
 *        The parameter ``axis`` specifies which axis of the input shape denotes
 *        the 'channel' (separately normalized groups).  The default is -1, which sets
 *        axis to be the last item in the input shape.
 *
 *
 *
 *        Defined in ../src/operator/nn/layer_norm.cc:L159
 * \param data Input data to layer normalization
 * \param gamma gamma array
 * \param beta beta array
 * \param axis The axis to perform layer normalization. Usually, this should be be axis
 * \param eps An `epsilon` parameter to prevent division by 0.
 * \param output_mean_var Output the mean and std calculated along the given axis.
 * \return new symbol
 */
inline Symbol LayerNorm(Symbol data,
                        Symbol gamma,
                        Symbol beta,
                        int axis = -1,
                        mx_float eps = 9.99999975e-06,
                        bool output_mean_var = false) {
  return Operator("LayerNorm")
           .SetParam("axis", axis)
           .SetParam("eps", eps)
           .SetParam("output_mean_var", output_mean_var)
           .SetInput("data", data)
           .SetInput("gamma", gamma)
           .SetInput("beta", beta)
           .CreateSymbol();
}

/*!
 * \brief Computes the log softmax of the input.
 *        This is equivalent to computing softmax followed by log.
 *
 *        Examples::
 *
 *        >>> x = mx.nd.array([1, 2, .1])
 *        >>> mx.nd.log_softmax(x).asnumpy()
 *        array([-1.41702998, -0.41702995, -2.31702995], dtype=float32)
 *
 *        >>> x = mx.nd.array( [[1, 2, .1],[.1, 2, 1]] )
 *        >>> mx.nd.log_softmax(x, axis=0).asnumpy()
 *        array([[-0.34115392, -0.69314718, -1.24115396],
 *        [-1.24115396, -0.69314718, -0.34115392]], dtype=float32)
 *
 *
 *
 * \param data The input array.
 * \param axis The axis along which to compute softmax.
 * \param temperature Temperature parameter in softmax
 * \param dtype DType of the output in case this can't be inferred. Defaults to the same
 * \param use_length Whether to use the length input as a mask over the data input.
 * \return new symbol
 */
inline Symbol log_softmax(Symbol data,
                          int axis = -1,
                          dmlc::optional<double> temperature = dmlc::optional<double>(),
                          Log_softmaxDtype dtype = Log_softmaxDtype::kNone,
                          dmlc::optional<bool> use_length = dmlc::optional<bool>(0)) {
  static const char *Log_softmaxDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("log_softmax")
           .SetParam("axis", axis)
           .SetParam("temperature", temperature)
           .SetParam("dtype", Log_softmaxDtypeValues[int(dtype)])
           .SetParam("use_length", use_length)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Applies local response normalization to the input.
 *
 *        The local response normalization layer performs "lateral inhibition" by
 *        over local input regions.
 *
 *        If :math:`a_{x,y}^{i}` is the activity of a neuron computed by applying kernel
 *        :math:`(x, y)` and then applying the ReLU nonlinearity, the response-normalized
 *        activity :math:`b_{x,y}^{i}` is given by the expression:
 *
 *        .. math::
 *        b_{x,y}^{i} = \frac{a_{x,y}^{i}}{\Bigg({k + \frac{\alpha}{n} \sum_{j=max(0,
 *
 *        where the sum runs over :math:`n` "adjacent" kernel maps at the same spatial
 *        number of kernels in the layer.
 *
 *
 *
 *        Defined in ../src/operator/nn/lrn.cc:L158
 * \param data Input data to LRN
 * \param nsize normalization window width in elements.
 * \param alpha The variance scaling parameter :math:`lpha` in the LRN expression.
 * \param beta The power parameter :math:`eta` in the LRN expression.
 * \param knorm The parameter :math:`k` in the LRN expression.
 * \return new symbol
 */
inline Symbol LRN(Symbol data,
                  uint32_t nsize,
                  mx_float alpha = 9.99999975e-05,
                  mx_float beta = 0.75,
                  mx_float knorm = 2) {
  return Operator("LRN")
           .SetParam("nsize", nsize)
           .SetParam("alpha", alpha)
           .SetParam("beta", beta)
           .SetParam("knorm", knorm)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief
 *        Calculate the mean and variance of `data`.
 *
 *        The mean and variance are calculated by aggregating the contents of data across
 *        If x is 1-D and axes = [0] this is just the mean and variance of a vector.
 *
 *        Example:
 *
 *        x = [[1, 2, 3], [4, 5, 6]]
 *        mean, var = moments(data=x, axes=[0])
 *        mean = [2.5, 3.5, 4.5]
 *        var = [2.25, 2.25, 2.25]
 *        mean, var = moments(data=x, axes=[1])
 *        mean = [2.0, 5.0]
 *        var = [0.66666667, 0.66666667]
 *        mean, var = moments(data=x, axis=[0, 1])
 *        mean = [3.5]
 *        var = [2.9166667]
 *
 *
 *
 *        Defined in ../src/operator/nn/moments.cc:L54
 * \param data Input ndarray
 * \param axes Array of ints. Axes along which to compute mean and variance.
 * \param keepdims produce moments with the same dimensionality as the input.
 * \return new symbol
 */
inline Symbol moments(Symbol data,
                      dmlc::optional<Shape> axes = dmlc::optional<Shape>(),
                      bool keepdims = false) {
  return Operator("moments")
           .SetParam("axes", axes)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Performs pooling on the input.
 *
 *        The shapes for 1-D pooling are
 *
 *        - **data** and **out**: *(batch_size, channel, width)* (NCW layout) or
 *        *(batch_size, width, channel)* (NWC layout),
 *
 *        The shapes for 2-D pooling are
 *
 *        - **data** and **out**: *(batch_size, channel, height, width)* (NCHW layout) or
 *        *(batch_size, height, width, channel)* (NHWC layout),
 *
 *        out_height = f(height, kernel[0], pad[0], stride[0])
 *        out_width = f(width, kernel[1], pad[1], stride[1])
 *
 *        The definition of *f* depends on ``pooling_convention``, which has two options:
 *
 *        - **valid** (default)::
 *
 *        f(x, k, p, s) = floor((x+2*p-k)/s)+1
 *
 *        - **full**, which is compatible with Caffe::
 *
 *        f(x, k, p, s) = ceil((x+2*p-k)/s)+1
 *
 *        When ``global_pool`` is set to be true, then global pooling is performed. It
 *        ``kernel=(height, width)`` and set the appropiate padding to 0.
 *
 *        Three pooling options are supported by ``pool_type``:
 *
 *        - **avg**: average pooling
 *        - **max**: max pooling
 *        - **sum**: sum pooling
 *        - **lp**: Lp pooling
 *
 *        For 3-D pooling, an additional *depth* dimension is added before
 *        *height*. Namely the input data and output will have shape *(batch_size,
 *        height, width)* (NCDHW layout) or *(batch_size, depth, height, width, channel)*
 *
 *        Notes on Lp pooling:
 *
 *        Lp pooling was first introduced by this paper:
 *        L-1 pooling is simply sum pooling, while L-inf pooling is simply max pooling.
 *        We can see that Lp pooling stands between those two, in practice the most
 *
 *        For each window ``X``, the mathematical expression for Lp pooling is:
 *
 *        :math:`f(X) = \sqrt[p]{\sum_{x}^{X} x^p}`
 *
 *
 *
 *        Defined in ../src/operator/nn/pooling.cc:L417
 * \param data Input data to the pooling operator.
 * \param kernel Pooling kernel size: (y, x) or (d, y, x)
 * \param pool_type Pooling type to be applied.
 * \param global_pool Ignore kernel size, do global pooling based on current input
 * \param cudnn_off Turn off cudnn pooling and use MXNet pooling operator.
 * \param pooling_convention Pooling convention to be applied.
 * \param stride Stride: for pooling (y, x) or (d, y, x). Defaults to 1 for each
 * \param pad Pad for pooling: (y, x) or (d, y, x). Defaults to no padding.
 * \param p_value Value of p for Lp pooling, can be 1 or 2, required for Lp Pooling.
 * \param count_include_pad Only used for AvgPool, specify whether to count padding
 *        elements for averagecalculation. For example, with a 5*5 kernel on a 3*3 corner
 *        of a image,the sum of the 9 valid elements will be divided by 25 if this is set
 * \param layout Set layout for input and output. Empty for
 *        default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.
 * \return new symbol
 */
inline Symbol Pooling(Symbol data,
                      Shape kernel = Shape(),
                      PoolingPoolType pool_type = PoolingPoolType::kMax,
                      bool global_pool = false,
                      bool cudnn_off = false,
                      PoolingPoolingConvention pooling_convention = PoolingPoolingConvention::kValid,
                      Shape stride = Shape(),
                      Shape pad = Shape(),
                      dmlc::optional<int> p_value = dmlc::optional<int>(),
                      dmlc::optional<bool> count_include_pad = dmlc::optional<bool>(),
                      PoolingLayout layout = PoolingLayout::kNone) {
  static const char *PoolingPoolTypeValues[] = {
    "avg",
    "lp",
    "max",
    "sum"
  };
  static const char *PoolingPoolingConventionValues[] = {
    "full",
    "same",
    "valid"
  };
  static const char *PoolingLayoutValues[] = {
    "None",
    "NCDHW",
    "NCHW",
    "NCW",
    "NDHWC",
    "NHWC",
    "NWC"
  };
  return Operator("Pooling")
           .SetParam("kernel", kernel)
           .SetParam("pool_type", PoolingPoolTypeValues[int(pool_type)])
           .SetParam("global_pool", global_pool)
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("pooling_convention", PoolingPoolingConventionValues[int(pooling_convention)])
           .SetParam("stride", stride)
           .SetParam("pad", pad)
           .SetParam("p_value", p_value)
           .SetParam("count_include_pad", count_include_pad)
           .SetParam("layout", PoolingLayoutValues[int(layout)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Applies the softmax function.
 *
 *        The resulting array contains elements in the range (0,1) and the elements along
 *
 *        .. math::
 *        softmax(\mathbf{z/t})_j = \frac{e^{z_j/t}}{\sum_{k=1}^K e^{z_k/t}}
 *
 *        for :math:`j = 1, ..., K`
 *
 *        t is the temperature parameter in softmax function. By default, t equals 1.0
 *
 *        Example::
 *
 *        x = [[ 1.  1.  1.]
 *        [ 1.  1.  1.]]
 *
 *        softmax(x,axis=0) = [[ 0.5  0.5  0.5]
 *        [ 0.5  0.5  0.5]]
 *
 *        softmax(x,axis=1) = [[ 0.33333334,  0.33333334,  0.33333334],
 *        [ 0.33333334,  0.33333334,  0.33333334]]
 *
 *
 *
 *        Defined in ../src/operator/nn/softmax.cc:L136
 * \param data The input array.
 * \param length The length array.
 * \param axis The axis along which to compute softmax.
 * \param temperature Temperature parameter in softmax
 * \param dtype DType of the output in case this can't be inferred. Defaults to the same
 * \param use_length Whether to use the length input as a mask over the data input.
 * \return new symbol
 */
inline Symbol softmax(Symbol data,
                      Symbol length,
                      int axis = -1,
                      dmlc::optional<double> temperature = dmlc::optional<double>(),
                      SoftmaxDtype dtype = SoftmaxDtype::kNone,
                      dmlc::optional<bool> use_length = dmlc::optional<bool>(0)) {
  static const char *SoftmaxDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("softmax")
           .SetParam("axis", axis)
           .SetParam("temperature", temperature)
           .SetParam("dtype", SoftmaxDtypeValues[int(dtype)])
           .SetParam("use_length", use_length)
           .SetInput("data", data)
           .SetInput("length", length)
           .CreateSymbol();
}

/*!
 * \brief Applies softmax activation to input. This is intended for internal layers.
 *
 *        .. note::
 *
 *        This operator has been deprecated, please use `softmax`.
 *
 *        If `mode` = ``instance``, this operator will compute a softmax for each
 *        This is the default mode.
 *
 *        If `mode` = ``channel``, this operator will compute a k-class softmax at each
 *        of each instance, where `k` = ``num_channel``. This mode can only be used when
 *        has at least 3 dimensions.
 *        This can be used for `fully convolutional network`, `image segmentation`, etc.
 *
 *        Example::
 *
 *        >>> input_array = mx.nd.array([[3., 0.5, -0.5, 2., 7.],
 *        >>>                            [2., -.4, 7.,   3., 0.2]])
 *        >>> softmax_act = mx.nd.SoftmaxActivation(input_array)
 *        >>> print softmax_act.asnumpy()
 *        [[  1.78322066e-02   1.46375655e-03   5.38485940e-04   6.56010211e-03
 *        [  6.56221947e-03   5.95310994e-04   9.73919690e-01   1.78379621e-02
 *
 *
 *
 *        Defined in ../src/operator/nn/softmax_activation.cc:L59
 * \param data The input array.
 * \param mode Specifies how to compute the softmax. If set to ``instance``, it computes
 *        softmax for each instance. If set to ``channel``, It computes cross channel
 * \return new symbol
 */
inline Symbol SoftmaxActivation(Symbol data,
                                SoftmaxActivationMode mode = SoftmaxActivationMode::kInstance) {
  static const char *SoftmaxActivationModeValues[] = {
    "channel",
    "instance"
  };
  return Operator("SoftmaxActivation")
           .SetParam("mode", SoftmaxActivationModeValues[int(mode)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Applies the softmin function.
 *
 *        The resulting array contains elements in the range (0,1) and the elements along
 *        up to 1.
 *
 *        .. math::
 *        softmin(\mathbf{z/t})_j = \frac{e^{-z_j/t}}{\sum_{k=1}^K e^{-z_k/t}}
 *
 *        for :math:`j = 1, ..., K`
 *
 *        t is the temperature parameter in softmax function. By default, t equals 1.0
 *
 *        Example::
 *
 *        x = [[ 1.  2.  3.]
 *        [ 3.  2.  1.]]
 *
 *        softmin(x,axis=0) = [[ 0.88079703,  0.5,  0.11920292],
 *        [ 0.11920292,  0.5,  0.88079703]]
 *
 *        softmin(x,axis=1) = [[ 0.66524094,  0.24472848,  0.09003057],
 *        [ 0.09003057,  0.24472848,  0.66524094]]
 *
 *
 *
 *        Defined in ../src/operator/nn/softmin.cc:L57
 * \param data The input array.
 * \param axis The axis along which to compute softmax.
 * \param temperature Temperature parameter in softmax
 * \param dtype DType of the output in case this can't be inferred. Defaults to the same
 * \param use_length Whether to use the length input as a mask over the data input.
 * \return new symbol
 */
inline Symbol softmin(Symbol data,
                      int axis = -1,
                      dmlc::optional<double> temperature = dmlc::optional<double>(),
                      SoftminDtype dtype = SoftminDtype::kNone,
                      dmlc::optional<bool> use_length = dmlc::optional<bool>(0)) {
  static const char *SoftminDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64"
  };
  return Operator("softmin")
           .SetParam("axis", axis)
           .SetParam("temperature", temperature)
           .SetParam("dtype", SoftminDtypeValues[int(dtype)])
           .SetParam("use_length", use_length)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Upsamples the given input data.
 *
 *        Two algorithms (``sample_type``) are available for upsampling:
 *
 *        - Nearest Neighbor
 *        - Bilinear
 *
 *        **Nearest Neighbor Upsampling**
 *
 *        Input data is expected to be NCHW.
 *
 *        Example::
 *
 *        x = [[[[1. 1. 1.]
 *        [1. 1. 1.]
 *        [1. 1. 1.]]]]
 *
 *        UpSampling(x, scale=2, sample_type='nearest') = [[[[1. 1. 1. 1. 1. 1.]
 *        [1. 1. 1. 1. 1. 1.]
 *        [1. 1. 1. 1. 1. 1.]
 *        [1. 1. 1. 1. 1. 1.]
 *        [1. 1. 1. 1. 1. 1.]
 *        [1. 1. 1. 1. 1. 1.]]]]
 *
 *        **Bilinear Upsampling**
 *
 *        Uses `deconvolution` algorithm under the hood. You need provide both input data
 *
 *        Input data is expected to be NCHW.
 *
 *        `num_filter` is expected to be same as the number of channels.
 *
 *        Example::
 *
 *        x = [[[[1. 1. 1.]
 *        [1. 1. 1.]
 *        [1. 1. 1.]]]]
 *
 *        w = [[[[1. 1. 1. 1.]
 *        [1. 1. 1. 1.]
 *        [1. 1. 1. 1.]
 *        [1. 1. 1. 1.]]]]
 *
 *        UpSampling(x, w, scale=2, sample_type='bilinear', num_filter=1) = [[[[1. 2. 2.
 *        [2. 4. 4. 4. 4. 2.]
 *        [2. 4. 4. 4. 4. 2.]
 *        [2. 4. 4. 4. 4. 2.]
 *        [2. 4. 4. 4. 4. 2.]
 *        [1. 2. 2. 2. 2. 1.]]]]
 *
 *
 *        Defined in ../src/operator/nn/upsampling.cc:L173
 * \param data Array of tensors to upsample. For bilinear upsampling, there should be 2
 * \param scale Up sampling scale
 * \param sample_type upsampling method
 * \param num_args Number of inputs to be upsampled. For nearest neighbor upsampling,
 *        this can be 1-N; the size of output will be(scale*h_0,scale*w_0) and all other
 *        inputs will be upsampled to thesame size. For bilinear upsampling this must be
 * \param num_filter Input filter. Only used by bilinear sample_type.Since bilinear
 * \param multi_input_mode How to handle multiple input. concat means concatenate
 *        upsampled images along the channel dimension. sum means add all images
 * \param workspace Tmp workspace for deconvolution (MB)
 * \return new symbol
 */
inline Symbol UpSampling(const std::vector<Symbol>& data,
                         int scale,
                         UpSamplingSampleType sample_type,
                         int num_args,
                         int num_filter = 0,
                         UpSamplingMultiInputMode multi_input_mode = UpSamplingMultiInputMode::kConcat,
                         uint64_t workspace = 512) {
  static const char *UpSamplingSampleTypeValues[] = {
    "bilinear",
    "nearest"
  };
  static const char *UpSamplingMultiInputModeValues[] = {
    "concat",
    "sum"
  };
  return Operator("UpSampling")
           .SetParam("scale", scale)
           .SetParam("sample_type", UpSamplingSampleTypeValues[int(sample_type)])
           .SetParam("num_args", num_args)
           .SetParam("num_filter", num_filter)
           .SetParam("multi_input_mode", UpSamplingMultiInputModeValues[int(multi_input_mode)])
           .SetParam("workspace", workspace)
(data)
           .CreateSymbol();
}

/*!
 * \brief Update function for SignSGD optimizer.
 *
 *        .. math::
 *
 *        g_t = \nabla J(W_{t-1})\\
 *        W_t = W_{t-1} - \eta_t \text{sign}(g_t)
 *
 *        It updates the weights using::
 *
 *        weight = weight - learning_rate * sign(gradient)
 *
 *        .. note::
 *        - sparse ndarray not supported for this optimizer yet.
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L63
 * \param weight Weight
 * \param grad Gradient
 * \param lr Learning rate
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \return new symbol
 */
inline Symbol signsgd_update(Symbol weight,
                             Symbol grad,
                             mx_float lr,
                             mx_float wd = 0,
                             mx_float rescale_grad = 1,
                             mx_float clip_gradient = -1) {
  return Operator("signsgd_update")
           .SetParam("lr", lr)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .CreateSymbol();
}

/*!
 * \brief SIGN momentUM (Signum) optimizer.
 *
 *        .. math::
 *
 *        g_t = \nabla J(W_{t-1})\\
 *        m_t = \beta m_{t-1} + (1 - \beta) g_t\\
 *        W_t = W_{t-1} - \eta_t \text{sign}(m_t)
 *
 *        It updates the weights using::
 *        state = momentum * state + (1-momentum) * gradient
 *        weight = weight - learning_rate * sign(state)
 *
 *        Where the parameter ``momentum`` is the decay rate of momentum estimates at
 *
 *        .. note::
 *        - sparse ndarray not supported for this optimizer yet.
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L92
 * \param weight Weight
 * \param grad Gradient
 * \param mom Momentum
 * \param lr Learning rate
 * \param momentum The decay rate of momentum estimates at each epoch.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param wd_lh The amount of weight decay that does not go into gradient/momentum
 * \return new symbol
 */
inline Symbol signum_update(Symbol weight,
                            Symbol grad,
                            Symbol mom,
                            mx_float lr,
                            mx_float momentum = 0,
                            mx_float wd = 0,
                            mx_float rescale_grad = 1,
                            mx_float clip_gradient = -1,
                            mx_float wd_lh = 0) {
  return Operator("signum_update")
           .SetParam("lr", lr)
           .SetParam("momentum", momentum)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("wd_lh", wd_lh)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("mom", mom)
           .CreateSymbol();
}

/*!
 * \brief Update function for Stochastic Gradient Descent (SDG) optimizer.
 *
 *        It updates the weights using::
 *
 *        weight = weight - learning_rate * (gradient + wd * weight)
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L329
 * \param data Weights
 * \param lrs Learning rates.
 * \param wds Weight decay augments the objective function with a regularization term
 *        that penalizes large weights. The penalty scales with the square of the
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param num_weights Number of updated weights.
 * \return new symbol
 */
inline Symbol multi_sgd_update(const std::vector<Symbol>& data,
                               nnvm::Tuple<mx_float> lrs,
                               nnvm::Tuple<mx_float> wds,
                               mx_float rescale_grad = 1,
                               mx_float clip_gradient = -1,
                               int num_weights = 1) {
  return Operator("multi_sgd_update")
           .SetParam("lrs", lrs)
           .SetParam("wds", wds)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("num_weights", num_weights)
(data)
           .CreateSymbol();
}

/*!
 * \brief Momentum update function for Stochastic Gradient Descent (SGD) optimizer.
 *
 *        Momentum update has better convergence rates on neural networks. Mathematically
 *        like below:
 *
 *        .. math::
 *
 *        v_1 = \alpha * \nabla J(W_0)\\
 *        v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\
 *        W_t = W_{t-1} + v_t
 *
 *        It updates the weights using::
 *
 *        v = momentum * v - learning_rate * gradient
 *        weight += v
 *
 *        Where the parameter ``momentum`` is the decay rate of momentum estimates at
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L374
 * \param data Weights, gradients and momentum
 * \param lrs Learning rates.
 * \param wds Weight decay augments the objective function with a regularization term
 *        that penalizes large weights. The penalty scales with the square of the
 * \param momentum The decay rate of momentum estimates at each epoch.
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param num_weights Number of updated weights.
 * \return new symbol
 */
inline Symbol multi_sgd_mom_update(const std::vector<Symbol>& data,
                                   nnvm::Tuple<mx_float> lrs,
                                   nnvm::Tuple<mx_float> wds,
                                   mx_float momentum = 0,
                                   mx_float rescale_grad = 1,
                                   mx_float clip_gradient = -1,
                                   int num_weights = 1) {
  return Operator("multi_sgd_mom_update")
           .SetParam("lrs", lrs)
           .SetParam("wds", wds)
           .SetParam("momentum", momentum)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("num_weights", num_weights)
(data)
           .CreateSymbol();
}

/*!
 * \brief Update function for multi-precision Stochastic Gradient Descent (SDG) optimizer.
 *
 *        It updates the weights using::
 *
 *        weight = weight - learning_rate * (gradient + wd * weight)
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L417
 * \param data Weights
 * \param lrs Learning rates.
 * \param wds Weight decay augments the objective function with a regularization term
 *        that penalizes large weights. The penalty scales with the square of the
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param num_weights Number of updated weights.
 * \return new symbol
 */
inline Symbol multi_mp_sgd_update(const std::vector<Symbol>& data,
                                  nnvm::Tuple<mx_float> lrs,
                                  nnvm::Tuple<mx_float> wds,
                                  mx_float rescale_grad = 1,
                                  mx_float clip_gradient = -1,
                                  int num_weights = 1) {
  return Operator("multi_mp_sgd_update")
           .SetParam("lrs", lrs)
           .SetParam("wds", wds)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("num_weights", num_weights)
(data)
           .CreateSymbol();
}

/*!
 * \brief Momentum update function for multi-precision Stochastic Gradient Descent (SGD)
 *
 *        Momentum update has better convergence rates on neural networks. Mathematically
 *        like below:
 *
 *        .. math::
 *
 *        v_1 = \alpha * \nabla J(W_0)\\
 *        v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\
 *        W_t = W_{t-1} + v_t
 *
 *        It updates the weights using::
 *
 *        v = momentum * v - learning_rate * gradient
 *        weight += v
 *
 *        Where the parameter ``momentum`` is the decay rate of momentum estimates at
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L472
 * \param data Weights
 * \param lrs Learning rates.
 * \param wds Weight decay augments the objective function with a regularization term
 *        that penalizes large weights. The penalty scales with the square of the
 * \param momentum The decay rate of momentum estimates at each epoch.
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param num_weights Number of updated weights.
 * \return new symbol
 */
inline Symbol multi_mp_sgd_mom_update(const std::vector<Symbol>& data,
                                      nnvm::Tuple<mx_float> lrs,
                                      nnvm::Tuple<mx_float> wds,
                                      mx_float momentum = 0,
                                      mx_float rescale_grad = 1,
                                      mx_float clip_gradient = -1,
                                      int num_weights = 1) {
  return Operator("multi_mp_sgd_mom_update")
           .SetParam("lrs", lrs)
           .SetParam("wds", wds)
           .SetParam("momentum", momentum)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("num_weights", num_weights)
(data)
           .CreateSymbol();
}

/*!
 * \brief Update function for Stochastic Gradient Descent (SGD) optimizer.
 *
 *        It updates the weights using::
 *
 *        weight = weight - learning_rate * (gradient + wd * weight)
 *
 *        However, if gradient is of ``row_sparse`` storage type and ``lazy_update`` is
 *        only the row slices whose indices appear in grad.indices are updated::
 *
 *        for row in gradient.indices:
 *        weight[row] = weight[row] - learning_rate * (gradient[row] + wd * weight[row])
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L524
 * \param weight Weight
 * \param grad Gradient
 * \param lr Learning rate
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param lazy_update If true, lazy updates are applied if gradient's stype is row_sparse.
 * \return new symbol
 */
inline Symbol sgd_update(Symbol weight,
                         Symbol grad,
                         mx_float lr,
                         mx_float wd = 0,
                         mx_float rescale_grad = 1,
                         mx_float clip_gradient = -1,
                         bool lazy_update = true) {
  return Operator("sgd_update")
           .SetParam("lr", lr)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("lazy_update", lazy_update)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .CreateSymbol();
}

/*!
 * \brief Momentum update function for Stochastic Gradient Descent (SGD) optimizer.
 *
 *        Momentum update has better convergence rates on neural networks. Mathematically
 *        like below:
 *
 *        .. math::
 *
 *        v_1 = \alpha * \nabla J(W_0)\\
 *        v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\
 *        W_t = W_{t-1} + v_t
 *
 *        It updates the weights using::
 *
 *        v = momentum * v - learning_rate * gradient
 *        weight += v
 *
 *        Where the parameter ``momentum`` is the decay rate of momentum estimates at
 *
 *        However, if grad's storage type is ``row_sparse``, ``lazy_update`` is True and
 *        type is the same as momentum's storage type,
 *        only the row slices whose indices appear in grad.indices are updated (for both
 *
 *        for row in gradient.indices:
 *        v[row] = momentum[row] * v[row] - learning_rate * gradient[row]
 *        weight[row] += v[row]
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L565
 * \param weight Weight
 * \param grad Gradient
 * \param mom Momentum
 * \param lr Learning rate
 * \param momentum The decay rate of momentum estimates at each epoch.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param lazy_update If true, lazy updates are applied if gradient's stype is row_sparse
 * \return new symbol
 */
inline Symbol sgd_mom_update(Symbol weight,
                             Symbol grad,
                             Symbol mom,
                             mx_float lr,
                             mx_float momentum = 0,
                             mx_float wd = 0,
                             mx_float rescale_grad = 1,
                             mx_float clip_gradient = -1,
                             bool lazy_update = true) {
  return Operator("sgd_mom_update")
           .SetParam("lr", lr)
           .SetParam("momentum", momentum)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("lazy_update", lazy_update)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("mom", mom)
           .CreateSymbol();
}

/*!
 * \brief Updater function for multi-precision sgd optimizer
 * \param weight Weight
 * \param grad gradient
 * \param weight32 Weight32
 * \param lr Learning rate
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param lazy_update If true, lazy updates are applied if gradient's stype is row_sparse.
 * \return new symbol
 */
inline Symbol mp_sgd_update(Symbol weight,
                            Symbol grad,
                            Symbol weight32,
                            mx_float lr,
                            mx_float wd = 0,
                            mx_float rescale_grad = 1,
                            mx_float clip_gradient = -1,
                            bool lazy_update = true) {
  return Operator("mp_sgd_update")
           .SetParam("lr", lr)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("lazy_update", lazy_update)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("weight32", weight32)
           .CreateSymbol();
}

/*!
 * \brief Updater function for multi-precision sgd optimizer
 * \param weight Weight
 * \param grad Gradient
 * \param mom Momentum
 * \param weight32 Weight32
 * \param lr Learning rate
 * \param momentum The decay rate of momentum estimates at each epoch.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param lazy_update If true, lazy updates are applied if gradient's stype is row_sparse
 * \return new symbol
 */
inline Symbol mp_sgd_mom_update(Symbol weight,
                                Symbol grad,
                                Symbol mom,
                                Symbol weight32,
                                mx_float lr,
                                mx_float momentum = 0,
                                mx_float wd = 0,
                                mx_float rescale_grad = 1,
                                mx_float clip_gradient = -1,
                                bool lazy_update = true) {
  return Operator("mp_sgd_mom_update")
           .SetParam("lr", lr)
           .SetParam("momentum", momentum)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("lazy_update", lazy_update)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("mom", mom)
           .SetInput("weight32", weight32)
           .CreateSymbol();
}

/*!
 * \brief The FTML optimizer described in
 *        *FTML - Follow the Moving Leader in Deep Learning*,
 *        available at http://proceedings.mlr.press/v70/zheng17a/zheng17a.pdf.
 *
 *        .. math::
 *
 *        g_t = \nabla J(W_{t-1})\\
 *        v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\
 *        d_t = \frac{ 1 - \beta_1^t }{ \eta_t } (\sqrt{ \frac{ v_t }{ 1 - \beta_2^t } }
 *        \sigma_t = d_t - \beta_1 d_{t-1}
 *        z_t = \beta_1 z_{ t-1 } + (1 - \beta_1^t) g_t - \sigma_t W_{t-1}
 *        W_t = - \frac{ z_t }{ d_t }
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L640
 * \param weight Weight
 * \param grad Gradient
 * \param d Internal state ``d_t``
 * \param v Internal state ``v_t``
 * \param z Internal state ``z_t``
 * \param lr Learning rate.
 * \param t Number of update.
 * \param beta1 Generally close to 0.5.
 * \param beta2 Generally close to 1.
 * \param epsilon Epsilon to prevent div 0.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_grad Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \return new symbol
 */
inline Symbol ftml_update(Symbol weight,
                          Symbol grad,
                          Symbol d,
                          Symbol v,
                          Symbol z,
                          mx_float lr,
                          int t,
                          mx_float beta1 = 0.600000024,
                          mx_float beta2 = 0.999000013,
                          double epsilon = 9.9999999392252903e-09,
                          mx_float wd = 0,
                          mx_float rescale_grad = 1,
                          mx_float clip_grad = -1) {
  return Operator("ftml_update")
           .SetParam("lr", lr)
           .SetParam("t", t)
           .SetParam("beta1", beta1)
           .SetParam("beta2", beta2)
           .SetParam("epsilon", epsilon)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_grad", clip_grad)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("d", d)
           .SetInput("v", v)
           .SetInput("z", z)
           .CreateSymbol();
}

/*!
 * \brief Update function for Adam optimizer. Adam is seen as a generalization
 *        of AdaGrad.
 *
 *        Adam update consists of the following steps, where g represents gradient and m,
 *        are 1st and 2nd order moment estimates (mean and variance).
 *
 *        .. math::
 *
 *        g_t = \nabla J(W_{t-1})\\
 *        m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\\
 *        v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\
 *        W_t = W_{t-1} - \alpha \frac{ m_t }{ \sqrt{ v_t } + \epsilon }
 *
 *        It updates the weights using::
 *
 *        m = beta1*m + (1-beta1)*grad
 *        v = beta2*v + (1-beta2)*(grad**2)
 *        w += - learning_rate * m / (sqrt(v) + epsilon)
 *
 *        However, if grad's storage type is ``row_sparse``, ``lazy_update`` is True and
 *        type of weight is the same as those of m and v,
 *        only the row slices whose indices appear in grad.indices are updated (for w, m
 *
 *        for row in grad.indices:
 *        m[row] = beta1*m[row] + (1-beta1)*grad[row]
 *        v[row] = beta2*v[row] + (1-beta2)*(grad[row]**2)
 *        w[row] += - learning_rate * m[row] / (sqrt(v[row]) + epsilon)
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L688
 * \param weight Weight
 * \param grad Gradient
 * \param mean Moving mean
 * \param var Moving variance
 * \param lr Learning rate
 * \param beta1 The decay rate for the 1st moment estimates.
 * \param beta2 The decay rate for the 2nd moment estimates.
 * \param epsilon A small constant for numerical stability.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param lazy_update If true, lazy updates are applied if gradient's stype is row_sparse
 * \return new symbol
 */
inline Symbol adam_update(Symbol weight,
                          Symbol grad,
                          Symbol mean,
                          Symbol var,
                          mx_float lr,
                          mx_float beta1 = 0.899999976,
                          mx_float beta2 = 0.999000013,
                          mx_float epsilon = 9.99999994e-09,
                          mx_float wd = 0,
                          mx_float rescale_grad = 1,
                          mx_float clip_gradient = -1,
                          bool lazy_update = true) {
  return Operator("adam_update")
           .SetParam("lr", lr)
           .SetParam("beta1", beta1)
           .SetParam("beta2", beta2)
           .SetParam("epsilon", epsilon)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("lazy_update", lazy_update)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("mean", mean)
           .SetInput("var", var)
           .CreateSymbol();
}

/*!
 * \brief Update function for Nesterov Accelerated Gradient( NAG) optimizer.
 *        It updates the weights using the following formula,
 *
 *        .. math::
 *        v_t = \gamma v_{t-1} + \eta * \nabla J(W_{t-1} - \gamma v_{t-1})\\
 *        W_t = W_{t-1} - v_t
 *
 *        Where
 *        :math:`\eta` is the learning rate of the optimizer
 *        :math:`\gamma` is the decay rate of the momentum estimate
 *        :math:`\v_t` is the update vector at time step `t`
 *        :math:`\W_t` is the weight vector at time step `t`
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L726
 * \param weight Weight
 * \param grad Gradient
 * \param mom Momentum
 * \param lr Learning rate
 * \param momentum The decay rate of momentum estimates at each epoch.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \return new symbol
 */
inline Symbol nag_mom_update(Symbol weight,
                             Symbol grad,
                             Symbol mom,
                             mx_float lr,
                             mx_float momentum = 0,
                             mx_float wd = 0,
                             mx_float rescale_grad = 1,
                             mx_float clip_gradient = -1) {
  return Operator("nag_mom_update")
           .SetParam("lr", lr)
           .SetParam("momentum", momentum)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("mom", mom)
           .CreateSymbol();
}

/*!
 * \brief Update function for multi-precision Nesterov Accelerated Gradient( NAG)
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L745
 * \param weight Weight
 * \param grad Gradient
 * \param mom Momentum
 * \param weight32 Weight32
 * \param lr Learning rate
 * \param momentum The decay rate of momentum estimates at each epoch.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \return new symbol
 */
inline Symbol mp_nag_mom_update(Symbol weight,
                                Symbol grad,
                                Symbol mom,
                                Symbol weight32,
                                mx_float lr,
                                mx_float momentum = 0,
                                mx_float wd = 0,
                                mx_float rescale_grad = 1,
                                mx_float clip_gradient = -1) {
  return Operator("mp_nag_mom_update")
           .SetParam("lr", lr)
           .SetParam("momentum", momentum)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("mom", mom)
           .SetInput("weight32", weight32)
           .CreateSymbol();
}

/*!
 * \brief Update function for `RMSProp` optimizer.
 *
 *        `RMSprop` is a variant of stochastic gradient descent where the gradients are
 *        divided by a cache which grows with the sum of squares of recent gradients?
 *
 *        `RMSProp` is similar to `AdaGrad`, a popular variant of `SGD` which adaptively
 *        tunes the learning rate of each parameter. `AdaGrad` lowers the learning rate
 *        each parameter monotonically over the course of training.
 *        While this is analytically motivated for convex optimizations, it may not be
 *        for non-convex problems. `RMSProp` deals with this heuristically by allowing the
 *        learning rates to rebound as the denominator decays over time.
 *
 *        Define the Root Mean Square (RMS) error criterion of the gradient as
 *        :math:`RMS[g]_t = \sqrt{E[g^2]_t + \epsilon}`, where :math:`g` represents
 *        gradient and :math:`E[g^2]_t` is the decaying average over past squared
 *
 *        The :math:`E[g^2]_t` is given by:
 *
 *        .. math::
 *        E[g^2]_t = \gamma * E[g^2]_{t-1} + (1-\gamma) * g_t^2
 *
 *        The update step is
 *
 *        .. math::
 *        \theta_{t+1} = \theta_t - \frac{\eta}{RMS[g]_t} g_t
 *
 *        The RMSProp code follows the version in
 *        http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
 *        Tieleman & Hinton, 2012.
 *
 *        Hinton suggests the momentum term :math:`\gamma` to be 0.9 and the learning rate
 *        :math:`\eta` to be 0.001.
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L797
 * \param weight Weight
 * \param grad Gradient
 * \param n n
 * \param lr Learning rate
 * \param gamma1 The decay rate of momentum estimates.
 * \param epsilon A small constant for numerical stability.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param clip_weights Clip weights to the range of [-clip_weights, clip_weights] If
 *        clip_weights <= 0, weight clipping is turned off. weights = max(min(weights,
 * \return new symbol
 */
inline Symbol rmsprop_update(Symbol weight,
                             Symbol grad,
                             Symbol n,
                             mx_float lr,
                             mx_float gamma1 = 0.949999988,
                             mx_float epsilon = 9.99999994e-09,
                             mx_float wd = 0,
                             mx_float rescale_grad = 1,
                             mx_float clip_gradient = -1,
                             mx_float clip_weights = -1) {
  return Operator("rmsprop_update")
           .SetParam("lr", lr)
           .SetParam("gamma1", gamma1)
           .SetParam("epsilon", epsilon)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("clip_weights", clip_weights)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("n", n)
           .CreateSymbol();
}

/*!
 * \brief Update function for RMSPropAlex optimizer.
 *
 *        `RMSPropAlex` is non-centered version of `RMSProp`.
 *
 *        Define :math:`E[g^2]_t` is the decaying average over past squared gradient and
 *        :math:`E[g]_t` is the decaying average over past gradient.
 *
 *        .. math::
 *        E[g^2]_t = \gamma_1 * E[g^2]_{t-1} + (1 - \gamma_1) * g_t^2\\
 *        E[g]_t = \gamma_1 * E[g]_{t-1} + (1 - \gamma_1) * g_t\\
 *        \Delta_t = \gamma_2 * \Delta_{t-1} - \frac{\eta}{\sqrt{E[g^2]_t - E[g]_t^2 +
 *
 *        The update step is
 *
 *        .. math::
 *        \theta_{t+1} = \theta_t + \Delta_t
 *
 *        The RMSPropAlex code follows the version in
 *        http://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45) by Alex Graves, 2013.
 *
 *        Graves suggests the momentum term :math:`\gamma_1` to be 0.95, :math:`\gamma_2`
 *        to be 0.9 and the learning rate :math:`\eta` to be 0.0001.
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L836
 * \param weight Weight
 * \param grad Gradient
 * \param n n
 * \param g g
 * \param delta delta
 * \param lr Learning rate
 * \param gamma1 Decay rate.
 * \param gamma2 Decay rate.
 * \param epsilon A small constant for numerical stability.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \param clip_weights Clip weights to the range of [-clip_weights, clip_weights] If
 *        clip_weights <= 0, weight clipping is turned off. weights = max(min(weights,
 * \return new symbol
 */
inline Symbol rmspropalex_update(Symbol weight,
                                 Symbol grad,
                                 Symbol n,
                                 Symbol g,
                                 Symbol delta,
                                 mx_float lr,
                                 mx_float gamma1 = 0.949999988,
                                 mx_float gamma2 = 0.899999976,
                                 mx_float epsilon = 9.99999994e-09,
                                 mx_float wd = 0,
                                 mx_float rescale_grad = 1,
                                 mx_float clip_gradient = -1,
                                 mx_float clip_weights = -1) {
  return Operator("rmspropalex_update")
           .SetParam("lr", lr)
           .SetParam("gamma1", gamma1)
           .SetParam("gamma2", gamma2)
           .SetParam("epsilon", epsilon)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetParam("clip_weights", clip_weights)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("n", n)
           .SetInput("g", g)
           .SetInput("delta", delta)
           .CreateSymbol();
}

/*!
 * \brief Update function for Ftrl optimizer.
 *        Referenced from *Ad Click Prediction: a View from the Trenches*, available at
 *        http://dl.acm.org/citation.cfm?id=2488200.
 *
 *        It updates the weights using::
 *
 *        rescaled_grad = clip(grad * rescale_grad, clip_gradient)
 *        z += rescaled_grad - (sqrt(n + rescaled_grad**2) - sqrt(n)) * weight /
 *        n += rescaled_grad**2
 *        w = (sign(z) * lamda1 - z) / ((beta + sqrt(n)) / learning_rate + wd) * (abs(z)
 *
 *        If w, z and n are all of ``row_sparse`` storage type,
 *        only the row slices whose indices appear in grad.indices are updated (for w, z
 *
 *        for row in grad.indices:
 *        rescaled_grad[row] = clip(grad[row] * rescale_grad, clip_gradient)
 *        z[row] += rescaled_grad[row] - (sqrt(n[row] + rescaled_grad[row]**2) -
 *        n[row] += rescaled_grad[row]**2
 *        w[row] = (sign(z[row]) * lamda1 - z[row]) / ((beta + sqrt(n[row])) /
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L876
 * \param weight Weight
 * \param grad Gradient
 * \param z z
 * \param n Square of grad
 * \param lr Learning rate
 * \param lamda1 The L1 regularization coefficient.
 * \param beta Per-Coordinate Learning Rate beta.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \return new symbol
 */
inline Symbol ftrl_update(Symbol weight,
                          Symbol grad,
                          Symbol z,
                          Symbol n,
                          mx_float lr,
                          mx_float lamda1 = 0.00999999978,
                          mx_float beta = 1,
                          mx_float wd = 0,
                          mx_float rescale_grad = 1,
                          mx_float clip_gradient = -1) {
  return Operator("ftrl_update")
           .SetParam("lr", lr)
           .SetParam("lamda1", lamda1)
           .SetParam("beta", beta)
           .SetParam("wd", wd)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("z", z)
           .SetInput("n", n)
           .CreateSymbol();
}

/*!
 * \brief Phase I of lamb update it performs the following operations and returns g:.
 *
 *        Link to paper: https://arxiv.org/pdf/1904.00962.pdf
 *
 *        .. math::
 *        \begin{gather*}
 *        grad = grad * rescale_grad
 *        if (grad < -clip_gradient)
 *        then
 *        grad = -clip_gradient
 *        if (grad > clip_gradient)
 *        then
 *        grad = clip_gradient
 *
 *        mean = beta1 * mean + (1 - beta1) * grad;
 *        variance = beta2 * variance + (1. - beta2) * grad ^ 2;
 *
 *        if (bias_correction)
 *        then
 *        mean_hat = mean / (1. - beta1^t);
 *        var_hat = var / (1 - beta2^t);
 *        g = mean_hat / (var_hat^(1/2) + epsilon) + wd * weight;
 *        else
 *        g = mean / (var_data^(1/2) + epsilon) + wd * weight;
 *        \end{gather*}
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L953
 * \param weight Weight
 * \param grad Gradient
 * \param mean Moving mean
 * \param var Moving variance
 * \param t Index update count.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param beta1 The decay rate for the 1st moment estimates.
 * \param beta2 The decay rate for the 2nd moment estimates.
 * \param epsilon A small constant for numerical stability.
 * \param bias_correction Whether to use bias correction.
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \return new symbol
 */
inline Symbol lamb_update_phase1(Symbol weight,
                                 Symbol grad,
                                 Symbol mean,
                                 Symbol var,
                                 int t,
                                 mx_float wd,
                                 mx_float beta1 = 0.899999976,
                                 mx_float beta2 = 0.999000013,
                                 mx_float epsilon = 9.99999997e-07,
                                 bool bias_correction = true,
                                 mx_float rescale_grad = 1,
                                 mx_float clip_gradient = -1) {
  return Operator("lamb_update_phase1")
           .SetParam("t", t)
           .SetParam("wd", wd)
           .SetParam("beta1", beta1)
           .SetParam("beta2", beta2)
           .SetParam("epsilon", epsilon)
           .SetParam("bias_correction", bias_correction)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("mean", mean)
           .SetInput("var", var)
           .CreateSymbol();
}

/*!
 * \brief Phase II of lamb update it performs the following operations and updates grad.
 *
 *        Link to paper: https://arxiv.org/pdf/1904.00962.pdf
 *
 *        .. math::
 *        \begin{gather*}
 *        if (lower_bound >= 0)
 *        then
 *        r1 = max(r1, lower_bound)
 *        if (upper_bound >= 0)
 *        then
 *        r1 = max(r1, upper_bound)
 *
 *        if (r1 == 0 or r2 == 0)
 *        then
 *        lr = lr
 *        else
 *        lr = lr * (r1/r2)
 *        weight = weight - lr * g
 *        \end{gather*}
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L992
 * \param weight Weight
 * \param g Output of lamb_update_phase 1
 * \param r1 r1
 * \param r2 r2
 * \param lr Learning rate
 * \param lower_bound Lower limit of norm of weight. If lower_bound <= 0, Lower limit is
 * \param upper_bound Upper limit of norm of weight. If upper_bound <= 0, Upper limit is
 * \return new symbol
 */
inline Symbol lamb_update_phase2(Symbol weight,
                                 Symbol g,
                                 Symbol r1,
                                 Symbol r2,
                                 mx_float lr,
                                 mx_float lower_bound = -1,
                                 mx_float upper_bound = -1) {
  return Operator("lamb_update_phase2")
           .SetParam("lr", lr)
           .SetParam("lower_bound", lower_bound)
           .SetParam("upper_bound", upper_bound)
           .SetInput("weight", weight)
           .SetInput("g", g)
           .SetInput("r1", r1)
           .SetInput("r2", r2)
           .CreateSymbol();
}

/*!
 * \brief Mixed Precision version of Phase I of lamb update
 *        it performs the following operations and returns g:.
 *
 *        Link to paper: https://arxiv.org/pdf/1904.00962.pdf
 *
 *        .. math::
 *        \begin{gather*}
 *        grad32 = grad(float16) * rescale_grad
 *        if (grad < -clip_gradient)
 *        then
 *        grad = -clip_gradient
 *        if (grad > clip_gradient)
 *        then
 *        grad = clip_gradient
 *
 *        mean = beta1 * mean + (1 - beta1) * grad;
 *        variance = beta2 * variance + (1. - beta2) * grad ^ 2;
 *
 *        if (bias_correction)
 *        then
 *        mean_hat = mean / (1. - beta1^t);
 *        var_hat = var / (1 - beta2^t);
 *        g = mean_hat / (var_hat^(1/2) + epsilon) + wd * weight32;
 *        else
 *        g = mean / (var_data^(1/2) + epsilon) + wd * weight32;
 *        \end{gather*}
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L1033
 * \param weight Weight
 * \param grad Gradient
 * \param mean Moving mean
 * \param var Moving variance
 * \param weight32 Weight32
 * \param t Index update count.
 * \param wd Weight decay augments the objective function with a regularization term that
 *        penalizes large weights. The penalty scales with the square of the magnitude of
 * \param beta1 The decay rate for the 1st moment estimates.
 * \param beta2 The decay rate for the 2nd moment estimates.
 * \param epsilon A small constant for numerical stability.
 * \param bias_correction Whether to use bias correction.
 * \param rescale_grad Rescale gradient to grad = rescale_grad*grad.
 * \param clip_gradient Clip gradient to the range of [-clip_gradient, clip_gradient] If
 *        clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad,
 * \return new symbol
 */
inline Symbol mp_lamb_update_phase1(Symbol weight,
                                    Symbol grad,
                                    Symbol mean,
                                    Symbol var,
                                    Symbol weight32,
                                    int t,
                                    mx_float wd,
                                    mx_float beta1 = 0.899999976,
                                    mx_float beta2 = 0.999000013,
                                    mx_float epsilon = 9.99999997e-07,
                                    bool bias_correction = true,
                                    mx_float rescale_grad = 1,
                                    mx_float clip_gradient = -1) {
  return Operator("mp_lamb_update_phase1")
           .SetParam("t", t)
           .SetParam("wd", wd)
           .SetParam("beta1", beta1)
           .SetParam("beta2", beta2)
           .SetParam("epsilon", epsilon)
           .SetParam("bias_correction", bias_correction)
           .SetParam("rescale_grad", rescale_grad)
           .SetParam("clip_gradient", clip_gradient)
           .SetInput("weight", weight)
           .SetInput("grad", grad)
           .SetInput("mean", mean)
           .SetInput("var", var)
           .SetInput("weight32", weight32)
           .CreateSymbol();
}

/*!
 * \brief Mixed Precision version Phase II of lamb update
 *        it performs the following operations and updates grad.
 *
 *        Link to paper: https://arxiv.org/pdf/1904.00962.pdf
 *
 *        .. math::
 *        \begin{gather*}
 *        if (lower_bound >= 0)
 *        then
 *        r1 = max(r1, lower_bound)
 *        if (upper_bound >= 0)
 *        then
 *        r1 = max(r1, upper_bound)
 *
 *        if (r1 == 0 or r2 == 0)
 *        then
 *        lr = lr
 *        else
 *        lr = lr * (r1/r2)
 *        weight32 = weight32 - lr * g
 *        weight(float16) = weight32
 *        \end{gather*}
 *
 *
 *
 *        Defined in ../src/operator/optimizer_op.cc:L1075
 * \param weight Weight
 * \param g Output of mp_lamb_update_phase 1
 * \param r1 r1
 * \param r2 r2
 * \param weight32 Weight32
 * \param lr Learning rate
 * \param lower_bound Lower limit of norm of weight. If lower_bound <= 0, Lower limit is
 * \param upper_bound Upper limit of norm of weight. If upper_bound <= 0, Upper limit is
 * \return new symbol
 */
inline Symbol mp_lamb_update_phase2(Symbol weight,
                                    Symbol g,
                                    Symbol r1,
                                    Symbol r2,
                                    Symbol weight32,
                                    mx_float lr,
                                    mx_float lower_bound = -1,
                                    mx_float upper_bound = -1) {
  return Operator("mp_lamb_update_phase2")
           .SetParam("lr", lr)
           .SetParam("lower_bound", lower_bound)
           .SetParam("upper_bound", upper_bound)
           .SetInput("weight", weight)
           .SetInput("g", g)
           .SetInput("r1", r1)
           .SetInput("r2", r2)
           .SetInput("weight32", weight32)
           .CreateSymbol();
}

/*!
 * \brief Pads an input array with a constant or edge values of the array.
 *
 *        .. note:: `Pad` is deprecated. Use `pad` instead.
 *
 *        .. note:: Current implementation only supports 4D and 5D input arrays with
 *        only on axes 1, 2 and 3. Expects axes 4 and 5 in `pad_width` to be zero.
 *
 *        This operation pads an input array with either a `constant_value` or edge values
 *        along each axis of the input array. The amount of padding is specified by
 *
 *        `pad_width` is a tuple of integer padding widths for each axis of the format
 *        ``(before_1, after_1, ... , before_N, after_N)``. The `pad_width` should be of
 *        where ``N`` is the number of dimensions of the array.
 *
 *        For dimension ``N`` of the input array, ``before_N`` and ``after_N`` indicates
 *        to add before and after the elements of the array along dimension ``N``.
 *        The widths of the higher two dimensions ``before_1``, ``after_1``, ``before_2``,
 *        ``after_2`` must be 0.
 *
 *        Example::
 *
 *        x = [[[[  1.   2.   3.]
 *        [  4.   5.   6.]]
 *
 *        [[  7.   8.   9.]
 *        [ 10.  11.  12.]]]
 *
 *
 *        [[[ 11.  12.  13.]
 *        [ 14.  15.  16.]]
 *
 *        [[ 17.  18.  19.]
 *        [ 20.  21.  22.]]]]
 *
 *        pad(x,mode="edge", pad_width=(0,0,0,0,1,1,1,1)) =
 *
 *        [[[[  1.   1.   2.   3.   3.]
 *        [  1.   1.   2.   3.   3.]
 *        [  4.   4.   5.   6.   6.]
 *        [  4.   4.   5.   6.   6.]]
 *
 *        [[  7.   7.   8.   9.   9.]
 *        [  7.   7.   8.   9.   9.]
 *        [ 10.  10.  11.  12.  12.]
 *        [ 10.  10.  11.  12.  12.]]]
 *
 *
 *        [[[ 11.  11.  12.  13.  13.]
 *        [ 11.  11.  12.  13.  13.]
 *        [ 14.  14.  15.  16.  16.]
 *        [ 14.  14.  15.  16.  16.]]
 *
 *        [[ 17.  17.  18.  19.  19.]
 *        [ 17.  17.  18.  19.  19.]
 *        [ 20.  20.  21.  22.  22.]
 *        [ 20.  20.  21.  22.  22.]]]]
 *
 *        pad(x, mode="constant", constant_value=0, pad_width=(0,0,0,0,1,1,1,1)) =
 *
 *        [[[[  0.   0.   0.   0.   0.]
 *        [  0.   1.   2.   3.   0.]
 *        [  0.   4.   5.   6.   0.]
 *        [  0.   0.   0.   0.   0.]]
 *
 *        [[  0.   0.   0.   0.   0.]
 *        [  0.   7.   8.   9.   0.]
 *        [  0.  10.  11.  12.   0.]
 *        [  0.   0.   0.   0.   0.]]]
 *
 *
 *        [[[  0.   0.   0.   0.   0.]
 *        [  0.  11.  12.  13.   0.]
 *        [  0.  14.  15.  16.   0.]
 *        [  0.   0.   0.   0.   0.]]
 *
 *        [[  0.   0.   0.   0.   0.]
 *        [  0.  17.  18.  19.   0.]
 *        [  0.  20.  21.  22.   0.]
 *        [  0.   0.   0.   0.   0.]]]]
 *
 *
 *
 *
 *        Defined in ../src/operator/pad.cc:L766
 * \param data An n-dimensional input array.
 * \param mode Padding type to use. "constant" pads with `constant_value` "edge" pads
 *        using the edge values of the input array "reflect" pads by reflecting values
 * \param pad_width Widths of the padding regions applied to the edges of each axis. It
 *        is a tuple of integer padding widths for each axis of the format ``(before_1,
 *        after_1, ... , before_N, after_N)``. It should be of length ``2*N`` where ``N``
 *        is the number of dimensions of the array.This is equivalent to pad_width in
 * \param constant_value The value used for padding when `mode` is "constant".
 * \return new symbol
 */
inline Symbol Pad(Symbol data,
                  PadMode mode,
                  Shape pad_width,
                  double constant_value = 0) {
  static const char *PadModeValues[] = {
    "constant",
    "edge",
    "reflect"
  };
  return Operator("Pad")
           .SetParam("mode", PadModeValues[int(mode)])
           .SetParam("pad_width", pad_width)
           .SetParam("constant_value", constant_value)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Adds arguments element-wise.
 *
 *        The storage type of ``elemwise_add`` output depends on storage types of inputs
 *
 *        - elemwise_add(row_sparse, row_sparse) = row_sparse
 *        - elemwise_add(csr, csr) = csr
 *        - elemwise_add(default, csr) = default
 *        - elemwise_add(csr, default) = default
 *        - elemwise_add(default, rsp) = default
 *        - elemwise_add(rsp, default) = default
 *        - otherwise, ``elemwise_add`` generates output with default storage
 *
 *
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol elemwise_add(Symbol lhs,
                           Symbol rhs) {
  return Operator("elemwise_add")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \brief Multiplies arguments element-wise.
 *
 *        The storage type of ``elemwise_mul`` output depends on storage types of inputs
 *
 *        - elemwise_mul(default, default) = default
 *        - elemwise_mul(row_sparse, row_sparse) = row_sparse
 *        - elemwise_mul(default, row_sparse) = row_sparse
 *        - elemwise_mul(row_sparse, default) = row_sparse
 *        - elemwise_mul(csr, csr) = csr
 *        - otherwise, ``elemwise_mul`` generates output with default storage
 *
 *
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol elemwise_mul(Symbol lhs,
                           Symbol rhs) {
  return Operator("elemwise_mul")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \brief Flattens the input array into a 2-D array by collapsing the higher dimensions.
 *        .. note:: `Flatten` is deprecated. Use `flatten` instead.
 *        For an input array with shape ``(d1, d2, ..., dk)``, `flatten` operation
 *        the input array into an output array of shape ``(d1, d2*...*dk)``.
 *        Note that the behavior of this function is different from numpy.ndarray.flatten,
 *        which behaves similar to mxnet.ndarray.reshape((-1,)).
 *        Example::
 *        x = [[
 *        [1,2,3],
 *        [4,5,6],
 *        [7,8,9]
 *        ],
 *        [    [1,2,3],
 *        [4,5,6],
 *        [7,8,9]
 *        ]],
 *        flatten(x) = [[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
 *        [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]]
 *
 *
 *        Defined in ../src/operator/tensor/matrix_op.cc:L250
 * \param data Input array.
 * \return new symbol
 */
inline Symbol Flatten(Symbol data) {
  return Operator("Flatten")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Maps integer indices to vector representations (embeddings).
 *
 *        This operator maps words to real-valued vectors in a high-dimensional space,
 *        called word embeddings. These embeddings can capture semantic and syntactic
 *        For example, it has been noted that in the learned embedding spaces, similar
 *        to be close to each other and dissimilar words far apart.
 *
 *        For an input array of shape (d1, ..., dK),
 *        the shape of an output array is (d1, ..., dK, output_dim).
 *        All the input values should be integers in the range [0, input_dim).
 *
 *        If the input_dim is ip0 and output_dim is op0, then shape of the embedding
 *        (ip0, op0).
 *
 *        When "sparse_grad" is False, if any index mentioned is too large, it is
 *        addresses the last vector in an embedding matrix.
 *        When "sparse_grad" is True, an error will be raised if invalid indices are
 *
 *        Examples::
 *
 *        input_dim = 4
 *        output_dim = 5
 *
 *        // Each row in weight matrix y represents a word. So, y = (w0,w1,w2,w3)
 *        y = [[  0.,   1.,   2.,   3.,   4.],
 *        [  5.,   6.,   7.,   8.,   9.],
 *        [ 10.,  11.,  12.,  13.,  14.],
 *        [ 15.,  16.,  17.,  18.,  19.]]
 *
 *        // Input array x represents n-grams(2-gram). So, x = [(w1,w3), (w0,w2)]
 *        x = [[ 1.,  3.],
 *        [ 0.,  2.]]
 *
 *        // Mapped input x to its vector representation y.
 *        Embedding(x, y, 4, 5) = [[[  5.,   6.,   7.,   8.,   9.],
 *        [ 15.,  16.,  17.,  18.,  19.]],
 *
 *        [[  0.,   1.,   2.,   3.,   4.],
 *        [ 10.,  11.,  12.,  13.,  14.]]]
 *
 *
 *        The storage type of weight can be either row_sparse or default.
 *
 *        .. Note::
 *
 *        If "sparse_grad" is set to True, the storage type of gradient w.r.t weights
 *        "row_sparse". Only a subset of optimizers support sparse gradients, including
 *        and Adam. Note that by default lazy updates is turned on, which may perform
 *        from standard updates. For more details, please check the Optimization API at:
 *        https://mxnet.incubator.apache.org/api/python/optimization/optimization.html
 *
 *
 *
 *        Defined in ../src/operator/tensor/indexing_op.cc:L598
 * \param data The input array to the embedding operator.
 * \param weight The embedding weight matrix.
 * \param input_dim Vocabulary size of the input indices.
 * \param output_dim Dimension of the embedding vectors.
 * \param dtype Data type of weight.
 * \param sparse_grad Compute row sparse gradient in the backward calculation. If set to
 * \return new symbol
 */
inline Symbol Embedding(Symbol data,
                        Symbol weight,
                        int input_dim,
                        int output_dim,
                        EmbeddingDtype dtype = EmbeddingDtype::kFloat32,
                        bool sparse_grad = false) {
  static const char *EmbeddingDtypeValues[] = {
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "int32",
    "int64",
    "int8",
    "uint8"
  };
  return Operator("Embedding")
           .SetParam("input_dim", input_dim)
           .SetParam("output_dim", output_dim)
           .SetParam("dtype", EmbeddingDtypeValues[int(dtype)])
           .SetParam("sparse_grad", sparse_grad)
           .SetInput("data", data)
           .SetInput("weight", weight)
           .CreateSymbol();
}

/*!
 * \brief Computes and optimizes for squared loss during backward propagation.
 *        Just outputs ``data`` during forward propagation.
 *
 *        If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i`
 *        then the squared loss estimated over :math:`n` samples is defined as
 *
 *        :math:`\text{SquaredLoss}(\textbf{Y}, \hat{\textbf{Y}} ) = \frac{1}{n}
 *
 *        .. note::
 *        Use the LinearRegressionOutput as the final output layer of a net.
 *
 *        The storage type of ``label`` can be ``default`` or ``csr``
 *
 *        - LinearRegressionOutput(default, default) = default
 *        - LinearRegressionOutput(default, csr) = default
 *
 *        By default, gradients of this loss function are scaled by factor `1/m`, where m
 *        The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.
 *
 *
 *
 *        Defined in ../src/operator/regression_output.cc:L92
 * \param data Input data to the function.
 * \param label Input label to the function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol LinearRegressionOutput(Symbol data,
                                     Symbol label,
                                     mx_float grad_scale = 1) {
  return Operator("LinearRegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \brief Computes mean absolute error of the input.
 *
 *        MAE is a risk metric corresponding to the expected value of the absolute error.
 *
 *        If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i`
 *        then the mean absolute error (MAE) estimated over :math:`n` samples is defined
 *
 *        :math:`\text{MAE}(\textbf{Y}, \hat{\textbf{Y}} ) = \frac{1}{n} \sum_{i=0}^{n-1}
 *
 *        .. note::
 *        Use the MAERegressionOutput as the final output layer of a net.
 *
 *        The storage type of ``label`` can be ``default`` or ``csr``
 *
 *        - MAERegressionOutput(default, default) = default
 *        - MAERegressionOutput(default, csr) = default
 *
 *        By default, gradients of this loss function are scaled by factor `1/m`, where m
 *        The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.
 *
 *
 *
 *        Defined in ../src/operator/regression_output.cc:L120
 * \param data Input data to the function.
 * \param label Input label to the function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol MAERegressionOutput(Symbol data,
                                  Symbol label,
                                  mx_float grad_scale = 1) {
  return Operator("MAERegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \brief Applies a logistic function to the input.
 *
 *        The logistic function, also known as the sigmoid function, is computed as
 *        :math:`\frac{1}{1+exp(-\textbf{x})}`.
 *
 *        Commonly, the sigmoid is used to squash the real-valued output of a linear model
 *        :math:`wTx+b` into the [0,1] range so that it can be interpreted as a
 *        It is suitable for binary classification or probability prediction tasks.
 *
 *        .. note::
 *        Use the LogisticRegressionOutput as the final output layer of a net.
 *
 *        The storage type of ``label`` can be ``default`` or ``csr``
 *
 *        - LogisticRegressionOutput(default, default) = default
 *        - LogisticRegressionOutput(default, csr) = default
 *
 *        The loss function used is the Binary Cross Entropy Loss:
 *
 *        :math:`-{(y\log(p) + (1 - y)\log(1 - p))}`
 *
 *        Where `y` is the ground truth probability of positive outcome for a given
 *        example, and `p` the probability predicted by the model. By default, gradients
 *        of this loss function are scaled by factor `1/m`, where m is the number of
 *        The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.
 *
 *
 *
 *        Defined in ../src/operator/regression_output.cc:L152
 * \param data Input data to the function.
 * \param label Input label to the function.
 * \param grad_scale Scale the gradient by a float factor
 * \return new symbol
 */
inline Symbol LogisticRegressionOutput(Symbol data,
                                       Symbol label,
                                       mx_float grad_scale = 1) {
  return Operator("LogisticRegressionOutput")
           .SetParam("grad_scale", grad_scale)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \brief Applies recurrent layers to input data. Currently, vanilla RNN, LSTM and GRU are
 *        implemented, with both multi-layer and bidirectional support.
 *
 *        When the input data is of type float32 and the environment variables
 *        and MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION are set to 1, this operator will
 *        pseudo-float16 precision (float32 math with float16 I/O) precision in order to
 *        Tensor Cores on suitable NVIDIA GPUs. This can sometimes give significant
 *
 *        **Vanilla RNN**
 *
 *        Applies a single-gate recurrent layer to input X. Two kinds of activation
 *        ReLU and Tanh.
 *
 *        With ReLU activation function:
 *
 *        .. math::
 *        h_t = relu(W_{ih} * x_t + b_{ih}  +  W_{hh} * h_{(t-1)} + b_{hh})
 *
 *        With Tanh activtion function:
 *
 *        .. math::
 *        h_t = \tanh(W_{ih} * x_t + b_{ih}  +  W_{hh} * h_{(t-1)} + b_{hh})
 *
 *        Reference paper: Finding structure in time - Elman, 1988.
 *        https://crl.ucsd.edu/~elman/Papers/fsit.pdf
 *
 *        **LSTM**
 *
 *        Long Short-Term Memory - Hochreiter, 1997.
 *
 *        .. math::
 *        \begin{array}{ll}
 *        i_t = \mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
 *        f_t = \mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
 *        g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hc} h_{(t-1)} + b_{hg}) \\
 *        o_t = \mathrm{sigmoid}(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
 *        c_t = f_t * c_{(t-1)} + i_t * g_t \\
 *        h_t = o_t * \tanh(c_t)
 *        \end{array}
 *
 *        With the projection size being set, LSTM could use the projection feature to
 *        size and give some speedups without significant damage to the accuracy.
 *
 *        Long Short-Term Memory Based Recurrent Neural Network Architectures for Large
 *        Recognition - Sak et al. 2014. https://arxiv.org/abs/1402.1128
 *
 *        .. math::
 *        \begin{array}{ll}
 *        i_t = \mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{ri} r_{(t-1)} + b_{ri}) \\
 *        f_t = \mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{rf} r_{(t-1)} + b_{rf}) \\
 *        g_t = \tanh(W_{ig} x_t + b_{ig} + W_{rc} r_{(t-1)} + b_{rg}) \\
 *        o_t = \mathrm{sigmoid}(W_{io} x_t + b_{o} + W_{ro} r_{(t-1)} + b_{ro}) \\
 *        c_t = f_t * c_{(t-1)} + i_t * g_t \\
 *        h_t = o_t * \tanh(c_t)
 *        r_t = W_{hr} h_t
 *        \end{array}
 *
 *        **GRU**
 *
 *        Gated Recurrent Unit - Cho et al. 2014. http://arxiv.org/abs/1406.1078
 *
 *        The definition of GRU here is slightly different from paper but compatible with
 *
 *        .. math::
 *        \begin{array}{ll}
 *        r_t = \mathrm{sigmoid}(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
 *        z_t = \mathrm{sigmoid}(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
 *        n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
 *        h_t = (1 - z_t) * n_t + z_t * h_{(t-1)} \\
 *        \end{array}
 *
 *
 *        Defined in ../src/operator/rnn.cc:L369
 * \param data Input data to RNN
 * \param parameters Vector of all RNN trainable parameters concatenated
 * \param state initial hidden state of the RNN
 * \param state_cell initial cell state for LSTM networks (only for LSTM)
 * \param sequence_length Vector of valid sequence lengths for each element in batch.
 * \param state_size size of the state for each layer
 * \param num_layers number of stacked layers
 * \param mode the type of RNN to compute
 * \param bidirectional whether to use bidirectional recurrent layers
 * \param p drop rate of the dropout on the outputs of each RNN layer, except the last
 * \param state_outputs Whether to have the states as symbol outputs.
 * \param projection_size size of project size
 * \param lstm_state_clip_min Minimum clip value of LSTM states. This option must be used
 * \param lstm_state_clip_max Maximum clip value of LSTM states. This option must be used
 * \param lstm_state_clip_nan Whether to stop NaN from propagating in state by clipping
 * \param use_sequence_length If set to true, this layer takes in an extra input
 * \return new symbol
 */
inline Symbol RNN(Symbol data,
                  Symbol parameters,
                  Symbol state,
                  Symbol state_cell,
                  Symbol sequence_length,
                  uint32_t state_size,
                  uint32_t num_layers,
                  RNNMode mode,
                  bool bidirectional = false,
                  mx_float p = 0,
                  bool state_outputs = false,
                  dmlc::optional<int> projection_size = dmlc::optional<int>(),
                  dmlc::optional<double> lstm_state_clip_min = dmlc::optional<double>(),
                  dmlc::optional<double> lstm_state_clip_max = dmlc::optional<double>(),
                  bool lstm_state_clip_nan = false,
                  bool use_sequence_length = false) {
  static const char *RNNModeValues[] = {
    "gru",
    "lstm",
    "rnn_relu",
    "rnn_tanh"
  };
  return Operator("RNN")
           .SetParam("state_size", state_size)
           .SetParam("num_layers", num_layers)
           .SetParam("mode", RNNModeValues[int(mode)])
           .SetParam("bidirectional", bidirectional)
           .SetParam("p", p)
           .SetParam("state_outputs", state_outputs)
           .SetParam("projection_size", projection_size)
           .SetParam("lstm_state_clip_min", lstm_state_clip_min)
           .SetParam("lstm_state_clip_max", lstm_state_clip_max)
           .SetParam("lstm_state_clip_nan", lstm_state_clip_nan)
           .SetParam("use_sequence_length", use_sequence_length)
           .SetInput("data", data)
           .SetInput("parameters", parameters)
           .SetInput("state", state)
           .SetInput("state_cell", state_cell)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol();
}

/*!
 * \brief Performs region of interest(ROI) pooling on the input array.
 *
 *        ROI pooling is a variant of a max pooling layer, in which the output size is
 *        region of interest is a parameter. Its purpose is to perform max pooling on the
 *        of non-uniform sizes to obtain fixed-size feature maps. ROI pooling is a
 *        layer mostly used in training a `Fast R-CNN` network for object detection.
 *
 *        This operator takes a 4D feature map as an input array and region proposals as
 *        then it pools over sub-regions of input and produces a fixed-sized output array
 *        regardless of the ROI size.
 *
 *        To crop the feature map accordingly, you can resize the bounding box coordinates
 *        by changing the parameters `rois` and `spatial_scale`.
 *
 *        The cropped feature maps are pooled by standard max pooling operation to a
 *        indicated by a `pooled_size` parameter. batch_size will change to the number of
 *        bounding boxes after `ROIPooling`.
 *
 *        The size of each region of interest doesn't have to be perfectly divisible by
 *        the number of pooling sections(`pooled_size`).
 *
 *        Example::
 *
 *        x = [[[[  0.,   1.,   2.,   3.,   4.,   5.],
 *        [  6.,   7.,   8.,   9.,  10.,  11.],
 *        [ 12.,  13.,  14.,  15.,  16.,  17.],
 *        [ 18.,  19.,  20.,  21.,  22.,  23.],
 *        [ 24.,  25.,  26.,  27.,  28.,  29.],
 *        [ 30.,  31.,  32.,  33.,  34.,  35.],
 *        [ 36.,  37.,  38.,  39.,  40.,  41.],
 *        [ 42.,  43.,  44.,  45.,  46.,  47.]]]]
 *
 *        // region of interest i.e. bounding box coordinates.
 *        y = [[0,0,0,4,4]]
 *
 *        // returns array of shape (2,2) according to the given roi with max pooling.
 *        ROIPooling(x, y, (2,2), 1.0) = [[[[ 14.,  16.],
 *        [ 26.,  28.]]]]
 *
 *        // region of interest is changed due to the change in `spacial_scale` parameter.
 *        ROIPooling(x, y, (2,2), 0.7) = [[[[  7.,   9.],
 *        [ 19.,  21.]]]]
 *
 *
 *
 *        Defined in ../src/operator/roi_pooling.cc:L225
 * \param data The input array to the pooling operator,  a 4D Feature maps
 * \param rois Bounding box coordinates, a 2D array of [[batch_index, x1, y1, x2, y2]],
 *        where (x1, y1) and (x2, y2) are top left and bottom right corners of designated
 *        region of interest. `batch_index` indicates the index of corresponding image in
 * \param pooled_size ROI pooling output shape (h,w)
 * \param spatial_scale Ratio of input feature map height (or w) to raw image height (or
 * \return new symbol
 */
inline Symbol ROIPooling(Symbol data,
                         Symbol rois,
                         Shape pooled_size,
                         mx_float spatial_scale) {
  return Operator("ROIPooling")
           .SetParam("pooled_size", pooled_size)
           .SetParam("spatial_scale", spatial_scale)
           .SetInput("data", data)
           .SetInput("rois", rois)
           .CreateSymbol();
}

/*!
 * \brief Sets all elements outside the sequence to a constant value.
 *
 *        This function takes an n-dimensional input array of the form
 *        [max_sequence_length, batch_size, other_feature_dims] and returns an array of
 *
 *        Parameter `sequence_length` is used to handle variable-length sequences.
 *        should be an input array of positive ints of dimension [batch_size].
 *        To use this parameter, set `use_sequence_length` to `True`,
 *        otherwise each example in the batch is assumed to have the max sequence length
 *        this operator works as the `identity` operator.
 *
 *        Example::
 *
 *        x = [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]]]
 *
 *        // Batch 1
 *        B1 = [[  1.,   2.,   3.],
 *        [  7.,   8.,   9.],
 *        [ 13.,  14.,  15.]]
 *
 *        // Batch 2
 *        B2 = [[  4.,   5.,   6.],
 *        [ 10.,  11.,  12.],
 *        [ 16.,  17.,  18.]]
 *
 *        // works as identity operator when sequence_length parameter is not used
 *        SequenceMask(x) = [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]]]
 *
 *        // sequence_length [1,1] means 1 of each batch will be kept
 *        // and other rows are masked with default mask value = 0
 *        SequenceMask(x, sequence_length=[1,1], use_sequence_length=True) =
 *        [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  0.,   0.,   0.],
 *        [  0.,   0.,   0.]],
 *
 *        [[  0.,   0.,   0.],
 *        [  0.,   0.,   0.]]]
 *
 *        // sequence_length [2,3] means 2 of batch B1 and 3 of batch B2 will be kept
 *        // and other rows are masked with value = 1
 *        SequenceMask(x, sequence_length=[2,3], use_sequence_length=True, value=1) =
 *        [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  7.,   8.,   9.],
 *        [  10.,  11.,  12.]],
 *
 *        [[   1.,   1.,   1.],
 *        [  16.,  17.,  18.]]]
 *
 *
 *
 *        Defined in ../src/operator/sequence_mask.cc:L186
 * \param data n-dimensional input array of the form [max_sequence_length, batch_size,
 * \param sequence_length vector of sequence lengths of the form [batch_size]
 * \param use_sequence_length If set to true, this layer takes in an extra input
 * \param value The value to be used as a mask.
 * \param axis The sequence axis. Only values of 0 and 1 are currently supported.
 * \return new symbol
 */
inline Symbol SequenceMask(Symbol data,
                           Symbol sequence_length,
                           bool use_sequence_length = false,
                           mx_float value = 0,
                           int axis = 0) {
  return Operator("SequenceMask")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetParam("value", value)
           .SetParam("axis", axis)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol();
}

/*!
 * \brief Splits an array along a particular axis into multiple sub-arrays.
 *
 *        .. note:: ``SliceChannel`` is deprecated. Use ``split`` instead.
 *
 *        **Note** that `num_outputs` should evenly divide the length of the axis
 *        along which to split the array.
 *
 *        Example::
 *
 *        x  = [[[ 1.]
 *        [ 2.]]
 *        [[ 3.]
 *        [ 4.]]
 *        [[ 5.]
 *        [ 6.]]]
 *        x.shape = (3, 2, 1)
 *
 *        y = split(x, axis=1, num_outputs=2) // a list of 2 arrays with shape (3, 1, 1)
 *        y = [[[ 1.]]
 *        [[ 3.]]
 *        [[ 5.]]]
 *
 *        [[[ 2.]]
 *        [[ 4.]]
 *        [[ 6.]]]
 *
 *        y[0].shape = (3, 1, 1)
 *
 *        z = split(x, axis=0, num_outputs=3) // a list of 3 arrays with shape (1, 2, 1)
 *        z = [[[ 1.]
 *        [ 2.]]]
 *
 *        [[[ 3.]
 *        [ 4.]]]
 *
 *        [[[ 5.]
 *        [ 6.]]]
 *
 *        z[0].shape = (1, 2, 1)
 *
 *        `squeeze_axis=1` removes the axis with length 1 from the shapes of the output
 *        **Note** that setting `squeeze_axis` to ``1`` removes axis with length 1 only
 *        along the `axis` which it is split.
 *        Also `squeeze_axis` can be set to true only if ``input.shape[axis] ==
 *
 *        Example::
 *
 *        z = split(x, axis=0, num_outputs=3, squeeze_axis=1) // a list of 3 arrays with
 *        z = [[ 1.]
 *        [ 2.]]
 *
 *        [[ 3.]
 *        [ 4.]]
 *
 *        [[ 5.]
 *        [ 6.]]
 *        z[0].shape = (2 ,1 )
 *
 *
 *
 *        Defined in ../src/operator/slice_channel.cc:L107
 * \param data The input
 * \param num_outputs Number of splits. Note that this should evenly divide the length of
 * \param axis Axis along which to split.
 * \param squeeze_axis If true, Removes the axis with length 1 from the shapes of the
 *        output arrays. **Note** that setting `squeeze_axis` to ``true`` removes axis
 *        with length 1 only along the `axis` which it is split. Also `squeeze_axis` can
 * \return new symbol
 */
inline Symbol SliceChannel(Symbol data,
                           int num_outputs,
                           int axis = 1,
                           bool squeeze_axis = false) {
  return Operator("SliceChannel")
           .SetParam("num_outputs", num_outputs)
           .SetParam("axis", axis)
           .SetParam("squeeze_axis", squeeze_axis)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Computes the gradient of cross entropy loss with respect to softmax output.
 *
 *        - This operator computes the gradient in two steps.
 *        The cross entropy loss does not actually need to be computed.
 *
 *        - Applies softmax function on the input array.
 *        - Computes and returns the gradient of cross entropy loss w.r.t. the softmax
 *
 *        - The softmax function, cross entropy loss and gradient is given by:
 *
 *        - Softmax Function:
 *
 *        .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}
 *
 *        - Cross Entropy Function:
 *
 *        .. math:: \text{CE(label, output)} = - \sum_i \text{label}_i
 *
 *        - The gradient of cross entropy loss w.r.t softmax output:
 *
 *        .. math:: \text{gradient} = \text{output} - \text{label}
 *
 *        - During forward propagation, the softmax function is computed for each
 *
 *        For general *N*-D input arrays with shape :math:`(d_1, d_2, ..., d_n)`. The
 *        :math:`s=d_1 \cdot d_2 \cdot \cdot \cdot d_n`. We can use the parameters
 *        and `multi_output` to specify the way to compute softmax:
 *
 *        - By default, `preserve_shape` is ``false``. This operator will reshape the
 *        into a 2-D array with shape :math:`(d_1, \frac{s}{d_1})` and then compute the
 *        each row in the reshaped array, and afterwards reshape it back to the original
 *        :math:`(d_1, d_2, ..., d_n)`.
 *        - If `preserve_shape` is ``true``, the softmax function will be computed along
 *        the last axis (`axis` = ``-1``).
 *        - If `multi_output` is ``true``, the softmax function will be computed along
 *        the second axis (`axis` = ``1``).
 *
 *        - During backward propagation, the gradient of cross-entropy loss w.r.t softmax
 *        The provided label can be a one-hot label array or a probability label array.
 *
 *        - If the parameter `use_ignore` is ``true``, `ignore_label` can specify input
 *        with a particular label to be ignored during backward propagation. **This has
 *        softmax `output` has same shape as `label`**.
 *
 *        Example::
 *
 *        data = [[1,2,3,4],[2,2,2,2],[3,3,3,3],[4,4,4,4]]
 *        label = [1,0,2,3]
 *        ignore_label = 1
 *        SoftmaxOutput(data=data, label = label,\
 *        multi_output=true, use_ignore=true,\
 *        ignore_label=ignore_label)
 *        ## forward softmax output
 *        [[ 0.0320586   0.08714432  0.23688284  0.64391428]
 *        [ 0.25        0.25        0.25        0.25      ]
 *        [ 0.25        0.25        0.25        0.25      ]
 *        [ 0.25        0.25        0.25        0.25      ]]
 *        ## backward gradient output
 *        [[ 0.    0.    0.    0.  ]
 *        [-0.75  0.25  0.25  0.25]
 *        [ 0.25  0.25 -0.75  0.25]
 *        [ 0.25  0.25  0.25 -0.75]]
 *        ## notice that the first row is all 0 because label[0] is 1, which is equal to
 *
 *        - The parameter `grad_scale` can be used to rescale the gradient, which is
 *        give each loss function different weights.
 *
 *        - This operator also supports various ways to normalize the gradient by
 *        The `normalization` is applied if softmax output has different shape than the
 *        The `normalization` mode can be set to the followings:
 *
 *        - ``'null'``: do nothing.
 *        - ``'batch'``: divide the gradient by the batch size.
 *        - ``'valid'``: divide the gradient by the number of instances which are not
 *
 *
 *
 *        Defined in ../src/operator/softmax_output.cc:L243
 * \param data Input array.
 * \param label Ground truth label.
 * \param grad_scale Scales the gradient by a float factor.
 * \param ignore_label The instances whose `labels` == `ignore_label` will be ignored
 * \param multi_output If set to ``true``, the softmax function will be computed along
 *        axis ``1``. This is applied when the shape of input array differs from the
 * \param use_ignore If set to ``true``, the `ignore_label` value will not contribute to
 * \param preserve_shape If set to ``true``, the softmax function will be computed along
 * \param normalization Normalizes the gradient.
 * \param out_grad Multiplies gradient with output gradient element-wise.
 * \param smooth_alpha Constant for computing a label smoothed version of
 *        cross-entropyfor the backwards pass.  This constant gets subtracted from
 *        theone-hot encoding of the gold label and distributed uniformly toall other
 * \return new symbol
 */
inline Symbol SoftmaxOutput(Symbol data,
                            Symbol label,
                            mx_float grad_scale = 1,
                            mx_float ignore_label = -1,
                            bool multi_output = false,
                            bool use_ignore = false,
                            bool preserve_shape = false,
                            SoftmaxOutputNormalization normalization = SoftmaxOutputNormalization::kNull,
                            bool out_grad = false,
                            mx_float smooth_alpha = 0) {
  static const char *SoftmaxOutputNormalizationValues[] = {
    "batch",
    "null",
    "valid"
  };
  return Operator("SoftmaxOutput")
           .SetParam("grad_scale", grad_scale)
           .SetParam("ignore_label", ignore_label)
           .SetParam("multi_output", multi_output)
           .SetParam("use_ignore", use_ignore)
           .SetParam("preserve_shape", preserve_shape)
           .SetParam("normalization", SoftmaxOutputNormalizationValues[int(normalization)])
           .SetParam("out_grad", out_grad)
           .SetParam("smooth_alpha", smooth_alpha)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \brief Interchanges two axes of an array.
 *
 *        Examples::
 *
 *        x = [[1, 2, 3]])
 *        swapaxes(x, 0, 1) = [[ 1],
 *        [ 2],
 *        [ 3]]
 *
 *        x = [[[ 0, 1],
 *        [ 2, 3]],
 *        [[ 4, 5],
 *        [ 6, 7]]]  // (2,2,2) array
 *
 *        swapaxes(x, 0, 2) = [[[ 0, 4],
 *        [ 2, 6]],
 *        [[ 1, 5],
 *        [ 3, 7]]]
 *
 *
 *        Defined in ../src/operator/swapaxis.cc:L70
 * \param data Input array.
 * \param dim1 the first axis to be swapped.
 * \param dim2 the second axis to be swapped.
 * \return new symbol
 */
inline Symbol SwapAxis(Symbol data,
                       int dim1 = 0,
                       int dim2 = 0) {
  return Operator("SwapAxis")
           .SetParam("dim1", dim1)
           .SetParam("dim2", dim2)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Cast function between low precision float/FP32 used by AMP.
 *
 *        It casts only between low precision float/FP32 and does not do anything for
 *
 *
 *        Defined in ../src/operator/tensor/amp_cast.cc:L121
 * \param data The input.
 * \param dtype Output data type.
 * \return new symbol
 */
inline Symbol amp_cast(Symbol data,
                       Amp_castDtype dtype) {
  static const char *Amp_castDtypeValues[] = {
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "int32",
    "int64",
    "int8",
    "uint8"
  };
  return Operator("amp_cast")
           .SetParam("dtype", Amp_castDtypeValues[int(dtype)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Cast function used by AMP, that casts its inputs to the common widest type.
 *
 *        It casts only between low precision float/FP32 and does not do anything for
 *
 *
 *
 *        Defined in ../src/operator/tensor/amp_cast.cc:L165
 * \param data Weights
 * \param num_outputs Number of input/output pairs to be casted to the widest type.
 * \param cast_narrow Whether to cast to the narrowest type
 * \return new symbol
 */
inline Symbol amp_multicast(const std::vector<Symbol>& data,
                            int num_outputs,
                            bool cast_narrow = false) {
  return Operator("amp_multicast")
           .SetParam("num_outputs", num_outputs)
           .SetParam("cast_narrow", cast_narrow)
(data)
           .CreateSymbol();
}

/*!
 * \brief Computes the max of array elements over given axes.
 *
 *        Defined in ../src/operator/tensor/./broadcast_reduce_op.h:L32
 * \param data The input
 * \param axis The axis or axes along which to perform the reduction.
 *
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *
 *        If `axis` is int, a reduction is performed on a particular axis.
 *
 *        If `axis` is a tuple of ints, a reduction is performed on all the axes
 *        specified in the tuple.
 *
 *        If `exclude` is true, reduction will be performed on the axes that are
 *        NOT in axis instead.
 *
 *        Negative values means indexing from right to left.
 * \param keepdims If this is set to `True`, the reduced axes are left in the result as
 * \param exclude Whether to perform reduction on axis that are NOT in axis instead.
 * \return new symbol
 */
inline Symbol max(Symbol data,
                  dmlc::optional<Shape> axis = dmlc::optional<Shape>(),
                  bool keepdims = false,
                  bool exclude = false) {
  return Operator("max")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("exclude", exclude)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Computes the min of array elements over given axes.
 *
 *        Defined in ../src/operator/tensor/./broadcast_reduce_op.h:L47
 * \param data The input
 * \param axis The axis or axes along which to perform the reduction.
 *
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *
 *        If `axis` is int, a reduction is performed on a particular axis.
 *
 *        If `axis` is a tuple of ints, a reduction is performed on all the axes
 *        specified in the tuple.
 *
 *        If `exclude` is true, reduction will be performed on the axes that are
 *        NOT in axis instead.
 *
 *        Negative values means indexing from right to left.
 * \param keepdims If this is set to `True`, the reduced axes are left in the result as
 * \param exclude Whether to perform reduction on axis that are NOT in axis instead.
 * \return new symbol
 */
inline Symbol min(Symbol data,
                  dmlc::optional<Shape> axis = dmlc::optional<Shape>(),
                  bool keepdims = false,
                  bool exclude = false) {
  return Operator("min")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("exclude", exclude)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Computes the norm on an NDArray.
 *
 *        This operator computes the norm on an NDArray with the specified axis, depending
 *        on the value of the ord parameter. By default, it computes the L2 norm on the
 *        array. Currently only ord=2 supports sparse ndarrays.
 *
 *        Examples::
 *
 *        x = [[[1, 2],
 *        [3, 4]],
 *        [[2, 2],
 *        [5, 6]]]
 *
 *        norm(x, ord=2, axis=1) = [[3.1622777 4.472136 ]
 *        [5.3851647 6.3245554]]
 *
 *        norm(x, ord=1, axis=1) = [[4., 6.],
 *        [7., 8.]]
 *
 *        rsp = x.cast_storage('row_sparse')
 *
 *        norm(rsp) = [5.47722578]
 *
 *        csr = x.cast_storage('csr')
 *
 *        norm(csr) = [5.47722578]
 *
 *
 *
 *        Defined in ../src/operator/tensor/broadcast_reduce_norm_value.cc:L89
 * \param data The input
 * \param ord Order of the norm. Currently ord=1 and ord=2 is supported.
 * \param axis The axis or axes along which to perform the reduction.
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *        If `axis` is int, a reduction is performed on a particular axis.
 *        If `axis` is a 2-tuple, it specifies the axes that hold 2-D matrices,
 *        and the matrix norms of these matrices are computed.
 * \param out_dtype The data type of the output.
 * \param keepdims If this is set to `True`, the reduced axis is left in the result as
 * \return new symbol
 */
inline Symbol norm(Symbol data,
                   int ord = 2,
                   dmlc::optional<Shape> axis = dmlc::optional<Shape>(),
                   NormOutDtype out_dtype = NormOutDtype::kNone,
                   bool keepdims = false) {
  static const char *NormOutDtypeValues[] = {
    "None",
    "float16",
    "float32",
    "float64",
    "int32",
    "int64",
    "int8"
  };
  return Operator("norm")
           .SetParam("ord", ord)
           .SetParam("axis", axis)
           .SetParam("out_dtype", NormOutDtypeValues[int(out_dtype)])
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns indices of the maximum values along an axis.
 *
 *        In the case of multiple occurrences of maximum values, the indices
 *        are returned.
 *
 *        Examples::
 *
 *        x = [[ 0.,  1.,  2.],
 *        [ 3.,  4.,  5.]]
 *
 *        // argmax along axis 0
 *        argmax(x, axis=0) = [ 1.,  1.,  1.]
 *
 *        // argmax along axis 1
 *        argmax(x, axis=1) = [ 2.,  2.]
 *
 *        // argmax along axis 1 keeping same dims as an input array
 *        argmax(x, axis=1, keepdims=True) = [[ 2.],
 *        [ 2.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/broadcast_reduce_op_index.cc:L52
 * \param data The input
 * \param axis The axis along which to perform the reduction. Negative values means
 *        indexing from right to left. ``Requires axis to be set as int, because global
 * \param keepdims If this is set to `True`, the reduced axis is left in the result as
 * \return new symbol
 */
inline Symbol argmax(Symbol data,
                     dmlc::optional<int> axis = dmlc::optional<int>(),
                     bool keepdims = false) {
  return Operator("argmax")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns indices of the minimum values along an axis.
 *
 *        In the case of multiple occurrences of minimum values, the indices
 *        are returned.
 *
 *        Examples::
 *
 *        x = [[ 0.,  1.,  2.],
 *        [ 3.,  4.,  5.]]
 *
 *        // argmin along axis 0
 *        argmin(x, axis=0) = [ 0.,  0.,  0.]
 *
 *        // argmin along axis 1
 *        argmin(x, axis=1) = [ 0.,  0.]
 *
 *        // argmin along axis 1 keeping same dims as an input array
 *        argmin(x, axis=1, keepdims=True) = [[ 0.],
 *        [ 0.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/broadcast_reduce_op_index.cc:L77
 * \param data The input
 * \param axis The axis along which to perform the reduction. Negative values means
 *        indexing from right to left. ``Requires axis to be set as int, because global
 * \param keepdims If this is set to `True`, the reduced axis is left in the result as
 * \return new symbol
 */
inline Symbol argmin(Symbol data,
                     dmlc::optional<int> axis = dmlc::optional<int>(),
                     bool keepdims = false) {
  return Operator("argmin")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns argmax indices of each channel from the input array.
 *
 *        The result will be an NDArray of shape (num_channel,).
 *
 *        In case of multiple occurrences of the maximum values, the indices
 *        are returned.
 *
 *        Examples::
 *
 *        x = [[ 0.,  1.,  2.],
 *        [ 3.,  4.,  5.]]
 *
 *        argmax_channel(x) = [ 2.,  2.]
 *
 *
 *
 *        Defined in ../src/operator/tensor/broadcast_reduce_op_index.cc:L97
 * \param data The input array
 * \return new symbol
 */
inline Symbol argmax_channel(Symbol data) {
  return Operator("argmax_channel")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Picks elements from an input array according to the input indices along the
 *
 *        Given an input array of shape ``(d0, d1)`` and indices of shape ``(i0,)``, the
 *        an output array of shape ``(i0,)`` with::
 *
 *        output[i] = input[i, indices[i]]
 *
 *        By default, if any index mentioned is too large, it is replaced by the index
 *        the last element along an axis (the `clip` mode).
 *
 *        This function supports n-dimensional input and (n-1)-dimensional indices arrays.
 *
 *        Examples::
 *
 *        x = [[ 1.,  2.],
 *        [ 3.,  4.],
 *        [ 5.,  6.]]
 *
 *        // picks elements with specified indices along axis 0
 *        pick(x, y=[0,1], 0) = [ 1.,  4.]
 *
 *        // picks elements with specified indices along axis 1
 *        pick(x, y=[0,1,0], 1) = [ 1.,  4.,  5.]
 *
 *        // picks elements with specified indices along axis 1 using 'wrap' mode
 *        // to place indicies that would normally be out of bounds
 *        pick(x, y=[2,-1,-2], 1, mode='wrap') = [ 1.,  4.,  5.]
 *
 *        y = [[ 1.],
 *        [ 0.],
 *        [ 2.]]
 *
 *        // picks elements with specified indices along axis 1 and dims are maintained
 *        pick(x, y, 1, keepdims=True) = [[ 2.],
 *        [ 3.],
 *        [ 6.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/broadcast_reduce_op_index.cc:L151
 * \param data The input array
 * \param index The index array
 * \param axis int or None. The axis to picking the elements. Negative values means
 *        indexing from right to left. If is `None`, the elements in the index w.r.t the
 * \param keepdims If true, the axis where we pick the elements is left in the result as
 * \param mode Specify how out-of-bound indices behave. Default is "clip". "clip" means
 *        clip to the range. So, if all indices mentioned are too large, they are
 *        replaced by the index that addresses the last element along an axis.  "wrap"
 * \return new symbol
 */
inline Symbol pick(Symbol data,
                   Symbol index,
                   dmlc::optional<int> axis = dmlc::optional<int>(-1),
                   bool keepdims = false,
                   PickMode mode = PickMode::kClip) {
  static const char *PickModeValues[] = {
    "clip",
    "wrap"
  };
  return Operator("pick")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("mode", PickModeValues[int(mode)])
           .SetInput("data", data)
           .SetInput("index", index)
           .CreateSymbol();
}

/*!
 * \brief Broadcasts the input array over particular axes.
 *
 *        Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
 *        `(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.
 *
 *        `broadcast_axes` is an alias to the function `broadcast_axis`.
 *
 *        Example::
 *
 *        // given x of shape (1,2,1)
 *        x = [[[ 1.],
 *        [ 2.]]]
 *
 *        // broadcast x on on axis 2
 *        broadcast_axis(x, axis=2, size=3) = [[[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]]]
 *        // broadcast x on on axes 0 and 2
 *        broadcast_axis(x, axis=(0,2), size=(2,3)) = [[[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]],
 *        [[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]]]
 *
 *
 *        Defined in ../src/operator/tensor/broadcast_reduce_op_value.cc:L93
 * \param data The input
 * \param axis The axes to perform the broadcasting.
 * \param size Target sizes of the broadcasting axes.
 * \return new symbol
 */
inline Symbol broadcast_axis(Symbol data,
                             Shape axis = Shape(),
                             Shape size = Shape()) {
  return Operator("broadcast_axis")
           .SetParam("axis", axis)
           .SetParam("size", size)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Broadcasts the input array to a new shape.
 *
 *        Broadcasting is a mechanism that allows NDArrays to perform arithmetic
 *        with arrays of different shapes efficiently without creating multiple copies of
 *        Also see, `Broadcasting
 *        <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ for more
 *
 *        Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
 *        `(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.
 *
 *        For example::
 *
 *        broadcast_to([[1,2,3]], shape=(2,3)) = [[ 1.,  2.,  3.],
 *        [ 1.,  2.,  3.]])
 *
 *        The dimension which you do not want to change can also be kept as `0` which
 *        So with `shape=(2,0)`, we will obtain the same result as in the above example.
 *
 *
 *
 *        Defined in ../src/operator/tensor/broadcast_reduce_op_value.cc:L117
 * \param data The input
 * \param shape The shape of the desired array. We can set the dim to zero if it's same
 *        as the original. E.g `A = broadcast_to(B, shape=(10, 0, 0))` has the same
 * \return new symbol
 */
inline Symbol broadcast_to(Symbol data,
                           Shape shape = Shape()) {
  return Operator("broadcast_to")
           .SetParam("shape", shape)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Broadcasts lhs to have the same shape as rhs.
 *
 *        Broadcasting is a mechanism that allows NDArrays to perform arithmetic
 *        with arrays of different shapes efficiently without creating multiple copies of
 *        Also see, `Broadcasting
 *        <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ for more
 *
 *        Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
 *        `(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.
 *
 *        For example::
 *
 *        broadcast_like([[1,2,3]], [[5,6,7],[7,8,9]]) = [[ 1.,  2.,  3.],
 *        [ 1.,  2.,  3.]])
 *
 *        broadcast_like([9], [1,2,3,4,5], lhs_axes=(0,), rhs_axes=(-1,)) = [9,9,9,9,9]
 *
 *
 *
 *        Defined in ../src/operator/tensor/broadcast_reduce_op_value.cc:L179
 * \param lhs First input.
 * \param rhs Second input.
 * \param lhs_axes Axes to perform broadcast on in the first input array
 * \param rhs_axes Axes to copy from the second input array
 * \return new symbol
 */
inline Symbol broadcast_like(Symbol lhs,
                             Symbol rhs,
                             dmlc::optional<Shape> lhs_axes = dmlc::optional<Shape>(),
                             dmlc::optional<Shape> rhs_axes = dmlc::optional<Shape>()) {
  return Operator("broadcast_like")
           .SetParam("lhs_axes", lhs_axes)
           .SetParam("rhs_axes", rhs_axes)
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \brief Computes the product of array elements over given axes.
 *
 *        Defined in ../src/operator/tensor/./broadcast_reduce_op.h:L31
 * \param data The input
 * \param axis The axis or axes along which to perform the reduction.
 *
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *
 *        If `axis` is int, a reduction is performed on a particular axis.
 *
 *        If `axis` is a tuple of ints, a reduction is performed on all the axes
 *        specified in the tuple.
 *
 *        If `exclude` is true, reduction will be performed on the axes that are
 *        NOT in axis instead.
 *
 *        Negative values means indexing from right to left.
 * \param keepdims If this is set to `True`, the reduced axes are left in the result as
 * \param exclude Whether to perform reduction on axis that are NOT in axis instead.
 * \return new symbol
 */
inline Symbol prod(Symbol data,
                   dmlc::optional<Shape> axis = dmlc::optional<Shape>(),
                   bool keepdims = false,
                   bool exclude = false) {
  return Operator("prod")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("exclude", exclude)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Computes the product of array elements over given axes treating Not a Numbers
 *
 *
 *
 *        Defined in ../src/operator/tensor/broadcast_reduce_prod_value.cc:L47
 * \param data The input
 * \param axis The axis or axes along which to perform the reduction.
 *
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *
 *        If `axis` is int, a reduction is performed on a particular axis.
 *
 *        If `axis` is a tuple of ints, a reduction is performed on all the axes
 *        specified in the tuple.
 *
 *        If `exclude` is true, reduction will be performed on the axes that are
 *        NOT in axis instead.
 *
 *        Negative values means indexing from right to left.
 * \param keepdims If this is set to `True`, the reduced axes are left in the result as
 * \param exclude Whether to perform reduction on axis that are NOT in axis instead.
 * \return new symbol
 */
inline Symbol nanprod(Symbol data,
                      dmlc::optional<Shape> axis = dmlc::optional<Shape>(),
                      bool keepdims = false,
                      bool exclude = false) {
  return Operator("nanprod")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("exclude", exclude)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Computes the sum of array elements over given axes.
 *
 *        .. Note::
 *
 *        `sum` and `sum_axis` are equivalent.
 *        For ndarray of csr storage type summation along axis 0 and axis 1 is supported.
 *        Setting keepdims or exclude to True will cause a fallback to dense operator.
 *
 *        Example::
 *
 *        data = [[[1, 2], [2, 3], [1, 3]],
 *        [[1, 4], [4, 3], [5, 2]],
 *        [[7, 1], [7, 2], [7, 3]]]
 *
 *        sum(data, axis=1)
 *        [[  4.   8.]
 *        [ 10.   9.]
 *        [ 21.   6.]]
 *
 *        sum(data, axis=[1,2])
 *        [ 12.  19.  27.]
 *
 *        data = [[1, 2, 0],
 *        [3, 0, 1],
 *        [4, 1, 0]]
 *
 *        csr = cast_storage(data, 'csr')
 *
 *        sum(csr, axis=0)
 *        [ 8.  3.  1.]
 *
 *        sum(csr, axis=1)
 *        [ 3.  4.  5.]
 *
 *
 *
 *        Defined in ../src/operator/tensor/broadcast_reduce_sum_value.cc:L67
 * \param data The input
 * \param axis The axis or axes along which to perform the reduction.
 *
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *
 *        If `axis` is int, a reduction is performed on a particular axis.
 *
 *        If `axis` is a tuple of ints, a reduction is performed on all the axes
 *        specified in the tuple.
 *
 *        If `exclude` is true, reduction will be performed on the axes that are
 *        NOT in axis instead.
 *
 *        Negative values means indexing from right to left.
 * \param keepdims If this is set to `True`, the reduced axes are left in the result as
 * \param exclude Whether to perform reduction on axis that are NOT in axis instead.
 * \return new symbol
 */
inline Symbol sum(Symbol data,
                  dmlc::optional<Shape> axis = dmlc::optional<Shape>(),
                  bool keepdims = false,
                  bool exclude = false) {
  return Operator("sum")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("exclude", exclude)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Computes the mean of array elements over given axes.
 *
 *        Defined in ../src/operator/tensor/./broadcast_reduce_op.h:L84
 * \param data The input
 * \param axis The axis or axes along which to perform the reduction.
 *
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *
 *        If `axis` is int, a reduction is performed on a particular axis.
 *
 *        If `axis` is a tuple of ints, a reduction is performed on all the axes
 *        specified in the tuple.
 *
 *        If `exclude` is true, reduction will be performed on the axes that are
 *        NOT in axis instead.
 *
 *        Negative values means indexing from right to left.
 * \param keepdims If this is set to `True`, the reduced axes are left in the result as
 * \param exclude Whether to perform reduction on axis that are NOT in axis instead.
 * \return new symbol
 */
inline Symbol mean(Symbol data,
                   dmlc::optional<Shape> axis = dmlc::optional<Shape>(),
                   bool keepdims = false,
                   bool exclude = false) {
  return Operator("mean")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("exclude", exclude)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Computes the sum of array elements over given axes treating Not a Numbers
 *
 *
 *
 *        Defined in ../src/operator/tensor/broadcast_reduce_sum_value.cc:L102
 * \param data The input
 * \param axis The axis or axes along which to perform the reduction.
 *
 *        The default, `axis=()`, will compute over all elements into a
 *        scalar array with shape `(1,)`.
 *
 *        If `axis` is int, a reduction is performed on a particular axis.
 *
 *        If `axis` is a tuple of ints, a reduction is performed on all the axes
 *        specified in the tuple.
 *
 *        If `exclude` is true, reduction will be performed on the axes that are
 *        NOT in axis instead.
 *
 *        Negative values means indexing from right to left.
 * \param keepdims If this is set to `True`, the reduced axes are left in the result as
 * \param exclude Whether to perform reduction on axis that are NOT in axis instead.
 * \return new symbol
 */
inline Symbol nansum(Symbol data,
                     dmlc::optional<Shape> axis = dmlc::optional<Shape>(),
                     bool keepdims = false,
                     bool exclude = false) {
  return Operator("nansum")
           .SetParam("axis", axis)
           .SetParam("keepdims", keepdims)
           .SetParam("exclude", exclude)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Casts tensor storage type to the new type.
 *
 *        When an NDArray with default storage type is cast to csr or row_sparse storage,
 *        the result is compact, which means:
 *
 *        - for csr, zero values will not be retained
 *        - for row_sparse, row slices of all zeros will not be retained
 *
 *        The storage type of ``cast_storage`` output depends on stype parameter:
 *
 *        - cast_storage(csr, 'default') = default
 *        - cast_storage(row_sparse, 'default') = default
 *        - cast_storage(default, 'csr') = csr
 *        - cast_storage(default, 'row_sparse') = row_sparse
 *        - cast_storage(csr, 'csr') = csr
 *        - cast_storage(row_sparse, 'row_sparse') = row_sparse
 *
 *        Example::
 *
 *        dense = [[ 0.,  1.,  0.],
 *        [ 2.,  0.,  3.],
 *        [ 0.,  0.,  0.],
 *        [ 0.,  0.,  0.]]
 *
 *        # cast to row_sparse storage type
 *        rsp = cast_storage(dense, 'row_sparse')
 *        rsp.indices = [0, 1]
 *        rsp.values = [[ 0.,  1.,  0.],
 *        [ 2.,  0.,  3.]]
 *
 *        # cast to csr storage type
 *        csr = cast_storage(dense, 'csr')
 *        csr.indices = [1, 0, 2]
 *        csr.values = [ 1.,  2.,  3.]
 *        csr.indptr = [0, 1, 3, 3, 3]
 *
 *
 *
 *        Defined in ../src/operator/tensor/cast_storage.cc:L71
 * \param data The input.
 * \param stype Output storage type.
 * \return new symbol
 */
inline Symbol cast_storage(Symbol data,
                           Cast_storageStype stype) {
  static const char *Cast_storageStypeValues[] = {
    "csr",
    "default",
    "row_sparse"
  };
  return Operator("cast_storage")
           .SetParam("stype", Cast_storageStypeValues[int(stype)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Return the elements, either from x or y, depending on the condition.
 *
 *        Given three ndarrays, condition, x, and y, return an ndarray with the elements
 *        depending on the elements from condition are true or false. x and y must have
 *        If condition has the same shape as x, each element in the output array is from
 *        corresponding element in the condition is true, and from y if false.
 *
 *        If condition does not have the same shape as x, it must be a 1D array whose
 *        the same as x's first dimension size. Each row of the output array is from x's
 *        if the corresponding element from condition is true, and from y's row if false.
 *
 *        Note that all non-zero values are interpreted as ``True`` in condition.
 *
 *        Examples::
 *
 *        x = [[1, 2], [3, 4]]
 *        y = [[5, 6], [7, 8]]
 *        cond = [[0, 1], [-1, 0]]
 *
 *        where(cond, x, y) = [[5, 2], [3, 8]]
 *
 *        csr_cond = cast_storage(cond, 'csr')
 *
 *        where(csr_cond, x, y) = [[5, 2], [3, 8]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/control_flow_op.cc:L57
 * \param condition condition array
 * \param x
 * \param y
 * \return new symbol
 */
inline Symbol where(Symbol condition,
                    Symbol x,
                    Symbol y) {
  return Operator("where")
           .SetInput("condition", condition)
           .SetInput("x", x)
           .SetInput("y", y)
           .CreateSymbol();
}

/*!
 * \brief Extracts a diagonal or constructs a diagonal array.
 *
 *        ``diag``'s behavior depends on the input array dimensions:
 *
 *        - 1-D arrays: constructs a 2-D array with the input as its diagonal, all other
 *        - N-D arrays: extracts the diagonals of the sub-arrays with axes specified by
 *        The output shape would be decided by removing the axes numbered ``axis1`` and
 *        input shape and appending to the result a new axis with the size of the
 *
 *        For example, when the input shape is `(2, 3, 4, 5)`, ``axis1`` and ``axis2``
 *        respectively and ``k`` is 0, the resulting shape would be `(3, 5, 2)`.
 *
 *        Examples::
 *
 *        x = [[1, 2, 3],
 *        [4, 5, 6]]
 *
 *        diag(x) = [1, 5]
 *
 *        diag(x, k=1) = [2, 6]
 *
 *        diag(x, k=-1) = [4]
 *
 *        x = [1, 2, 3]
 *
 *        diag(x) = [[1, 0, 0],
 *        [0, 2, 0],
 *        [0, 0, 3]]
 *
 *        diag(x, k=1) = [[0, 1, 0],
 *        [0, 0, 2],
 *        [0, 0, 0]]
 *
 *        diag(x, k=-1) = [[0, 0, 0],
 *        [1, 0, 0],
 *        [0, 2, 0]]
 *
 *        x = [[[1, 2],
 *        [3, 4]],
 *
 *        [[5, 6],
 *        [7, 8]]]
 *
 *        diag(x) = [[1, 7],
 *        [2, 8]]
 *
 *        diag(x, k=1) = [[3],
 *        [4]]
 *
 *        diag(x, axis1=-2, axis2=-1) = [[1, 4],
 *        [5, 8]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/diag_op.cc:L87
 * \param data Input ndarray
 * \param k Diagonal in question. The default is 0. Use k>0 for diagonals above the main
 *        diagonal, and k<0 for diagonals below the main diagonal. If input has shape (S0
 * \param axis1 The first axis of the sub-arrays of interest. Ignored when the input is a
 * \param axis2 The second axis of the sub-arrays of interest. Ignored when the input is
 * \return new symbol
 */
inline Symbol diag(Symbol data,
                   int k = 0,
                   int axis1 = 0,
                   int axis2 = 1) {
  return Operator("diag")
           .SetParam("k", k)
           .SetParam("axis1", axis1)
           .SetParam("axis2", axis2)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Dot product of two arrays.
 *
 *        ``dot``'s behavior depends on the input array dimensions:
 *
 *        - 1-D arrays: inner product of vectors
 *        - 2-D arrays: matrix multiplication
 *        - N-D arrays: a sum product over the last axis of the first input and the first
 *        axis of the second input
 *
 *        For example, given 3-D ``x`` with shape `(n,m,k)` and ``y`` with shape
 *        result array will have shape `(n,m,r,s)`. It is computed by::
 *
 *        dot(x,y)[i,j,a,b] = sum(x[i,j,:]*y[:,a,b])
 *
 *        Example::
 *
 *        x = reshape([0,1,2,3,4,5,6,7], shape=(2,2,2))
 *        y = reshape([7,6,5,4,3,2,1,0], shape=(2,2,2))
 *        dot(x,y)[0,0,1,1] = 0
 *        sum(x[0,0,:]*y[:,1,1]) = 0
 *
 *        The storage type of ``dot`` output depends on storage types of inputs,
 *        forward_stype option for output storage type. Implemented sparse operations
 *
 *        - dot(default, default, transpose_a=True/False, transpose_b=True/False) =
 *        - dot(csr, default, transpose_a=True) = default
 *        - dot(csr, default, transpose_a=True) = row_sparse
 *        - dot(csr, default) = default
 *        - dot(csr, row_sparse) = default
 *        - dot(default, csr) = csr (CPU only)
 *        - dot(default, csr, forward_stype='default') = default
 *        - dot(default, csr, transpose_b=True, forward_stype='default') = default
 *
 *        If the combination of input storage types and forward_stype does not match any
 *        above patterns, ``dot`` will fallback and generate output with default storage.
 *
 *        .. Note::
 *
 *        If the storage type of the lhs is "csr", the storage type of gradient w.r.t rhs
 *        "row_sparse". Only a subset of optimizers support sparse gradients, including
 *        and Adam. Note that by default lazy updates is turned on, which may perform
 *        from standard updates. For more details, please check the Optimization API at:
 *        https://mxnet.incubator.apache.org/api/python/optimization/optimization.html
 *
 *
 *
 *        Defined in ../src/operator/tensor/dot.cc:L77
 * \param lhs The first input
 * \param rhs The second input
 * \param transpose_a If true then transpose the first input before dot.
 * \param transpose_b If true then transpose the second input before dot.
 * \param forward_stype The desired storage type of the forward output given by user, if
 *        thecombination of input storage types and this hint does not matchany
 *        implemented ones, the dot operator will perform fallback operationand still
 * \return new symbol
 */
inline Symbol dot(Symbol lhs,
                  Symbol rhs,
                  bool transpose_a = false,
                  bool transpose_b = false,
                  DotForwardStype forward_stype = DotForwardStype::kNone) {
  static const char *DotForwardStypeValues[] = {
    "None",
    "csr",
    "default",
    "row_sparse"
  };
  return Operator("dot")
           .SetParam("transpose_a", transpose_a)
           .SetParam("transpose_b", transpose_b)
           .SetParam("forward_stype", DotForwardStypeValues[int(forward_stype)])
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \brief Batchwise dot product.
 *
 *        ``batch_dot`` is used to compute dot product of ``x`` and ``y`` when ``x`` and
 *        ``y`` are data in batch, namely N-D (N >= 3) arrays in shape of `(B0, ..., B_i,
 *
 *        For example, given ``x`` with shape `(B_0, ..., B_i, N, M)` and ``y`` with shape
 *        `(B_0, ..., B_i, M, K)`, the result array will have shape `(B_0, ..., B_i, N,
 *        which is computed by::
 *
 *        batch_dot(x,y)[b_0, ..., b_i, :, :] = dot(x[b_0, ..., b_i, :, :], y[b_0, ...,
 *
 *
 *
 *        Defined in ../src/operator/tensor/dot.cc:L127
 * \param lhs The first input
 * \param rhs The second input
 * \param transpose_a If true then transpose the first input before dot.
 * \param transpose_b If true then transpose the second input before dot.
 * \param forward_stype The desired storage type of the forward output given by user, if
 *        thecombination of input storage types and this hint does not matchany
 *        implemented ones, the dot operator will perform fallback operationand still
 * \return new symbol
 */
inline Symbol batch_dot(Symbol lhs,
                        Symbol rhs,
                        bool transpose_a = false,
                        bool transpose_b = false,
                        Batch_dotForwardStype forward_stype = Batch_dotForwardStype::kNone) {
  static const char *Batch_dotForwardStypeValues[] = {
    "None",
    "csr",
    "default",
    "row_sparse"
  };
  return Operator("batch_dot")
           .SetParam("transpose_a", transpose_a)
           .SetParam("transpose_b", transpose_b)
           .SetParam("forward_stype", Batch_dotForwardStypeValues[int(forward_stype)])
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise sum of the input arrays with broadcasting.
 *
 *        `broadcast_plus` is an alias to the function `broadcast_add`.
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_add(x, y) = [[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]]
 *
 *        broadcast_plus(x, y) = [[ 1.,  1.,  1.],
 *        [ 2.,  2.,  2.]]
 *
 *        Supported sparse operations:
 *
 *        broadcast_add(csr, dense(1D)) = dense
 *        broadcast_add(dense(1D), csr) = dense
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L58
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_add(Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_add")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise difference of the input arrays with broadcasting.
 *
 *        `broadcast_minus` is an alias to the function `broadcast_sub`.
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_sub(x, y) = [[ 1.,  1.,  1.],
 *        [ 0.,  0.,  0.]]
 *
 *        broadcast_minus(x, y) = [[ 1.,  1.,  1.],
 *        [ 0.,  0.,  0.]]
 *
 *        Supported sparse operations:
 *
 *        broadcast_sub/minus(csr, dense(1D)) = dense
 *        broadcast_sub/minus(dense(1D), csr) = dense
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L106
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_sub(Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_sub")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise product of the input arrays with broadcasting.
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_mul(x, y) = [[ 0.,  0.,  0.],
 *        [ 1.,  1.,  1.]]
 *
 *        Supported sparse operations:
 *
 *        broadcast_mul(csr, dense(1D)) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L146
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_mul(Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_mul")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise division of the input arrays with broadcasting.
 *
 *        Example::
 *
 *        x = [[ 6.,  6.,  6.],
 *        [ 6.,  6.,  6.]]
 *
 *        y = [[ 2.],
 *        [ 3.]]
 *
 *        broadcast_div(x, y) = [[ 3.,  3.,  3.],
 *        [ 2.,  2.,  2.]]
 *
 *        Supported sparse operations:
 *
 *        broadcast_div(csr, dense(1D)) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L187
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_div(Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_div")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise modulo of the input arrays with broadcasting.
 *
 *        Example::
 *
 *        x = [[ 8.,  8.,  8.],
 *        [ 8.,  8.,  8.]]
 *
 *        y = [[ 2.],
 *        [ 3.]]
 *
 *        broadcast_mod(x, y) = [[ 0.,  0.,  0.],
 *        [ 2.,  2.,  2.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L222
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_mod(Symbol lhs,
                            Symbol rhs) {
  return Operator("broadcast_mod")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \brief Returns result of first array elements raised to powers from second array,
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_power(x, y) = [[ 2.,  2.,  2.],
 *        [ 4.,  4.,  4.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L45
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_power(Symbol lhs,
                              Symbol rhs) {
  return Operator("broadcast_power")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise maximum of the input arrays with broadcasting.
 *
 *        This function compares two input arrays and returns a new array having the
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_maximum(x, y) = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L81
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_maximum(Symbol lhs,
                                Symbol rhs) {
  return Operator("broadcast_maximum")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise minimum of the input arrays with broadcasting.
 *
 *        This function compares two input arrays and returns a new array having the
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_maximum(x, y) = [[ 0.,  0.,  0.],
 *        [ 1.,  1.,  1.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L117
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_minimum(Symbol lhs,
                                Symbol rhs) {
  return Operator("broadcast_minimum")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \brief Returns the hypotenuse of a right angled triangle, given its "legs"
 *        with broadcasting.
 *
 *        It is equivalent to doing :math:`sqrt(x_1^2 + x_2^2)`.
 *
 *        Example::
 *
 *        x = [[ 3.,  3.,  3.]]
 *
 *        y = [[ 4.],
 *        [ 4.]]
 *
 *        broadcast_hypot(x, y) = [[ 5.,  5.,  5.],
 *        [ 5.,  5.,  5.]]
 *
 *        z = [[ 0.],
 *        [ 4.]]
 *
 *        broadcast_hypot(x, z) = [[ 3.,  3.,  3.],
 *        [ 5.,  5.,  5.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L158
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_hypot(Symbol lhs,
                              Symbol rhs) {
  return Operator("broadcast_hypot")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \brief Returns the result of element-wise **equal to** (==) comparison operation with
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_equal(x, y) = [[ 0.,  0.,  0.],
 *        [ 1.,  1.,  1.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L46
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_equal(Symbol lhs,
                              Symbol rhs) {
  return Operator("broadcast_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \brief Returns the result of element-wise **not equal to** (!=) comparison operation
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_not_equal(x, y) = [[ 1.,  1.,  1.],
 *        [ 0.,  0.,  0.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L64
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_not_equal(Symbol lhs,
                                  Symbol rhs) {
  return Operator("broadcast_not_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \brief Returns the result of element-wise **greater than** (>) comparison operation
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_greater(x, y) = [[ 1.,  1.,  1.],
 *        [ 0.,  0.,  0.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L82
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_greater(Symbol lhs,
                                Symbol rhs) {
  return Operator("broadcast_greater")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \brief Returns the result of element-wise **greater than or equal to** (>=) comparison
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_greater_equal(x, y) = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L100
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_greater_equal(Symbol lhs,
                                      Symbol rhs) {
  return Operator("broadcast_greater_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \brief Returns the result of element-wise **lesser than** (<) comparison operation
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_lesser(x, y) = [[ 0.,  0.,  0.],
 *        [ 0.,  0.,  0.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L118
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_lesser(Symbol lhs,
                               Symbol rhs) {
  return Operator("broadcast_lesser")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \brief Returns the result of element-wise **lesser than or equal to** (<=) comparison
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_lesser_equal(x, y) = [[ 0.,  0.,  0.],
 *        [ 1.,  1.,  1.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L136
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_lesser_equal(Symbol lhs,
                                     Symbol rhs) {
  return Operator("broadcast_lesser_equal")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \brief Returns the result of element-wise **logical and** with broadcasting.
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        y = [[ 0.],
 *        [ 1.]]
 *
 *        broadcast_logical_and(x, y) = [[ 0.,  0.,  0.],
 *        [ 1.,  1.,  1.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L154
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_logical_and(Symbol lhs,
                                    Symbol rhs) {
  return Operator("broadcast_logical_and")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \brief Returns the result of element-wise **logical or** with broadcasting.
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  0.],
 *        [ 1.,  1.,  0.]]
 *
 *        y = [[ 1.],
 *        [ 0.]]
 *
 *        broadcast_logical_or(x, y) = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  0.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L172
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_logical_or(Symbol lhs,
                                   Symbol rhs) {
  return Operator("broadcast_logical_or")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \brief Returns the result of element-wise **logical xor** with broadcasting.
 *
 *        Example::
 *
 *        x = [[ 1.,  1.,  0.],
 *        [ 1.,  1.,  0.]]
 *
 *        y = [[ 1.],
 *        [ 0.]]
 *
 *        broadcast_logical_xor(x, y) = [[ 0.,  0.,  1.],
 *        [ 1.,  1.,  0.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L190
 * \param lhs First input to the function
 * \param rhs Second input to the function
 * \return new symbol
 */
inline Symbol broadcast_logical_xor(Symbol lhs,
                                    Symbol rhs) {
  return Operator("broadcast_logical_xor")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \brief Subtracts arguments element-wise.
 *
 *        The storage type of ``elemwise_sub`` output depends on storage types of inputs
 *
 *        - elemwise_sub(row_sparse, row_sparse) = row_sparse
 *        - elemwise_sub(csr, csr) = csr
 *        - elemwise_sub(default, csr) = default
 *        - elemwise_sub(csr, default) = default
 *        - elemwise_sub(default, rsp) = default
 *        - elemwise_sub(rsp, default) = default
 *        - otherwise, ``elemwise_sub`` generates output with default storage
 *
 *
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol elemwise_sub(Symbol lhs,
                           Symbol rhs) {
  return Operator("elemwise_sub")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \brief Divides arguments element-wise.
 *
 *        The storage type of ``elemwise_div`` output is always dense
 *
 *
 * \param lhs first input
 * \param rhs second input
 * \return new symbol
 */
inline Symbol elemwise_div(Symbol lhs,
                           Symbol rhs) {
  return Operator("elemwise_div")
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \brief Calculate Smooth L1 Loss(lhs, scalar) by summing
 *
 *        .. math::
 *
 *        f(x) =
 *        \begin{cases}
 *        (\sigma x)^2/2,& \text{if }x < 1/\sigma^2\\
 *        |x|-0.5/\sigma^2,& \text{otherwise}
 *        \end{cases}
 *
 *        where :math:`x` is an element of the tensor *lhs* and :math:`\sigma` is the
 *
 *        Example::
 *
 *        smooth_l1([1, 2, 3, 4]) = [0.5, 1.5, 2.5, 3.5]
 *        smooth_l1([1, 2, 3, 4], scalar=1) = [0.5, 1.5, 2.5, 3.5]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_binary_scalar_op_extended.cc:L109
 * \param data source input
 * \param scalar scalar input
 * \return new symbol
 */
inline Symbol smooth_l1(Symbol data,
                        mx_float scalar) {
  return Operator("smooth_l1")
           .SetParam("scalar", scalar)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Adds all input arguments element-wise.
 *
 *        .. math::
 *        add\_n(a_1, a_2, ..., a_n) = a_1 + a_2 + ... + a_n
 *
 *        ``add_n`` is potentially more efficient than calling ``add`` by `n` times.
 *
 *        The storage type of ``add_n`` output depends on storage types of inputs
 *
 *        - add_n(row_sparse, row_sparse, ..) = row_sparse
 *        - add_n(default, csr, default) = default
 *        - add_n(any input combinations longer than 4 (>4) with at least one default
 *        - otherwise, ``add_n`` falls all inputs back to default storage and generates
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_sum.cc:L156
 * \param args Positional input arguments
 * \return new symbol
 */
inline Symbol add_n(const std::vector<Symbol>& args) {
  return Operator("add_n")
(args)
           .CreateSymbol();
}

/*!
 * \brief Computes rectified linear activation.
 *
 *        .. math::
 *        max(features, 0)
 *
 *        The storage type of ``relu`` output depends upon the input storage type:
 *
 *        - relu(default) = default
 *        - relu(row_sparse) = row_sparse
 *        - relu(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L85
 * \param data The input array.
 * \return new symbol
 */
inline Symbol relu(Symbol data) {
  return Operator("relu")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Computes sigmoid of x element-wise.
 *
 *        .. math::
 *        y = 1 / (1 + exp(-x))
 *
 *        The storage type of ``sigmoid`` output is always dense
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L119
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sigmoid(Symbol data) {
  return Operator("sigmoid")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Computes hard sigmoid of x element-wise.
 *
 *        .. math::
 *        y = max(0, min(1, alpha * x + beta))
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L161
 * \param data The input array.
 * \param alpha Slope of hard sigmoid
 * \param beta Bias of hard sigmoid.
 * \return new symbol
 */
inline Symbol hard_sigmoid(Symbol data,
                           mx_float alpha = 0.200000003,
                           mx_float beta = 0.5) {
  return Operator("hard_sigmoid")
           .SetParam("alpha", alpha)
           .SetParam("beta", beta)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Computes softsign of x element-wise.
 *
 *        .. math::
 *        y = x / (1 + abs(x))
 *
 *        The storage type of ``softsign`` output is always dense
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L191
 * \param data The input array.
 * \return new symbol
 */
inline Symbol softsign(Symbol data) {
  return Operator("softsign")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Stops gradient computation.
 *
 *        Stops the accumulated gradient of the inputs from flowing through this operator
 *        in the backward direction. In other words, this operator prevents the
 *        of its inputs to be taken into account for computing gradients.
 *
 *        Example::
 *
 *        v1 = [1, 2]
 *        v2 = [0, 1]
 *        a = Variable('a')
 *        b = Variable('b')
 *        b_stop_grad = stop_gradient(3 * b)
 *        loss = MakeLoss(b_stop_grad + a)
 *
 *        executor = loss.simple_bind(ctx=cpu(), a=(1,2), b=(1,2))
 *        executor.forward(is_train=True, a=v1, b=v2)
 *        executor.outputs
 *        [ 1.  5.]
 *
 *        executor.backward()
 *        executor.grad_arrays
 *        [ 0.  0.]
 *        [ 1.  1.]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L325
 * \param data The input array.
 * \return new symbol
 */
inline Symbol BlockGrad(Symbol data) {
  return Operator("BlockGrad")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Make your own loss function in network construction.
 *
 *        This operator accepts a customized loss function symbol as a terminal loss and
 *        the symbol should be an operator with no backward dependency.
 *        The output of this function is the gradient of loss with respect to the input
 *
 *        For example, if you are a making a cross entropy loss function. Assume ``out``
 *        predicted output and ``label`` is the true label, then the cross entropy can be
 *
 *        cross_entropy = label * log(out) + (1 - label) * log(1 - out)
 *        loss = make_loss(cross_entropy)
 *
 *        We will need to use ``make_loss`` when we are creating our own loss function or
 *        combine multiple loss functions. Also we may want to stop some variables'
 *        from backpropagation. See more detail in ``BlockGrad`` or ``stop_gradient``.
 *
 *        The storage type of ``make_loss`` output depends upon the input storage type:
 *
 *        - make_loss(default) = default
 *        - make_loss(row_sparse) = row_sparse
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L358
 * \param data The input array.
 * \return new symbol
 */
inline Symbol make_loss(Symbol data) {
  return Operator("make_loss")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Reshape some or all dimensions of `lhs` to have the same shape as some or all
 *
 *        Returns a **view** of the `lhs` array with a new shape without altering any
 *
 *        Example::
 *
 *        x = [1, 2, 3, 4, 5, 6]
 *        y = [[0, -4], [3, 2], [2, 2]]
 *        reshape_like(x, y) = [[1, 2], [3, 4], [5, 6]]
 *
 *        More precise control over how dimensions are inherited is achieved by
 *        slices over the `lhs` and `rhs` array dimensions. Only the sliced `lhs`
 *        are reshaped to the `rhs` sliced dimensions, with the non-sliced `lhs`
 *
 *        Examples::
 *
 *        - lhs shape = (30,7), rhs shape = (15,2,4), lhs_begin=0, lhs_end=1,
 *        - lhs shape = (3, 5), rhs shape = (1,15,4), lhs_begin=0, lhs_end=2,
 *
 *        Negative indices are supported, and `None` can be used for either `lhs_end` or
 *
 *        Example::
 *
 *        - lhs shape = (30, 12), rhs shape = (4, 2, 2, 3), lhs_begin=-1, lhs_end=None,
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L511
 * \param lhs First input.
 * \param rhs Second input.
 * \param lhs_begin Defaults to 0. The beginning index along which the lhs dimensions are
 * \param lhs_end Defaults to None. The ending index along which the lhs dimensions are
 * \param rhs_begin Defaults to 0. The beginning index along which the rhs dimensions are
 * \param rhs_end Defaults to None. The ending index along which the rhs dimensions are
 * \return new symbol
 */
inline Symbol reshape_like(Symbol lhs,
                           Symbol rhs,
                           dmlc::optional<int> lhs_begin = dmlc::optional<int>(),
                           dmlc::optional<int> lhs_end = dmlc::optional<int>(),
                           dmlc::optional<int> rhs_begin = dmlc::optional<int>(),
                           dmlc::optional<int> rhs_end = dmlc::optional<int>()) {
  return Operator("reshape_like")
           .SetParam("lhs_begin", lhs_begin)
           .SetParam("lhs_end", lhs_end)
           .SetParam("rhs_begin", rhs_begin)
           .SetParam("rhs_end", rhs_end)
           .SetInput("lhs", lhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

/*!
 * \brief Returns a 1D int64 array containing the shape of data.
 *
 *        Example::
 *
 *        shape_array([[1,2,3,4], [5,6,7,8]]) = [2,4]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L573
 * \param data Input Array.
 * \return new symbol
 */
inline Symbol shape_array(Symbol data) {
  return Operator("shape_array")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns a 1D int64 array containing the size of data.
 *
 *        Example::
 *
 *        size_array([[1,2,3,4], [5,6,7,8]]) = [8]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L624
 * \param data Input Array.
 * \return new symbol
 */
inline Symbol size_array(Symbol data) {
  return Operator("size_array")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Casts all elements of the input to a new type.
 *
 *        .. note:: ``Cast`` is deprecated. Use ``cast`` instead.
 *
 *        Example::
 *
 *        cast([0.9, 1.3], dtype='int32') = [0, 1]
 *        cast([1e20, 11.1], dtype='float16') = [inf, 11.09375]
 *        cast([300, 11.1, 10.9, -1, -3], dtype='uint8') = [44, 11, 10, 255, 253]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L664
 * \param data The input.
 * \param dtype Output data type.
 * \return new symbol
 */
inline Symbol Cast(Symbol data,
                   CastDtype dtype) {
  static const char *CastDtypeValues[] = {
    "bfloat16",
    "bool",
    "float16",
    "float32",
    "float64",
    "int32",
    "int64",
    "int8",
    "uint8"
  };
  return Operator("Cast")
           .SetParam("dtype", CastDtypeValues[int(dtype)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Numerical negative of the argument, element-wise.
 *
 *        The storage type of ``negative`` output depends upon the input storage type:
 *
 *        - negative(default) = default
 *        - negative(row_sparse) = row_sparse
 *        - negative(csr) = csr
 *
 *
 * \param data The input array.
 * \return new symbol
 */
inline Symbol negative(Symbol data) {
  return Operator("negative")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise absolute value of the input.
 *
 *        Example::
 *
 *        abs([-2, 0, 3]) = [2, 0, 3]
 *
 *        The storage type of ``abs`` output depends upon the input storage type:
 *
 *        - abs(default) = default
 *        - abs(row_sparse) = row_sparse
 *        - abs(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L720
 * \param data The input array.
 * \return new symbol
 */
inline Symbol abs(Symbol data) {
  return Operator("abs")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise sign of the input.
 *
 *        Example::
 *
 *        sign([-2, 0, 3]) = [-1, 0, 1]
 *
 *        The storage type of ``sign`` output depends upon the input storage type:
 *
 *        - sign(default) = default
 *        - sign(row_sparse) = row_sparse
 *        - sign(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L758
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sign(Symbol data) {
  return Operator("sign")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise rounded value to the nearest integer of the input.
 *
 *        Example::
 *
 *        round([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  2., -2.,  2.,  2.]
 *
 *        The storage type of ``round`` output depends upon the input storage type:
 *
 *        - round(default) = default
 *        - round(row_sparse) = row_sparse
 *        - round(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L777
 * \param data The input array.
 * \return new symbol
 */
inline Symbol round(Symbol data) {
  return Operator("round")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise rounded value to the nearest integer of the input.
 *
 *        .. note::
 *        - For input ``n.5`` ``rint`` returns ``n`` while ``round`` returns ``n+1``.
 *        - For input ``-n.5`` both ``rint`` and ``round`` returns ``-n-1``.
 *
 *        Example::
 *
 *        rint([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  1., -2.,  2.,  2.]
 *
 *        The storage type of ``rint`` output depends upon the input storage type:
 *
 *        - rint(default) = default
 *        - rint(row_sparse) = row_sparse
 *        - rint(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L798
 * \param data The input array.
 * \return new symbol
 */
inline Symbol rint(Symbol data) {
  return Operator("rint")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise ceiling of the input.
 *
 *        The ceil of the scalar x is the smallest integer i, such that i >= x.
 *
 *        Example::
 *
 *        ceil([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  2.,  2.,  3.]
 *
 *        The storage type of ``ceil`` output depends upon the input storage type:
 *
 *        - ceil(default) = default
 *        - ceil(row_sparse) = row_sparse
 *        - ceil(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L817
 * \param data The input array.
 * \return new symbol
 */
inline Symbol ceil(Symbol data) {
  return Operator("ceil")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise floor of the input.
 *
 *        The floor of the scalar x is the largest integer i, such that i <= x.
 *
 *        Example::
 *
 *        floor([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-3., -2.,  1.,  1.,  2.]
 *
 *        The storage type of ``floor`` output depends upon the input storage type:
 *
 *        - floor(default) = default
 *        - floor(row_sparse) = row_sparse
 *        - floor(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L836
 * \param data The input array.
 * \return new symbol
 */
inline Symbol floor(Symbol data) {
  return Operator("floor")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Return the element-wise truncated value of the input.
 *
 *        The truncated value of the scalar x is the nearest integer i which is closer to
 *        zero than x is. In short, the fractional part of the signed number x is
 *
 *        Example::
 *
 *        trunc([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  1.,  1.,  2.]
 *
 *        The storage type of ``trunc`` output depends upon the input storage type:
 *
 *        - trunc(default) = default
 *        - trunc(row_sparse) = row_sparse
 *        - trunc(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L856
 * \param data The input array.
 * \return new symbol
 */
inline Symbol trunc(Symbol data) {
  return Operator("trunc")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise rounded value to the nearest \
 *        integer towards zero of the input.
 *
 *        Example::
 *
 *        fix([-2.1, -1.9, 1.9, 2.1]) = [-2., -1.,  1., 2.]
 *
 *        The storage type of ``fix`` output depends upon the input storage type:
 *
 *        - fix(default) = default
 *        - fix(row_sparse) = row_sparse
 *        - fix(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L874
 * \param data The input array.
 * \return new symbol
 */
inline Symbol fix(Symbol data) {
  return Operator("fix")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise gauss error function of the input.
 *
 *        Example::
 *
 *        erf([0, -1., 10.]) = [0., -0.8427, 1.]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L886
 * \param data The input array.
 * \return new symbol
 */
inline Symbol erf(Symbol data) {
  return Operator("erf")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise inverse gauss error function of the input.
 *
 *        Example::
 *
 *        erfinv([0, 0.5., -1.]) = [0., 0.4769, -inf]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L908
 * \param data The input array.
 * \return new symbol
 */
inline Symbol erfinv(Symbol data) {
  return Operator("erfinv")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns the gamma function (extension of the factorial function \
 *        to the reals), computed element-wise on the input array.
 *
 *        The storage type of ``gamma`` output is always dense
 *
 *
 * \param data The input array.
 * \return new symbol
 */
inline Symbol gamma(Symbol data) {
  return Operator("gamma")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise log of the absolute value of the gamma function \
 *        of the input.
 *
 *        The storage type of ``gammaln`` output is always dense
 *
 *
 * \param data The input array.
 * \return new symbol
 */
inline Symbol gammaln(Symbol data) {
  return Operator("gammaln")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns the result of logical NOT (!) function
 *
 *        Example:
 *        logical_not([-2., 0., 1.]) = [0., 1., 0.]
 *
 *
 * \param data The input array.
 * \return new symbol
 */
inline Symbol logical_not(Symbol data) {
  return Operator("logical_not")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise exponential value of the input.
 *
 *        .. math::
 *        exp(x) = e^x \approx 2.718^x
 *
 *        Example::
 *
 *        exp([0, 1, 2]) = [1., 2.71828175, 7.38905621]
 *
 *        The storage type of ``exp`` output is always dense
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_logexp.cc:L64
 * \param data The input array.
 * \return new symbol
 */
inline Symbol exp(Symbol data) {
  return Operator("exp")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise Natural logarithmic value of the input.
 *
 *        The natural logarithm is logarithm in base *e*, so that ``log(exp(x)) = x``
 *
 *        The storage type of ``log`` output is always dense
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_logexp.cc:L77
 * \param data The input array.
 * \return new symbol
 */
inline Symbol log(Symbol data) {
  return Operator("log")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise Base-10 logarithmic value of the input.
 *
 *        ``10**log10(x) = x``
 *
 *        The storage type of ``log10`` output is always dense
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_logexp.cc:L94
 * \param data The input array.
 * \return new symbol
 */
inline Symbol log10(Symbol data) {
  return Operator("log10")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise Base-2 logarithmic value of the input.
 *
 *        ``2**log2(x) = x``
 *
 *        The storage type of ``log2`` output is always dense
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_logexp.cc:L106
 * \param data The input array.
 * \return new symbol
 */
inline Symbol log2(Symbol data) {
  return Operator("log2")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise ``log(1 + x)`` value of the input.
 *
 *        This function is more accurate than ``log(1 + x)``  for small ``x`` so that
 *        :math:`1+x\approx 1`
 *
 *        The storage type of ``log1p`` output depends upon the input storage type:
 *
 *        - log1p(default) = default
 *        - log1p(row_sparse) = row_sparse
 *        - log1p(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_logexp.cc:L199
 * \param data The input array.
 * \return new symbol
 */
inline Symbol log1p(Symbol data) {
  return Operator("log1p")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns ``exp(x) - 1`` computed element-wise on the input.
 *
 *        This function provides greater precision than ``exp(x) - 1`` for small values
 *
 *        The storage type of ``expm1`` output depends upon the input storage type:
 *
 *        - expm1(default) = default
 *        - expm1(row_sparse) = row_sparse
 *        - expm1(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_logexp.cc:L244
 * \param data The input array.
 * \return new symbol
 */
inline Symbol expm1(Symbol data) {
  return Operator("expm1")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns the reciprocal of the argument, element-wise.
 *
 *        Calculates 1/x.
 *
 *        Example::
 *
 *        reciprocal([-2, 1, 3, 1.6, 0.2]) = [-0.5, 1.0, 0.33333334, 0.625, 5.0]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_pow.cc:L43
 * \param data The input array.
 * \return new symbol
 */
inline Symbol reciprocal(Symbol data) {
  return Operator("reciprocal")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise squared value of the input.
 *
 *        .. math::
 *        square(x) = x^2
 *
 *        Example::
 *
 *        square([2, 3, 4]) = [4, 9, 16]
 *
 *        The storage type of ``square`` output depends upon the input storage type:
 *
 *        - square(default) = default
 *        - square(row_sparse) = row_sparse
 *        - square(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_pow.cc:L119
 * \param data The input array.
 * \return new symbol
 */
inline Symbol square(Symbol data) {
  return Operator("square")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise square-root value of the input.
 *
 *        .. math::
 *        \textrm{sqrt}(x) = \sqrt{x}
 *
 *        Example::
 *
 *        sqrt([4, 9, 16]) = [2, 3, 4]
 *
 *        The storage type of ``sqrt`` output depends upon the input storage type:
 *
 *        - sqrt(default) = default
 *        - sqrt(row_sparse) = row_sparse
 *        - sqrt(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_pow.cc:L170
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sqrt(Symbol data) {
  return Operator("sqrt")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise inverse square-root value of the input.
 *
 *        .. math::
 *        rsqrt(x) = 1/\sqrt{x}
 *
 *        Example::
 *
 *        rsqrt([4,9,16]) = [0.5, 0.33333334, 0.25]
 *
 *        The storage type of ``rsqrt`` output is always dense
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_pow.cc:L221
 * \param data The input array.
 * \return new symbol
 */
inline Symbol rsqrt(Symbol data) {
  return Operator("rsqrt")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise cube-root value of the input.
 *
 *        .. math::
 *        cbrt(x) = \sqrt[3]{x}
 *
 *        Example::
 *
 *        cbrt([1, 8, -125]) = [1, 2, -5]
 *
 *        The storage type of ``cbrt`` output depends upon the input storage type:
 *
 *        - cbrt(default) = default
 *        - cbrt(row_sparse) = row_sparse
 *        - cbrt(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_pow.cc:L270
 * \param data The input array.
 * \return new symbol
 */
inline Symbol cbrt(Symbol data) {
  return Operator("cbrt")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise inverse cube-root value of the input.
 *
 *        .. math::
 *        rcbrt(x) = 1/\sqrt[3]{x}
 *
 *        Example::
 *
 *        rcbrt([1,8,-125]) = [1.0, 0.5, -0.2]
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_pow.cc:L323
 * \param data The input array.
 * \return new symbol
 */
inline Symbol rcbrt(Symbol data) {
  return Operator("rcbrt")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Computes the element-wise sine of the input array.
 *
 *        The input should be in radians (:math:`2\pi` rad equals 360 degrees).
 *
 *        .. math::
 *        sin([0, \pi/4, \pi/2]) = [0, 0.707, 1]
 *
 *        The storage type of ``sin`` output depends upon the input storage type:
 *
 *        - sin(default) = default
 *        - sin(row_sparse) = row_sparse
 *        - sin(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_trig.cc:L47
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sin(Symbol data) {
  return Operator("sin")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Computes the element-wise cosine of the input array.
 *
 *        The input should be in radians (:math:`2\pi` rad equals 360 degrees).
 *
 *        .. math::
 *        cos([0, \pi/4, \pi/2]) = [1, 0.707, 0]
 *
 *        The storage type of ``cos`` output is always dense
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_trig.cc:L90
 * \param data The input array.
 * \return new symbol
 */
inline Symbol cos(Symbol data) {
  return Operator("cos")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Computes the element-wise tangent of the input array.
 *
 *        The input should be in radians (:math:`2\pi` rad equals 360 degrees).
 *
 *        .. math::
 *        tan([0, \pi/4, \pi/2]) = [0, 1, -inf]
 *
 *        The storage type of ``tan`` output depends upon the input storage type:
 *
 *        - tan(default) = default
 *        - tan(row_sparse) = row_sparse
 *        - tan(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_trig.cc:L140
 * \param data The input array.
 * \return new symbol
 */
inline Symbol tan(Symbol data) {
  return Operator("tan")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise inverse sine of the input array.
 *
 *        The input should be in the range `[-1, 1]`.
 *        The output is in the closed interval of [:math:`-\pi/2`, :math:`\pi/2`].
 *
 *        .. math::
 *        arcsin([-1, -.707, 0, .707, 1]) = [-\pi/2, -\pi/4, 0, \pi/4, \pi/2]
 *
 *        The storage type of ``arcsin`` output depends upon the input storage type:
 *
 *        - arcsin(default) = default
 *        - arcsin(row_sparse) = row_sparse
 *        - arcsin(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_trig.cc:L187
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arcsin(Symbol data) {
  return Operator("arcsin")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise inverse cosine of the input array.
 *
 *        The input should be in range `[-1, 1]`.
 *        The output is in the closed interval :math:`[0, \pi]`
 *
 *        .. math::
 *        arccos([-1, -.707, 0, .707, 1]) = [\pi, 3\pi/4, \pi/2, \pi/4, 0]
 *
 *        The storage type of ``arccos`` output is always dense
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_trig.cc:L233
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arccos(Symbol data) {
  return Operator("arccos")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns element-wise inverse tangent of the input array.
 *
 *        The output is in the closed interval :math:`[-\pi/2, \pi/2]`
 *
 *        .. math::
 *        arctan([-1, 0, 1]) = [-\pi/4, 0, \pi/4]
 *
 *        The storage type of ``arctan`` output depends upon the input storage type:
 *
 *        - arctan(default) = default
 *        - arctan(row_sparse) = row_sparse
 *        - arctan(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_trig.cc:L282
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arctan(Symbol data) {
  return Operator("arctan")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Converts each element of the input array from radians to degrees.
 *
 *        .. math::
 *        degrees([0, \pi/2, \pi, 3\pi/2, 2\pi]) = [0, 90, 180, 270, 360]
 *
 *        The storage type of ``degrees`` output depends upon the input storage type:
 *
 *        - degrees(default) = default
 *        - degrees(row_sparse) = row_sparse
 *        - degrees(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_trig.cc:L332
 * \param data The input array.
 * \return new symbol
 */
inline Symbol degrees(Symbol data) {
  return Operator("degrees")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Converts each element of the input array from degrees to radians.
 *
 *        .. math::
 *        radians([0, 90, 180, 270, 360]) = [0, \pi/2, \pi, 3\pi/2, 2\pi]
 *
 *        The storage type of ``radians`` output depends upon the input storage type:
 *
 *        - radians(default) = default
 *        - radians(row_sparse) = row_sparse
 *        - radians(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_trig.cc:L351
 * \param data The input array.
 * \return new symbol
 */
inline Symbol radians(Symbol data) {
  return Operator("radians")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns the hyperbolic sine of the input array, computed element-wise.
 *
 *        .. math::
 *        sinh(x) = 0.5\times(exp(x) - exp(-x))
 *
 *        The storage type of ``sinh`` output depends upon the input storage type:
 *
 *        - sinh(default) = default
 *        - sinh(row_sparse) = row_sparse
 *        - sinh(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_trig.cc:L371
 * \param data The input array.
 * \return new symbol
 */
inline Symbol sinh(Symbol data) {
  return Operator("sinh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns the hyperbolic cosine  of the input array, computed element-wise.
 *
 *        .. math::
 *        cosh(x) = 0.5\times(exp(x) + exp(-x))
 *
 *        The storage type of ``cosh`` output is always dense
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_trig.cc:L409
 * \param data The input array.
 * \return new symbol
 */
inline Symbol cosh(Symbol data) {
  return Operator("cosh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns the hyperbolic tangent of the input array, computed element-wise.
 *
 *        .. math::
 *        tanh(x) = sinh(x) / cosh(x)
 *
 *        The storage type of ``tanh`` output depends upon the input storage type:
 *
 *        - tanh(default) = default
 *        - tanh(row_sparse) = row_sparse
 *        - tanh(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_trig.cc:L451
 * \param data The input array.
 * \return new symbol
 */
inline Symbol tanh(Symbol data) {
  return Operator("tanh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns the element-wise inverse hyperbolic sine of the input array, \
 *        computed element-wise.
 *
 *        The storage type of ``arcsinh`` output depends upon the input storage type:
 *
 *        - arcsinh(default) = default
 *        - arcsinh(row_sparse) = row_sparse
 *        - arcsinh(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_trig.cc:L494
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arcsinh(Symbol data) {
  return Operator("arcsinh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns the element-wise inverse hyperbolic cosine of the input array, \
 *        computed element-wise.
 *
 *        The storage type of ``arccosh`` output is always dense
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_trig.cc:L535
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arccosh(Symbol data) {
  return Operator("arccosh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns the element-wise inverse hyperbolic tangent of the input array, \
 *        computed element-wise.
 *
 *        The storage type of ``arctanh`` output depends upon the input storage type:
 *
 *        - arctanh(default) = default
 *        - arctanh(row_sparse) = row_sparse
 *        - arctanh(csr) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/elemwise_unary_op_trig.cc:L579
 * \param data The input array.
 * \return new symbol
 */
inline Symbol arctanh(Symbol data) {
  return Operator("arctanh")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Takes elements from an input array along the given axis.
 *
 *        This function slices the input array along a particular axis with the provided
 *
 *        Given data tensor of rank r >= 1, and indices tensor of rank q, gather entries
 *        dimension of data (by default outer-most one as axis=0) indexed by indices, and
 *        in an output tensor of rank q + (r - 1).
 *
 *        Examples::
 *
 *        x = [4.  5.  6.]
 *
 *        // Trivial case, take the second element along the first axis.
 *
 *        take(x, [1]) = [ 5. ]
 *
 *        // The other trivial case, axis=-1, take the third element along the first axis
 *
 *        take(x, [3], axis=-1, mode='clip') = [ 6. ]
 *
 *        x = [[ 1.,  2.],
 *        [ 3.,  4.],
 *        [ 5.,  6.]]
 *
 *        // In this case we will get rows 0 and 1, then 1 and 2. Along axis 0
 *
 *        take(x, [[0,1],[1,2]]) = [[[ 1.,  2.],
 *        [ 3.,  4.]],
 *
 *        [[ 3.,  4.],
 *        [ 5.,  6.]]]
 *
 *        // In this case we will get rows 0 and 1, then 1 and 2 (calculated by wrapping
 *        // Along axis 1
 *
 *        take(x, [[0, 3], [-1, -2]], axis=1, mode='wrap') = [[[ 1.  2.]
 *        [ 2.  1.]]
 *
 *        [[ 3.  4.]
 *        [ 4.  3.]]
 *
 *        [[ 5.  6.]
 *        [ 6.  5.]]]
 *
 *        The storage type of ``take`` output depends upon the input storage type:
 *
 *        - take(default, default) = default
 *        - take(csr, default, axis=0) = csr
 *
 *
 *
 *        Defined in ../src/operator/tensor/indexing_op.cc:L777
 * \param a The input array.
 * \param indices The indices of the values to be extracted.
 * \param axis The axis of input array to be taken.For input tensor of rank r, it could
 * \param mode Specify how out-of-bound indices bahave. Default is "clip". "clip" means
 *        clip to the range. So, if all indices mentioned are too large, they are
 *        replaced by the index that addresses the last element along an axis. "wrap"
 * \return new symbol
 */
inline Symbol take(Symbol a,
                   Symbol indices,
                   int axis = 0,
                   TakeMode mode = TakeMode::kClip) {
  static const char *TakeModeValues[] = {
    "clip",
    "raise",
    "wrap"
  };
  return Operator("take")
           .SetParam("axis", axis)
           .SetParam("mode", TakeModeValues[int(mode)])
           .SetInput("a", a)
           .SetInput("indices", indices)
           .CreateSymbol();
}

/*!
 * \brief Takes elements from a data batch.
 *
 *        .. note::
 *        `batch_take` is deprecated. Use `pick` instead.
 *
 *        Given an input array of shape ``(d0, d1)`` and indices of shape ``(i0,)``, the
 *        an output array of shape ``(i0,)`` with::
 *
 *        output[i] = input[i, indices[i]]
 *
 *        Examples::
 *
 *        x = [[ 1.,  2.],
 *        [ 3.,  4.],
 *        [ 5.,  6.]]
 *
 *        // takes elements with specified indices
 *        batch_take(x, [0,1,0]) = [ 1.  4.  5.]
 *
 *
 *
 *        Defined in ../src/operator/tensor/indexing_op.cc:L836
 * \param a The input array
 * \param indices The index array
 * \return new symbol
 */
inline Symbol batch_take(Symbol a,
                         Symbol indices) {
  return Operator("batch_take")
           .SetInput("a", a)
           .SetInput("indices", indices)
           .CreateSymbol();
}

/*!
 * \brief Returns a one-hot array.
 *
 *        The locations represented by `indices` take value `on_value`, while all
 *        other locations take value `off_value`.
 *
 *        `one_hot` operation with `indices` of shape ``(i0, i1)`` and `depth`  of ``d``
 *        in an output array of shape ``(i0, i1, d)`` with::
 *
 *        output[i,j,:] = off_value
 *        output[i,j,indices[i,j]] = on_value
 *
 *        Examples::
 *
 *        one_hot([1,0,2,0], 3) = [[ 0.  1.  0.]
 *        [ 1.  0.  0.]
 *        [ 0.  0.  1.]
 *        [ 1.  0.  0.]]
 *
 *        one_hot([1,0,2,0], 3, on_value=8, off_value=1,
 *        dtype='int32') = [[1 8 1]
 *        [8 1 1]
 *        [1 1 8]
 *        [8 1 1]]
 *
 *        one_hot([[1,0],[1,0],[2,0]], 3) = [[[ 0.  1.  0.]
 *        [ 1.  0.  0.]]
 *
 *        [[ 0.  1.  0.]
 *        [ 1.  0.  0.]]
 *
 *        [[ 0.  0.  1.]
 *        [ 1.  0.  0.]]]
 *
 *
 *        Defined in ../src/operator/tensor/indexing_op.cc:L883
 * \param indices array of locations where to set on_value
 * \param depth Depth of the one hot dimension.
 * \param on_value The value assigned to the locations represented by indices.
 * \param off_value The value assigned to the locations not represented by indices.
 * \param dtype DType of the output
 * \return new symbol
 */
inline Symbol one_hot(Symbol indices,
                      int depth,
                      double on_value = 1,
                      double off_value = 0,
                      One_hotDtype dtype = One_hotDtype::kFloat32) {
  static const char *One_hotDtypeValues[] = {
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "int32",
    "int64",
    "int8",
    "uint8"
  };
  return Operator("one_hot")
           .SetParam("depth", depth)
           .SetParam("on_value", on_value)
           .SetParam("off_value", off_value)
           .SetParam("dtype", One_hotDtypeValues[int(dtype)])
           .SetInput("indices", indices)
           .CreateSymbol();
}

/*!
 * \brief Gather elements or slices from `data` and store to a tensor whose
 *        shape is defined by `indices`.
 *
 *        Given `data` with shape `(X_0, X_1, ..., X_{N-1})` and indices with shape
 *        `(M, Y_0, ..., Y_{K-1})`, the output will have shape `(Y_0, ..., Y_{K-1}, X_M,
 *        where `M <= N`. If `M == N`, output shape will simply be `(Y_0, ..., Y_{K-1})`.
 *
 *        The elements in output is defined as follows::
 *
 *        output[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}] = data[indices[0, y_0, ...,
 *        ...,
 *        indices[M-1, y_0, ..., y_{K-1}],
 *        x_M, ..., x_{N-1}]
 *
 *        Examples::
 *
 *        data = [[0, 1], [2, 3]]
 *        indices = [[1, 1, 0], [0, 1, 0]]
 *        gather_nd(data, indices) = [2, 3, 0]
 *
 *        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
 *        indices = [[0, 1], [1, 0]]
 *        gather_nd(data, indices) = [[3, 4], [5, 6]]
 *
 *
 * \param data data
 * \param indices indices
 * \return new symbol
 */
inline Symbol gather_nd(Symbol data,
                        Symbol indices) {
  return Operator("gather_nd")
           .SetInput("data", data)
           .SetInput("indices", indices)
           .CreateSymbol();
}

/*!
 * \brief Scatters data into a new tensor according to indices.
 *
 *        Given `data` with shape `(Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1})` and indices
 *        `(M, Y_0, ..., Y_{K-1})`, the output will have shape `(X_0, X_1, ..., X_{N-1})`,
 *        where `M <= N`. If `M == N`, data shape should simply be `(Y_0, ..., Y_{K-1})`.
 *
 *        The elements in output is defined as follows::
 *
 *        output[indices[0, y_0, ..., y_{K-1}],
 *        ...,
 *        indices[M-1, y_0, ..., y_{K-1}],
 *        x_M, ..., x_{N-1}] = data[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}]
 *
 *        all other entries in output are 0.
 *
 *        .. warning::
 *
 *        If the indices have duplicates, the result will be non-deterministic and
 *        the gradient of `scatter_nd` will not be correct!!
 *
 *
 *        Examples::
 *
 *        data = [2, 3, 0]
 *        indices = [[1, 1, 0], [0, 1, 0]]
 *        shape = (2, 2)
 *        scatter_nd(data, indices, shape) = [[0, 0], [2, 3]]
 *
 *        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
 *        indices = [[0, 1], [1, 1]]
 *        shape = (2, 2, 2, 2)
 *        scatter_nd(data, indices, shape) = [[[[0, 0],
 *        [0, 0]],
 *
 *        [[1, 2],
 *        [3, 4]]],
 *
 *        [[[0, 0],
 *        [0, 0]],
 *
 *        [[5, 6],
 *        [7, 8]]]]
 *
 *
 * \param data data
 * \param indices indices
 * \param shape Shape of output.
 * \return new symbol
 */
inline Symbol scatter_nd(Symbol data,
                         Symbol indices,
                         Shape shape) {
  return Operator("scatter_nd")
           .SetParam("shape", shape)
           .SetInput("data", data)
           .SetInput("indices", indices)
           .CreateSymbol();
}

/*!
 * \brief Return an array of zeros with the same shape, type and storage type
 *        as the input array.
 *
 *        The storage type of ``zeros_like`` output depends on the storage type of the
 *
 *        - zeros_like(row_sparse) = row_sparse
 *        - zeros_like(csr) = csr
 *        - zeros_like(default) = default
 *
 *        Examples::
 *
 *        x = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *        zeros_like(x) = [[ 0.,  0.,  0.],
 *        [ 0.,  0.,  0.]]
 *
 *
 * \param data The input
 * \return new symbol
 */
inline Symbol zeros_like(Symbol data) {
  return Operator("zeros_like")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Return an array of ones with the same shape and type
 *        as the input array.
 *
 *        Examples::
 *
 *        x = [[ 0.,  0.,  0.],
 *        [ 0.,  0.,  0.]]
 *
 *        ones_like(x) = [[ 1.,  1.,  1.],
 *        [ 1.,  1.,  1.]]
 *
 *
 * \param data The input
 * \return new symbol
 */
inline Symbol ones_like(Symbol data) {
  return Operator("ones_like")
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Reshapes the input array.
 *        .. note:: ``Reshape`` is deprecated, use ``reshape``
 *        Given an array and a shape, this function returns a copy of the array in the
 *        The shape is a tuple of integers such as (2,3,4). The size of the new shape
 *        Example::
 *        reshape([1,2,3,4], shape=(2,2)) = [[1,2], [3,4]]
 *        Some dimensions of the shape can take special values from the set {0, -1, -2,
 *        - ``0``  copy this dimension from the input to the output shape.
 *        Example::
 *        - input shape = (2,3,4), shape = (4,0,2), output shape = (4,3,2)
 *        - input shape = (2,3,4), shape = (2,0,0), output shape = (2,3,4)
 *        - ``-1`` infers the dimension of the output shape by using the remainder of the
 *        keeping the size of the new array same as that of the input array.
 *        At most one dimension of shape can be -1.
 *        Example::
 *        - input shape = (2,3,4), shape = (6,1,-1), output shape = (6,1,4)
 *        - input shape = (2,3,4), shape = (3,-1,8), output shape = (3,1,8)
 *        - input shape = (2,3,4), shape=(-1,), output shape = (24,)
 *        - ``-2`` copy all/remainder of the input dimensions to the output shape.
 *        Example::
 *        - input shape = (2,3,4), shape = (-2,), output shape = (2,3,4)
 *        - input shape = (2,3,4), shape = (2,-2), output shape = (2,3,4)
 *        - input shape = (2,3,4), shape = (-2,1,1), output shape = (2,3,4,1,1)
 *        - ``-3`` use the product of two consecutive dimensions of the input shape as
 *        Example::
 *        - input shape = (2,3,4), shape = (-3,4), output shape = (6,4)
 *        - input shape = (2,3,4,5), shape = (-3,-3), output shape = (6,20)
 *        - input shape = (2,3,4), shape = (0,-3), output shape = (2,12)
 *        - input shape = (2,3,4), shape = (-3,-2), output shape = (6,4)
 *        - ``-4`` split one dimension of the input into two dimensions passed subsequent
 *        Example::
 *        - input shape = (2,3,4), shape = (-4,1,2,-2), output shape =(1,2,3,4)
 *        - input shape = (2,3,4), shape = (2,-4,-1,3,-2), output shape = (2,1,3,4)
 *        If the argument `reverse` is set to 1, then the special values are inferred
 *        Example::
 *        - without reverse=1, for input shape = (10,5,4), shape = (-1,0), output shape
 *        - with reverse=1, output shape will be (50,4).
 *
 *
 *        Defined in ../src/operator/tensor/matrix_op.cc:L175
 * \param data Input data to reshape.
 * \param shape The target shape
 * \param reverse If true then the special values are inferred from right to left
 * \param target_shape (Deprecated! Use ``shape`` instead.) Target new shape. One and
 * \param keep_highest (Deprecated! Use ``shape`` instead.) Whether keep the highest dim
 *        unchanged.If set to true, then the first dim in target_shape is ignored,and
 * \return new symbol
 */
inline Symbol Reshape(Symbol data,
                      Shape shape = Shape(),
                      bool reverse = false,
                      Shape target_shape = Shape(),
                      bool keep_highest = false) {
  return Operator("Reshape")
           .SetParam("shape", shape)
           .SetParam("reverse", reverse)
           .SetParam("target_shape", target_shape)
           .SetParam("keep_highest", keep_highest)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Permutes the dimensions of an array.
 *        Examples::
 *        x = [[ 1, 2],
 *        [ 3, 4]]
 *        transpose(x) = [[ 1.,  3.],
 *        [ 2.,  4.]]
 *        x = [[[ 1.,  2.],
 *        [ 3.,  4.]],
 *        [[ 5.,  6.],
 *        [ 7.,  8.]]]
 *        transpose(x) = [[[ 1.,  5.],
 *        [ 3.,  7.]],
 *        [[ 2.,  6.],
 *        [ 4.,  8.]]]
 *        transpose(x, axes=(1,0,2)) = [[[ 1.,  2.],
 *        [ 5.,  6.]],
 *        [[ 3.,  4.],
 *        [ 7.,  8.]]]
 *
 *
 *        Defined in ../src/operator/tensor/matrix_op.cc:L328
 * \param data Source input
 * \param axes Target axis order. By default the axes will be inverted.
 * \return new symbol
 */
inline Symbol transpose(Symbol data,
                        Shape axes = Shape()) {
  return Operator("transpose")
           .SetParam("axes", axes)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Inserts a new axis of size 1 into the array shape
 *        For example, given ``x`` with shape ``(2,3,4)``, then ``expand_dims(x, axis=1)``
 *        will return a new array with shape ``(2,1,3,4)``.
 *
 *
 *        Defined in ../src/operator/tensor/matrix_op.cc:L395
 * \param data Source input
 * \param axis Position where new axis is to be inserted. Suppose that the input
 *        `NDArray`'s dimension is `ndim`, the range of the inserted axis is `[-ndim,
 * \return new symbol
 */
inline Symbol expand_dims(Symbol data,
                          int axis) {
  return Operator("expand_dims")
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Slices a region of the array.
 *        .. note:: ``crop`` is deprecated. Use ``slice`` instead.
 *        This function returns a sliced array between the indices given
 *        by `begin` and `end` with the corresponding `step`.
 *        For an input array of ``shape=(d_0, d_1, ..., d_n-1)``,
 *        slice operation with ``begin=(b_0, b_1...b_m-1)``,
 *        ``end=(e_0, e_1, ..., e_m-1)``, and ``step=(s_0, s_1, ..., s_m-1)``,
 *        where m <= n, results in an array with the shape
 *        ``(|e_0-b_0|/|s_0|, ..., |e_m-1-b_m-1|/|s_m-1|, d_m, ..., d_n-1)``.
 *        The resulting array's *k*-th dimension contains elements
 *        from the *k*-th dimension of the input array starting
 *        from index ``b_k`` (inclusive) with step ``s_k``
 *        until reaching ``e_k`` (exclusive).
 *        If the *k*-th elements are `None` in the sequence of `begin`, `end`,
 *        and `step`, the following rule will be used to set default values.
 *        If `s_k` is `None`, set `s_k=1`. If `s_k > 0`, set `b_k=0`, `e_k=d_k`;
 *        else, set `b_k=d_k-1`, `e_k=-1`.
 *        The storage type of ``slice`` output depends on storage types of inputs
 *        - slice(csr) = csr
 *        - otherwise, ``slice`` generates output with default storage
 *        .. note:: When input data storage type is csr, it only supports
 *        step=(), or step=(None,), or step=(1,) to generate a csr output.
 *        For other step parameter values, it falls back to slicing
 *        a dense tensor.
 *        Example::
 *        x = [[  1.,   2.,   3.,   4.],
 *        [  5.,   6.,   7.,   8.],
 *        [  9.,  10.,  11.,  12.]]
 *        slice(x, begin=(0,1), end=(2,4)) = [[ 2.,  3.,  4.],
 *        [ 6.,  7.,  8.]]
 *        slice(x, begin=(None, 0), end=(None, 3), step=(-1, 2)) = [[9., 11.],
 *        [5.,  7.],
 *        [1.,  3.]]
 *
 *
 *        Defined in ../src/operator/tensor/matrix_op.cc:L482
 * \param data Source input
 * \param begin starting indices for the slice operation, supports negative indices.
 * \param end ending indices for the slice operation, supports negative indices.
 * \param step step for the slice operation, supports negative values.
 * \return new symbol
 */
inline Symbol slice(Symbol data,
                    Shape begin,
                    Shape end,
                    Shape step = Shape()) {
  return Operator("slice")
           .SetParam("begin", begin)
           .SetParam("end", end)
           .SetParam("step", step)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Slices along a given axis.
 *        Returns an array slice along a given `axis` starting from the `begin` index
 *        to the `end` index.
 *        Examples::
 *        x = [[  1.,   2.,   3.,   4.],
 *        [  5.,   6.,   7.,   8.],
 *        [  9.,  10.,  11.,  12.]]
 *        slice_axis(x, axis=0, begin=1, end=3) = [[  5.,   6.,   7.,   8.],
 *        [  9.,  10.,  11.,  12.]]
 *        slice_axis(x, axis=1, begin=0, end=2) = [[  1.,   2.],
 *        [  5.,   6.],
 *        [  9.,  10.]]
 *        slice_axis(x, axis=1, begin=-3, end=-1) = [[  2.,   3.],
 *        [  6.,   7.],
 *        [ 10.,  11.]]
 *
 *
 *        Defined in ../src/operator/tensor/matrix_op.cc:L571
 * \param data Source input
 * \param axis Axis along which to be sliced, supports negative indexes.
 * \param begin The beginning index along the axis to be sliced,  supports negative
 * \param end The ending index along the axis to be sliced,  supports negative indexes.
 * \return new symbol
 */
inline Symbol slice_axis(Symbol data,
                         int axis,
                         int begin,
                         dmlc::optional<int> end) {
  return Operator("slice_axis")
           .SetParam("axis", axis)
           .SetParam("begin", begin)
           .SetParam("end", end)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Slices a region of the array like the shape of another array.
 *        This function is similar to ``slice``, however, the `begin` are always `0`s
 *        and `end` of specific axes are inferred from the second input `shape_like`.
 *        Given the second `shape_like` input of ``shape=(d_0, d_1, ..., d_n-1)``,
 *        a ``slice_like`` operator with default empty `axes`, it performs the
 *        following operation:
 *        `` out = slice(input, begin=(0, 0, ..., 0), end=(d_0, d_1, ..., d_n-1))``.
 *        When `axes` is not empty, it is used to speficy which axes are being sliced.
 *        Given a 4-d input data, ``slice_like`` operator with ``axes=(0, 2, -1)``
 *        will perform the following operation:
 *        `` out = slice(input, begin=(0, 0, 0, 0), end=(d_0, None, d_2, d_3))``.
 *        Note that it is allowed to have first and second input with different
 *        however, you have to make sure the `axes` are specified and not exceeding the
 *        dimension limits.
 *        For example, given `input_1` with ``shape=(2,3,4,5)`` and `input_2` with
 *        ``shape=(1,2,3)``, it is not allowed to use:
 *        `` out = slice_like(a, b)`` because ndim of `input_1` is 4, and ndim of
 *        is 3.
 *        The following is allowed in this situation:
 *        `` out = slice_like(a, b, axes=(0, 2))``
 *        Example::
 *        x = [[  1.,   2.,   3.,   4.],
 *        [  5.,   6.,   7.,   8.],
 *        [  9.,  10.,  11.,  12.]]
 *        y = [[  0.,   0.,   0.],
 *        [  0.,   0.,   0.]]
 *        slice_like(x, y) = [[ 1.,  2.,  3.]
 *        [ 5.,  6.,  7.]]
 *        slice_like(x, y, axes=(0, 1)) = [[ 1.,  2.,  3.]
 *        [ 5.,  6.,  7.]]
 *        slice_like(x, y, axes=(0)) = [[ 1.,  2.,  3.,  4.]
 *        [ 5.,  6.,  7.,  8.]]
 *        slice_like(x, y, axes=(-1)) = [[  1.,   2.,   3.]
 *        [  5.,   6.,   7.]
 *        [  9.,  10.,  11.]]
 *
 *
 *        Defined in ../src/operator/tensor/matrix_op.cc:L625
 * \param data Source input
 * \param shape_like Shape like input
 * \param axes List of axes on which input data will be sliced according to the
 *        corresponding size of the second input. By default will slice on all axes.
 * \return new symbol
 */
inline Symbol slice_like(Symbol data,
                         Symbol shape_like,
                         Shape axes = Shape()) {
  return Operator("slice_like")
           .SetParam("axes", axes)
           .SetInput("data", data)
           .SetInput("shape_like", shape_like)
           .CreateSymbol();
}

/*!
 * \brief Clips (limits) the values in an array.
 *        Given an interval, values outside the interval are clipped to the interval
 *        Clipping ``x`` between `a_min` and `a_max` would be::
 *        .. math::
 *        clip(x, a_min, a_max) = \max(\min(x, a_max), a_min))
 *        Example::
 *        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
 *        clip(x,1,8) = [ 1.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  8.]
 *        The storage type of ``clip`` output depends on storage types of inputs and the
 *        parameter values:
 *        - clip(default) = default
 *        - clip(row_sparse, a_min <= 0, a_max >= 0) = row_sparse
 *        - clip(csr, a_min <= 0, a_max >= 0) = csr
 *        - clip(row_sparse, a_min < 0, a_max < 0) = default
 *        - clip(row_sparse, a_min > 0, a_max > 0) = default
 *        - clip(csr, a_min < 0, a_max < 0) = csr
 *        - clip(csr, a_min > 0, a_max > 0) = csr
 *
 *
 *        Defined in ../src/operator/tensor/matrix_op.cc:L677
 * \param data Input array.
 * \param a_min Minimum value
 * \param a_max Maximum value
 * \return new symbol
 */
inline Symbol clip(Symbol data,
                   mx_float a_min,
                   mx_float a_max) {
  return Operator("clip")
           .SetParam("a_min", a_min)
           .SetParam("a_max", a_max)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Repeats elements of an array.
 *        By default, ``repeat`` flattens the input array into 1-D and then repeats the
 *        elements::
 *        x = [[ 1, 2],
 *        [ 3, 4]]
 *        repeat(x, repeats=2) = [ 1.,  1.,  2.,  2.,  3.,  3.,  4.,  4.]
 *        The parameter ``axis`` specifies the axis along which to perform repeat::
 *        repeat(x, repeats=2, axis=1) = [[ 1.,  1.,  2.,  2.],
 *        [ 3.,  3.,  4.,  4.]]
 *        repeat(x, repeats=2, axis=0) = [[ 1.,  2.],
 *        [ 1.,  2.],
 *        [ 3.,  4.],
 *        [ 3.,  4.]]
 *        repeat(x, repeats=2, axis=-1) = [[ 1.,  1.,  2.,  2.],
 *        [ 3.,  3.,  4.,  4.]]
 *
 *
 *        Defined in ../src/operator/tensor/matrix_op.cc:L744
 * \param data Input data array
 * \param repeats The number of repetitions for each element.
 * \param axis The axis along which to repeat values. The negative numbers are
 *        interpreted counting from the backward. By default, use the flattened input
 * \return new symbol
 */
inline Symbol repeat(Symbol data,
                     int repeats,
                     dmlc::optional<int> axis = dmlc::optional<int>()) {
  return Operator("repeat")
           .SetParam("repeats", repeats)
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Repeats the whole array multiple times.
 *        If ``reps`` has length *d*, and input array has dimension of *n*. There are
 *        three cases:
 *        - **n=d**. Repeat *i*-th dimension of the input by ``reps[i]`` times::
 *        x = [[1, 2],
 *        [3, 4]]
 *        tile(x, reps=(2,3)) = [[ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.],
 *        [ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.]]
 *        - **n>d**. ``reps`` is promoted to length *n* by pre-pending 1's to it. Thus for
 *        an input shape ``(2,3)``, ``repos=(2,)`` is treated as ``(1,2)``::
 *        tile(x, reps=(2,)) = [[ 1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.]]
 *        - **n<d**. The input is promoted to be d-dimensional by prepending new axes. So
 *        shape ``(2,2)`` array is promoted to ``(1,2,2)`` for 3-D replication::
 *        tile(x, reps=(2,2,3)) = [[[ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.],
 *        [ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.]],
 *        [[ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.],
 *        [ 1.,  2.,  1.,  2.,  1.,  2.],
 *        [ 3.,  4.,  3.,  4.,  3.,  4.]]]
 *
 *
 *        Defined in ../src/operator/tensor/matrix_op.cc:L796
 * \param data Input data array
 * \param reps The number of times for repeating the tensor a. Each dim size of reps must
 *        be a positive integer. If reps has length d, the result will have dimension of
 *        max(d, a.ndim); If a.ndim < d, a is promoted to be d-dimensional by prepending
 * \return new symbol
 */
inline Symbol tile(Symbol data,
                   Shape reps) {
  return Operator("tile")
           .SetParam("reps", reps)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Reverses the order of elements along given axis while preserving array shape.
 *        Note: reverse and flip are equivalent. We use reverse in the following examples.
 *        Examples::
 *        x = [[ 0.,  1.,  2.,  3.,  4.],
 *        [ 5.,  6.,  7.,  8.,  9.]]
 *        reverse(x, axis=0) = [[ 5.,  6.,  7.,  8.,  9.],
 *        [ 0.,  1.,  2.,  3.,  4.]]
 *        reverse(x, axis=1) = [[ 4.,  3.,  2.,  1.,  0.],
 *        [ 9.,  8.,  7.,  6.,  5.]]
 *
 *
 *        Defined in ../src/operator/tensor/matrix_op.cc:L832
 * \param data Input data array
 * \param axis The axis which to reverse elements.
 * \return new symbol
 */
inline Symbol reverse(Symbol data,
                      Shape axis) {
  return Operator("reverse")
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Join a sequence of arrays along a new axis.
 *        The axis parameter specifies the index of the new axis in the dimensions of the
 *        result. For example, if axis=0 it will be the first dimension and if axis=-1 it
 *        will be the last dimension.
 *        Examples::
 *        x = [1, 2]
 *        y = [3, 4]
 *        stack(x, y) = [[1, 2],
 *        [3, 4]]
 *        stack(x, y, axis=1) = [[1, 3],
 *        [2, 4]]
 *
 * \param data List of arrays to stack
 * \param num_args Number of inputs to be stacked.
 * \param axis The axis in the result array along which the input arrays are stacked.
 * \return new symbol
 */
inline Symbol stack(const std::vector<Symbol>& data,
                    int num_args,
                    int axis = 0) {
  return Operator("stack")
           .SetParam("num_args", num_args)
           .SetParam("axis", axis)
(data)
           .CreateSymbol();
}

/*!
 * \brief Remove single-dimensional entries from the shape of an array.
 *        Same behavior of defining the output tensor shape as numpy.squeeze for the most
 *        See the following note for exception.
 *        Examples::
 *        data = [[[0], [1], [2]]]
 *        squeeze(data) = [0, 1, 2]
 *        squeeze(data, axis=0) = [[0], [1], [2]]
 *        squeeze(data, axis=2) = [[0, 1, 2]]
 *        squeeze(data, axis=(0, 2)) = [0, 1, 2]
 *        .. Note::
 *        The output of this operator will keep at least one dimension not removed. For
 *        squeeze([[[4]]]) = [4], while in numpy.squeeze, the output will become a scalar.
 *
 * \param data data to squeeze
 * \param axis Selects a subset of the single-dimensional entries in the shape. If an
 * \return new symbol
 */
inline Symbol squeeze(Symbol data,
                      dmlc::optional<Shape> axis = dmlc::optional<Shape>()) {
  return Operator("squeeze")
           .SetParam("axis", axis)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Rearranges(permutes) data from depth into blocks of spatial data.
 *        Similar to ONNX DepthToSpace operator:
 *        https://github.com/onnx/onnx/blob/master/docs/Operators.md#DepthToSpace.
 *        The output is a new tensor where the values from depth dimension are moved in
 *        to height and width dimension. The reverse of this operation is
 *        .. math::
 *        \begin{gather*}
 *        x \prime = reshape(x, [N, block\_size, block\_size, C / (block\_size ^ 2), H *
 *        x \prime \prime = transpose(x \prime, [0, 3, 4, 1, 5, 2]) \\
 *        y = reshape(x \prime \prime, [N, C / (block\_size ^ 2), H * block\_size, W *
 *        \end{gather*}
 *        where :math:`x` is an input tensor with default layout as :math:`[N, C, H, W]`:
 *        and :math:`y` is the output tensor of layout :math:`[N, C / (block\_size ^ 2),
 *        Example::
 *        x = [[[[0, 1, 2],
 *        [3, 4, 5]],
 *        [[6, 7, 8],
 *        [9, 10, 11]],
 *        [[12, 13, 14],
 *        [15, 16, 17]],
 *        [[18, 19, 20],
 *        [21, 22, 23]]]]
 *        depth_to_space(x, 2) = [[[[0, 6, 1, 7, 2, 8],
 *        [12, 18, 13, 19, 14, 20],
 *        [3, 9, 4, 10, 5, 11],
 *        [15, 21, 16, 22, 17, 23]]]]
 *
 *
 *        Defined in ../src/operator/tensor/matrix_op.cc:L972
 * \param data Input ndarray
 * \param block_size Blocks of [block_size. block_size] are moved
 * \return new symbol
 */
inline Symbol depth_to_space(Symbol data,
                             int block_size) {
  return Operator("depth_to_space")
           .SetParam("block_size", block_size)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Rearranges(permutes) blocks of spatial data into depth.
 *        Similar to ONNX SpaceToDepth operator:
 *        https://github.com/onnx/onnx/blob/master/docs/Operators.md#SpaceToDepth
 *        The output is a new tensor where the values from height and width dimension are
 *        moved to the depth dimension. The reverse of this operation is
 *        .. math::
 *        \begin{gather*}
 *        x \prime = reshape(x, [N, C, H / block\_size, block\_size, W / block\_size,
 *        x \prime \prime = transpose(x \prime, [0, 3, 5, 1, 2, 4]) \\
 *        y = reshape(x \prime \prime, [N, C * (block\_size ^ 2), H / block\_size, W /
 *        \end{gather*}
 *        where :math:`x` is an input tensor with default layout as :math:`[N, C, H, W]`:
 *        and :math:`y` is the output tensor of layout :math:`[N, C * (block\_size ^ 2),
 *        Example::
 *        x = [[[[0, 6, 1, 7, 2, 8],
 *        [12, 18, 13, 19, 14, 20],
 *        [3, 9, 4, 10, 5, 11],
 *        [15, 21, 16, 22, 17, 23]]]]
 *        space_to_depth(x, 2) = [[[[0, 1, 2],
 *        [3, 4, 5]],
 *        [[6, 7, 8],
 *        [9, 10, 11]],
 *        [[12, 13, 14],
 *        [15, 16, 17]],
 *        [[18, 19, 20],
 *        [21, 22, 23]]]]
 *
 *
 *        Defined in ../src/operator/tensor/matrix_op.cc:L1019
 * \param data Input ndarray
 * \param block_size Blocks of [block_size. block_size] are moved
 * \return new symbol
 */
inline Symbol space_to_depth(Symbol data,
                             int block_size) {
  return Operator("space_to_depth")
           .SetParam("block_size", block_size)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns the indices of the top *k* elements in an input array along the given
 *        axis (by default).
 *        If ret_type is set to 'value' returns the value of top *k* elements (instead of
 *        In case of ret_type = 'both', both value and index would be returned.
 *        The returned elements will be sorted.
 *
 *        Examples::
 *
 *        x = [[ 0.3,  0.2,  0.4],
 *        [ 0.1,  0.3,  0.2]]
 *
 *        // returns an index of the largest element on last axis
 *        topk(x) = [[ 2.],
 *        [ 1.]]
 *
 *        // returns the value of top-2 largest elements on last axis
 *        topk(x, ret_typ='value', k=2) = [[ 0.4,  0.3],
 *        [ 0.3,  0.2]]
 *
 *        // returns the value of top-2 smallest elements on last axis
 *        topk(x, ret_typ='value', k=2, is_ascend=1) = [[ 0.2 ,  0.3],
 *        [ 0.1 ,  0.2]]
 *
 *        // returns the value of top-2 largest elements on axis 0
 *        topk(x, axis=0, ret_typ='value', k=2) = [[ 0.3,  0.3,  0.4],
 *        [ 0.1,  0.2,  0.2]]
 *
 *        // flattens and then returns list of both values and indices
 *        topk(x, ret_typ='both', k=2) = [[[ 0.4,  0.3], [ 0.3,  0.2]] ,  [[ 2.,  0.], [
 *
 *
 *
 *        Defined in ../src/operator/tensor/ordering_op.cc:L68
 * \param data The input array
 * \param axis Axis along which to choose the top k indices. If not given, the flattened
 * \param k Number of top elements to select, should be always smaller than or equal to
 * \param ret_typ The return type.
 *        "value" means to return the top k values, "indices" means to return the indices
 *        of the top k values, "mask" means to return a mask array containing 0 and 1. 1
 *        means the top k values. "both" means to return a list of both values and
 * \param is_ascend Whether to choose k largest or k smallest elements. Top K largest
 * \param dtype DType of the output indices when ret_typ is "indices" or "both". An error
 * \return new symbol
 */
inline Symbol topk(Symbol data,
                   dmlc::optional<int> axis = dmlc::optional<int>(-1),
                   int k = 1,
                   TopkRetTyp ret_typ = TopkRetTyp::kIndices,
                   bool is_ascend = false,
                   TopkDtype dtype = TopkDtype::kFloat32) {
  static const char *TopkRetTypValues[] = {
    "both",
    "indices",
    "mask",
    "value"
  };
  static const char *TopkDtypeValues[] = {
    "float16",
    "float32",
    "float64",
    "int32",
    "int64",
    "uint8"
  };
  return Operator("topk")
           .SetParam("axis", axis)
           .SetParam("k", k)
           .SetParam("ret_typ", TopkRetTypValues[int(ret_typ)])
           .SetParam("is_ascend", is_ascend)
           .SetParam("dtype", TopkDtypeValues[int(dtype)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns a sorted copy of an input array along the given axis.
 *
 *        Examples::
 *
 *        x = [[ 1, 4],
 *        [ 3, 1]]
 *
 *        // sorts along the last axis
 *        sort(x) = [[ 1.,  4.],
 *        [ 1.,  3.]]
 *
 *        // flattens and then sorts
 *        sort(x, axis=None) = [ 1.,  1.,  3.,  4.]
 *
 *        // sorts along the first axis
 *        sort(x, axis=0) = [[ 1.,  1.],
 *        [ 3.,  4.]]
 *
 *        // in a descend order
 *        sort(x, is_ascend=0) = [[ 4.,  1.],
 *        [ 3.,  1.]]
 *
 *
 *
 *        Defined in ../src/operator/tensor/ordering_op.cc:L133
 * \param data The input array
 * \param axis Axis along which to choose sort the input tensor. If not given, the
 * \param is_ascend Whether to sort in ascending or descending order.
 * \return new symbol
 */
inline Symbol sort(Symbol data,
                   dmlc::optional<int> axis = dmlc::optional<int>(-1),
                   bool is_ascend = true) {
  return Operator("sort")
           .SetParam("axis", axis)
           .SetParam("is_ascend", is_ascend)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Returns the indices that would sort an input array along the given axis.
 *
 *        This function performs sorting along the given axis and returns an array of
 *        as an input array that index data in sorted order.
 *
 *        Examples::
 *
 *        x = [[ 0.3,  0.2,  0.4],
 *        [ 0.1,  0.3,  0.2]]
 *
 *        // sort along axis -1
 *        argsort(x) = [[ 1.,  0.,  2.],
 *        [ 0.,  2.,  1.]]
 *
 *        // sort along axis 0
 *        argsort(x, axis=0) = [[ 1.,  0.,  1.]
 *        [ 0.,  1.,  0.]]
 *
 *        // flatten and then sort
 *        argsort(x, axis=None) = [ 3.,  1.,  5.,  0.,  4.,  2.]
 *
 *
 *        Defined in ../src/operator/tensor/ordering_op.cc:L185
 * \param data The input array
 * \param axis Axis along which to sort the input tensor. If not given, the flattened
 * \param is_ascend Whether to sort in ascending or descending order.
 * \param dtype DType of the output indices. It is only valid when ret_typ is "indices"
 *        or "both". An error will be raised if the selected data type cannot precisely
 * \return new symbol
 */
inline Symbol argsort(Symbol data,
                      dmlc::optional<int> axis = dmlc::optional<int>(-1),
                      bool is_ascend = true,
                      ArgsortDtype dtype = ArgsortDtype::kFloat32) {
  static const char *ArgsortDtypeValues[] = {
    "float16",
    "float32",
    "float64",
    "int32",
    "int64",
    "uint8"
  };
  return Operator("argsort")
           .SetParam("axis", axis)
           .SetParam("is_ascend", is_ascend)
           .SetParam("dtype", ArgsortDtypeValues[int(dtype)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Applies bilinear sampling to input feature map.
 *
 *        Bilinear Sampling is the key of  [NIPS2015] \"Spatial Transformer Networks\".
 *        except that the operator has the backward pass.
 *
 *        Given :math:`data` and :math:`grid`, then the output is computed by
 *
 *        .. math::
 *        x_{src} = grid[batch, 0, y_{dst}, x_{dst}] \\
 *        y_{src} = grid[batch, 1, y_{dst}, x_{dst}] \\
 *        output[batch, channel, y_{dst}, x_{dst}] = G(data[batch, channel, y_{src},
 *
 *        :math:`x_{dst}`, :math:`y_{dst}` enumerate all spatial locations in
 *        The out-boundary points will be padded with zeros.The shape of the output will
 *
 *        The operator assumes that :math:`data` has 'NCHW' layout and :math:`grid` has
 *
 *        BilinearSampler often cooperates with GridGenerator which generates sampling
 *        GridGenerator supports two kinds of transformation: ``affine`` and ``warp``.
 *        If users want to design a CustomOp to manipulate :math:`grid`, please firstly
 *
 *        Example 1::
 *
 *        ## Zoom out data two times
 *        data = array([[[[1, 4, 3, 6],
 *        [1, 8, 8, 9],
 *        [0, 4, 1, 5],
 *        [1, 0, 1, 3]]]])
 *
 *        affine_matrix = array([[2, 0, 0],
 *        [0, 2, 0]])
 *
 *        affine_matrix = reshape(affine_matrix, shape=(1, 6))
 *
 *        grid = GridGenerator(data=affine_matrix, transform_type='affine',
 *
 *        out = BilinearSampler(data, grid)
 *
 *        out
 *        [[[[ 0,   0,     0,   0],
 *        [ 0,   3.5,   6.5, 0],
 *        [ 0,   1.25,  2.5, 0],
 *        [ 0,   0,     0,   0]]]
 *
 *
 *        Example 2::
 *
 *        ## shift data horizontally by -1 pixel
 *
 *        data = array([[[[1, 4, 3, 6],
 *        [1, 8, 8, 9],
 *        [0, 4, 1, 5],
 *        [1, 0, 1, 3]]]])
 *
 *        warp_maxtrix = array([[[[1, 1, 1, 1],
 *        [1, 1, 1, 1],
 *        [1, 1, 1, 1],
 *        [1, 1, 1, 1]],
 *        [[0, 0, 0, 0],
 *        [0, 0, 0, 0],
 *        [0, 0, 0, 0],
 *        [0, 0, 0, 0]]]])
 *
 *        grid = GridGenerator(data=warp_matrix, transform_type='warp')
 *        out = BilinearSampler(data, grid)
 *
 *        out
 *        [[[[ 4,  3,  6,  0],
 *        [ 8,  8,  9,  0],
 *        [ 4,  1,  5,  0],
 *        [ 0,  1,  3,  0]]]
 *
 *
 *        Defined in ../src/operator/bilinear_sampler.cc:L256
 * \param data Input data to the BilinearsamplerOp.
 * \param grid Input grid to the BilinearsamplerOp.grid has two channels: x_src, y_src
 * \param cudnn_off whether to turn cudnn off
 * \return new symbol
 */
inline Symbol BilinearSampler(Symbol data,
                              Symbol grid,
                              dmlc::optional<bool> cudnn_off = dmlc::optional<bool>()) {
  return Operator("BilinearSampler")
           .SetParam("cudnn_off", cudnn_off)
           .SetInput("data", data)
           .SetInput("grid", grid)
           .CreateSymbol();
}

/*!
 * \brief This operator is DEPRECATED. Apply convolution to input then add a bias.
 * \param data Input data to the ConvolutionV1Op.
 * \param weight Weight matrix.
 * \param bias Bias parameter.
 * \param kernel convolution kernel size: (h, w) or (d, h, w)
 * \param num_filter convolution filter(channel) number
 * \param stride convolution stride: (h, w) or (d, h, w)
 * \param dilate convolution dilate: (h, w) or (d, h, w)
 * \param pad pad for convolution: (h, w) or (d, h, w)
 * \param num_group Number of group partitions. Equivalent to slicing input into num_group
 *        partitions, apply convolution on each, then concatenate the results
 * \param workspace Maximum temporary workspace allowed for convolution (MB).This
 *        parameter determines the effective batch size of the convolution kernel, which
 *        may be smaller than the given batch size. Also, the workspace will be
 * \param no_bias Whether to disable bias parameter.
 * \param cudnn_tune Whether to pick convolution algo by running performance test.
 *        Leads to higher startup time but may give faster speed. Options are:
 *        'off': no tuning
 *        'limited_workspace': run test and pick the fastest algorithm that doesn't
 *        'fastest': pick the fastest algorithm and ignore workspace limit.
 *        If set to None (default), behavior is determined by environment
 *        variable MXNET_CUDNN_AUTOTUNE_DEFAULT: 0 for off,
 *        1 for limited workspace (default), 2 for fastest.
 * \param cudnn_off Turn off cudnn for this layer.
 * \param layout Set layout for input, output and weight. Empty for
 *        default layout: NCHW for 2d and NCDHW for 3d.
 * \return new symbol
 */
inline Symbol Convolution_v1(Symbol data,
                             Symbol weight,
                             Symbol bias,
                             Shape kernel,
                             uint32_t num_filter,
                             Shape stride = Shape(),
                             Shape dilate = Shape(),
                             Shape pad = Shape(),
                             uint32_t num_group = 1,
                             uint64_t workspace = 1024,
                             bool no_bias = false,
                             Convolution_v1CudnnTune cudnn_tune = Convolution_v1CudnnTune::kNone,
                             bool cudnn_off = false,
                             Convolution_v1Layout layout = Convolution_v1Layout::kNone) {
  static const char *Convolution_v1CudnnTuneValues[] = {
    "None",
    "fastest",
    "limited_workspace",
    "off"
  };
  static const char *Convolution_v1LayoutValues[] = {
    "None",
    "NCDHW",
    "NCHW",
    "NDHWC",
    "NHWC"
  };
  return Operator("Convolution_v1")
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("dilate", dilate)
           .SetParam("pad", pad)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
           .SetParam("cudnn_tune", Convolution_v1CudnnTuneValues[int(cudnn_tune)])
           .SetParam("cudnn_off", cudnn_off)
           .SetParam("layout", Convolution_v1LayoutValues[int(layout)])
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol();
}

/*!
 * \brief Applies correlation to inputs.
 *
 *        The correlation layer performs multiplicative patch comparisons between two
 *
 *        Given two multi-channel feature maps :math:`f_{1}, f_{2}`, with :math:`w`,
 *        the correlation layer lets the network compare each patch from :math:`f_{1}`
 *
 *        For now we consider only a single comparison of two patches. The 'correlation'
 *        :math:`x_{2}` in the second map is then defined as:
 *
 *        .. math::
 *
 *        c(x_{1}, x_{2}) = \sum_{o \in [-k,k] \times [-k,k]} <f_{1}(x_{1} + o),
 *
 *        for a square patch of size :math:`K:=2k+1`.
 *
 *        Note that the equation above is identical to one step of a convolution in
 *        neural networks, but instead of convolving data with a filter, it convolves
 *        data. For this reason, it has no training weights.
 *
 *        Computing :math:`c(x_{1}, x_{2})` involves :math:`c * K^{2}` multiplications.
 *
 *        Given a maximum displacement :math:`d`, for each location :math:`x_{1}` it
 *        computes correlations :math:`c(x_{1}, x_{2})` only in a neighborhood of size
 *        by limiting the range of :math:`x_{2}`. We use strides :math:`s_{1}, s_{2}`, to
 *        quantize :math:`x_{1}` globally and to quantize :math:`x_{2}` within the
 *        centered around :math:`x_{1}`.
 *
 *        The final output is defined by the following expression:
 *
 *        .. math::
 *        out[n, q, i, j] = c(x_{i, j}, x_{q})
 *
 *        where :math:`i` and :math:`j` enumerate spatial locations in :math:`f_{1}`, and
 *
 *
 *        Defined in ../src/operator/correlation.cc:L198
 * \param data1 Input data1 to the correlation.
 * \param data2 Input data2 to the correlation.
 * \param kernel_size kernel size for Correlation must be an odd number
 * \param max_displacement Max displacement of Correlation
 * \param stride1 stride1 quantize data1 globally
 * \param stride2 stride2 quantize data2 within the neighborhood centered around data1
 * \param pad_size pad for Correlation
 * \param is_multiply operation type is either multiplication or subduction
 * \return new symbol
 */
inline Symbol Correlation(Symbol data1,
                          Symbol data2,
                          uint32_t kernel_size = 1,
                          uint32_t max_displacement = 1,
                          uint32_t stride1 = 1,
                          uint32_t stride2 = 1,
                          uint32_t pad_size = 0,
                          bool is_multiply = true) {
  return Operator("Correlation")
           .SetParam("kernel_size", kernel_size)
           .SetParam("max_displacement", max_displacement)
           .SetParam("stride1", stride1)
           .SetParam("stride2", stride2)
           .SetParam("pad_size", pad_size)
           .SetParam("is_multiply", is_multiply)
           .SetInput("data1", data1)
           .SetInput("data2", data2)
           .CreateSymbol();
}

/*!
 * \brief
 *
 *        .. note:: `Crop` is deprecated. Use `slice` instead.
 *
 *        Crop the 2nd and 3rd dim of input data, with the corresponding size of h_w or
 *        with width and height of the second input symbol, i.e., with one input, we need
 *        specify the crop height and width, otherwise the second input symbol's size
 *
 *
 *        Defined in ../src/operator/crop.cc:L50
 * \param data Tensor or List of Tensors, the second input will be used as crop_like
 * \param num_args Number of inputs for crop, if equals one, then we will use the h_wfor
 *        crop height and width, else if equals two, then we will use the heightand width
 * \param offset crop offset coordinate: (y, x)
 * \param h_w crop height and width: (h, w)
 * \param center_crop If set to true, then it will use be the center_crop,or it will crop
 * \return new symbol
 */
inline Symbol Crop(const std::vector<Symbol>& data,
                   int num_args,
                   Shape offset = Shape(0,0),
                   Shape h_w = Shape(0,0),
                   bool center_crop = false) {
  return Operator("Crop")
           .SetParam("num_args", num_args)
           .SetParam("offset", offset)
           .SetParam("h_w", h_w)
           .SetParam("center_crop", center_crop)
(data)
           .CreateSymbol();
}

/*!
 * \brief Generates 2D sampling grid for bilinear sampling.
 * \param data Input data to the function.
 * \param transform_type The type of transformation. For `affine`, input data should be
 *        an affine matrix of size (batch, 6). For `warp`, input data should be an
 * \param target_shape Specifies the output shape (H, W). This is required if
 *        transformation type is `affine`. If transformation type is `warp`, this
 * \return new symbol
 */
inline Symbol GridGenerator(Symbol data,
                            GridGeneratorTransformType transform_type,
                            Shape target_shape = Shape(0,0)) {
  static const char *GridGeneratorTransformTypeValues[] = {
    "affine",
    "warp"
  };
  return Operator("GridGenerator")
           .SetParam("transform_type", GridGeneratorTransformTypeValues[int(transform_type)])
           .SetParam("target_shape", target_shape)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Applies instance normalization to the n-dimensional input array.
 *
 *        This operator takes an n-dimensional input array where (n>2) and normalizes
 *        the input using the following formula:
 *
 *        .. math::
 *
 *        out = \frac{x - mean[data]}{ \sqrt{Var[data]} + \epsilon} * gamma + beta
 *
 *        This layer is similar to batch normalization layer (`BatchNorm`)
 *        with two differences: first, the normalization is
 *        carried out per example (instance), not over a batch. Second, the
 *        same normalization is applied both at test and train time. This
 *        operation is also known as `contrast normalization`.
 *
 *        If the input data is of shape [batch, channel, spacial_dim1, spacial_dim2, ...],
 *        `gamma` and `beta` parameters must be vectors of shape [channel].
 *
 *        This implementation is based on this paper [1]_
 *
 *        .. [1] Instance Normalization: The Missing Ingredient for Fast Stylization,
 *        D. Ulyanov, A. Vedaldi, V. Lempitsky, 2016 (arXiv:1607.08022v2).
 *
 *        Examples::
 *
 *        // Input of shape (2,1,2)
 *        x = [[[ 1.1,  2.2]],
 *        [[ 3.3,  4.4]]]
 *
 *        // gamma parameter of length 1
 *        gamma = [1.5]
 *
 *        // beta parameter of length 1
 *        beta = [0.5]
 *
 *        // Instance normalization is calculated with the above formula
 *        InstanceNorm(x,gamma,beta) = [[[-0.997527  ,  1.99752665]],
 *        [[-0.99752653,  1.99752724]]]
 *
 *
 *
 *        Defined in ../src/operator/instance_norm.cc:L95
 * \param data An n-dimensional input array (n > 2) of the form [batch, channel,
 * \param gamma A vector of length 'channel', which multiplies the normalized input.
 * \param beta A vector of length 'channel', which is added to the product of the
 * \param eps An `epsilon` parameter to prevent division by 0.
 * \return new symbol
 */
inline Symbol InstanceNorm(Symbol data,
                           Symbol gamma,
                           Symbol beta,
                           mx_float eps = 0.00100000005) {
  return Operator("InstanceNorm")
           .SetParam("eps", eps)
           .SetInput("data", data)
           .SetInput("gamma", gamma)
           .SetInput("beta", beta)
           .CreateSymbol();
}

/*!
 * \brief Normalize the input array using the L2 norm.
 *
 *        For 1-D NDArray, it computes::
 *
 *        out = data / sqrt(sum(data ** 2) + eps)
 *
 *        For N-D NDArray, if the input array has shape (N, N, ..., N),
 *
 *        with ``mode`` = ``instance``, it normalizes each instance in the
 *        array by its L2 norm.::
 *
 *        for i in 0...N
 *        out[i,:,:,...,:] = data[i,:,:,...,:] / sqrt(sum(data[i,:,:,...,:] ** 2) + eps)
 *
 *        with ``mode`` = ``channel``, it normalizes each channel in the array by its L2
 *
 *        for i in 0...N
 *        out[:,i,:,...,:] = data[:,i,:,...,:] / sqrt(sum(data[:,i,:,...,:] ** 2) + eps)
 *
 *        with ``mode`` = ``spatial``, it normalizes the cross channel norm for each
 *        in the array by its L2 norm.::
 *
 *        for dim in 2...N
 *        for i in 0...N
 *        out[.....,i,...] = take(out, indices=i, axis=dim) / sqrt(sum(take(out,
 *        -dim-
 *
 *        Example::
 *
 *        x = [[[1,2],
 *        [3,4]],
 *        [[2,2],
 *        [5,6]]]
 *
 *        L2Normalization(x, mode='instance')
 *        =[[[ 0.18257418  0.36514837]
 *        [ 0.54772252  0.73029673]]
 *        [[ 0.24077171  0.24077171]
 *        [ 0.60192931  0.72231513]]]
 *
 *        L2Normalization(x, mode='channel')
 *        =[[[ 0.31622776  0.44721359]
 *        [ 0.94868326  0.89442718]]
 *        [[ 0.37139067  0.31622776]
 *        [ 0.92847669  0.94868326]]]
 *
 *        L2Normalization(x, mode='spatial')
 *        =[[[ 0.44721359  0.89442718]
 *        [ 0.60000002  0.80000001]]
 *        [[ 0.70710677  0.70710677]
 *        [ 0.6401844   0.76822126]]]
 *
 *
 *
 *        Defined in ../src/operator/l2_normalization.cc:L196
 * \param data Input array to normalize.
 * \param eps A small constant for numerical stability.
 * \param mode Specify the dimension along which to compute L2 norm.
 * \return new symbol
 */
inline Symbol L2Normalization(Symbol data,
                              mx_float eps = 1.00000001e-10,
                              L2NormalizationMode mode = L2NormalizationMode::kInstance) {
  static const char *L2NormalizationModeValues[] = {
    "channel",
    "instance",
    "spatial"
  };
  return Operator("L2Normalization")
           .SetParam("eps", eps)
           .SetParam("mode", L2NormalizationModeValues[int(mode)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Make your own loss function in network construction.
 *
 *        This operator accepts a customized loss function symbol as a terminal loss and
 *        the symbol should be an operator with no backward dependency.
 *        The output of this function is the gradient of loss with respect to the input
 *
 *        For example, if you are a making a cross entropy loss function. Assume ``out``
 *        predicted output and ``label`` is the true label, then the cross entropy can be
 *
 *        cross_entropy = label * log(out) + (1 - label) * log(1 - out)
 *        loss = MakeLoss(cross_entropy)
 *
 *        We will need to use ``MakeLoss`` when we are creating our own loss function or
 *        combine multiple loss functions. Also we may want to stop some variables'
 *        from backpropagation. See more detail in ``BlockGrad`` or ``stop_gradient``.
 *
 *        In addition, we can give a scale to the loss by setting ``grad_scale``,
 *        so that the gradient of the loss will be rescaled in the backpropagation.
 *
 *        .. note:: This operator should be used as a Symbol instead of NDArray.
 *
 *
 *
 *        Defined in ../src/operator/make_loss.cc:L71
 * \param data Input array.
 * \param grad_scale Gradient scale as a supplement to unary and binary operators
 * \param valid_thresh clip each element in the array to 0 when it is less than
 * \param normalization If this is set to null, the output gradient will not be
 *        normalized. If this is set to batch, the output gradient will be divided by the
 *        batch size. If this is set to valid, the output gradient will be divided by the
 * \return new symbol
 */
inline Symbol MakeLoss(Symbol data,
                       mx_float grad_scale = 1,
                       mx_float valid_thresh = 0,
                       MakeLossNormalization normalization = MakeLossNormalization::kNull) {
  static const char *MakeLossNormalizationValues[] = {
    "batch",
    "null",
    "valid"
  };
  return Operator("MakeLoss")
           .SetParam("grad_scale", grad_scale)
           .SetParam("valid_thresh", valid_thresh)
           .SetParam("normalization", MakeLossNormalizationValues[int(normalization)])
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief This operator is DEPRECATED.
 *        Perform pooling on the input.
 *
 *        The shapes for 2-D pooling is
 *
 *        - **data**: *(batch_size, channel, height, width)*
 *        - **out**: *(batch_size, num_filter, out_height, out_width)*, with::
 *
 *        out_height = f(height, kernel[0], pad[0], stride[0])
 *        out_width = f(width, kernel[1], pad[1], stride[1])
 *
 *        The definition of *f* depends on ``pooling_convention``, which has two options:
 *
 *        - **valid** (default)::
 *
 *        f(x, k, p, s) = floor((x+2*p-k)/s)+1
 *
 *        - **full**, which is compatible with Caffe::
 *
 *        f(x, k, p, s) = ceil((x+2*p-k)/s)+1
 *
 *        But ``global_pool`` is set to be true, then do a global pooling, namely reset
 *        ``kernel=(height, width)``.
 *
 *        Three pooling options are supported by ``pool_type``:
 *
 *        - **avg**: average pooling
 *        - **max**: max pooling
 *        - **sum**: sum pooling
 *
 *        1-D pooling is special case of 2-D pooling with *weight=1* and
 *        *kernel[1]=1*.
 *
 *        For 3-D pooling, an additional *depth* dimension is added before
 *        *height*. Namely the input data will have shape *(batch_size, channel, depth,
 *        height, width)*.
 *
 *
 *
 *        Defined in ../src/operator/pooling_v1.cc:L104
 * \param data Input data to the pooling operator.
 * \param kernel pooling kernel size: (y, x) or (d, y, x)
 * \param pool_type Pooling type to be applied.
 * \param global_pool Ignore kernel size, do global pooling based on current input
 * \param pooling_convention Pooling convention to be applied.
 * \param stride stride: for pooling (y, x) or (d, y, x)
 * \param pad pad for pooling: (y, x) or (d, y, x)
 * \return new symbol
 */
inline Symbol Pooling_v1(Symbol data,
                         Shape kernel = Shape(),
                         Pooling_v1PoolType pool_type = Pooling_v1PoolType::kMax,
                         bool global_pool = false,
                         Pooling_v1PoolingConvention pooling_convention = Pooling_v1PoolingConvention::kValid,
                         Shape stride = Shape(),
                         Shape pad = Shape()) {
  static const char *Pooling_v1PoolTypeValues[] = {
    "avg",
    "max",
    "sum"
  };
  static const char *Pooling_v1PoolingConventionValues[] = {
    "full",
    "valid"
  };
  return Operator("Pooling_v1")
           .SetParam("kernel", kernel)
           .SetParam("pool_type", Pooling_v1PoolTypeValues[int(pool_type)])
           .SetParam("global_pool", global_pool)
           .SetParam("pooling_convention", Pooling_v1PoolingConventionValues[int(pooling_convention)])
           .SetParam("stride", stride)
           .SetParam("pad", pad)
           .SetInput("data", data)
           .CreateSymbol();
}

/*!
 * \brief Takes the last element of a sequence.
 *
 *        This function takes an n-dimensional input array of the form
 *        [max_sequence_length, batch_size, other_feature_dims] and returns a
 *        of the form [batch_size, other_feature_dims].
 *
 *        Parameter `sequence_length` is used to handle variable-length sequences.
 *        an input array of positive ints of dimension [batch_size]. To use this
 *        set `use_sequence_length` to `True`, otherwise each example in the batch is
 *        to have the max sequence length.
 *
 *        .. note:: Alternatively, you can also use `take` operator.
 *
 *        Example::
 *
 *        x = [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.],
 *        [  7.,   8.,   9.]],
 *
 *        [[ 10.,   11.,   12.],
 *        [ 13.,   14.,   15.],
 *        [ 16.,   17.,   18.]],
 *
 *        [[  19.,   20.,   21.],
 *        [  22.,   23.,   24.],
 *        [  25.,   26.,   27.]]]
 *
 *        // returns last sequence when sequence_length parameter is not used
 *        SequenceLast(x) = [[  19.,   20.,   21.],
 *        [  22.,   23.,   24.],
 *        [  25.,   26.,   27.]]
 *
 *        // sequence_length is used
 *        SequenceLast(x, sequence_length=[1,1,1], use_sequence_length=True) =
 *        [[  1.,   2.,   3.],
 *        [  4.,   5.,   6.],
 *        [  7.,   8.,   9.]]
 *
 *        // sequence_length is used
 *        SequenceLast(x, sequence_length=[1,2,3], use_sequence_length=True) =
 *        [[  1.,    2.,   3.],
 *        [  13.,  14.,  15.],
 *        [  25.,  26.,  27.]]
 *
 *
 *
 *        Defined in ../src/operator/sequence_last.cc:L106
 * \param data n-dimensional input array of the form [max_sequence_length, batch_size,
 * \param sequence_length vector of sequence lengths of the form [batch_size]
 * \param use_sequence_length If set to true, this layer takes in an extra input
 * \param axis The sequence axis. Only values of 0 and 1 are currently supported.
 * \return new symbol
 */
inline Symbol SequenceLast(Symbol data,
                           Symbol sequence_length,
                           bool use_sequence_length = false,
                           int axis = 0) {
  return Operator("SequenceLast")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetParam("axis", axis)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol();
}

/*!
 * \brief Reverses the elements of each sequence.
 *
 *        This function takes an n-dimensional input array of the form
 *        and returns an array of the same shape.
 *
 *        Parameter `sequence_length` is used to handle variable-length sequences.
 *        `sequence_length` should be an input array of positive ints of dimension
 *        To use this parameter, set `use_sequence_length` to `True`,
 *        otherwise each example in the batch is assumed to have the max sequence length.
 *
 *        Example::
 *
 *        x = [[[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]]]
 *
 *        // Batch 1
 *        B1 = [[  1.,   2.,   3.],
 *        [  7.,   8.,   9.],
 *        [ 13.,  14.,  15.]]
 *
 *        // Batch 2
 *        B2 = [[  4.,   5.,   6.],
 *        [ 10.,  11.,  12.],
 *        [ 16.,  17.,  18.]]
 *
 *        // returns reverse sequence when sequence_length parameter is not used
 *        SequenceReverse(x) = [[[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]],
 *
 *        [[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]]]
 *
 *        // sequence_length [2,2] means 2 rows of
 *        // both batch B1 and B2 will be reversed.
 *        SequenceReverse(x, sequence_length=[2,2], use_sequence_length=True) =
 *        [[[  7.,   8.,   9.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[  1.,   2.,   3.],
 *        [  4.,   5.,   6.]],
 *
 *        [[ 13.,  14.,   15.],
 *        [ 16.,  17.,   18.]]]
 *
 *        // sequence_length [2,3] means 2 of batch B2 and 3 of batch B3
 *        // will be reversed.
 *        SequenceReverse(x, sequence_length=[2,3], use_sequence_length=True) =
 *        [[[  7.,   8.,   9.],
 *        [ 16.,  17.,  18.]],
 *
 *        [[  1.,   2.,   3.],
 *        [ 10.,  11.,  12.]],
 *
 *        [[ 13.,  14,   15.],
 *        [  4.,   5.,   6.]]]
 *
 *
 *
 *        Defined in ../src/operator/sequence_reverse.cc:L122
 * \param data n-dimensional input array of the form [max_sequence_length, batch_size,
 * \param sequence_length vector of sequence lengths of the form [batch_size]
 * \param use_sequence_length If set to true, this layer takes in an extra input
 * \param axis The sequence axis. Only 0 is currently supported.
 * \return new symbol
 */
inline Symbol SequenceReverse(Symbol data,
                              Symbol sequence_length,
                              bool use_sequence_length = false,
                              int axis = 0) {
  return Operator("SequenceReverse")
           .SetParam("use_sequence_length", use_sequence_length)
           .SetParam("axis", axis)
           .SetInput("data", data)
           .SetInput("sequence_length", sequence_length)
           .CreateSymbol();
}

/*!
 * \brief Applies a spatial transformer to input feature map.
 * \param data Input data to the SpatialTransformerOp.
 * \param loc localisation net, the output dim should be 6 when transform_type is affine.
 * \param transform_type transformation type
 * \param sampler_type sampling type
 * \param target_shape output shape(h, w) of spatial transformer: (y, x)
 * \param cudnn_off whether to turn cudnn off
 * \return new symbol
 */
inline Symbol SpatialTransformer(Symbol data,
                                 Symbol loc,
                                 SpatialTransformerTransformType transform_type,
                                 SpatialTransformerSamplerType sampler_type,
                                 Shape target_shape = Shape(0,0),
                                 dmlc::optional<bool> cudnn_off = dmlc::optional<bool>()) {
  static const char *SpatialTransformerTransformTypeValues[] = {
    "affine"
  };
  static const char *SpatialTransformerSamplerTypeValues[] = {
    "bilinear"
  };
  return Operator("SpatialTransformer")
           .SetParam("transform_type", SpatialTransformerTransformTypeValues[int(transform_type)])
           .SetParam("sampler_type", SpatialTransformerSamplerTypeValues[int(sampler_type)])
           .SetParam("target_shape", target_shape)
           .SetParam("cudnn_off", cudnn_off)
           .SetInput("data", data)
           .SetInput("loc", loc)
           .CreateSymbol();
}

/*!
 * \brief Computes support vector machine based transformation of the input.
 *
 *        This tutorial demonstrates using SVM as output layer for classification instead
 *        https://github.com/dmlc/mxnet/tree/master/example/svm_mnist.
 *
 *
 * \param data Input data for SVM transformation.
 * \param label Class label for the input data.
 * \param margin The loss function penalizes outputs that lie outside this margin.
 * \param regularization_coefficient Regularization parameter for the SVM. This balances
 * \param use_linear Whether to use L1-SVM objective. L2-SVM objective is used by default.
 * \return new symbol
 */
inline Symbol SVMOutput(Symbol data,
                        Symbol label,
                        mx_float margin = 1,
                        mx_float regularization_coefficient = 1,
                        bool use_linear = false) {
  return Operator("SVMOutput")
           .SetParam("margin", margin)
           .SetParam("regularization_coefficient", regularization_coefficient)
           .SetParam("use_linear", use_linear)
           .SetInput("data", data)
           .SetInput("label", label)
           .CreateSymbol();
}

/*!
 * \brief Fill one element of each line(row for python, column for R/Julia) in lhs
 *        according to index indicated by rhs and values indicated by mhs. This function
 * \param lhs Left operand to the function.
 * \param mhs Middle operand to the function.
 * \param rhs Right operand to the function.
 * \return new symbol
 */
inline Symbol fill_element_0index(Symbol lhs,
                                  Symbol mhs,
                                  Symbol rhs) {
  return Operator("fill_element_0index")
           .SetInput("lhs", lhs)
           .SetInput("mhs", mhs)
           .SetInput("rhs", rhs)
           .CreateSymbol();
}

} //namespace cpp
} //namespace mxnet
#endif  // MXNET_CPP_OP_H_
