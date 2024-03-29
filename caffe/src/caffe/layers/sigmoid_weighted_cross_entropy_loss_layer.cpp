#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sigmoid_weighted_cross_entropy_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidWeightedCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  const SigmoidWeightedCELossParameter  sigmoid_wce_loss_param = this->layer_param_.sigmoid_wce_loss_param();
  alpha_ = sigmoid_wce_loss_param.alpha();
}

template <typename Dtype>
void SigmoidWeightedCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_WEIGHTED_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void SigmoidWeightedCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  const Dtype* in_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();

  Dtype loss_pos = 0;
  Dtype loss_neg = 0;
  Dtype count_pos = 0;
  Dtype count_neg = 0;

  int dim = bottom[0]->count() / bottom[0]->num();
  for (int j = 0; j < dim; j++) {
    const int target_value = static_cast<int>(target[j]);
    if (target_value == 1) {
      count_pos++;
      loss_pos -= in_data[j]*(1-(in_data[j]>=0))-log(1+exp(in_data[j]-2*in_data[j]*(in_data[j]>= 0)));
	  } else {
	    count_neg++;
      loss_neg -= in_data[j]*(0-(in_data[j]>=0))-log(1+exp(in_data[j]-2*in_data[j]*(in_data[j]>= 0)));
	  }
  }

  weight_pos_ = 1.0 * count_neg / (count_pos + count_neg);
  weight_neg_ = 1.0 * count_pos / (count_pos + count_neg);
	weight_pos_ *= (1.0    / (alpha_ + 1));
	weight_neg_ *= (alpha_ / (alpha_ + 1));
	  
  loss_pos *= weight_pos_;
  loss_neg *= weight_neg_;

  top[0]->mutable_cpu_data()[0] = (loss_pos + loss_neg);
}

template <typename Dtype>
void SigmoidWeightedCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);

    int dim = bottom[0]->count() / bottom[0]->num();
	  for (int j = 0; j < dim; j++) {
    const int target_value = static_cast<int>(target[j]);
	    if (target_value == 1) {
	  	  bottom_diff[j] *= weight_pos_;
	    } else {
	    	bottom_diff[j] *= weight_neg_;
			}
	  }

    const Dtype loss_weight = top [0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(SigmoidWeightedCrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(SigmoidWeightedCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(SigmoidWeightedCrossEntropyLoss);

}  // namespace caffe
