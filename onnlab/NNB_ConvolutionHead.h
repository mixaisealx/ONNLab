#pragma once
#include "NNBasicsInterfaces.h"
#include "BasicLayerI.h"
#include "BasicConvolutionI.h"
#include "InputNeuronI.h"
#include "BasicBackPropgI.h"
#include "BasicWghOptI.h"
#include "OptimizerI.h"
#include "AtomicSpinlock.h"

#include <vector>
#include <functional>
#include <algorithm>
#include <barrier>
#include <thread>

namespace nn
{
	template<interfaces::OptimizerInherit OptimizerT>
	class NNB_ConvolutionHead : public interfaces::BasicConvolutionI, public interfaces::BasicLayerInterface, public interfaces::BasicWeightOptimizableInterface, public interfaces::BackPropMetaLayerMark {
		nn::interfaces::BasicConvolutionEssenceI *frame;
		unsigned places_count, kernel_size;
		unsigned batch_size, last_batch_size;
		unsigned threads_count;
		OptimizerT *optimizer;
		struct Weight {
			float weight;
			OptimizerT::State optimizer_context;

			Weight() {
				weight = 0.0f;
			}
		};

		void(NNB_ConvolutionHead<OptimizerT>::*ResetAccumulator)(std::vector<float> accums[], unsigned count) const;
		
		std::vector<std::vector<Weight>> kernels_cells;
		std::vector<Weight> biases;

		AtomicSpinlock locker;
		std::barrier<> *threads_barrier;

		std::vector<std::vector<float>> forwardconv_accumulators; // Temp variable for single-thread operations
		std::vector<float> backpropbias_accumulator; // Temp variable for multi-thread bias grad store
		std::vector<float> backpropbias_accumulator_compensation; // Temp variable for multi-thread bias compensation 

		NNB_ConvolutionHead(const NNB_ConvolutionHead &) = delete;
		NNB_ConvolutionHead &operator=(const NNB_ConvolutionHead &) = delete;

		const std::vector<nn::interfaces::NBI *> &Neurons() override {
			throw std::exception("Neurons not allowed in ConvolutionHead!");
		}

		void AddNeuron(nn::interfaces::NeuronBasicInterface *) override {
			throw std::exception("AddNeuron not allowed in ConvolutionHead!");
		}

		void WeightOptimDoUpdate(float) override {
			throw std::exception("WeightOptimDoUpdate not allowed in ConvolutionHead!");
		}

		void BatchSizeUpdatedNotify(unsigned new_size) override {
			batch_size = new_size;
		}

		void ResetAccumulatorUnbiased(std::vector<float> accums[], unsigned count) const {
			for (unsigned i = 0; i != count; ++i) {
				std::fill_n(accums[i].begin(), batch_size, 0.0f);
			}
		}
		void ResetAccumulatorBiased(std::vector<float> accums[], unsigned count) const {
			for (unsigned i = 0; i != count; ++i) {
				std::fill_n(accums[i].begin(), batch_size, biases[i].weight);
			}
		}

		inline void InterlockedBackPropAccumulateError(interfaces::BasicBackPropogableInterface *nrn, float error, unsigned channel) {
			locker.lock();
			nrn->BackPropAccumulateError(error, channel);
			locker.unlock();
		}

		inline void DoThSfBiasKahanSummation(float value, unsigned kernelid) {
			locker.lock();
			float &sum = backpropbias_accumulator[kernelid];
			float &compensation = backpropbias_accumulator_compensation[kernelid];
			float y = value - compensation;
			float t = sum + y;
			compensation = (t - sum) - y;
			sum = t;
			locker.unlock();
		}
	public:
		NNB_ConvolutionHead(nn::interfaces::BasicConvolutionEssenceI *frame, OptimizerT *optimizer, std::function<float(bool is_bias)> weight_initializer, bool learn_bias = true):frame(frame), optimizer(optimizer) {
			places_count = frame->GetPlacesCount();
			kernel_size = frame->GetKernelSize();
			kernels_cells = std::vector<std::vector<Weight>>(frame->GetKernelsCount(), std::vector<Weight>(kernel_size));
			batch_size = frame->GetBatchSize();
			last_batch_size = 0;
			threads_count = 1;
			if (learn_bias) {
				biases.resize(kernels_cells.size());
				for (auto &elem : biases) {
					elem.weight = weight_initializer(true);
				}
				ResetAccumulator = &NNB_ConvolutionHead<OptimizerT>::ResetAccumulatorBiased;
			} else {
				ResetAccumulator = &NNB_ConvolutionHead<OptimizerT>::ResetAccumulatorUnbiased;
			}
			for (auto &kernel : kernels_cells) {
				for (auto &elem : kernel) {
					elem.weight = weight_initializer(false);
				}
			}
			this->WeightOptimReset();
			BCE_AddBatchUpdateSubscriber(frame, this);
			threads_barrier = new std::barrier<>(threads_count);
		}

		~NNB_ConvolutionHead() override {
			delete threads_barrier;
			BCE_RemoveBatchUpdateSubscriber(frame, this);
		}

		unsigned CalcWeightsCount() override {
			return kernels_cells.size() * kernel_size + biases.size();
		}

		std::vector<float> RetrieveWeights() override {
			std::vector<float> result; // Compiler should apply Return-Value-Optimization
			result.reserve(CalcWeightsCount());
			// Saving biases
			for (auto &elem : biases) {
				result.push_back(elem.weight);
			}
			// Saving cells
			for (auto &kernel : kernels_cells) {
				for (auto &elem : kernel) {
					result.push_back(elem.weight);
				}
			}
			return result;
		}

		void PushWeights(const std::vector<float> &weights) override {
			if (weights.size() != CalcWeightsCount()) {
				throw std::exception("Weights vector have a wrong size!");
			}
			auto current = weights.cbegin();
			// Restoring biases
			for (auto &elem : biases) {
				elem.weight = *current;
				++current;
			}
			// Restoring cells
			for (auto &kernel : kernels_cells) {
				for (auto &elem : kernel) {
					elem.weight = *current;
					++current;
				}
			}
		}

		bool HasTrainable() override {
			return true;
		}

		// Single-thread implementation
		void PerformFullConvolution() override {
			if (batch_size > last_batch_size) {
				if (!last_batch_size) {
					forwardconv_accumulators = std::vector<std::vector<float>>(frame->GetKernelsCount());
				}
				for (auto &accum : forwardconv_accumulators) {
					if (accum.size() < batch_size) {
						accum.resize(batch_size);
					}
				}
				last_batch_size = batch_size;
			}
			
			interfaces::NBI *nrn;
			interfaces::InputNeuronI *nri;
			float value;
			for (unsigned place = 0; place != places_count; ++place) {
				(this->*ResetAccumulator)(&forwardconv_accumulators[0], kernels_cells.size());
				for (unsigned cell = 0; cell != kernel_size; ++cell) {
					nrn = frame->GetInputforLocation(place, cell);
					for (unsigned channel = 0; channel != batch_size; ++channel) {
						value = nrn->OwnLevel(channel);
						for (unsigned kernelid = 0; kernelid != kernels_cells.size(); ++kernelid) {
							forwardconv_accumulators[kernelid][channel] += value * kernels_cells[kernelid][cell].weight;
						}
					}
				}
				for (unsigned kernelid = 0; kernelid != kernels_cells.size(); ++kernelid) {
					nri = dynamic_cast<interfaces::InputNeuronI *>(frame->GetOutputforLocation(place, kernelid));
					auto &kernel = forwardconv_accumulators[kernelid];
					for (unsigned channel = 0; channel != batch_size; ++channel) {
						nri->SetOwnLevel(kernel[channel], channel);
					}
				}
			}
		}

		// Single-thread implementation
		void BackPropagateFullConvolution() override {
			float weight_bias_optim;
			float weight_optim_grad;
			float error;
			interfaces::NBI *input;
			interfaces::BasicBackPropogableInterface *bpo, *bpi;

			for (unsigned kernelid = 0; kernelid != kernels_cells.size(); ++kernelid) {
				weight_bias_optim = 0.0f;

				for (unsigned cell = 0; cell != kernel_size; ++cell) {
					Weight &curr_cell = kernels_cells[kernelid][cell];
					weight_optim_grad = 0.0f;
					for (unsigned place = 0; place != places_count; ++place) {
						input = frame->GetInputforLocation(place, cell);
						bpo = dynamic_cast<interfaces::BasicBackPropogableInterface *>(frame->GetOutputforLocation(place, kernelid));
						if (input->IsTrainable()) {
							bpi = dynamic_cast<interfaces::BasicBackPropogableInterface *>(input);
							for (unsigned channel = 0; channel != batch_size; ++channel) {
								error = bpo->BackPropGetFinalError(channel);
								weight_optim_grad += error * input->OwnLevel(channel);
								bpi->BackPropAccumulateError(error * curr_cell.weight, channel);
							}
						} else {
							for (unsigned channel = 0; channel != batch_size; ++channel) {
								weight_optim_grad += bpo->BackPropGetFinalError(channel) * input->OwnLevel(channel);
							}
						}
					}
					weight_optim_grad /= batch_size; // Averaging the gradient across the batch
					weight_bias_optim += weight_optim_grad;
					curr_cell.weight -= optimizer->CalcDelta(weight_optim_grad, &curr_cell.optimizer_context);
				}
				
				if (biases.size()) {
					biases[kernelid].weight -= optimizer->CalcDelta(weight_bias_optim, &biases[kernelid].optimizer_context);
				}
				for (unsigned place = 0; place != places_count; ++place) {
					dynamic_cast<interfaces::BasicBackPropogableInterface *>(frame->GetOutputforLocation(place, kernelid))->BackPropResetError();
				}
			}
		}

		// Single-thread implementation
		void BackPropagateFullConvolutionSpecial(bool update_weights, bool cleanup_grads, void *) override {
			float weight_bias_optim;
			float weight_optim_grad;
			float error;
			interfaces::NBI *input;
			interfaces::BasicBackPropogableInterface *bpo, *bpi;

			for (unsigned kernelid = 0; kernelid != kernels_cells.size(); ++kernelid) {
				weight_bias_optim = 0.0f;

				for (unsigned cell = 0; cell != kernel_size; ++cell) {
					Weight &curr_cell = kernels_cells[kernelid][cell];
					weight_optim_grad = 0.0f;
					for (unsigned place = 0; place != places_count; ++place) {
						input = frame->GetInputforLocation(place, cell);
						bpo = dynamic_cast<interfaces::BasicBackPropogableInterface *>(frame->GetOutputforLocation(place, kernelid));
						if (input->IsTrainable()) {
							bpi = dynamic_cast<interfaces::BasicBackPropogableInterface *>(input);
							for (unsigned channel = 0; channel != batch_size; ++channel) {
								error = bpo->BackPropGetFinalError(channel);
								weight_optim_grad += error * input->OwnLevel(channel);
								bpi->BackPropAccumulateError(error * curr_cell.weight, channel);
							}
						} else {
							for (unsigned channel = 0; channel != batch_size; ++channel) {
								weight_optim_grad += bpo->BackPropGetFinalError(channel) * input->OwnLevel(channel);
							}
						}
					}
					weight_optim_grad /= batch_size; // Averaging the gradient across the batch
					weight_bias_optim += weight_optim_grad;
					if (update_weights) {
						curr_cell.weight -= optimizer->CalcDelta(weight_optim_grad, &curr_cell.optimizer_context);
					}
				}

				if (update_weights && biases.size()) {
					biases[kernelid].weight -= optimizer->CalcDelta(weight_bias_optim, &biases[kernelid].optimizer_context);
				}
				if (cleanup_grads) {
					for (unsigned place = 0; place != places_count; ++place) {
						dynamic_cast<interfaces::BasicBackPropogableInterface *>(frame->GetOutputforLocation(place, kernelid))->BackPropResetError();
					}
				}
			}
		}

		unsigned GetRequiredCachesCount(bool for_backprop) override {
			return (for_backprop ? 0 : 1);
		}

		// Multi-thread implementation
		void PerformPartialConvolution(unsigned worker_id, std::vector<float> caches[]) override {
			if (caches->size() < batch_size) {
				caches->resize(batch_size);
			}
			unsigned job_id = 0;

			interfaces::NBI *nrn;
			interfaces::InputNeuronI *nri;
			float weight;
			for (unsigned place = 0; place != places_count; ++place) {
				for (unsigned kernelid = 0; kernelid != kernels_cells.size(); ++kernelid) {
					if (job_id == worker_id) {
						(this->*ResetAccumulator)(caches, 1);
						auto &kernelcl = kernels_cells[kernelid];
						for (unsigned cell = 0; cell != kernel_size; ++cell) {
							nrn = frame->GetInputforLocation(place, cell);
							weight = kernelcl[cell].weight;
							for (unsigned channel = 0; channel != batch_size; ++channel) {
								caches[0][channel] += nrn->OwnLevel(channel) * weight;
							}
						}
						nri = dynamic_cast<interfaces::InputNeuronI *>(frame->GetOutputforLocation(place, kernelid));
						for (unsigned channel = 0; channel != batch_size; ++channel) {
							nri->SetOwnLevel(caches[0][channel], channel);
						}
					}
					if (++job_id == threads_count) {
						job_id = 0;
					}
				}
			}
		}

		// Multi-thread implementation
		void BackPropagatePartialConvolution(unsigned worker_id, std::vector<float> caches[]) override {
			if (biases.size()) { // Use biases
				if (!worker_id) { // worker_id == 0
					if (backpropbias_accumulator.size() != biases.size()) {
						backpropbias_accumulator.resize(biases.size());
						backpropbias_accumulator_compensation.resize(biases.size());
					}
					std::fill(backpropbias_accumulator.begin(), backpropbias_accumulator.end(), 0.0f);
					std::fill(backpropbias_accumulator_compensation.begin(), backpropbias_accumulator_compensation.end(), 0.0f);
				}

				threads_barrier->arrive_and_wait();
			}

			unsigned job_id = 0;

			float weight_bias_optim;
			float weight_optim_grad;
			float error;
			interfaces::NBI *input;
			interfaces::BasicBackPropogableInterface *bpo, *bpi;
			for (unsigned kernelid = 0; kernelid != kernels_cells.size(); ++kernelid) {
				weight_bias_optim = 0.0f;

				for (unsigned cell = 0; cell != kernel_size; ++cell) {
					if (job_id == worker_id) {
						Weight &curr_cell = kernels_cells[kernelid][cell];
						weight_optim_grad = 0.0f;
						for (unsigned place = 0; place != places_count; ++place) {
							input = frame->GetInputforLocation(place, cell);
							bpo = dynamic_cast<interfaces::BasicBackPropogableInterface *>(frame->GetOutputforLocation(place, kernelid));
							if (input->IsTrainable()) {
								bpi = dynamic_cast<interfaces::BasicBackPropogableInterface *>(input);
								for (unsigned channel = 0; channel != batch_size; ++channel) {
									error = bpo->BackPropGetFinalError(channel);
									weight_optim_grad += error * input->OwnLevel(channel);
									InterlockedBackPropAccumulateError(bpi, error * curr_cell.weight, channel);
								}
							} else {
								for (unsigned channel = 0; channel != batch_size; ++channel) {
									weight_optim_grad += bpo->BackPropGetFinalError(channel) * input->OwnLevel(channel);
								}
							}
						}
						weight_optim_grad /= batch_size; // Averaging the gradient across the batch
						weight_bias_optim += weight_optim_grad;
						curr_cell.weight -= optimizer->CalcDelta(weight_optim_grad, &curr_cell.optimizer_context);
					}

					if (++job_id == threads_count) {
						job_id = 0;
					}
				}

				if (biases.size()) 
					DoThSfBiasKahanSummation(weight_bias_optim, kernelid);
			}

			threads_barrier->arrive_and_wait();

			if (biases.size()) {
				for (unsigned kernelid = 0; kernelid != kernels_cells.size(); ++kernelid) {
					if (job_id == worker_id) {
						biases[kernelid].weight -= optimizer->CalcDelta(backpropbias_accumulator[kernelid], &biases[kernelid].optimizer_context);
					}

					if (++job_id == threads_count) {
						job_id = 0;
					}
				}
			}

			for (unsigned kernelid = 0; kernelid != kernels_cells.size(); ++kernelid) {
				for (unsigned place = 0; place != places_count; ++place) {
					if (job_id == worker_id) {
						dynamic_cast<interfaces::BasicBackPropogableInterface *>(frame->GetOutputforLocation(place, kernelid))->BackPropResetError();
					}

					if (++job_id == threads_count) {
						job_id = 0;
					}
				}
			}
		}

		unsigned GetThreadsCount() override {
			return threads_count;
		}

		void SetThreadsCount(unsigned threads_count) override {
			if (!threads_count)
				throw std::exception("Zero threads count!");

			if (this->threads_count != threads_count) {
				this->threads_count = threads_count;

				delete threads_barrier;
				threads_barrier = new std::barrier<>(threads_count);
			}
		}

		void CleanupLocalCaches() override {
			last_batch_size = 0;
			std::vector<std::vector<float>>().swap(forwardconv_accumulators);
			std::vector<float>().swap(backpropbias_accumulator);
			std::vector<float>().swap(backpropbias_accumulator_compensation);
		}

		void WeightOptimReset() override {
			for (auto &kernel : kernels_cells) {
				for (auto &elem : kernel) {
					optimizer->Reset(&elem.optimizer_context);
				}
			}
		}
	};
}
