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
	template<bool MaxPooling>
	class NNB_ConvolutionMinMaxPoolingHead : public interfaces::BasicConvolutionI, public interfaces::BasicLayerInterface, public interfaces::BackPropMetaLayerMark {
		nn::interfaces::BasicConvolutionEssenceI *frame;
		unsigned places_count, kernel_size;
		unsigned batch_size, last_batch_size;
		unsigned threads_count;

		AtomicSpinlock locker;

		std::vector<float> batch_boundvals; // Temp variable for single-thread operations

		NNB_ConvolutionMinMaxPoolingHead(const NNB_ConvolutionMinMaxPoolingHead &) = delete;
		NNB_ConvolutionMinMaxPoolingHead &operator=(const NNB_ConvolutionMinMaxPoolingHead &) = delete;

		const std::vector<nn::interfaces::NBI *> &Neurons() override {
			throw std::exception("Neurons not allowed in ConvolutionHead!");
		}

		void AddNeuron(nn::interfaces::NeuronBasicInterface *) override {
			throw std::exception("AddNeuron not allowed in ConvolutionHead!");
		}

		void BatchSizeUpdatedNotify(unsigned new_size) override {
			batch_size = new_size;
		}

		inline void InterlockedBackPropAccumulateError(interfaces::BasicBackPropogableInterface *nrn, float error, unsigned channel) {
			locker.lock();
			nrn->BackPropAccumulateError(error, channel);
			locker.unlock();
		}

		static inline bool Compare(float val_new, float val_known) {
			if constexpr (MaxPooling) {
				return val_new > val_known;
			} else {
				return val_new < val_known;
			}
		}
	public:
		NNB_ConvolutionMinMaxPoolingHead(nn::interfaces::BasicConvolutionEssenceI *frame):frame(frame) {
			if (frame->GetKernelsCount() != 1) {
				throw std::exception("Frame's Kernels Count number is wrong for pooling! It must be exactly equal to one!");
			}
			places_count = frame->GetPlacesCount();
			kernel_size = frame->GetKernelSize();
			batch_size = frame->GetBatchSize();

			last_batch_size = 0;
			threads_count = 1;

			BCE_AddBatchUpdateSubscriber(frame, this);
		}

		~NNB_ConvolutionMinMaxPoolingHead() override {
			BCE_RemoveBatchUpdateSubscriber(frame, this);
		}

		unsigned CalcWeightsCount() override {
			return 0;
		}

		std::vector<float> RetrieveWeights() override {
			return std::vector<float>(); // Compiler should apply Return-Value-Optimization
		}

		void PushWeights(const std::vector<float> &weights) override {
			if (!weights.empty()) {
				throw std::exception("Weights vector have a wrong size!");
			}
		}

		bool HasTrainable() override {
			return true;
		}

		// Single-thread implementation
		void PerformFullConvolution() override {
			if (batch_size > last_batch_size) {
				if (!last_batch_size) {
					batch_boundvals = std::vector<float>(frame->GetKernelsCount());
				} else if (batch_boundvals.size() < batch_size) {
					batch_boundvals.resize(batch_size);
				}
				last_batch_size = batch_size;
			}

			interfaces::NBI *nrn;
			interfaces::InputNeuronI *nri;
			float value;
			for (unsigned place = 0; place != places_count; ++place) {
				if constexpr (MaxPooling) {
					std::fill(batch_boundvals.begin(), batch_boundvals.end(), -std::numeric_limits<float>::infinity());
				} else {
					std::fill(batch_boundvals.begin(), batch_boundvals.end(), std::numeric_limits<float>::infinity());
				}
				for (unsigned cell = 0; cell != kernel_size; ++cell) {
					nrn = frame->GetInputforLocation(place, cell);
					for (unsigned channel = 0; channel != batch_size; ++channel) {
						value = nrn->OwnLevel(channel);
						if (Compare(value, batch_boundvals[channel])) {
							batch_boundvals[channel] = value;
						}
					}
				}
				nri = dynamic_cast<interfaces::InputNeuronI *>(frame->GetOutputforLocation(place, 0));
				for (unsigned channel = 0; channel != batch_size; ++channel) {
					nri->SetOwnLevel(batch_boundvals[channel], channel);
				}
			}
		}

		// Single-thread implementation
		void BackPropagateFullConvolution() override {
			if (batch_size > last_batch_size) {
				if (!last_batch_size) {
					batch_boundvals = std::vector<float>(frame->GetKernelsCount());
				} else if (batch_boundvals.size() < batch_size) {
					batch_boundvals.resize(batch_size);
				}
				last_batch_size = batch_size;
			}

			interfaces::NBI *input, *output;
			interfaces::BasicBackPropogableInterface *bpo, *bpi;
			unsigned propagation_remains;

			for (unsigned place = 0; place != places_count; ++place) {
				output = frame->GetOutputforLocation(place, 0);
				for (unsigned channel = 0; channel != batch_size; ++channel) {
					batch_boundvals[channel] = output->OwnLevel(channel);
				}
				bpo = dynamic_cast<interfaces::BasicBackPropogableInterface *>(output);
				
				propagation_remains = batch_size;
				for (unsigned cell = 0; cell != kernel_size; ++cell) {
					input = frame->GetInputforLocation(place, cell);
					if (input->IsTrainable()) {
						bpi = dynamic_cast<interfaces::BasicBackPropogableInterface *>(input);
						for (unsigned channel = 0; channel != batch_size; ++channel) {
							if (batch_boundvals[channel] == input->OwnLevel(channel)) {
								// Propagating error to the first found
								bpi->BackPropAccumulateError(bpo->BackPropGetFinalError(channel), channel);
								// Resetting channel to disable further propagation for it
								batch_boundvals[channel] = std::numeric_limits<float>::quiet_NaN(); // Even NaN != NaN
								// Decreasing counter of remaining errors to propagate
								if (!(--propagation_remains)) {
									break;
								}
							}
						}
						if (!propagation_remains) {
							break;
						}
					}
				}
				bpo->BackPropResetError();
			}
		}

		// Single-thread implementation
		void BackPropagateFullConvolutionSpecial(bool, bool cleanup_grads, void *) override {
			if (batch_size > last_batch_size) {
				if (!last_batch_size) {
					batch_boundvals = std::vector<float>(frame->GetKernelsCount());
				} else if (batch_boundvals.size() < batch_size) {
					batch_boundvals.resize(batch_size);
				}
				last_batch_size = batch_size;
			}

			interfaces::NBI *input, *output;
			interfaces::BasicBackPropogableInterface *bpo, *bpi;
			unsigned propagation_remains;

			for (unsigned place = 0; place != places_count; ++place) {
				output = frame->GetOutputforLocation(place, 0);
				for (unsigned channel = 0; channel != batch_size; ++channel) {
					batch_boundvals[channel] = output->OwnLevel(channel);
				}
				bpo = dynamic_cast<interfaces::BasicBackPropogableInterface *>(output);

				propagation_remains = batch_size;
				for (unsigned cell = 0; cell != kernel_size; ++cell) {
					input = frame->GetInputforLocation(place, cell);
					if (input->IsTrainable()) {
						bpi = dynamic_cast<interfaces::BasicBackPropogableInterface *>(input);
						for (unsigned channel = 0; channel != batch_size; ++channel) {
							if (batch_boundvals[channel] == input->OwnLevel(channel)) {
								// Propagating error to the first found
								bpi->BackPropAccumulateError(bpo->BackPropGetFinalError(channel), channel);
								// Resetting channel to disable further propagation for it
								batch_boundvals[channel] = std::numeric_limits<float>::quiet_NaN(); // Even NaN != NaN
								// Decreasing counter of remaining errors to propagate
								if (!(--propagation_remains)) {
									break;
								}
							}
						}
						if (!propagation_remains) {
							break;
						}
					}
				}
				if (cleanup_grads) {
					bpo->BackPropResetError();
				}
			}
		}

		unsigned GetRequiredCachesCount(bool for_backprop) override {
			return 1U;
		}

		// Multi-thread implementation
		void PerformPartialConvolution(unsigned worker_id, std::vector<float> caches[]) override {
			if (caches->size() < batch_size) {
				caches->resize(batch_size);
			}

			interfaces::NBI *nrn;
			interfaces::InputNeuronI *nri;
			float value;
			for (unsigned place = worker_id; place < places_count; place += threads_count) {
				if constexpr (MaxPooling) {
					std::fill(caches->begin(), caches->end(), -std::numeric_limits<float>::infinity());
				} else {
					std::fill(caches->begin(), caches->end(), std::numeric_limits<float>::infinity());
				}
				for (unsigned cell = 0; cell != kernel_size; ++cell) {
					nrn = frame->GetInputforLocation(place, cell);
					for (unsigned channel = 0; channel != batch_size; ++channel) {
						value = nrn->OwnLevel(channel);
						if (Compare(value, (*caches)[channel])) {
							(*caches)[channel] = value;
						}
					}
				}
				nri = dynamic_cast<interfaces::InputNeuronI *>(frame->GetOutputforLocation(place, 0));
				for (unsigned channel = 0; channel != batch_size; ++channel) {
					nri->SetOwnLevel((*caches)[channel], channel);
				}
			}
		}

		// Multi-thread implementation
		void BackPropagatePartialConvolution(unsigned worker_id, std::vector<float> caches[]) override {
			if (caches->size() < batch_size) {
				caches->resize(batch_size);
			}

			interfaces::NBI *input, *output;
			interfaces::BasicBackPropogableInterface *bpo, *bpi;
			unsigned propagation_remains;

			for (unsigned place = worker_id; place < places_count; place += threads_count) {
				output = frame->GetOutputforLocation(place, 0);
				for (unsigned channel = 0; channel != batch_size; ++channel) {
					(*caches)[channel] = output->OwnLevel(channel);
				}
				bpo = dynamic_cast<interfaces::BasicBackPropogableInterface *>(output);

				propagation_remains = batch_size;
				for (unsigned cell = 0; cell != kernel_size; ++cell) {
					input = frame->GetInputforLocation(place, cell);
					if (input->IsTrainable()) {
						bpi = dynamic_cast<interfaces::BasicBackPropogableInterface *>(input);
						for (unsigned channel = 0; channel != batch_size; ++channel) {
							if ((*caches)[channel] == input->OwnLevel(channel)) {
								// Propagating error to the first found
								InterlockedBackPropAccumulateError(bpi, bpo->BackPropGetFinalError(channel), channel);
								// Resetting channel to disable further propagation for it
								 (*caches)[channel] = std::numeric_limits<float>::quiet_NaN(); // Even NaN != NaN
								// Decreasing counter of remaining errors to propagate
								if (!(--propagation_remains)) {
									break;
								}
							}
						}
						if (!propagation_remains) {
							break;
						}
					}
				}
				bpo->BackPropResetError();
			}
		}

		unsigned GetThreadsCount() override {
			return threads_count;
		}

		void SetThreadsCount(unsigned threads_count) override {
			if (!threads_count)
				throw std::exception("Zero threads count!");

			this->threads_count = threads_count;
		}

		void CleanupLocalCaches() override {
			last_batch_size = 0;
			std::vector<float>().swap(batch_boundvals);
		}
	};

	using NNB_ConvolutionMinPoolingHead = NNB_ConvolutionMinMaxPoolingHead<false>;
	using NNB_ConvolutionMaxPoolingHead = NNB_ConvolutionMinMaxPoolingHead<true>;
}
