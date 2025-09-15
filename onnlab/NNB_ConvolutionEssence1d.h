#pragma once
#include "NNBasicsInterfaces.h"
#include "BasicLayerI.h"
#include "BasicConvolutionI.h"
#include "InputNeuronI.h"
#include "BasicBackPropgI.h"
#include "BatchNeuronBasicI.h"

#include <limits>
#include <stdexcept>

namespace nn
{
	class NNB_ConvolutionEssence1d : public nn::interfaces::BasicConvolutionEssenceI {
		interfaces::BasicLayerInterface *input_layer, *output_layer;
		unsigned kernel_size, kernels_count, stride, dilation, places_count;
		unsigned batch_size;
		std::vector<interfaces::BasicConvolutionI *> subscribers;

		void AddBatchUpdateSubscriber(interfaces::BasicConvolutionI *subscriber) override {
			subscribers.push_back(subscriber);
		}

		void RemoveBatchUpdateSubscriber(interfaces::BasicConvolutionI *subscriber) override {
			subscribers.erase(std::remove(subscribers.begin(), subscribers.end(), subscriber), subscribers.end());
		}

		NNB_ConvolutionEssence1d(const NNB_ConvolutionEssence1d &) = delete;
		NNB_ConvolutionEssence1d &operator=(const NNB_ConvolutionEssence1d &) = delete;
	public:
		NNB_ConvolutionEssence1d(interfaces::BasicLayerInterface *input_layer,
								 interfaces::BasicLayerInterface *output_layer,
								 unsigned kernel_size, unsigned kernels_count = 1, unsigned stride = 1, unsigned dilation = 1, bool allow_input_kernel_underfit = false):input_layer(input_layer), output_layer(output_layer), kernel_size(kernel_size), kernels_count(kernels_count), stride(stride), dilation(dilation) {

			if (kernel_size < 2 || !stride || !dilation) {
				throw std::runtime_error("Zeros in stride, dilation and kernel_size < 2 is not allowed!");
			}
			this->RecalcBatchSize();

			for (auto elem : output_layer->Neurons()) {
				if (!dynamic_cast<nn::interfaces::InputNeuronI *>(elem) || !dynamic_cast<nn::interfaces::BasicBackPropogableInterface *>(elem)) {
					throw std::runtime_error("The output layer must consist of neurons based on interfaces::InputNeuronI!");
				}
			}

			unsigned emplaceable_size = input_layer->Neurons().size() - (kernel_size + (kernel_size - 1) * (dilation - 1));
			places_count = emplaceable_size / stride + 1;
			if (!allow_input_kernel_underfit && emplaceable_size % stride) { // Check if layers have appropriate sizes
				throw std::runtime_error("That kernel_size and stride in not appropriate for input layer of such size!");
			} else if (places_count * kernels_count != output_layer->Neurons().size()) {
				throw std::runtime_error("The input layer in not appropriate for output layer of such size!");
			}
		}

		static inline NNB_ConvolutionEssence1d BuildPoolingSetup(interfaces::BasicLayerInterface *input_layer,
														  interfaces::BasicLayerInterface *output_layer, 
														  unsigned kernel_size,
														  bool allow_input_kernel_underfit = false) {
			return NNB_ConvolutionEssence1d(input_layer, output_layer, kernel_size, 1, kernel_size, 1, allow_input_kernel_underfit);
		}

		void RecalcBatchSize() override {
			unsigned old_batch = batch_size;
			unsigned tmp;
			interfaces::BatchNeuronBasicI *nrn;
			batch_size = std::numeric_limits<unsigned>::max(); // Temporal value
			for (auto elem : input_layer->Neurons()) {
				nrn = dynamic_cast<interfaces::BatchNeuronBasicI *>(elem);
				tmp = (nrn ? nrn->GetCurrentBatchSize() : 1);
				if (tmp != batch_size && tmp != std::numeric_limits<unsigned>::max()) {
					if (batch_size == std::numeric_limits<unsigned>::max()) {
						batch_size = tmp;
					} else {
						throw std::runtime_error("Different batch sizes (input layer) is not allowed!");
					}
				}
			}
			for (auto elem : output_layer->Neurons()) {
				nrn = dynamic_cast<interfaces::BatchNeuronBasicI *>(elem);
				tmp = (nrn ? nrn->GetCurrentBatchSize() : 1);
				if (tmp != batch_size && tmp != std::numeric_limits<unsigned>::max()) {
					if (batch_size == std::numeric_limits<unsigned>::max()) {
						batch_size = tmp;
					} else {
						throw std::runtime_error("Different batch sizes (output layer) is not allowed!");
					}
				}
			}
			if (batch_size != old_batch) {
				for (auto conv : subscribers) {
					BC_BatchUpdateNotify(conv, batch_size);
				}
			}
		}

		unsigned GetBatchSize() override {
			return batch_size;
		}

		unsigned GetPlacesCount() override {
			return places_count;
		}

		unsigned GetKernelSize() override {
			return kernel_size;
		}

		nn::interfaces::NBI *GetInputforLocation(unsigned place_id, unsigned weight_id) override {
			return input_layer->Neurons()[place_id * stride + weight_id * dilation];
		}

		nn::interfaces::NBI *GetOutputforLocation(unsigned place_id, unsigned kernel_id) override {
			return dynamic_cast<nn::interfaces::NBI *>(output_layer->Neurons()[kernel_id * places_count + place_id]);
		}

		unsigned GetKernelsCount() override {
			return kernels_count;
		}

		virtual std::vector<unsigned> CalcInputShape() override {
			return std::vector<unsigned> {input_layer->Neurons().size(), 1};
		}

		virtual std::vector<unsigned> CalcOutputShape() override {
			return std::vector<unsigned> {places_count, kernels_count};
		}
	};
}
