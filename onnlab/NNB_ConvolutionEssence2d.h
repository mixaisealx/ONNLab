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
	class NNB_ConvolutionEssence2d : public nn::interfaces::BasicConvolutionEssenceI {
		interfaces::BasicLayerInterface *input_layer, *output_layer;
		unsigned width, places_x_count, kernel_x_size, kernels_count;
		unsigned stride_x, stride_y, dilation_x, dilation_y;
		unsigned kernel_area, places_count;
		unsigned batch_size;
		std::vector<interfaces::BasicConvolutionI *> subscribers;

		void AddBatchUpdateSubscriber(interfaces::BasicConvolutionI *subscriber) override {
			subscribers.push_back(subscriber);
		}

		void RemoveBatchUpdateSubscriber(interfaces::BasicConvolutionI *subscriber) override {
			subscribers.erase(std::remove(subscribers.begin(), subscribers.end(), subscriber), subscribers.end());
		}

		NNB_ConvolutionEssence2d(const NNB_ConvolutionEssence2d &) = delete;
		NNB_ConvolutionEssence2d &operator=(const NNB_ConvolutionEssence2d &) = delete;

		inline void Init(unsigned kernel_size_y, bool allow_input_kernel_underfit) {
			RecalcBatchSize();

			for (auto elem : output_layer->Neurons()) {
				if (!dynamic_cast<nn::interfaces::InputNeuronI *>(elem) || !dynamic_cast<nn::interfaces::BasicBackPropogableInterface *>(elem)) {
					throw std::runtime_error("The output layer must consist of neurons based on interfaces::InputNeuronI!");
				}
			}

			unsigned emplaceable_x_size = width - (kernel_x_size + (kernel_x_size - 1) * (dilation_x - 1));
			places_x_count = emplaceable_x_size / stride_x + 1;
			if (!allow_input_kernel_underfit && emplaceable_x_size % stride_x) { // Check if layers have appropriate sizes
				throw std::runtime_error("That kernel_size_x and stride in not appropriate for input layer of such size!");
			}
			unsigned height = input_layer->Neurons().size() / width;
			unsigned emplaceable_y_size = height - (kernel_size_y + (kernel_size_y - 1) * (dilation_y - 1));
			unsigned places_y_count = emplaceable_y_size / stride_y + 1;
			if (!allow_input_kernel_underfit && emplaceable_y_size % stride_y) { // Check if layers have appropriate sizes
				throw std::runtime_error("That kernel_size_y and stride in not appropriate for input layer of such size!");
			}
			places_count = places_x_count * places_y_count;
			if (places_count * kernels_count != output_layer->Neurons().size()) {
				throw std::runtime_error("The input layer in not appropriate for output layer of such size!");
			}
			kernel_area = kernel_x_size * kernel_size_y;
		}

	public:
		struct InputShape {
			unsigned input_size_x, input_size_y;
			InputShape(unsigned input_size_x, unsigned input_size_y): input_size_x(input_size_x), input_size_y(input_size_y) {}
		};

		struct KernelShape {
			unsigned kernel_size_x, kernel_size_y;
			KernelShape(unsigned square_kernel_size): kernel_size_x(square_kernel_size), kernel_size_y(square_kernel_size) {}
			KernelShape(unsigned kernel_size_x, unsigned kernel_size_y): kernel_size_x(kernel_size_x), kernel_size_y(kernel_size_y) {}
		};

		struct KernelMovement {
			unsigned stride_x, stride_y, dilation_x, dilation_y;
			KernelMovement(unsigned stride = 1, unsigned dilation = 1): stride_x(stride), stride_y(stride), dilation_x(dilation), dilation_y(dilation) {}
			KernelMovement(unsigned stride_x, unsigned stride_y, unsigned dilation_x, unsigned dilation_y): stride_x(stride_x), stride_y(stride_y), dilation_x(dilation_x), dilation_y(dilation_y) {}
		};

		NNB_ConvolutionEssence2d(interfaces::BasicLayerInterface *input_layer,
								 interfaces::BasicLayerInterface *output_layer,
								 unsigned input_shape_x,
								 KernelShape kernel_shape,
								 KernelMovement kernel_movement,
								 unsigned kernels_count = 1,
								 bool allow_input_kernel_underfit = false):
			input_layer(input_layer), output_layer(output_layer), width(input_shape_x), kernel_x_size(kernel_shape.kernel_size_x), kernels_count(kernels_count), 
			stride_x(kernel_movement.stride_x), stride_y(kernel_movement.stride_y), dilation_x(kernel_movement.dilation_x), dilation_y(kernel_movement.dilation_y) {
			
			unsigned div = input_layer->Neurons().size() / input_shape_x; // For compiler optimization
			unsigned mod = input_layer->Neurons().size() % input_shape_x; // For compiler optimization

			if (input_shape_x < 2 || div < 2) {
				throw std::runtime_error("Shape of input must be at least 2! (otherwise use 1d convolution)");
			}
			if (mod) {
				throw std::runtime_error("input_shape_x is uncorrect for currect input size!");
			}
			if (kernel_shape.kernel_size_x < 2 && kernel_shape.kernel_size_y < 2 || !kernel_shape.kernel_size_x || !kernel_shape.kernel_size_y || !stride_x || !stride_y || !dilation_x || !dilation_y) {
				throw std::runtime_error("Zeros in stride, dilation and kernel_size < 2 is not allowed!");
			}
			if (kernel_shape.kernel_size_x == 1 && dilation_x != 1 || kernel_shape.kernel_size_y == 1 && dilation_y != 1) {
				throw std::runtime_error("For kernel_size==1, dilation must be ==1");
			}
			
			Init(kernel_shape.kernel_size_y, allow_input_kernel_underfit);
		}

		NNB_ConvolutionEssence2d(interfaces::BasicLayerInterface *input_layer,
								 interfaces::BasicLayerInterface *output_layer,
								 InputShape input_shape,
								 KernelShape kernel_shape,
								 KernelMovement kernel_movement,
								 unsigned kernels_count = 1,
								 bool allow_input_kernel_underfit = false):
			input_layer(input_layer), output_layer(output_layer), width(input_shape.input_size_x), kernel_x_size(kernel_shape.kernel_size_x), kernels_count(kernels_count),
			stride_x(kernel_movement.stride_x), stride_y(kernel_movement.stride_y), dilation_x(kernel_movement.dilation_x), dilation_y(kernel_movement.dilation_y) {

			if (input_shape.input_size_x < 2 || input_shape.input_size_y < 2) {
				throw std::runtime_error("Shape of input must be at least 2! (otherwise use 1d convolution)");
			}
			if (input_layer->Neurons().size() != input_shape.input_size_x * input_shape.input_size_y) {
				throw std::runtime_error("input_shape is uncorrect for currect input size!");
			}
			if (kernel_shape.kernel_size_x < 2 && kernel_shape.kernel_size_y < 2 || !kernel_shape.kernel_size_x || !kernel_shape.kernel_size_y || !stride_x || !stride_y || !dilation_x || !dilation_y) {
				throw std::runtime_error("Zeros in stride, dilation and kernel_size < 2 is not allowed!");
			}
			if (kernel_shape.kernel_size_x == 1 && dilation_x != 1 || kernel_shape.kernel_size_y == 1 && dilation_y != 1) {
				throw std::runtime_error("For kernel_size==1, dilation must be ==1");
			}

			Init(kernel_shape.kernel_size_y, allow_input_kernel_underfit);
		}

		static inline NNB_ConvolutionEssence2d BuildPoolingSetup(interfaces::BasicLayerInterface *input_layer,
														  interfaces::BasicLayerInterface *output_layer,
														  unsigned input_shape_x,
														  KernelShape kernel_shape,
														  bool allow_input_kernel_underfit = false) {
			return NNB_ConvolutionEssence2d(input_layer, output_layer, input_shape_x, kernel_shape, KernelMovement(kernel_shape.kernel_size_x, kernel_shape.kernel_size_y, 1, 1), 1, allow_input_kernel_underfit);
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
			return kernel_area;
		}

		nn::interfaces::NBI *GetInputforLocation(unsigned place_id, unsigned weight_id) override {
			// It is assumed that the compiler will make the necessary optimizations (e.g. divmod)
			unsigned place_y = place_id / places_x_count;
			unsigned place_x = place_id % places_x_count;
			unsigned kernel_y = weight_id / kernel_x_size;
			unsigned kernel_x = weight_id % kernel_x_size;
			unsigned place_linear_offset = (place_y * stride_y + kernel_y * dilation_y) * width + place_x * stride_x + kernel_x * dilation_x;
			return input_layer->Neurons()[place_linear_offset];
		}

		nn::interfaces::NBI *GetOutputforLocation(unsigned place_id, unsigned kernel_id) override {
			return dynamic_cast<nn::interfaces::NBI *>(output_layer->Neurons()[kernel_id * places_count + place_id]);
		}

		unsigned GetKernelsCount() override {
			return kernels_count;
		}

		virtual std::vector<unsigned> CalcInputShape() override {
			return std::vector<unsigned> {width, input_layer->Neurons().size() / width, 1};
		}

		virtual std::vector<unsigned> CalcOutputShape() override {
			return std::vector<unsigned> {places_x_count, places_count / places_x_count, kernels_count};
		}

		KernelShape CalcKernelShape() const {
			return KernelShape(kernel_x_size, kernel_area / kernel_x_size);
		}
	};
}
