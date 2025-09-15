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
	class NNB_ConvolutionEssence3d : public nn::interfaces::BasicConvolutionEssenceI {
		struct Size3d { unsigned x, y, z; };

		interfaces::BasicLayerInterface *input_layer, *output_layer;
		Size3d canvas_dims, places_dims, kernel_dims;
		Size3d stride, dilation;
		unsigned kernels_count, places_count, kernel_volume;
		unsigned batch_size;
		std::vector<interfaces::BasicConvolutionI *> subscribers;

		void AddBatchUpdateSubscriber(interfaces::BasicConvolutionI *subscriber) override {
			subscribers.push_back(subscriber);
		}

		void RemoveBatchUpdateSubscriber(interfaces::BasicConvolutionI *subscriber) override {
			subscribers.erase(std::remove(subscribers.begin(), subscribers.end(), subscriber), subscribers.end());
		}

		NNB_ConvolutionEssence3d(const NNB_ConvolutionEssence3d &) = delete;
		NNB_ConvolutionEssence3d &operator=(const NNB_ConvolutionEssence3d &) = delete;

		static inline void Assert_kernel_size(unsigned canval_sz, unsigned kernel_sz, unsigned stride, unsigned dilation, unsigned &places_count, bool allow_input_kernel_underfit) {
			unsigned emplaceable_size = canval_sz - (kernel_sz + (kernel_sz - 1) * (dilation - 1));
			places_count = emplaceable_size / stride + 1;
			if (!allow_input_kernel_underfit && emplaceable_size % stride) { // Check if layers have appropriate sizes
				throw std::runtime_error("That kernel_size and stride in not appropriate for input layer of such size!");
			}
		}

		inline void Init(bool allow_input_kernel_underfit) {
			RecalcBatchSize();

			for (auto elem : output_layer->Neurons()) {
				if (!dynamic_cast<nn::interfaces::InputNeuronI *>(elem) || !dynamic_cast<nn::interfaces::BasicBackPropogableInterface *>(elem)) {
					throw std::runtime_error("The output layer must consist of neurons based on interfaces::InputNeuronI!");
				}
			}

			Assert_kernel_size(canvas_dims.x, kernel_dims.x, stride.x, dilation.x, places_dims.x, allow_input_kernel_underfit);
			Assert_kernel_size(canvas_dims.y, kernel_dims.y, stride.y, dilation.y, places_dims.y, allow_input_kernel_underfit);
			Assert_kernel_size(canvas_dims.z, kernel_dims.z, stride.z, dilation.z, places_dims.z, allow_input_kernel_underfit);

			places_dims.y *= places_dims.x;
			places_dims.z *= places_dims.y;

			kernel_dims.y *= kernel_dims.x;
			kernel_dims.z *= kernel_dims.y;

			canvas_dims.y *= canvas_dims.x;
			canvas_dims.z *= canvas_dims.y;

			kernel_volume = kernel_dims.z;
			places_count = places_dims.z;
			if (places_count * kernels_count != output_layer->Neurons().size()) {
				throw std::runtime_error("The input layer in not appropriate for output layer of such size!");
			}
		}

	public:
		struct InputShape {
			unsigned input_size_x, input_size_y, input_size_z;
			InputShape(unsigned input_size_x, unsigned input_size_y, unsigned input_size_z): input_size_x(input_size_x), input_size_y(input_size_y), input_size_z(input_size_z) {}
			// Automatically calculates input_size_z
			InputShape(unsigned input_size_x, unsigned input_size_y): input_size_x(input_size_x), input_size_y(input_size_y), input_size_z(0) {}
		};

		struct KernelShape {
			unsigned kernel_size_x, kernel_size_y, kernel_size_z;
			KernelShape(unsigned cubic_kernel_size): kernel_size_x(cubic_kernel_size), kernel_size_y(cubic_kernel_size), kernel_size_z(cubic_kernel_size) {}
			KernelShape(unsigned kernel_size_x, unsigned kernel_size_y, unsigned kernel_size_z): kernel_size_x(kernel_size_x), kernel_size_y(kernel_size_y), kernel_size_z(kernel_size_z) {}
		};

		struct KernelMovement {
			unsigned stride_x, stride_y, stride_z, dilation_x, dilation_y, dilation_z;
			KernelMovement(unsigned stride = 1, unsigned dilation = 1): stride_x(stride), stride_y(stride), stride_z(stride), dilation_x(dilation), dilation_y(dilation), dilation_z(dilation) {}
			KernelMovement(unsigned stride_x, unsigned stride_y, unsigned stride_z, unsigned dilation_x, unsigned dilation_y, unsigned dilation_z): stride_x(stride_x), stride_y(stride_y), stride_z(stride_z),
				dilation_x(dilation_x), dilation_y(dilation_y), dilation_z(dilation_z) {}
		};

		NNB_ConvolutionEssence3d(interfaces::BasicLayerInterface *input_layer,
								 interfaces::BasicLayerInterface *output_layer,
								 InputShape input_shape,
								 KernelShape kernel_shape,
								 KernelMovement kernel_movement,
								 unsigned kernels_count = 1,
								 bool allow_input_kernel_underfit = false): input_layer(input_layer), output_layer(output_layer), kernels_count(kernels_count) {

			if (kernel_shape.kernel_size_x < 2 && kernel_shape.kernel_size_y < 2 && kernel_shape.kernel_size_z < 2 ||
				!kernel_shape.kernel_size_x || !kernel_shape.kernel_size_y || !kernel_shape.kernel_size_z ||
				!kernel_movement.stride_x || !kernel_movement.stride_y || !kernel_movement.stride_z ||
				!kernel_movement.dilation_x || !kernel_movement.dilation_y || !kernel_movement.dilation_z) {
				throw std::runtime_error("Zeros in stride, dilation and kernel_size < 2 is not allowed!");
			}
			if (kernel_shape.kernel_size_x == 1 && kernel_movement.dilation_x != 1 ||
				kernel_shape.kernel_size_y == 1 && kernel_movement.dilation_y != 1 ||
				kernel_shape.kernel_size_z == 1 && kernel_movement.dilation_z != 1) {
				throw std::runtime_error("For kernel_size==1, dilation must be ==1");
			}
			
			unsigned volume = input_shape.input_size_x * input_shape.input_size_y;
			if (!input_shape.input_size_z) {
				input_shape.input_size_z = input_layer->Neurons().size() / volume;
			}
			volume *= input_shape.input_size_z;

			if (input_layer->Neurons().size() != volume) {
				throw std::runtime_error("input_shape is uncorrect for currect input size!");
			}
			if (input_shape.input_size_x < 2 || input_shape.input_size_y < 2 || input_shape.input_size_z < 2) {
				throw std::runtime_error("Shape of input must be at least 2! (otherwise use 2d convolution)");
			}

			canvas_dims = Size3d{ input_shape.input_size_x, input_shape.input_size_y, input_shape.input_size_z };
			kernel_dims = Size3d{ kernel_shape.kernel_size_x, kernel_shape.kernel_size_y, kernel_shape.kernel_size_z };
			stride = Size3d{ kernel_movement.stride_x, kernel_movement.stride_y, kernel_movement.stride_z };
			dilation = Size3d{ kernel_movement.dilation_x, kernel_movement.dilation_y, kernel_movement.dilation_z };

			Init(allow_input_kernel_underfit);
		}

		static inline NNB_ConvolutionEssence3d BuildPoolingSetup(interfaces::BasicLayerInterface *input_layer,
																 interfaces::BasicLayerInterface *output_layer,
																 InputShape input_shape,
																 KernelShape kernel_shape,
																 bool allow_input_kernel_underfit = false) {
			return NNB_ConvolutionEssence3d(input_layer, output_layer, input_shape, kernel_shape, KernelMovement(kernel_shape.kernel_size_x, kernel_shape.kernel_size_y, kernel_shape.kernel_size_z, 1, 1, 1), 1, allow_input_kernel_underfit);
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
			return kernel_volume;
		}

		nn::interfaces::NBI *GetInputforLocation(unsigned place_id, unsigned weight_id) override {
			// It is assumed that the compiler will make the necessary optimizations (e.g. divmod)
			unsigned place_z = place_id / places_dims.y;
			unsigned place_xy = place_id % places_dims.y;
			unsigned place_y = place_xy / places_dims.x;
			unsigned place_x = place_xy % places_dims.x;

			unsigned kernel_z = weight_id / kernel_dims.y;
			unsigned kernel_xy = weight_id % kernel_dims.y;
			unsigned kernel_y = kernel_xy / kernel_dims.x;
			unsigned kernel_x = kernel_xy % kernel_dims.x;

			unsigned linear_offset_z = (place_z * stride.z + kernel_z * dilation.z) * canvas_dims.y;
			unsigned linear_offset_y = (place_y * stride.y + kernel_y * dilation.y) * canvas_dims.x;
			unsigned linear_offset_x = place_x * stride.x + kernel_x * dilation.x;

			return input_layer->Neurons()[linear_offset_z + linear_offset_y + linear_offset_x];
		}

		nn::interfaces::NBI *GetOutputforLocation(unsigned place_id, unsigned kernel_id) override {
			return dynamic_cast<nn::interfaces::NBI *>(output_layer->Neurons()[kernel_id * places_count + place_id]);
		}

		unsigned GetKernelsCount() override {
			return kernels_count;
		}

		virtual std::vector<unsigned> CalcInputShape() override {
			unsigned y = canvas_dims.y / canvas_dims.x;
			unsigned z = canvas_dims.z / y;
			return std::vector<unsigned> {canvas_dims.x, y, z, 1};
		}

		virtual std::vector<unsigned> CalcOutputShape() override {
			unsigned y = places_dims.y / places_dims.x;
			unsigned z = places_dims.z / y;
			return std::vector<unsigned> {places_dims.x, y, z, kernels_count};
		}

		KernelShape CalcKernelShape() const {
			unsigned y = kernel_dims.y / kernel_dims.x;
			unsigned z = kernel_dims.z / y;
			return KernelShape(kernel_dims.x, y, z);
		}
	};
}
