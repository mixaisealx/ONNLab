#pragma once
#include "NNBasicsInterfaces.h"
#include "InputNeuronI.h"
#include "BatchNeuronBasicI.h"

#include <array>
#include <algorithm>
#include <functional>
#include <stdexcept>

namespace nn
{
	template<unsigned BATCH_SIZE> requires (BATCH_SIZE > 0)
	class NNB_InputB : public interfaces::NeuronBasicInterface, public interfaces::InputNeuronI, public interfaces::BatchNeuronBasicI {
		std::array<float *, BATCH_SIZE> values_source;
		unsigned current_batch_size;
		std::vector<interfaces::ConnectionBasicInterface *> outputs;

		void AddInputConnection(interfaces::ConnectionBasicInterface *) override {
			throw std::runtime_error("Logic error! InputConnection on Input neuron!");
		}

		void RemoveInputConnection(interfaces::ConnectionBasicInterface *) override {
			throw std::runtime_error("Logic error! InputConnection on Input neuron!");
		}

		const std::vector<interfaces::ConnectionBasicInterface *> &InputConnections() override {
			throw std::runtime_error("Logic error! InputConnection on Input neuron!");
		}

		float ActivationFunction(float x) const override {
			return x;
		}

		float ActivationFunctionDerivative(float x) const override {
			return 1;
		}

		void AddOutputConnection(interfaces::ConnectionBasicInterface *output) override {
			outputs.push_back(output);
		}

		void RemoveOutputConnection(interfaces::ConnectionBasicInterface *output) override {
			outputs.erase(std::remove(outputs.begin(), outputs.end(), output), outputs.end());
		}

		NNB_InputB(const NNB_InputB &) = delete;
		NNB_InputB &operator=(const NNB_InputB &) = delete;
		public:
			NNB_InputB(float values_storage_array[], unsigned count = 1) {
				if (count > BATCH_SIZE || count == 0) throw std::runtime_error("Batch initializer bigger than batch size or zero!");
				current_batch_size = count;
				for (unsigned i = 0; i != count; ++i) {
					values_source[i] = values_storage_array + i;
				}
			}

			NNB_InputB(std::initializer_list<float *> values_sources) {
				if (values_sources.size() > BATCH_SIZE || values_sources.size() == 0) throw std::runtime_error("Batch initializer bigger than batch size or zero!");
				current_batch_size = values_sources.size();
				std::copy(values_sources.begin(), values_sources.end(), values_source.begin());
			}


			NNB_InputB(std::function<void(float **storage, unsigned capacity, unsigned &used_capacity)> initializer) {
				unsigned count;
				initializer(values_source.data(), BATCH_SIZE, count);
				if (count > BATCH_SIZE || count == 0) throw std::runtime_error("Batch initializer bigger than batch size or zero!");
				current_batch_size = count;
			}
			

			~NNB_InputB() override {
				for (auto out : outputs) {
					out->~ConnectionBasicInterface();
				}
			}

			const std::vector<interfaces::ConnectionBasicInterface *> &OutputConnections() override {
				return outputs;
			}

			float *ValueStoragePtr(unsigned channel = 0) {
				return values_source[channel];
			}

			void ValueStoragePtr(float *value_storage, unsigned channel = 0) {
				this->values_source[channel] = value_storage;
			}

			void UpdateOwnLevel() override {}

			void SetOwnLevel(float value, unsigned channel = 0) override {
				*values_source[channel] = value;
			}

			float OwnLevel(unsigned channel = 0) override {
				return *values_source[channel];
			}

			bool IsTrainable() override {
				return false;
			}

			unsigned GetMaxBatchSize() override {
				return BATCH_SIZE;
			}

			unsigned GetCurrentBatchSize() override {
				return current_batch_size;
			}

			void SetCurrentBatchSize(unsigned batch_size) override {
				if (batch_size && batch_size <= BATCH_SIZE)
					current_batch_size = batch_size;
				else
					throw std::runtime_error("batch_size cannot be zero or greater than BATCH_SIZE!");
			}
	};

	using NNB_Input = NNB_InputB<1>;
}
