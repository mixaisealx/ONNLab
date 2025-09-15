#pragma once
#include "NNBasicsInterfaces.h"
#include "InputNeuronI.h"
#include "BasicBackPropgI.h"
#include "BatchNeuronBasicI.h"

#include <array>
#include <vector>
#include <algorithm>
#include <type_traits>
#include <stdexcept>
#include <variant>

namespace nn
{
	template<unsigned BATCH_SIZE, bool KahanErrorSummation = false> requires (BATCH_SIZE > 0)
	class NNB_StorageB : public interfaces::NeuronBasicInterface, public interfaces::BasicBackPropogableInterface, public interfaces::InputNeuronI, public interfaces::BatchNeuronBasicI {
		std::array<float, BATCH_SIZE> value;
		std::array<float, BATCH_SIZE> backprop_error_accumulator;
		[[msvc::no_unique_address]] [[no_unique_address]] std::conditional<KahanErrorSummation, std::array<float, BATCH_SIZE>, std::monostate>::type backprop_error_accumulator_kahan_compensation;
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

		void AddOutputConnection(interfaces::ConnectionBasicInterface *output) override {
			outputs.push_back(output);
		}

		void RemoveOutputConnection(interfaces::ConnectionBasicInterface *output) override {
			outputs.erase(std::remove(outputs.begin(), outputs.end(), output), outputs.end());
		}

		NNB_StorageB(const NNB_StorageB &) = delete;
		NNB_StorageB &operator=(const NNB_StorageB &) = delete;
	public:
		NNB_StorageB(unsigned batch_size = BATCH_SIZE):current_batch_size(batch_size) {
			if (batch_size > BATCH_SIZE || batch_size == 0) throw std::runtime_error("batch_size bigger than max batch size or is zero!");
			std::fill_n(value.begin(), current_batch_size, 0.0f);
			std::fill_n(backprop_error_accumulator.begin(), current_batch_size, 0.0f);
			if constexpr (KahanErrorSummation) {
				std::fill_n(backprop_error_accumulator_kahan_compensation.begin(), current_batch_size, 0.0f);
			}
		}

		~NNB_StorageB() override {
			for (auto out : outputs) {
				out->~ConnectionBasicInterface();
			}
		}

		const std::vector<interfaces::ConnectionBasicInterface *> &OutputConnections() override {
			return outputs;
		}

		bool IsTrainable() override {
			return true;
		}

		float ActivationFunction(float x) const override {
			return x;
		}

		float ActivationFunctionDerivative(float x) const override {
			return 1.0f;
		}

		void UpdateOwnLevel() override {
		}

		float OwnLevel(unsigned channel = 0) override {
			return value[channel];
		}

		void SetOwnLevel(float value, unsigned channel = 0) override {
			this->value[channel] = value;
		}

		void BackPropResetError() override {
			std::fill_n(backprop_error_accumulator.begin(), current_batch_size, 0.0f);
			if constexpr (KahanErrorSummation) {
				std::fill_n(backprop_error_accumulator_kahan_compensation.begin(), current_batch_size, 0.0f);
			}
		}

		void BackPropAccumulateError(float error, unsigned channel = 0) override {
			if constexpr (KahanErrorSummation) {
				float &sum = backprop_error_accumulator[channel];
				float &compensation = backprop_error_accumulator_kahan_compensation[channel];
				float y = error - compensation;
				float t = sum + y;
				compensation = (t - sum) - y;
				sum = t;
			} else {
				backprop_error_accumulator[channel] += error;
			}
		}

		float BackPropGetFinalError(unsigned channel = 0) override {
			return backprop_error_accumulator[channel];
		}

		unsigned GetMaxBatchSize() override {
			return BATCH_SIZE;
		}

		unsigned GetCurrentBatchSize() override {
			return current_batch_size;
		}

		void SetCurrentBatchSize(unsigned batch_size) override {
			if (batch_size && batch_size <= BATCH_SIZE) {
				current_batch_size = batch_size;
				std::fill_n(value.begin(), current_batch_size, 0.0f);
				std::fill_n(backprop_error_accumulator.begin(), current_batch_size, 0.0f);
				if constexpr (KahanErrorSummation) {
					std::fill_n(backprop_error_accumulator_kahan_compensation.begin(), current_batch_size, 0.0f);
				}
			} else
				throw std::runtime_error("batch_size cannot be zero or greater than BATCH_SIZE!");
		}
	};

	using NNB_Storage = NNB_StorageB<1, false>;
}
