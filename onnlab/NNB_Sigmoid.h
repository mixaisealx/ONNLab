#pragma once
#include "NNBasicsInterfaces.h"
#include "BasicBackPropgI.h"
#include "LimitedNeuronI.h"
#include "BatchNeuronBasicI.h"

#include <cmath>
#include <algorithm>
#include <array>

namespace nn
{
	template<unsigned BATCH_SIZE, bool KahanErrorSummation = false> requires (BATCH_SIZE > 0)
	class NNB_SigmoidB : public interfaces::NeuronBasicInterface, public interfaces::BasicBackPropogableInterface, public interfaces::BatchNeuronBasicI, public interfaces::LimitedNeuronI {
		std::array<float, BATCH_SIZE> accumulator;
		std::array<float, BATCH_SIZE> backprop_error_accumulator;
		[[msvc::no_unique_address]] [[no_unique_address]] std::conditional<KahanErrorSummation, std::array<float, BATCH_SIZE>, std::monostate>::type backprop_error_accumulator_kahan_compensation;
		unsigned current_batch_size;
		std::vector<interfaces::ConnectionBasicInterface *> outputs;
		std::vector<interfaces::ConnectionBasicInterface *> inputs;

		const float MC_E = 2.71828182f;

		void AddInputConnection(interfaces::ConnectionBasicInterface *input) override {
			inputs.push_back(input);
		}

		void AddOutputConnection(interfaces::ConnectionBasicInterface *output) override {
			outputs.push_back(output);
		}

		void RemoveInputConnection(interfaces::ConnectionBasicInterface *input) override {
			inputs.erase(std::remove(inputs.begin(), inputs.end(), input), inputs.end());
		}

		void RemoveOutputConnection(interfaces::ConnectionBasicInterface *output) override {
			outputs.erase(std::remove(outputs.begin(), outputs.end(), output), outputs.end());
		}

		inline void InnerActivationFunction(float &x) const {
			x = 1.0f / (1.0f + std::powf(MC_E, -x));
		}

		NNB_SigmoidB(const NNB_SigmoidB &) = delete;
		NNB_SigmoidB &operator=(const NNB_SigmoidB &) = delete;
	public:
		NNB_SigmoidB(unsigned batch_size = BATCH_SIZE):current_batch_size(batch_size) {
			if (batch_size > BATCH_SIZE || batch_size == 0) throw std::exception("batch_size bigger than max batch size or is zero!");
			std::fill_n(accumulator.begin(), current_batch_size, 0.0f);
			std::fill_n(backprop_error_accumulator.begin(), current_batch_size, 0.0f);
			if constexpr (KahanErrorSummation) {
				std::fill_n(backprop_error_accumulator_kahan_compensation.begin(), current_batch_size, 0.0f);
			}
		}

		~NNB_SigmoidB() override {
			for (auto inp : inputs) {
				inp->~ConnectionBasicInterface();
			}
			for (auto out : outputs) {
				out->~ConnectionBasicInterface();
			}
		}

		const std::vector<interfaces::ConnectionBasicInterface *> &InputConnections() override {
			return inputs;
		}
		const std::vector<interfaces::ConnectionBasicInterface *> &OutputConnections() override {
			return outputs;
		}

		bool IsTrainable() override {
			return true;
		}

		float ActivationFunction(float x) const override {
			return 1.0f / (1.0f + std::powf(MC_E, -x));
		}

		float ActivationFunctionDerivative(float x) const override {
			return x * (1 - x);
		}

		void UpdateOwnLevel() override {
			std::fill_n(accumulator.begin(), current_batch_size, 0.0f);

			interfaces::NBI *nrn;
			for (const auto inp : inputs) {
				nrn = inp->From();
				for (unsigned channel = 0; channel != current_batch_size; ++channel) {
					accumulator[channel] += nrn->OwnLevel(channel) * inp->Weight();
				}
			}
			for (unsigned channel = 0; channel != current_batch_size; ++channel) {
				InnerActivationFunction(accumulator[channel]);
			}
		}

		float OwnLevel(unsigned channel = 0) override {
			return accumulator[channel];
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
				unsigned tmp;
				interfaces::BatchNeuronBasicI *nrn;
				for (auto elem : inputs) {
					nrn = dynamic_cast<interfaces::BatchNeuronBasicI *>(elem->From());
					tmp = (nrn ? nrn->GetCurrentBatchSize() : 1);
					if (tmp != batch_size && tmp != std::numeric_limits<unsigned>::max()) {
						throw std::exception("Different batch sizes (batch_size vs \"input layer\") is not allowed!");
					}
				}
				current_batch_size = batch_size;
				std::fill_n(accumulator.begin(), current_batch_size, 0.0f);
				std::fill_n(backprop_error_accumulator.begin(), current_batch_size, 0.0f);
				if constexpr (KahanErrorSummation) {
					std::fill_n(backprop_error_accumulator_kahan_compensation.begin(), current_batch_size, 0.0f);
				}
			} else
				throw std::exception("batch_size cannot be zero or greater than BATCH_SIZE!");
		}

		float UpperLimitValue() const override {
			return 1.0f;
		}

		float LowerLimitValue() const override {
			return 0.0f;
		}
	};

	using NNB_Sigmoid = NNB_SigmoidB<1, false>;
}
