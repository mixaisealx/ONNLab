#pragma once
#include "NNBasicsInterfaces.h"
#include "BasicBackPropgI.h"
#include "LimitedNeuronI.h"
#include "BatchNeuronBasicI.h"

#include <algorithm>
#include <array>
#include <limits>
#include <type_traits>
#include <stdexcept>
#include <variant>

namespace nn
{
	// immortal ReLU
	template<unsigned BATCH_SIZE, bool KahanErrorSummation = false> requires (BATCH_SIZE > 0)
	class NNB_iReLUb : public interfaces::NeuronBasicInterface, public interfaces::ZeroGradBackPropogableInterface, public interfaces::LimitedNeuronI, public interfaces::BatchNeuronBasicI {
		std::array<float, BATCH_SIZE> accumulator;
		std::array<float, BATCH_SIZE> backprop_error_accumulator;
		[[msvc::no_unique_address]] [[no_unique_address]] std::conditional<KahanErrorSummation, std::array<float, BATCH_SIZE>, std::monostate>::type backprop_error_accumulator_kahan_compensation;
		unsigned current_batch_size;
		float hidden_derivative, hidden_error_backprop_factor, backprop_overkill_factor;
		std::vector<interfaces::ConnectionBasicInterface *> outputs;
		std::vector<interfaces::ConnectionBasicInterface *> inputs;

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

		static inline void InnerActivationFunction(float &x) {
			x = (x > 0.0f ? x : 0.0f); // f(0) == 0
		}

		NNB_iReLUb(const NNB_iReLUb &) = delete;
		NNB_iReLUb &operator=(const NNB_iReLUb &) = delete;
		public:
			NNB_iReLUb(float hidden_derivative = 0.01f, float hidden_error_backprop_factor = 1.0f, float backprop_overkill_factor = 0.0f, unsigned batch_size = BATCH_SIZE):
				hidden_derivative(hidden_derivative),
				hidden_error_backprop_factor(hidden_error_backprop_factor),
				backprop_overkill_factor(backprop_overkill_factor),
				current_batch_size(batch_size) {
				if (batch_size > BATCH_SIZE || batch_size == 0) throw std::runtime_error("batch_size bigger than max batch size or is zero!");
				std::fill_n(accumulator.begin(), current_batch_size, 0.0f);
				std::fill_n(backprop_error_accumulator.begin(), current_batch_size, 0.0f);
				if constexpr (KahanErrorSummation) {
					std::fill_n(backprop_error_accumulator_kahan_compensation.begin(), current_batch_size, 0.0f);
				}
			}

			~NNB_iReLUb() override {
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
				return (x > 0.0f ? x : 0.0f); // f(0) == 0
			}

			float ActivationFunctionDerivative(float x) const override {
				return (x > 0.0f ? 1.0f : 0.0f); // df(0) == 0.0f
			}

			float HiddenActivationFunctionDerivative(float x, float backprop_error) const override {
				return (x > 0.0f ? 1.0f : hidden_derivative * (backprop_error < 0.0f ? 1.0f : backprop_overkill_factor)); // x <= 0 && backprop_error > 0 -> "zeroing" zero (moving to x-negative field making "much zero")
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

			float BackPropErrorFactor(float accumulator_value) override {
				return accumulator_value > 0.0f ? 1.0f : hidden_error_backprop_factor;
			}

			float UpperLimitValue() const override {
				return std::numeric_limits<float>::infinity();
			}

			float LowerLimitValue() const override {
				return 0.0f;
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
							throw std::runtime_error("Different batch sizes (batch_size vs \"input layer\") is not allowed!");
						}
					}
					current_batch_size = batch_size;
					std::fill_n(accumulator.begin(), current_batch_size, 0.0f);
					std::fill_n(backprop_error_accumulator.begin(), current_batch_size, 0.0f);
					if constexpr (KahanErrorSummation) {
						std::fill_n(backprop_error_accumulator_kahan_compensation.begin(), current_batch_size, 0.0f);
					}
				} else
					throw std::runtime_error("batch_size cannot be zero or greater than BATCH_SIZE!");
			}
	};

	using NNB_iReLU = NNB_iReLUb<1, false>;
}
