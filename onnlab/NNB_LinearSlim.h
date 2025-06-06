#pragma once
#include "NNBasicsInterfaces.h"
#include "BasicBackPropgI.h"
#include "CustomBackPropgI.h"
#include "SelectableInputI.h"

#include <array>
#include <vector>
#include <algorithm>

namespace nn
{
	template<unsigned BATCH_SIZE, bool KahanErrorSummation = false, bool do_mangle = false, float scale = 1.0f, float offset = 0.0f> requires (BATCH_SIZE > 0)
	class NNB_LinearSlimT : public interfaces::NeuronBasicInterface, public interfaces::BasicBackPropogableInterface, public interfaces::BatchNeuronBasicI {
		std::array<float, BATCH_SIZE> accumulator;
		std::array<float, BATCH_SIZE> backprop_error_accumulator;
		[[msvc::no_unique_address]] [[no_unique_address]] std::conditional<KahanErrorSummation, std::array<float, BATCH_SIZE>, std::monostate>::type backprop_error_accumulator_kahan_compensation;
		unsigned current_batch_size;
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

		inline void InnerMangleFunction(float &x) const {
			x = x * scale + offset;
		}

		NNB_LinearSlimT(const NNB_LinearSlimT &) = delete;
		NNB_LinearSlimT &operator=(const NNB_LinearSlimT &) = delete;
		public:
			NNB_LinearSlimT(unsigned batch_size = BATCH_SIZE):current_batch_size(batch_size) {
				if (batch_size > BATCH_SIZE || batch_size == 0) throw std::exception("batch_size bigger than max batch size or is zero!");
				std::fill_n(accumulator.begin(), current_batch_size, 0.0f);
				std::fill_n(backprop_error_accumulator.begin(), current_batch_size, 0.0f);
				if constexpr (KahanErrorSummation) {
					std::fill_n(backprop_error_accumulator_kahan_compensation.begin(), current_batch_size, 0.0f);
				}
			}

			~NNB_LinearSlimT() override {
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
				if constexpr (!do_mangle) {
					return x;
				} else {
					return x * scale + offset;
				}
			}

			float ActivationFunctionDerivative(float x) const override {
				if constexpr (!do_mangle) {
					return 1.0f;
				} else {
					return scale;
				}
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
				if constexpr (do_mangle) {
					for (unsigned channel = 0; channel != current_batch_size; ++channel) {
						InnerMangleFunction(accumulator[channel]);
					}
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
	};

	using NNB_LinearSlim = NNB_LinearSlimT<1, false>;
}
