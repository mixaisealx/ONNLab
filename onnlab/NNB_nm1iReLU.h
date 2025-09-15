#pragma once
#include "NNBasicsInterfaces.h"
#include "BasicBackPropgI.h"
#include "LimitedNeuronI.h"
#include "BatchNeuronBasicI.h"

#include <array>
#include <limits>
#include <type_traits>
#include <stdexcept>
#include <variant>

namespace nn
{
	// non-monotonic (1) immortal ReLU
	template<unsigned BATCH_SIZE, bool KahanErrorSummation = false, bool StoreRealAccumulator = false> requires (BATCH_SIZE > 0)
	class NNB_nm1iReLUb : public interfaces::NeuronBasicInterface, public interfaces::MaccBackPropogableInterface, public interfaces::ZeroGradBackPropogableInterface, public interfaces::LimitedNeuronI, public interfaces::BatchNeuronBasicI {
		std::array<float, BATCH_SIZE> accumulator;
		std::array<float, BATCH_SIZE> backprop_error_accumulator;
		[[msvc::no_unique_address]] [[no_unique_address]] std::conditional<KahanErrorSummation, std::array<float, BATCH_SIZE>, std::monostate>::type backprop_error_accumulator_kahan_compensation;
		std::array<typename std::conditional<StoreRealAccumulator, float, uint8_t>::type, BATCH_SIZE> accumulator_backup;
		unsigned current_batch_size;
		float max_positive_out;
		float hidden_derivative, hidden_error_backprop_factor, backprop_overkill_factor;
		std::vector<interfaces::ConnectionBasicInterface *> outputs;
		std::vector<interfaces::ConnectionBasicInterface *> inputs;

		[[msvc::no_unique_address]] [[no_unique_address]] std::conditional<StoreRealAccumulator, std::monostate, std::array<float, 4>>::type surrogate_precalc_table;
		std::array<float, 2> batch_analyzer_field_activation_min_distance;
		std::array<std::pair<const void *, unsigned>, 2> batch_analyzer_field_activation_min_distance_drop_on;
		bool batch_analyzer_disabled;
		const void **batch_analyzer_payload;
		const void *batch_analyzer_payload_plug = nullptr;

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

		inline void SuroCalcActivationFunction(float &accum, std::conditional<StoreRealAccumulator, float, uint8_t>::type &rsacc) const {
			// -x _/0\_ x
			//std::max(max_positive_out-abs(x), 0)
			if constexpr (StoreRealAccumulator) {
				rsacc = accum;
				if (accum > -max_positive_out && accum < max_positive_out) {
					accum = (accum < 0 ? accum : -accum) + max_positive_out; /* this is "/\" part */
				} else {
					accum = 0.0f;
				}
			} else {
				if (accum < -max_positive_out) { /* x is negaitive, this is "-x__" part */
					rsacc = 0;
					accum = 0;
				} else if (accum > max_positive_out) { /* x is positive, this is "__+x" part */
					rsacc = 3;
					accum = 0;
				} else if (accum < 0) {
					rsacc = 1;
					accum = accum + max_positive_out; /* this is "/" part */
				} else {
					rsacc = 2;
					accum = -accum + max_positive_out; /* this is "\" part */
				}
			}
		}

		NNB_nm1iReLUb(const NNB_nm1iReLUb &) = delete;
		NNB_nm1iReLUb &operator=(const NNB_nm1iReLUb &) = delete;
		public:
			NNB_nm1iReLUb(float max_positive_out = 1.0f, float hidden_derivative = 0.01f, float hidden_error_backprop_factor = 1.0f, float backprop_overkill_factor = 0.0f, unsigned batch_size = BATCH_SIZE):
				max_positive_out(max_positive_out),
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
				if constexpr (StoreRealAccumulator) {
					std::fill_n(accumulator_backup.begin(), current_batch_size, 0.0f);
				} else {
					std::fill_n(accumulator_backup.begin(), current_batch_size, 0);
					surrogate_precalc_table[3] = max_positive_out + 1.0f; /* x is positive, this is "__+x" part */
					surrogate_precalc_table[0] = -surrogate_precalc_table[3]; /* x is negaitive, this is "-x__" part */
					surrogate_precalc_table[2] = max_positive_out * 0.5f; /* this is "\" part */
					surrogate_precalc_table[1] = -surrogate_precalc_table[2]; /* this is "/" part */
				}
				batch_analyzer_disabled = true;
				BatchAnalyzer_Reset();
			}

			~NNB_nm1iReLUb() override {
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
				// -x _/0\_ x
				//std::max(max_positive_out-abs(x), 0.0f)
				if (x > -max_positive_out && x < max_positive_out) {
					return (x < 0 ? x : -x) + max_positive_out; /* this is "/\" part */
				} else {
					return 0.0f;
				}
			}

			float ActivationFunctionDerivative(float x) const override {
				if (x > -max_positive_out && x < max_positive_out) {
					return (x < 0 ? 1.0f : -1.0f); /* this is "/\" part */
				} else {
					return 0.0f;
				}
			}

			float HiddenActivationFunctionDerivative(float x, float backprop_error) const override {
				if (x > -max_positive_out) {
					if (x < max_positive_out) {
						return (x < 0 ? 1.0f : -1.0f); /* this is "/\" part */
					} else { // x > max_positive_out
						return -hidden_derivative * (backprop_error < 0.0f ? 1.0f : backprop_overkill_factor); // x >= max_positive_out && backprop_error > 0 -> "zeroing" zero (moving to x-positive field making "much zero")
					}
				} else { // x < -max_positive_out
					return hidden_derivative * (backprop_error < 0.0f ? 1.0f : backprop_overkill_factor); // x <= -max_positive_out && backprop_error > 0 -> "zeroing" zero (moving to x-negative field making "much zero")
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

				if (batch_analyzer_disabled) { // Batch analyzer is not enabled (common case)
					for (unsigned channel = 0; channel != current_batch_size; ++channel) {
						SuroCalcActivationFunction(accumulator[channel], accumulator_backup[channel]);
					}
				} else { // Batch analyzer is enabled
					for (unsigned channel = 0; channel != current_batch_size; ++channel) {
						if (accumulator[channel] < 0) {
							if (batch_analyzer_field_activation_min_distance[0] > 0.0f) {
								batch_analyzer_field_activation_min_distance[0] = 0.0f;
								batch_analyzer_field_activation_min_distance_drop_on[0].first = *batch_analyzer_payload;
								batch_analyzer_field_activation_min_distance_drop_on[0].second = channel;
							}
							if (batch_analyzer_field_activation_min_distance[1] > -accumulator[channel]) {
								batch_analyzer_field_activation_min_distance[1] = -accumulator[channel];
								batch_analyzer_field_activation_min_distance_drop_on[1].first = *batch_analyzer_payload;
								batch_analyzer_field_activation_min_distance_drop_on[1].second = channel;
							}
						} else {
							if (batch_analyzer_field_activation_min_distance[1] > 0.0f) {
								batch_analyzer_field_activation_min_distance[1] = 0.0f;
								batch_analyzer_field_activation_min_distance_drop_on[1].first = *batch_analyzer_payload;
								batch_analyzer_field_activation_min_distance_drop_on[1].second = channel;
							}
							if (batch_analyzer_field_activation_min_distance[0] > accumulator[channel]) {
								batch_analyzer_field_activation_min_distance[0] = accumulator[channel];
								batch_analyzer_field_activation_min_distance_drop_on[0].first = *batch_analyzer_payload;
								batch_analyzer_field_activation_min_distance_drop_on[0].second = channel;
							}
						}
						SuroCalcActivationFunction(accumulator[channel], accumulator_backup[channel]);
					}
				}
			}

			float RealAccumulatorValue(unsigned data_channel = 0) {
				if constexpr (StoreRealAccumulator) {
					return accumulator_backup[data_channel];
				} else {
					throw std::runtime_error("RealAccumulatorValue is not enabled! (StoreRealAccumulator == false)");
				}
			}

			float SurrogateAccumulatorValue(unsigned channel = 0) override {
				if constexpr (StoreRealAccumulator) {
					return accumulator_backup[channel];
				} else {
					return surrogate_precalc_table[accumulator_backup[channel]];
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
				return std::abs(accumulator_value) < max_positive_out ? 1.0f : hidden_error_backprop_factor;
			}

			void BatchAnalyzer_SetState(bool is_enabled) {
				batch_analyzer_disabled = !is_enabled;
			}

			// Usage Tip: 
			// If you see a value of 0, it means that the corresponding field has been activated at one of the inputs and 
			// additional correction of the relative value of this field is not required. Otherwise, you can "transfer" 
			// the most suitable data output to this field (the most suitable output is the one where the last decrease 
			// in the number occurred) using a special function.
			// Caution: the weights should not be adjusted during the batch, otherwise the statistics may not be representative!
			const std::array<float, 2> &BatchAnalyzer_GetFieldsActivateMinDistance() const {
				return batch_analyzer_field_activation_min_distance;
			}

			// Usage Tip: 
			// If you have set the value of the data pointer storage, then pointers to data sets will be available 
			// in this array, for which the distance of transferring the value to another area is minimal.
			const std::array<std::pair<const void *, unsigned>, 2> &BatchAnalyzer_GetFieldsActivateMinDistanceDropOnPayload() const {
				return batch_analyzer_field_activation_min_distance_drop_on;
			}

			void BatchAnalyzer_Reset() {
				batch_analyzer_field_activation_min_distance[1] = batch_analyzer_field_activation_min_distance[0] = std::numeric_limits<float>::max();
				batch_analyzer_field_activation_min_distance_drop_on[1].first = batch_analyzer_field_activation_min_distance_drop_on[0].first = nullptr;
				batch_analyzer_field_activation_min_distance_drop_on[1].second = batch_analyzer_field_activation_min_distance_drop_on[0].second = 0;
				batch_analyzer_payload = &batch_analyzer_payload_plug;
			}

			void BatchAnalyzer_SetDataPayloadPtrSource(const void **payload_source) {
				batch_analyzer_payload = payload_source;
			}

			void FlipCurrentInputToAnotherField(unsigned relative_channel = 0, float nan_correcting_epsilon = 1e-5f) {
				float unweighted_summ = 0.0f;
				float accum = 0.0f;
				{
					float tmp;
					for (const auto inp : inputs) {
						tmp = inp->From()->OwnLevel(relative_channel);
						unweighted_summ += tmp;
						accum += tmp * inp->Weight();
					}
				}

				if (unweighted_summ > 1e-10f) { // Check if inputs are not 0
					if (accum < 0) { // Flip to positive field
						// Need to set value small bigger than 0 
						float weights_delta = (accum - nan_correcting_epsilon) / unweighted_summ; // nan_correcting_epsilon is some "epsilon" for guaratees
						for (auto inp : inputs) {
							inp->Weight(inp->Weight() - weights_delta);
						}
					} else { // Flip to negative field
						// Need to set value small lesser than 0 
						float weights_delta = (accum + nan_correcting_epsilon) / unweighted_summ; // nan_correcting_epsilon is some "epsilon" for guaratees
						for (auto inp : inputs) {
							inp->Weight(inp->Weight() - weights_delta);
						}
					}
				}
			}

			float UpperLimitValue() const override {
				return max_positive_out;
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
					if constexpr (StoreRealAccumulator) {
						std::fill_n(accumulator_backup.begin(), current_batch_size, 0.0f);
					} else {
						std::fill_n(accumulator_backup.begin(), current_batch_size, 0);
					}
					std::fill_n(accumulator.begin(), current_batch_size, 0.0f);
					std::fill_n(backprop_error_accumulator.begin(), current_batch_size, 0.0f);
					if constexpr (KahanErrorSummation) {
						std::fill_n(backprop_error_accumulator_kahan_compensation.begin(), current_batch_size, 0.0f);
					}
				} else
					throw std::runtime_error("batch_size cannot be zero or greater than BATCH_SIZE!");
			}
	};

	using NNB_nm1iReLU = NNB_nm1iReLUb<1, false>;
}
