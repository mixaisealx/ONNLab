#pragma once
#include "NNBasicsInterfaces.h"
#include "BasicBackPropgI.h"
#include "SelectableInputI.h"


namespace nn
{
	template<bool direction_is_asc>
	class NNB_m1hReLU : public interfaces::NeuronBasicInterface, public interfaces::MaccBackPropogableInterface, public interfaces::SelectableInputInterface {
		float accumulator;
		float output_value;
		float backprop_error_accumulator;
		float max_positive_out;
		std::vector<interfaces::ConnectionBasicInterface *> outputs;
		std::vector<interfaces::ConnectionBasicInterface *> inputs;

		float negative_multiplier;
		float nan_correcting_epsilon;

		bool is_backprop_enabled;
		
		
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

		NNB_m1hReLU(const NNB_m1hReLU &) = delete;
		NNB_m1hReLU &operator=(const NNB_m1hReLU &) = delete;
	public:
		const bool is_asc = direction_is_asc;

		NNB_m1hReLU(float negative_multiplier = 0.1f, float max_positive_out = 1.0f, float nan_correcting_epsilon = 1e-5f): negative_multiplier(negative_multiplier), max_positive_out(max_positive_out), nan_correcting_epsilon(nan_correcting_epsilon){
			output_value = accumulator = 0;
			backprop_error_accumulator = 0;
			is_backprop_enabled = true;
		}

		~NNB_m1hReLU() override {
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
			if constexpr (direction_is_asc) {
				if (x < -max_positive_out) { /* x is negaitive, this is "-x__" part */
					return negative_multiplier * (x + max_positive_out);
				} else if (x < 0) {
					return x + max_positive_out; /* this is "/" part */
				} else {
					return std::numeric_limits<float>::quiet_NaN();
				}
			} else {
				if (x > max_positive_out) { /* x is positive, this is "__+x" part */
					return -negative_multiplier * (x - max_positive_out);
				} else if (x >= 0) {
					return -x + max_positive_out; /* this is "\" part */
				} else {
					return -std::numeric_limits<float>::quiet_NaN();
				}
			}
		}

		float ActivationFunctionDerivative(float x) const override {
			if (is_backprop_enabled) {
				if constexpr (direction_is_asc) {
					if (x < -max_positive_out) { /* x is negaitive, this is "-x__" part */
						return negative_multiplier;
					} else if (x < 0) {
						return 1.0; /* this is "/" part */
					} else {
						return std::numeric_limits<float>::quiet_NaN();
					}
				} else {
					if (x > max_positive_out) { /* x is positive, this is "__+x" part */
						return -negative_multiplier;
					} else if (x >= 0) {
						return -1.0; /* this is "\" part */
					} else {
						return -std::numeric_limits<float>::quiet_NaN();
					}
				}
			}
			return 1.0f;
		}

		void UpdateOwnLevel() override {
			accumulator = 0; // 1 * bias_weight
			for (const auto inp : inputs) {
				accumulator += inp->From()->OwnLevel() * inp->Weight();
			}
			output_value = ActivationFunction(accumulator);
		}

		float OwnAccumulatorValue() override {
			return accumulator;
		}

		float OwnLevel() override {
			return output_value;
		}

		bool IsBackPropEnabled() override {
			return is_backprop_enabled;
		}

		void SetBackPropEnabledState(bool is_enabled) override {
			is_backprop_enabled = is_enabled;
		}

		void BackPropResetError() override {
			backprop_error_accumulator = 0;
		}

		void BackPropAccumulateError(float error) override {
			if (is_backprop_enabled) { // Selected as out
				backprop_error_accumulator += error;
			} else if (!std::isnan(output_value)) { // value is NOT NaN, but not selected for out - good time to make NaN
				Accumulator_make_NaN();
			}
		}

		float BackPropGetFinalError() override {
			if (is_backprop_enabled)
				return backprop_error_accumulator;
			
			return 0.0f;
		}

		float Accumulator_unNaN_distance() override {
			if constexpr (direction_is_asc) {
				if (accumulator < 0) {
					return 0.0f;
				} else {
					return accumulator;
				}
			} else {
				if (accumulator >= 0) {
					return 0.0f;
				} else {
					return -accumulator;
				}
			}
		}

		float Accumulator_NaN_distance() override {
			if constexpr (direction_is_asc) {
				if (accumulator < 0) {
					return -accumulator;
				} else {
					return 0.0f;
				}
			} else {
				if (accumulator >= 0) {
					return accumulator;
				} else {
					return 0.0f;
				}
			}
		}

		void Accumulator_make_NaN() override {
			float unweighted_summ = 0;
			for (const auto inp : inputs) {
				unweighted_summ += inp->From()->OwnLevel();
			}
			if (unweighted_summ > 1e-10f) { // Check if inputs are not 0
				if constexpr (direction_is_asc) { // Need to set value small bigger than 0 
					float weights_delta = (accumulator - nan_correcting_epsilon) / unweighted_summ; // nan_correcting_epsilon is some "epsilon" for guaratees
					for (auto inp : inputs) {
						inp->Weight(inp->Weight() - weights_delta);
					}
				} else { // Need to set value small lesser than 0 
					float weights_delta = (accumulator + nan_correcting_epsilon) / unweighted_summ; // nan_correcting_epsilon is some "epsilon" for guaratees
					for (auto inp : inputs) {
						inp->Weight(inp->Weight() - weights_delta);
					}
				}
			}
		}

		void Accumulator_make_unNaN() override {
			float unweighted_summ = 0;
			for (const auto inp : inputs) {
				unweighted_summ += inp->From()->OwnLevel();
			}
			if (unweighted_summ > 1e-10f) { // Check if inputs are not 0
				if constexpr (direction_is_asc) { // Need to set value small lesser than 0 
					float weights_delta = (accumulator + nan_correcting_epsilon) / unweighted_summ; // nan_correcting_epsilon is some "epsilon" for guaratees
					for (auto inp : inputs) {
						inp->Weight(inp->Weight() - weights_delta);
					}
				} else { // Need to set value small bigger than 0 
					float weights_delta = (accumulator - nan_correcting_epsilon) / unweighted_summ; // nan_correcting_epsilon is some "epsilon" for guaratees
					for (auto inp : inputs) {
						inp->Weight(inp->Weight() - weights_delta);
					}
				}
			}
		}
	};
}
