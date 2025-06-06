#pragma once
#include "NNBasicsInterfaces.h"
#include "BasicBackPropgI.h"
#include "SelectableInputI.h"


namespace nn
{
	template<bool direction_is_asc>
	class NNB_m1h_sqReLU : public interfaces::NeuronBasicInterface, public interfaces::MaccBackPropogableInterface {
		float accumulator;
		float output_value;
		float backprop_error_accumulator;
		float max_positive_out;
		std::vector<interfaces::ConnectionBasicInterface *> outputs;
		std::vector<interfaces::ConnectionBasicInterface *> inputs;

		float negative_multiplier;

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

		NNB_m1h_sqReLU(const NNB_m1h_sqReLU &) = delete;
		NNB_m1h_sqReLU &operator=(const NNB_m1h_sqReLU &) = delete;
	public:
		const bool is_asc = direction_is_asc;

		NNB_m1h_sqReLU(float negative_multiplier = 0.1f, float max_positive_out = 1.0f): negative_multiplier(negative_multiplier), max_positive_out(max_positive_out) {
			output_value = accumulator = 0;
			backprop_error_accumulator = 0;
		}

		~NNB_m1h_sqReLU() override {
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
					return (x + max_positive_out) * (x + max_positive_out);
				}
			} else {
				if (x > max_positive_out) { /* x is positive, this is "__+x" part */
					return -negative_multiplier * (x - max_positive_out);
				} else if (x >= 0) {
					return -x + max_positive_out; /* this is "\" part */
				} else {
					return (x - max_positive_out) * (x - max_positive_out);
				}
			}
		}

		float ActivationFunctionDerivative(float x) const override {
			if constexpr (direction_is_asc) {
				if (x < -max_positive_out) { /* x is negaitive, this is "-x__" part */
					return negative_multiplier;
				} else if (x < 0) {
					return 1.0; /* this is "/" part */
				} else {
					return 2 * (x + max_positive_out);
				}
			} else {
				if (x > max_positive_out) { /* x is positive, this is "__+x" part */
					return -negative_multiplier;
				} else if (x >= 0) {
					return -1.0; /* this is "\" part */
				} else {
					return 2 * (x - max_positive_out);
				}
			}
		}

		void UpdateOwnLevel() override {
			accumulator = 0; // 1 * bias_weight
			for (const auto inp : inputs) {
				accumulator += inp->From()->OwnLevel() * inp->Weight();
			}
			output_value = ActivationFunction(accumulator);
		}

		float RealAccumulatorValue(unsigned _ = 0) {
			return accumulator;
		}

		float SurrogateAccumulatorValue(unsigned _ = 0) override {
			return accumulator;
		}

		float OwnLevel(unsigned _ = 0) override {
			return output_value;
		}

		void BackPropResetError() override {
			backprop_error_accumulator = 0;
		}

		void BackPropAccumulateError(float error, unsigned _ = 0) override {
			backprop_error_accumulator += error;
		}

		float BackPropGetFinalError(unsigned _ = 0) override {
			return backprop_error_accumulator;
		}
	};
}
