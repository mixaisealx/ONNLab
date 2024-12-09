#pragma once
#include "NNBasicsInterfaces.h"
#include "BasicBackPropgI.h"

#include <array>


namespace nn
{
	class NNB_m1ReLU : public interfaces::NeuronBasicInterface, public interfaces::MaccBackPropogableInterface {
		float output_value;
		float accumulator;
		float backprop_error_accumulator;
		float max_positive_out;
		std::vector<interfaces::ConnectionBasicInterface *> outputs;
		std::vector<interfaces::ConnectionBasicInterface *> inputs;

		float negative_multiplier;
		float nan_correcting_epsilon;

		std::array<float, 2> batch_analyzer_feild_activation_min_distance;
		std::array<const void*, 2> batch_analyzer_feild_activation_min_distance_drop_on;
		bool batch_analyzer_enabled;
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

		NNB_m1ReLU(const NNB_m1ReLU &) = delete;
		NNB_m1ReLU &operator=(const NNB_m1ReLU &) = delete;
	public:
		NNB_m1ReLU(float negative_multiplier = 0.1f, float max_positive_out = 1.0f, float nan_correcting_epsilon = 1e-5f):negative_multiplier(negative_multiplier), max_positive_out(max_positive_out), nan_correcting_epsilon(nan_correcting_epsilon) {
			output_value = accumulator = 0;
			backprop_error_accumulator = 0;
			batch_analyzer_enabled = false;
			BatchAnalyzer_Reset();
		}

		~NNB_m1ReLU() override {
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
			//std::max(max_positive_out-abs(x), -0.1f*(max_positive_out-abs(x)))
			if (x < -max_positive_out) { /* x is negaitive, this is "-x__" part */
				return negative_multiplier * (x + max_positive_out);
			} else if (x > max_positive_out) { /* x is positive, this is "__+x" part */
				return -negative_multiplier * (x - max_positive_out);
			} else {
				return (x < 0 ? x : -x) + max_positive_out; /* this is "/\" part */
			}
		}

		float ActivationFunctionDerivative(float x) const override {
			if (x < -max_positive_out) { /* x is negaitive, this is "-x__" part */
				return negative_multiplier;
			} else if (x > max_positive_out) { /* x is positive, this is "__+x" part */
				return -negative_multiplier;
			} else {
				return (x < 0 ? 1.0f : -1.0f); /* this is "/\" part */
			}
		}

		void UpdateOwnLevel() override {
			accumulator = 0; // 1 * bias_weight
			for (const auto inp : inputs) {
				accumulator += inp->From()->OwnLevel() * inp->Weight();
			}
			output_value = ActivationFunction(accumulator);
			if (batch_analyzer_enabled) {
				if (accumulator < 0) {
					if (batch_analyzer_feild_activation_min_distance[0] > 0.0f) {
						batch_analyzer_feild_activation_min_distance[0] = 0.0f;
						batch_analyzer_feild_activation_min_distance_drop_on[0] = *batch_analyzer_payload;
					}
					if (batch_analyzer_feild_activation_min_distance[1] > -accumulator) {
						batch_analyzer_feild_activation_min_distance[1] = -accumulator;
						batch_analyzer_feild_activation_min_distance_drop_on[1] = *batch_analyzer_payload;
					}
				} else {
					if (batch_analyzer_feild_activation_min_distance[1] > 0.0f) {
						batch_analyzer_feild_activation_min_distance[1] = 0.0f;
						batch_analyzer_feild_activation_min_distance_drop_on[1] = *batch_analyzer_payload;
					}
					if (batch_analyzer_feild_activation_min_distance[0] > accumulator) {
						batch_analyzer_feild_activation_min_distance[0] = accumulator;
						batch_analyzer_feild_activation_min_distance_drop_on[0] = *batch_analyzer_payload;
					}
				}
			}
		}

		float OwnAccumulatorValue() override {
			return accumulator;
		}

		float OwnLevel() override {
			return output_value;
		}

		void BackPropResetError() override {
			backprop_error_accumulator = 0;
		}

		void BackPropAccumulateError(float error) override {
			backprop_error_accumulator += error;
		}

		float BackPropGetFinalError() override {
			return backprop_error_accumulator;
		}

		void BatchAnalyzer_SetState(bool is_enabled) {
			batch_analyzer_enabled = is_enabled;
		}

		// Usage Tip: 
		// If you see a value of 0, it means that the corresponding feild has been activated at one of the inputs and 
		// additional correction of the relative value of this feild is not required. Otherwise, you can "transfer" 
		// the most suitable data output to this feild (the most suitable output is the one where the last decrease 
		// in the number occurred) using a special function.
		// Caution: the weights should not be adjusted during the batch, otherwise the statistics may not be representative!
		const std::array<float, 2> &BatchAnalyzer_GetFeildsActivateMinDistance() const {
			return batch_analyzer_feild_activation_min_distance;
		}

		// Usage Tip: 
		// If you have set the value of the data pointer storage, then pointers to data sets will be available 
		// in this array, for which the distance of transferring the value to another area is minimal.
		const std::array<const void*, 2> &BatchAnalyzer_GetFeildsActivateMinDistanceDropOnPayload() const {
			return batch_analyzer_feild_activation_min_distance_drop_on;
		}

		void BatchAnalyzer_Reset() {
			batch_analyzer_feild_activation_min_distance[1] = batch_analyzer_feild_activation_min_distance[0] = std::numeric_limits<float>::max();
			batch_analyzer_feild_activation_min_distance_drop_on[1] = batch_analyzer_feild_activation_min_distance_drop_on[0] = nullptr;
			batch_analyzer_payload = &batch_analyzer_payload_plug;
		}

		void BatchAnalyzer_SetDataPayloadPtrSource(const void **payload_source) {
			batch_analyzer_payload = payload_source;
		}

		void FlipCurrentInputToAnotherFeild() {
			float unweighted_summ = 0;
			for (const auto inp : inputs) {
				unweighted_summ += inp->From()->OwnLevel();
			}
			if (unweighted_summ > 1e-10f) { // Check if inputs are not 0
				if (accumulator < 0) { // Flip to positive feild
					// Need to set value small bigger than 0 
					float weights_delta = (accumulator - nan_correcting_epsilon) / unweighted_summ; // nan_correcting_epsilon is some "epsilon" for guaratees
					for (auto inp : inputs) {
						inp->Weight(inp->Weight() - weights_delta);
					}
				} else { // Flip to negative feild
					// Need to set value small lesser than 0 
					float weights_delta = (accumulator + nan_correcting_epsilon) / unweighted_summ; // nan_correcting_epsilon is some "epsilon" for guaratees
					for (auto inp : inputs) {
						inp->Weight(inp->Weight() - weights_delta);
					}
				}
			}
		}
	};
}
