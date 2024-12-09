#pragma once
#include "NNBasicsInterfaces.h"
#include "BasicBackPropgI.h"

#include "math.h"

namespace nn
{
	class NNB_Sigmoid : public interfaces::NeuronBasicInterface, public interfaces::BasicBackPropogableInterface {
		float accumulator;
		float backprop_error_accumulator;
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

		NNB_Sigmoid(const NNB_Sigmoid &) = delete;
		NNB_Sigmoid &operator=(const NNB_Sigmoid &) = delete;
	public:
		NNB_Sigmoid() {
			accumulator = 0;
			backprop_error_accumulator = 0;
		}

		~NNB_Sigmoid() override {
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
			return 1.0f / (1.0f + powf(MC_E, -x));
		}

		float ActivationFunctionDerivative(float x) const override {
			return x * (1 - x);
		}

		void UpdateOwnLevel() override {
			accumulator = 0; // 1 * bias_weight
			for (const auto inp : inputs) {
				accumulator += inp->From()->OwnLevel() * inp->Weight();
			}
			accumulator = ActivationFunction(accumulator);
		}

		float OwnLevel() override {
			return accumulator;
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
	};
}
