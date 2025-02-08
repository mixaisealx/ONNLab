#pragma once
#include "NNBasicsInterfaces.h"
#include "BasicBackPropgI.h"
#include "CustomBackPropgI.h"
#include "SelectableInputI.h"

#include <vector>
#include <algorithm>

namespace nn
{
	class NNB_m1h_SumHead : public interfaces::NeuronBasicInterface, public interfaces::BasicBackPropogableInterface {
		float accumulator;
		float backprop_error_accumulator;
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

		NNB_m1h_SumHead(const NNB_m1h_SumHead &) = delete;
		NNB_m1h_SumHead &operator=(const NNB_m1h_SumHead &) = delete;
	public:
		NNB_m1h_SumHead() {
			accumulator = 0;
			backprop_error_accumulator = 0;
		}

		~NNB_m1h_SumHead() override {
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
			return x;
		}

		float ActivationFunctionDerivative(float x) const override {
			return 1.0f;
		}

		void UpdateOwnLevel() override {
			accumulator = 0;
			for (const auto inp : inputs) {
				accumulator += inp->From()->OwnLevel();
			}
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
