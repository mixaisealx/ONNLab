#pragma once
#include "NNBasicsInterfaces.h"
#include "BasicBackPropgI.h"
#include "CustomBackPropgI.h"
#include "SelectableInputI.h"

#include <vector>
#include <algorithm>

namespace nn
{
	template<bool do_mangle = false, float scale = 1.0f, float offset = 0.0f>
	class NNB_m1h_SumHeadT : public interfaces::NeuronBasicInterface, public interfaces::BasicBackPropogableInterface {
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

		NNB_m1h_SumHeadT(const NNB_m1h_SumHeadT &) = delete;
		NNB_m1h_SumHeadT &operator=(const NNB_m1h_SumHeadT &) = delete;
	public:
		NNB_m1h_SumHeadT() {
			accumulator = 0;
			backprop_error_accumulator = 0;
		}

		~NNB_m1h_SumHeadT() override {
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
			accumulator = 0;
			for (const auto inp : inputs) {
				accumulator += inp->From()->OwnLevel();
			}

			if constexpr (do_mangle) {
				accumulator = ActivationFunction(accumulator);
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

	using NNB_m1h_SumHead = NNB_m1h_SumHeadT<>;
}
